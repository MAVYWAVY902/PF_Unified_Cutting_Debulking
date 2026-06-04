"""3-mesh × 3-inclusion-size × partial-obs hier evaluation.

Setup matches real phantom deployment:
   - iter=2 (real-time feasible)
   - damping=0.95, gravity=0, floor below bbox (corrected physics)
   - partial obs only (surface mask, top 10%) — D405 cannot see interior
   - 3 mesh shapes (cube, tumor, torus) × 3 inclusion sizes per mesh
   - calibrate on T_A, test sim-real overlap on T_B (held-out trajectory)

Question: does hier truly help under realistic phantom obs across mesh shapes?

Output per cell (3 meshes × 3 incl sizes = 9 cells):
   - stiff_in mean (Hier vs Liang vs truth)
   - stiff_IoU @ 0.5 threshold
   - sim-real surface pos_mm, cav_IoU, alpha_MSE on T_B
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))

import warp as wp  # noqa: E402
wp.init()

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.stageA_inclusion_recovery import build_inclusion_truth, gen_obs_per_particle_stiff  # noqa: E402
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec  # noqa: E402
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402
from tools.pointnetpp.eval_hier_phantom_realistic import (  # noqa: E402
    predict_on_traj, surface_overlap_metrics, param_iou,
)


# Per-mesh inclusion size sweep (absolute meters)
# Sized to span ~1% to ~10% of mesh volume.
INCL_SIZES_PER_MESH = {
    "M1_cube":  [("small", 0.04), ("medium", 0.07), ("large", 0.10)],
    "M2_tumor": [("small", 0.003), ("medium", 0.005), ("large", 0.008)],
    "M3_torus": [("small", 0.04), ("medium", 0.08), ("large", 0.12)],
}


def run_cell(cfg, incl_radius, device, T_calib=10, T_pred=10, N_epochs=15):
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    # Corrected physics
    solver.damping = 0.95
    solver.num_iterations = 2
    solver.gravity = wp.vec3(0.0, 0.0, 0.0)
    pos_rest = mesh_info["vertices"]
    solver.floor_y = float(pos_rest[:, 2].min() - 0.1 * (pos_rest[:, 2].max() - pos_rest[:, 2].min()))

    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size
    top_axis = cfg["top_axis"]

    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=cfg["stiff_in"], stiff_out=cfg["stiff_out"],
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    n_incl = int(inclusion_mask.sum())
    if n_incl == 0 or n_incl == N:
        return None

    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx_np = np.where(surface_mask)[0]

    theta_truth_np = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    traj_calib = make_traj_from_spec(cfg["traj_specs"]["A"], bbox_min, bbox_max, T_calib)
    traj_B = make_traj_from_spec(cfg["traj_specs"]["B"], bbox_min, bbox_max, T_pred)

    solver.reset_to_rest(pos_rest)
    obs_pos_calib, obs_alpha_calib = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_truth, traj_calib, knife_radius, device,
        tau_truth_np=tau_truth)
    solver.reset_to_rest(pos_rest)
    obs_pos_B, obs_alpha_B = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_truth, traj_B, knife_radius, device,
        tau_truth_np=tau_truth)

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos_calib]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha_calib]
    surface_idx = torch.tensor(surface_idx_np, dtype=torch.long, device=device)

    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edges_a_t = torch.tensor(edges_a_np, dtype=torch.long, device=device)
    edges_b_t = torch.tensor(edges_b_np, dtype=torch.long, device=device)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)

    x_rest = torch.from_numpy(pos_rest).to(device, dtype=torch.float32)
    M1 = min(512, max(32, N // 20))
    M2 = min(64, max(8, N // 200))
    H = build_hierarchy(x_rest, M1=M1, M2=M2, k_neighbors=3)

    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)

    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=traj_calib, knife_radius=knife_radius,
        pos_rest=pos_rest, theta_truth=theta_truth_np, theta_3d_np=theta_truth_np,
        stiff_init_np=stiff_init, stiff_truth=stiff_truth,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        inclusion_mask=inclusion_mask,
        surface_idx=surface_idx, surface_mask=surface_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H,
    )

    results = {}
    for method in ["Liang_inner", "Hier+Liang_inner"]:
        t0 = time.time()
        stiff, tau, hist = calibrate(method, ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                          lambda_smooth=0.0, lambda_E=1e-5)
        dt = time.time() - t0
        interior_incl = inclusion_mask & ~surface_mask
        stiff_in_int = float(stiff[interior_incl].mean()) if interior_incl.sum() > 0 else float("nan")
        stiff_iou = param_iou(stiff, stiff_truth, threshold=0.5, direction=">")
        tau_in_int = float(tau[interior_incl].mean()) if interior_incl.sum() > 0 else float("nan")
        tau_iou = param_iou(tau, tau_truth, threshold=0.3, direction="<")

        pred_pos, pred_alpha = predict_on_traj(stiff, tau, traj_B, ctx)
        sr = surface_overlap_metrics(pred_pos, obs_pos_B, pred_alpha, obs_alpha_B, surface_idx_np)
        results[method] = dict(
            stiff_in_int=stiff_in_int, stiff_iou=stiff_iou,
            tau_in_int=tau_in_int, tau_iou=tau_iou,
            pos_mm=sr["pos_mm"], cav_iou=sr["cav_iou"], alpha_mse=sr["alpha_mse"],
            runtime=dt,
        )
    return dict(n_incl=n_incl, N=N, n_surface=int(surface_mask.sum()),
                 truth_stiff_in=cfg["stiff_in"], truth_tau_in=cfg["tau_in"],
                 results=results)


def main():
    device = "cuda:0"
    all_data = []  # list of (mesh, incl_name, incl_r, n_incl, ...)
    t0_all = time.time()
    for cfg in MESH_CONFIGS:
        sizes = INCL_SIZES_PER_MESH.get(cfg["name"], [("default", cfg["incl_radius_abs"])])
        for incl_name, incl_r in sizes:
            print(f"\n{'='*80}\nMESH: {cfg['name']}  incl={incl_name}(r={incl_r})\n{'='*80}", flush=True)
            try:
                res = run_cell(cfg, incl_r, device, T_calib=10, T_pred=10, N_epochs=15)
            except Exception as e:
                import traceback
                print(f"  FAILED: {e}")
                traceback.print_exc()
                res = None
            if res is None:
                print(f"  skipped"); continue
            all_data.append((cfg["name"], incl_name, incl_r, res))
            n = res["n_incl"]; N = res["N"]
            print(f"  n_incl={n}/{N} ({n/N*100:.2f}%)  n_surface={res['n_surface']}", flush=True)
            for m in ["Liang_inner", "Hier+Liang_inner"]:
                r = res["results"][m]
                print(f"  {m:<18}  stiff_in={r['stiff_in_int']:.3f} (truth {res['truth_stiff_in']:.2f}) "
                       f"iou={r['stiff_iou']:.3f}  "
                       f"tau_in={r['tau_in_int']:.3f} iou={r['tau_iou']:.3f}  "
                       f"sim-real pos={r['pos_mm']:.3f}mm cav_IoU={r['cav_iou']:.3f} "
                       f"αMSE={r['alpha_mse']:.4f}  ({r['runtime']:.0f}s)", flush=True)

    print(f"\n{'='*120}")
    print(f"3-MESH × 3-INCL PARTIAL-OBS HIER VS LIANG (iter=2, real-time, corrected physics)")
    print(f"{'='*120}")
    print(f"{'mesh':<10} {'incl':<8} {'n%':>5} {'Liang stiff_in':>15} {'Hier stiff_in':>14} "
          f"{'Δstiff%':>9} {'L pos_mm':>10} {'H pos_mm':>10} {'L cav_IoU':>11} {'H cav_IoU':>11} {'verdict':>14}")
    print("-" * 130)
    helps = hurts = marginal = 0
    for mesh, incl_name, incl_r, res in all_data:
        L = res["results"]["Liang_inner"]
        H = res["results"]["Hier+Liang_inner"]
        truth = res["truth_stiff_in"]
        gap_L = abs(L["stiff_in_int"] - truth)
        gap_H = abs(H["stiff_in_int"] - truth)
        d_stiff = (gap_L - gap_H) / max(gap_L, 1e-9) * 100
        d_pos = H["pos_mm"] - L["pos_mm"]
        d_iou = H["cav_iou"] - L["cav_iou"]
        d_mse = H["alpha_mse"] - L["alpha_mse"]
        sim_real_score = sum([d_pos < 0, d_iou > 0, d_mse < 0])
        param_score = d_stiff > 5
        if sim_real_score >= 2 and d_stiff > 5:
            verdict = "★★ hier wins"
            helps += 1
        elif sim_real_score >= 2 or d_stiff > 5:
            verdict = "★ hier helps"
            helps += 1
        elif sim_real_score <= 1 and d_stiff < -5:
            verdict = "✗ hier hurts"
            hurts += 1
        else:
            verdict = "≈ marginal"
            marginal += 1
        n_pct = res["n_incl"] / res["N"] * 100
        print(f"{mesh:<10} {incl_name:<8} {n_pct:>5.1f} {L['stiff_in_int']:>15.3f} {H['stiff_in_int']:>14.3f} "
              f"{d_stiff:>+8.1f}% {L['pos_mm']:>10.3f} {H['pos_mm']:>10.3f} "
              f"{L['cav_iou']:>11.3f} {H['cav_iou']:>11.3f} {verdict:>14}")
    print("-" * 130)
    print(f"\nFinal verdict: {helps} helps  /  {hurts} hurts  /  {marginal} marginal  (out of {len(all_data)})")
    print(f"Total: {(time.time()-t0_all)/60:.1f} min")

    np.savez(REPO / "tools" / "pointnetpp" / "eval_hier_3mesh_partial_sweep.npz",
              all_data=all_data)


if __name__ == "__main__":
    main()
