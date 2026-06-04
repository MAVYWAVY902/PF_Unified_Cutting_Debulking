"""Phantom-realistic multi-mesh hier vs Liang evaluation.

KEY DESIGN — sim-vs-real surface obs overlap is the criterion.
The point of calibration is to make our SIM produce surface observations
(positions + cavity mask) that look like REAL observations (D405 surface only).
Numerical stiff_in recovery is a diagnostic, NOT the goal.

Phantom-obtainable signals (D405 surface RGB-D):
    1. Surface particle positions (visible from above)
    2. Surface cavity binary mask (alpha < 0.5 visible as missing surface)
NOT phantom-obtainable:
    - Interior particle positions/alpha
    - Per-particle stiff or tau (these are the unknowns!)

We evaluate on a HELD-OUT trajectory T_B (different sweep direction than
calibration T_A) — generalization across cut patterns is what we need.

Methods compared on each mesh:
    Liang_inner       — per-particle Adam, surface-masked L_pos + L_cav
    Hier+Liang_inner  — same loss, but z_L0 + z_L1 + z_L2 hierarchical reparam

Reported (3 categories):

  A. SIM-REAL SURFACE OVERLAP (phantom-realistic, primary):
       surface_pos_mm   surface mean L2 distance over T frames (mm)
       surface_iou      surface cavity IoU (binary, alpha<0.5)
       surface_alpha_mse surface alpha MSE (continuous, alpha values)

  B. PARAMETER RECOVERY (diagnostic, not phantom-obtainable):
       stiff_in_int     interior particles inside inclusion (un-observable)
       stiff_iou        per-particle stiff > 0.5 IoU vs truth
       tau_in_int       interior tau (un-observable)
       tau_iou          per-particle tau < 0.3 IoU vs truth

  C. CALIBRATION CONVERGENCE:
       L_gap_final      Liang surrogate loss at convergence
       runtime_s        wall-time

Mesh set: cube + tumor + torus (M1/M2/M3 from multi_mesh_test.MESH_CONFIGS).
Regime: iter=2 (real-time-feasible, hier winning regime per value map v2).
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
from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff,
)
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec  # noqa: E402
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402


def predict_on_traj(stiff_pp, tau_pp, knife_traj, ctx):
    device = ctx["device"]
    theta_torch = torch.tensor(ctx["theta_3d_np"], dtype=torch.float32, device=device,
                                  requires_grad=False)
    stiff_torch = torch.tensor(stiff_pp, dtype=torch.float32, device=device, requires_grad=False)
    tau_torch = torch.tensor(tau_pp, dtype=torch.float32, device=device, requires_grad=False)
    solver = ctx["solver"]
    rest_state = ctx["rest_state"]
    knife_radius = ctx["knife_radius"]
    pos_rest = ctx["pos_rest"]
    solver.reset_to_rest(pos_rest)
    state_in = rest_state
    pred_pos, pred_alpha = [], []
    with torch.no_grad():
        for t, knife_pos in enumerate(knife_traj):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, knife_pos, knife_radius,
                stiff_torch, tau_torch,
            )
            pred_pos.append(pos_t.cpu().numpy().copy())
            pred_alpha.append(alpha_t.cpu().numpy().copy())
            state_in = solver_save_state(solver)
    return pred_pos, pred_alpha


def surface_overlap_metrics(pred_pos, truth_pos, pred_alpha, truth_alpha, surface_idx_np):
    """Phantom-realistic similarity on surface particles only.
    Returns dict with:
       pos_mm     — mean L2 surface position error (mm), averaged over T
       cav_iou    — surface cavity binary IoU (alpha<0.5), averaged over T
       alpha_mse  — surface alpha continuous MSE, averaged over T
    """
    T = len(pred_pos)
    pos_errs, ious, mses = [], [], []
    for t in range(T):
        d = np.linalg.norm(pred_pos[t][surface_idx_np] - truth_pos[t][surface_idx_np], axis=1)
        pos_errs.append(float(d.mean()))
        pa = pred_alpha[t][surface_idx_np]
        ta = truth_alpha[t][surface_idx_np]
        mses.append(float(((pa - ta) ** 2).mean()))
        # Cavity IoU: missing/dead particles
        pred_dead = pa < 0.5
        truth_dead = ta < 0.5
        inter = int((pred_dead & truth_dead).sum())
        union = int((pred_dead | truth_dead).sum())
        ious.append(1.0 if union == 0 else inter / union)
    return {
        "pos_mm": float(np.mean(pos_errs)) * 1000.0,
        "cav_iou": float(np.mean(ious)),
        "alpha_mse": float(np.mean(mses)),
    }


def param_iou(pred, truth, threshold, direction):
    """IoU on per-particle params at given threshold.
    direction='>': mask where pred > threshold (for stiff inclusion is HIGH).
    direction='<': mask where pred < threshold (for tau inclusion is LOW).
    """
    if direction == ">":
        pm = pred > threshold
        tm = truth > threshold
    else:
        pm = pred < threshold
        tm = truth < threshold
    inter = int((pm & tm).sum())
    union = int((pm | tm).sum())
    return 1.0 if union == 0 else inter / union


def run_one_mesh(cfg, device, T_calib=10, T_pred=10, N_epochs=15):
    print(f"\n{'='*78}")
    print(f"MESH: {cfg['name']}  ({cfg['path']})")
    print(f"{'='*78}", flush=True)
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    # 2026-05-30 PHYSICS CALIBRATION (after sanity audit):
    # - damping 0.99 → 0.95 (match cpp_sim, kill zombie momentum on dead particles)
    # - num_iter = 2 (cube/tumor solid at this regime; iter=16 makes torus alive
    #   particles drift 10mm worse, no general win)
    # - gravity=0 (avoid axis mismatch: cube/tumor top=Y but solver gravity=−Z;
    #   for paper phantom deployment, enable with per-mesh axis fix)
    # - floor below bbox z_min (tumor mesh z range [-0.012, 0.012] was half-
    #   buried under floor_y=0, causing violent collision push = "fluid" effect)
    solver.damping = 0.95
    solver.num_iterations = 2
    solver.gravity = wp.vec3(0.0, 0.0, 0.0)
    pos_rest_for_floor = mesh_info["vertices"]
    solver.floor_y = float(pos_rest_for_floor[:, 2].min()
                            - 0.1 * (pos_rest_for_floor[:, 2].max() - pos_rest_for_floor[:, 2].min()))

    pos_rest = mesh_info["vertices"]
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size
    incl_radius = cfg["incl_radius_abs"]
    top_axis = cfg["top_axis"]
    print(f"  N={N}  bbox=({bbox_size[0]:.4f},{bbox_size[1]:.4f},{bbox_size[2]:.4f})")
    print(f"  top_axis={['x','y','z'][top_axis]}  knife_radius={knife_radius:.4f}")
    print(f"  incl_center={incl_center}  incl_radius={incl_radius:.4f}")

    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=cfg["stiff_in"], stiff_out=cfg["stiff_out"],
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    n_incl = int(inclusion_mask.sum())
    if n_incl == 0 or n_incl == N:
        print(f"  ⚠ skip {cfg['name']}: degenerate inclusion ({n_incl}/{N})")
        return None

    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx_np = np.where(surface_mask)[0]
    n_surface = len(surface_idx_np)
    print(f"  inclusion: {n_incl}/{N} ({n_incl/N*100:.2f}%)  surface: {n_surface}/{N} ({n_surface/N*100:.2f}%)")

    theta_truth_np = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    truth = {}
    for name, spec in cfg["traj_specs"].items():
        traj = make_traj_from_spec(spec, bbox_min, bbox_max, T_calib if name == "A" else T_pred)
        solver.reset_to_rest(pos_rest)
        obs_pos, obs_alpha = gen_obs_per_particle_stiff(
            solver, rest_state, theta_truth_np, stiff_truth, traj, knife_radius, device,
            tau_truth_np=tau_truth)
        cav_surface = float((obs_alpha[-1][surface_idx_np] < 0.5).mean())
        truth[name] = dict(traj=traj, pos=obs_pos, alpha=obs_alpha)
        print(f"    T_{name}: surface cav={cav_surface*100:.2f}%", flush=True)

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in truth["A"]["pos"]]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in truth["A"]["alpha"]]
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
    print(f"  Hier: M1={M1}, M2={M2}", flush=True)

    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)

    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=truth["A"]["traj"], knife_radius=knife_radius,
        pos_rest=pos_rest, theta_truth=theta_truth_np, theta_3d_np=theta_truth_np,
        stiff_init_np=stiff_init, stiff_truth=stiff_truth,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        inclusion_mask=inclusion_mask,
        surface_idx=surface_idx, surface_mask=surface_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H,
    )

    out = {}
    for method in ["Liang_inner", "Hier+Liang_inner"]:
        print(f"\n  [{method}] calibrate ...", flush=True)
        t0 = time.time()
        stiff, tau, hist = calibrate(method, ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                          lambda_smooth=0.0, lambda_E=1e-5)
        dt = time.time() - t0
        Lgap_final = float(hist[-1]["mean_L"]) if hist else float("nan")

        # Param recovery (diagnostic)
        interior_incl = inclusion_mask & ~surface_mask
        stiff_in_int = float(stiff[interior_incl].mean()) if interior_incl.sum() > 0 else float("nan")
        stiff_out_int = float(stiff[~inclusion_mask & ~surface_mask].mean()) if (~inclusion_mask & ~surface_mask).sum() > 0 else float("nan")
        tau_in_int = float(tau[interior_incl].mean()) if interior_incl.sum() > 0 else float("nan")
        tau_out_int = float(tau[~inclusion_mask & ~surface_mask].mean()) if (~inclusion_mask & ~surface_mask).sum() > 0 else float("nan")
        stiff_iou_val = param_iou(stiff, stiff_truth, threshold=0.5, direction=">")
        tau_iou_val = param_iou(tau, tau_truth, threshold=0.3, direction="<")

        # Sim-real overlap on held-out T_B (phantom-realistic, primary metric)
        traj_B = truth["B"]["traj"]
        pred_pos_B, pred_alpha_B = predict_on_traj(stiff, tau, traj_B, ctx)
        sim_real_B = surface_overlap_metrics(pred_pos_B, truth["B"]["pos"],
                                                  pred_alpha_B, truth["B"]["alpha"], surface_idx_np)

        # Also surface metrics on T_A (in-distribution sanity)
        traj_A = truth["A"]["traj"]
        pred_pos_A, pred_alpha_A = predict_on_traj(stiff, tau, traj_A, ctx)
        sim_real_A = surface_overlap_metrics(pred_pos_A, truth["A"]["pos"],
                                                  pred_alpha_A, truth["A"]["alpha"], surface_idx_np)

        out[method] = dict(
            runtime_s=dt, Lgap_final=Lgap_final,
            stiff_in_int=stiff_in_int, stiff_out_int=stiff_out_int,
            tau_in_int=tau_in_int, tau_out_int=tau_out_int,
            stiff_iou=stiff_iou_val, tau_iou=tau_iou_val,
            simreal_A=sim_real_A, simreal_B=sim_real_B,
        )
        print(f"    runtime={dt:.0f}s  L_gap={Lgap_final:.3e}", flush=True)
        print(f"    Param recovery: stiff_in={stiff_in_int:.3f} (truth {cfg['stiff_in']:.2f})  "
              f"stiff_out={stiff_out_int:.3f} (truth {cfg['stiff_out']:.2f})  IoU={stiff_iou_val:.3f}", flush=True)
        print(f"                    tau_in  ={tau_in_int:.3f} (truth {cfg['tau_in']:.2f})  "
              f"tau_out  ={tau_out_int:.3f} (truth {cfg['tau_out']:.2f})  IoU={tau_iou_val:.3f}", flush=True)
        print(f"    Sim-Real T_A: pos {sim_real_A['pos_mm']:.3f} mm  cav_IoU {sim_real_A['cav_iou']:.3f}  alpha_MSE {sim_real_A['alpha_mse']:.4f}", flush=True)
        print(f"    Sim-Real T_B: pos {sim_real_B['pos_mm']:.3f} mm  cav_IoU {sim_real_B['cav_iou']:.3f}  alpha_MSE {sim_real_B['alpha_mse']:.4f}", flush=True)

    return dict(mesh=cfg["name"], N=N, n_incl=n_incl, n_surface=n_surface, results=out)


def summarize(all_res):
    print("\n" + "=" * 110)
    print("PHANTOM-REALISTIC HIER vs LIANG — SIM-REAL SURFACE OVERLAP (held-out T_B)")
    print("=" * 110)
    print(f"{'mesh':<10} {'method':<20} {'pos_mm':>9} {'cav_IoU':>9} {'α_MSE':>9} "
          f"{'stiff_in':>9} {'stiff_iou':>10} {'tau_in':>9} {'tau_iou':>9} {'L_gap':>11}")
    print("-" * 110)
    for r in all_res:
        if r is None:
            continue
        for m in ["Liang_inner", "Hier+Liang_inner"]:
            x = r["results"][m]
            print(f"{r['mesh']:<10} {m:<20} "
                  f"{x['simreal_B']['pos_mm']:>9.3f} {x['simreal_B']['cav_iou']:>9.3f} {x['simreal_B']['alpha_mse']:>9.4f} "
                  f"{x['stiff_in_int']:>9.3f} {x['stiff_iou']:>10.3f} {x['tau_in_int']:>9.3f} {x['tau_iou']:>9.3f} "
                  f"{x['Lgap_final']:>11.3e}")
    print("=" * 110)

    # Per-mesh delta
    print("\nDELTA (Hier - Liang) — positive = hier closer to real obs (better for pos/MSE, lower is better)")
    print("                       — positive cav_IoU = hier improves cavity overlap")
    print(f"{'mesh':<10} {'Δpos_mm':>10} {'Δcav_IoU':>10} {'Δα_MSE':>10} "
          f"{'Δstiff_iou':>12} {'Δtau_iou':>11} {'verdict (sim-real)':<22}")
    print("-" * 95)
    for r in all_res:
        if r is None:
            continue
        L = r["results"]["Liang_inner"]
        H = r["results"]["Hier+Liang_inner"]
        d_pos = H["simreal_B"]["pos_mm"] - L["simreal_B"]["pos_mm"]
        d_iou = H["simreal_B"]["cav_iou"] - L["simreal_B"]["cav_iou"]
        d_mse = H["simreal_B"]["alpha_mse"] - L["simreal_B"]["alpha_mse"]
        d_siou = H["stiff_iou"] - L["stiff_iou"]
        d_tiou = H["tau_iou"] - L["tau_iou"]
        # Verdict: hier helps if pos_mm decreases AND (cav_IoU increases OR alpha_MSE decreases)
        # Conservative: count helps if 2 of 3 sim-real metrics improve
        helps = sum([d_pos < 0, d_iou > 0, d_mse < 0])
        if helps >= 2:
            v = "★ hier helps"
        elif helps == 1:
            v = "≈ marginal"
        else:
            v = "✗ hier hurts"
        print(f"{r['mesh']:<10} {d_pos:>+10.3f} {d_iou:>+10.3f} {d_mse:>+10.4f} "
              f"{d_siou:>+12.3f} {d_tiou:>+11.3f} {v:<22}")
    print("=" * 95)


def main():
    device = "cuda:0"
    all_res = []
    t0 = time.time()
    for cfg in MESH_CONFIGS:
        try:
            res = run_one_mesh(cfg, device, T_calib=10, T_pred=10, N_epochs=15)
            all_res.append(res)
        except Exception as e:
            import traceback
            print(f"  ⚠ {cfg['name']} FAILED: {e}")
            traceback.print_exc()
            all_res.append(None)
    summarize(all_res)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")

    # Save raw
    np.savez(REPO / "tools" / "pointnetpp" / "eval_hier_phantom_realistic.npz",
              results=all_res)


if __name__ == "__main__":
    main()
