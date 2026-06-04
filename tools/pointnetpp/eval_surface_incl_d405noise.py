"""D405 noise robustness study on surface inclusion setup.

Real Intel RealSense D405 depth noise: ~0.5-2mm at 10-30cm range.
Adds Gaussian noise N(0, σ) to observed surface positions, then calibrates
Liang + Hier and measures degradation of P/R/IoU.

For each mesh × noise_σ ∈ {0, 0.5mm, 1.0mm, 2.0mm} × 2 methods:
   - Calibrate with noisy obs
   - Report tau_IoU/P/R against GT
   - Sim-real overlap on noisy held-out T_B (= real-world test metric)
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
from tools.stageA_inclusion_recovery import build_inclusion_truth, gen_obs_per_particle_stiff  # noqa: E402
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec  # noqa: E402
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402
from tools.pointnetpp.eval_hier_phantom_realistic import (  # noqa: E402
    predict_on_traj, surface_overlap_metrics,
)
from tools.pointnetpp.eval_surface_inclusion import SURFACE_INCL_CFG  # noqa: E402


def add_pos_noise_to_surface(obs_pos_list, surface_idx_np, sigma_m, seed=0):
    """Add Gaussian noise N(0, sigma) to surface positions only.
    Returns new list with noisy obs (clean elsewhere).
    Noise is per-frame, per-surface-particle, isotropic.
    """
    if sigma_m <= 0:
        return obs_pos_list
    rng = np.random.RandomState(seed)
    noisy = []
    for p in obs_pos_list:
        p_new = p.copy()
        # Only noise surface particles
        noise = rng.normal(0.0, sigma_m, size=(len(surface_idx_np), 3)).astype(np.float32)
        p_new[surface_idx_np] = p_new[surface_idx_np] + noise
        noisy.append(p_new)
    return noisy


def run_cell(cfg, sigma_m, device, T_calib=10, T_pred=10, N_epochs=15):
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.95
    solver.num_iterations = 2
    solver.gravity = wp.vec3(0.0, 0.0, 0.0)
    pos_rest = mesh_info["vertices"]
    solver.floor_y = float(pos_rest[:, 2].min() - 0.1 * (pos_rest[:, 2].max() - pos_rest[:, 2].min()))

    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    top_axis = cfg["top_axis"]
    incl_rel, incl_radius = SURFACE_INCL_CFG[cfg["name"]]
    incl_center = bbox_min + np.array(incl_rel) * bbox_size

    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=cfg["stiff_in"], stiff_out=cfg["stiff_out"],
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)

    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx_np = np.where(surface_mask)[0]

    theta_truth_np = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    traj_calib = make_traj_from_spec(cfg["traj_specs"]["A"], bbox_min, bbox_max, T_calib)
    traj_B = make_traj_from_spec(cfg["traj_specs"]["B"], bbox_min, bbox_max, T_pred)

    solver.reset_to_rest(pos_rest)
    obs_pos_calib_clean, obs_alpha_calib = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_truth, traj_calib, knife_radius, device,
        tau_truth_np=tau_truth)
    solver.reset_to_rest(pos_rest)
    obs_pos_B_clean, obs_alpha_B = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_truth, traj_B, knife_radius, device,
        tau_truth_np=tau_truth)

    # Add D405 noise
    obs_pos_calib_noisy = add_pos_noise_to_surface(obs_pos_calib_clean, surface_idx_np, sigma_m, seed=42)
    obs_pos_B_noisy = add_pos_noise_to_surface(obs_pos_B_clean, surface_idx_np, sigma_m, seed=43)

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos_calib_noisy]
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
    H_hier = build_hierarchy(x_rest, M1=M1, M2=M2, k_neighbors=3)

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
        H=H_hier,
    )

    results = {}
    for method in ["Liang_inner", "Hier+Liang_inner"]:
        t0 = time.time()
        stiff, tau, _ = calibrate(method, ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                          lambda_smooth=0.0, lambda_E=1e-5)
        dt = time.time() - t0
        truth_in = tau_truth < 0.3
        pred_in = tau < 0.3
        tp = int((pred_in & truth_in).sum())
        fn = int((~pred_in & truth_in).sum())
        fp = int((pred_in & ~truth_in).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        union = (pred_in | truth_in).sum()
        iou = 0.0 if union == 0 else tp / union

        # Sim-real on NOISY held-out (= real metric)
        pred_pos_B, pred_alpha_B = predict_on_traj(stiff, tau, traj_B, ctx)
        sr_noisy = surface_overlap_metrics(pred_pos_B, obs_pos_B_noisy, pred_alpha_B,
                                                obs_alpha_B, surface_idx_np)
        sr_clean = surface_overlap_metrics(pred_pos_B, obs_pos_B_clean, pred_alpha_B,
                                                obs_alpha_B, surface_idx_np)

        results[method] = dict(
            stiff_iou=float(((stiff > 0.5) & (stiff_truth > 0.5)).sum() / max(((stiff > 0.5) | (stiff_truth > 0.5)).sum(), 1)),
            tau_IoU=iou, P=prec, R=rec,
            pos_mm_noisy=sr_noisy["pos_mm"], cav_iou_noisy=sr_noisy["cav_iou"],
            pos_mm_clean=sr_clean["pos_mm"], cav_iou_clean=sr_clean["cav_iou"],
            n_pred_in=int(pred_in.sum()), n_truth_in=int(truth_in.sum()),
            runtime=dt,
        )
    return results


def main():
    device = "cuda:0"
    NOISE_SIGMAS = [0.0, 0.0005, 0.001, 0.002]   # 0, 0.5mm, 1mm, 2mm
    t0_all = time.time()
    print(f"{'mesh':<10} {'method':<18} {'σ_mm':>6} "
          f"{'tau_IoU':>8} {'P':>5} {'R':>5} "
          f"{'pos_noisy':>10} {'cav_noisy':>10} {'pos_clean':>10} {'cav_clean':>10}")
    print("-" * 120)
    for cfg in MESH_CONFIGS:
        for sigma_m in NOISE_SIGMAS:
            print(f"  ... {cfg['name']} σ={sigma_m*1000:.1f}mm", flush=True)
            r = run_cell(cfg, sigma_m, device, T_calib=10, T_pred=10, N_epochs=15)
            for m in ["Liang_inner", "Hier+Liang_inner"]:
                d = r[m]
                print(f"{cfg['name']:<10} {m:<18} {sigma_m*1000:>6.1f} "
                       f"{d['tau_IoU']:>8.3f} {d['P']:>5.2f} {d['R']:>5.2f} "
                       f"{d['pos_mm_noisy']:>10.3f} {d['cav_iou_noisy']:>10.3f} "
                       f"{d['pos_mm_clean']:>10.3f} {d['cav_iou_clean']:>10.3f}", flush=True)
    print(f"\nTotal: {(time.time()-t0_all)/60:.1f} min")


if __name__ == "__main__":
    main()
