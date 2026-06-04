"""Sanity check — Hier 真 不 真 working? 还 是 silently broken?

Two diagnostics:

  CHECK 1: Compare Liang vs Hier calibrated stiff fields directly
     - Are stiff_L and stiff_H actually different per particle?
     - How much different?
     - Where (inclusion vs outside)?

  CHECK 2: Forward sim sensitivity probe
     - Run forward with uniform stiff in {0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5}
     - Measure sim deformation magnitude
     - Determine: is forward sim REALLY insensitive in [0.7, 0.85] range?
     - Quantify how much stiff difference is needed for visually visible change

  CHECK 3: Compare pred_L vs pred_H at frame 29 directly
     - Per-particle position diff
     - Per-particle alpha diff
     - Is the difference "noise" (≤ 1e-9) or "small but real" (~1e-5)?
"""

import sys
import time
import math
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))

import warp as wp  # noqa: E402
wp.init()
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402

from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff,
)
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import (  # noqa: E402
    MESH_CONFIGS, make_traj_from_spec, predict_on_traj,
)
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402


def main():
    cfg = MESH_CONFIGS[0]
    device = "cuda:0"
    T = 10
    N_epochs = 60
    lambda_E = 1e-5
    stiff_in_truth = 0.8

    print(f"\n=== Hier sanity check ===")
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size
    incl_radius = cfg["incl_radius_abs"]
    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=stiff_in_truth, stiff_out=0.3,
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)
    top_axis = cfg["top_axis"]
    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold

    theta_truth = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    traj_A = make_traj_from_spec(cfg["traj_specs"]["A"], bbox_min, bbox_max, T)
    solver.reset_to_rest(pos_rest)
    obs_pos, obs_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth, stiff_truth, traj_A, knife_radius, device,
        tau_truth_np=tau_truth)

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]
    surface_idx = torch.tensor(np.where(surface_mask)[0], dtype=torch.long, device=device)
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64), dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64), dtype=torch.long, device=device)
    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)
    x_rest = torch.from_numpy(pos_rest).to(device, dtype=torch.float32)
    H = build_hierarchy(x_rest, M1=512, M2=64, k_neighbors=3)
    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=traj_A, knife_radius=knife_radius,
        pos_rest=pos_rest, theta_truth=theta_truth, theta_3d_np=theta_truth,
        stiff_init_np=stiff_init, stiff_truth=stiff_truth,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        inclusion_mask=inclusion_mask,
        surface_idx=surface_idx, surface_mask=surface_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H,
    )

    # Calibrate
    print(f"\n[Calibrate Liang on T={T}]")
    stiff_L, tau_L, _ = calibrate("Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                     lambda_smooth=0.0, lambda_E=lambda_E)
    print(f"\n[Calibrate Hier+Liang on T={T}]")
    stiff_H, tau_H, _ = calibrate("Hier+Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                     lambda_smooth=0.0, lambda_E=lambda_E)

    # ===== CHECK 1: stiff field differences =====
    print(f"\n{'='*72}")
    print(f"CHECK 1: Liang vs Hier stiff field comparison")
    print(f"{'='*72}")
    stiff_diff = stiff_H - stiff_L
    print(f"  Per-particle stiff diff (Hier - Liang):")
    print(f"    mean = {stiff_diff.mean():+.4f}")
    print(f"    std  = {stiff_diff.std():.4f}")
    print(f"    min  = {stiff_diff.min():+.4f}")
    print(f"    max  = {stiff_diff.max():+.4f}")
    print(f"    |diff|.mean = {np.abs(stiff_diff).mean():.4f}")
    print(f"  Per-region breakdown:")
    print(f"    Inclusion particles: Liang {stiff_L[inclusion_mask].mean():.3f} vs Hier {stiff_H[inclusion_mask].mean():.3f}  Δ {(stiff_H-stiff_L)[inclusion_mask].mean():+.3f}")
    print(f"    Outside particles:   Liang {stiff_L[~inclusion_mask].mean():.3f} vs Hier {stiff_H[~inclusion_mask].mean():.3f}  Δ {(stiff_H-stiff_L)[~inclusion_mask].mean():+.3f}")
    print(f"    Surface particles:   Liang {stiff_L[surface_mask].mean():.3f} vs Hier {stiff_H[surface_mask].mean():.3f}  Δ {(stiff_H-stiff_L)[surface_mask].mean():+.3f}")
    print(f"    Interior particles:  Liang {stiff_L[~surface_mask].mean():.3f} vs Hier {stiff_H[~surface_mask].mean():.3f}  Δ {(stiff_H-stiff_L)[~surface_mask].mean():+.3f}")
    if np.abs(stiff_diff).mean() > 0.01:
        print(f"  ✓ stiff fields ARE substantially different → hier IS modifying parameters")
    else:
        print(f"  ✗ stiff fields essentially identical → POSSIBLE BUG in hier")

    # ===== CHECK 2: Forward sim sensitivity probe =====
    print(f"\n{'='*72}")
    print(f"CHECK 2: Forward sim sensitivity probe — uniform stiff")
    print(f"{'='*72}")
    print(f"  Run forward T=10 with uniform stiff field, compare deformation magnitude")
    uniform_stiff_grid = [0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5]
    sim_outputs = {}
    for s_unif in uniform_stiff_grid:
        stiff_test = np.full(N, s_unif, dtype=np.float32)
        tau_test = np.full(N, 0.3, dtype=np.float32)
        pred_pos, pred_alpha = predict_on_traj(stiff_test, tau_test, traj_A, ctx)
        # Compare final frame deformation magnitude
        displ = np.linalg.norm(pred_pos[-1] - pos_rest, axis=1)
        cav = (pred_alpha[-1] < 0.5).mean()
        sim_outputs[s_unif] = dict(
            mean_displ=float(displ.mean()),
            max_displ=float(displ.max()),
            cavity=float(cav),
            pos=pred_pos[-1],
            alpha=pred_alpha[-1],
        )
        print(f"    uniform stiff = {s_unif:.2f}:  mean displ = {displ.mean():.3e} m  max = {displ.max():.3e} m  cavity = {cav*100:.2f}%")

    # Compare sensitivity: how much does mean displacement change per unit stiff?
    s_arr = np.array(uniform_stiff_grid)
    d_arr = np.array([sim_outputs[s]["mean_displ"] for s in uniform_stiff_grid])
    sensitivity = np.diff(d_arr) / np.diff(s_arr)
    print(f"\n  Sensitivity ∂(mean_displ)/∂stiff:")
    for i, (s0, s1) in enumerate(zip(uniform_stiff_grid[:-1], uniform_stiff_grid[1:])):
        print(f"    {s0:.2f} → {s1:.2f}: sensitivity = {sensitivity[i]:.3e} m/unit_stiff")
    # In stiff range [0.7, 0.85] (where calibrated values live), what's the sensitivity?
    print(f"\n  Across L/H calibrated range (0.73 → 0.81, Δ=0.08):")
    s_low, s_high = 0.7, 0.8
    d_low = sim_outputs[0.7]["mean_displ"]
    d_high = sim_outputs[0.8]["mean_displ"]
    print(f"    sim at stiff 0.7: mean displ {d_low:.3e}")
    print(f"    sim at stiff 0.8: mean displ {d_high:.3e}")
    expected_diff = abs(d_high - d_low) * (0.08 / 0.1)  # 0.08 change vs 0.1 step
    print(f"    Expected mean displ diff for Δstiff=0.08: {expected_diff:.3e}")

    # ===== CHECK 3: pred_L vs pred_H direct compare =====
    print(f"\n{'='*72}")
    print(f"CHECK 3: pred_L vs pred_H direct forward sim comparison")
    print(f"{'='*72}")
    pred_L_pos, pred_L_alpha = predict_on_traj(stiff_L, tau_L, traj_A, ctx)
    pred_H_pos, pred_H_alpha = predict_on_traj(stiff_H, tau_H, traj_A, ctx)
    print(f"  At final frame T={T-1}:")
    diff_per_particle = np.linalg.norm(pred_L_pos[-1] - pred_H_pos[-1], axis=1)
    diff_alpha = np.abs(pred_L_alpha[-1] - pred_H_alpha[-1])
    print(f"    Position L vs H diff: mean = {diff_per_particle.mean():.3e} m  max = {diff_per_particle.max():.3e} m")
    print(f"    Alpha    L vs H diff: mean = {diff_alpha.mean():.3e}      max = {diff_alpha.max():.3e}")
    print(f"  GPU noise floor (measured earlier): ~2.2e-9 m mean")
    if diff_per_particle.mean() > 1e-6:
        print(f"  ✓ pred_L ≠ pred_H by {diff_per_particle.mean()/2.2e-9:.0f}× noise floor — hier IS producing different sim")
    else:
        print(f"  ✗ pred_L ≈ pred_H — possible hier bug or saturation extreme")

    # ===== Verdict =====
    print(f"\n{'='*72}")
    print(f"VERDICT")
    print(f"{'='*72}")
    print(f"  CHECK 1 (stiff fields differ):  Δ_mean = {np.abs(stiff_diff).mean():.4f}")
    print(f"  CHECK 2 (sim sensitivity):       cavity span = {(sim_outputs[1.5]['cavity']-sim_outputs[0.3]['cavity'])*100:.2f}%")
    print(f"  CHECK 3 (pred L vs H diff):     mean = {diff_per_particle.mean():.3e} m")
    print(f"")
    print(f"  Interpretation:")
    if np.abs(stiff_diff).mean() > 0.01 and diff_per_particle.mean() > 1e-6:
        print(f"  ✓ Hier IS modifying stiff field AND producing different sim")
        print(f"  ✓ But the sim difference may be small in absolute terms (< human perception)")
        print(f"  → Visual indistinguishability is a SENSITIVITY limit of forward sim, not bug")
    elif np.abs(stiff_diff).mean() < 1e-4:
        print(f"  ✗ POSSIBLE BUG: hier not actually changing stiff field much")
    elif diff_per_particle.mean() < 1e-8:
        print(f"  ✗ POSSIBLE BUG: sim ignoring stiff field difference")
    else:
        print(f"  ? Mixed signals, need more investigation")


if __name__ == "__main__":
    main()
