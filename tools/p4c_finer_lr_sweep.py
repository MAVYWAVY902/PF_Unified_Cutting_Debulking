"""P4c: Finer lr sweep around 0.03 with sigmoid bounding + best-epoch tracking.

Phase 1 (P4b) found lr=0.03 best among {0.03, 0.1, 0.3}.
Phase 2 showed Adam drifts past best (ep 30 best, ep 59 worse).

This sweep: lr ∈ {0.01, 0.02, 0.03, 0.05, 0.07} × 30 epoch.
Tracks per-epoch metrics, reports BEST epoch's stiff_in (vs truth 0.8).

Sigmoid bound: stiff ∈ [0.05, 2.0], theta=(3, 100, 0.3).
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))

import torch
import warp as wp
wp.init()

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa
from tools.day8_stable_inner_loop import gd_inner_loop_stable, SHARPNESS_INNER, TAU_INNER  # noqa
from tools.day6f_v0_multistep_synth import make_knife_trajectory  # noqa
from tools.stageA_inclusion_recovery import (  # noqa
    build_inclusion_truth, gen_obs_per_particle_stiff, compute_metrics,
)
from tools.p4b_sigmoid_bound_lr_sweep import (  # noqa
    STIFF_LOW, STIFF_HIGH, inv_sigmoid_to_z, z_to_stiff,
)


def run_with_tracking(solver, rest_state, knife_traj, knife_radius,
                        obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
                        theta_3d_np, stiff_init_np, stiff_truth_np, inclusion_mask,
                        device, N_epochs, lr, n_inner=30):
    """Run + track per-epoch stiff_in mean and L_gap."""
    T = len(knife_traj)
    theta_torch = torch.tensor(theta_3d_np, dtype=torch.float32, device=device,
                                 requires_grad=False)
    z_init = inv_sigmoid_to_z(stiff_init_np, STIFF_LOW, STIFF_HIGH)
    z_torch = torch.tensor(z_init, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([z_torch], lr=lr)

    truth_in = float(stiff_truth_np[inclusion_mask].mean())
    truth_out = float(stiff_truth_np[~inclusion_mask].mean())

    trajectory = []  # list of dict per epoch
    for epoch in range(N_epochs):
        state = rest_state
        L_gaps = []
        for t in range(T):
            stiff_pp_torch = z_to_stiff(z_torch)
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state, knife_traj[t], knife_radius,
                stiff_pp_torch,
            )
            delta_x, delta_dv, _ = gd_inner_loop_stable(
                pos_t, dv_t, obs_pos_t[t], obs_alpha_t[t],
                theta_torch, edges_a_t, edges_b_t, edges_rest_t,
                n_iter=n_inner, lr_pos=0.5, lr_dv_factor=1.0,
                lambda_cav=1.0, lambda_E=0.0,
            )
            L_gap = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
            opt.zero_grad()
            L_gap.backward()
            if torch.isnan(z_torch.grad).any():
                return trajectory  # bail out
            opt.step()
            L_gaps.append(float(L_gap.item()))
            state = solver_save_state(solver)
            dx_np = delta_x.detach().cpu().numpy()
            ddv_np = delta_dv.detach().cpu().numpy()
            state["pos"] = state["pos"] + dx_np
            state["particle_damage"] = np.maximum(
                state["particle_damage"] + ddv_np, 0.0)
            new_dv = state["particle_damage"]
            state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(
                SHARPNESS_INNER * (new_dv - TAU_INNER), -50, 50)))
        # Per-epoch metrics
        stiff_now = z_to_stiff(z_torch).detach().cpu().numpy()
        in_mean = float(stiff_now[inclusion_mask].mean())
        out_mean = float(stiff_now[~inclusion_mask].mean())
        err_in = abs(in_mean - truth_in) / truth_in * 100
        trajectory.append(dict(
            epoch=epoch, mean_L_gap=float(np.mean(L_gaps)),
            stiff_in=in_mean, stiff_out=out_mean, err_in=err_in,
        ))
    return trajectory


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = 2
    device = "cuda:0"
    knife_radius = 0.08
    T = 10
    knife_traj = make_knife_trajectory(T)
    theta_3D = np.array([3.0, 100.0, 0.3], dtype=np.float32)

    pos_rest = mesh_info["vertices"]
    stiff_truth, inclusion_mask, center, incl_r = build_inclusion_truth(
        pos_rest, stiff_in=0.8, stiff_out=0.3,
        center_override=(0.15, 0.25, 0.15),
        radius_override=0.06)

    print(f"\n[Setup] T={T} theta=(3,100,0.3) sigmoid bound [{STIFF_LOW},{STIFF_HIGH}]")
    print(f"[Truth] stiff_in=0.8, stiff_out=0.3")
    print(f"[Init]  stiff uniform 0.5")

    print(f"\nGenerating obs ...")
    obs_pos, obs_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_3D, stiff_truth, knife_traj, knife_radius, device)
    obs_pos_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                                dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                                dtype=torch.long, device=device)
    edge_rest_len = np.linalg.norm(
        pos_rest[edges_a_t.cpu().numpy()] - pos_rest[edges_b_t.cpu().numpy()], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)
    stiff_init = np.full(N, 0.5, dtype=np.float32)

    # ========== Sweep ==========
    lr_grid = [0.01, 0.02, 0.03, 0.05, 0.07]
    print(f"\n{'='*78}\nSweep: lr ∈ {lr_grid} × 30 epoch\n{'='*78}")

    all_results = {}
    for lr in lr_grid:
        print(f"\n  ── lr = {lr} ─────────────────")
        traj = run_with_tracking(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
            theta_3D, stiff_init, stiff_truth, inclusion_mask, device,
            N_epochs=30, lr=lr,
        )
        all_results[lr] = traj
        # Best epoch by min err_in
        if traj:
            best = min(traj, key=lambda d: d["err_in"])
            final = traj[-1]
            print(f"    BEST ep {best['epoch']:>2d}: stiff_in={best['stiff_in']:.3f}  err={best['err_in']:.1f}%  L_gap={best['mean_L_gap']:.2e}")
            print(f"    FINAL ep{final['epoch']:>2d}: stiff_in={final['stiff_in']:.3f}  err={final['err_in']:.1f}%  L_gap={final['mean_L_gap']:.2e}")
            drift = final['err_in'] - best['err_in']
            print(f"    Drift from best to final: +{drift:.1f}%  ({'over' if final['stiff_in'] > best['stiff_in'] else 'under'})")

    # ========== Summary ==========
    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    print(f"  lr      best_ep   stiff_in_best   err_best   stiff_in_final   err_final   drift")
    print(f"  ────────────────────────────────────────────────────────────────────────────────")
    for lr, traj in all_results.items():
        if not traj:
            print(f"  {lr:.3f}   (NaN bail out)")
            continue
        best = min(traj, key=lambda d: d["err_in"])
        final = traj[-1]
        drift = final['err_in'] - best['err_in']
        print(f"  {lr:.3f}     {best['epoch']:>3d}     {best['stiff_in']:.3f}        "
              f"{best['err_in']:>5.2f}%      {final['stiff_in']:.3f}         "
              f"{final['err_in']:>5.2f}%     +{drift:.1f}%")

    # Best lr by min best-epoch err
    valid = {lr: traj for lr, traj in all_results.items() if traj}
    if valid:
        best_lr = min(valid, key=lambda lr: min(t["err_in"] for t in valid[lr]))
        best_traj = valid[best_lr]
        best_pt = min(best_traj, key=lambda d: d["err_in"])
        print(f"\n  ★ Best (by min err at any epoch): lr={best_lr}, ep={best_pt['epoch']}")
        print(f"     stiff_in = {best_pt['stiff_in']:.3f} (truth 0.8, err {best_pt['err_in']:.2f}%)")

    # Save
    np.savez(REPO / "tools" / "p4c_lr_sweep_results.npz",
              lr_grid=lr_grid,
              **{f"lr_{lr}_stiff_in": [d['stiff_in'] for d in all_results.get(lr, [])]
                 for lr in lr_grid},
              **{f"lr_{lr}_err_in": [d['err_in'] for d in all_results.get(lr, [])]
                 for lr in lr_grid},
              **{f"lr_{lr}_L_gap": [d['mean_L_gap'] for d in all_results.get(lr, [])]
                 for lr in lr_grid})
    print(f"\n  Saved: p4c_lr_sweep_results.npz")


if __name__ == "__main__":
    main()
