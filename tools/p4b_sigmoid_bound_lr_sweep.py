"""P4b: Liang-style sigmoid parameter bounding + lr sweep.

Replaces hard clamp(stiff, 0.05, 5.0) with sigmoid reparameterization:
   z ∈ R unconstrained, optimized by Adam
   stiff = STIFF_LOW + (STIFF_HIGH - STIFF_LOW) * sigmoid(z)

This matches Liang 2023 Section III-B: "We impose bounds on our simulation
parameters with a sigmoid operation."

Two phases:
   1. Short sweep: lr ∈ {0.03, 0.1, 0.3} × 30 epochs, theta=(3,100,0.3)
      Pick lr with best mid-training stiff_in convergence + stability
   2. Full run: best lr × 60 epochs with 4 λ_smooth values

Theta=(3, 100, 0.3) Config B propagation.
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

from tools.day2_solver_wrapper import (  # noqa
    setup_solver_with_rest, solver_save_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa
from tools.day8_stable_inner_loop import (  # noqa
    gd_inner_loop_stable, SHARPNESS_INNER, TAU_INNER,
)
from tools.day6f_v0_multistep_synth import make_knife_trajectory  # noqa
from tools.stageA_inclusion_recovery import (  # noqa
    build_inclusion_truth, gen_obs_per_particle_stiff, compute_metrics,
)


# Sigmoid bounding constants
STIFF_LOW = 0.05
STIFF_HIGH = 2.0


def inv_sigmoid_to_z(s, lo, hi):
    """Inverse sigmoid map: stiff ∈ (lo, hi) → z ∈ R."""
    s_clamped = np.clip(s, lo + 1e-4, hi - 1e-4)
    return np.log((s_clamped - lo) / (hi - s_clamped)).astype(np.float32)


def z_to_stiff(z_torch):
    """Forward sigmoid: z ∈ R → stiff ∈ (lo, hi)."""
    return STIFF_LOW + (STIFF_HIGH - STIFF_LOW) * torch.sigmoid(z_torch)


def run_recovery_sigbound(solver, rest_state, knife_traj, knife_radius,
                            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
                            theta_3d_np, stiff_init_np, stiff_truth_np, inclusion_mask,
                            device, N_epochs, lr, n_inner=30, log_every=5,
                            lambda_smooth=0.0):
    """Adam on z (sigmoid-bounded stiff). θ_global frozen."""
    T = len(knife_traj)
    theta_torch = torch.tensor(theta_3d_np, dtype=torch.float32, device=device,
                                 requires_grad=False)
    z_init = inv_sigmoid_to_z(stiff_init_np, STIFF_LOW, STIFF_HIGH)
    z_torch = torch.tensor(z_init, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([z_torch], lr=lr)
    epoch_logs = []
    explosion_at = None

    for epoch in range(N_epochs):
        state = rest_state
        L_gaps = []
        for t in range(T):
            stiff_pp_torch = z_to_stiff(z_torch)  # NEW: derive stiff from z
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
            if lambda_smooth > 0:
                neighbor_diff = stiff_pp_torch[edges_a_t] - stiff_pp_torch[edges_b_t]
                L_smooth = neighbor_diff.abs().sum()
                L_total = L_gap + lambda_smooth * L_smooth
            else:
                L_total = L_gap
            opt.zero_grad()
            L_total.backward()
            if torch.isnan(z_torch.grad).any():
                print(f"  ✗ NaN at epoch {epoch}, t={t}")
                explosion_at = epoch
                return dict(stiff_final=z_to_stiff(z_torch).detach().cpu().numpy(),
                              epoch_logs=epoch_logs, exploded=True, explosion_at=explosion_at)
            opt.step()
            # NO CLAMP — sigmoid handles bounding
            L_gaps.append(float(L_gap.item()))
            # Mode C inject
            state = solver_save_state(solver)
            dx_np = delta_x.detach().cpu().numpy()
            ddv_np = delta_dv.detach().cpu().numpy()
            state["pos"] = state["pos"] + dx_np
            state["particle_damage"] = np.maximum(
                state["particle_damage"] + ddv_np, 0.0)
            new_dv = state["particle_damage"]
            state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(
                SHARPNESS_INNER * (new_dv - TAU_INNER), -50, 50)))
        mean_L = float(np.mean(L_gaps))
        # Detect explosion: L_gap jumps by 10× compared to previous
        if epoch >= 1 and len(epoch_logs) > 0:
            prev_L = epoch_logs[-1].get("mean_L_gap", mean_L)
            if mean_L > 10 * prev_L and prev_L > 0:
                print(f"  ⚠ L_gap explosion at ep {epoch}: {prev_L:.2e} → {mean_L:.2e}")
                explosion_at = epoch
        if epoch % log_every == 0 or epoch == N_epochs - 1:
            stiff_now = z_to_stiff(z_torch).detach().cpu().numpy()
            log = dict(epoch=epoch, mean_L_gap=mean_L,
                          stiff_snapshot=stiff_now.copy())
            epoch_logs.append(log)
            m = compute_metrics(stiff_now, stiff_truth_np, inclusion_mask)
            print(f"  ep {epoch:3d}: L_gap={mean_L:.3e}  "
                  f"L1_in={m['L1_inside']:.4f}  IoU={m['iou']:.3f}  "
                  f"stiff_in={stiff_now[inclusion_mask].mean():.3f}  "
                  f"stiff_out={stiff_now[~inclusion_mask].mean():.3f}")
    return dict(
        stiff_final=z_to_stiff(z_torch).detach().cpu().numpy(),
        epoch_logs=epoch_logs,
        exploded=(explosion_at is not None),
        explosion_at=explosion_at,
    )


def score_run(result, stiff_truth, inclusion_mask):
    """Score: prefer (no explosion + good stiff_in recovery)."""
    if result is None:
        return -1, "None"
    stiff_final = result["stiff_final"]
    stiff_in_mean = float(stiff_final[inclusion_mask].mean())
    truth_in = float(stiff_truth[inclusion_mask].mean())
    err = abs(stiff_in_mean - truth_in) / max(truth_in, 1e-6) * 100
    penalty = 100.0 if result.get("exploded", False) else 0.0
    score = -err - penalty
    note = f"in={stiff_in_mean:.3f}(truth {truth_in:.2f}, err {err:.1f}%){' EXPLODE' if result.get('exploded') else ''}"
    return score, note


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    N = solver.N

    # Step 1 + Step 2 setup
    solver.damping = 0.99
    solver.num_iterations = 2
    print(f"\nSolver: N={N}, iters=2, damping=0.99")

    device = "cuda:0"
    knife_radius = 0.08
    T = 10
    knife_traj = make_knife_trajectory(T)
    theta_3D = np.array([3.0, 100.0, 0.3], dtype=np.float32)  # Config B
    print(f"  Theta = (3, 100, 0.3)  ← Config B anti-zombie")
    print(f"  Sigmoid bound: stiff ∈ [{STIFF_LOW}, {STIFF_HIGH}]")

    # Ground truth same as Step 2
    pos_rest = mesh_info["vertices"]
    stiff_truth, inclusion_mask, center, incl_r = build_inclusion_truth(
        pos_rest, stiff_in=0.8, stiff_out=0.3,
        center_override=(0.15, 0.25, 0.15),
        radius_override=0.06)
    n_incl = int(inclusion_mask.sum())
    print(f"\n[Truth] Inclusion {n_incl}/{N}, in=0.8 out=0.3")

    # Obs
    print(f"\n[Obs] Generating T={T} ...")
    obs_pos, obs_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_3D, stiff_truth, knife_traj, knife_radius, device)
    cav = [f"{(a<0.5).mean():.3f}" for a in obs_alpha]
    print(f"  cavity per frame: {cav}")
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
    print(f"\n[Init] stiff uniform 0.5 → z_init = {inv_sigmoid_to_z(stiff_init[:1], STIFF_LOW, STIFF_HIGH)[0]:.3f}")

    # ========== Phase 1: Short lr sweep (30 epoch each) ==========
    print(f"\n{'='*72}\nPhase 1: lr sweep × 30 epochs (λ_smooth=0)\n{'='*72}")
    lr_grid = [0.03, 0.1, 0.3]
    results = {}
    for lr in lr_grid:
        print(f"\n  ── lr = {lr} ─────────────────")
        result = run_recovery_sigbound(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
            theta_3D, stiff_init, stiff_truth, inclusion_mask, device,
            N_epochs=30, lr=lr, n_inner=30, log_every=10, lambda_smooth=0.0,
        )
        results[lr] = result
        score, note = score_run(result, stiff_truth, inclusion_mask)
        print(f"    score = {score:.1f}  ({note})")

    # Pick best
    best_lr = max(lr_grid, key=lambda lr: score_run(results[lr], stiff_truth, inclusion_mask)[0])
    print(f"\n  Best lr: {best_lr}")

    # ========== Phase 2: Full run at best lr ==========
    print(f"\n{'='*72}\nPhase 2: full 60 epochs at lr={best_lr}, sweep λ_smooth\n{'='*72}")
    final_results = {}
    for lam in [0.0, 1e-6, 1e-4, 1e-2]:
        print(f"\n  ── λ_smooth = {lam} ─────────────────")
        result = run_recovery_sigbound(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
            theta_3D, stiff_init, stiff_truth, inclusion_mask, device,
            N_epochs=60, lr=best_lr, n_inner=30, log_every=10, lambda_smooth=lam,
        )
        final_results[lam] = result

    # ========== Summary ==========
    print(f"\n{'='*72}\nSUMMARY — sigmoid bound, theta=(3,100,0.3), lr={best_lr}\n{'='*72}")
    print(f"  λ_smooth   L1_total       L1_in      L1_out     IoU   stiff_in_mean  stiff_out_mean  exploded")
    print(f"  (truth)         0           0           0     1.0           0.800           0.300         /")
    for lam, result in final_results.items():
        stiff = result["stiff_final"]
        m = compute_metrics(stiff, stiff_truth, inclusion_mask)
        ex = "YES" if result.get("exploded") else "no"
        print(f"  {lam:.0e}    {m['L1_total']:>8.4f}    {m['L1_inside']:>8.4f}    "
              f"{m['L1_outside']:>8.4f}   {m['iou']:.3f}        {stiff[inclusion_mask].mean():>6.3f}          "
              f"{stiff[~inclusion_mask].mean():>6.3f}        {ex}")

    # Save best
    best_lam = max(final_results, key=lambda l: score_run(final_results[l], stiff_truth, inclusion_mask)[0])
    print(f"\n  Best (λ={best_lam}): IoU={compute_metrics(final_results[best_lam]['stiff_final'], stiff_truth, inclusion_mask)['iou']:.3f}")
    np.savez(REPO / "tools" / "stageA_inclusion_recovery_p4b.npz",
              positions=pos_rest, stiff_truth=stiff_truth, inclusion_mask=inclusion_mask,
              stiff_recovered=final_results[best_lam]["stiff_final"], stiff_init=stiff_init,
              best_lr=best_lr, best_lambda=best_lam)
    print(f"  Saved: stageA_inclusion_recovery_p4b.npz")


if __name__ == "__main__":
    main()
