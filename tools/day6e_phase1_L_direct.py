"""Day 6e Phase 1: synth-to-synth θ recovery on cube_tet with L_direct.

Hypothesis chain:
  - Day 6b: WarpFullForwardOp gradient correct (FD < 0.04%)
  - Day 6d v2: end-to-end chain with inner GD gradient correct (FD < 5%)
  - Day 6e Phase 1: does *outer Adam* using that gradient actually recover θ_true?

Loss:
  L_direct = ‖pos(θ) − pos_target‖² + λ_cav · ‖α(θ) − α_target‖²

This is the simplest possible outer loss — no inner loop, no residual mapping.
If this doesn't recover θ_true, no inner-loop variant will. If it does, we
have a working baseline to compare L_gap against in Phase 2.

θ_true = (3.0, 20.0, 0.5)      # ground truth (same as Day 6d_v2)
θ_init = (5.0, 15.0, 0.6)      # close-init: +67% / -25% / +20%

PASS criteria:
  - tight:   per-dim rel_err < 5%
  - partial: per-dim rel_err < 20%
  - FAIL:    some param drifts away from truth or NaN
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pf_shape_matching_warp"))

import torch
import warp as wp
wp.init()
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

from tools.day2_solver_wrapper import (  # noqa: E402
    setup_solver_with_rest, solver_load_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402


# Clamp ranges — keep θ physical. α_min here is the EDGE SDF gain k in
# σ(-k · sdist/r), bounded only by wp.clamp(exp_arg,-50,50) in the kernel.
# Typical values 1-10; cap at 50 to match kernel's own clamp.
THETA_BOUNDS = {
    "alpha_min":  (0.1,  50.0),
    "sharpness":  (1.0,  50.0),
    "threshold":  (0.05, 2.0),
}


def clamp_theta_inplace(theta_tensor):
    """Clamp θ in-place to physical ranges via .data (avoids autograd)."""
    with torch.no_grad():
        theta_tensor.data[0].clamp_(*THETA_BOUNDS["alpha_min"])
        theta_tensor.data[1].clamp_(*THETA_BOUNDS["sharpness"])
        theta_tensor.data[2].clamp_(*THETA_BOUNDS["threshold"])


def generate_obs(solver, rest_state, theta_true, knife_pos, knife_radius):
    """Run forward at θ_true via legacy float path → (pos_target, alpha_target)."""
    solver.set_theta_wp(None)
    solver.alpha_min = float(theta_true[0])
    solver.soft_removal_sharpness = float(theta_true[1])
    solver.aspiration_threshold = float(theta_true[2])
    solver.knife_radius = knife_radius
    solver.knife_active = 1
    solver.knife_pos = wp.array(np.array([knife_pos], dtype=np.float32),
                                  dtype=wp.vec3, device=solver.device,
                                  requires_grad=False)
    solver_load_state(solver, rest_state)
    solver.step()
    if solver.use_ssa_soft_removal:
        ci = solver._ssa_cycle_index
        alpha_np = solver._ssa_alpha_buffers[ci].numpy().copy()
    else:
        alpha_np = solver.p["alpha"].numpy().copy()
    pos_np = solver.p["pos"].numpy().copy()
    return pos_np, alpha_np


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_pos = np.array([0.15, 0.30, 0.15], dtype=np.float32)
    knife_radius = 0.04

    theta_true = np.array([3.0, 20.0, 0.5], dtype=np.float32)
    theta_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)
    param_names = ["α_min", "sharpness", "τ"]

    # ---- Generate observation target ----
    print(f"\n[Setup] θ_true = ({theta_true[0]}, {theta_true[1]}, {theta_true[2]})")
    print(f"[Setup] θ_init = ({theta_init[0]}, {theta_init[1]}, {theta_init[2]})")
    print(f"[Setup] init perturbation: "
          f"α_min {(theta_init[0]-theta_true[0])/theta_true[0]*100:+.1f}%, "
          f"sharp {(theta_init[1]-theta_true[1])/theta_true[1]*100:+.1f}%, "
          f"τ {(theta_init[2]-theta_true[2])/theta_true[2]*100:+.1f}%")

    pos_target_np, alpha_target_np = generate_obs(
        solver, rest_state, theta_true, knife_pos, knife_radius)
    cavity_frac = float((alpha_target_np < 0.5).mean())
    pos_norm = float(np.linalg.norm(pos_target_np))
    print(f"[Setup] target cavity fraction: {cavity_frac:.4f}")
    print(f"[Setup] ‖pos_target‖ = {pos_norm:.2f}")

    pos_target_t = torch.tensor(pos_target_np, dtype=torch.float32, device=device)
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)

    # ---- Choose λ_cav so two terms are comparable at θ_init (value-balance) ----
    print("\n[Probe] computing L_pos / L_cav scales at θ_init …")
    theta_probe = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                 requires_grad=False)
    with torch.no_grad():
        alpha_p, pos_p, dv_p = WarpFullForwardOp.apply(
            theta_probe, solver, rest_state, knife_pos, knife_radius,
        )
        L_pos_probe = float(((pos_p - pos_target_t) ** 2).sum())
        L_cav_probe = float(((alpha_p - alpha_target_t) ** 2).sum())
    print(f"  L_pos(θ_init) = {L_pos_probe:.4e}")
    print(f"  L_cav(θ_init) = {L_cav_probe:.4e}")
    lambda_cav = max(L_pos_probe / max(L_cav_probe, 1e-12), 1e-6)
    print(f"  → λ_cav = {lambda_cav:.4e}  (value-balance L_pos ≈ λ_cav · L_cav)")

    # ---- Outer Adam loop ----
    print("\n[Recover] outer Adam loop, L_direct = L_pos + λ_cav · L_cav")
    print(f"  N_outer = 50, lr = 0.10")

    theta_torch = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                 requires_grad=True)
    opt = torch.optim.Adam([theta_torch], lr=0.10)

    trajectory = []
    losses = []
    L_pos_hist = []
    L_cav_hist = []

    n_outer = 50
    print(f"\n  {'step':>4s}  {'L_outer':>12s}  {'L_pos':>12s}  {'L_cav':>12s}  "
          f"{'α_min':>8s}  {'sharp':>8s}  {'τ':>8s}  {'grad α':>10s}  "
          f"{'grad s':>10s}  {'grad τ':>10s}")
    for step in range(n_outer):
        alpha, pos, dv = WarpFullForwardOp.apply(
            theta_torch, solver, rest_state, knife_pos, knife_radius,
        )
        L_pos = ((pos - pos_target_t) ** 2).sum()
        L_cav = ((alpha - alpha_target_t) ** 2).sum()
        L_outer = L_pos + lambda_cav * L_cav

        opt.zero_grad()
        L_outer.backward()
        grad_vals = theta_torch.grad.detach().cpu().numpy().copy()
        opt.step()
        clamp_theta_inplace(theta_torch)

        th = theta_torch.detach().cpu().numpy().copy()
        trajectory.append(th)
        losses.append(float(L_outer.item()))
        L_pos_hist.append(float(L_pos.item()))
        L_cav_hist.append(float(L_cav.item()))

        if step < 5 or step % 5 == 4 or step == n_outer - 1:
            print(f"  {step:>4d}  {L_outer.item():>12.4e}  {L_pos.item():>12.4e}  "
                  f"{L_cav.item():>12.4e}  {th[0]:>8.4f}  {th[1]:>8.4f}  "
                  f"{th[2]:>8.4f}  {grad_vals[0]:>+10.3e}  "
                  f"{grad_vals[1]:>+10.3e}  {grad_vals[2]:>+10.3e}")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("Day 6e Phase 1 — L_direct synth-to-synth verdict")
    print("=" * 70)
    theta_final = trajectory[-1]
    rel_err = np.abs(theta_final - theta_true) / np.maximum(np.abs(theta_true), 1e-12)
    L_init = losses[0]; L_final = losses[-1]
    print(f"  L_outer: {L_init:.4e} → {L_final:.4e}  "
          f"(reduction {L_init/max(L_final,1e-12):.2f}×)")
    print(f"  L_pos  : {L_pos_hist[0]:.4e} → {L_pos_hist[-1]:.4e}")
    print(f"  L_cav  : {L_cav_hist[0]:.4e} → {L_cav_hist[-1]:.4e}")
    print(f"\n  per-dim recovery:")
    for d, name in enumerate(param_names):
        print(f"    {name:>10s}: true={theta_true[d]:.4f}  init={theta_init[d]:.4f}  "
              f"final={theta_final[d]:.4f}  rel_err={rel_err[d]*100:.2f}%")

    pass_tight = bool((rel_err < 0.05).all())
    pass_partial = bool((rel_err < 0.20).all())
    monotone = (L_final < L_init)
    if pass_tight and monotone:
        print(f"\n✓ Day 6e Phase 1 PASS (tight) — all params within 5%, L decreasing")
    elif pass_partial and monotone:
        print(f"\n⚠ Day 6e Phase 1 PARTIAL — params within 20% but not 5%")
    else:
        print(f"\n✗ Day 6e Phase 1 FAIL — params drift or L not decreasing")
        # Diagnostics
        for d, name in enumerate(param_names):
            if rel_err[d] > 0.20:
                drift = "OVERSHOOT" if abs(theta_final[d] - theta_true[d]) > abs(theta_init[d] - theta_true[d]) else "STUCK"
                print(f"    {name}: {drift} (init err {abs(theta_init[d]-theta_true[d])/theta_true[d]*100:.1f}% → "
                      f"final err {rel_err[d]*100:.1f}%)")

    # Save trajectory for inspection
    npz_path = REPO_ROOT / "tools" / "day6e_phase1_trajectory.npz"
    np.savez(npz_path,
             trajectory=np.array(trajectory),
             losses=np.array(losses),
             L_pos=np.array(L_pos_hist),
             L_cav=np.array(L_cav_hist),
             theta_true=theta_true,
             theta_init=theta_init,
             lambda_cav=lambda_cav,
             rel_err_final=rel_err)
    print(f"\n  → trajectory saved to {npz_path}")


if __name__ == "__main__":
    main()
