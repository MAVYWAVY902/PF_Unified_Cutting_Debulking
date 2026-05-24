"""Inner lr_pos sweep — discover if our 5e-3 is undertuned vs Liang's 50.

Background: Liang paper §II-B says inner GD uses lr=50 for 30 steps. Their
Δ is physical position in meters. Our setup: Δx is also in meters (cube 0.3m³),
but we use lr_pos=5e-3 — 10⁴ smaller than Liang's value.

Prior inner self-test showed Δ_GD ≈ 25% × Δ_closed-form at 30 steps, consistent
with lr_pos × Hessian × 30 ≈ 30% (GD's geometric convergence rate). If we increase
lr_pos to Liang scale, GD should reach closed-form.

Tested lr_pos: 5e-3 (current), 0.5, 5, 50
Tested at:     θ_init baseline = (5, 15, 0.6, 0.7)

For each: measure ||Δ_GD - Δ_cf|| / ||Δ_cf||  (close-to-0 = converged)
Also: L_inner drop magnitude.
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
    setup_solver_with_rest,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day8_stable_inner_loop import (  # noqa: E402
    gd_inner_loop_stable, SHARPNESS_INNER, TAU_INNER,
)
from tools.day6f_v0_multistep_synth import make_knife_trajectory  # noqa: E402


def closed_form_with_constants(pos_pred, dv_pred, pos_target, alpha_target,
                                  s_inner, tau_inner):
    delta_x = pos_target - pos_pred
    eps = 1e-7
    alpha_safe = torch.clamp(alpha_target, eps, 1 - eps)
    logit_obs = torch.log(alpha_safe / (1 - alpha_safe))
    delta_dv_unclamped = tau_inner - logit_obs / s_inner - dv_pred
    delta_dv = torch.maximum(delta_dv_unclamped, -dv_pred)
    return delta_x, delta_dv


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, _ = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  Cube extent (meters): 0.3 × 0.3 × 0.3")
    print(f"Inner constants: SHARPNESS_INNER={SHARPNESS_INNER}, TAU_INNER={TAU_INNER}")

    device = "cuda:0"
    knife_radius = 0.04
    knife_traj = make_knife_trajectory(3)
    theta_true_4d = np.array([3.0, 20.0, 0.5, 0.85], dtype=np.float32)
    theta_init_4d = np.array([5.0, 15.0, 0.6, 0.7], dtype=np.float32)

    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_rest_t = torch.tensor(solver.e["pf_rest"].numpy(),
                                  dtype=torch.float32, device=device)

    # Generate obs at θ_true (frame 0)
    theta_obs_torch = torch.tensor(theta_true_4d, dtype=torch.float32, device=device,
                                     requires_grad=False)
    with torch.no_grad():
        alpha_obs, pos_obs, _ = WarpFullForwardOp.apply(
            theta_obs_torch, solver, rest_state, knife_traj[0], knife_radius,
        )

    # Forward at θ_init
    theta_eval_torch = torch.tensor(theta_init_4d, dtype=torch.float32, device=device,
                                       requires_grad=False)
    with torch.no_grad():
        _, pos_eval, dv_eval = WarpFullForwardOp.apply(
            theta_eval_torch, solver, rest_state, knife_traj[0], knife_radius,
        )

    # Closed-form (using TAU_INNER constants)
    with torch.no_grad():
        delta_x_cf, delta_dv_cf = closed_form_with_constants(
            pos_eval, dv_eval, pos_obs, alpha_obs,
            SHARPNESS_INNER, TAU_INNER,
        )
    cf_dx_norm = float(delta_x_cf.norm())
    cf_ddv_norm = float(delta_dv_cf.norm())
    print(f"\nClosed-form target:")
    print(f"  ‖Δx_cf‖  = {cf_dx_norm:.4e}  (physical meters; cube spans 0.3m)")
    print(f"  ‖Δdv_cf‖ = {cf_ddv_norm:.4e}  (damage units)")

    # Sweep lr_pos
    print(f"\n{'='*72}")
    print(f"lr_pos sweep at θ_init = (5, 15, 0.6, 0.7), 30 inner GD steps")
    print(f"lr_dv kept adaptive (1/s² = 0.0025 since s=20)")
    print(f"{'='*72}")
    print(f"\n  {'lr_pos':>10s}  {'L_init':>12s}  {'L_final':>12s}  {'drop':>8s}  "
          f"{'rel(Δx)':>10s}  {'rel(Δdv)':>10s}  {'mono?':>6s}")

    for lr_pos in [5e-3, 0.5, 5.0, 50.0]:
        delta_x_gd, delta_dv_gd, info = gd_inner_loop_stable(
            pos_eval, dv_eval, pos_obs, alpha_obs,
            theta_eval_torch.detach(),
            edges_a_t, edges_b_t, edges_rest_t,
            n_iter=30, lr_pos=lr_pos, lr_dv_factor=1.0,
            lambda_cav=1.0, lambda_E=0.0,
        )
        L_init = info["L_init"]
        L_final = info["L_final"]
        L_drop = L_init / max(L_final, 1e-30)
        L_history = info["L_history"]
        n_incr = sum(1 for i in range(1, len(L_history))
                      if L_history[i] > L_history[i-1] * 1.001)
        rel_x = float((delta_x_gd.detach() - delta_x_cf).norm()) / max(cf_dx_norm, 1e-12)
        rel_dv = float((delta_dv_gd.detach() - delta_dv_cf).norm()) / max(cf_ddv_norm, 1e-12)
        print(f"  {lr_pos:10.4g}  {L_init:12.4e}  {L_final:12.4e}  "
              f"{L_drop:8.1f}×  {rel_x*100:9.2f}%  {rel_dv*100:9.2f}%  "
              f"{'✓' if n_incr<3 else '✗ (' + str(n_incr) + ')':>6s}")

    print(f"\n{'='*72}")
    print(f"Interpretation:")
    print(f"  - rel(Δx) ≈ 0%  → GD matches closed-form (good)")
    print(f"  - rel(Δx) ≈ 100% → GD didn't move (lr too small)")
    print(f"  - rel(Δx) >> 100% → GD overshoots (lr too big)")
    print(f"  - mono ✗ → GD diverges (lr way too big)")


if __name__ == "__main__":
    main()
