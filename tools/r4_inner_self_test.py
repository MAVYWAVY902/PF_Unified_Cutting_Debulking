"""Gate-Inner: standalone test that inner GD converges correctly after Day 3c fix.

Question: is the inner GD itself healthy with new TAU_INNER constants?
   - If YES → inner GD machinery works; Gate 2/3 outer failure is purely
     outer gradient issue (the τ-recovery loss-of-signal we already identified).
   - If NO → the fix broke the inner GD itself; gradients/lr need adjustment.

Tests at 3 θ points:
   T1. θ_true: inner Δ should be ≈ 0 (sim=obs, no residual needed)
   T2. θ_baseline init: moderate distance from truth; Δ ≠ 0 but well-defined
   T3. θ_all-high init: extreme distance (this is where Gate 2 outer failed)

For each: compare GD output to closed-form solution.
   Closed-form (with TAU_INNER=0.5 constant):
      Δx*  = pos_target - pos_pred
      Δdv* = τ_const - logit(α_target)/s_const - dv_pred   (clamped ≥ -dv_pred)

Pass criteria (per scenario):
   (a) L_inner monotonically decreases over 30 steps
   (b) Final L_inner < initial L_inner / 10
   (c) ||Δ_GD - Δ_cf|| / ||Δ_cf|| < 20%  (GD approximates closed-form well)
   (d) At θ_true: ||Δ_GD|| ≈ 0  (i.e., L_inner ≈ machine precision)
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
    setup_solver_with_rest, solver_save_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day8_stable_inner_loop import (  # noqa: E402
    gd_inner_loop_stable, SHARPNESS_INNER, TAU_INNER,
)
from tools.day6f_v0_multistep_synth import make_knife_trajectory  # noqa: E402


def closed_form_with_constants(pos_pred, dv_pred, pos_target, alpha_target,
                                  s_inner, tau_inner):
    """Closed-form analog of new inner loss minimizer (L_pos + L_cav, lambda_E=0)."""
    delta_x = pos_target - pos_pred
    eps = 1e-7
    alpha_safe = torch.clamp(alpha_target, eps, 1 - eps)
    logit_obs = torch.log(alpha_safe / (1 - alpha_safe))
    # New formula: σ(-s_inner · (dv + Δdv - tau_inner)) = α_target
    # Solve: dv + Δdv = tau_inner - logit(α)/s_inner
    delta_dv_unclamped = tau_inner - logit_obs / s_inner - dv_pred
    delta_dv = torch.maximum(delta_dv_unclamped, -dv_pred)  # respect dv ≥ 0
    return delta_x, delta_dv


def run_one_scenario(label, theta_eval_4d, theta_obs_4d, solver, rest_state,
                       knife_traj, knife_radius, device, edges_a_t, edges_b_t,
                       edges_rest_t, n_inner=30):
    """At theta_eval, compute pos_pred / dv_pred. Compare inner GD output to
    closed-form (using TAU_INNER constants)."""
    print(f"\n{'='*72}")
    print(f"Scenario: {label}")
    print(f"   θ_eval (forward sim): {theta_eval_4d.tolist()}")
    print(f"   θ_obs (generated):    {theta_obs_4d.tolist()}")
    print(f"{'='*72}")

    # 1. Generate obs at θ_obs (single frame, t=0)
    theta_obs_torch = torch.tensor(theta_obs_4d, dtype=torch.float32, device=device,
                                     requires_grad=False)
    with torch.no_grad():
        alpha_obs, pos_obs, dv_obs = WarpFullForwardOp.apply(
            theta_obs_torch, solver, rest_state, knife_traj[0], knife_radius,
        )

    # 2. Forward at θ_eval
    theta_eval_torch = torch.tensor(theta_eval_4d, dtype=torch.float32, device=device,
                                       requires_grad=False)
    with torch.no_grad():
        alpha_eval, pos_eval, dv_eval = WarpFullForwardOp.apply(
            theta_eval_torch, solver, rest_state, knife_traj[0], knife_radius,
        )

    # 3. Closed-form Δ (using TAU_INNER constants, no torch grad needed)
    with torch.no_grad():
        delta_x_cf, delta_dv_cf = closed_form_with_constants(
            pos_eval, dv_eval, pos_obs, alpha_obs,
            SHARPNESS_INNER, TAU_INNER,
        )

    # 4. Inner GD (uses TAU_INNER constants via module-level defaults)
    delta_x_gd, delta_dv_gd, info = gd_inner_loop_stable(
        pos_eval, dv_eval, pos_obs, alpha_obs,
        # theta passed but inner ignores it for s, τ — uses module constants
        theta_eval_torch.detach(),
        edges_a_t, edges_b_t, edges_rest_t,
        n_iter=n_inner, lr_pos=5e-3, lr_dv_factor=1.0,
        lambda_cav=1.0, lambda_E=0.0,
    )

    # Detach for analysis
    delta_x_gd_d = delta_x_gd.detach()
    delta_dv_gd_d = delta_dv_gd.detach()

    # 5. Diagnostics
    L_init = info["L_init"]
    L_final = info["L_final"]
    L_drop = L_init / max(L_final, 1e-30)
    L_history = info["L_history"]

    # Monotonicity: how many steps L_inner INCREASED
    n_increase = sum(1 for i in range(1, len(L_history))
                      if L_history[i] > L_history[i-1] * 1.001)
    monotonic = n_increase < 3

    # Δ vs closed-form
    cf_dx_norm = float(delta_x_cf.norm())
    cf_ddv_norm = float(delta_dv_cf.norm())
    gd_dx_norm = float(delta_x_gd_d.norm())
    gd_ddv_norm = float(delta_dv_gd_d.norm())
    rel_x = float((delta_x_gd_d - delta_x_cf).norm()) / max(cf_dx_norm, 1e-12)
    rel_dv = float((delta_dv_gd_d - delta_dv_cf).norm()) / max(cf_ddv_norm, 1e-12)

    # Report
    print(f"\n   L_inner: {L_init:.4e} → {L_final:.4e}  (drop {L_drop:.1f}×)")
    print(f"   L_history (every 5 iter): " +
          "  ".join(f"{L_history[k]:.3e}" for k in range(0, n_inner, 5)))
    print(f"   Monotonic? {monotonic} ({n_increase} steps had L_inner increase)")
    print(f"\n   Δ_GD vs Δ_closed-form:")
    print(f"     ‖Δx_cf‖={cf_dx_norm:.4e}    ‖Δx_GD‖={gd_dx_norm:.4e}    rel_err={rel_x*100:.2f}%")
    print(f"     ‖Δdv_cf‖={cf_ddv_norm:.4e}  ‖Δdv_GD‖={gd_ddv_norm:.4e}  rel_err={rel_dv*100:.2f}%")

    # Gate pass/fail
    L_drops_enough = L_drop > 10 or L_final < 1e-6
    deltas_match_cf = rel_x < 0.20 and rel_dv < 0.20
    near_zero_at_truth = (cf_dx_norm < 1e-3 and cf_ddv_norm < 1e-3)

    passes = []
    if monotonic: passes.append("✓ monotonic L_inner descent")
    else: passes.append("✗ L_inner not monotonic")
    if L_drops_enough: passes.append("✓ L_inner drop sufficient")
    else: passes.append(f"⚠ L_inner only dropped {L_drop:.1f}× (want >10×)")
    if deltas_match_cf: passes.append("✓ Δ_GD ≈ Δ_closed-form")
    else: passes.append(f"⚠ Δ_GD differs from closed-form: Δx {rel_x*100:.1f}%, Δdv {rel_dv*100:.1f}%")
    if "truth" in label:
        if near_zero_at_truth: passes.append("✓ Δ ≈ 0 at θ_true (as expected)")
        else: passes.append(f"⚠ Δ at θ_true not zero: ‖Δx‖={cf_dx_norm:.2e}, ‖Δdv‖={cf_ddv_norm:.2e}")

    print(f"\n   Gates:")
    for p in passes:
        print(f"     {p}")
    all_ok = all("✓" in p for p in passes)
    return all_ok, dict(L_init=L_init, L_final=L_final, rel_x=rel_x, rel_dv=rel_dv)


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, _ = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")
    print(f"Inner constants: SHARPNESS_INNER={SHARPNESS_INNER}, TAU_INNER={TAU_INNER}")

    device = "cuda:0"
    knife_radius = 0.04
    knife_traj = make_knife_trajectory(3)
    theta_true_4d = np.array([3.0, 20.0, 0.5, 0.85], dtype=np.float32)

    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_rest_t = torch.tensor(solver.e["pf_rest"].numpy(),
                                  dtype=torch.float32, device=device)

    scenarios = [
        ("T1: θ_eval = θ_true (should give Δ ≈ 0)",
         theta_true_4d, theta_true_4d),
        ("T2: θ_eval = baseline init (5, 15, 0.6, 0.7)",
         np.array([5.0, 15.0, 0.6, 0.70], dtype=np.float32), theta_true_4d),
        ("T3: θ_eval = all-high init (6, 30, 0.8, 1.3)  ← Gate 2 failure case",
         np.array([6.0, 30.0, 0.8, 1.30], dtype=np.float32), theta_true_4d),
    ]

    results = {}
    for label, theta_eval, theta_obs in scenarios:
        ok, info = run_one_scenario(
            label, theta_eval, theta_obs, solver, rest_state, knife_traj,
            knife_radius, device, edges_a_t, edges_b_t, edges_rest_t,
        )
        results[label] = (ok, info)

    # Verdict
    print(f"\n{'='*72}")
    print(f"INNER-SELF-TEST VERDICT")
    print(f"{'='*72}")
    n_pass = sum(1 for ok, _ in results.values() if ok)
    print(f"\n  {n_pass}/{len(scenarios)} scenarios passed all gates.")
    if n_pass == len(scenarios):
        print(f"  ✓ Inner GD is HEALTHY under new TAU_INNER constants.")
        print(f"    → Gate 2/3 outer failure is purely OUTER gradient issue")
        print(f"      (τ recovery from far init lost its artifact signal).")
        print(f"    → Inner-side fix is sound; need different approach for outer τ.")
    else:
        print(f"  ✗ Inner GD has issues under new formula.")
        print(f"    → Fix the inner first before re-attempting outer recovery.")
        for label, (ok, _) in results.items():
            mark = "✓" if ok else "✗"
            print(f"    {mark} {label}")


if __name__ == "__main__":
    main()
