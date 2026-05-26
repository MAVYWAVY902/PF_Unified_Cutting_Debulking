"""Day 8 stable GD inner loop + closed-form ground truth verification.

Fix the inner loop divergence found in T_fwd. Approach:

  Stable GD: lr_dv = c / s²  (theta-adaptive Hessian-aware lr)
             Hessian of L_cav w.r.t. Δ_dv ≈ 2(s·σ')² ≤ s²/8
             For stable GD: lr × Hess < 2 → lr_dv < 16/s²
             Conservative: lr_dv = 1/s² → lr·Hess ≤ 0.25 (fast convergence)

  Closed-form (synth-only ground truth):
             L_pos minimizer per-particle: Δ_x = pos_obs - pos_sim
             L_cav minimizer per-particle: Δ_dv = -logit(α_obs)/s - dv_sim + τ
             (Sigmoid inversion gives exact answer for decoupled L_inner.)
             Subject to dv_corr ≥ 0 (clamp Δ_dv ≥ -dv_sim)

Tests:
  T1: Closed-form is exact (L_gap < 1e-10 at θ_true)
  T2: Stable GD converges to closed-form result (within tolerance)
  T3: Stable GD across s ∈ {5, 20, 35} all stable + correct
  T4: Stable GD at θ_init (NOT truth) still gives meaningful Δ
  T5: L_gap(θ_true) ≈ 0 with stable GD (vs 0.84 with broken GD)

Note: closed-form is synth-only (needs per-particle alpha_obs).
Production inner solver is stable GD — extends to real phantom by
swapping L_inner (add L_F, surface mask) without touching the GD core.
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
    setup_solver_with_rest, solver_load_state, solver_save_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Day 3c (2026-05-24) — Liang-style inner constants (analog to k_c' in Eq. 5)
#
# Liang 2023 §II-B: "k_c' is an uniform user-defined weight matrix" — i.e.,
# the inner residual mapping g uses CONSTANT weights, kept separate from the
# outer-learned k. This preserves two-level optimization invariant:
#     ∂Δ*/∂(outer θ)  flows ONLY through forward sim x_t(θ), never directly.
#
# Before this fix, our inner L_cav used s = θ[1], τ = θ[2] from outer θ —
# violating Liang's invariant and creating asymmetric Hessian (τ had 350×
# more curvature than α/stiff due to dual-path gradient — see R4-v3 result).
#
# After fix: inner uses SHARPNESS_INNER, TAU_INNER as constants. Outer τ
# enters L_gap only through forward sim → pos_sim/dv_sim chain (single path,
# same as α_min and stiffness).
#
# Tuning: for synth-to-synth, TAU_INNER = truth (0.5); for phantom calibration,
# TAU_INNER = init guess (e.g., from Stage 2.0 literature value).
# ─────────────────────────────────────────────────────────────────────────────
SHARPNESS_INNER = 20.0   # = SHARPNESS_FIXED (Day 9 lock); inner uses constant
TAU_INNER       = 0.5    # user-defined; synth=truth, phantom=init-guess


def gd_inner_loop_stable(
    pos_pred, dv_pred, pos_target, alpha_target, theta,  # noqa: ARG001 (theta DEPRECATED)
    edges_a, edges_b, edges_rest,
    n_iter=30, lr_pos=5e-3, lr_dv_factor=1.0,
    lambda_cav=1.0, lambda_E=0.0,
    s_inner=None, tau_inner=None,
    surface_idx=None,
):
    """GD inner loop with Hessian-aware adaptive lr_dv (stable across all s).

    Args:
        pos_pred, dv_pred       : Current sim state (autograd).
        pos_target, alpha_target: Observation targets (no grad).
        theta                   : DEPRECATED since Day 3c (2026-05-24). Function
                                  IGNORES this argument. Kept in 5th positional
                                  slot for backward compat with 17+ legacy callers.
                                  Use s_inner / tau_inner kwargs instead.
                                  Can pass None or any tensor — value discarded.
        edges_a, edges_b, edges_rest: Edge topology + rest lengths (for L_E).
        n_iter, lr_pos, lr_dv_factor : GD hyperparams.
        lambda_cav, lambda_E         : Loss weights.
        s_inner, tau_inner           : Inner sigmoid sharpness + threshold.
                                       Pass as Python float (via .item() detach
                                       from outer θ) to sever autograd path.
                                       Defaults to SHARPNESS_INNER, TAU_INNER
                                       module constants if None.
        surface_idx             : Optional surface-particle mask (Test 1/2).

    lr_dv = lr_dv_factor / s²   (s is now CONSTANT, no .detach() needed).
       For stable convergence: lr × Hessian < 2.
       Hessian ≈ 2(s·σ_max')² ≤ s²/8, so lr < 16/s².
       lr_dv_factor=1.0 → lr·Hess ≤ 0.25, fast convergence in ~10-20 iter.
    """
    # theta arg: DEPRECATED (Day 3c lock). Explicitly discard so reader knows
    # it's ignored. New callers should pass None; legacy callers still pass
    # theta_torch positionally — function ignores either way.
    del theta  # noqa: F841

    N = pos_pred.shape[0]
    # Day 3c: inner uses CONSTANTS (Liang-style k_c' analog), not outer θ.
    s_val = float(SHARPNESS_INNER if s_inner is None else s_inner)
    tau_val = float(TAU_INNER if tau_inner is None else tau_inner)
    # lr_dv now uses constant s (no torch tensor / no .detach())
    lr_dv = lr_dv_factor / (s_val ** 2)

    delta_x = torch.zeros((N, 3), dtype=pos_pred.dtype, device=pos_pred.device,
                            requires_grad=True)
    delta_dv = torch.zeros(N, dtype=pos_pred.dtype, device=pos_pred.device,
                              requires_grad=True)
    L_init = None
    L_history = []
    for k in range(n_iter):
        pos_corr = pos_pred + delta_x
        dv_corr = torch.relu(dv_pred + delta_dv)
        # Day 3c: s_val, tau_val are constants — no autograd to outer θ here.
        alpha_corr = torch.sigmoid(-s_val * (dv_corr - tau_val))
        diff_pos = pos_corr - pos_target
        diff_alpha = alpha_corr - alpha_target
        # Test 1 (phantom-readiness): if surface_idx is provided, restrict L_pos
        # to surface particles only (camera-visible). Interior particles' Δx
        # then has NO direct signal — relies on L_E (Test 2) to propagate.
        if surface_idx is None:
            L_pos = (diff_pos ** 2).sum()
        else:
            L_pos = (diff_pos[surface_idx] ** 2).sum()
        L_cav = (diff_alpha ** 2).sum()
        if lambda_E > 0:
            edge_vec = pos_corr[edges_a] - pos_corr[edges_b]
            edge_len = edge_vec.norm(dim=1).clamp(min=1e-12)
            L_E = ((edge_len - edges_rest) ** 2).sum()
        else:
            L_E = torch.tensor(0.0, device=pos_pred.device)
        L_inner = L_pos + lambda_cav * L_cav + lambda_E * L_E
        if L_init is None:
            L_init = float(L_inner)
        L_history.append(float(L_inner))
        grad_x, grad_dv = torch.autograd.grad(
            L_inner, (delta_x, delta_dv),
            create_graph=True, retain_graph=True,
        )
        delta_x = delta_x - lr_pos * grad_x
        delta_dv = delta_dv - lr_dv * grad_dv
    return delta_x, delta_dv, dict(
        L_init=L_init, L_final=float(L_inner),
        lr_dv=float(lr_dv), L_history=L_history,
    )


def closed_form_inner(pos_pred, dv_pred, pos_target, alpha_target, theta):
    """Closed-form inner solver (synth-only ground truth).

    Both L_pos and L_cav decouple per-particle and have analytical minimizers:
      Δ_x_i* = pos_obs_i − pos_sim_i
      Δ_dv_i* = -(1/s)·logit(α_obs_i) − dv_sim_i + τ
    Subject to dv_corr ≥ 0 → Δ_dv ≥ -dv_sim.
    """
    s = theta[1]; tau = theta[2]
    delta_x = pos_target - pos_pred
    eps = 1e-7
    alpha_safe = torch.clamp(alpha_target, eps, 1 - eps)
    logit_obs = torch.log(alpha_safe / (1 - alpha_safe))
    delta_dv_unclamped = -logit_obs / s - dv_pred + tau
    delta_dv = torch.maximum(delta_dv_unclamped, -dv_pred)
    return delta_x, delta_dv


def gen_obs_param_arr(solver, rest_state, theta_true, knife_pos, knife_radius, device):
    """Single-frame obs at theta_true via param_arr path (same as sim)."""
    theta_torch = torch.tensor(theta_true, dtype=torch.float32, device=device,
                                 requires_grad=False)
    with torch.no_grad():
        alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
            theta_torch, solver, rest_state, knife_pos, knife_radius,
        )
    return pos_t.cpu().numpy().copy(), alpha_t.cpu().numpy().copy(), dv_t.cpu().numpy().copy()


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_pos = np.array([0.15, 0.30, 0.15], dtype=np.float32)
    knife_radius = 0.04

    # ============ Generate obs at θ_true=(3,20,0.5) ============
    theta_true = np.array([3.0, 20.0, 0.5], dtype=np.float32)
    pos_target_np, alpha_target_np, dv_target_np = gen_obs_param_arr(
        solver, rest_state, theta_true, knife_pos, knife_radius, device)

    pos_target_t = torch.tensor(pos_target_np, dtype=torch.float32, device=device)
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_rest_t = torch.tensor(solver.e["pf_rest"].numpy(),
                                  dtype=torch.float32, device=device)

    # ============ T1: closed-form at θ_true should give Δ ≈ 0 ============
    print(f"\n[T1] Closed-form at θ_true (sim == obs, Δ should be ≈ 0)")
    theta_torch = torch.tensor(theta_true, dtype=torch.float32, device=device,
                                 requires_grad=False)
    with torch.no_grad():
        alpha_sim, pos_sim, dv_sim = WarpFullForwardOp.apply(
            theta_torch, solver, rest_state, knife_pos, knife_radius,
        )
    delta_x_cf, delta_dv_cf = closed_form_inner(
        pos_sim, dv_sim, pos_target_t, alpha_target_t, theta_torch,
    )
    Lx_cf = float((delta_x_cf ** 2).sum())
    Ldv_cf = float((delta_dv_cf ** 2).sum())
    print(f"  closed-form Δ at θ_true: ‖Δx‖²={Lx_cf:.4e}  ‖Δdv‖²={Ldv_cf:.4e}  "
          f"L_gap={Lx_cf+Ldv_cf:.4e}")
    cf_clean = (Lx_cf + Ldv_cf) < 1e-6
    print(f"  {'✓ closed-form L_gap < 1e-6 at truth' if cf_clean else '✗ closed-form not clean!'}")

    # ============ T2: stable GD vs closed-form at θ_true ============
    print(f"\n[T2] Stable GD vs closed-form at θ_true (should match)")
    delta_x_gd, delta_dv_gd, info_gd = gd_inner_loop_stable(
        pos_sim, dv_sim, pos_target_t, alpha_target_t, theta_torch,
        edges_a_t, edges_b_t, edges_rest_t,
        n_iter=30, lr_pos=5e-3, lr_dv_factor=1.0,
        lambda_cav=1.0, lambda_E=0.0,
    )
    Lx_gd = float((delta_x_gd ** 2).sum())
    Ldv_gd = float((delta_dv_gd ** 2).sum())
    print(f"  stable GD lr_dv={info_gd['lr_dv']:.4e}")
    print(f"  GD Δ at θ_true: ‖Δx‖²={Lx_gd:.4e}  ‖Δdv‖²={Ldv_gd:.4e}  L_gap={Lx_gd+Ldv_gd:.4e}")
    print(f"  L_inner history (every 5 iter): " +
          "  ".join(f"{info_gd['L_history'][k]:.3e}" for k in range(0, 30, 5)))
    L_inner_grew = info_gd['L_history'][-1] > info_gd['L_history'][0] * 2
    print(f"  L_inner diverged? {'✗ YES (BAD)' if L_inner_grew else '✓ NO (good)'}")

    # ============ T3: across multiple sharpness values ============
    print(f"\n[T3] Stable GD across s ∈ {{5, 10, 20, 35, 50}}")
    for s_val in [5.0, 10.0, 20.0, 35.0, 50.0]:
        theta_s = torch.tensor([3.0, s_val, 0.5], dtype=torch.float32, device=device,
                                 requires_grad=False)
        # obs at this s
        pos_s, alpha_s, dv_s = gen_obs_param_arr(
            solver, rest_state, theta_s.cpu().numpy(), knife_pos, knife_radius, device)
        alpha_s_t = torch.tensor(alpha_s, dtype=torch.float32, device=device)
        pos_s_t = torch.tensor(pos_s, dtype=torch.float32, device=device)
        # sim at same theta
        with torch.no_grad():
            alpha_sim_s, pos_sim_s, dv_sim_s = WarpFullForwardOp.apply(
                theta_s, solver, rest_state, knife_pos, knife_radius,
            )
        delta_x_s, delta_dv_s, info_s = gd_inner_loop_stable(
            pos_sim_s, dv_sim_s, pos_s_t, alpha_s_t, theta_s,
            edges_a_t, edges_b_t, edges_rest_t,
            n_iter=30, lr_pos=5e-3, lr_dv_factor=1.0,
            lambda_cav=1.0, lambda_E=0.0,
        )
        L_gap = float((delta_x_s ** 2).sum() + (delta_dv_s ** 2).sum())
        L_inner_first = info_s['L_history'][0]
        L_inner_last = info_s['L_history'][-1]
        stable = L_inner_last < L_inner_first * 2
        print(f"  s={s_val:5.1f}: lr_dv={info_s['lr_dv']:.4e}  "
              f"L_inner {L_inner_first:.3e} → {L_inner_last:.3e}  "
              f"L_gap={L_gap:.4e}  {'✓ stable' if stable else '✗ divergent'}")

    # ============ T4: GD vs closed-form at θ_init (off-truth) ============
    print(f"\n[T4] At θ_init=(5,15,0.6), GD should converge to closed-form result")
    theta_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)
    theta_init_t = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                  requires_grad=False)
    with torch.no_grad():
        alpha_sim_i, pos_sim_i, dv_sim_i = WarpFullForwardOp.apply(
            theta_init_t, solver, rest_state, knife_pos, knife_radius,
        )
    # Closed-form
    dx_cf, ddv_cf = closed_form_inner(
        pos_sim_i, dv_sim_i, pos_target_t, alpha_target_t, theta_init_t,
    )
    Lx_cf_i = float((dx_cf ** 2).sum())
    Ldv_cf_i = float((ddv_cf ** 2).sum())
    L_gap_cf_i = Lx_cf_i + Ldv_cf_i
    print(f"  closed-form: ‖Δx‖²={Lx_cf_i:.4e}  ‖Δdv‖²={Ldv_cf_i:.4e}  L_gap={L_gap_cf_i:.4e}")

    # Stable GD
    dx_gd_i, ddv_gd_i, info_gd_i = gd_inner_loop_stable(
        pos_sim_i, dv_sim_i, pos_target_t, alpha_target_t, theta_init_t,
        edges_a_t, edges_b_t, edges_rest_t,
        n_iter=30, lr_pos=5e-3, lr_dv_factor=1.0,
        lambda_cav=1.0, lambda_E=0.0,
    )
    Lx_gd_i = float((dx_gd_i ** 2).sum())
    Ldv_gd_i = float((ddv_gd_i ** 2).sum())
    L_gap_gd_i = Lx_gd_i + Ldv_gd_i
    print(f"  stable GD:   ‖Δx‖²={Lx_gd_i:.4e}  ‖Δdv‖²={Ldv_gd_i:.4e}  L_gap={L_gap_gd_i:.4e}")
    print(f"  L_inner GD history: " +
          "  ".join(f"{info_gd_i['L_history'][k]:.3e}" for k in [0, 5, 10, 20, 29]))

    # Compare Δ values directly (per-component)
    dx_diff = float((dx_gd_i - dx_cf).norm())
    ddv_diff = float((ddv_gd_i - ddv_cf).norm())
    print(f"  Δ_x diff norm: {dx_diff:.4e}   Δ_dv diff norm: {ddv_diff:.4e}")
    L_gap_ratio = L_gap_gd_i / max(L_gap_cf_i, 1e-12)
    print(f"  L_gap ratio GD/closed-form: {L_gap_ratio:.3f}")

    # ============ T5: side-by-side broken vs stable GD ============
    print(f"\n[T5] Broken GD (lr_dv=0.1) vs stable GD on same problem")
    print(f"  Broken (Day 7's lr_dv=0.1):")
    # Simulate the broken version by using fixed lr_dv=0.1
    delta_x_br = torch.zeros((solver.N, 3), dtype=torch.float32, device=device, requires_grad=True)
    delta_dv_br = torch.zeros(solver.N, dtype=torch.float32, device=device, requires_grad=True)
    s_init = theta_init_t[1]; tau_init = theta_init_t[2]
    L_history_broken = []
    for k in range(30):
        pos_corr = pos_sim_i + delta_x_br
        dv_corr = torch.relu(dv_sim_i + delta_dv_br)
        alpha_corr = torch.sigmoid(-s_init * (dv_corr - tau_init))
        L_pos_b = ((pos_corr - pos_target_t) ** 2).sum()
        L_cav_b = ((alpha_corr - alpha_target_t) ** 2).sum()
        L_inner_b = L_pos_b + L_cav_b
        L_history_broken.append(float(L_inner_b))
        grad_x_b, grad_dv_b = torch.autograd.grad(
            L_inner_b, (delta_x_br, delta_dv_br),
            create_graph=True, retain_graph=True,
        )
        delta_x_br = delta_x_br - 5e-3 * grad_x_b
        delta_dv_br = delta_dv_br - 0.1 * grad_dv_b
    print(f"    L_inner: " +
          "  ".join(f"{L_history_broken[k]:.3e}" for k in [0, 5, 10, 20, 29]))
    L_gap_broken = float((delta_x_br ** 2).sum() + (delta_dv_br ** 2).sum())
    print(f"    L_gap_broken = {L_gap_broken:.4e}")
    print(f"  Stable (lr_dv=1/s²={1.0/15**2:.4e}):")
    print(f"    L_inner: " +
          "  ".join(f"{info_gd_i['L_history'][k]:.3e}" for k in [0, 5, 10, 20, 29]))
    print(f"    L_gap_stable = {L_gap_gd_i:.4e}")
    print(f"  L_gap_broken / L_gap_stable = {L_gap_broken / max(L_gap_gd_i, 1e-12):.1f}×")

    # ============ Verdict ============
    print(f"\n" + "=" * 70)
    print(f"Day 8 inner loop verdict")
    print(f"=" * 70)
    print(f"\n  T1 closed-form at θ_true: {'✓' if cf_clean else '✗'} L_gap < 1e-6")
    print(f"  T2 stable GD at θ_true: {'✓' if not L_inner_grew else '✗'} L_inner doesn't diverge")
    print(f"  T2 stable GD L_gap value: {Lx_gd + Ldv_gd:.4e} "
          f"({'✓ ≈ 0' if Lx_gd + Ldv_gd < 1e-3 else '⚠ not zero'})")
    print(f"  T4 GD ≈ closed-form at θ_init: ratio={L_gap_ratio:.2f} "
          f"({'✓ close' if 0.5 < L_gap_ratio < 2.0 else '⚠'})")
    print(f"  T5 stable GD vs broken GD: {L_gap_broken / max(L_gap_gd_i, 1e-12):.1f}× reduction")


if __name__ == "__main__":
    main()
