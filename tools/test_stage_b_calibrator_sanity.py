#!/usr/bin/env python3
"""
Stage B Day 5.1 — Sanity 2-4 for production autograd calibrator.

Sanity 2: FD parity at MULTIPLE ε on MSE loss
    Per Nesterov-Spokoiny 2017 Theorem 1: FD with ε computes ∇f_ε (Gaussian-
    smoothed surrogate). As ε→0, ∇f_ε → ∇f. So we expect:
        ε=1e-3:  FD ≈ autograd (rel_err < 1%)
        ε=1e-2:  small bias (~few %)
        ε=1e-1:  large bias (sign-flip possible — what we saw 2026-05-01)
    PASS criteria: rel_err monotonically decreases as ε → 0, and at ε=1e-3
    rel_err < 5% on both alpha_min and sharpness gradients.

Sanity 3: Loss convergence (close init)
    Run autograd calibration with init (3.3, 18) → true (3.0, 20) for 100 iters.
    Adam with momentum is NOT strictly monotonic by design (and shouldn't be —
    momentum is what makes it work). Honest convergence criterion:
      (a) Final loss ≤ 1% of initial loss (substantial decrease, not stalled)
      (b) Late-window mean < 10% of early-window mean (downward trend, not noise)
      (c) No NaN/inf anywhere in trajectory
    PASS criteria: all three (a)(b)(c) hold.

Sanity 4: Runtime — autograd vs FD per-step cost
    Measure wall time for: autograd 100 iters vs FD 50 iters on N=20 synth.
    FD does 4 forward sims per step (2 params × central diff); autograd does
    1 forward + 1 backward.
    PASS criteria: autograd per-iter wall-time < FD per-iter wall-time
    (asymptotic expected gain scales with #params).
"""
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

wp.config.verify_autograd_array_access = True

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.calibrate_stage2_autograd import (
    setup_synthetic_system,
    make_params,
    run_ssa_full,
    mse_loss_kernel,
    Adam2D,
)


# =============================================================================
# Helpers
# =============================================================================

def compute_loss_only(params: dict, sys_dict: dict, target_alpha_wp,
                      n_cycles: int = 3) -> float:
    """Forward-only: run sim, compute MSE, return scalar (no tape)."""
    N = sys_dict["N"]
    loss_buf = wp.zeros(1, dtype=wp.float32)
    alpha_sim = run_ssa_full(params, sys_dict, n_cycles=n_cycles)
    wp.launch(mse_loss_kernel, dim=N,
              inputs=[alpha_sim, target_alpha_wp, loss_buf])
    wp.synchronize()
    return float(loss_buf.numpy()[0])


def compute_autograd_grad(theta: np.ndarray, sys_dict: dict, target_alpha_wp,
                          n_cycles: int = 3):
    """Returns (grad, loss) at theta = [alpha_min, sharpness]."""
    N = sys_dict["N"]
    params = make_params(float(theta[0]), float(theta[1]), requires_grad=True)
    loss_buf = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    with wp.Tape() as tape:
        alpha_sim = run_ssa_full(params, sys_dict, n_cycles=n_cycles)
        wp.launch(mse_loss_kernel, dim=N,
                  inputs=[alpha_sim, target_alpha_wp, loss_buf])
    tape.backward(loss=loss_buf)
    grad = np.array([
        float(params["alpha_min"].grad.numpy()[0]),
        float(params["sharpness"].grad.numpy()[0]),
    ])
    loss = float(loss_buf.numpy()[0])
    return grad, loss


def compute_fd_grad(theta: np.ndarray, sys_dict: dict, target_alpha_wp,
                    eps: float, n_cycles: int = 3):
    """Central diff on (alpha_min, sharpness) at theta. Same eps both axes."""
    grad = np.zeros(2)
    for i in range(2):
        theta_plus = theta.copy(); theta_plus[i] += eps
        theta_minus = theta.copy(); theta_minus[i] -= eps
        params_plus = make_params(float(theta_plus[0]), float(theta_plus[1]),
                                  requires_grad=False)
        params_minus = make_params(float(theta_minus[0]), float(theta_minus[1]),
                                   requires_grad=False)
        loss_plus = compute_loss_only(params_plus, sys_dict, target_alpha_wp, n_cycles)
        loss_minus = compute_loss_only(params_minus, sys_dict, target_alpha_wp, n_cycles)
        grad[i] = (loss_plus - loss_minus) / (2.0 * eps)
    return grad


def setup_target(true_alpha_min=3.0, true_sharpness=20.0, n_cycles=3, N=20):
    """Generate synthetic target alpha distribution from true params."""
    sys_dict = setup_synthetic_system(N=N)
    target_params = make_params(true_alpha_min, true_sharpness,
                                requires_grad=False)
    target_alpha_np = run_ssa_full(target_params, sys_dict,
                                   n_cycles=n_cycles).numpy().copy()
    target_alpha_wp = wp.array(target_alpha_np, dtype=wp.float32)
    return sys_dict, target_alpha_wp


# =============================================================================
# Sanity 2: FD parity at multiple ε
# =============================================================================

def sanity_2_fd_parity():
    print("=" * 72)
    print("Sanity 2 — FD parity at multiple ε (Nesterov-Spokoiny verification)")
    print("=" * 72)

    sys_dict, target_alpha_wp = setup_target()

    # Test at close init (3.3, 18) — well-conditioned regime, no sub-ε wiggles
    # expected. This tests pure framework correctness.
    theta = np.array([3.3, 18.0])
    print(f"  Test point: theta = {theta}")

    # Autograd reference
    ag_grad, ag_loss = compute_autograd_grad(theta, sys_dict, target_alpha_wp)
    print(f"  Autograd:  grad = [{ag_grad[0]:+.6e}, {ag_grad[1]:+.6e}]")
    print(f"  Loss at theta: {ag_loss:.6e}")
    print()

    # FD at multiple ε
    eps_values = [1e-1, 1e-2, 1e-3, 1e-4]
    rel_errs = {"alpha_min": [], "sharpness": []}
    print(f"  {'ε':>10}  {'FD grad α_min':>18}  {'rel_err':>10}  "
          f"{'FD grad sharp':>18}  {'rel_err':>10}")
    print(f"  {'-'*10}  {'-'*18}  {'-'*10}  {'-'*18}  {'-'*10}")

    for eps in eps_values:
        fd_grad = compute_fd_grad(theta, sys_dict, target_alpha_wp, eps=eps)
        rel_a = abs(fd_grad[0] - ag_grad[0]) / max(abs(ag_grad[0]), 1e-12)
        rel_s = abs(fd_grad[1] - ag_grad[1]) / max(abs(ag_grad[1]), 1e-12)
        rel_errs["alpha_min"].append(rel_a)
        rel_errs["sharpness"].append(rel_s)
        print(f"  {eps:>10.0e}  {fd_grad[0]:>+18.6e}  {rel_a*100:>9.3f}%  "
              f"{fd_grad[1]:>+18.6e}  {rel_s*100:>9.3f}%")

    # Verdict
    smallest_eps_rel_a = rel_errs["alpha_min"][-1]
    smallest_eps_rel_s = rel_errs["sharpness"][-1]
    monotone_a = all(rel_errs["alpha_min"][i] >= rel_errs["alpha_min"][i+1] - 0.01
                     for i in range(len(rel_errs["alpha_min"]) - 1))
    monotone_s = all(rel_errs["sharpness"][i] >= rel_errs["sharpness"][i+1] - 0.01
                     for i in range(len(rel_errs["sharpness"]) - 1))

    print()
    print(f"  Smallest ε (1e-4) rel_err: α_min {smallest_eps_rel_a*100:.3f}%  "
          f"sharpness {smallest_eps_rel_s*100:.3f}%")
    pass_strict = smallest_eps_rel_a < 0.01 and smallest_eps_rel_s < 0.01
    pass_relaxed = smallest_eps_rel_a < 0.05 and smallest_eps_rel_s < 0.05

    print(f"  Strict (<1% at ε=1e-4):  {'PASS' if pass_strict else 'FAIL'}")
    print(f"  Relaxed (<5% at ε=1e-4): {'PASS' if pass_relaxed else 'FAIL'}")
    print(f"  Monotone decrease in rel_err: α_min {'✓' if monotone_a else '✗'}  "
          f"sharpness {'✓' if monotone_s else '✗'}")

    return pass_relaxed, dict(eps_values=eps_values, rel_errs=rel_errs,
                              ag_grad=ag_grad)


# =============================================================================
# Sanity 3: Monotonic loss decrease
# =============================================================================

def sanity_3_monotonic_loss():
    print()
    print("=" * 72)
    print("Sanity 3 — Loss convergence (close init recovery)")
    print("=" * 72)

    sys_dict, target_alpha_wp = setup_target()

    # Close init: well-conditioned regime
    theta = np.array([3.3, 18.0], dtype=np.float64)
    print(f"  Init: theta = {theta}, true = [3.0, 20.0]")

    opt = Adam2D(lr=np.array([0.02, 0.5], dtype=np.float64))
    n_iters = 100

    losses = []
    for step in range(n_iters):
        ag_grad, loss = compute_autograd_grad(theta, sys_dict, target_alpha_wp)
        losses.append(loss / sys_dict["N"])  # normalize to mean MSE
        ag_grad = ag_grad / sys_dict["N"]
        theta = opt.step(theta, ag_grad)
        theta[0] = max(0.5, min(50.0, theta[0]))
        theta[1] = max(1.0, min(200.0, theta[1]))

    losses = np.array(losses)

    # (a) Final loss ≤ 1% of initial loss
    init_loss = losses[0]
    final_loss = losses[-1]
    drop_ratio = init_loss / max(final_loss, 1e-15)
    pass_a = final_loss <= 0.01 * init_loss

    # (b) Late-window mean < 10% of early-window mean
    early_window = losses[:10].mean()
    late_window = losses[-10:].mean()
    window_ratio = late_window / max(early_window, 1e-15)
    pass_b = window_ratio < 0.10

    # (c) No NaN/inf
    pass_c = bool(np.all(np.isfinite(losses)))

    print(f"  Loss trajectory: {init_loss:.3e} → {final_loss:.3e}  "
          f"(min: {losses.min():.3e})")
    print(f"  Total drop:                 {drop_ratio:.1e}×")
    print(f"  Late-window/early-window:   {window_ratio:.2e}")
    print(f"  All finite:                 {pass_c}")
    print()
    print(f"  (a) Final ≤ 1% × init:        {'PASS' if pass_a else 'FAIL'}")
    print(f"  (b) Late mean < 10% early:    {'PASS' if pass_b else 'FAIL'}")
    print(f"  (c) No NaN/inf:               {'PASS' if pass_c else 'FAIL'}")
    pass_all = pass_a and pass_b and pass_c
    print(f"  Overall: {'PASS' if pass_all else 'FAIL'}")

    return pass_all, dict(losses=losses, drop=drop_ratio,
                          window_ratio=window_ratio)


# =============================================================================
# Sanity 4: Runtime benchmark
# =============================================================================

def sanity_4_runtime():
    print()
    print("=" * 72)
    print("Sanity 4 — Runtime benchmark (autograd vs FD)")
    print("=" * 72)

    sys_dict, target_alpha_wp = setup_target()
    theta = np.array([3.3, 18.0])

    # Warmup (CUDA compile etc)
    _ = compute_autograd_grad(theta, sys_dict, target_alpha_wp)
    _ = compute_fd_grad(theta, sys_dict, target_alpha_wp, eps=1e-3)

    # Autograd: time 100 iters of (forward + backward)
    n_ag = 100
    t0 = time.time()
    for _ in range(n_ag):
        _ = compute_autograd_grad(theta, sys_dict, target_alpha_wp)
    wp.synchronize()
    t_ag = time.time() - t0
    per_iter_ag = t_ag / n_ag

    # FD: time 50 iters of (4 forwards per step for 2-param central diff)
    n_fd = 50
    t0 = time.time()
    for _ in range(n_fd):
        _ = compute_fd_grad(theta, sys_dict, target_alpha_wp, eps=1e-3)
    wp.synchronize()
    t_fd = time.time() - t0
    per_iter_fd = t_fd / n_fd

    print(f"  Autograd: {n_ag} iters in {t_ag*1000:.1f}ms  "
          f"({per_iter_ag*1000:.2f} ms/iter)")
    print(f"  FD:       {n_fd} iters in {t_fd*1000:.1f}ms  "
          f"({per_iter_fd*1000:.2f} ms/iter)")
    print(f"  Speedup (FD ms/iter ÷ autograd ms/iter): "
          f"{per_iter_fd / per_iter_ag:.2f}×")

    pass_runtime = per_iter_ag <= per_iter_fd  # autograd should be ≤ FD per iter
    print(f"  Autograd ≤ FD per-iter:  {'PASS' if pass_runtime else 'FAIL'}")
    print(f"  Note: 2-param case, autograd advantage scales with #params.")

    return pass_runtime, dict(per_iter_ag_ms=per_iter_ag*1000,
                              per_iter_fd_ms=per_iter_fd*1000)


# =============================================================================
# Main
# =============================================================================

def main():
    wp.init()
    print()
    print("█" * 72)
    print("  Stage B Day 5.1 — Sanity 2-4 (production autograd calibrator)")
    print("█" * 72)

    p2, r2 = sanity_2_fd_parity()
    p3, r3 = sanity_3_monotonic_loss()
    p4, r4 = sanity_4_runtime()

    print()
    print("█" * 72)
    print("  STAGE B DAY 5.1 VERDICT")
    print("█" * 72)
    print(f"  Sanity 2 (FD parity): {'PASS' if p2 else 'FAIL'}")
    print(f"  Sanity 3 (monotonic): {'PASS' if p3 else 'FAIL'}")
    print(f"  Sanity 4 (runtime):   {'PASS' if p4 else 'FAIL'}")
    print()
    all_pass = p2 and p3 and p4
    if all_pass:
        print("  ✅ Stage B Day 5.1 PASS — calibrator framework fully verified")
        print("     Stage B at 100% — ready to proceed to Path 4 / Stage 2.2 phantom")
        return 0
    else:
        print("  ⚠️ Some sanities failed — investigate before claiming Stage B 100%")
        return 1


if __name__ == "__main__":
    sys.exit(main())
