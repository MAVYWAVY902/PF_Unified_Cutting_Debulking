#!/usr/bin/env python3
"""
Diagnostic for Stage 2 calibration: figure out what's actually going on
in the loss landscape before claiming "non-identifiability".

Three independent tests:

  Test A (Newton): replace Adam with 2D Newton-Raphson using FD-computed
    Hessian. If Newton converges to TRUE in <10 steps → it's an optimizer
    issue, not a forward-map issue.

  Test B (N scan): try N ∈ {20, 100, 500}. If valley narrows with N → it's a
    data-quantity issue, not a parameter-coupling issue.

  Test C (multi-knife): try 1, 3, 5 knife positions. If multi-knife converges
    to TRUE → it's observation-design, not intrinsic.

Also produces:
  - Actual 2x2 Hessian at TRUE (eigenvalues + eigenvectors)
  - Loss heatmap on a 30x30 grid around TRUE
"""
import sys, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import warp as wp
wp.init()

from tools.calibrate_stage2_FD import (
    setup_synthetic_system, run_cutting_forward, loss_synth,
)

TRUE_AM, TRUE_SP = 3.0, 20.0
INIT_AM, INIT_SP = 5.0, 10.0
DEFAULT_FWD = dict(knife_radius=0.1, threshold=0.5, n_cycles=3)


def make_loss_fn(N=20, knife_x=0.5, fwd=None, seed=42):
    """Generate target with TRUE params, return closure that computes loss(am, sp)."""
    fwd = fwd or DEFAULT_FWD
    sys_dict = setup_synthetic_system(N=N, knife_x=knife_x, seed=seed)
    target = run_cutting_forward(sys_dict, TRUE_AM, TRUE_SP, **fwd)
    def L(am, sp):
        a = run_cutting_forward(sys_dict, float(am), float(sp), **fwd)
        return loss_synth(a, target)
    return L, sys_dict


def fd_grad_hess(L, am, sp, eps_a=0.05, eps_s=0.25):
    """Central diff gradient + Hessian (2x2). Costs 9 evaluations."""
    L0 = L(am, sp)
    # gradient
    g_a = (L(am + eps_a, sp) - L(am - eps_a, sp)) / (2 * eps_a)
    g_s = (L(am, sp + eps_s) - L(am, sp - eps_s)) / (2 * eps_s)
    # Hessian diagonals
    h_aa = (L(am + eps_a, sp) - 2 * L0 + L(am - eps_a, sp)) / (eps_a ** 2)
    h_ss = (L(am, sp + eps_s) - 2 * L0 + L(am, sp - eps_s)) / (eps_s ** 2)
    # cross term (4 evals)
    h_as = (L(am + eps_a, sp + eps_s) - L(am + eps_a, sp - eps_s)
            - L(am - eps_a, sp + eps_s) + L(am - eps_a, sp - eps_s)) / (4 * eps_a * eps_s)
    grad = np.array([g_a, g_s])
    hess = np.array([[h_aa, h_as], [h_as, h_ss]])
    return L0, grad, hess


def newton_step(L, am, sp, eps_a=0.05, eps_s=0.25, max_step_a=0.3, max_step_s=2.0,
                damping=1e-6):
    """One Newton step with trust-region clipping."""
    L0, grad, hess = fd_grad_hess(L, am, sp, eps_a, eps_s)
    # Damped Newton: H + λI to avoid singularity / negative-definite issues
    H_damped = hess + damping * np.eye(2)
    try:
        delta = -np.linalg.solve(H_damped, grad)
    except np.linalg.LinAlgError:
        delta = -grad  # fallback to gradient descent
    # Clip step size (trust region)
    delta[0] = np.clip(delta[0], -max_step_a, max_step_a)
    delta[1] = np.clip(delta[1], -max_step_s, max_step_s)
    return am + delta[0], sp + delta[1], L0, grad, hess


# =========================================================================
# Test A: Newton from same init as Adam
# =========================================================================

def test_A_newton(N=20):
    print("\n" + "=" * 72)
    print(f"  TEST A: Newton-Raphson (N={N})")
    print("=" * 72)
    L, _ = make_loss_fn(N=N)
    am, sp = INIT_AM, INIT_SP
    print(f"  init: ({am:.3f}, {sp:.3f}), L={L(am, sp):.3e}")
    print(f"  TRUE: ({TRUE_AM}, {TRUE_SP})")

    for step in range(15):
        am, sp, L0, grad, hess = newton_step(L, am, sp)
        # bounds
        am = max(0.5, min(50.0, am)); sp = max(1.0, min(200.0, sp))
        eigs = np.linalg.eigvalsh(hess)
        print(f"  step {step:2d}: ({am:.4f}, {sp:.4f})  L={L0:.3e}  "
              f"|grad|={np.linalg.norm(grad):.2e}  H_eigs=[{eigs[0]:+.2e}, {eigs[1]:+.2e}]")

    err_a = abs(am - TRUE_AM) / TRUE_AM
    err_s = abs(sp - TRUE_SP) / TRUE_SP
    print(f"\n  RESULT: rel_err α_min = {err_a*100:.1f}%, sharp = {err_s*100:.1f}%")
    return am, sp, err_a, err_s


# =========================================================================
# Test B: N-scan with Newton
# =========================================================================

def test_B_n_scan():
    print("\n" + "=" * 72)
    print("  TEST B: N scan (does valley narrow with more particles?)")
    print("=" * 72)
    results = {}
    for N in [20, 100, 500]:
        print(f"\n  --- N = {N} ---")
        am, sp, err_a, err_s = test_A_newton(N=N)
        results[N] = (am, sp, err_a, err_s)
    print("\n  Summary:")
    print(f"  {'N':>5}  {'α_min':>8}  {'sharp':>8}  {'err_α':>7}  {'err_s':>7}")
    for N, (am, sp, ea, es) in results.items():
        print(f"  {N:>5}  {am:>8.4f}  {sp:>8.4f}  {ea*100:>6.1f}%  {es*100:>6.1f}%")
    return results


# =========================================================================
# Test C: multi-knife
# =========================================================================

def make_multi_knife_loss_fn(N=20, knife_xs=(0.4, 0.5, 0.6)):
    """Loss = sum over knife positions of MSE(alpha_k vs target_k)."""
    fwd = DEFAULT_FWD
    sys_dicts = [setup_synthetic_system(N=N, knife_x=kx) for kx in knife_xs]
    targets = [run_cutting_forward(s, TRUE_AM, TRUE_SP, **fwd) for s in sys_dicts]
    def L(am, sp):
        total = 0.0
        for s, t in zip(sys_dicts, targets):
            a = run_cutting_forward(s, float(am), float(sp), **fwd)
            total += loss_synth(a, t)
        return total / len(sys_dicts)
    return L


def test_C_multi_knife():
    print("\n" + "=" * 72)
    print("  TEST C: Multi-knife (1 vs 3 vs 5 cuts)")
    print("=" * 72)
    configs = [
        ("1 cut",  (0.5,)),
        ("3 cuts", (0.4, 0.5, 0.6)),
        ("5 cuts", (0.35, 0.425, 0.5, 0.575, 0.65)),
    ]
    results = {}
    for label, kxs in configs:
        print(f"\n  --- {label}: knives at {kxs} ---")
        L = make_multi_knife_loss_fn(N=20, knife_xs=kxs)
        am, sp = INIT_AM, INIT_SP
        for step in range(15):
            am, sp, L0, grad, hess = newton_step(L, am, sp)
            am = max(0.5, min(50.0, am)); sp = max(1.0, min(200.0, sp))
        err_a = abs(am - TRUE_AM) / TRUE_AM
        err_s = abs(sp - TRUE_SP) / TRUE_SP
        print(f"  final: ({am:.4f}, {sp:.4f})  err_α={err_a*100:.1f}%, err_s={err_s*100:.1f}%")
        results[label] = (am, sp, err_a, err_s)
    print("\n  Summary:")
    print(f"  {'config':>10}  {'α_min':>8}  {'sharp':>8}  {'err_α':>7}  {'err_s':>7}")
    for label, (am, sp, ea, es) in results.items():
        print(f"  {label:>10}  {am:>8.4f}  {sp:>8.4f}  {ea*100:>6.1f}%  {es*100:>6.1f}%")
    return results


# =========================================================================
# Bonus: actual Hessian at TRUE + condition number
# =========================================================================

def hessian_at_true(N=20):
    print("\n" + "=" * 72)
    print(f"  HESSIAN AT TRUE (N={N})")
    print("=" * 72)
    L, _ = make_loss_fn(N=N)
    L0, grad, hess = fd_grad_hess(L, TRUE_AM, TRUE_SP)
    eigs, vecs = np.linalg.eigh(hess)
    print(f"  L(TRUE) = {L0:.3e}")
    print(f"  grad    = {grad}")
    print(f"  Hessian =\n{hess}")
    print(f"  eigenvalues: {eigs}")
    print(f"  condition number κ(H) = {eigs.max() / max(abs(eigs.min()), 1e-12):.2e}")
    print(f"  eigenvector (small eig): {vecs[:, 0]}  ← valley axis")
    print(f"  eigenvector (large eig): {vecs[:, 1]}  ← perpendicular")
    return hess, eigs, vecs


# =========================================================================
# Newton SANITY CHECK: start near TRUE, must converge back
# =========================================================================
# Prior Newton-from-far-init test was inconclusive (Hessian indefinite far
# from optimum → Newton diverged). This separates "tool broken" from
# "tool can't escape that init region".

def newton_sanity_check():
    print("\n" + "=" * 72)
    print("  NEWTON SANITY CHECK: start near TRUE, expect convergence back to TRUE")
    print("=" * 72)
    L, _ = make_loss_fn(N=20)

    perturbations = [
        ("+α small", TRUE_AM + 0.10, TRUE_SP),
        ("-α small", TRUE_AM - 0.10, TRUE_SP),
        ("+s small", TRUE_AM, TRUE_SP + 1.0),
        ("-s small", TRUE_AM, TRUE_SP - 1.0),
        ("both small", TRUE_AM + 0.10, TRUE_SP + 1.0),
        ("medium offset", TRUE_AM + 0.30, TRUE_SP + 3.0),
    ]
    results = []
    for label, am0, sp0 in perturbations:
        am, sp = am0, sp0
        for step in range(8):
            am, sp, L0, grad, hess = newton_step(L, am, sp,
                                                 max_step_a=0.15, max_step_s=1.0)
            am = max(0.5, min(50.0, am)); sp = max(1.0, min(200.0, sp))
        err_a = abs(am - TRUE_AM); err_s = abs(sp - TRUE_SP)
        L_final = L(am, sp)
        results.append((label, am0, sp0, am, sp, err_a, err_s, L_final))
        print(f"  {label:>15}  init=({am0:.3f}, {sp0:.3f})  "
              f"final=({am:.4f}, {sp:.4f})  "
              f"err=({err_a:.3f}, {err_s:.3f})  L={L_final:.2e}")

    all_ok = all(r[5] < 0.05 and r[6] < 0.5 for r in results)
    print(f"\n  Newton sanity: {'PASS' if all_ok else 'FAIL'} "
          f"(all converge to within α±0.05, s±0.5 of TRUE)")
    return all_ok


# =========================================================================
# Adam-based optimizer (matches calibrate_stage2_FD.py settings)
# =========================================================================

def run_adam_on_loss(L, init_am=INIT_AM, init_sp=INIT_SP,
                     n_steps=200, lr_a=0.05, lr_s=0.3,
                     eps_a=0.05, eps_s=0.25, verbose=False):
    """Adam with per-param lr + early stopping. Returns best (am, sp, loss)."""
    am, sp = float(init_am), float(init_sp)
    m = np.zeros(2); v = np.zeros(2); t = 0
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    lr = np.array([lr_a, lr_s])

    best_loss = float("inf"); best = (am, sp, 0)
    for step in range(n_steps):
        # central diff gradient
        L0 = L(am, sp)
        g_a = (L(am + eps_a, sp) - L(am - eps_a, sp)) / (2 * eps_a)
        g_s = (L(am, sp + eps_s) - L(am, sp - eps_s)) / (2 * eps_s)
        grad = np.array([g_a, g_s])

        if L0 < best_loss:
            best_loss = L0; best = (am, sp, step)

        # Adam step
        t += 1
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        m_hat = m / (1 - b1 ** t); v_hat = v / (1 - b2 ** t)
        params = np.array([am, sp]) - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        am, sp = params[0], params[1]
        am = max(0.5, min(50.0, am)); sp = max(1.0, min(200.0, sp))
        if verbose and step % 50 == 0:
            print(f"    [adam {step:3d}]  ({am:.3f}, {sp:.3f})  L={L0:.3e}")
    return best  # (am, sp, step_at_best)


# =========================================================================
# Test B (redone): N scan with Adam
# =========================================================================

def test_B_n_scan_adam():
    print("\n" + "=" * 72)
    print("  TEST B (redone): N scan with Adam optimizer")
    print("=" * 72)
    print("  Question: does valley narrow with more particles? (Newton failed -- using Adam)")
    print()

    results = {}
    for N in [20, 100, 500]:
        L, _ = make_loss_fn(N=N)
        best_am, best_sp, best_step = run_adam_on_loss(L, n_steps=200)
        err_a = abs(best_am - TRUE_AM) / TRUE_AM
        err_s = abs(best_sp - TRUE_SP) / TRUE_SP
        results[N] = (best_am, best_sp, err_a, err_s, best_step)
        print(f"  N={N:>4}: best_step={best_step:3d}  "
              f"params=({best_am:.4f}, {best_sp:.4f})  "
              f"err=({err_a*100:.1f}%, {err_s*100:.1f}%)")

    print("\n  Summary:")
    print(f"  {'N':>5}  {'α_min':>8}  {'sharp':>8}  {'err_α':>7}  {'err_s':>7}")
    for N, (am, sp, ea, es, _) in results.items():
        print(f"  {N:>5}  {am:>8.4f}  {sp:>8.4f}  {ea*100:>6.1f}%  {es*100:>6.1f}%")
    return results


# =========================================================================
# Test C (redone): multi-knife with Adam
# =========================================================================

def test_C_multi_knife_adam():
    print("\n" + "=" * 72)
    print("  TEST C (redone): multi-knife scan with Adam")
    print("=" * 72)
    print("  Question: do more independent knife positions break the valley?")
    print()

    configs = [
        ("1 cut",  (0.5,)),
        ("3 cuts", (0.4, 0.5, 0.6)),
        ("5 cuts", (0.35, 0.425, 0.5, 0.575, 0.65)),
    ]
    results = {}
    for label, kxs in configs:
        L = make_multi_knife_loss_fn(N=20, knife_xs=kxs)
        best_am, best_sp, best_step = run_adam_on_loss(L, n_steps=200)
        err_a = abs(best_am - TRUE_AM) / TRUE_AM
        err_s = abs(best_sp - TRUE_SP) / TRUE_SP
        results[label] = (best_am, best_sp, err_a, err_s, best_step)
        print(f"  {label}: best_step={best_step:3d}  "
              f"params=({best_am:.4f}, {best_sp:.4f})  "
              f"err=({err_a*100:.1f}%, {err_s*100:.1f}%)")

    print("\n  Summary:")
    print(f"  {'config':>10}  {'α_min':>8}  {'sharp':>8}  {'err_α':>7}  {'err_s':>7}")
    for label, (am, sp, ea, es, _) in results.items():
        print(f"  {label:>10}  {am:>8.4f}  {sp:>8.4f}  {ea*100:>6.1f}%  {es*100:>6.1f}%")
    return results


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    t0 = time.time()
    hessian_at_true(N=20)
    sanity_ok = newton_sanity_check()
    print(f"\n  → Newton sanity {'PASSED, but' if sanity_ok else 'FAILED — '} "
          f"the original far-init Test A is still inconclusive.")
    test_B_n_scan_adam()
    test_C_multi_knife_adam()
    print(f"\n[total time] {time.time() - t0:.1f}s")
