#!/usr/bin/env python3
"""Phase 2B Day 4 — FD parity test with cutting ACTIVE.

Phase 4a: 1-step no-cutting via production solver — replicate POC v3 ratio.
Phase 4b: 1-step WITH cutting fired — verify gradient flows through cutting.

This addresses the gap: all earlier gradient parity tests had enable_cutting=False.
Phase 4b proves Layer A backward works in true production sys-ID conditions.

Pass:
  4a — ratio at num_iter=6 in (0.85, 1.05)
  4b — ratio at num_iter=6 in (0.5, 2.0) — wider band since cutting adds complexity
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

wp.config.verify_autograd_array_access = False

from path4_falsification_test_H_isolate import setup_env, loss_sum_pos_z_kernel
from path4_poc_v1_dual_purpose import snap_solver_state, reset_solver_state


def fd_test_via_solver(num_iter, knife_pos_np, cutting_enabled, eps=5e-3):
    """One-step forward + autograd backward via PRODUCTION solver
    (use_split_cluster_solve=True). Compare to FD perturbation."""

    env = setup_env()
    env.solver.num_iterations = num_iter
    env.solver.use_split_cluster_solve = True
    env.solver.use_ssa_soft_removal = True
    if cutting_enabled:
        env.solver.cutting_update_interval = 1   # force cutting every step
    else:
        env.solver.cutting_update_interval = 100000  # effectively disabled
    env.solver.knife_active = 1
    env.solver.step_count = 0

    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)

    state_snap = snap_solver_state(env)

    # ---------- Autograd ----------
    knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                          device=env.device, requires_grad=True)
    env.solver.knife_pos = knife_pos
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)

    with wp.Tape() as tape:
        env.solver.step()
        wp.launch(loss_sum_pos_z_kernel, dim=env.solver.N,
                  inputs=[env.solver.p["pos"], interior_mask_wp,
                          interior_count, loss_buf], device=env.device)
    tape.backward(loss=loss_buf)
    wp.synchronize()
    ag_kp = knife_pos.grad.numpy()[0].copy()

    # ---------- FD via 6 perturbations ----------
    fd_kp = np.zeros(3)
    for axis in range(3):
        for sign in (+1, -1):
            reset_solver_state(env, state_snap)
            env.solver.step_count = 0
            kp_np = list(knife_pos_np); kp_np[axis] += sign * eps
            env.solver.knife_pos = wp.array([wp.vec3(*kp_np)],
                                              dtype=wp.vec3, device=env.device)
            l_buf = wp.zeros(1, dtype=wp.float32, device=env.device)
            env.solver.step()
            wp.launch(loss_sum_pos_z_kernel, dim=env.solver.N,
                      inputs=[env.solver.p["pos"], interior_mask_wp,
                              interior_count, l_buf], device=env.device)
            wp.synchronize()
            if sign == +1: lp = float(l_buf.numpy()[0])
            else: lm = float(l_buf.numpy()[0])
        fd_kp[axis] = (lp - lm) / (2.0 * eps)

    ag_mag = float(np.linalg.norm(ag_kp))
    fd_mag = float(np.linalg.norm(fd_kp))
    ratio = ag_mag / max(fd_mag, 1e-12)
    return dict(num_iter=num_iter, ag_kp=ag_kp, fd_kp=fd_kp,
                ag_mag=ag_mag, fd_mag=fd_mag, ratio=ratio)


def main():
    wp.init()
    print("=" * 78)
    print("  Phase 2B Day 4 — Production solver FD parity (with/without cutting)")
    print("=" * 78)

    knife_pos_np = (0.85, 0.5, 0.12)

    print("\n  ---- Phase 4a: NO CUTTING (replicate POC v3 via solver) ----")
    results_a = {}
    for n in (1, 3, 6):
        r = fd_test_via_solver(n, knife_pos_np, cutting_enabled=False)
        results_a[n] = r
        print(f"    num_iter={n}: autograd={r['ag_mag']:.3e}  FD={r['fd_mag']:.3e}  ratio={r['ratio']:.3e}")

    print("\n  ---- Phase 4b: CUTTING ACTIVE (production-like) ----")
    results_b = {}
    for n in (1, 3, 6):
        r = fd_test_via_solver(n, knife_pos_np, cutting_enabled=True)
        results_b[n] = r
        print(f"    num_iter={n}: autograd={r['ag_mag']:.3e}  FD={r['fd_mag']:.3e}  ratio={r['ratio']:.3e}")

    # Verdicts
    print()
    print("=" * 78)
    print("  RESULTS")
    print("=" * 78)
    print(f"  {'num_iter':>8}  {'no-cut ratio':>14}  {'with-cut ratio':>16}")
    for n in (1, 3, 6):
        ra = results_a[n]['ratio']
        rb = results_b[n]['ratio']
        print(f"  {n:>8}  {ra:>14.3e}  {rb:>16.3e}")

    r6_a = results_a[6]['ratio']
    r6_b = results_b[6]['ratio']
    print()
    print(f"  Phase 4a verdict (no-cut, target 0.85-1.05): ", end="")
    if 0.85 <= r6_a <= 1.05:
        print(f"🎯 PASS ({r6_a:.3e})")
    else:
        print(f"❌ FAIL ({r6_a:.3e})")

    print(f"  Phase 4b verdict (cutting, target 0.5-2.0):  ", end="")
    if 0.5 <= r6_b <= 2.0:
        print(f"🎯 PASS ({r6_b:.3e}) — Layer A IS differentiable in cutting context!")
    elif r6_b < 1e3:
        print(f"⚠️ partial ({r6_b:.3e})")
    else:
        print(f"❌ FAIL ({r6_b:.3e}) — cutting + cluster_solve combined breaks")

    return 0


if __name__ == "__main__":
    sys.exit(main())
