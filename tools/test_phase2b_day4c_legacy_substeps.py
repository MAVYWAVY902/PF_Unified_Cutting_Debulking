#!/usr/bin/env python3
"""Day 4-C: Legacy mode (no split) with substeps=1 and substeps=50.

Goal: confirm "bias is from substep accumulation, not Phase 2B split mode".

Compare ratios:
  legacy substeps=1   → if 0.94, legacy 1-substep clean (matches POC v1's R=I expectation)
                         if huge, legacy is broken (we already know it's 4.5e10 with v2q)
  legacy substeps=50  → expected similar pattern as split substeps=50

If split substeps=50 (Day 4) ≈ legacy substeps=50 (Day 4-C),
  → confirmed: long-horizon bias is NOT introduced by Phase 2B; it's PBD physics.
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


def fd_test_via_solver_legacy(num_iter, substeps, knife_pos_np, eps=5e-3):
    env = setup_env()
    env.solver.num_iterations = num_iter
    env.solver.substeps = substeps
    env.solver.use_split_cluster_solve = False    # ★ LEGACY mode
    env.solver.use_ssa_soft_removal = True
    env.solver.cutting_update_interval = 100000
    env.solver.knife_active = 1
    env.solver.step_count = 0

    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)
    state_snap = snap_solver_state(env)

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
    return ag_mag, fd_mag, ag_mag / max(fd_mag, 1e-12)


def main():
    wp.init()
    print("=" * 78)
    print("  Phase 2B Day 4-C — LEGACY mode substeps sweep (no split)")
    print("=" * 78)

    knife_pos_np = (0.85, 0.5, 0.12)

    print(f"\n  {'num_iter':>10} {'substeps':>10} {'autograd':>12} {'FD':>12} {'ratio':>10}")

    for num_iter in (1, 3, 6):
        for substeps in (1, 50):
            ag, fd, ratio = fd_test_via_solver_legacy(num_iter, substeps, knife_pos_np)
            print(f"  {num_iter:>10} {substeps:>10} {ag:>12.3e} {fd:>12.3e} {ratio:>10.3e}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
