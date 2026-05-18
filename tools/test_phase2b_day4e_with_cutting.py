#!/usr/bin/env python3
"""Day 4-E: FD parity with cutting truly active (cutting_update_interval=1, step_count=0).

Now that collision_delta is cycle-scoped (Day 4-D v3 fix), test gradient flow
when cutting kernels actually fire. This is the production sys-ID condition.

Pass: ratio in (0.85, 1.15) at num_iter=6 — match POC v3 precision even with
cutting kernels active in the tape.
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


def fd_test(num_iter, cutting_active, knife_pos_np, eps=5e-3):
    env = setup_env()
    env.solver.num_iterations = num_iter
    env.solver.substeps = 1
    env.solver.use_split_cluster_solve = True
    env.solver.use_ssa_soft_removal = True
    if cutting_active:
        env.solver.cutting_update_interval = 1   # cutting fires every step
    else:
        env.solver.cutting_update_interval = 100000
    env.solver.knife_active = 1
    env.solver.step_count = 0    # 0 % anything == 0 → with cutting=1 fires; with 100000 also fires (we test both)

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
    print("  Day 4-E: split mode FD parity WITH cutting active")
    print("=" * 78)
    print("  Now that collision_delta is cycle-scoped, verify cutting kernels in")
    print("  tape don't break gradient. Production sys-ID condition.")
    print()

    knife_pos_np = (0.85, 0.5, 0.12)

    print(f"  {'num_iter':>10} {'cutting':>10} {'autograd':>12} {'FD':>12} {'ratio':>10}")
    for num_iter in (1, 3, 6):
        for cutting_active in (False, True):
            ag, fd, r = fd_test(num_iter, cutting_active, knife_pos_np)
            print(f"  {num_iter:>10} {str(cutting_active):>10} {ag:>12.3e} {fd:>12.3e} {r:>10.3e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
