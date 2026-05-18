#!/usr/bin/env python3
"""Phase 2C hypothesis test — replace discrete wp.where mask with sigmoid.

Hypothesis: with-cutting ratio 0.094 because update_velocity's alive mask is
discrete: ∂wp.where(a<0.01, 0, 1)/∂a = 0. Replace with sigmoid:
    alive = wp.sigmoid(40 * (alpha - 0.01))
which is smooth and has non-zero gradient through alpha.

If hypothesis correct → ratio jumps to ~0.94 with cutting active.
If still 0.094 → real bug is elsewhere, mask discreteness is NOT the issue.
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

wp.config.verify_autograd_array_access = False

# Monkey-patch update_velocity_kernel_ssa with sigmoid mask version BEFORE
# importing solver (so solver imports this version).

import pf_shape_matching_warp.sim.kernels as km


@wp.kernel
def update_velocity_kernel_ssa_sigmoid(
    pos_in: wp.array(dtype=wp.vec3),
    vel_in: wp.array(dtype=wp.vec3),
    pred_pos: wp.array(dtype=wp.vec3),
    old_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    inv_dt: float,
    pos_out: wp.array(dtype=wp.vec3),
    vel_out: wp.array(dtype=wp.vec3),
):
    """Sigmoid mask variant — smooth around alpha=0.01 threshold."""
    i = wp.tid()
    a = alpha[i]
    f = fixed[i]
    # Smooth alive mask via sigmoid (k=40 → narrow transition around 0.01)
    alive = float(1.0) / (float(1.0) + wp.exp(-40.0 * (a - 0.01)))
    not_fixed = wp.where(f == 1, 0.0, 1.0)
    mask = alive * not_fixed

    new_vel = (pred_pos[i] - old_pos[i]) * inv_dt
    new_pos = pred_pos[i]

    vel_out[i] = mask * new_vel + (1.0 - mask) * vel_in[i]
    pos_out[i] = mask * new_pos + (1.0 - mask) * pos_in[i]


# Replace the production kernel reference with sigmoid version
km.update_velocity_kernel_ssa = update_velocity_kernel_ssa_sigmoid

# Also patch the solver's import (already imported the OLD version, so re-bind)
import pf_shape_matching_warp.sim.solver as solver_mod
solver_mod.update_velocity_kernel_ssa = update_velocity_kernel_ssa_sigmoid

from path4_falsification_test_H_isolate import setup_env, loss_sum_pos_z_kernel
from path4_poc_v1_dual_purpose import snap_solver_state, reset_solver_state


def fd_test(num_iter, knife_pos_np, eps=5e-3):
    env = setup_env()
    env.solver.num_iterations = num_iter
    env.solver.substeps = 1
    env.solver.use_split_cluster_solve = True
    env.solver.use_ssa_soft_removal = True
    env.solver.cutting_update_interval = 1
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
    print("  Phase 2C hypothesis test — sigmoid mask in update_velocity_kernel_ssa")
    print("=" * 78)
    print()
    knife_pos_np = (0.85, 0.5, 0.12)
    print(f"  {'num_iter':>10} {'autograd':>12} {'FD':>12} {'ratio':>10}")
    for num_iter in (1, 3, 6):
        ag, fd, r = fd_test(num_iter, knife_pos_np)
        print(f"  {num_iter:>10} {ag:>12.3e} {fd:>12.3e} {r:>10.3e}")
    print()
    print("  Expected if mask discreteness is core issue:")
    print("    num_iter=6: ratio ~0.94 (recovers POC v3 precision)")
    print("  Expected if NOT core issue:")
    print("    num_iter=6: ratio still ~0.10 (some other bug)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
