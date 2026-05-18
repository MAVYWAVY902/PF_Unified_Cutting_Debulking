#!/usr/bin/env python3
"""Sanity check — run Day 3 FD parity in LEGACY mode (use_ssa_soft_removal=False).
Should reproduce original Test E result (knife_pos rel_err ~10^9).
If yes, our test framework is correct and Day 3 result reflects real
post-fix behavior.

If legacy mode also shows much worse than 10^9, the test framework is buggy.
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

wp.config.verify_autograd_array_access = False

from path4_poc_v1_dual_purpose import snap_solver_state, reset_solver_state


@wp.kernel
def loss_sum_pos_z_kernel(
    pos: wp.array(dtype=wp.vec3),
    interior_mask: wp.array(dtype=wp.int32),
    interior_count: float,
    loss_buf: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if interior_mask[i] == 1:
        wp.atomic_add(loss_buf, 0, pos[i][2] / interior_count)


def setup_env(use_ssa: bool):
    from warp_sim.envs.debulk_env import DebulkEnv
    cfg_path = REPO_ROOT / "warp_sim" / "config" / "debulk_cap_armA.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["obs_mode"] = "baseline"
    cfg["task_type"] = "capsule_only"
    cfg["use_ssa_soft_removal"] = use_ssa
    env = DebulkEnv(cfg)
    env.reset()
    env.solver.cutting_update_interval = 1
    env.solver.knife_active = 1
    env.solver.knife_radius = 0.15
    env.solver.aspiration_threshold = 0.05
    env.solver.alpha_min = 3.0
    env.solver.soft_removal_sharpness = 20.0
    if use_ssa:
        env.solver._alloc_ssa_buffers()
    return env


def run_substep(env, knife_pos_xyz, requires_grad, interior_mask_wp, interior_count):
    solver = env.solver
    solver.knife_pos = wp.array(
        [wp.vec3(*knife_pos_xyz)], dtype=wp.vec3,
        device=solver.device, requires_grad=requires_grad,
    )
    loss_buf = wp.zeros(1, dtype=wp.float32, device=solver.device,
                         requires_grad=requires_grad)
    if requires_grad:
        tape = wp.Tape()
        with tape:
            solver._substep()
            wp.launch(loss_sum_pos_z_kernel, dim=solver.N,
                      inputs=[solver.p["pos"], interior_mask_wp,
                              interior_count, loss_buf], device=solver.device)
        return loss_buf, solver.knife_pos, tape
    else:
        solver._substep()
        wp.launch(loss_sum_pos_z_kernel, dim=solver.N,
                  inputs=[solver.p["pos"], interior_mask_wp,
                          interior_count, loss_buf], device=solver.device)
        wp.synchronize()
        return float(loss_buf.numpy()[0])


def measure(env, num_iter, knife_pos_xyz=(0.85, 0.5, 0.12), eps=5e-3):
    env.solver.num_iterations = num_iter
    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)
    snap = snap_solver_state(env)

    reset_solver_state(env, snap)
    loss_buf, kp_array, tape = run_substep(env, knife_pos_xyz, True,
                                              interior_mask_wp, interior_count)
    tape.backward(loss=loss_buf)
    wp.synchronize()
    autograd_grad = kp_array.grad.numpy()[0].copy()

    fd_grad = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        reset_solver_state(env, snap)
        kp_p = list(knife_pos_xyz); kp_p[axis] += eps
        loss_p = run_substep(env, kp_p, False, interior_mask_wp, interior_count)
        reset_solver_state(env, snap)
        kp_m = list(knife_pos_xyz); kp_m[axis] -= eps
        loss_m = run_substep(env, kp_m, False, interior_mask_wp, interior_count)
        fd_grad[axis] = (loss_p - loss_m) / (2 * eps)

    return autograd_grad, fd_grad


def main():
    wp.init()
    print("=" * 78)
    print("LEGACY MODE BASELINE — reproduce original bug (use_ssa_soft_removal=False)")
    print("=" * 78)
    env = setup_env(use_ssa=False)
    print(f"  num_iter   ‖autograd‖             ‖FD‖           ratio")
    for num_iter in [0, 1, 6]:
        ag, fd = measure(env, num_iter)
        ag_norm = float(np.linalg.norm(ag))
        fd_norm = float(np.linalg.norm(fd))
        ratio = ag_norm / max(fd_norm, 1e-12)
        print(f"  {num_iter:>8}   {ag_norm:>15.4e}   {fd_norm:>15.4e}   {ratio:>12.3e}")

    print()
    print("=" * 78)
    print("SSA MODE — post-fix (use_ssa_soft_removal=True)")
    print("=" * 78)
    env = setup_env(use_ssa=True)
    print(f"  num_iter   ‖autograd‖             ‖FD‖           ratio")
    for num_iter in [0, 1, 6]:
        ag, fd = measure(env, num_iter)
        ag_norm = float(np.linalg.norm(ag))
        fd_norm = float(np.linalg.norm(fd))
        ratio = ag_norm / max(fd_norm, 1e-12)
        print(f"  {num_iter:>8}   {ag_norm:>15.4e}   {fd_norm:>15.4e}   {ratio:>12.3e}")


if __name__ == "__main__":
    main()
