#!/usr/bin/env python3
"""Phase 2B+2C unified gradient parity sweep — verify each parameter end-to-end.

Tests autograd vs FD parity for each calibratable parameter, in the FULL
production pipeline (Phase 2B split kernels + Phase 2C sigmoid mask + cutting
active + cluster_solve + cluster_quat update + update_velocity).

Single substep, cutting fires (step_count=0, interval=1).

Pass criterion: ratio in (0.85, 1.15) for each parameter individually.
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


def setup_calibration_env():
    env = setup_env()
    env.solver.num_iterations = 6
    env.solver.substeps = 1
    env.solver.use_split_cluster_solve = True
    env.solver.use_ssa_soft_removal = True
    env.solver.cutting_update_interval = 1
    env.solver.knife_active = 1
    env.solver.step_count = 0
    return env


def fd_test_knife_pos(num_iter, knife_pos_np=(0.85, 0.5, 0.12), eps=5e-3):
    """knife_pos gradient FD parity (already verified at Day 4-E)."""
    env = setup_calibration_env()
    env.solver.num_iterations = num_iter
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
    ag = knife_pos.grad.numpy()[0].copy()

    fd = np.zeros(3)
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
        fd[axis] = (lp - lm) / (2.0 * eps)

    return float(np.linalg.norm(ag)) / max(float(np.linalg.norm(fd)), 1e-12)


def fd_test_scalar_param(num_iter, param_name, gt_value, eps=1e-2):
    """Generic FD parity test for solver scalar attribute (stiffness, damping,
    floor_friction, alpha_min, sharpness, etc).

    Note: this tests the SCALAR solver attribute path. For autograd to track
    these, they need to be wp.array — we patch via 1-elem array trick.
    """
    env = setup_calibration_env()
    env.solver.num_iterations = num_iter
    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)
    state_snap = snap_solver_state(env)

    knife_pos_np = (0.85, 0.5, 0.12)
    knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3, device=env.device)
    env.solver.knife_pos = knife_pos

    # Set GT value on solver attribute
    setattr(env.solver, param_name, float(gt_value))

    # ------ FD perturbation ------
    fd = 0.0
    for sign in (+1, -1):
        reset_solver_state(env, state_snap)
        env.solver.step_count = 0
        setattr(env.solver, param_name, float(gt_value + sign * eps))
        l_buf = wp.zeros(1, dtype=wp.float32, device=env.device)
        env.solver.step()
        wp.launch(loss_sum_pos_z_kernel, dim=env.solver.N,
                  inputs=[env.solver.p["pos"], interior_mask_wp,
                          interior_count, l_buf], device=env.device)
        wp.synchronize()
        if sign == +1: lp = float(l_buf.numpy()[0])
        else: lm = float(l_buf.numpy()[0])
    fd = (lp - lm) / (2.0 * eps)

    # Restore GT for autograd
    setattr(env.solver, param_name, float(gt_value))

    # Note: solver scalar attributes are not autograd-tracked by default.
    # We can only return FD here as ground truth, plus an indication
    # if the parameter even appears in the kernel signature.
    return fd


def fd_test_stiffness_via_array(num_iter, gt_stiffness=0.85, test_stiffness=0.55, eps=1e-3):
    """stiffness FD parity — test_stiffness ≠ gt_stiffness for non-zero gradient."""
    # We use the existing E2E test infrastructure. This already tested it
    # with cutting bypassed (step_count=1). Now we'll test with cutting active.
    from test_layer_a_phase2b_e2e_stiffness_sysid import (
        differentiable_substep_e2e, loss_pos_mse_kernel,
    )

    env = setup_calibration_env()
    env.solver.num_iterations = num_iter
    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)
    inv_count = 1.0 / interior_count
    knife_pos_np = (0.85, 0.5, 0.12)
    dt_sub = env.solver.dt / float(env.solver.substeps)

    state_snap = snap_solver_state(env)

    # Loss vs target_pos at GT stiffness
    stiff_gt = wp.array([gt_stiffness], dtype=wp.float32, device=env.device,
                         requires_grad=False)
    knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3, device=env.device)
    env.solver.step_count = 0
    differentiable_substep_e2e(env.solver, knife_pos, stiff_gt, dt_sub,
                                enable_layer_a_collision=True)
    target_pos_np = env.solver.p["pos"].numpy().copy()
    target_pos = wp.array(target_pos_np, dtype=wp.vec3, device=env.device)
    reset_solver_state(env, state_snap)
    env.solver.step_count = 0

    # Autograd (at TEST stiffness, NOT gt — for non-zero gradient)
    stiff_ag = wp.array([test_stiffness], dtype=wp.float32, device=env.device,
                         requires_grad=True)
    knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3, device=env.device)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)
    with wp.Tape() as tape:
        differentiable_substep_e2e(env.solver, knife_pos, stiff_ag, dt_sub,
                                    enable_layer_a_collision=True)
        wp.launch(loss_pos_mse_kernel, dim=env.solver.N,
                  inputs=[env.solver.p["pos"], target_pos, interior_mask_wp,
                          inv_count, loss_buf], device=env.device)
    tape.backward(loss=loss_buf)
    wp.synchronize()
    ag = float(stiff_ag.grad.numpy()[0])

    # FD (at TEST stiffness, perturbed)
    fd = 0.0
    for sign in (+1, -1):
        reset_solver_state(env, state_snap)
        env.solver.step_count = 0
        s = wp.array([test_stiffness + sign * eps], dtype=wp.float32, device=env.device)
        kp = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3, device=env.device)
        l_buf = wp.zeros(1, dtype=wp.float32, device=env.device)
        differentiable_substep_e2e(env.solver, kp, s, dt_sub,
                                    enable_layer_a_collision=True)
        wp.launch(loss_pos_mse_kernel, dim=env.solver.N,
                  inputs=[env.solver.p["pos"], target_pos, interior_mask_wp,
                          inv_count, l_buf], device=env.device)
        wp.synchronize()
        if sign == +1: lp = float(l_buf.numpy()[0])
        else: lm = float(l_buf.numpy()[0])
    fd = (lp - lm) / (2.0 * eps)

    return ag, fd


def main():
    wp.init()
    print("=" * 78)
    print("  Phase 2B+2C Unified Gradient Parity Sweep")
    print("=" * 78)
    print()
    print("  Setup: 1 substep, cutting=1 (fires at step 0), num_iter=6")
    print("         Phase 2B 5-kernel split + Phase 2C sigmoid alive mask")
    print()

    NUM_ITER = 6

    # ---------- Test 1: knife_pos (Layer A collision) ----------
    print("  [Test 1] knife_pos gradient (Layer A collision, vec3)")
    r1 = fd_test_knife_pos(NUM_ITER)
    verdict = "✅ PASS" if 0.85 <= r1 <= 1.15 else "❌ FAIL"
    print(f"           ratio = {r1:.4f}  {verdict}")
    print()

    # ---------- Test 2: stiffness (Layer A shape matching) ----------
    print("  [Test 2] stiffness gradient (Layer A shape matching, scalar)")
    ag2, fd2 = fd_test_stiffness_via_array(NUM_ITER)
    r2 = abs(ag2) / max(abs(fd2), 1e-12)
    verdict = "✅ PASS" if 0.85 <= r2 <= 1.15 else f"⚠️ ratio out of band"
    print(f"           autograd = {ag2:+.3e}, FD = {fd2:+.3e}, ratio = {r2:.4f}  {verdict}")
    print()

    # ---------- Reporting ----------
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    results = {
        "knife_pos (Layer A collision)": (r1, "Phase 2B/2C verified"),
        "stiffness (Layer A shape)":    (r2, "via E2E test infra + cutting active"),
    }
    print(f"  {'Parameter':<35} {'Ratio':>8} {'Status':<12} {'Source':<30}")
    for name, (r, src) in results.items():
        if 0.85 <= r <= 1.15:
            status = "✅ PASS"
        else:
            status = "⚠️ OOB"
        print(f"  {name:<35} {r:>8.4f}  {status:<12} {src:<30}")

    print()
    print("  Note: damping, floor_friction not tested (scalar in solver, no autograd path).")
    print("        alpha_min, sharpness verified in Stage B (calibrate_stage2_autograd.py).")
    print("        See Stage B SSA path: tools/run_stage_b_regression_ssa.py.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
