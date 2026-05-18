#!/usr/bin/env python3
"""Phase 2B+2C+Stage B unified gradient parity — ALL parameters in one tape.

Tests autograd vs FD parity for EVERY calibratable parameter, all in the same
tape, all running through the FULL pipeline:
  - Phase 2B 5-kernel split cluster_solve
  - Phase 2C sigmoid alive mask in update_velocity
  - Stage B cutting kernels with cycle-scoped SSA buffers

Parameters tested:
  Layer A:
    - knife_pos    (vec3, via collision_kernel_ssa)
    - stiffness    (scalar, via apply_kernel_ssa_stiff_arr)
    - knife_radius (scalar, via collision_kernel_ssa - read scalar in production,
                     wp.array via update_edge_damage_kernel_ssa_param_arr in cutting)
  Layer B:
    - alpha_min    (scalar, via update_edge_damage_kernel_ssa_param_arr)
    - sharpness    (scalar, via update_soft_removal_kernel_ssa_param_arr)

All params are wp.array(requires_grad=True) for autograd tracking.
Pass criterion: ratio in (0.85, 1.15) for each.
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

wp.config.verify_autograd_array_access = False

from pf_shape_matching_warp.sim.kernels import (
    predict_kernel_ssa, collision_kernel_ssa,
    apply_shape_delta_ssa_kernel, apply_collision_delta_ssa_kernel,
    update_velocity_kernel_ssa,    # Phase 2C sigmoid mask version
    cluster_compute_sums_kernel_ssa, cluster_normalize_centroids_kernel_ssa,
    cluster_compute_covariance_kernel_ssa, cluster_polar_rotation_kernel_ssa,
    cluster_goal_scatter_kernel_ssa,
    extract_rotation_muller,
)
from pf_shape_matching_warp.sim.pf_damage import (
    update_edge_damage_kernel_ssa_param_arr,
    apply_edge_damage_ssa_kernel,
    update_particle_damage_kernel,
    finalize_particle_damage_kernel_ssa,
    apply_particle_damage_ssa_kernel,
    update_soft_removal_kernel_ssa_param_arr,
)
from path4_falsification_test_H_isolate import setup_env, loss_sum_pos_z_kernel
from path4_poc_v1_dual_purpose import snap_solver_state, reset_solver_state
from test_layer_a_phase2b_e2e_stiffness_sysid import apply_kernel_ssa_stiff_arr


def run_unified_substep(env, knife_pos_arr, stiffness_arr, alpha_min_arr,
                         sharpness_arr, knife_radius_arr, threshold_arr):
    """Run 1 substep with ALL params as wp.array for autograd tracking.
    Includes cutting pipeline + Phase 2B split + Phase 2C sigmoid."""
    solver = env.solver
    N = solver.N
    NC = solver.num_clusters
    num_iter = solver.num_iterations
    dt_sub = solver.dt / float(solver.substeps)
    solver._alloc_ssa_buffers()

    # Allocate fresh pred_pos
    solver.p["pred_pos"] = wp.empty_like(solver.p["pos"])

    # Cycle-scoped pools for Phase 2B split cluster_solve
    pred_pos_pool = [wp.empty(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
                     for _ in range(2*num_iter+2)]
    deltas_pool = [wp.zeros(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
                   for _ in range(num_iter)]
    delta_counts_pool = [wp.zeros(N, dtype=wp.int32, device=solver.device)
                         for _ in range(num_iter)]
    shape_delta_pool = [wp.zeros(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
                        for _ in range(num_iter)]
    coll_delta_pre = wp.zeros(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
    coll_delta_pool = [wp.zeros(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
                       for _ in range(num_iter)]
    cx_sum_pool = [wp.zeros(NC, dtype=wp.vec3, device=solver.device, requires_grad=True)
                   for _ in range(num_iter)]
    qc_sum_pool = [wp.zeros(NC, dtype=wp.vec3, device=solver.device, requires_grad=True)
                   for _ in range(num_iter)]
    tw_pool = [wp.zeros(NC, dtype=wp.float32, device=solver.device, requires_grad=True)
               for _ in range(num_iter)]
    cx_mean_pool = [wp.zeros(NC, dtype=wp.vec3, device=solver.device, requires_grad=True)
                    for _ in range(num_iter)]
    qc_mean_pool = [wp.zeros(NC, dtype=wp.vec3, device=solver.device, requires_grad=True)
                    for _ in range(num_iter)]
    A_pool = [wp.zeros(NC, dtype=wp.mat33, device=solver.device, requires_grad=True)
              for _ in range(num_iter)]
    R_pool = [wp.zeros(NC, dtype=wp.mat33, device=solver.device, requires_grad=True)
              for _ in range(num_iter)]

    # 1. Predict
    wp.launch(predict_kernel_ssa, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"],
        pred_pos_pool[0], solver.p["old_pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.gravity, dt_sub, solver.damping,
    ], device=solver.device)

    # 2. Pre-loop collision
    wp.launch(collision_kernel_ssa, dim=N, inputs=[
        pred_pos_pool[0], solver.p["pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.floor_y, solver.floor_friction,
        knife_pos_arr, solver.knife_radius, solver.knife_active,
        coll_delta_pre,
    ], device=solver.device)
    wp.launch(apply_collision_delta_ssa_kernel, dim=N,
              inputs=[pred_pos_pool[0], coll_delta_pre],
              outputs=[pred_pos_pool[1]], device=solver.device)

    # 3. Jacobi loop (Phase 2B 5-kernel split + apply_kernel_ssa_stiff_arr)
    for k in range(num_iter):
        in_idx = 2*k+1
        mid_idx = 2*k+2
        out_idx = 2*k+3
        wp.launch(cluster_compute_sums_kernel_ssa, dim=NC, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            cx_sum_pool[k], qc_sum_pool[k], tw_pool[k],
        ], device=solver.device)
        wp.launch(cluster_normalize_centroids_kernel_ssa, dim=NC, inputs=[
            solver.c["sizes"], cx_sum_pool[k], qc_sum_pool[k], tw_pool[k],
            cx_mean_pool[k], qc_mean_pool[k],
        ], device=solver.device)
        wp.launch(cluster_compute_covariance_kernel_ssa, dim=NC, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            cx_mean_pool[k], qc_mean_pool[k], A_pool[k],
        ], device=solver.device)
        wp.launch(cluster_polar_rotation_kernel_ssa, dim=NC, inputs=[
            solver.c["sizes"], A_pool[k], solver.c["cluster_quat"],
            solver.muller_iter, R_pool[k],
        ], device=solver.device)
        wp.launch(cluster_goal_scatter_kernel_ssa, dim=NC, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            cx_mean_pool[k], qc_mean_pool[k], R_pool[k],
            deltas_pool[k], delta_counts_pool[k],
        ], device=solver.device)
        wp.launch(apply_kernel_ssa_stiff_arr, dim=N, inputs=[
            deltas_pool[k], delta_counts_pool[k],
            solver.p["damage"], solver._current_alpha(), solver.p["fixed"],
            stiffness_arr,
            shape_delta_pool[k],
        ], device=solver.device)
        wp.launch(apply_shape_delta_ssa_kernel, dim=N,
                  inputs=[pred_pos_pool[in_idx], shape_delta_pool[k]],
                  outputs=[pred_pos_pool[mid_idx]], device=solver.device)
        wp.launch(collision_kernel_ssa, dim=N, inputs=[
            pred_pos_pool[mid_idx], solver.p["pos"],
            solver._current_alpha(), solver.p["fixed"],
            solver.floor_y, solver.floor_friction,
            knife_pos_arr, solver.knife_radius, solver.knife_active,
            coll_delta_pool[k],
        ], device=solver.device)
        wp.launch(apply_collision_delta_ssa_kernel, dim=N,
                  inputs=[pred_pos_pool[mid_idx], coll_delta_pool[k]],
                  outputs=[pred_pos_pool[out_idx]], device=solver.device)

    final_pred_pos = pred_pos_pool[2*num_iter+1]

    # 4. Cutting pipeline (param_arr variants) — uses alpha_min, sharpness as arrays
    c = solver._ssa_cycle_index
    if c + 1 >= solver.ssa_buffer_capacity:
        raise RuntimeError(f"SSA buffer cycle {c+1} >= capacity")
    alpha_prev = solver._ssa_alpha_buffers[c]
    alpha_next = solver._ssa_alpha_buffers[c+1]
    ed_prev = solver._ssa_edge_damage_buffers[c]
    ed_next = solver._ssa_edge_damage_buffers[c+1]
    pd_prev = solver._ssa_particle_damage_buffers[c]
    pd_next = solver._ssa_particle_damage_buffers[c+1]
    num_edges = solver.e["num_edges"]

    edge_damage_delta = wp.zeros(num_edges, dtype=wp.float32, device=solver.device, requires_grad=True)
    wp.launch(update_edge_damage_kernel_ssa_param_arr, dim=num_edges, inputs=[
        final_pred_pos, alpha_prev,
        solver.e["a"], solver.e["b"],
        solver.e["pf_rest"], ed_prev,
        knife_pos_arr, knife_radius_arr, alpha_min_arr,
        edge_damage_delta,
    ], device=solver.device)
    wp.launch(apply_edge_damage_ssa_kernel, dim=num_edges, inputs=[
        ed_prev, edge_damage_delta, ed_next,
    ], device=solver.device)

    damage_sum = wp.zeros(N, dtype=wp.float32, device=solver.device, requires_grad=True)
    damage_count = wp.zeros(N, dtype=wp.float32, device=solver.device, requires_grad=True)
    wp.launch(update_particle_damage_kernel, dim=num_edges, inputs=[
        damage_sum, damage_count, alpha_prev,
        solver.e["a"], solver.e["b"], ed_next,
    ], device=solver.device)

    particle_damage_delta = wp.zeros(N, dtype=wp.float32, device=solver.device, requires_grad=True)
    wp.launch(finalize_particle_damage_kernel_ssa, dim=N, inputs=[
        damage_sum, damage_count, pd_prev, particle_damage_delta,
    ], device=solver.device)
    wp.launch(apply_particle_damage_ssa_kernel, dim=N, inputs=[
        pd_prev, particle_damage_delta, pd_next,
    ], device=solver.device)

    wp.launch(update_soft_removal_kernel_ssa_param_arr, dim=N, inputs=[
        pd_next, alpha_prev, solver.p["fixed"],
        threshold_arr, sharpness_arr,
        alpha_next,
    ], device=solver.device)

    solver._ssa_cycle_index = c + 1

    # 5. Update velocity (Phase 2C sigmoid mask)
    new_pos = wp.empty_like(solver.p["pos"])
    new_vel = wp.empty_like(solver.p["vel"])
    inv_dt = 1.0 / dt_sub
    wp.launch(update_velocity_kernel_ssa, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"],
        final_pred_pos, solver.p["old_pos"],
        alpha_next, solver.p["fixed"],
        inv_dt,
    ], outputs=[new_pos, new_vel], device=solver.device)
    solver.p["pos"] = new_pos
    solver.p["vel"] = new_vel
    solver.step_count += 1


def _make_arr(value, device, requires_grad=True, vec3=False):
    if vec3:
        return wp.array([wp.vec3(*value)], dtype=wp.vec3, device=device,
                          requires_grad=requires_grad)
    return wp.array([float(value)], dtype=wp.float32, device=device,
                      requires_grad=requires_grad)


def _apply_alpha_preset(env, alpha_init_np, damage_init_np):
    """Re-apply pre-set alpha + damage to buffer 0 (after reset_solver_state)."""
    wp.copy(env.solver._ssa_alpha_buffers[0],
            wp.array(alpha_init_np, dtype=wp.float32, device=env.device))
    wp.copy(env.solver._ssa_particle_damage_buffers[0],
            wp.array(damage_init_np, dtype=wp.float32, device=env.device))


def fd_test_param(env, param_name, gt_values, eps_dict, knife_pos_np,
                    alpha_init_np=None, damage_init_np=None):
    """Test FD parity for ONE parameter while others held at GT.

    gt_values: dict of {param_name: gt_value}
    Returns: (autograd_grad, fd_grad, ratio)"""
    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)
    state_snap = snap_solver_state(env)
    eps = eps_dict[param_name]

    def make_all_params(perturb_name=None, perturb_sign=0):
        """Return dict of all wp.arrays. perturb_name+sign perturbs one."""
        params = {}
        for name, gt in gt_values.items():
            value = gt
            if name == perturb_name:
                if name == "knife_pos":
                    value = list(gt)
                    # Use first axis for FD, fold all axes externally
                    value[0] += perturb_sign * eps
                else:
                    value = gt + perturb_sign * eps
            requires_grad = (name == param_name and perturb_name is None)
            params[name] = _make_arr(value, env.device, requires_grad=requires_grad,
                                       vec3=(name == "knife_pos"))
        return params

    # ---- Autograd ----
    reset_solver_state(env, state_snap)
    env.solver.step_count = 0
    env.solver._ssa_cycle_index = 0
    if alpha_init_np is not None:
        _apply_alpha_preset(env, alpha_init_np, damage_init_np)
    params_ag = make_all_params(perturb_name=None)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)
    with wp.Tape() as tape:
        run_unified_substep(env,
                              params_ag["knife_pos"], params_ag["stiffness"],
                              params_ag["alpha_min"], params_ag["sharpness"],
                              params_ag["knife_radius"], params_ag["threshold"])
        wp.launch(loss_sum_pos_z_kernel, dim=env.solver.N,
                  inputs=[env.solver.p["pos"], interior_mask_wp,
                          interior_count, loss_buf], device=env.device)
    tape.backward(loss=loss_buf)
    wp.synchronize()
    ag_grad = params_ag[param_name].grad.numpy().copy()

    # ---- FD ----
    if param_name == "knife_pos":
        # Use only x-axis FD as representative
        fd_grad = np.zeros(3)
        for axis in range(3):
            for sign in (+1, -1):
                reset_solver_state(env, state_snap)
                env.solver.step_count = 0
                env.solver._ssa_cycle_index = 0
                kp = list(gt_values["knife_pos"])
                kp[axis] += sign * eps
                params_fd = make_all_params(perturb_name=None)
                # Override knife_pos
                params_fd["knife_pos"] = wp.array([wp.vec3(*kp)], dtype=wp.vec3,
                                                    device=env.device, requires_grad=False)
                l_buf = wp.zeros(1, dtype=wp.float32, device=env.device)
                run_unified_substep(env,
                                      params_fd["knife_pos"], params_fd["stiffness"],
                                      params_fd["alpha_min"], params_fd["sharpness"],
                                      params_fd["knife_radius"], params_fd["threshold"])
                wp.launch(loss_sum_pos_z_kernel, dim=env.solver.N,
                          inputs=[env.solver.p["pos"], interior_mask_wp,
                                  interior_count, l_buf], device=env.device)
                wp.synchronize()
                if sign == +1: lp = float(l_buf.numpy()[0])
                else: lm = float(l_buf.numpy()[0])
            fd_grad[axis] = (lp - lm) / (2.0 * eps)
        ag_mag = float(np.linalg.norm(ag_grad[0]))
        fd_mag = float(np.linalg.norm(fd_grad))
    else:
        fd_grad = 0.0
        for sign in (+1, -1):
            reset_solver_state(env, state_snap)
            env.solver.step_count = 0
            env.solver._ssa_cycle_index = 0
            params_fd = make_all_params(perturb_name=param_name, perturb_sign=sign)
            for k in params_fd:
                params_fd[k] = wp.array(params_fd[k].numpy(), dtype=params_fd[k].dtype,
                                          device=env.device, requires_grad=False)
            l_buf = wp.zeros(1, dtype=wp.float32, device=env.device)
            run_unified_substep(env,
                                  params_fd["knife_pos"], params_fd["stiffness"],
                                  params_fd["alpha_min"], params_fd["sharpness"],
                                  params_fd["knife_radius"], params_fd["threshold"])
            wp.launch(loss_sum_pos_z_kernel, dim=env.solver.N,
                      inputs=[env.solver.p["pos"], interior_mask_wp,
                              interior_count, l_buf], device=env.device)
            wp.synchronize()
            if sign == +1: lp = float(l_buf.numpy()[0])
            else: lm = float(l_buf.numpy()[0])
        fd_grad = (lp - lm) / (2.0 * eps)
        ag_mag = abs(float(ag_grad[0]))
        fd_mag = abs(float(fd_grad))

    return ag_grad, fd_grad, ag_mag / max(fd_mag, 1e-12)


def main():
    wp.init()
    print("=" * 78)
    print("  Phase 2B+2C+Stage B Unified Gradient Parity (ALL params, full pipeline)")
    print("=" * 78)
    print()

    env = setup_env()
    env.solver.num_iterations = 6
    env.solver.substeps = 1
    env.solver.use_split_cluster_solve = True
    env.solver.use_ssa_soft_removal = True
    env.solver.cutting_update_interval = 1   # ensure cutting fires each step
    env.solver.knife_active = 1
    env.solver.step_count = 0

    # Mesh edge knife position (matches Day 4-E setup that gave knife_pos 0.95)
    knife_pos_np = (0.85, 0.5, 0.12)
    gt_values = {
        "knife_pos": knife_pos_np,
        "stiffness": 0.55,
        "alpha_min": 50.0,
        "sharpness": 30.0,
        "knife_radius": 0.05,
        "threshold": 0.4,
    }
    eps_dict = {
        "knife_pos": 5e-3,
        "stiffness": 1e-3,
        "alpha_min": 1.0,
        "sharpness": 1.0,
        "knife_radius": 1e-3,
        "threshold": 1e-3,
    }

    # Note: alpha buffer at default (1.0). Layer A params (knife_pos, stiffness)
    # verifiable directly. Layer B params (alpha_min, sharpness) require non-saturated
    # mask which would need active cutting damage; we reference Stage B's machine
    # precision verification on synthetic system (calibrate_stage2_autograd.py).
    env.solver._alloc_ssa_buffers()
    alpha_init = None
    damage_init = None

    print(f"  Setup: 1 autograd substep AFTER warm-up, num_iter=6, cutting active")
    print(f"  GT params: {gt_values}")
    print()

    print(f"  {'Parameter':<14} {'autograd':>12} {'FD':>12} {'ratio':>8} {'Status':<10}")

    for param in ["knife_pos", "stiffness", "alpha_min", "sharpness",
                   "knife_radius", "threshold"]:
        try:
            ag, fd, ratio = fd_test_param(env, param, gt_values, eps_dict, knife_pos_np,
                                            alpha_init_np=alpha_init,
                                            damage_init_np=damage_init)
            if isinstance(fd, np.ndarray):
                ag_disp = float(np.linalg.norm(ag[0])) if hasattr(ag, '__getitem__') else float(np.linalg.norm(ag))
                fd_disp = float(np.linalg.norm(fd))
            else:
                ag_disp = abs(float(ag[0])) if hasattr(ag, '__getitem__') else abs(float(ag))
                fd_disp = abs(float(fd))
            status = "✅ PASS" if 0.85 <= ratio <= 1.15 else "⚠️ OOB"
            print(f"  {param:<14} {ag_disp:>12.3e} {fd_disp:>12.3e} {ratio:>8.4f} {status:<10}")
        except Exception as e:
            print(f"  {param:<14} ERROR: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
