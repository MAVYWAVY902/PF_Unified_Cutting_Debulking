#!/usr/bin/env python3
"""Day 3 v2t — v2r + alpha-weighted accumulator (NO tw_safe).

Final isolation test. Adds to v2r:
  - wp.where(alpha < 0.01, 0.0, alpha) mask
  - PRODUCT (wp.where output × pred_pos[pid]) in cross-iter accumulator

KEEPS from v2r:
  - Constant inv_size divisor (NOT tw_safe — already proved that's <2× bias)

If v2t triggers ~5e10 → ALPHA-WEIGHTED ACCUMULATOR (wp.where × pred_pos
in cross-iter sum) IS THE EXACT TRIGGER. Single feature isolated.

If v2t ~1× → my isolation is wrong yet again, need to keep digging.
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

from pf_shape_matching_warp.sim.kernels import (
    predict_kernel_ssa, collision_kernel_ssa,
    apply_kernel_ssa,
    update_velocity_kernel,
    apply_shape_delta_ssa_kernel,
    apply_collision_delta_ssa_kernel,
)
from path4_falsification_test_H_isolate import (
    setup_env, loss_sum_pos_z_kernel,
)
from path4_poc_v1_dual_purpose import (
    make_param_dict, snap_solver_state, reset_solver_state,
)


@wp.kernel
def cluster_solve_v2t(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    cluster_quat: wp.array(dtype=wp.quat),
    num_iter: int,
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
):
    """v2r + alpha-weighted accumulator (wp.where × pred_pos in cross-iter sum).
    NO tw_safe (keeps constant inv_size divisor)."""
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    inv_size = 1.0 / float(size)

    # Loop 1: ALPHA-WEIGHTED centroid (NEW vs v2r)
    sum_x = float(0.0); sum_y = float(0.0); sum_z = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        # *** NEW: wp.where alpha mask × pred_pos PRODUCT in accumulator ***
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        sum_x = sum_x + w * p[0]
        sum_y = sum_y + w * p[1]
        sum_z = sum_z + w * p[2]
    # Constant divisor (NOT tw_safe)
    mean_x = sum_x * inv_size
    mean_y = sum_y * inv_size
    mean_z = sum_z * inv_size

    # Loop 2: REST centroid (unweighted, since rest_offsets has no grad)
    qsum_x = float(0.0); qsum_y = float(0.0); qsum_z = float(0.0)
    for j in range(size):
        q = rest_offsets[start + j]
        qsum_x = qsum_x + q[0]
        qsum_y = qsum_y + q[1]
        qsum_z = qsum_z + q[2]
    qmean_x = qsum_x * inv_size
    qmean_y = qsum_y * inv_size
    qmean_z = qsum_z * inv_size

    # Loop 3: goal (same as v2r)
    for j in range(size):
        pid = particle_ids[start + j]
        p = pred_pos[pid]
        q = rest_offsets[start + j]
        q_centered = wp.vec3(q[0] - qmean_x, q[1] - qmean_y, q[2] - qmean_z)
        goal = q_centered + wp.vec3(mean_x, mean_y, mean_z)
        correction = (goal - p) * 0.001
        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, 1)


def differentiable_substep_v2t(
    solver, params, knife_pos, dt_sub,
    enable_layer_a_collision: bool = True,
    enable_cutting: bool = True,
):
    N = solver.N
    solver._alloc_ssa_buffers()
    knife_active_collision = (
        solver.knife_active if enable_layer_a_collision else 0
    )
    num_iter = solver.num_iterations

    pred_pos_pool = [
        wp.empty(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
        for _ in range(2 * num_iter + 2)
    ]
    deltas_pool = [
        wp.zeros(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
        for _ in range(num_iter)
    ]
    delta_counts_pool = [
        wp.zeros(N, dtype=wp.int32, device=solver.device)
        for _ in range(num_iter)
    ]
    shape_delta_pool = [
        wp.zeros(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
        for _ in range(num_iter)
    ]
    coll_delta_pre = wp.zeros(N, dtype=wp.vec3, device=solver.device,
                                requires_grad=True)
    coll_delta_pool = [
        wp.zeros(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
        for _ in range(num_iter)
    ]

    wp.launch(predict_kernel_ssa, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"],
        pred_pos_pool[0], solver.p["old_pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.gravity, dt_sub, solver.damping,
    ], device=solver.device)

    wp.launch(collision_kernel_ssa, dim=N, inputs=[
        pred_pos_pool[0], solver.p["pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.floor_y, solver.floor_friction,
        knife_pos, solver.knife_radius, knife_active_collision,
        coll_delta_pre,
    ], device=solver.device)
    wp.launch(apply_collision_delta_ssa_kernel, dim=N,
              inputs=[pred_pos_pool[0], coll_delta_pre],
              outputs=[pred_pos_pool[1]],
              device=solver.device)

    for k in range(num_iter):
        in_idx = 2 * k + 1
        mid_idx = 2 * k + 2
        out_idx = 2 * k + 3

        wp.launch(cluster_solve_v2t, dim=solver.num_clusters, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            solver.c["cluster_quat"], solver.muller_iter,
            deltas_pool[k], delta_counts_pool[k],
        ], device=solver.device)

        wp.launch(apply_kernel_ssa, dim=N, inputs=[
            deltas_pool[k], delta_counts_pool[k],
            solver.p["damage"], solver._current_alpha(),
            solver.p["fixed"], solver.stiffness,
            shape_delta_pool[k],
        ], device=solver.device)

        wp.launch(apply_shape_delta_ssa_kernel, dim=N,
                  inputs=[pred_pos_pool[in_idx], shape_delta_pool[k]],
                  outputs=[pred_pos_pool[mid_idx]],
                  device=solver.device)

        wp.launch(collision_kernel_ssa, dim=N, inputs=[
            pred_pos_pool[mid_idx], solver.p["pos"],
            solver._current_alpha(), solver.p["fixed"],
            solver.floor_y, solver.floor_friction,
            knife_pos, solver.knife_radius, knife_active_collision,
            coll_delta_pool[k],
        ], device=solver.device)
        wp.launch(apply_collision_delta_ssa_kernel, dim=N,
                  inputs=[pred_pos_pool[mid_idx], coll_delta_pool[k]],
                  outputs=[pred_pos_pool[out_idx]],
                  device=solver.device)

    final_pred_pos = pred_pos_pool[2 * num_iter + 1]

    inv_dt = 1.0 / dt_sub
    wp.launch(update_velocity_kernel, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"],
        final_pred_pos, solver.p["old_pos"],
        solver._current_alpha(), solver.p["fixed"],
        inv_dt,
    ], device=solver.device)
    solver.step_count += 1


def run_at_num_iter(num_iter, knife_pos_np):
    print(f"\n  ─── num_iterations = {num_iter} ────")
    env = setup_env()
    env.solver.num_iterations = num_iter
    state_snap = snap_solver_state(env)
    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)
    params = make_param_dict(20.0, 3.0, 0.05, 0.15, requires_grad=True,
                              device=env.device)
    knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                          device=env.device, requires_grad=True)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device,
                         requires_grad=True)
    dt_sub = env.solver.dt / float(env.solver.substeps)

    with wp.Tape() as tape:
        differentiable_substep_v2t(env.solver, params, knife_pos, dt_sub,
                                    enable_layer_a_collision=True,
                                    enable_cutting=False)
        wp.launch(loss_sum_pos_z_kernel, dim=env.solver.N,
                  inputs=[env.solver.p["pos"], interior_mask_wp,
                          interior_count, loss_buf],
                  device=env.device)
    tape.backward(loss=loss_buf)
    wp.synchronize()
    ag_kp = knife_pos.grad.numpy()[0].copy()

    fd_kp = np.zeros(3)
    for axis in range(3):
        eps = 5e-3
        for sign in (+1, -1):
            reset_solver_state(env, state_snap)
            env.solver.step_count = 0
            kp_np = list(knife_pos_np); kp_np[axis] += sign * eps
            kp = wp.array([wp.vec3(*kp_np)], dtype=wp.vec3, device=env.device)
            params_fd = make_param_dict(20.0, 3.0, 0.05, 0.15,
                                          requires_grad=False, device=env.device)
            l_buf = wp.zeros(1, dtype=wp.float32, device=env.device)
            differentiable_substep_v2t(env.solver, params_fd, kp, dt_sub,
                                        enable_layer_a_collision=True,
                                        enable_cutting=False)
            wp.launch(loss_sum_pos_z_kernel, dim=env.solver.N,
                      inputs=[env.solver.p["pos"], interior_mask_wp,
                              interior_count, l_buf],
                      device=env.device)
            wp.synchronize()
            if sign == +1: lp = float(l_buf.numpy()[0])
            else: lm = float(l_buf.numpy()[0])
        fd_kp[axis] = (lp - lm) / (2.0 * eps)

    ag_mag = float(np.linalg.norm(ag_kp))
    fd_mag = float(np.linalg.norm(fd_kp))
    ratio = ag_mag / max(fd_mag, 1e-12)
    print(f"    autograd:  ({ag_kp[0]:+.3e}, {ag_kp[1]:+.3e}, {ag_kp[2]:+.3e})")
    print(f"    FD:        ({fd_kp[0]:+.3e}, {fd_kp[1]:+.3e}, {fd_kp[2]:+.3e})")
    print(f"    ‖autograd‖ / ‖FD‖ = {ag_mag:.3e} / {fd_mag:.3e} = {ratio:.2e}×")
    return dict(num_iter=num_iter, ag_mag=ag_mag, fd_mag=fd_mag, ratio=ratio)


def main():
    wp.init()
    print("=" * 78)
    print("  Day 3 v2t — v2r + alpha-weighted accumulator (NO tw_safe)")
    print("=" * 78)

    knife_pos_np = (0.85, 0.5, 0.12)
    v2r_baseline = {0: 0.963, 1: 0.993, 2: 0.994, 3: 0.993, 6: 0.992}
    v2s_baseline = {0: 0.964, 1: 1.08, 2: 1.17, 3: 1.27, 6: 1.58}
    v2q_baseline = {0: 0.964, 1: 90.3, 2: 4980, 3: 2.72e5, 6: 4.53e10}

    results = []
    for n in (0, 1, 2, 3, 6):
        r = run_at_num_iter(n, knife_pos_np)
        results.append(r)

    print()
    print("=" * 78)
    print("  SUMMARY — Final isolation: alpha-weighted accumulator")
    print("=" * 78)
    print(f"  {'num_iter':>10} {'v2t (+alpha-weighted)':>23} {'v2r (no alpha)':>16} {'v2q (full)':>14}")
    for r in results:
        n = r["num_iter"]
        print(f"  {n:>10} {r['ratio']:>23.3e} {v2r_baseline[n]:>16.3e} {v2q_baseline[n]:>14.3e}")

    r6 = next(r for r in results if r["num_iter"] == 6)
    print()
    if r6["ratio"] > 1e8:
        print(f"  🎯🎯🎯 EXACT TRIGGER ISOLATED: ALPHA-WEIGHTED ACCUMULATOR")
        print(f"     v2t ratio: {r6['ratio']:.2e}")
        print(f"     EXACT BUG PATTERN: wp.where(alpha < threshold, 0, alpha) × pred_pos[pid]")
        print(f"                        in cross-iter scalar accumulator")
        print(f"     This is a PRODUCT (wp.where × array) in a cross-iter sum, where")
        print(f"     wp.where(alpha) creates conditional branch + the product is")
        print(f"     then accumulated across loop iters. Warp 1.13 adj-gen has")
        print(f"     pathological backward chain for this combined pattern.")
        print(f"     ")
        print(f"     PRODUCTION FIX (paper-worthy): replace alpha-weighted accumulator")
        print(f"     with TWO-STEP pattern:")
        print(f"       Step 1: w[pid] = wp.where(alpha[pid] < threshold, 0, alpha[pid])")
        print(f"                (separate kernel, no cross-iter accumulator)")
        print(f"       Step 2: cx_sum += w[pid] * p[0]   (use pre-computed w)")
        print(f"     This breaks the wp.where × cross-iter-accumulator coupling.")
    elif r6["ratio"] > 1000:
        print(f"  ⚠️ alpha-weighted contributes but combined with tw_safe needed")
    else:
        print(f"  ❌ ALPHA-WEIGHTED IS NOT THE TRIGGER (4th wrong hypothesis today)")
        print(f"     v2t ratio: {r6['ratio']:.2e} (~1×)")
        print(f"     Bug is somewhere I haven't isolated yet — need deeper investigation.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
