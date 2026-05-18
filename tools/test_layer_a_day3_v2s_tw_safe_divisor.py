#!/usr/bin/env python3
"""Day 3 v2s — v2r + tw_safe wp.where divisor (no alpha weighting).

Adds to v2r:
  - tw cross-iter accumulator (count, += 1 per iter)
  - tw_safe = wp.where(tw < 1.5, 1.0, tw)
  - Use tw_safe as divisor for centroid and rest centroid means

KEEPS from v2r (NOT v2q):
  - No alpha weighting in accumulators (just unweighted sum_x += p[0])

If v2s triggers → tw_safe wp.where on cross-iter accumulator IS the bug
If v2s ~1× → bug is alpha-weighted accumulator pattern (need v2t)
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
def cluster_solve_v2s(
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
    """v2r + tw cross-iter accumulator + tw_safe wp.where divisor.
    NO alpha weighting (unweighted sums)."""
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    # Loop 1: pred_pos centroid (unweighted) + tw count accumulator
    sum_x = float(0.0); sum_y = float(0.0); sum_z = float(0.0)
    tw = float(0.0)    # ← NEW: count accumulator
    for j in range(size):
        pid = particle_ids[start + j]
        p = pred_pos[pid]
        sum_x = sum_x + p[0]
        sum_y = sum_y + p[1]
        sum_z = sum_z + p[2]
        tw = tw + 1.0    # unweighted += 1 (= size at end)

    # ★ NEW: tw_safe wp.where on cross-iter accumulator ★
    tw_safe = wp.where(tw < 1.5, 1.0, tw)

    # Use tw_safe as divisor (mimics v2q)
    mean_x = sum_x / tw_safe
    mean_y = sum_y / tw_safe
    mean_z = sum_z / tw_safe

    # Loop 2: rest centroid (unweighted)
    qsum_x = float(0.0); qsum_y = float(0.0); qsum_z = float(0.0)
    for j in range(size):
        q = rest_offsets[start + j]
        qsum_x = qsum_x + q[0]
        qsum_y = qsum_y + q[1]
        qsum_z = qsum_z + q[2]

    # Use SAME tw_safe (cross-loop shared)
    qmean_x = qsum_x / tw_safe
    qmean_y = qsum_y / tw_safe
    qmean_z = qsum_z / tw_safe

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


def differentiable_substep_v2s(
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

        wp.launch(cluster_solve_v2s, dim=solver.num_clusters, inputs=[
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
        differentiable_substep_v2s(env.solver, params, knife_pos, dt_sub,
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
            differentiable_substep_v2s(env.solver, params_fd, kp, dt_sub,
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
    print("  Day 3 v2s — v2r + tw_safe wp.where divisor (no alpha weighting)")
    print("=" * 78)

    knife_pos_np = (0.85, 0.5, 0.12)
    v2r_baseline = {0: 0.963, 1: 0.993, 2: 0.994, 3: 0.993, 6: 0.992}
    v2q_baseline = {0: 0.964, 1: 90.3, 2: 4980, 3: 2.72e5, 6: 4.53e10}

    results = []
    for n in (0, 1, 2, 3, 6):
        r = run_at_num_iter(n, knife_pos_np)
        results.append(r)

    print()
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  {'num_iter':>10} {'v2s (+tw_safe)':>17} {'v2r (no tw_safe)':>18} {'v2q (full)':>14}")
    for r in results:
        n = r["num_iter"]
        print(f"  {n:>10} {r['ratio']:>17.3e} {v2r_baseline[n]:>18.3e} {v2q_baseline[n]:>14.3e}")

    r6 = next(r for r in results if r["num_iter"] == 6)
    print()
    if r6["ratio"] > 1e8:
        print(f"  🎯 tw_safe wp.where ON CROSS-ITER ACCUMULATOR IS THE BUG")
        print(f"     v2s ratio: {r6['ratio']:.2e}")
        print(f"     wp.where(tw < threshold, ...) where tw is built across iters")
        print(f"     creates pathological adj backward chain.")
        print(f"     Production fix: replace tw_safe with constant inv_size,")
        print(f"     OR use clamp/max instead of wp.where on the threshold.")
    elif r6["ratio"] > 1000:
        print(f"  ⚠️ tw_safe contributes but not sole — alpha weighting also needed")
    else:
        print(f"  ❌ tw_safe wp.where divisor IS NOT THE TRIGGER")
        print(f"     v2s ratio: {r6['ratio']:.2e} (~1×, like v2r)")
        print(f"     Bug is in ALPHA-WEIGHTED accumulator (alpha * pred_pos products)")
        print(f"     Next: v2t (add alpha weighting to v2r WITHOUT tw_safe)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
