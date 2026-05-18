#!/usr/bin/env python3
"""Phase 2B Day 0 POC v1 — Stage B-style 3-kernel split of cluster_solve.

Goal: keep production physics (alpha-weighted accumulator + variable divisor)
but isolate variable-divisor compute from atomic_add scatter. Test if Stage B
pattern fixes the F3 trigger we pinned in v2x.

3 kernels (each single-responsibility):
  1. cluster_compute_sums:    reduce → cx_sum, qc_sum, tw per cluster (NO div, NO scatter)
  2. cluster_normalize:       per-cluster scalar math: mean = sum / safe(tw)
  3. cluster_goal_scatter:    read means, compute correction, atomic_add to deltas

Variable divisor `/ tw_safe` now ISOLATED to kernel 2 (no multi-loop, no scatter).

Pass criterion: ratio at num_iter=6 in (0.95, 1.05) — comparable to v2v's 0.993.
  - v2q (broken):  4.5e10
  - v2x (no F3):   0.93   (correct sign, slight drift from dead-code F2)
  - v2v (no F2,F3): 0.99
  - POC target:    > 0.95 with full production physics
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


# ============================================================================
#                     SPLIT KERNELS — Stage B pattern
# ============================================================================

@wp.kernel
def cluster_compute_sums(
    pred_pos:       wp.array(dtype=wp.vec3),
    alpha:          wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes:  wp.array(dtype=wp.int32),
    particle_ids:   wp.array(dtype=wp.int32),
    rest_offsets:   wp.array(dtype=wp.vec3),
    # Outputs (per-cluster):
    cx_sum_buf:     wp.array(dtype=wp.vec3),    # weighted pred_pos sum
    qc_sum_buf:     wp.array(dtype=wp.vec3),    # weighted rest sum
    tw_buf:         wp.array(dtype=wp.float32), # alive-weighted total
):
    """Kernel 1: pure reduce. NO division, NO atomic_add to particle arrays."""
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    cx_sum = wp.vec3(0.0, 0.0, 0.0)
    qc_sum = wp.vec3(0.0, 0.0, 0.0)
    tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)    # KEEP F1 alpha-weighted
        p = pred_pos[pid]
        q = rest_offsets[start + j]
        cx_sum = cx_sum + w * p
        qc_sum = qc_sum + w * q
        tw = tw + w

    cx_sum_buf[cid] = cx_sum
    qc_sum_buf[cid] = qc_sum
    tw_buf[cid] = tw


@wp.kernel
def cluster_normalize_centroids(
    cluster_sizes:  wp.array(dtype=wp.int32),
    cx_sum_buf:     wp.array(dtype=wp.vec3),
    qc_sum_buf:     wp.array(dtype=wp.vec3),
    tw_buf:         wp.array(dtype=wp.float32),
    # Outputs:
    cx_mean_buf:    wp.array(dtype=wp.vec3),
    qc_mean_buf:    wp.array(dtype=wp.vec3),
):
    """Kernel 2: per-cluster scalar normalize — variable divisor ISOLATED here.
    No reduce loop, no scatter — single-thread-per-cluster math."""
    cid = wp.tid()
    size = cluster_sizes[cid]
    if size < 2:
        cx_mean_buf[cid] = wp.vec3(0.0, 0.0, 0.0)
        qc_mean_buf[cid] = wp.vec3(0.0, 0.0, 0.0)
        return

    tw = tw_buf[cid]
    tw_safe = wp.where(tw < 1.5, 1.0, tw)        # KEEP F2 safety guard
    cx_mean_buf[cid] = cx_sum_buf[cid] / tw_safe
    qc_mean_buf[cid] = qc_sum_buf[cid] / tw_safe


@wp.kernel
def cluster_goal_scatter(
    pred_pos:       wp.array(dtype=wp.vec3),
    alpha:          wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes:  wp.array(dtype=wp.int32),
    particle_ids:   wp.array(dtype=wp.int32),
    rest_offsets:   wp.array(dtype=wp.vec3),
    cx_mean_buf:    wp.array(dtype=wp.vec3),
    qc_mean_buf:    wp.array(dtype=wp.vec3),
    # Outputs:
    deltas:         wp.array(dtype=wp.vec3),
    delta_counts:   wp.array(dtype=wp.int32),
):
    """Kernel 3: scatter — atomic_add to particle deltas. NO reduce, NO divide."""
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    cx_mean = cx_mean_buf[cid]
    qc_mean = qc_mean_buf[cid]

    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        particle_alive = wp.where(w_raw < 0.01, 0.0, 1.0)
        q = rest_offsets[start + j]
        q_centered = q - qc_mean
        goal = q_centered + cx_mean
        correction = (goal - pred_pos[pid]) * particle_alive
        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, int(particle_alive))


# ============================================================================
#                     SUBSTEP DRIVER — wires the 3 kernels together
# ============================================================================

def differentiable_substep_split(
    solver, params, knife_pos, dt_sub,
    enable_layer_a_collision: bool = True,
    enable_cutting: bool = True,
):
    N = solver.N
    NC = solver.num_clusters
    solver._alloc_ssa_buffers()
    knife_active_collision = (
        solver.knife_active if enable_layer_a_collision else 0
    )
    num_iter = solver.num_iterations

    # Particle pred_pos pool — same as v2x
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

    # NEW: per-cluster cycle-scoped buffers (one per Jacobi iter)
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

        # Kernel 1: reduce sums
        wp.launch(cluster_compute_sums, dim=NC, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            cx_sum_pool[k], qc_sum_pool[k], tw_pool[k],
        ], device=solver.device)

        # Kernel 2: normalize (variable divisor isolated)
        wp.launch(cluster_normalize_centroids, dim=NC, inputs=[
            solver.c["sizes"],
            cx_sum_pool[k], qc_sum_pool[k], tw_pool[k],
            cx_mean_pool[k], qc_mean_pool[k],
        ], device=solver.device)

        # Kernel 3: scatter via atomic_add
        wp.launch(cluster_goal_scatter, dim=NC, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            cx_mean_pool[k], qc_mean_pool[k],
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
        differentiable_substep_split(env.solver, params, knife_pos, dt_sub,
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
            differentiable_substep_split(env.solver, params_fd, kp, dt_sub,
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
    print("  Phase 2B Day 0 POC v1 — Stage B-style 3-kernel split of cluster_solve")
    print("=" * 78)
    print("  Production physics PRESERVED (alpha-weighted accum + variable divisor)")
    print("  3 kernels: compute_sums → normalize → goal_scatter")
    print("  PASS: ratio at num_iter=6 in (0.95, 1.05)")
    print()

    knife_pos_np = (0.85, 0.5, 0.12)
    v2q_baseline = {0: 0.964, 1: 90.3, 2: 4980, 3: 2.72e5, 6: 4.53e10}
    v2x_baseline = {0: 0.963, 1: 0.965, 2: 0.958, 3: 0.949, 6: 0.932}
    v2v_baseline = {0: 0.963, 1: 0.993, 2: 0.993, 3: 0.993, 6: 0.993}

    results = []
    for n in (0, 1, 2, 3, 6):
        r = run_at_num_iter(n, knife_pos_np)
        results.append(r)

    print()
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  {'num_iter':>10} {'POC (split)':>14} {'v2q (broken)':>14} {'v2x (no F3)':>14} {'v2v (no F2,F3)':>16}")
    for r in results:
        n = r["num_iter"]
        print(f"  {n:>10} {r['ratio']:>14.3e} {v2q_baseline[n]:>14.3e} {v2x_baseline[n]:>14.3e} {v2v_baseline[n]:>16.3e}")

    r6 = next(r for r in results if r["num_iter"] == 6)
    print()
    if 0.95 <= r6["ratio"] <= 1.05:
        print(f"  🎯🎯🎯 POC PASS! Stage B pattern fixes Layer A WITH production physics.")
        print(f"     POC ratio at num_iter=6: {r6['ratio']:.3e} (target 0.95-1.05)")
        print(f"     CONCLUSION: Phase 2B production refactor (8-12 days) is GREENLIT.")
        print(f"     Trigger F3 (variable divisor) was killed by isolating it to a")
        print(f"     single-purpose normalize kernel separated from scatter atomic_add.")
    elif 0.5 < r6["ratio"] < 2.0:
        print(f"  ⚠️ POC PARTIAL — close but not in (0.95, 1.05) target band")
        print(f"     POC ratio: {r6['ratio']:.3e}")
        print(f"     Investigate: residual bias from per-cluster buffer adjoint chain?")
    else:
        print(f"  ❌ POC FAIL — split alone insufficient")
        print(f"     POC ratio: {r6['ratio']:.3e}")
        print(f"     Either: (a) need more aggressive split (4-5 kernels)")
        print(f"     Or:     (b) F3 trigger is more subtle than 'variable divisor in same kernel'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
