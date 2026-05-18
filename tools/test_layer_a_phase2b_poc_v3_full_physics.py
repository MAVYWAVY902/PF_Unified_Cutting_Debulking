#!/usr/bin/env python3
"""Phase 2B Day 0 POC v3 — Full production physics in Stage B-style 5-kernel split.

Adds to POC v1 (which used R=identity):
  - Kernel 3: cluster_compute_covariance — A_pq matrix per cluster (9 accumulators)
  - Kernel 4: cluster_polar_rotation     — Müller 2016 quaternion polar → R per cluster
  - Kernel 5: cluster_goal_scatter       — uses R_buf (NOT identity) in goal computation

Pipeline:
  pred_pos →
    [K1 cluster_compute_sums] → (cx_sum, qc_sum, tw) buffers per cluster →
    [K2 cluster_normalize] → (cx_mean, qc_mean) per cluster →
    [K3 cluster_compute_covariance] → A matrix per cluster →
    [K4 cluster_polar_rotation] → R matrix per cluster →
    [K5 cluster_goal_scatter] → atomic_add to deltas

Pass criterion: ratio at num_iter=6 in (0.85, 1.15) — wider band than POC v1 since
covariance + polar adds adjoint complexity. Catastrophic blow-up = >>1e3.

Goal: prove FULL production physics (not just centroid+goal) survives split pattern.
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
    extract_rotation_muller,    # @wp.func (Müller 2016 quaternion polar)
)
from path4_falsification_test_H_isolate import (
    setup_env, loss_sum_pos_z_kernel,
)
from path4_poc_v1_dual_purpose import (
    make_param_dict, snap_solver_state, reset_solver_state,
)


# ============================================================================
#                     SPLIT KERNELS — 5-kernel chain
# ============================================================================

@wp.kernel
def cluster_compute_sums(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    cx_sum_buf: wp.array(dtype=wp.vec3),
    qc_sum_buf: wp.array(dtype=wp.vec3),
    tw_buf: wp.array(dtype=wp.float32),
):
    """K1: Pure reduce. Per-cluster (cx_sum, qc_sum, tw)."""
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
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
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
    cluster_sizes: wp.array(dtype=wp.int32),
    cx_sum_buf: wp.array(dtype=wp.vec3),
    qc_sum_buf: wp.array(dtype=wp.vec3),
    tw_buf: wp.array(dtype=wp.float32),
    cx_mean_buf: wp.array(dtype=wp.vec3),
    qc_mean_buf: wp.array(dtype=wp.vec3),
):
    """K2: Per-cluster scalar normalize — variable divisor isolated."""
    cid = wp.tid()
    size = cluster_sizes[cid]
    if size < 2:
        cx_mean_buf[cid] = wp.vec3(0.0, 0.0, 0.0)
        qc_mean_buf[cid] = wp.vec3(0.0, 0.0, 0.0)
        return
    tw = tw_buf[cid]
    tw_safe = wp.where(tw < 1.5, 1.0, tw)
    cx_mean_buf[cid] = cx_sum_buf[cid] / tw_safe
    qc_mean_buf[cid] = qc_sum_buf[cid] / tw_safe


@wp.kernel
def cluster_compute_covariance(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    cx_mean_buf: wp.array(dtype=wp.vec3),
    qc_mean_buf: wp.array(dtype=wp.vec3),
    A_buf: wp.array(dtype=wp.mat33),
):
    """K3: Compute 3x3 covariance A_pq per cluster.
    Mirrors production loop 2b: alive (binary) mask, NOT alpha-weighted.
    """
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        A_buf[cid] = wp.mat33(0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0)
        return

    cx_mean = cx_mean_buf[cid]
    qc_mean = qc_mean_buf[cid]

    a00 = float(0.0); a01 = float(0.0); a02 = float(0.0)
    a10 = float(0.0); a11 = float(0.0); a12 = float(0.0)
    a20 = float(0.0); a21 = float(0.0); a22 = float(0.0)

    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        alive = wp.where(w_raw < 0.01, 0.0, 1.0)    # BINARY mask (matches production)
        p = pred_pos[pid]
        px = p[0] - cx_mean[0]
        py = p[1] - cx_mean[1]
        pz = p[2] - cx_mean[2]
        q = rest_offsets[start + j]
        qx = q[0] - qc_mean[0]
        qy = q[1] - qc_mean[1]
        qz = q[2] - qc_mean[2]
        a00 = a00 + alive * px * qx; a01 = a01 + alive * px * qy; a02 = a02 + alive * px * qz
        a10 = a10 + alive * py * qx; a11 = a11 + alive * py * qy; a12 = a12 + alive * py * qz
        a20 = a20 + alive * pz * qx; a21 = a21 + alive * pz * qy; a22 = a22 + alive * pz * qz

    A_buf[cid] = wp.mat33(a00, a01, a02,
                             a10, a11, a12,
                             a20, a21, a22)


@wp.kernel
def cluster_polar_rotation(
    cluster_sizes: wp.array(dtype=wp.int32),
    A_buf: wp.array(dtype=wp.mat33),
    cluster_quat: wp.array(dtype=wp.quat),    # warm-start (read only)
    num_iter: int,
    R_buf: wp.array(dtype=wp.mat33),
):
    """K4: Per-cluster Müller 2016 polar decomposition. Output R as mat33."""
    cid = wp.tid()
    size = cluster_sizes[cid]
    if size < 2:
        R_buf[cid] = wp.mat33(1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0)
        return
    A = A_buf[cid]
    q_warm = cluster_quat[cid]
    q_new = extract_rotation_muller(A, q_warm, num_iter)
    R_buf[cid] = wp.quat_to_matrix(q_new)


@wp.kernel
def cluster_goal_scatter_with_R(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    cx_mean_buf: wp.array(dtype=wp.vec3),
    qc_mean_buf: wp.array(dtype=wp.vec3),
    R_buf: wp.array(dtype=wp.mat33),
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
):
    """K5: scatter atomic_add. Uses R_buf rotation (NOT identity)."""
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return
    cx_mean = cx_mean_buf[cid]
    qc_mean = qc_mean_buf[cid]
    R = R_buf[cid]
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        particle_alive = wp.where(w_raw < 0.01, 0.0, 1.0)
        q = rest_offsets[start + j]
        q_centered = wp.vec3(q[0] - qc_mean[0], q[1] - qc_mean[1], q[2] - qc_mean[2])
        goal = R * q_centered + cx_mean    # Full production: R applied
        correction = (goal - pred_pos[pid]) * particle_alive
        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, int(particle_alive))


# ============================================================================
#                     SUBSTEP DRIVER
# ============================================================================

def differentiable_substep_split_v3(
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

    pred_pos_pool = [wp.empty(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
                      for _ in range(2 * num_iter + 2)]
    deltas_pool = [wp.zeros(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
                    for _ in range(num_iter)]
    delta_counts_pool = [wp.zeros(N, dtype=wp.int32, device=solver.device)
                          for _ in range(num_iter)]
    shape_delta_pool = [wp.zeros(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
                         for _ in range(num_iter)]
    coll_delta_pre = wp.zeros(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
    coll_delta_pool = [wp.zeros(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
                        for _ in range(num_iter)]

    # Per-cluster cycle-scoped buffers (one per Jacobi iter):
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

    wp.launch(predict_kernel_ssa, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"], pred_pos_pool[0], solver.p["old_pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.gravity, dt_sub, solver.damping,
    ], device=solver.device)
    wp.launch(collision_kernel_ssa, dim=N, inputs=[
        pred_pos_pool[0], solver.p["pos"], solver._current_alpha(), solver.p["fixed"],
        solver.floor_y, solver.floor_friction,
        knife_pos, solver.knife_radius, knife_active_collision, coll_delta_pre,
    ], device=solver.device)
    wp.launch(apply_collision_delta_ssa_kernel, dim=N,
              inputs=[pred_pos_pool[0], coll_delta_pre],
              outputs=[pred_pos_pool[1]], device=solver.device)

    for k in range(num_iter):
        in_idx = 2 * k + 1
        mid_idx = 2 * k + 2
        out_idx = 2 * k + 3

        # K1: reduce sums
        wp.launch(cluster_compute_sums, dim=NC, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            cx_sum_pool[k], qc_sum_pool[k], tw_pool[k],
        ], device=solver.device)
        # K2: normalize
        wp.launch(cluster_normalize_centroids, dim=NC, inputs=[
            solver.c["sizes"], cx_sum_pool[k], qc_sum_pool[k], tw_pool[k],
            cx_mean_pool[k], qc_mean_pool[k],
        ], device=solver.device)
        # K3: covariance
        wp.launch(cluster_compute_covariance, dim=NC, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            cx_mean_pool[k], qc_mean_pool[k],
            A_pool[k],
        ], device=solver.device)
        # K4: Müller polar rotation
        wp.launch(cluster_polar_rotation, dim=NC, inputs=[
            solver.c["sizes"], A_pool[k], solver.c["cluster_quat"],
            solver.muller_iter,
            R_pool[k],
        ], device=solver.device)
        # K5: scatter goal with R
        wp.launch(cluster_goal_scatter_with_R, dim=NC, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            cx_mean_pool[k], qc_mean_pool[k],
            R_pool[k],
            deltas_pool[k], delta_counts_pool[k],
        ], device=solver.device)

        wp.launch(apply_kernel_ssa, dim=N, inputs=[
            deltas_pool[k], delta_counts_pool[k],
            solver.p["damage"], solver._current_alpha(),
            solver.p["fixed"], solver.stiffness, shape_delta_pool[k],
        ], device=solver.device)
        wp.launch(apply_shape_delta_ssa_kernel, dim=N,
                  inputs=[pred_pos_pool[in_idx], shape_delta_pool[k]],
                  outputs=[pred_pos_pool[mid_idx]], device=solver.device)
        wp.launch(collision_kernel_ssa, dim=N, inputs=[
            pred_pos_pool[mid_idx], solver.p["pos"],
            solver._current_alpha(), solver.p["fixed"],
            solver.floor_y, solver.floor_friction,
            knife_pos, solver.knife_radius, knife_active_collision,
            coll_delta_pool[k],
        ], device=solver.device)
        wp.launch(apply_collision_delta_ssa_kernel, dim=N,
                  inputs=[pred_pos_pool[mid_idx], coll_delta_pool[k]],
                  outputs=[pred_pos_pool[out_idx]], device=solver.device)

    final_pred_pos = pred_pos_pool[2 * num_iter + 1]
    inv_dt = 1.0 / dt_sub
    wp.launch(update_velocity_kernel, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"], final_pred_pos, solver.p["old_pos"],
        solver._current_alpha(), solver.p["fixed"], inv_dt,
    ], device=solver.device)
    solver.step_count += 1


def run_at_num_iter(num_iter, knife_pos_np):
    print(f"\n  ─── num_iterations = {num_iter} ────")
    env = setup_env()
    env.solver.num_iterations = num_iter
    state_snap = snap_solver_state(env)
    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)
    params = make_param_dict(20.0, 3.0, 0.05, 0.15, requires_grad=True, device=env.device)
    knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                          device=env.device, requires_grad=True)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)
    dt_sub = env.solver.dt / float(env.solver.substeps)

    with wp.Tape() as tape:
        differentiable_substep_split_v3(env.solver, params, knife_pos, dt_sub,
                                          enable_layer_a_collision=True,
                                          enable_cutting=False)
        wp.launch(loss_sum_pos_z_kernel, dim=env.solver.N,
                  inputs=[env.solver.p["pos"], interior_mask_wp,
                          interior_count, loss_buf], device=env.device)
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
            params_fd = make_param_dict(20.0, 3.0, 0.05, 0.15, requires_grad=False, device=env.device)
            l_buf = wp.zeros(1, dtype=wp.float32, device=env.device)
            differentiable_substep_split_v3(env.solver, params_fd, kp, dt_sub,
                                              enable_layer_a_collision=True,
                                              enable_cutting=False)
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
    print(f"    autograd:  ({ag_kp[0]:+.3e}, {ag_kp[1]:+.3e}, {ag_kp[2]:+.3e})")
    print(f"    FD:        ({fd_kp[0]:+.3e}, {fd_kp[1]:+.3e}, {fd_kp[2]:+.3e})")
    print(f"    ‖autograd‖ / ‖FD‖ = {ag_mag:.3e} / {fd_mag:.3e} = {ratio:.2e}×")
    return dict(num_iter=num_iter, ag_mag=ag_mag, fd_mag=fd_mag, ratio=ratio)


def main():
    wp.init()
    print("=" * 78)
    print("  Phase 2B Day 0 POC v3 — Full production physics (covariance + Müller polar)")
    print("=" * 78)
    print("  5-kernel split: sums → normalize → covariance → polar → goal_scatter")
    print("  PASS: ratio at num_iter=6 in (0.85, 1.15) — no catastrophic blow-up")
    print()
    knife_pos_np = (0.85, 0.5, 0.12)
    poc_v1_baseline = {0: 0.963, 1: 0.964, 2: 0.958, 3: 0.949, 6: 0.932}
    v2q_baseline = {0: 0.964, 1: 90.3, 2: 4980, 3: 2.72e5, 6: 4.53e10}
    results = []
    for n in (0, 1, 2, 3, 6):
        results.append(run_at_num_iter(n, knife_pos_np))
    print()
    print("=" * 78)
    print("  SUMMARY — POC v3 (full physics: cov + polar + R)")
    print("=" * 78)
    print(f"  {'num_iter':>10} {'POC v3':>14} {'POC v1 (R=I)':>14} {'v2q (broken)':>14}")
    for r in results:
        n = r["num_iter"]
        print(f"  {n:>10} {r['ratio']:>14.3e} {poc_v1_baseline[n]:>14.3e} {v2q_baseline[n]:>14.3e}")
    r6 = next(r for r in results if r["num_iter"] == 6)
    print()
    if 0.85 <= r6["ratio"] <= 1.15:
        print(f"  🎯🎯🎯 POC v3 PASS! Full production physics in split mode is clean.")
        print(f"     POC v3 ratio at num_iter=6: {r6['ratio']:.3e} (target 0.85-1.15)")
        print(f"     CONCLUSION: covariance + polar layers add NO catastrophic adjoint bias.")
        print(f"     Phase 2B production refactor is FULLY GREENLIT.")
    elif 0.5 <= r6["ratio"] < 0.85 or 1.15 < r6["ratio"] <= 2.0:
        print(f"  ⚠️ partial — covariance/polar adds some bias but not catastrophic")
        print(f"     POC v3 ratio: {r6['ratio']:.3e}")
    else:
        print(f"  ❌ POC v3 FAIL — {r6['ratio']:.3e}")
        print(f"     Investigate: covariance multi-loop or polar quaternion adj?")

    return 0


if __name__ == "__main__":
    sys.exit(main())
