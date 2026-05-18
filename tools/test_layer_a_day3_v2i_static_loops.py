#!/usr/bin/env python3
"""Day 3 v2i — Static-range loops in cluster_solve (Issue #1293 isolation).

Replaces the 4 sequential `for j in range(size)` dynamic-range loops in
cluster_solve_kernel_ssa with `for j in range(MAX_CLUSTER_SIZE=100)` +
alive-mask. Also replaces Müller polar's `for _ in range(num_iter)` with
fixed `for _ in range(8)` static range.

WHY: Issue #1293 ("Warp 1.13 nested dynamic-range loops give 2× wrong
gradient, wontfix") — agent research flagged this as remaining hypothesis.
Our cluster_solve has 4 sequential dynamic-range loops + Müller's inner
dynamic loop; if Issue #1293 is dominant, converting to static range
should restore gradient correctness.

FORWARD PHYSICS PRESERVED: out-of-cluster iterations have in_cluster=0
masking → zero contribution. Particle ids clamped to start+0 (always
in-cluster) so reads are safe. Atomic_add of zero corrections is no-op.

GRADIENT FLOW: out-of-cluster atomic_add propagates adj × 0 = 0 → no
spurious gradient. Static range is what Warp claims to handle correctly.

PASS condition: num_iter=6 ratio < 100× (down from 4.4×10¹⁰).
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
    cluster_quat_update_kernel,
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


# ===========================================================================
# Static-range Müller polar (mirror extract_rotation_muller but unrolled)
# ===========================================================================

@wp.func
def extract_rotation_muller_static8(A: wp.mat33, q_init: wp.quat) -> wp.quat:
    """Same as extract_rotation_muller but with fixed range(8) — STATIC.
    8 iterations is solver default (muller_iter=8); enough for warm-start."""
    q = q_init
    for _ in range(8):    # ← STATIC range, Python literal
        R = wp.quat_to_matrix(q)
        r0 = wp.vec3(R[0, 0], R[1, 0], R[2, 0])
        r1 = wp.vec3(R[0, 1], R[1, 1], R[2, 1])
        r2 = wp.vec3(R[0, 2], R[1, 2], R[2, 2])
        a0 = wp.vec3(A[0, 0], A[1, 0], A[2, 0])
        a1 = wp.vec3(A[0, 1], A[1, 1], A[2, 1])
        a2 = wp.vec3(A[0, 2], A[1, 2], A[2, 2])
        omega_num = wp.cross(r0, a0) + wp.cross(r1, a1) + wp.cross(r2, a2)
        omega_denom = wp.abs(wp.dot(r0, a0) + wp.dot(r1, a1) + wp.dot(r2, a2)) + 1.0e-9
        omega = omega_num / omega_denom
        w = wp.length(omega)
        axis = omega / (w + 1.0e-20)
        dq = wp.quat_from_axis_angle(axis, w)
        q = wp.normalize(wp.mul(dq, q))
    return q


# ===========================================================================
# Static-range cluster_solve — verified math-equivalent to cluster_solve_kernel_ssa
# 4 loops × `range(100)` instead of `range(size)`, with alive-mask.
# ===========================================================================

@wp.kernel
def cluster_solve_kernel_static(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    cluster_quat: wp.array(dtype=wp.quat),
    num_iter: int,    # IGNORED — we use fixed 8 in extract_rotation_muller_static8
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
):
    """Static-range version of cluster_solve_kernel_ssa.

    Forward equivalent: when j < size, full contribution; when j >= size,
    in_cluster mask zeroes contribution. Index clamped to start+0 (safe).

    Gradient flow: out-of-cluster atomic_add of zero correction propagates
    adj × 0 = 0 — no spurious gradient. Static range avoids Issue #1293.
    """
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    # --- 1. Weighted centroid (STATIC range) ---
    cx_sum = float(0.0)
    cy_sum = float(0.0)
    cz_sum = float(0.0)
    tw = float(0.0)

    for j in range(100):    # ← STATIC, MAX_CLUSTER_SIZE=100 (verified empirical max=95)
        in_cluster = wp.where(j < size, 1.0, 0.0)
        j_safe = wp.where(j < size, j, 0)    # clamp to in-cluster
        pid = particle_ids[start + j_safe]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw) * in_cluster    # mask out-of-cluster
        p = pred_pos[pid]
        cx_sum = cx_sum + w * p[0]
        cy_sum = cy_sum + w * p[1]
        cz_sum = cz_sum + w * p[2]
        tw = tw + w

    tw_safe = wp.where(tw < 1.5, 1.0, tw)
    cx_mean = cx_sum / tw_safe
    cy_mean = cy_sum / tw_safe
    cz_mean = cz_sum / tw_safe

    # --- 2a. Rest-offset centroid (STATIC range) ---
    qcx_sum = float(0.0)
    qcy_sum = float(0.0)
    qcz_sum = float(0.0)

    for j in range(100):
        in_cluster = wp.where(j < size, 1.0, 0.0)
        j_safe = wp.where(j < size, j, 0)
        pid = particle_ids[start + j_safe]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw) * in_cluster
        q = rest_offsets[start + j_safe]
        qcx_sum = qcx_sum + w * q[0]
        qcy_sum = qcy_sum + w * q[1]
        qcz_sum = qcz_sum + w * q[2]

    qcx_mean = qcx_sum / tw_safe
    qcy_mean = qcy_sum / tw_safe
    qcz_mean = qcz_sum / tw_safe

    # --- 2b. Covariance A_pq (STATIC range) ---
    a00 = float(0.0); a01 = float(0.0); a02 = float(0.0)
    a10 = float(0.0); a11 = float(0.0); a12 = float(0.0)
    a20 = float(0.0); a21 = float(0.0); a22 = float(0.0)

    for j in range(100):
        in_cluster = wp.where(j < size, 1.0, 0.0)
        j_safe = wp.where(j < size, j, 0)
        pid = particle_ids[start + j_safe]
        w_raw = alpha[pid]
        # binary alive mask × in_cluster mask
        alive = wp.where(w_raw < 0.01, 0.0, 1.0) * in_cluster
        p = pred_pos[pid]
        px = p[0] - cx_mean; py = p[1] - cy_mean; pz = p[2] - cz_mean
        q = rest_offsets[start + j_safe]
        qx = q[0] - qcx_mean; qy = q[1] - qcy_mean; qz = q[2] - qcz_mean
        a00 = a00 + alive * px * qx; a01 = a01 + alive * px * qy; a02 = a02 + alive * px * qz
        a10 = a10 + alive * py * qx; a11 = a11 + alive * py * qy; a12 = a12 + alive * py * qz
        a20 = a20 + alive * pz * qx; a21 = a21 + alive * pz * qy; a22 = a22 + alive * pz * qz

    A = wp.mat33(a00, a01, a02, a10, a11, a12, a20, a21, a22)

    # --- 3. Müller polar (STATIC, hardcoded range(8)) ---
    q_warm = cluster_quat[cid]
    q_new = extract_rotation_muller_static8(A, q_warm)
    R = wp.quat_to_matrix(q_new)

    # --- 4. Goal positions → atomic_add deltas (STATIC range) ---
    for j in range(100):
        in_cluster = wp.where(j < size, 1.0, 0.0)
        j_safe = wp.where(j < size, j, 0)
        pid = particle_ids[start + j_safe]
        w_raw = alpha[pid]
        # particle_alive masked by in_cluster — zero correction for out-of-cluster
        particle_alive = wp.where(w_raw < 0.01, 0.0, 1.0) * in_cluster
        q = rest_offsets[start + j_safe]
        q_centered = wp.vec3(q[0] - qcx_mean, q[1] - qcy_mean, q[2] - qcz_mean)
        goal = R * q_centered + wp.vec3(cx_mean, cy_mean, cz_mean)
        correction = (goal - pred_pos[pid]) * particle_alive

        # atomic_add of zero (when out-of-cluster) is mathematical no-op;
        # adj propagates × 0 = 0 — no spurious gradient.
        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, int(particle_alive))


# ===========================================================================
# fluid_checkpoint helper using cluster_solve_kernel_static
# ===========================================================================

def differentiable_substep_static(
    solver, params, knife_pos, dt_sub,
    enable_layer_a_collision: bool = True,
    enable_cutting: bool = True,
):
    """v2e fluid_checkpoint pattern + cluster_solve_kernel_static."""
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

        # *** STATIC-RANGE cluster_solve ***
        wp.launch(cluster_solve_kernel_static, dim=solver.num_clusters, inputs=[
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
        differentiable_substep_static(env.solver, params, knife_pos, dt_sub,
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
            differentiable_substep_static(env.solver, params_fd, kp, dt_sub,
                                              enable_layer_a_collision=True,
                                              enable_cutting=False)
            wp.launch(loss_sum_pos_z_kernel, dim=env.solver.N,
                      inputs=[env.solver.p["pos"], interior_mask_wp,
                              interior_count, l_buf],
                      device=env.device)
            wp.synchronize()
            if sign == +1:
                lp = float(l_buf.numpy()[0])
            else:
                lm = float(l_buf.numpy()[0])
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
    print("  Day 3 v2i — STATIC-range loops in cluster_solve (Issue #1293 isolation)")
    print("  4 loops × range(100) + Müller × range(8) — all STATIC")
    print("  Forward physics preserved via in_cluster mask")
    print("=" * 78)

    knife_pos_np = (0.85, 0.5, 0.12)
    v2e_baseline = {0: 0.964, 1: 90.4, 2: 4950, 3: 2.69e5, 6: 4.38e10}

    results = []
    for n in (0, 1, 2, 3, 6):
        r = run_at_num_iter(n, knife_pos_np)
        results.append(r)

    print()
    print("=" * 78)
    print("  SUMMARY — STATIC-range cluster_solve")
    print("=" * 78)
    print(f"  {'num_iter':>10} {'v2i (static)':>15} {'v2e (dynamic)':>17}  improvement")
    print(f"  {'-'*10} {'-'*15} {'-'*17}  {'-'*15}")
    for r in results:
        n = r["num_iter"]
        v2e = v2e_baseline[n]
        improvement = v2e / max(r["ratio"], 1e-12)
        print(f"  {n:>10} {r['ratio']:>15.3e} {v2e:>17.3e}  {improvement:>10.2f}×")

    r6 = next(r for r in results if r["num_iter"] == 6)
    print()
    if r6["ratio"] < 100:
        print(f"  🎯 ISSUE #1293 IS ROOT CAUSE")
        print(f"     num_iter=6 ratio: {v2e_baseline[6]:.0e} → {r6['ratio']:.2e}")
        print(f"     Path forward: production solver._substep should use static-range cluster_solve.")
        print(f"     Paper §3 contribution: 'first empirical isolation of Warp dynamic-range adjoint")
        print(f"     limit (Issue #1293) in differentiable shape-matching PBD'.")
    elif r6["ratio"] < v2e_baseline[6] / 10:
        print(f"  ⚠️ PARTIAL — dynamic-range loops contribute but aren't sole cause")
        print(f"     num_iter=6 ratio: {v2e_baseline[6]:.0e} → {r6['ratio']:.2e}")
    else:
        print(f"  ❌ ISSUE #1293 RULED OUT — dynamic-range is NOT the bug")
        print(f"     num_iter=6 ratio: {v2e_baseline[6]:.0e} → {r6['ratio']:.2e}")
        print(f"     Bug is elsewhere — most likely multi-writer atomic_add proper test required.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
