#!/usr/bin/env python3
"""Day 3 v2k — Scatter-gather pattern (PROPER multi-writer atomic_add isolation).

Replaces cluster_solve_kernel_ssa's `wp.atomic_add(deltas, pid, correction)`
multi-writer pattern with a TWO-STAGE scatter-gather:

  Stage 1 (cluster_scatter_kernel):
    Per-cluster thread computes correction for each particle-in-cluster,
    writes to cluster_corrections[cluster_start + j_local] via PLAIN
    ASSIGNMENT (single writer per slot — no atomic_add).

  Stage 2 (gather_corrections_kernel):
    Per-particle thread iterates its cluster memberships (via inverse CSR)
    and sums corrections from all clusters containing it. Writes deltas[pid]
    via PLAIN ASSIGNMENT (single writer per particle — no atomic_add).

Forward physics is PRESERVED exactly (same sum of corrections from all
clusters per particle, same delta_counts per particle).

THE TEST: this is the only remaining hypothesis we haven't isolated.
Forward atomic_add into deltas is now eliminated. If gradient still wrong
~10⁹×, the bug is in some OTHER aspect of cluster_solve's backward chain
(possibly multi-writer atomic_add in adj_pred_pos which is auto-generated
by Warp in backward — independent of forward pattern).
"""
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

wp.config.verify_autograd_array_access = False

from pf_shape_matching_warp.sim.kernels import (
    extract_rotation_muller,
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
# Inverse CSR build (numpy, vectorized)
# ===========================================================================

def build_inverse_csr(cluster_starts_np, cluster_sizes_np, particle_ids_np, N_particles):
    """Build inverse CSR: for each particle, list of global CSR positions
    (== cluster_starts[cid] + j_local for clusters containing this particle).

    Returns:
        inv_starts (N,)           — starting index in inv_global_indices
        inv_sizes (N,)            — number of clusters containing each particle
        inv_global_indices (M,)   — flat array, M = sum(cluster_sizes)
                                    inv_global_indices[inv_starts[pid] + k] is the
                                    global CSR position of particle pid's k-th cluster
    """
    all_pids = particle_ids_np  # already a flat list of length M = sum(sizes)
    M = len(all_pids)

    # Sort positions by particle ID (stable to preserve cluster order)
    order = np.argsort(all_pids, kind='stable').astype(np.int32)
    sorted_pids = all_pids[order]

    # inv_starts[pid] = first position in `order` where the particle is pid
    inv_starts = np.searchsorted(sorted_pids, np.arange(N_particles)).astype(np.int32)

    # inv_sizes[pid] = count via bincount
    counts = np.bincount(all_pids, minlength=N_particles).astype(np.int32)

    return inv_starts, counts, order


# ===========================================================================
# Stage 1: cluster_scatter_kernel — writes per-(cluster, j_local) corrections
# ===========================================================================

@wp.kernel
def cluster_scatter_kernel(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    cluster_quat: wp.array(dtype=wp.quat),
    num_iter: int,
    cluster_corrections: wp.array(dtype=wp.vec3),     # OUTPUT: shape M, single writer per slot
    cluster_alive_flags: wp.array(dtype=wp.float32),  # OUTPUT: shape M, particle_alive per (cluster,j)
):
    """Same shape-matching math as cluster_solve_kernel_ssa, but writes
    per-(cluster, j_local) corrections to cluster_corrections[start+j]
    via assignment (NO atomic_add, single writer per slot)."""
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        # Zero out this cluster's slots; can't return because j may not all be visited
        for j in range(100):
            in_cluster = wp.where(j < size, 1.0, 0.0)
            j_safe = wp.where(j < size, j, 0)
            cluster_corrections[start + j_safe] = wp.vec3(0.0, 0.0, 0.0) * in_cluster
            cluster_alive_flags[start + j_safe] = 0.0 * in_cluster
        return

    # --- 1. Weighted centroid ---
    cx_sum = float(0.0); cy_sum = float(0.0); cz_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        cx_sum = cx_sum + w * p[0]; cy_sum = cy_sum + w * p[1]; cz_sum = cz_sum + w * p[2]
        tw = tw + w
    tw_safe = wp.where(tw < 1.5, 1.0, tw)
    cx_mean = cx_sum / tw_safe
    cy_mean = cy_sum / tw_safe
    cz_mean = cz_sum / tw_safe

    # --- 2a. Rest centroid ---
    qcx_sum = float(0.0); qcy_sum = float(0.0); qcz_sum = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        q = rest_offsets[start + j]
        qcx_sum = qcx_sum + w * q[0]; qcy_sum = qcy_sum + w * q[1]; qcz_sum = qcz_sum + w * q[2]
    qcx_mean = qcx_sum / tw_safe
    qcy_mean = qcy_sum / tw_safe
    qcz_mean = qcz_sum / tw_safe

    # --- 2b. Covariance ---
    a00 = float(0.0); a01 = float(0.0); a02 = float(0.0)
    a10 = float(0.0); a11 = float(0.0); a12 = float(0.0)
    a20 = float(0.0); a21 = float(0.0); a22 = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        alive = wp.where(w_raw < 0.01, 0.0, 1.0)
        p = pred_pos[pid]
        px = p[0] - cx_mean; py = p[1] - cy_mean; pz = p[2] - cz_mean
        q = rest_offsets[start + j]
        qx = q[0] - qcx_mean; qy = q[1] - qcy_mean; qz = q[2] - qcz_mean
        a00 = a00 + alive*px*qx; a01 = a01 + alive*px*qy; a02 = a02 + alive*px*qz
        a10 = a10 + alive*py*qx; a11 = a11 + alive*py*qy; a12 = a12 + alive*py*qz
        a20 = a20 + alive*pz*qx; a21 = a21 + alive*pz*qy; a22 = a22 + alive*pz*qz
    A = wp.mat33(a00, a01, a02, a10, a11, a12, a20, a21, a22)

    # --- 3. Müller polar ---
    q_warm = cluster_quat[cid]
    q_new = extract_rotation_muller(A, q_warm, num_iter)
    R = wp.quat_to_matrix(q_new)

    # --- 4. Goal positions → ASSIGNMENT to per-(cluster, j_local) buffer ---
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        particle_alive = wp.where(w_raw < 0.01, 0.0, 1.0)
        q = rest_offsets[start + j]
        q_centered = wp.vec3(q[0] - qcx_mean, q[1] - qcy_mean, q[2] - qcz_mean)
        goal = R * q_centered + wp.vec3(cx_mean, cy_mean, cz_mean)
        correction = (goal - pred_pos[pid]) * particle_alive

        # *** ASSIGNMENT (single writer per slot — no atomic_add) ***
        cluster_corrections[start + j] = correction
        cluster_alive_flags[start + j] = particle_alive


# ===========================================================================
# Stage 2: gather_corrections_kernel — sum per-particle from all its clusters
# ===========================================================================

@wp.kernel
def gather_corrections_kernel(
    cluster_corrections: wp.array(dtype=wp.vec3),     # IN: M slots, written by scatter
    cluster_alive_flags: wp.array(dtype=wp.float32),  # IN: M slots
    inv_starts: wp.array(dtype=wp.int32),             # IN: per-particle offsets
    inv_sizes: wp.array(dtype=wp.int32),              # IN: per-particle counts
    inv_global_indices: wp.array(dtype=wp.int32),     # IN: M slots, maps inv-position→global CSR position
    deltas: wp.array(dtype=wp.vec3),                  # OUT: per-particle correction sum (single writer)
    delta_counts: wp.array(dtype=wp.int32),           # OUT: per-particle alive count (single writer)
):
    """Per-particle thread sums corrections from all clusters containing it.
    Single writer per (deltas, delta_counts) cell — no atomic_add."""
    pid = wp.tid()
    inv_start = inv_starts[pid]
    inv_size = inv_sizes[pid]

    sum_x = float(0.0); sum_y = float(0.0); sum_z = float(0.0)
    count_alive = float(0.0)

    # Static range for safety; cap at 200 (max known per-particle ~150 assuming worst-case)
    # Actually for our setup max is ~150. Use 200 for safety.
    for k in range(200):
        in_range = wp.where(k < inv_size, 1.0, 0.0)
        k_safe = wp.where(k < inv_size, k, 0)
        global_idx = inv_global_indices[inv_start + k_safe]
        c = cluster_corrections[global_idx]
        a = cluster_alive_flags[global_idx]
        sum_x = sum_x + c[0] * in_range
        sum_y = sum_y + c[1] * in_range
        sum_z = sum_z + c[2] * in_range
        count_alive = count_alive + a * in_range

    # *** ASSIGNMENT ***
    deltas[pid] = wp.vec3(sum_x, sum_y, sum_z)
    delta_counts[pid] = int(count_alive)


# ===========================================================================
# fluid_checkpoint helper using scatter+gather instead of cluster_solve atomic_add
# ===========================================================================

def differentiable_substep_scatter_gather(
    solver, params, knife_pos, dt_sub,
    inv_starts_wp, inv_sizes_wp, inv_global_indices_wp, total_memberships,
    enable_layer_a_collision: bool = True,
    enable_cutting: bool = True,
):
    """v2e fluid_checkpoint pattern + scatter-gather replacement of cluster_solve."""
    N = solver.N
    solver._alloc_ssa_buffers()

    knife_active_collision = (
        solver.knife_active if enable_layer_a_collision else 0
    )
    num_iter = solver.num_iterations
    M = total_memberships

    pred_pos_pool = [
        wp.empty(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
        for _ in range(2 * num_iter + 2)
    ]
    cluster_corrections_pool = [
        wp.zeros(M, dtype=wp.vec3, device=solver.device, requires_grad=True)
        for _ in range(num_iter)
    ]
    cluster_alive_pool = [
        wp.zeros(M, dtype=wp.float32, device=solver.device, requires_grad=True)
        for _ in range(num_iter)
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

        # *** SCATTER (no atomic_add) ***
        wp.launch(cluster_scatter_kernel, dim=solver.num_clusters, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            solver.c["cluster_quat"], solver.muller_iter,
            cluster_corrections_pool[k], cluster_alive_pool[k],
        ], device=solver.device)

        # *** GATHER (no atomic_add) ***
        wp.launch(gather_corrections_kernel, dim=N, inputs=[
            cluster_corrections_pool[k], cluster_alive_pool[k],
            inv_starts_wp, inv_sizes_wp, inv_global_indices_wp,
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


def run_at_num_iter(num_iter, knife_pos_np, inv_starts_wp, inv_sizes_wp,
                      inv_global_indices_wp, total_memberships):
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
        differentiable_substep_scatter_gather(
            env.solver, params, knife_pos, dt_sub,
            inv_starts_wp, inv_sizes_wp, inv_global_indices_wp, total_memberships,
            enable_layer_a_collision=True, enable_cutting=False)
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
            differentiable_substep_scatter_gather(
                env.solver, params_fd, kp, dt_sub,
                inv_starts_wp, inv_sizes_wp, inv_global_indices_wp, total_memberships,
                enable_layer_a_collision=True, enable_cutting=False)
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
    print("  Day 3 v2k — SCATTER-GATHER (proper multi-writer atomic_add isolation)")
    print("  Forward physics PRESERVED. No forward atomic_add into deltas.")
    print("=" * 78)

    # Build inverse CSR once
    print("\n  Building inverse CSR...")
    t0 = time.time()
    env_init = setup_env()
    cluster_starts_np = env_init.solver.c["starts"].numpy()
    cluster_sizes_np = env_init.solver.c["sizes"].numpy()
    particle_ids_np = env_init.solver.c["particle_ids"].numpy()
    N_particles = env_init.solver.N

    inv_starts_np, inv_sizes_np, inv_global_indices_np = build_inverse_csr(
        cluster_starts_np, cluster_sizes_np, particle_ids_np, N_particles
    )
    total_memberships = int(inv_sizes_np.sum())
    max_per_particle = int(inv_sizes_np.max())
    print(f"  Built in {time.time()-t0:.1f}s. "
          f"total memberships M={total_memberships}, "
          f"per-particle max={max_per_particle}, mean={inv_sizes_np.mean():.1f}")

    inv_starts_wp = wp.array(inv_starts_np, dtype=wp.int32, device=env_init.device)
    inv_sizes_wp = wp.array(inv_sizes_np, dtype=wp.int32, device=env_init.device)
    inv_global_indices_wp = wp.array(inv_global_indices_np, dtype=wp.int32,
                                       device=env_init.device)

    if max_per_particle > 200:
        print(f"  ⚠️ max_per_particle={max_per_particle} > 200 hardcoded loop limit!")
        print(f"     gather_corrections_kernel will silently miss memberships.")
        return 1

    knife_pos_np = (0.85, 0.5, 0.12)
    v2e_baseline = {0: 0.964, 1: 90.4, 2: 4950, 3: 2.69e5, 6: 4.38e10}

    results = []
    for n in (0, 1, 2, 3, 6):
        r = run_at_num_iter(n, knife_pos_np, inv_starts_wp, inv_sizes_wp,
                              inv_global_indices_wp, total_memberships)
        results.append(r)

    print()
    print("=" * 78)
    print("  SUMMARY — SCATTER-GATHER")
    print("=" * 78)
    print(f"  {'num_iter':>10} {'v2k (scatter-gather)':>22} {'v2e (atomic_add)':>20}  improvement")
    print(f"  {'-'*10} {'-'*22} {'-'*20}  {'-'*15}")
    for r in results:
        n = r["num_iter"]
        v2e = v2e_baseline[n]
        improvement = v2e / max(r["ratio"], 1e-12)
        print(f"  {n:>10} {r['ratio']:>22.3e} {v2e:>20.3e}  {improvement:>10.2f}×")

    r6 = next(r for r in results if r["num_iter"] == 6)
    print()
    if r6["ratio"] < 100:
        print(f"  🎯 FORWARD MULTI-WRITER ATOMIC_ADD IS ROOT CAUSE")
        print(f"     num_iter=6 ratio: {v2e_baseline[6]:.0e} → {r6['ratio']:.2e}")
        print(f"     Path forward: scatter-gather refactor in production solver.")
        print(f"     Paper §3 contribution: 'first scatter-gather diff shape-matching PBD'")
    elif r6["ratio"] < v2e_baseline[6] / 10:
        print(f"  ⚠️ PARTIAL — multi-writer atomic_add forward contributes but not sole cause")
        print(f"     num_iter=6 ratio: {v2e_baseline[6]:.0e} → {r6['ratio']:.2e}")
        print(f"     Multi-writer adj_atomic_add in BACKWARD chain (auto-generated by Warp")
        print(f"     for adj_pred_pos) is likely additional bias source.")
    else:
        print(f"  ❌ FORWARD MULTI-WRITER ATOMIC_ADD IS NOT THE BUG")
        print(f"     num_iter=6 ratio: {v2e_baseline[6]:.0e} → {r6['ratio']:.2e}")
        print(f"     Bug is in BACKWARD adj propagation through cluster_solve's adjoint.")
        print(f"     Even with scatter-gather, Warp auto-generates multi-writer adj_atomic_add")
        print(f"     for adj_pred_pos[pid] across ~65 clusters, and that's likely the bug.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
