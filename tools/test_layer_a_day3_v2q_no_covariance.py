#!/usr/bin/env python3
"""Day 3 v2q — v2f WITHOUT covariance loop (keep rest centroid + R=identity goal).

Tests whether the covariance MATRIX accumulator (a00..a22 9-scalar
cross-product loop) is the trigger by REMOVING it from v2f's structure.

Forward physics changes minimally — covariance computed-but-not-used in v2f
becomes "covariance not even computed" in v2q. Forward output unchanged
(R=identity, so A doesn't enter goal anyway).

Mimics v2f's kernel structure:
  Loop 1: pred_pos centroid (alpha-weighted, with tw_safe guard)
  Loop 2: rest_offsets centroid (alpha-weighted)
  [SKIPPED] Loop 3: covariance A_pq
  Loop 4: goal = q_centered + centroid - pred_pos (multiplied by particle_alive)

Predictions:
  v2q ratio ~5e10 → covariance loop is NOT trigger; bug is in
                    centroid+rest_centroid+goal chain
  v2q ratio ~1× → covariance computed-but-unused IS trigger
                  (Warp adj-gen bug for dead code chain)
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
def cluster_solve_v2q(
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
    """Same as cluster_solve_kernel_no_muller (v2f) but WITHOUT covariance loop.

    Forward output identical to v2f (covariance was computed-but-unused due to
    R=identity bypass). Only structural change: 3 sequential loops instead of 4.
    """
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    # --- 1. Weighted centroid (alpha-weighted, mirrors v2e/v2f) ---
    cx_sum = float(0.0); cy_sum = float(0.0); cz_sum = float(0.0)
    tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        cx_sum = cx_sum + w * p[0]
        cy_sum = cy_sum + w * p[1]
        cz_sum = cz_sum + w * p[2]
        tw = tw + w

    tw_safe = wp.where(tw < 1.5, 1.0, tw)
    cx_mean = cx_sum / tw_safe
    cy_mean = cy_sum / tw_safe
    cz_mean = cz_sum / tw_safe

    # --- 2. Rest centroid (alpha-weighted, mirrors v2e/v2f) ---
    qcx_sum = float(0.0); qcy_sum = float(0.0); qcz_sum = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        q = rest_offsets[start + j]
        qcx_sum = qcx_sum + w * q[0]
        qcy_sum = qcy_sum + w * q[1]
        qcz_sum = qcz_sum + w * q[2]

    qcx_mean = qcx_sum / tw_safe
    qcy_mean = qcy_sum / tw_safe
    qcz_mean = qcz_sum / tw_safe

    # *** SKIPPED: covariance loop (compared to v2f) ***
    # No A_pq accumulators

    # --- 3. Goal positions (R=identity, mirrors v2f) ---
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        particle_alive = wp.where(w_raw < 0.01, 0.0, 1.0)
        q = rest_offsets[start + j]
        q_centered = wp.vec3(q[0] - qcx_mean, q[1] - qcy_mean, q[2] - qcz_mean)
        # R=identity → goal = q_centered + centroid (NO covariance, NO Müller)
        goal = q_centered + wp.vec3(cx_mean, cy_mean, cz_mean)
        correction = (goal - pred_pos[pid]) * particle_alive

        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, int(particle_alive))


def differentiable_substep_v2q(
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

        wp.launch(cluster_solve_v2q, dim=solver.num_clusters, inputs=[
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
        differentiable_substep_v2q(env.solver, params, knife_pos, dt_sub,
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
            differentiable_substep_v2q(env.solver, params_fd, kp, dt_sub,
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
    print("  Day 3 v2q — v2f WITHOUT covariance loop (keep rest centroid + R=identity)")
    print("=" * 78)

    knife_pos_np = (0.85, 0.5, 0.12)
    v2f_baseline = {0: 0.964, 1: 90.3, 2: 4980, 3: 2.72e5, 6: 4.53e10}
    v2p_baseline = {0: 0.963, 1: 0.993, 2: 0.991, 3: 0.993, 6: 0.994}
    v2e_baseline = {0: 0.964, 1: 90.4, 2: 4950, 3: 2.69e5, 6: 4.38e10}

    results = []
    for n in (0, 1, 2, 3, 6):
        r = run_at_num_iter(n, knife_pos_np)
        results.append(r)

    print()
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  {'num_iter':>10} {'v2q (no cov)':>15} {'v2f (with cov)':>16} {'v2p (no rest)':>15}")
    for r in results:
        n = r["num_iter"]
        print(f"  {n:>10} {r['ratio']:>15.3e} {v2f_baseline[n]:>16.3e} {v2p_baseline[n]:>15.3e}")

    r6 = next(r for r in results if r["num_iter"] == 6)
    print()
    if r6["ratio"] > 1e8:
        print(f"  🎯 REST CENTROID + GOAL CHAIN IS THE TRIGGER (covariance NOT needed)")
        print(f"     v2q ratio: {r6['ratio']:.2e}, similar to v2f's {v2f_baseline[6]:.2e}")
        print(f"     Bug: 2 cross-loop accumulator pairs (centroid + rest centroid)")
        print(f"     interacting in goal computation (q_centered + centroid - pred_pos)")
    elif r6["ratio"] > 1000:
        print(f"  ⚠️ Both covariance and rest centroid contribute")
    else:
        print(f"  🎯 COVARIANCE COMPUTED-BUT-UNUSED IS THE TRIGGER")
        print(f"     v2q ratio: {r6['ratio']:.2e} (~1×, removing covariance fixed it)")
        print(f"     This is a Warp adj-gen bug: dead code (computed-but-unused)")
        print(f"     covariance accumulator chain leaks adj into pred_pos via ")
        print(f"     auto-generated backward kernel, even when R=identity.")
        print(f"     Production fix: remove covariance computation entirely OR")
        print(f"     wrap it in @wp.func_replay with no-op adjoint.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
