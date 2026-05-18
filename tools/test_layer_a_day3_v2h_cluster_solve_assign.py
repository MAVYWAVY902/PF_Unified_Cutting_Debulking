#!/usr/bin/env python3
"""Day 3 v2h — cluster_solve atomic_add → assignment.

THE missing test. Replaces multi-writer atomic_add in cluster_solve_kernel_ssa
with plain assignment. Forward physics is BROKEN (last-cluster-wins race,
non-deterministic order) but FD and autograd both measure THIS broken physics,
so ratio is still informative about backward correctness.

Empirical hypothesis:
  - Each particle is in ~65.6 clusters (14886 clusters × avg size 65.6 / 14886 particles)
  - cluster_solve does atomic_add(deltas, pid, correction) per particle per cluster
  - If multi-writer atomic_add adjoint scales with #writers (N), 65^6 ≈ 7.5×10¹⁰
    closely matches observed 4.4×10¹⁰

Predictions:
  - ratio drops 10× or more → multi-writer atomic_add adjoint IS root cause
  - ratio essentially same as v2e (4.4×10¹⁰) → bug is elsewhere

Setup mirrors v2e fluid_checkpoint pattern (cleanest, no rebind).
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
# Replacement cluster_solve WITHOUT atomic_add (assignment for both deltas
# and delta_counts). Otherwise IDENTICAL to cluster_solve_kernel_ssa
# (Müller polar kept, alpha-weighted centroid kept).
# ===========================================================================

@wp.kernel
def cluster_solve_kernel_assign(
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
    """Same physics as cluster_solve_kernel_ssa BUT:
      - wp.atomic_add(deltas, pid, correction) → deltas[pid] = correction
      - wp.atomic_add(delta_counts, pid, ...) → delta_counts[pid] = ...
    With ~65 cluster writers per particle, this introduces RACE in forward
    (last cluster wins). Non-deterministic but FD/autograd both see same
    broken physics → ratio still meaningful."""
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    # --- 1. Weighted centroid ---
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

    # --- 2a. Rest centroid ---
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

    # --- 2b. Covariance A_pq ---
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
        a00 = a00 + alive * px * qx; a01 = a01 + alive * px * qy; a02 = a02 + alive * px * qz
        a10 = a10 + alive * py * qx; a11 = a11 + alive * py * qy; a12 = a12 + alive * py * qz
        a20 = a20 + alive * pz * qx; a21 = a21 + alive * pz * qy; a22 = a22 + alive * pz * qz

    A = wp.mat33(a00, a01, a02, a10, a11, a12, a20, a21, a22)

    # --- 3. Müller polar (unchanged) ---
    q_warm = cluster_quat[cid]
    q_new = extract_rotation_muller(A, q_warm, num_iter)
    R = wp.quat_to_matrix(q_new)

    # --- 4. Goal positions → ASSIGNMENT (was wp.atomic_add) ---
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        particle_alive = wp.where(w_raw < 0.01, 0.0, 1.0)
        q = rest_offsets[start + j]
        q_centered = wp.vec3(q[0] - qcx_mean, q[1] - qcy_mean, q[2] - qcz_mean)
        goal = R * q_centered + wp.vec3(cx_mean, cy_mean, cz_mean)
        correction = (goal - pred_pos[pid]) * particle_alive

        # ★ KEY DIFFERENCE: assignment instead of atomic_add ★
        deltas[pid] = correction
        delta_counts[pid] = int(particle_alive)


# ===========================================================================
# fluid_checkpoint helper using cluster_solve_kernel_assign
# ===========================================================================

def differentiable_substep_cluster_assign(
    solver, params, knife_pos, dt_sub,
    enable_layer_a_collision: bool = True,
    enable_cutting: bool = True,
):
    """Same as v2e fluid_checkpoint but uses cluster_solve_kernel_assign."""
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

        # *** USE cluster_solve_kernel_assign (no atomic_add) ***
        wp.launch(cluster_solve_kernel_assign, dim=solver.num_clusters, inputs=[
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
        differentiable_substep_cluster_assign(env.solver, params, knife_pos, dt_sub,
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
            differentiable_substep_cluster_assign(env.solver, params_fd, kp, dt_sub,
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
    print("  Day 3 v2h — cluster_solve atomic_add → assignment")
    print("  THE MISSING TEST. Tests #1: multi-writer atomic_add adjoint")
    print("  ⚠️ Forward physics broken (race) but FD vs autograd ratio still valid")
    print("=" * 78)

    knife_pos_np = (0.85, 0.5, 0.12)
    v2e_baseline = {0: 0.964, 1: 90.4, 2: 4950, 3: 2.69e5, 6: 4.38e10}

    results = []
    for n in (0, 1, 2, 3, 6):
        r = run_at_num_iter(n, knife_pos_np)
        results.append(r)

    print()
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  {'num_iter':>10} {'v2h (cs assign)':>17} {'v2e (cs atomic_add)':>20}  improvement")
    print(f"  {'-'*10} {'-'*17} {'-'*20}  {'-'*15}")
    for r in results:
        n = r["num_iter"]
        v2e = v2e_baseline[n]
        improvement = v2e / max(r["ratio"], 1e-12)
        print(f"  {n:>10} {r['ratio']:>17.3e} {v2e:>20.3e}  {improvement:>10.2f}×")

    r6 = next(r for r in results if r["num_iter"] == 6)
    print()
    if r6["ratio"] < 100:
        print(f"  🎯 HYPOTHESIS #1 CONFIRMED — multi-writer atomic_add IS root cause")
        print(f"     num_iter=6 ratio: {v2e_baseline[6]:.0e} → {r6['ratio']:.2e}")
        print(f"     ({v2e_baseline[6]/max(r6['ratio'],1e-12):.0e}× improvement)")
        print(f"     Path forward: scatter→gather rewrite (Fix #3 from agent research)")
        print(f"     OR: live with it (Layer A FD-only) until Warp ships proper multi-writer adj")
    elif r6["ratio"] < v2e_baseline[6] / 10:
        print(f"  ⚠️ PARTIAL — atomic_add contributes but isn't sole cause")
        print(f"     num_iter=6 ratio: {v2e_baseline[6]:.0e} → {r6['ratio']:.2e}")
        print(f"     Need to investigate other untested hypotheses (#2 dynamic-range loops, etc.)")
    else:
        print(f"  ❌ HYPOTHESIS #1 RULED OUT — multi-writer atomic_add is NOT the bug")
        print(f"     num_iter=6 ratio: {v2e_baseline[6]:.0e} → {r6['ratio']:.2e}")
        print(f"     Bug is in Warp 1.13 stock autodiff for our pattern, beyond atomic_add.")
        print(f"     Strategic decision: accept Layer A FD-only.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
