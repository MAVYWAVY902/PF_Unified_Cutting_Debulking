#!/usr/bin/env python3
"""Day 3 v2l — Trivial linear cluster_solve (TRUE multi-writer adj isolation).

This is the FINAL isolation test. It strips ALL complexity from cluster_solve
EXCEPT the multi-reader/multi-writer pattern itself:

    @wp.kernel
    def cluster_solve_trivial(...):
        cid = wp.tid()
        for j in range(size):
            pid = particle_ids[start + j]
            correction = pred_pos[pid] * 0.001    # ← TRIVIAL LINEAR
            wp.atomic_add(deltas, pid, correction)
            wp.atomic_add(delta_counts, pid, 1)

What's KEPT:
  - Multi-reader pred_pos[pid] (~65 cluster threads read same pid)
  - Multi-writer atomic_add into deltas (~65/cell)
  - Auto-generated multi-writer adj_atomic_add into adj_pred_pos in BACKWARD

What's REMOVED:
  - Centroid computation (3 loops)
  - Covariance matrix (1 loop)
  - Müller polar (8 inner iters)
  - Cross-loop accumulators (cx_mean, qcx_mean, A used across loops)
  - Cluster_quat warm-start

PREDICTIONS:
  - num_iter=6 ratio still ~50× per iter (>1e9): multi-reader → multi-writer
    adj_atomic_add IS the structural bug. Confirmed framework limit.
  - num_iter=6 ratio ~1×: bug requires specific cluster_solve math
    (centroid/covariance/Müller chain). My hypothesis is WRONG.
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


# ===========================================================================
# Trivial linear cluster_solve — strips ALL complexity except multi-reader pattern
# ===========================================================================

@wp.kernel
def cluster_solve_trivial(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),         # IGNORED (no alive masking)
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),     # IGNORED
    cluster_quat: wp.array(dtype=wp.quat),     # IGNORED
    num_iter: int,                              # IGNORED
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
):
    """TRIVIAL LINEAR: correction = pred_pos[pid] * 0.001, atomic_add into deltas.

    Forward physics:
      - Each cluster reads pred_pos[pid] for ~65 particles
      - Each particle gets ~65 atomic_adds of 0.001*pred_pos[pid] → deltas[pid] = 0.065 * pred_pos[pid]
      - apply_kernel + apply_shape_delta_ssa → pred_pos *= ~1.00085 per Jacobi iter
      - Stable, monotonic scaling

    Backward (Warp auto-generated):
      - adj_correction = adj_deltas[pid] (multi-reader: ~65 cluster threads read same)
      - ∂correction/∂pred_pos[pid] = 0.001 * I (TRIVIAL LINEAR)
      - adj_pred_pos[pid] += 0.001 * adj_correction (multi-writer atomic_add: ~65/cell)
      - This is the ONLY pattern remaining — no Müller, no centroid, no covariance
    """
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    for j in range(size):
        pid = particle_ids[start + j]
        # *** TRIVIAL LINEAR — direct multi-reader pred_pos → multi-writer adj_pred_pos in backward ***
        correction = pred_pos[pid] * 0.001
        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, 1)


# ===========================================================================
# fluid_checkpoint helper using cluster_solve_trivial
# ===========================================================================

def differentiable_substep_trivial_cluster(
    solver, params, knife_pos, dt_sub,
    enable_layer_a_collision: bool = True,
    enable_cutting: bool = True,
):
    """v2e fluid_checkpoint structure but cluster_solve replaced by trivial linear."""
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

        # *** TRIVIAL LINEAR cluster_solve ***
        wp.launch(cluster_solve_trivial, dim=solver.num_clusters, inputs=[
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
        differentiable_substep_trivial_cluster(env.solver, params, knife_pos, dt_sub,
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
            differentiable_substep_trivial_cluster(env.solver, params_fd, kp, dt_sub,
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
    print("  Day 3 v2l — TRIVIAL LINEAR cluster_solve (multi-writer adj isolation)")
    print("  Stripped: centroid, covariance, Müller polar, cross-loop accumulators")
    print("  Kept:     multi-reader pred_pos → multi-writer adj backward")
    print("=" * 78)

    knife_pos_np = (0.85, 0.5, 0.12)
    v2e_baseline = {0: 0.964, 1: 90.4, 2: 4950, 3: 2.69e5, 6: 4.38e10}

    results = []
    for n in (0, 1, 2, 3, 6):
        r = run_at_num_iter(n, knife_pos_np)
        results.append(r)

    print()
    print("=" * 78)
    print("  SUMMARY — TRIVIAL LINEAR cluster_solve")
    print("=" * 78)
    print(f"  {'num_iter':>10} {'v2l (trivial)':>18} {'v2e (full math)':>17}  v2e/v2l")
    print(f"  {'-'*10} {'-'*18} {'-'*17}  {'-'*10}")
    for r in results:
        n = r["num_iter"]
        v2e = v2e_baseline[n]
        ratio_diff = v2e / max(r["ratio"], 1e-12)
        print(f"  {n:>10} {r['ratio']:>18.3e} {v2e:>17.3e}  {ratio_diff:>10.2f}×")

    r6 = next(r for r in results if r["num_iter"] == 6)
    r1 = next(r for r in results if r["num_iter"] == 1)
    print()
    if r6["ratio"] > 1e9 and r1["ratio"] > 10:
        print(f"  🎯 MULTI-READER → MULTI-WRITER ADJ_ATOMIC_ADD IS ROOT CAUSE")
        print(f"     num_iter=1 ratio: {r1['ratio']:.2e}  (~50× amp from single Jacobi iter)")
        print(f"     num_iter=6 ratio: {r6['ratio']:.2e}  (compounds 50^6 ≈ 1.5e10)")
        print(f"     Stripping ALL cluster_solve math (centroid/cov/Müller) → ratio")
        print(f"     STAYS HIGH. Confirms structural Warp 1.13 limit on multi-reader")
        print(f"     forward → multi-writer adj backward chains compounded over Jacobi.")
    elif r6["ratio"] < 100:
        print(f"  🎯 cluster_solve's SPECIFIC MATH IS THE BUG (not multi-writer adj)")
        print(f"     num_iter=6 ratio: {r6['ratio']:.2e}  (vs v2e {v2e_baseline[6]:.0e})")
        print(f"     Trivial linear gives ~1× → bug requires centroid/cov/Müller chain.")
        print(f"     My hypothesis was WRONG. Bug is in specific shape-matching adj.")
    else:
        print(f"  ⚠️ INTERMEDIATE — multi-writer adj contributes but not sole cause")
        print(f"     num_iter=6 ratio: {r6['ratio']:.2e}")
        print(f"     Cluster_solve's math AND multi-writer adj BOTH contribute.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
