#!/usr/bin/env python3
"""Day 3 v2m — Multi-read same pred_pos same thread (Issue #717 falsifier).

Tests Warp Issue #717: "Multiple Copies of an array breaks the computational graph"
(open, milestone Warp 1.14, gradients scaled by 1/N when array read N times).

EXACT delta vs v2l (which gave ~1× ratio):
    v2l: ONE read of pred_pos[pid] per cluster thread per j
    v2m: THREE reads of pred_pos[pid] per cluster thread per j (same pid)

Forward math is mathematically equivalent (correction = pred_pos[pid] * 0.001).
But backward: v2m generates 3 adj_atomic_add calls into adj_pred_pos[pid] per
cluster thread (vs v2l's 1).

If v2m ratio ~5×10¹⁰ (catastrophic like v2e):
    → Issue #717 (multi-read-same-element-same-thread) IS our root cause
    → Fix: cache pred_pos[pid] in local var, read once per cluster thread per particle
    → Production fix is loop fusion (~half day)

If v2m ratio ~1× (clean like v2l):
    → Multi-read alone is NOT the bug
    → Bug is in cross-loop accumulator (centroid/cov) or Müller chain
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
# Multi-read trivial cluster_solve — same forward as v2l, 3× backward adj
# ===========================================================================

@wp.kernel
def cluster_solve_multi_read(
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
    """Trivial linear math (same forward as v2l) but each cluster thread reads
    pred_pos[pid] THREE TIMES per j (mimicking v2e's 3 reads from loops 1/3/4
    but without the centroid/cov/Müller chain complexity).

    Forward: correction = (p1+p2+p3)*0.001/3.0 = pred_pos[pid]*0.001 (same as v2l)
    Backward: 3 adj_atomic_add(adj_pred_pos, pid, ...) per cluster thread per j
              vs v2l's 1 per cluster thread per j
    """
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    for j in range(size):
        pid = particle_ids[start + j]
        # *** THREE reads of SAME pred_pos[pid] in SAME thread ***
        p1 = pred_pos[pid]   # READ 1
        p2 = pred_pos[pid]   # READ 2 — same pid, same thread
        p3 = pred_pos[pid]   # READ 3 — same pid, same thread
        # Forward math: 3*pred_pos/3 = pred_pos (* 0.001 scaling)
        correction = (p1 + p2 + p3) * (0.001 / 3.0)
        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, 1)


# ===========================================================================
# fluid_checkpoint helper
# ===========================================================================

def differentiable_substep_multi_read(
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

        wp.launch(cluster_solve_multi_read, dim=solver.num_clusters, inputs=[
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
        differentiable_substep_multi_read(env.solver, params, knife_pos, dt_sub,
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
            differentiable_substep_multi_read(env.solver, params_fd, kp, dt_sub,
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
    print("  Day 3 v2m — Multi-read SAME pred_pos SAME thread (Issue #717 falsifier)")
    print("  3 reads of pred_pos[pid] per cluster thread per j (mimics v2e adj pattern)")
    print("  Forward math = v2l (correction = pred_pos * 0.001)")
    print("=" * 78)

    knife_pos_np = (0.85, 0.5, 0.12)
    v2l_baseline = {0: 0.964, 1: 0.994, 2: 0.993, 3: 0.993, 6: 0.983}
    v2e_baseline = {0: 0.964, 1: 90.4, 2: 4950, 3: 2.69e5, 6: 4.38e10}

    results = []
    for n in (0, 1, 2, 3, 6):
        r = run_at_num_iter(n, knife_pos_np)
        results.append(r)

    print()
    print("=" * 78)
    print("  SUMMARY — Multi-read same pred_pos")
    print("=" * 78)
    print(f"  {'num_iter':>10} {'v2m (3-read)':>17} {'v2l (1-read)':>15} {'v2e (full math)':>17}")
    print(f"  {'-'*10} {'-'*17} {'-'*15} {'-'*17}")
    for r in results:
        n = r["num_iter"]
        v2l = v2l_baseline[n]
        v2e = v2e_baseline[n]
        print(f"  {n:>10} {r['ratio']:>17.3e} {v2l:>15.3e} {v2e:>17.3e}")

    r6 = next(r for r in results if r["num_iter"] == 6)
    print()
    if r6["ratio"] > 1e8:
        print(f"  🎯 ISSUE #717 CONFIRMED (multi-read same pred_pos same thread IS the bug)")
        print(f"     v2m num_iter=6 ratio: {r6['ratio']:.2e} — explosive like v2e")
        print(f"     Fix path: cache pred_pos[pid] in local var, read ONCE per cluster thread.")
        print(f"     Loop fusion: combine 3 cluster_solve loops into 1 with single pred_pos read.")
        print(f"     Production fix estimate: ~half day.")
    elif r6["ratio"] > 100:
        print(f"  ⚠️ PARTIAL — multi-read contributes but isn't sole cause")
        print(f"     v2m: {r6['ratio']:.2e}, v2e: {v2e_baseline[6]:.2e}")
        print(f"     Multi-read adds bias but cross-loop accumulator pattern adds more.")
    else:
        print(f"  ❌ ISSUE #717 RULED OUT — multi-read same pred_pos NOT the bug")
        print(f"     v2m: {r6['ratio']:.2e} (~1×, like v2l)")
        print(f"     Bug requires SPECIFIC math chain (centroid/cov/Müller cross-loop).")
        print(f"     Issue #717 doesn't explain our pattern despite surface match.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
