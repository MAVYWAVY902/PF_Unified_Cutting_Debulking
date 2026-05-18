#!/usr/bin/env python3
"""Day 3 v2g — apply_kernel: atomic_add → assignment.

Tests Hypothesis #2: Warp 1.12's atomic_add adjoint has a bias for
single-writer cells (where atomic_add semantically equals assignment).

apply_kernel_ssa launches with dim=N (one thread per particle), and each
thread writes shape_delta[tid] only — NO contention. So:
    wp.atomic_add(shape_delta, i, correction)  ≡  shape_delta[i] = correction

forward-wise. But Warp's auto-derived BACKWARD might differ:
  - atomic_add adjoint: reads adj_shape_delta[i] (atomic load), accumulates
    into adj_correction → adj_deltas → ...
  - assignment adjoint: reads adj_shape_delta[i], assigns adj_correction
    directly (no atomic accumulation logic).

If ratio drops → the atomic_add adjoint path has bias (Hypothesis #2)
If ratio unchanged → bug is upstream in cluster_solve (Hypothesis #1)

Setup mirrors v2e (fluid_checkpoint pattern, no Müller polar bypass).
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
    cluster_solve_kernel_ssa,
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
# Replacement apply_kernel WITHOUT atomic_add (pure assignment)
# ===========================================================================

@wp.kernel
def apply_kernel_assign(
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    stiffness: float,
    shape_delta: wp.array(dtype=wp.vec3),  # OUTPUT
):
    """Like apply_kernel_ssa but uses assignment instead of atomic_add.
    Forward equivalent (single thread per cell). Backward may differ —
    that's exactly what we're testing.

    Caller must zero-initialize shape_delta before launch.
    """
    i = wp.tid()
    if alpha[i] < 0.01 or fixed[i] == 1 or delta_counts[i] == 0:
        return

    dmg = damage[i]
    local_stiff = stiffness * (1.0 - dmg * 0.95)
    inv_count = 1.0 / wp.float32(delta_counts[i])
    correction = local_stiff * inv_count * deltas[i]

    # ASSIGNMENT instead of wp.atomic_add
    shape_delta[i] = correction


# ===========================================================================
# Helper — fluid_checkpoint pattern with apply_kernel_assign
# ===========================================================================

def differentiable_substep_assign(
    solver, params, knife_pos, dt_sub,
    enable_layer_a_collision: bool = True,
    enable_cutting: bool = True,
):
    """Same as v2e fluid_checkpoint but uses apply_kernel_assign (no atomic_add)."""
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

        wp.launch(cluster_solve_kernel_ssa, dim=solver.num_clusters, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            solver.c["cluster_quat"], solver.muller_iter,
            deltas_pool[k], delta_counts_pool[k],
        ], device=solver.device)

        # *** ASSIGNMENT VARIANT ***
        wp.launch(apply_kernel_assign, dim=N, inputs=[
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
        differentiable_substep_assign(env.solver, params, knife_pos, dt_sub,
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
            differentiable_substep_assign(env.solver, params_fd, kp, dt_sub,
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
    print("  Day 3 v2g — apply_kernel: atomic_add → assignment isolation")
    print("  Tests #2: atomic_add adjoint bias on single-writer cell")
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
    print(f"  {'num_iter':>10} {'v2g (assign)':>15} {'v2e (atomic_add)':>20}  improvement")
    print(f"  {'-'*10} {'-'*15} {'-'*20}  {'-'*15}")
    for r in results:
        n = r["num_iter"]
        v2e = v2e_baseline[n]
        improvement = v2e / max(r["ratio"], 1e-12)
        print(f"  {n:>10} {r['ratio']:>15.3e} {v2e:>20.3e}  {improvement:>10.2f}×")

    r6 = next(r for r in results if r["num_iter"] == 6)
    print()
    if r6["ratio"] < 100:
        print(f"  ✅ HYPOTHESIS #2 CONFIRMED — apply_kernel atomic_add adjoint IS the bug")
        print(f"     num_iter=6 ratio: {v2e_baseline[6]:.0e} → {r6['ratio']:.2e}")
        print(f"     Path forward: replace atomic_add with assignment in apply_kernel.")
    elif r6["ratio"] < v2e_baseline[6] / 10:
        print(f"  ⚠️ PARTIAL — apply_kernel atomic_add contributes but isn't sole cause")
        print(f"     num_iter=6 ratio: {v2e_baseline[6]:.0e} → {r6['ratio']:.2e}")
        print(f"     Move on to test #1 (cluster_solve atomic_add).")
    else:
        print(f"  ❌ HYPOTHESIS #2 RULED OUT — apply_kernel atomic_add is NOT the bug")
        print(f"     num_iter=6 ratio: {v2e_baseline[6]:.0e} → {r6['ratio']:.2e} (essentially same)")
        print(f"     Proceed to test #1 (cluster_solve atomic_add into deltas).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
