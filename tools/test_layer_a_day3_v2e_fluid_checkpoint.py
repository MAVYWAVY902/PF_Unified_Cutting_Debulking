#!/usr/bin/env python3
"""Day 3 v2e — fluid_checkpoint pattern (pre-allocated indexed buffer pool).

Eliminates Python dict attribute rebind. Pre-allocates pred_pos buffers
[0..2N+1] BEFORE wp.Tape context, passes indexed buffers directly to
kernel launches. Mathematically equivalent to v2d (Newton idiom) but
avoids any potential Warp tape interaction with dict rebind.

Reference: NVIDIA fluid_checkpoint.py lines 197+294-303 (50-iter Jacobi
pressure solver):
    for _iter in range(self.pressure_iterations):
        self.pressure_arrays.append(wp.zeros(...))
    ...
    for k in range(self.pressure_iterations):
        input_index = self.pressure_iterations * (step_index - 1) + k
        wp.launch(jacobi_iter,
                  inputs=[..., self.pressure_arrays[input_index]],
                  outputs=[self.pressure_arrays[input_index + 1]])

Buffer layout for num_iter=N:
    pred_pos[0] = predict output
    pred_pos[1] = after pre-loop collision
    pred_pos[2k+2] = iter k after apply_shape_delta_ssa  (for k=0..N-1)
    pred_pos[2k+3] = iter k after apply_collision_delta_ssa
    Final pred_pos = pred_pos[2N+1]

If v2e PASSES: bug was in dict rebind. Use this pattern in production.
If v2e FAILS:  bug is deeper (Müller adjoint, cluster_solve adjoint).
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
    cluster_solve_kernel_ssa, apply_kernel_ssa,
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


def differentiable_substep_fluid_checkpoint(
    solver, params, knife_pos, dt_sub,
    enable_layer_a_collision: bool = True,
    enable_cutting: bool = True,
):
    """fluid_checkpoint pattern: pre-allocated indexed buffer pool, no
    Python dict rebind. Mathematically equivalent to v2d but bypasses
    solver.p["pred_pos"] dict access in the iter loop.
    """
    N = solver.N
    solver._alloc_ssa_buffers()

    knife_active_collision = (
        solver.knife_active if enable_layer_a_collision else 0
    )
    num_iter = solver.num_iterations

    # ---- Pre-allocate ALL buffers up-front ----
    # pred_pos pool: 2*num_iter + 2 buffers
    # buf 0 = predict output
    # buf 1 = after pre-loop collision
    # buf (2k+2) = iter k after apply_shape_delta_ssa (k=0..N-1)
    # buf (2k+3) = iter k after apply_collision_delta_ssa
    # Final = buf (2*num_iter + 1)
    pred_pos_pool = [
        wp.empty(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
        for _ in range(2 * num_iter + 2)
    ]
    # Per-iter scratch buffer pools
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

    # ---- 1. Predict (writes to pred_pos_pool[0]) ----
    wp.launch(predict_kernel_ssa, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"],
        pred_pos_pool[0], solver.p["old_pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.gravity, dt_sub, solver.damping,
    ], device=solver.device)

    # ---- 2. Pre-loop collision: pool[0] -> pool[1] ----
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

    # ---- 3. Jacobi loop: pool[2k+1] -> pool[2k+2] -> pool[2k+3] ----
    for k in range(num_iter):
        in_idx = 2 * k + 1     # input: pre-iter pred_pos
        mid_idx = 2 * k + 2     # output of apply_shape_delta_ssa
        out_idx = 2 * k + 3     # output of apply_collision_delta_ssa

        # cluster_solve reads pool[in_idx]
        wp.launch(cluster_solve_kernel_ssa, dim=solver.num_clusters, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            solver.c["cluster_quat"], solver.muller_iter,
            deltas_pool[k], delta_counts_pool[k],
        ], device=solver.device)

        # apply_kernel writes shape_delta_pool[k]
        wp.launch(apply_kernel_ssa, dim=N, inputs=[
            deltas_pool[k], delta_counts_pool[k],
            solver.p["damage"], solver._current_alpha(),
            solver.p["fixed"], solver.stiffness,
            shape_delta_pool[k],
        ], device=solver.device)

        # apply_shape_delta_ssa: pool[in_idx] -> pool[mid_idx]
        wp.launch(apply_shape_delta_ssa_kernel, dim=N,
                  inputs=[pred_pos_pool[in_idx], shape_delta_pool[k]],
                  outputs=[pred_pos_pool[mid_idx]],
                  device=solver.device)

        # In-loop collision: pool[mid_idx] -> coll_delta_pool[k]
        wp.launch(collision_kernel_ssa, dim=N, inputs=[
            pred_pos_pool[mid_idx], solver.p["pos"],
            solver._current_alpha(), solver.p["fixed"],
            solver.floor_y, solver.floor_friction,
            knife_pos, solver.knife_radius, knife_active_collision,
            coll_delta_pool[k],
        ], device=solver.device)

        # apply_collision_delta_ssa: pool[mid_idx] -> pool[out_idx]
        wp.launch(apply_collision_delta_ssa_kernel, dim=N,
                  inputs=[pred_pos_pool[mid_idx], coll_delta_pool[k]],
                  outputs=[pred_pos_pool[out_idx]],
                  device=solver.device)

    # Final pred_pos
    final_pred_pos = pred_pos_pool[2 * num_iter + 1]

    # ---- 4. cluster_quat warm-start ----
    wp.launch(cluster_quat_update_kernel, dim=solver.num_clusters, inputs=[
        final_pred_pos, solver._current_alpha(),
        solver.c["starts"], solver.c["sizes"],
        solver.c["particle_ids"], solver.c["rest_offsets"],
        solver.c["cluster_quat"], solver.muller_iter,
    ], device=solver.device)

    # ---- 5. Cutting block skipped (enable_cutting=False) ----

    # ---- 6. Update velocity (reads final_pred_pos, writes pos, vel) ----
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
        differentiable_substep_fluid_checkpoint(env.solver, params, knife_pos, dt_sub,
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
            differentiable_substep_fluid_checkpoint(env.solver, params_fd, kp, dt_sub,
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
    print("  Day 3 v2e — fluid_checkpoint pattern (pre-alloc indexed buffer pool)")
    print("  Setup: H1 (collision ON, cutting OFF, loss=mean pos.z)")
    print("  Helper: differentiable_substep_fluid_checkpoint (NO dict rebind)")
    print("=" * 78)

    knife_pos_np = (0.85, 0.5, 0.12)
    test_i_baseline = {0: 1.23, 1: 20.4, 2: 843, 3: 4.0e4, 6: 5.3e9}
    v2d_baseline   = {0: 0.96, 1: 90.4, 2: 4948, 3: 2.69e5, 6: 4.38e10}

    results = []
    for n in (0, 1, 2, 3, 6):
        r = run_at_num_iter(n, knife_pos_np)
        results.append(r)

    print()
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  {'num_iter':>10} {'v2e (no rebind)':>18} {'v2d (rebind)':>15} {'Test I (legacy)':>18} verdict")
    print(f"  {'-'*10} {'-'*18} {'-'*15} {'-'*18} {'-'*10}")
    for r in results:
        n = r["num_iter"]
        v2d = v2d_baseline[n]
        bl = test_i_baseline[n]
        ssa_thr = {0: 2, 1: 2, 2: 3, 3: 5, 6: 10}[n]
        if r["ratio"] < ssa_thr:
            verdict = "✅ PASS"
        elif r["ratio"] < bl:
            verdict = "⚠️ IMPROVED"
        else:
            verdict = "❌ FAIL"
        print(f"  {n:>10} {r['ratio']:>18.3e} {v2d:>15.3e} {bl:>18.2e}  {verdict}")

    r6 = next(r for r in results if r["num_iter"] == 6)
    if r6["ratio"] < 10:
        print()
        print(f"  ✅ Day 3 v2e PASS — fluid_checkpoint pattern works.")
        print(f"     num_iter=6 ratio = {r6['ratio']:.2e} (<10× threshold).")
        print(f"     ROOT CAUSE: Python dict attribute rebind interferes with Warp tape.")
        print(f"     PATH FORWARD: rewrite production solver._substep with indexed buffer pool.")
        return 0
    elif r6["ratio"] < 5.3e9:
        print()
        print(f"  ⚠️ Day 3 v2e IMPROVED but not enough.")
        print(f"     num_iter=6 ratio = {r6['ratio']:.2e}.")
        print(f"     Direction is right — but need deeper fix (probably Müller polar).")
        return 1
    else:
        print()
        print(f"  ❌ Day 3 v2e FAIL — fluid_checkpoint pattern doesn't help either.")
        print(f"     num_iter=6 ratio = {r6['ratio']:.2e}.")
        print(f"     Implication: bug is NOT in dict rebind. It's deeper —")
        print(f"     likely Müller polar 8-iter quaternion adjoint, or cluster_solve")
        print(f"     adjoint specifically. Need design rethink.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
