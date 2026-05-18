#!/usr/bin/env python3
"""Day 3 v2c — SSA-ified Test I helper FD parity.

Verify whether our SSA fix IDIOM (cycle-scoped pred_pos via wp.empty_like +
attribute rebind) is correct, by porting it into the Test I helper
(differentiable_substep_isolated) instead of into production solver._substep.

If this PASSES (num_iter=6 ratio < 10×):
  → idiom is correct. Production solver's residual bug is from another
    interaction (likely scratch buffers in production: deltas, shape_delta,
    collision_delta — they have R-W-R compounding in production but the
    helper has the same pattern, so this should also fail there if that's it).

If this FAILS (num_iter=6 ratio still ~5e9 or worse):
  → idiom itself is buggy. Reconsider design (maybe explicit ssa cycle-index
    pool like Stage B instead of attribute rebind, or DiffXPBD adjoint route).

Compare to memory baseline: Test I unchanged shows num_iter=6 ratio ≈ 5.3e9.
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
    predict_kernel_ssa, collision_kernel_ssa, apply_collision_delta_kernel,
    clear_deltas_kernel, cluster_solve_kernel_ssa, apply_kernel_ssa,
    apply_shape_delta_kernel, cluster_quat_update_kernel,
    update_velocity_kernel,
    # New SSA kernels (Day 1 of Layer A fix)
    apply_shape_delta_ssa_kernel,
    apply_collision_delta_ssa_kernel,
)
from path4_falsification_test_H_isolate import (
    setup_env, loss_sum_pos_z_kernel,
)
from path4_poc_v1_dual_purpose import (
    make_param_dict, snap_solver_state, reset_solver_state,
)


def differentiable_substep_isolated_ssa(
    solver, params, knife_pos, dt_sub,
    enable_layer_a_collision: bool = True,
    enable_cutting: bool = True,
):
    """SSA variant of differentiable_substep_isolated.

    Replaces apply_shape_delta_kernel + apply_collision_delta_kernel
    (in-place atomic_add) with cycle-scoped wp.empty_like + rebind pattern,
    matching production solver._substep when use_ssa_soft_removal=True.

    Layer B cutting block unchanged (Stage B already SSA).
    """
    N = solver.N
    num_edges = solver.e["num_edges"] if solver.e is not None else 0
    solver._alloc_ssa_buffers()

    knife_active_collision = (
        solver.knife_active if enable_layer_a_collision else 0
    )

    # ---- 1. Predict ----
    wp.launch(predict_kernel_ssa, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"],
        solver.p["pred_pos"], solver.p["old_pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.gravity, dt_sub, solver.damping,
    ], device=solver.device)

    # ---- 2. Initial collision (with SSA dispatch) ----
    solver.p["collision_delta"].zero_()
    wp.launch(collision_kernel_ssa, dim=N, inputs=[
        solver.p["pred_pos"], solver.p["pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.floor_y, solver.floor_friction,
        knife_pos, solver.knife_radius, knife_active_collision,
        solver.p["collision_delta"],
    ], device=solver.device)
    # SSA: cycle-scoped buffer + rebind
    new_pred_pos = wp.empty_like(solver.p["pred_pos"])
    wp.launch(apply_collision_delta_ssa_kernel, dim=N,
              inputs=[solver.p["pred_pos"], solver.p["collision_delta"]],
              outputs=[new_pred_pos], device=solver.device)
    solver.p["pred_pos"] = new_pred_pos

    # ---- 3. Jacobi loop (with SSA dispatch on apply_shape_delta + apply_collision_delta) ----
    for _ in range(solver.num_iterations):
        wp.launch(clear_deltas_kernel, dim=N,
                  inputs=[solver.p["deltas"], solver.p["delta_counts"]],
                  device=solver.device)
        wp.launch(cluster_solve_kernel_ssa, dim=solver.num_clusters, inputs=[
            solver.p["pred_pos"], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            solver.c["cluster_quat"], solver.muller_iter,
            solver.p["deltas"], solver.p["delta_counts"],
        ], device=solver.device)

        solver.p["shape_delta"].zero_()
        wp.launch(apply_kernel_ssa, dim=N, inputs=[
            solver.p["deltas"], solver.p["delta_counts"],
            solver.p["damage"], solver._current_alpha(),
            solver.p["fixed"], solver.stiffness,
            solver.p["shape_delta"],
        ], device=solver.device)

        # SSA: apply_shape_delta with cycle-scoped buffer + rebind
        new_pred_pos = wp.empty_like(solver.p["pred_pos"])
        wp.launch(apply_shape_delta_ssa_kernel, dim=N,
                  inputs=[solver.p["pred_pos"], solver.p["shape_delta"]],
                  outputs=[new_pred_pos], device=solver.device)
        solver.p["pred_pos"] = new_pred_pos

        # In-loop collision
        solver.p["collision_delta"].zero_()
        wp.launch(collision_kernel_ssa, dim=N, inputs=[
            solver.p["pred_pos"], solver.p["pos"],
            solver._current_alpha(), solver.p["fixed"],
            solver.floor_y, solver.floor_friction,
            knife_pos, solver.knife_radius, knife_active_collision,
            solver.p["collision_delta"],
        ], device=solver.device)
        # SSA: apply_collision_delta with cycle-scoped buffer + rebind
        new_pred_pos = wp.empty_like(solver.p["pred_pos"])
        wp.launch(apply_collision_delta_ssa_kernel, dim=N,
                  inputs=[solver.p["pred_pos"], solver.p["collision_delta"]],
                  outputs=[new_pred_pos], device=solver.device)
        solver.p["pred_pos"] = new_pred_pos

    # ---- 4. cluster_quat warm-start (out-of-tape state update) ----
    wp.launch(cluster_quat_update_kernel, dim=solver.num_clusters, inputs=[
        solver.p["pred_pos"], solver._current_alpha(),
        solver.c["starts"], solver.c["sizes"],
        solver.c["particle_ids"], solver.c["rest_offsets"],
        solver.c["cluster_quat"], solver.muller_iter,
    ], device=solver.device)

    # ---- 5. Cutting block (skipped for H1 setup) ----
    # enable_cutting=False → no kernels launched

    # ---- 6. Update velocity ----
    inv_dt = 1.0 / dt_sub
    wp.launch(update_velocity_kernel, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"],
        solver.p["pred_pos"], solver.p["old_pos"],
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
        differentiable_substep_isolated_ssa(env.solver, params, knife_pos, dt_sub,
                                              enable_layer_a_collision=True,
                                              enable_cutting=False)
        wp.launch(loss_sum_pos_z_kernel, dim=env.solver.N,
                  inputs=[env.solver.p["pos"], interior_mask_wp,
                          interior_count, loss_buf],
                  device=env.device)
    tape.backward(loss=loss_buf)
    wp.synchronize()

    ag_kp = knife_pos.grad.numpy()[0].copy()

    # FD per-axis
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
            differentiable_substep_isolated_ssa(env.solver, params_fd, kp, dt_sub,
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
    print("=" * 72)
    print("  Day 3 v2c — SSA-ified Test I helper (cutting disabled, num_iter scan)")
    print("  Setup: H1 (collision ON, cutting OFF, loss=mean pos.z)")
    print("  Helper: differentiable_substep_isolated_ssa (with our Layer A SSA fix)")
    print("=" * 72)

    knife_pos_np = (0.85, 0.5, 0.12)

    results = []
    for n in (0, 1, 2, 3, 6):
        r = run_at_num_iter(n, knife_pos_np)
        results.append(r)

    print()
    print("=" * 72)
    print("  SUMMARY: SSA-ified helper, gradient magnitude scaling with num_iter")
    print("=" * 72)
    print(f"  {'num_iter':>10} {'‖autograd‖':>15} {'‖FD‖':>15} {'ratio':>15}  baseline_test_i  ssa_verdict")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}  {'-'*15}  {'-'*15}")
    test_i_baseline = {0: 1.23, 1: 20.4, 2: 843, 3: 4.0e4, 6: 5.3e9}
    for r in results:
        n = r["num_iter"]
        bl = test_i_baseline[n]
        ssa_thr = {0: 2, 1: 2, 2: 3, 3: 5, 6: 10}[n]
        verdict = ("✅ PASS" if r["ratio"] < ssa_thr else
                   "⚠️ SAME-ORDER" if r["ratio"] < bl * 2 else
                   "❌ FAIL")
        print(f"  {n:>10} {r['ag_mag']:>15.3e} {r['fd_mag']:>15.3e} "
              f"{r['ratio']:>15.3e}  {bl:>15.2e}  {verdict}")

    # Final verdict
    r6 = next(r for r in results if r["num_iter"] == 6)
    if r6["ratio"] < 10:
        print()
        print(f"  ✅ Day 3 v2c PASS — SSA idiom verified in helper.")
        print(f"     num_iter=6 ratio dropped {test_i_baseline[6]:.0e} → {r6['ratio']:.2e} (<10× threshold).")
        print(f"     Implication: production solver._substep has additional bugs beyond")
        print(f"     apply_shape_delta_kernel R-W-R that were masked by helper's bug.")
        return 0
    else:
        print()
        print(f"  ❌ Day 3 v2c FAIL — SSA idiom does NOT work even in helper.")
        print(f"     num_iter=6 ratio = {r6['ratio']:.2e}, expected <10×.")
        print(f"     Implication: idiom itself is buggy. Reconsider design.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
