#!/usr/bin/env python3
"""Day 3 v2d — Full Newton idiom (all scratch buffers cycle-scoped per iter).

Verify whether porting Newton's _apply_particle_deltas pattern VERBATIM —
including fresh wp.zeros allocation for ALL accumulator buffers per Jacobi
iter — fixes the gradient bug.

Differences vs v2c:
  - v2c only cycle-scoped pred_pos (apply_shape_delta_ssa, apply_collision_delta_ssa)
  - v2d ALSO allocates fresh deltas, delta_counts, shape_delta, collision_delta
    per iter via wp.zeros (no clear_deltas_kernel, no in-place .zero_())
  - This matches Newton solver_xpbd.py:374, 380:
      particle_deltas = wp.zeros_like(particle_deltas) if requires_grad and i > 0
      else particle_deltas.zero_()

Hypothesis: scratch-buffer R-W-R was the residual bug. With ALL state and
accumulator buffers fresh per iter, backward should be value-correct.

PASS condition: num_iter=6 ratio < 10× (down from Test I's 5.3e9 baseline).
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


def differentiable_substep_full_newton_ssa(
    solver, params, knife_pos, dt_sub,
    enable_layer_a_collision: bool = True,
    enable_cutting: bool = True,
):
    """FULL Newton idiom: every accumulator buffer freshly allocated per iter.

    Buffers cycle-scoped per Jacobi iter:
      - pred_pos    (via wp.empty_like + apply_*_ssa output rebind)
      - deltas      (wp.zeros_like + cluster_solve writes)
      - delta_counts (wp.zeros_like, no requires_grad — int)
      - shape_delta (wp.zeros + apply_kernel_ssa atomic_add)
      - collision_delta (wp.zeros + collision_kernel_ssa atomic_add)
    """
    N = solver.N
    solver._alloc_ssa_buffers()

    knife_active_collision = (
        solver.knife_active if enable_layer_a_collision else 0
    )

    # ---- 1. Predict (no change) ----
    wp.launch(predict_kernel_ssa, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"],
        solver.p["pred_pos"], solver.p["old_pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.gravity, dt_sub, solver.damping,
    ], device=solver.device)

    # ---- 2. Initial collision: fresh collision_delta + cycle-scoped pred_pos ----
    coll_delta_pre = wp.zeros(N, dtype=wp.vec3, device=solver.device,
                                requires_grad=True)
    wp.launch(collision_kernel_ssa, dim=N, inputs=[
        solver.p["pred_pos"], solver.p["pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.floor_y, solver.floor_friction,
        knife_pos, solver.knife_radius, knife_active_collision,
        coll_delta_pre,
    ], device=solver.device)
    new_pred_pos = wp.empty_like(solver.p["pred_pos"])
    wp.launch(apply_collision_delta_ssa_kernel, dim=N,
              inputs=[solver.p["pred_pos"], coll_delta_pre],
              outputs=[new_pred_pos], device=solver.device)
    solver.p["pred_pos"] = new_pred_pos

    # ---- 3. Jacobi loop: ALL scratch buffers fresh per iter ----
    for k in range(solver.num_iterations):
        # Fresh deltas, delta_counts (Newton idiom: wp.zeros_like per iter)
        deltas_k = wp.zeros(N, dtype=wp.vec3, device=solver.device,
                              requires_grad=True)
        delta_counts_k = wp.zeros(N, dtype=wp.int32, device=solver.device)

        wp.launch(cluster_solve_kernel_ssa, dim=solver.num_clusters, inputs=[
            solver.p["pred_pos"], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            solver.c["cluster_quat"], solver.muller_iter,
            deltas_k, delta_counts_k,
        ], device=solver.device)

        # Fresh shape_delta
        shape_delta_k = wp.zeros(N, dtype=wp.vec3, device=solver.device,
                                   requires_grad=True)
        wp.launch(apply_kernel_ssa, dim=N, inputs=[
            deltas_k, delta_counts_k,
            solver.p["damage"], solver._current_alpha(),
            solver.p["fixed"], solver.stiffness,
            shape_delta_k,
        ], device=solver.device)

        # Apply shape_delta with cycle-scoped pred_pos
        new_pred_pos = wp.empty_like(solver.p["pred_pos"])
        wp.launch(apply_shape_delta_ssa_kernel, dim=N,
                  inputs=[solver.p["pred_pos"], shape_delta_k],
                  outputs=[new_pred_pos], device=solver.device)
        solver.p["pred_pos"] = new_pred_pos

        # Fresh collision_delta
        coll_delta_k = wp.zeros(N, dtype=wp.vec3, device=solver.device,
                                  requires_grad=True)
        wp.launch(collision_kernel_ssa, dim=N, inputs=[
            solver.p["pred_pos"], solver.p["pos"],
            solver._current_alpha(), solver.p["fixed"],
            solver.floor_y, solver.floor_friction,
            knife_pos, solver.knife_radius, knife_active_collision,
            coll_delta_k,
        ], device=solver.device)
        new_pred_pos = wp.empty_like(solver.p["pred_pos"])
        wp.launch(apply_collision_delta_ssa_kernel, dim=N,
                  inputs=[solver.p["pred_pos"], coll_delta_k],
                  outputs=[new_pred_pos], device=solver.device)
        solver.p["pred_pos"] = new_pred_pos

    # ---- 4. cluster_quat warm-start (out-of-tape state update) ----
    wp.launch(cluster_quat_update_kernel, dim=solver.num_clusters, inputs=[
        solver.p["pred_pos"], solver._current_alpha(),
        solver.c["starts"], solver.c["sizes"],
        solver.c["particle_ids"], solver.c["rest_offsets"],
        solver.c["cluster_quat"], solver.muller_iter,
    ], device=solver.device)

    # ---- 5. Cutting block skipped (enable_cutting=False) ----

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
        differentiable_substep_full_newton_ssa(env.solver, params, knife_pos, dt_sub,
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
            differentiable_substep_full_newton_ssa(env.solver, params_fd, kp, dt_sub,
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
    print("  Day 3 v2d — Full Newton idiom (ALL scratch buffers cycle-scoped)")
    print("  Setup: H1 (collision ON, cutting OFF, loss=mean pos.z)")
    print("  Helper: differentiable_substep_full_newton_ssa")
    print("=" * 78)

    knife_pos_np = (0.85, 0.5, 0.12)
    test_i_baseline = {0: 1.23, 1: 20.4, 2: 843, 3: 4.0e4, 6: 5.3e9}
    v2c_baseline = {0: 0.964, 1: 91.3, 2: 5216, 3: 2.93e5, 6: 5.30e10}

    results = []
    for n in (0, 1, 2, 3, 6):
        r = run_at_num_iter(n, knife_pos_np)
        results.append(r)

    print()
    print("=" * 78)
    print("  SUMMARY: Full Newton idiom (all buffers cycle-scoped)")
    print("=" * 78)
    print(f"  {'num_iter':>10} {'ratio_v2d':>12} {'v2c (pred_pos only)':>22} {'Test I (legacy)':>18} verdict")
    print(f"  {'-'*10} {'-'*12} {'-'*22} {'-'*18} {'-'*10}")
    for r in results:
        n = r["num_iter"]
        v2c = v2c_baseline[n]
        bl = test_i_baseline[n]
        ssa_thr = {0: 2, 1: 2, 2: 3, 3: 5, 6: 10}[n]
        verdict = ("✅ PASS" if r["ratio"] < ssa_thr else
                   "⚠️ IMPROVED" if r["ratio"] < bl else
                   "❌ FAIL (worse than baseline)")
        print(f"  {n:>10} {r['ratio']:>12.3e} {v2c:>22.3e} {bl:>18.2e}  {verdict}")

    r6 = next(r for r in results if r["num_iter"] == 6)
    if r6["ratio"] < 10:
        print()
        print(f"  ✅ Day 3 v2d PASS — Full Newton idiom verified.")
        print(f"     num_iter=6 ratio = {r6['ratio']:.2e} (<10× threshold).")
        print(f"     Path forward: port v2d pattern into production solver._substep.")
        return 0
    elif r6["ratio"] < 5.3e9:
        print()
        print(f"  ⚠️ Day 3 v2d IMPROVED but doesn't meet <10× threshold.")
        print(f"     num_iter=6 ratio = {r6['ratio']:.2e}.")
        print(f"     Direction is right — scratch buffer cycle-scoping helps but isn't sufficient.")
        print(f"     Consider deeper investigation (maybe Müller polar adjoint interaction).")
        return 1
    else:
        print()
        print(f"  ❌ Day 3 v2d FAIL — Full Newton idiom doesn't help either.")
        print(f"     num_iter=6 ratio = {r6['ratio']:.2e}, expected <10×.")
        print(f"     Implication: bug is somewhere else (Müller polar / cluster-quat / wp.atomic_add adjoint).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
