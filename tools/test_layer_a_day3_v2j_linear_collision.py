#!/usr/bin/env python3
"""Day 3 v2j — Linear collision kernel (isolate knife sphere wp.where backward).

Replaces the nonlinear knife-sphere collision in collision_kernel_ssa:
    diff = p_floor - kp
    dist_sq = wp.dot(diff, diff)        ← nonlinear (quadratic in kp)
    safe_dist = wp.sqrt(dist_sq + eps)   ← nonlinear (sqrt of quadratic)
    p_knife = kp + diff * (r/safe_dist)  ← multiplicative through 1/safe_dist
    p_new = wp.where(dist_sq < r², p_knife, p_floor)  ← discontinuous boundary

with a simple LINEAR pull:
    p_new = p_floor + (kp - p_floor) * 0.05

Forward physics is DIFFERENT (every particle pulled 5% toward knife, not
just particles inside sphere). But knife_pos gradient path is now LINEAR:
    ∂p_new/∂kp = 0.05 * I

PREDICTIONS:
  - ratio at num_iter=6 < 100×: knife sphere wp.where nonlinear backward IS
    the dominant root cause. Replacing with simpler differentiable contact
    (e.g., signed distance with smooth gating) would fix our gradient.
  - ratio still ~5×10¹⁰: bug is NOT in collision's nonlinearity. It's in
    the chain through cluster_solve's adjoint that propagates back to
    knife_pos via pred_pos modifications.

Either result is highly informative.
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
    predict_kernel_ssa,
    cluster_solve_kernel_ssa,
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
# Linear collision kernel — knife branch is LINEAR in knife_pos
# ===========================================================================

@wp.kernel
def collision_kernel_linear(
    pred_pos: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    floor_y: float,
    floor_friction: float,
    knife_pos: wp.array(dtype=wp.vec3),
    knife_radius: float,    # IGNORED in linear version
    knife_active: int,
    collision_delta: wp.array(dtype=wp.vec3),
):
    """Linear-in-knife_pos variant of collision_kernel_ssa.
    Floor branch unchanged. Knife branch replaced with LINEAR pull:
        p_new = p_floor + (kp - p_floor) * 0.05
    ∂p_new/∂kp = 0.05 * I (constant Jacobian — should be trivial for autograd).
    """
    i = wp.tid()
    if alpha[i] < 0.01 or fixed[i] == 1:
        return

    p0 = pred_pos[i]
    p_anchor = pos[i]

    # Floor branch (UNCHANGED — same as original)
    tx = p0[0] - p_anchor[0]
    ty = p0[1] - p_anchor[1]
    fx = p_anchor[0] + tx * (1.0 - floor_friction)
    fy = p_anchor[1] + ty * (1.0 - floor_friction)
    p_floor = wp.where(
        p0[2] < floor_y,
        wp.vec3(fx, fy, floor_y),
        p0,
    )

    # *** LINEAR knife branch — no sphere math, no wp.where boundary ***
    if knife_active == 1:
        kp = knife_pos[0]
        # Pull every particle 5% toward kp — purely LINEAR in kp
        # p_new = 0.95 * p_floor + 0.05 * kp
        p_new = p_floor + (kp - p_floor) * 0.05
    else:
        p_new = p_floor

    wp.atomic_add(collision_delta, i, p_new - p0)


# ===========================================================================
# fluid_checkpoint helper using collision_kernel_linear (everything else same)
# ===========================================================================

def differentiable_substep_linear_collision(
    solver, params, knife_pos, dt_sub,
    enable_layer_a_collision: bool = True,
    enable_cutting: bool = True,
):
    """v2e fluid_checkpoint pattern + collision_kernel_linear (for both
    pre-loop and in-loop collision). cluster_solve unchanged."""
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

    # *** LINEAR collision (pre-loop) ***
    wp.launch(collision_kernel_linear, dim=N, inputs=[
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

        # *** LINEAR collision (in-loop) ***
        wp.launch(collision_kernel_linear, dim=N, inputs=[
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
        differentiable_substep_linear_collision(env.solver, params, knife_pos, dt_sub,
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
            differentiable_substep_linear_collision(env.solver, params_fd, kp, dt_sub,
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
    print("  Day 3 v2j — LINEAR collision (isolate knife sphere wp.where backward)")
    print("  Knife branch: p_new = p_floor + (kp - p_floor) * 0.05  (LINEAR in kp)")
    print("  cluster_solve UNCHANGED")
    print("=" * 78)

    knife_pos_np = (0.85, 0.5, 0.12)
    v2e_baseline = {0: 0.964, 1: 90.4, 2: 4950, 3: 2.69e5, 6: 4.38e10}

    results = []
    for n in (0, 1, 2, 3, 6):
        r = run_at_num_iter(n, knife_pos_np)
        results.append(r)

    print()
    print("=" * 78)
    print("  SUMMARY — LINEAR collision")
    print("=" * 78)
    print(f"  {'num_iter':>10} {'v2j (linear coll)':>18} {'v2e (sphere coll)':>18}  improvement")
    print(f"  {'-'*10} {'-'*18} {'-'*18}  {'-'*15}")
    for r in results:
        n = r["num_iter"]
        v2e = v2e_baseline[n]
        improvement = v2e / max(r["ratio"], 1e-12)
        print(f"  {n:>10} {r['ratio']:>18.3e} {v2e:>18.3e}  {improvement:>10.2f}×")

    r6 = next(r for r in results if r["num_iter"] == 6)
    print()
    if r6["ratio"] < 100:
        print(f"  🎯 KNIFE SPHERE wp.where BACKWARD IS DOMINANT BUG")
        print(f"     num_iter=6 ratio: {v2e_baseline[6]:.0e} → {r6['ratio']:.2e}")
        print(f"     Path forward: replace sphere collision with smooth signed-distance")
        print(f"     contact (penalty form, smooth gating). Production fix possible.")
    elif r6["ratio"] < v2e_baseline[6] / 10:
        print(f"  ⚠️ PARTIAL — knife sphere contributes but not sole cause")
        print(f"     num_iter=6 ratio: {v2e_baseline[6]:.0e} → {r6['ratio']:.2e}")
    else:
        print(f"  ❌ KNIFE SPHERE BACKWARD IS NOT THE BUG")
        print(f"     num_iter=6 ratio: {v2e_baseline[6]:.0e} → {r6['ratio']:.2e}")
        print(f"     Bug is in chain through cluster_solve adjoint, not collision.")
        print(f"     Multi-writer atomic_add proper test (scatter-gather) remains untested.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
