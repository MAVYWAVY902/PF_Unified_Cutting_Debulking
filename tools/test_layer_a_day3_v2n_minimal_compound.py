#!/usr/bin/env python3
"""Day 3 v2n — Minimal A+B+C compound trigger pattern.

Tests whether the bug is triggered by THE MINIMAL COMPOUND of:
  A: cross-iter scalar accumulator (sum across loop iterations)
  B: cross-loop variable consumption (use accumulator-derived value in 2nd loop)
  C: affine subtraction (pred_pos[pid] - cross_loop_var)

Strips ALL other complexity from cluster_solve:
  - NO covariance matrix (no A_pq accumulator, just 1 scalar sum)
  - NO Müller polar
  - NO multi-component centroid (just scalar mean)
  - NO alpha-weighted masks
  - NO 4 sequential loops (just 2)

If v2n ratio ~50× per iter (~5e10 at num_iter=6):
    → Confirms minimal A+B+C compound IS the trigger
    → Production fix: avoid this pattern via loop fusion or cache pred_pos in local
    → Could be paper-worthy specific framework limit finding

If v2n ratio ~1× (clean like v2l/v2m):
    → A+B+C alone NOT enough — bug requires more (cov matrix or Müller)
    → My hypothesis WRONG yet again, need to keep isolating
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
# Minimal A+B+C compound cluster_solve
# ===========================================================================

@wp.kernel
def cluster_solve_minimal_abc(
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
    """Minimal A+B+C compound:
      A: sum_val = cross-iter scalar accumulator (built across loop 1)
      B: mean = cross-loop variable (used in loop 2)
      C: diff = pred_pos[pid] - mean (affine subtraction in loop 2)

    Forward: each particle gets correction = (pred_pos[pid] - mean) * 0.001
             where mean = average of cluster's pred_pos x-coords
    Backward: adj propagates back through both loops + cross-loop dep + subtraction
    """
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    # ─── A: cross-iter scalar accumulator (single float) ───
    sum_val = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        sum_val = sum_val + pred_pos[pid][0]   # accumulate x component

    # ─── B: cross-loop variable consumption ───
    mean = sum_val / float(size)

    # ─── C: affine subtraction in 2nd loop ───
    for j in range(size):
        pid = particle_ids[start + j]
        # affine: pred_pos[pid] - vec3(mean, mean, mean)
        diff = pred_pos[pid] - wp.vec3(mean, mean, mean)
        correction = diff * 0.001
        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, 1)


# ===========================================================================
# fluid_checkpoint helper
# ===========================================================================

def differentiable_substep_minimal_abc(
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

        wp.launch(cluster_solve_minimal_abc, dim=solver.num_clusters, inputs=[
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
        differentiable_substep_minimal_abc(env.solver, params, knife_pos, dt_sub,
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
            differentiable_substep_minimal_abc(env.solver, params_fd, kp, dt_sub,
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
    print("  Day 3 v2n — Minimal A+B+C compound trigger pattern test")
    print("  A: cross-iter scalar accumulator (sum_val)")
    print("  B: cross-loop variable (mean)")
    print("  C: affine subtraction (pred_pos[pid] - mean)")
    print("  NO covariance matrix, NO Müller, NO alpha mask, NO 4 loops")
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
    print("  SUMMARY — Minimal A+B+C")
    print("=" * 78)
    print(f"  {'num_iter':>10} {'v2n (A+B+C)':>15} {'v2l (trivial)':>15} {'v2e (full)':>15}")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
    for r in results:
        n = r["num_iter"]
        print(f"  {n:>10} {r['ratio']:>15.3e} {v2l_baseline[n]:>15.3e} {v2e_baseline[n]:>15.3e}")

    r6 = next(r for r in results if r["num_iter"] == 6)
    print()
    if r6["ratio"] > 1e8:
        print(f"  🎯 A+B+C MINIMAL COMPOUND IS THE TRIGGER")
        print(f"     v2n ratio: {r6['ratio']:.2e} (catastrophic, like v2e)")
        print(f"     Verified: cross-iter accumulator + cross-loop var + affine subtraction")
        print(f"     ALONE triggers the bug. No covariance, no Müller needed.")
        print(f"     Production fix: cache mean OUTSIDE tape, OR loop fusion.")
    elif r6["ratio"] > 1000:
        print(f"  ⚠️ A+B+C contributes but not sole — bug needs MORE")
        print(f"     v2n ratio: {r6['ratio']:.2e}")
    else:
        print(f"  ❌ A+B+C IS NOT ENOUGH — bug requires covariance/Müller chain")
        print(f"     v2n ratio: {r6['ratio']:.2e} (~1×, like v2l)")
        print(f"     My hypothesis WRONG (4th time today). Bug needs full")
        print(f"     centroid+covariance+Müller chain to trigger.")
        print(f"     Need v2o (add covariance) or v2p (add Müller) to isolate.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
