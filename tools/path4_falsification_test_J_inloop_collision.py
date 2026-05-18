#!/usr/bin/env python3
"""Path 4 Falsification Test J — disable in-loop collision specifically.

Test I showed Jacobi loop introduces ~40× amplification per iter on
∂L/∂knife_pos. This narrows further: WITHIN the Jacobi loop, 5 kernels
run. Which is the amplifier?

  J1: Initial collision ON, in-loop collision OFF, cluster_solve/apply ON
      → if amp gone → in-loop collision_kernel is the bug source
      → if amp still there → cluster_solve_kernel_ssa or apply_kernel_ssa or
        apply_shape_delta_kernel is the source

  J2: Initial collision ON, in-loop collision ON, apply_kernel stiffness=0
      → if amp gone → apply_kernel/apply_shape_delta is the source
      → if amp still there → cluster_solve_kernel_ssa is the source

(J2 may not work cleanly because cluster_solve still produces deltas, just
the apply_kernel scales them by stiffness=0 → shape_delta=0 → effectively
no-op. But cluster_solve still runs, so its backward path is still active.)
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
)
from path4_falsification_test_H_isolate import (
    loss_sum_pos_z_kernel, setup_env,
)
from path4_poc_v1_dual_purpose import (
    make_param_dict, snap_solver_state, reset_solver_state,
)


def differentiable_substep_J(
    solver, knife_pos, dt_sub,
    enable_initial_collision: bool = True,
    enable_in_loop_collision: bool = True,
    apply_stiffness: float = None,   # None = use solver.stiffness; 0.0 = no shape correction
):
    """Variant: separate initial vs in-loop collision; tunable stiffness."""
    N = solver.N
    solver._alloc_ssa_buffers()

    knife_active_initial = solver.knife_active if enable_initial_collision else 0
    knife_active_in_loop = solver.knife_active if enable_in_loop_collision else 0
    stiffness_use = solver.stiffness if apply_stiffness is None else apply_stiffness

    # Predict
    wp.launch(predict_kernel_ssa, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"],
        solver.p["pred_pos"], solver.p["old_pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.gravity, dt_sub, solver.damping,
    ], device=solver.device)

    # Initial collision
    solver.p["collision_delta"].zero_()
    wp.launch(collision_kernel_ssa, dim=N, inputs=[
        solver.p["pred_pos"], solver.p["pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.floor_y, solver.floor_friction,
        knife_pos, solver.knife_radius, knife_active_initial,
        solver.p["collision_delta"],
    ], device=solver.device)
    wp.launch(apply_collision_delta_kernel, dim=N,
              inputs=[solver.p["pred_pos"], solver.p["collision_delta"]],
              device=solver.device)

    # Jacobi loop
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
            solver.p["fixed"], stiffness_use,
            solver.p["shape_delta"],
        ], device=solver.device)
        wp.launch(apply_shape_delta_kernel, dim=N,
                  inputs=[solver.p["pred_pos"], solver.p["shape_delta"]],
                  device=solver.device)

        solver.p["collision_delta"].zero_()
        wp.launch(collision_kernel_ssa, dim=N, inputs=[
            solver.p["pred_pos"], solver.p["pos"],
            solver._current_alpha(), solver.p["fixed"],
            solver.floor_y, solver.floor_friction,
            knife_pos, solver.knife_radius, knife_active_in_loop,
            solver.p["collision_delta"],
        ], device=solver.device)
        wp.launch(apply_collision_delta_kernel, dim=N,
                  inputs=[solver.p["pred_pos"], solver.p["collision_delta"]],
                  device=solver.device)

    # cluster_quat
    wp.launch(cluster_quat_update_kernel, dim=solver.num_clusters, inputs=[
        solver.p["pred_pos"], solver._current_alpha(),
        solver.c["starts"], solver.c["sizes"],
        solver.c["particle_ids"], solver.c["rest_offsets"],
        solver.c["cluster_quat"], solver.muller_iter,
    ], device=solver.device)

    # Update velocity
    inv_dt = 1.0 / dt_sub
    wp.launch(update_velocity_kernel, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"],
        solver.p["pred_pos"], solver.p["old_pos"],
        solver._current_alpha(), solver.p["fixed"],
        inv_dt,
    ], device=solver.device)
    solver.step_count += 1


def evaluate(name, num_iter, **kwargs):
    print(f"\n  ─── {name}  num_iter={num_iter} ────")
    env = setup_env()
    env.solver.num_iterations = num_iter
    state_snap = snap_solver_state(env)
    knife_pos_np = (0.85, 0.5, 0.12)

    # Autograd
    knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                         device=env.device, requires_grad=True)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device,
                        requires_grad=True)
    dt_sub = env.solver.dt / float(env.solver.substeps)
    interior_count = float(env._total_interior)

    with wp.Tape() as tape:
        differentiable_substep_J(env.solver, knife_pos, dt_sub, **kwargs)
        wp.launch(loss_sum_pos_z_kernel, dim=env.solver.N,
                  inputs=[env.solver.p["pos"], env._interior_mask,
                          interior_count, loss_buf],
                  device=env.device)
    tape.backward(loss=loss_buf)
    wp.synchronize()
    ag_kp = knife_pos.grad.numpy()[0].copy()

    # FD
    fd_kp = np.zeros(3)
    for axis in range(3):
        eps = 5e-3
        for sign in (+1, -1):
            reset_solver_state(env, state_snap)
            env.solver.step_count = 0
            kp_np = list(knife_pos_np); kp_np[axis] += sign * eps
            kp = wp.array([wp.vec3(*kp_np)], dtype=wp.vec3, device=env.device)
            l_buf = wp.zeros(1, dtype=wp.float32, device=env.device)
            differentiable_substep_J(env.solver, kp, dt_sub, **kwargs)
            wp.launch(loss_sum_pos_z_kernel, dim=env.solver.N,
                      inputs=[env.solver.p["pos"], env._interior_mask,
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
    print(f"    autograd: ({ag_kp[0]:+.3e}, {ag_kp[1]:+.3e}, {ag_kp[2]:+.3e})")
    print(f"    FD:       ({fd_kp[0]:+.3e}, {fd_kp[1]:+.3e}, {fd_kp[2]:+.3e})")
    print(f"    ‖ag‖={ag_mag:.3e}  ‖FD‖={fd_mag:.3e}  ratio={ratio:.2e}×")
    return ratio


def main():
    wp.init()
    print("=" * 72)
    print("  Test J — narrow within Jacobi: in-loop collision vs cluster_solve")
    print("  H1 setup: knife on tumor, cutting OFF, loss=mean pos.z")
    print("=" * 72)

    # Baseline: full Layer A (matches Test I num_iter=6)
    r_baseline = evaluate("Baseline (full Layer A, num_iter=6)",
                          num_iter=6,
                          enable_initial_collision=True,
                          enable_in_loop_collision=True)

    # J1: disable in-loop collision (only initial), full Jacobi
    r_J1 = evaluate("J1 (only initial collision, NO in-loop, num_iter=6)",
                    num_iter=6,
                    enable_initial_collision=True,
                    enable_in_loop_collision=False)

    # J2: disable apply_kernel (stiffness=0)
    r_J2 = evaluate("J2 (full collision, stiffness=0 → no shape correction)",
                    num_iter=6,
                    enable_initial_collision=True,
                    enable_in_loop_collision=True,
                    apply_stiffness=0.0)

    # J3: disable BOTH in-loop collision AND stiffness (only cluster_solve runs)
    r_J3 = evaluate("J3 (NO in-loop collision, stiffness=0 → cluster_solve isolated)",
                    num_iter=6,
                    enable_initial_collision=True,
                    enable_in_loop_collision=False,
                    apply_stiffness=0.0)

    # J4: num_iter=0 (no Jacobi at all) for sanity baseline
    r_J4 = evaluate("J4 (num_iter=0, no Jacobi)",
                    num_iter=0,
                    enable_initial_collision=True,
                    enable_in_loop_collision=True)

    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  J4 (num_iter=0):                                 ratio={r_J4:.2e}×")
    print(f"  J3 (Jacobi: cluster_solve only, no apply, no col): ratio={r_J3:.2e}×")
    print(f"  J2 (Jacobi: cluster_solve + collision, no apply):  ratio={r_J2:.2e}×")
    print(f"  J1 (Jacobi: cluster_solve + apply, no col):       ratio={r_J1:.2e}×")
    print(f"  Baseline (full Layer A, num_iter=6):              ratio={r_baseline:.2e}×")
    print()
    print("INTERPRETATION:")
    if r_J3 < 5 and r_baseline > 1e3:
        print("  → cluster_solve_kernel_ssa ALONE doesn't blow up backward")
        if r_J1 > 1e3:
            print("  → in-loop apply_kernel + apply_shape_delta is amplifier")
        elif r_J2 > 1e3:
            print("  → in-loop collision_kernel + apply_collision_delta is amplifier")
        else:
            print("  → mixed; both in-loop kernels contribute")
    elif r_J3 > 1e3:
        print("  → cluster_solve_kernel_ssa ITSELF amplifies (Müller'16 SVD adjoint)")


if __name__ == "__main__":
    main()
