#!/usr/bin/env python3
"""Path 4 Falsification Test H — isolate Layer A vs Layer B path for knife_pos.

THE CORE QUESTION: knife_pos has TWO backward paths:
  Layer A:  loss → pos → ... → pred_pos ← collision_delta ← knife_pos
  Layer B:  loss → alpha → ... → edge_damage ← knife_pos[0] (read directly)

Test E showed knife_pos catastrophic 10^9× off when both paths active.
We need to isolate which path is the bug source.

H1 — Layer A only:
  - enable_layer_a_collision = True   (collision fires)
  - enable_cutting = False             (cutting block skipped → Layer B path dead)
  - Loss = sum(pos[interior].z)        (position-dependent so Layer A signal exists)
  - knife_pos's only path to loss = via collision → pred_pos → final pos

H2 — Layer B only:
  - enable_layer_a_collision = False   (collision_kernel runs but knife_active=0
                                          inside it — sees only floor)
  - enable_cutting = True              (cutting block fires)
  - Loss = mean(alpha[interior])       (alpha-dependent)
  - knife_pos's only path to loss = via update_edge_damage's `kp = knife_pos[0]`
  - Note: collision still runs floor projection (needed for stable physics) but
    skips the knife wp.where branch.

PREDICTIONS:
  H1 ≈ FD AND H2 ≈ FD  → both paths OK; bug is somewhere else
  H1 ≠ FD AND H2 ≈ FD  → ✓ Layer A is the bug
  H1 ≈ FD AND H2 ≠ FD  → Layer B is the bug
  H1 ≠ FD AND H2 ≠ FD  → both have bugs (worse situation)
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
from pf_shape_matching_warp.sim.pf_damage import (
    update_edge_damage_kernel_ssa_param_arr, apply_edge_damage_ssa_kernel,
    update_particle_damage_kernel, finalize_particle_damage_kernel_ssa,
    apply_particle_damage_ssa_kernel,
    update_soft_removal_kernel_ssa_param_arr,
)
from path4_poc_v1_dual_purpose import (
    make_param_dict, snap_solver_state, reset_solver_state,
)
from warp_sim.envs.debulk_env import DebulkEnv


# =============================================================================
# Substep variant with controllable enable flags
# =============================================================================

def differentiable_substep_isolated(
    solver, params, knife_pos, dt_sub,
    enable_layer_a_collision: bool = True,
    enable_cutting: bool = True,
):
    N = solver.N
    num_edges = solver.e["num_edges"] if solver.e is not None else 0
    solver._alloc_ssa_buffers()

    # Knife is on/off for Layer A collision branch ONLY
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

    # ---- 2. Initial collision ----
    solver.p["collision_delta"].zero_()
    wp.launch(collision_kernel_ssa, dim=N, inputs=[
        solver.p["pred_pos"], solver.p["pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.floor_y, solver.floor_friction,
        knife_pos, solver.knife_radius, knife_active_collision,
        solver.p["collision_delta"],
    ], device=solver.device)
    wp.launch(apply_collision_delta_kernel, dim=N,
              inputs=[solver.p["pred_pos"], solver.p["collision_delta"]],
              device=solver.device)

    # ---- 3. Jacobi loop ----
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
        wp.launch(apply_shape_delta_kernel, dim=N,
                  inputs=[solver.p["pred_pos"], solver.p["shape_delta"]],
                  device=solver.device)

        solver.p["collision_delta"].zero_()
        wp.launch(collision_kernel_ssa, dim=N, inputs=[
            solver.p["pred_pos"], solver.p["pos"],
            solver._current_alpha(), solver.p["fixed"],
            solver.floor_y, solver.floor_friction,
            knife_pos, solver.knife_radius, knife_active_collision,
            solver.p["collision_delta"],
        ], device=solver.device)
        wp.launch(apply_collision_delta_kernel, dim=N,
                  inputs=[solver.p["pred_pos"], solver.p["collision_delta"]],
                  device=solver.device)

    # ---- 4. cluster_quat warm start ----
    wp.launch(cluster_quat_update_kernel, dim=solver.num_clusters, inputs=[
        solver.p["pred_pos"], solver._current_alpha(),
        solver.c["starts"], solver.c["sizes"],
        solver.c["particle_ids"], solver.c["rest_offsets"],
        solver.c["cluster_quat"], solver.muller_iter,
    ], device=solver.device)

    # ---- 5. Cutting block (controllable) ----
    if (enable_cutting and solver.knife_active == 1 and solver.e is not None
            and solver.step_count % solver.cutting_update_interval == 0):
        c = solver._ssa_cycle_index
        if c + 1 >= solver.ssa_buffer_capacity:
            raise RuntimeError("SSA capacity exhausted")

        alpha_prev = solver._ssa_alpha_buffers[c]
        alpha_next = solver._ssa_alpha_buffers[c + 1]
        ed_prev = solver._ssa_edge_damage_buffers[c]
        ed_next = solver._ssa_edge_damage_buffers[c + 1]
        pd_prev = solver._ssa_particle_damage_buffers[c]
        pd_next = solver._ssa_particle_damage_buffers[c + 1]

        solver.e["damage_delta"].zero_()
        wp.launch(update_edge_damage_kernel_ssa_param_arr, dim=num_edges,
                  inputs=[
                      solver.p["pred_pos"], alpha_prev,
                      solver.e["a"], solver.e["b"], solver.e["pf_rest"],
                      ed_prev,
                      knife_pos, params["knife_radius"], params["alpha_min"],
                      solver.e["damage_delta"],
                  ], device=solver.device)
        wp.launch(apply_edge_damage_ssa_kernel, dim=num_edges,
                  inputs=[ed_prev, solver.e["damage_delta"], ed_next],
                  device=solver.device)

        damage_sum = wp.zeros(N, dtype=wp.float32, device=solver.device,
                              requires_grad=True)
        damage_count = wp.zeros(N, dtype=wp.float32, device=solver.device,
                                requires_grad=True)
        wp.launch(update_particle_damage_kernel, dim=num_edges, inputs=[
            damage_sum, damage_count, alpha_prev,
            solver.e["a"], solver.e["b"], ed_next,
        ], device=solver.device)
        solver.p["particle_damage_delta"].zero_()
        wp.launch(finalize_particle_damage_kernel_ssa, dim=N,
                  inputs=[damage_sum, damage_count, pd_prev,
                          solver.p["particle_damage_delta"]],
                  device=solver.device)
        wp.launch(apply_particle_damage_ssa_kernel, dim=N,
                  inputs=[pd_prev, solver.p["particle_damage_delta"], pd_next],
                  device=solver.device)

        wp.launch(update_soft_removal_kernel_ssa_param_arr, dim=N, inputs=[
            pd_next, alpha_prev, solver.p["fixed"],
            params["threshold"], params["sharpness"],
            alpha_next,
        ], device=solver.device)
        solver._ssa_cycle_index = c + 1

    # ---- 6. Update velocity ----
    inv_dt = 1.0 / dt_sub
    wp.launch(update_velocity_kernel, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"],
        solver.p["pred_pos"], solver.p["old_pos"],
        solver._current_alpha(), solver.p["fixed"],
        inv_dt,
    ], device=solver.device)
    solver.step_count += 1


# =============================================================================
# Loss kernels
# =============================================================================

@wp.kernel
def loss_sum_pos_z_kernel(
    pos: wp.array(dtype=wp.vec3),
    interior_mask: wp.array(dtype=wp.int32),
    interior_count: float,
    loss_buf: wp.array(dtype=wp.float32),
):
    """loss = mean(pos[interior].z) — knife-push displacement signal for H1."""
    i = wp.tid()
    if interior_mask[i] == 1:
        wp.atomic_add(loss_buf, 0, pos[i][2] / interior_count)


@wp.kernel
def loss_mean_alpha_kernel(
    alpha: wp.array(dtype=wp.float32),
    interior_mask: wp.array(dtype=wp.int32),
    interior_count: float,
    loss_buf: wp.array(dtype=wp.float32),
):
    """loss = mean(alpha[interior]) — for H2."""
    i = wp.tid()
    if interior_mask[i] == 1:
        wp.atomic_add(loss_buf, 0, alpha[i] / interior_count)


# =============================================================================
# Test runners
# =============================================================================

def setup_env():
    cfg_path = REPO_ROOT / "warp_sim" / "config" / "debulk_cap_armA.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["obs_mode"] = "baseline"
    cfg["task_type"] = "capsule_only"
    cfg["use_ssa_soft_removal"] = True

    env = DebulkEnv(cfg)
    env.reset()
    env.solver.cutting_update_interval = 1
    env.solver.knife_active = 1
    env.solver.knife_radius = 0.15
    env.solver.aspiration_threshold = 0.05
    env.solver.alpha_min = 3.0
    env.solver.soft_removal_sharpness = 20.0
    env.solver._alloc_ssa_buffers()
    return env


def run_test(name: str, enable_layer_a: bool, enable_cutting: bool,
             loss_kernel, knife_pos_np):
    print()
    print("=" * 72)
    print(f"  {name}")
    print(f"  enable_layer_a_collision = {enable_layer_a}")
    print(f"  enable_cutting           = {enable_cutting}")
    print("=" * 72)

    env = setup_env()
    state_snap = snap_solver_state(env)
    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)

    # Autograd
    params = make_param_dict(20.0, 3.0, 0.05, 0.15, requires_grad=True,
                             device=env.device)
    knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                         device=env.device, requires_grad=True)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device,
                        requires_grad=True)
    dt_sub = env.solver.dt / float(env.solver.substeps)

    with wp.Tape() as tape:
        differentiable_substep_isolated(env.solver, params, knife_pos, dt_sub,
                                        enable_layer_a, enable_cutting)
        # Loss target depends on which kernel
        if loss_kernel is loss_sum_pos_z_kernel:
            loss_input = env.solver.p["pos"]
        else:
            loss_input = env.solver._current_alpha()
        wp.launch(loss_kernel, dim=env.solver.N,
                  inputs=[loss_input, interior_mask_wp, interior_count, loss_buf],
                  device=env.device)
    tape.backward(loss=loss_buf)
    wp.synchronize()

    ag_loss = float(loss_buf.numpy()[0])
    ag_kp = knife_pos.grad.numpy()[0].copy() if knife_pos.grad is not None else None
    print(f"  Forward loss: {ag_loss:.6e}")
    print(f"  Autograd ∂L/∂knife_pos: ({ag_kp[0]:+.4e}, {ag_kp[1]:+.4e}, {ag_kp[2]:+.4e})")

    # FD per-axis
    print(f"  FD per-axis (ε=5e-3):")
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
            for _ in range(1):
                differentiable_substep_isolated(env.solver, params_fd, kp,
                                                dt_sub, enable_layer_a, enable_cutting)
            if loss_kernel is loss_sum_pos_z_kernel:
                loss_input = env.solver.p["pos"]
            else:
                loss_input = env.solver._current_alpha()
            wp.launch(loss_kernel, dim=env.solver.N,
                      inputs=[loss_input, interior_mask_wp, interior_count, l_buf],
                      device=env.device)
            wp.synchronize()
            if sign == +1:
                lp = float(l_buf.numpy()[0])
            else:
                lm = float(l_buf.numpy()[0])
        fd_kp[axis] = (lp - lm) / (2.0 * eps)

    print(f"  FD       ∂L/∂knife_pos: ({fd_kp[0]:+.4e}, {fd_kp[1]:+.4e}, {fd_kp[2]:+.4e})")
    print()
    print(f"  {'axis':<4} {'autograd':>14} {'FD':>14} {'rel_err':>10} {'verdict':>10}")
    for axis, axis_name in enumerate(("x", "y", "z")):
        ag = ag_kp[axis]; fd = fd_kp[axis]
        rel = abs(ag - fd) / max(abs(fd), 1e-12)
        sign_ok = (np.sign(ag) == np.sign(fd)) if (abs(fd) > 1e-9 and abs(ag) > 1e-9) else None
        verdict = ("PASS" if rel < 0.10 else
                   "MAGNITUDE_WAY_OFF" if rel > 100 else "FAIL")
        print(f"  {axis_name:<4} {ag:>+14.4e} {fd:>+14.4e} {rel*100:>9.1f}% {verdict:>10}")

    return ag_kp, fd_kp


def main():
    wp.init()
    print()
    print("█" * 72)
    print("  Test H — isolate Layer A vs Layer B path for knife_pos")
    print("█" * 72)

    knife_pos_np = (0.85, 0.5, 0.12)

    # H1: Layer A only (collision fires, cutting OFF, position-dependent loss)
    h1_ag, h1_fd = run_test(
        name="H1 — Layer A only (collision ON, cutting OFF, loss=mean pos.z)",
        enable_layer_a=True, enable_cutting=False,
        loss_kernel=loss_sum_pos_z_kernel,
        knife_pos_np=knife_pos_np,
    )

    # H2: Layer B only (collision OFF, cutting ON, alpha-dependent loss)
    h2_ag, h2_fd = run_test(
        name="H2 — Layer B only (collision OFF, cutting ON, loss=mean alpha)",
        enable_layer_a=False, enable_cutting=True,
        loss_kernel=loss_mean_alpha_kernel,
        knife_pos_np=knife_pos_np,
    )

    # H_baseline: both ON (matches Test E)
    hb_ag, hb_fd = run_test(
        name="H_baseline — both ON, loss=mean alpha (matches Test E)",
        enable_layer_a=True, enable_cutting=True,
        loss_kernel=loss_mean_alpha_kernel,
        knife_pos_np=knife_pos_np,
    )

    # ===== Final analysis =====
    print()
    print("█" * 72)
    print("  COMBINED ANALYSIS")
    print("█" * 72)

    def status(ag, fd):
        rels = [abs(a - f) / max(abs(f), 1e-12) for a, f in zip(ag, fd)]
        if all(r < 0.10 for r in rels):
            return "PASS (≤10% on all axes)"
        elif any(r > 100 for r in rels):
            return f"CATASTROPHIC ({max(rels)*100:.0f}%)"
        else:
            return f"FAIL ({max(rels)*100:.1f}%)"

    print(f"  H1 (Layer A only):     {status(h1_ag, h1_fd)}")
    print(f"  H2 (Layer B only):     {status(h2_ag, h2_fd)}")
    print(f"  H_baseline (both):     {status(hb_ag, hb_fd)}")
    print()
    print("  INTERPRETATION:")

    h1_ok = all(abs(a - f) / max(abs(f), 1e-12) < 0.10 for a, f in zip(h1_ag, h1_fd))
    h2_ok = all(abs(a - f) / max(abs(f), 1e-12) < 0.10 for a, f in zip(h2_ag, h2_fd))

    if h1_ok and h2_ok:
        print("    Both isolated paths PASS → bug is in the INTERACTION between")
        print("    Layer A and Layer B when both are active. Cycle-scoped buffer")
        print("    pattern between the two layers may be needed.")
    elif (not h1_ok) and h2_ok:
        print("    ✓ Layer A path is the bug source.")
        print("    Layer B's knife_pos backward (via update_edge_damage) is correct.")
        print("    Fix: Stage B-style cycle-scoped pred_pos / collision_delta in")
        print("    the Layer A physics path.")
    elif h1_ok and (not h2_ok):
        print("    Layer B path is the bug source!")
        print("    Layer A's knife_pos backward (via collision_delta) is correct.")
        print("    Fix: Layer B's update_edge_damage_kernel knife_pos[0] read or")
        print("    its interaction with subsequent cutting block.")
    else:
        print("    Both paths broken. Bug is widespread in solver._substep backward.")
        print("    Need deeper falsification to find common subsystem.")


if __name__ == "__main__":
    main()
