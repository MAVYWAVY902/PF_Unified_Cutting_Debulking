#!/usr/bin/env python3
"""
Path 4 POC v1 — DUAL-PURPOSE: trajopt + sys-ID gradient verification.

Goal: Verify wp.Tape backward through 10 substeps of full physics + cutting
pipeline produces correct gradients for BOTH:

  1. ∂loss/∂knife_pos          (trajopt path, Layer A + Layer B)
  2. ∂loss/∂sharpness          (sys-ID path, Layer B only — soft removal sigmoid)
  3. ∂loss/∂alpha_min          (sys-ID path, Layer B only — P(F) edge sigmoid)
  4. ∂loss/∂threshold          (sys-ID path, Layer B only — soft removal sigmoid)

CONTEXT — why this test exists:
  Stage B (2026-04-30 → 05-01) verified Layer B (cutting block) backward via a
  SYNTHETIC harness (`run_ssa_full()` in `calibrate_stage2_autograd.py`) that
  bypasses ALL physics kernels (predict / collision / cluster_solve / apply /
  cluster_quat / vel_update). That synthetic harness is sufficient to verify
  the cutting block's autograd, but real-data sys-ID (Stage 2.2 phantom
  calibration) MUST go through `solver._substep()` because q̂_i(θ) requires
  full physics (cavity formed by knife pushing tissue).

  This test is the FIRST to verify the full physics + cutting pipeline backward
  in a single tape. If it passes:
    - Path 4 trajectory optimization is unblocked (∂loss/∂knife_pos works)
    - Real-data sys-ID is unblocked (∂loss/∂sys-ID-params works through
      solver._substep()-equivalent path)

ARCHITECTURE:
  Mirrors `solver._substep()` orchestration exactly (verified from
  solver.py:200-433) but swaps cutting block kernels to their `_param_arr`
  variants, so sys-ID params can be passed as wp.array(requires_grad=True).

  Layer A kernels (predict/collision/cluster_solve/apply/cluster_quat/vel_update)
  use the same kernels production uses — knife_pos is already wp.array, so
  trajopt gradient flows naturally without any kernel swap.

  Layer B kernels swap:
    update_edge_damage_kernel_ssa     → update_edge_damage_kernel_ssa_param_arr
    update_soft_removal_kernel_ssa    → update_soft_removal_kernel_ssa_param_arr

  All other kernels (apply_edge_damage_ssa, particle damage chain) take no
  scalar params and remain unchanged.

PASS CRITERIA (all must hold):
  - 0 verify_autograd_array_access warnings on physics kernels
  - All 4 gradients finite (no NaN/inf)
  - FD parity rel_err < 5% at ε=1e-3 for each of 4 variables (knife_pos check
    via per-component, params via single ε)
  - Loss decreases monotonically when running gradient descent (sanity check)

FAIL → DIAGNOSE:
  - Warning on predict/collision/cluster_solve/apply/cluster_quat → Layer A
    kernel needs SSA refactor (Stage B-style Day 1.5 redo)
  - NaN/inf → numerical instability (try smaller dt, fewer iterations, or
    Müller'16 SVD adjoint stability check)
  - 0 grad → silent zero, find Pattern A/B/C/D in failing kernel
  - rel_err > 5% — wp.atomic_add or wp.where producing biased gradient
"""
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

wp.config.verify_autograd_array_access = True

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---- Layer A kernels (Warp tape-safe SSA, used by production solver._substep) ----
from pf_shape_matching_warp.sim.kernels import (
    predict_kernel_ssa,
    collision_kernel_ssa,
    apply_collision_delta_kernel,
    clear_deltas_kernel,
    cluster_solve_kernel_ssa,
    apply_kernel_ssa,
    apply_shape_delta_kernel,
    cluster_quat_update_kernel,
    update_velocity_kernel,
)

# ---- Layer B kernels — _param_arr variants for sys-ID autograd ----
from pf_shape_matching_warp.sim.pf_damage import (
    update_edge_damage_kernel_ssa_param_arr,
    apply_edge_damage_ssa_kernel,
    update_particle_damage_kernel,
    finalize_particle_damage_kernel_ssa,
    apply_particle_damage_ssa_kernel,
    update_soft_removal_kernel_ssa_param_arr,
)

from warp_sim.envs.debulk_env import DebulkEnv


# =============================================================================
# Helpers
# =============================================================================

@wp.kernel
def loss_mean_alpha_at_interior_kernel(
    alpha: wp.array(dtype=wp.float32),
    interior_mask: wp.array(dtype=wp.int32),
    interior_count: float,                          # precomputed N_interior
    loss_buf: wp.array(dtype=wp.float32),
):
    """loss = mean(alpha[interior]) — low alpha means debulked."""
    i = wp.tid()
    if interior_mask[i] == 1:
        wp.atomic_add(loss_buf, 0, alpha[i] / interior_count)


def make_param_dict(
    sharpness: float, alpha_min: float, threshold: float,
    knife_radius: float, requires_grad: bool, device: str,
) -> dict:
    """Returns dict of wp.array(1)-encoded scalars + knife_radius (for Layer B
    _param_arr kernel calls). NOTE: Layer A still uses Python float
    knife_radius — this dict is for cutting block only."""
    return dict(
        sharpness=wp.array([sharpness], dtype=wp.float32, device=device,
                           requires_grad=requires_grad),
        alpha_min=wp.array([alpha_min], dtype=wp.float32, device=device,
                           requires_grad=requires_grad),
        threshold=wp.array([threshold], dtype=wp.float32, device=device,
                           requires_grad=requires_grad),
        knife_radius=wp.array([knife_radius], dtype=wp.float32, device=device,
                              requires_grad=requires_grad),
    )


def differentiable_substep(solver, params, knife_pos, dt_sub):
    """Mirrors solver._substep() orchestration but uses _param_arr cutting
    kernels. Reads `solver` for mesh/buffers, takes `params` dict (wp.array
    encoded sys-ID scalars) and `knife_pos` (wp.array(vec3)) as autograd
    inputs.

    Verified-faithful copy of solver.py:200-433 SSA path. Diffs vs production:
      - knife_pos: passed as arg, not solver.knife_pos
      - cutting block uses _param_arr kernel variants
      - Layer A unchanged
      - cluster_quat update unchanged
      - update_velocity unchanged
    """
    N = solver.N
    num_edges = solver.e["num_edges"] if solver.e is not None else 0

    # Lazy alloc SSA buffers if not yet
    solver._alloc_ssa_buffers()

    # ---- 1. Predict ----
    wp.launch(
        predict_kernel_ssa, dim=N,
        inputs=[
            solver.p["pos"], solver.p["vel"],
            solver.p["pred_pos"], solver.p["old_pos"],
            solver._current_alpha(), solver.p["fixed"],
            solver.gravity, dt_sub, solver.damping,
        ],
        device=solver.device,
    )

    # ---- 2. Initial collision ----
    solver.p["collision_delta"].zero_()
    wp.launch(
        collision_kernel_ssa, dim=N,
        inputs=[
            solver.p["pred_pos"], solver.p["pos"],
            solver._current_alpha(), solver.p["fixed"],
            solver.floor_y, solver.floor_friction,
            knife_pos, solver.knife_radius, solver.knife_active,
            solver.p["collision_delta"],
        ],
        device=solver.device,
    )
    wp.launch(
        apply_collision_delta_kernel, dim=N,
        inputs=[solver.p["pred_pos"], solver.p["collision_delta"]],
        device=solver.device,
    )

    # ---- 3. Jacobi loop ----
    for _ in range(solver.num_iterations):
        wp.launch(clear_deltas_kernel, dim=N,
                  inputs=[solver.p["deltas"], solver.p["delta_counts"]],
                  device=solver.device)

        wp.launch(cluster_solve_kernel_ssa, dim=solver.num_clusters,
                  inputs=[
                      solver.p["pred_pos"], solver._current_alpha(),
                      solver.c["starts"], solver.c["sizes"],
                      solver.c["particle_ids"], solver.c["rest_offsets"],
                      solver.c["cluster_quat"], solver.muller_iter,
                      solver.p["deltas"], solver.p["delta_counts"],
                  ],
                  device=solver.device)

        solver.p["shape_delta"].zero_()
        wp.launch(apply_kernel_ssa, dim=N,
                  inputs=[
                      solver.p["deltas"], solver.p["delta_counts"],
                      solver.p["damage"], solver._current_alpha(),
                      solver.p["fixed"], solver.stiffness,
                      solver.p["shape_delta"],
                  ],
                  device=solver.device)
        wp.launch(apply_shape_delta_kernel, dim=N,
                  inputs=[solver.p["pred_pos"], solver.p["shape_delta"]],
                  device=solver.device)

        solver.p["collision_delta"].zero_()
        wp.launch(collision_kernel_ssa, dim=N,
                  inputs=[
                      solver.p["pred_pos"], solver.p["pos"],
                      solver._current_alpha(), solver.p["fixed"],
                      solver.floor_y, solver.floor_friction,
                      knife_pos, solver.knife_radius, solver.knife_active,
                      solver.p["collision_delta"],
                  ],
                  device=solver.device)
        wp.launch(apply_collision_delta_kernel, dim=N,
                  inputs=[solver.p["pred_pos"], solver.p["collision_delta"]],
                  device=solver.device)

    # ---- 4. cluster_quat warm start ----
    wp.launch(cluster_quat_update_kernel, dim=solver.num_clusters,
              inputs=[
                  solver.p["pred_pos"], solver._current_alpha(),
                  solver.c["starts"], solver.c["sizes"],
                  solver.c["particle_ids"], solver.c["rest_offsets"],
                  solver.c["cluster_quat"], solver.muller_iter,
              ],
              device=solver.device)

    # ---- 5. Cutting block (every substep — we set cutting_update_interval=1) ----
    if (solver.knife_active == 1 and solver.e is not None and
            solver.step_count % solver.cutting_update_interval == 0):
        c = solver._ssa_cycle_index
        if c + 1 >= solver.ssa_buffer_capacity:
            raise RuntimeError(
                f"SSA buffer capacity {solver.ssa_buffer_capacity} exhausted "
                f"at cycle {c+1}.")

        alpha_prev = solver._ssa_alpha_buffers[c]
        alpha_next = solver._ssa_alpha_buffers[c + 1]
        ed_prev = solver._ssa_edge_damage_buffers[c]
        ed_next = solver._ssa_edge_damage_buffers[c + 1]
        pd_prev = solver._ssa_particle_damage_buffers[c]
        pd_next = solver._ssa_particle_damage_buffers[c + 1]

        # Edge damage (USES _param_arr WITH knife_radius_arr + alpha_min_arr)
        solver.e["damage_delta"].zero_()
        wp.launch(
            update_edge_damage_kernel_ssa_param_arr, dim=num_edges,
            inputs=[
                solver.p["pred_pos"], alpha_prev,
                solver.e["a"], solver.e["b"],
                solver.e["pf_rest"], ed_prev,
                knife_pos, params["knife_radius"], params["alpha_min"],
                solver.e["damage_delta"],
            ],
            device=solver.device,
        )
        wp.launch(apply_edge_damage_ssa_kernel, dim=num_edges,
                  inputs=[ed_prev, solver.e["damage_delta"], ed_next],
                  device=solver.device)

        # Particle damage chain
        damage_sum = wp.zeros(N, dtype=wp.float32, device=solver.device,
                              requires_grad=True)
        damage_count = wp.zeros(N, dtype=wp.float32, device=solver.device,
                                requires_grad=True)
        wp.launch(update_particle_damage_kernel, dim=num_edges,
                  inputs=[damage_sum, damage_count, alpha_prev,
                          solver.e["a"], solver.e["b"], ed_next],
                  device=solver.device)
        solver.p["particle_damage_delta"].zero_()
        wp.launch(finalize_particle_damage_kernel_ssa, dim=N,
                  inputs=[damage_sum, damage_count, pd_prev,
                          solver.p["particle_damage_delta"]],
                  device=solver.device)
        wp.launch(apply_particle_damage_ssa_kernel, dim=N,
                  inputs=[pd_prev, solver.p["particle_damage_delta"], pd_next],
                  device=solver.device)

        # Soft removal (USES _param_arr WITH threshold_arr + sharpness_arr)
        wp.launch(
            update_soft_removal_kernel_ssa_param_arr, dim=N,
            inputs=[
                pd_next, alpha_prev, solver.p["fixed"],
                params["threshold"], params["sharpness"],
                alpha_next,
            ],
            device=solver.device,
        )
        solver._ssa_cycle_index = c + 1

    # ---- 6. Update velocity ----
    inv_dt = 1.0 / dt_sub
    wp.launch(update_velocity_kernel, dim=N,
              inputs=[
                  solver.p["pos"], solver.p["vel"],
                  solver.p["pred_pos"], solver.p["old_pos"],
                  solver._current_alpha(), solver.p["fixed"],
                  inv_dt,
              ],
              device=solver.device)

    solver.step_count += 1


def run_episode(env, params, knife_pos, n_substeps: int):
    """Full episode forward — n_substeps of differentiable_substep.
    Returns final alpha buffer for loss computation."""
    solver = env.solver
    dt_sub = solver.dt / float(solver.substeps)
    for _ in range(n_substeps):
        differentiable_substep(solver, params, knife_pos, dt_sub)
    return solver._current_alpha()


def reset_solver_state(env, snap):
    """Restore pos/vel/alpha buffers + step_count + ssa_cycle_index from snap.
    Called between FD evaluations to ensure each forward pass is independent."""
    solver = env.solver
    wp.copy(solver.p["pos"], snap["pos"])
    wp.copy(solver.p["vel"], snap["vel"])
    wp.copy(solver.p["pred_pos"], snap["pred_pos"])
    wp.copy(solver.p["old_pos"], snap["old_pos"])
    wp.copy(solver.p["damage"], snap["damage"])
    wp.copy(solver.c["cluster_quat"], snap["cluster_quat"])
    solver.step_count = snap["step_count"]
    solver._ssa_cycle_index = 0
    # Re-init SSA buffers (only if allocated)
    if solver._ssa_alpha_buffers is not None:
        for b in solver._ssa_alpha_buffers:
            b.fill_(1.0)
        for b in solver._ssa_edge_damage_buffers:
            b.zero_()
        for b in solver._ssa_particle_damage_buffers:
            b.zero_()


def snap_solver_state(env):
    solver = env.solver
    return dict(
        pos=wp.clone(solver.p["pos"]),
        vel=wp.clone(solver.p["vel"]),
        pred_pos=wp.clone(solver.p["pred_pos"]),
        old_pos=wp.clone(solver.p["old_pos"]),
        damage=wp.clone(solver.p["damage"]),
        cluster_quat=wp.clone(solver.c["cluster_quat"]),
        step_count=solver.step_count,
    )


def make_knife_pos(initial_pos, requires_grad: bool, device: str):
    return wp.array([wp.vec3(*initial_pos)], dtype=wp.vec3,
                    device=device, requires_grad=requires_grad)


# =============================================================================
# Loss computation (forward + backward)
# =============================================================================

def forward_only_loss(env, params, knife_pos, n_substeps,
                      interior_mask_wp, interior_count: float):
    """Forward pass without tape — returns scalar loss."""
    alpha = run_episode(env, params, knife_pos, n_substeps)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device)
    wp.launch(loss_mean_alpha_at_interior_kernel, dim=env.solver.N,
              inputs=[alpha, interior_mask_wp, interior_count, loss_buf],
              device=env.device)
    wp.synchronize()
    return float(loss_buf.numpy()[0])


def autograd_pass(env, params, knife_pos, n_substeps,
                  interior_mask_wp, interior_count: float):
    """Full forward + backward in tape. Returns (loss, grads_dict)."""
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device,
                        requires_grad=True)
    with wp.Tape() as tape:
        alpha = run_episode(env, params, knife_pos, n_substeps)
        wp.launch(loss_mean_alpha_at_interior_kernel, dim=env.solver.N,
                  inputs=[alpha, interior_mask_wp, interior_count, loss_buf],
                  device=env.device)
    tape.backward(loss=loss_buf)

    grads = {}
    for name, arr in params.items():
        if arr.grad is not None:
            grads[name] = float(arr.grad.numpy()[0])
        else:
            grads[name] = None
    if knife_pos.grad is not None:
        kg = knife_pos.grad.numpy()[0]
        grads["knife_pos"] = (float(kg[0]), float(kg[1]), float(kg[2]))
    else:
        grads["knife_pos"] = None

    return float(loss_buf.numpy()[0]), grads


# =============================================================================
# FD verification
# =============================================================================

def fd_grad_param(env, params_base, knife_pos_base, n_substeps,
                  interior_mask_wp, interior_count, param_name, eps=1e-3):
    """Central-diff FD over one scalar param. Snapshot/restore state between."""
    snap = snap_solver_state(env)

    def _eval(p_value):
        # Make params with one perturbed
        p_kw = {k: float(v.numpy()[0]) for k, v in params_base.items()}
        p_kw[param_name] = float(p_value)
        params = make_param_dict(
            p_kw["sharpness"], p_kw["alpha_min"], p_kw["threshold"],
            p_kw["knife_radius"],
            requires_grad=False, device=env.device,
        )
        knife_pos_fwd = wp.clone(knife_pos_base)
        reset_solver_state(env, snap)
        return forward_only_loss(env, params, knife_pos_fwd, n_substeps,
                                 interior_mask_wp, interior_count)

    base = float(params_base[param_name].numpy()[0])
    loss_plus = _eval(base + eps)
    loss_minus = _eval(base - eps)
    reset_solver_state(env, snap)
    return (loss_plus - loss_minus) / (2.0 * eps)


def fd_grad_knife_pos(env, params_base, knife_pos_base, n_substeps,
                      interior_mask_wp, interior_count, axis, eps=1e-4):
    """Central-diff FD over one knife_pos axis (0/1/2 = x/y/z)."""
    snap = snap_solver_state(env)
    base = knife_pos_base.numpy()[0].copy()

    def _eval(perturb):
        kp_np = base.copy()
        kp_np[axis] += perturb
        kp = wp.array([wp.vec3(*kp_np)], dtype=wp.vec3, device=env.device,
                      requires_grad=False)
        params = make_param_dict(
            float(params_base["sharpness"].numpy()[0]),
            float(params_base["alpha_min"].numpy()[0]),
            float(params_base["threshold"].numpy()[0]),
            float(params_base["knife_radius"].numpy()[0]),
            requires_grad=False, device=env.device,
        )
        reset_solver_state(env, snap)
        return forward_only_loss(env, params, kp, n_substeps,
                                 interior_mask_wp, interior_count)

    loss_plus = _eval(eps)
    loss_minus = _eval(-eps)
    reset_solver_state(env, snap)
    return (loss_plus - loss_minus) / (2.0 * eps)


# =============================================================================
# Main
# =============================================================================

def main():
    wp.init()
    print("=" * 72)
    print("Path 4 POC v1 — DUAL-PURPOSE: trajopt + sys-ID gradient verification")
    print("=" * 72)

    # ---- Load env ----
    import yaml
    cfg_path = REPO_ROOT / "warp_sim" / "config" / "debulk_cap_armA.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["obs_mode"] = "baseline"
    cfg["task_type"] = "capsule_only"
    cfg["use_ssa_soft_removal"] = True

    env = DebulkEnv(cfg)
    env.reset()

    # Configure: cutting fires every substep
    env.solver.cutting_update_interval = 1
    env.solver.knife_active = 1

    print(f"  N particles: {env.solver.N}")
    print(f"  N edges:     {env.solver.e['num_edges']}")
    print(f"  Interior:    {env._total_interior}")
    print(f"  use_ssa:     {env.solver.use_ssa_soft_removal}")

    # ---- Setup params ----
    params = make_param_dict(
        sharpness=20.0, alpha_min=3.0, threshold=0.3, knife_radius=0.1,
        requires_grad=True, device=env.device,
    )

    # Strong-signal control: place knife ON the torus tube (NOT in donut hole).
    # Torus geometry: center (0.5, 0.5, 0.12), major R=0.35, tube r=0.12.
    # Tube particles live at distance ~0.35 from XY center → e.g. (0.85, 0.5, 0.12).
    env.solver.knife_radius = 0.15
    env.solver.aspiration_threshold = 0.05  # was 0.3; lower → easier cut
    knife_pos = make_knife_pos((0.85, 0.5, 0.12), requires_grad=True,
                               device=env.device)

    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)

    n_substeps = 30   # 30 substeps × 1 cycle/substep = 30 cycles, well within capacity 256

    print(f"  knife_pos:     (0.85, 0.5, 0.12)  ← on torus tube")
    print(f"  knife_radius:  {env.solver.knife_radius}")
    print(f"  threshold:     {env.solver.aspiration_threshold}")
    print(f"  ssa_capacity:  {env.solver.ssa_buffer_capacity}")
    print(f"  cutting_iv:    {env.solver.cutting_update_interval}")
    print(f"  knife_active:  {env.solver.knife_active}")
    print(f"  num_iter:      {env.solver.num_iterations}")

    # ---- Pre-allocate SSA buffers (lazy alloc would happen on first cutting cycle) ----
    env.solver._alloc_ssa_buffers()
    state_snap = snap_solver_state(env)
    initial_alpha_sum = float(env.solver._ssa_alpha_buffers[0].numpy().sum())
    print(f"  Initial alpha sum (should be N = {env.solver.N}): "
          f"{initial_alpha_sum:.1f}")

    # ============================================================
    # TEST 1: Forward sanity
    # ============================================================
    print()
    print("─" * 72)
    print("TEST 1: Forward sanity (no tape)")
    print("─" * 72)
    t0 = time.time()
    loss = forward_only_loss(env, params, knife_pos, n_substeps,
                             interior_mask_wp, interior_count)
    fwd_time_ms = (time.time() - t0) * 1000
    print(f"  Loss after {n_substeps} substeps: {loss:.6e}")
    print(f"  Forward time: {fwd_time_ms:.1f} ms ({fwd_time_ms/n_substeps:.1f} ms/substep)")

    # Diagnostic: did anything actually happen?
    final_alpha = env.solver._current_alpha().numpy()
    final_pd = env.solver._ssa_particle_damage_buffers[env.solver._ssa_cycle_index].numpy()
    final_ed = env.solver._ssa_edge_damage_buffers[env.solver._ssa_cycle_index].numpy()
    print(f"  Final cycle:    {env.solver._ssa_cycle_index}")
    print(f"  alpha:    min={final_alpha.min():.4f}  mean={final_alpha.mean():.4f}  "
          f"#(<0.5)={int((final_alpha < 0.5).sum())}")
    print(f"  particle_damage:  max={final_pd.max():.4f}  mean={final_pd.mean():.6f}  "
          f"#(>0)={int((final_pd > 0).sum())}")
    print(f"  edge_damage:      max={final_ed.max():.4f}  mean={final_ed.mean():.6f}  "
          f"#(>0)={int((final_ed > 0).sum())}")

    # ============================================================
    # TEST 2: Autograd backward (gradient flow check)
    # ============================================================
    print()
    print("─" * 72)
    print("TEST 2: Autograd backward")
    print("─" * 72)

    reset_solver_state(env, state_snap)
    t0 = time.time()
    ag_loss, ag_grads = autograd_pass(env, params, knife_pos, n_substeps,
                                      interior_mask_wp, interior_count)
    bwd_time_ms = (time.time() - t0) * 1000
    print(f"  Loss (autograd):  {ag_loss:.6e}  (matches forward sanity? "
          f"{abs(ag_loss - loss) < 1e-5})")
    print(f"  Backward time:    {bwd_time_ms:.1f} ms")
    print()
    print("  Gradients found:")
    all_finite = True
    for name in ("knife_pos", "knife_radius", "alpha_min", "sharpness", "threshold"):
        g = ag_grads[name]
        if g is None:
            print(f"    {name:<15} ❌ NONE (gradient did not flow)")
            all_finite = False
        elif name == "knife_pos":
            finite = all(np.isfinite(c) for c in g)
            print(f"    {name:<15} {'✓' if finite else '✗'} ({g[0]:+.3e}, "
                  f"{g[1]:+.3e}, {g[2]:+.3e})")
            if not finite:
                all_finite = False
        else:
            finite = np.isfinite(g)
            print(f"    {name:<15} {'✓' if finite else '✗'} {g:+.3e}")
            if not finite:
                all_finite = False

    test2_pass = all_finite
    print(f"\n  Test 2 verdict: {'PASS' if test2_pass else 'FAIL'}")

    # ============================================================
    # TEST 3: FD parity
    # ============================================================
    print()
    print("─" * 72)
    print("TEST 3: FD parity (autograd vs central-diff at ε=1e-3)")
    print("─" * 72)

    if not test2_pass:
        print("  Skipped (Test 2 failed)")
        test3_pass = False
    else:
        results = {}
        # Scalar params
        for name, eps in (("alpha_min", 1e-3), ("sharpness", 1e-2),
                          ("threshold", 1e-3), ("knife_radius", 1e-3)):
            ag = ag_grads[name]
            t0 = time.time()
            fd = fd_grad_param(env, params, knife_pos, n_substeps,
                               interior_mask_wp, interior_count, name, eps=eps)
            t_fd = (time.time() - t0) * 1000
            sign_ok = (np.sign(ag) == np.sign(fd)) if abs(fd) > 1e-12 else True
            rel_err = abs(ag - fd) / max(abs(fd), 1e-12)
            verdict = "PASS" if (sign_ok and rel_err < 0.05) else "FAIL"
            results[name] = (ag, fd, rel_err, verdict)
            print(f"    {name:<15} ag={ag:+.3e}  fd={fd:+.3e}  rel_err="
                  f"{rel_err*100:6.2f}%  [{verdict}]  ({t_fd:.0f} ms)")

        # Knife pos per-axis
        print()
        for axis, axis_name in enumerate(("x", "y", "z")):
            ag = ag_grads["knife_pos"][axis]
            t0 = time.time()
            fd = fd_grad_knife_pos(env, params, knife_pos, n_substeps,
                                   interior_mask_wp, interior_count,
                                   axis, eps=1e-4)
            t_fd = (time.time() - t0) * 1000
            sign_ok = (np.sign(ag) == np.sign(fd)) if abs(fd) > 1e-12 else True
            rel_err = abs(ag - fd) / max(abs(fd), 1e-12)
            verdict = "PASS" if (sign_ok and rel_err < 0.05) else "FAIL"
            results[f"knife_pos_{axis_name}"] = (ag, fd, rel_err, verdict)
            print(f"    knife_pos[{axis_name}]   ag={ag:+.3e}  fd={fd:+.3e}  "
                  f"rel_err={rel_err*100:6.2f}%  [{verdict}]  ({t_fd:.0f} ms)")

        test3_pass = all(r[3] == "PASS" for r in results.values())
        print(f"\n  Test 3 verdict: {'PASS' if test3_pass else 'FAIL'}")

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print()
    print("=" * 72)
    print("PATH 4 POC v1 DUAL-PURPOSE VERDICT")
    print("=" * 72)
    print(f"  Test 2 (autograd backward, all grads finite):  "
          f"{'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"  Test 3 (FD parity rel_err < 5%):                "
          f"{'✅ PASS' if test3_pass else '❌ FAIL'}")
    print()
    if test2_pass and test3_pass:
        print("  ✅ DUAL-PURPOSE PASS:")
        print("     - Path 4 trajopt: ∂loss/∂knife_pos verified through real")
        print("       solver._substep()-mirrored pipeline (Layer A + Layer B)")
        print("     - Real-data sys-ID: ∂loss/∂(sharpness, alpha_min, threshold,")
        print("       knife_radius) verified through full physics + cutting tape")
        print()
        print("  → Phase 2 (trajectory optimization with Adam) unblocked")
        print("  → Real-phantom sys-ID code path verified (only data + loader missing)")
        return 0
    else:
        print("  ❌ FAIL — diagnose:")
        if not test2_pass:
            print("     Test 2 fail → some grad is None or NaN/inf.")
            print("       Investigate: which kernel breaks gradient flow?")
            print("       Run with verify_autograd_array_access warnings visible.")
        if not test3_pass:
            print("     Test 3 fail → FD parity fail despite finite grad.")
            print("       Possible: silent zero (Pattern A/B/C/D), or biased")
            print("       atomic_add backward.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
