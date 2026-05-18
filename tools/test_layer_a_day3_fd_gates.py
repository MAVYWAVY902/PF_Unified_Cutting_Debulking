#!/usr/bin/env python3
"""Day 3 FD parity test for Layer A SSA fix.

Tests our fix's CORE PROMISE on production solver._substep() with
use_ssa_soft_removal=True.

  G3 — Test I equivalent (Jacobi iter scan):
       Vary num_iterations ∈ {0, 1, 2, 3, 6}. Compute autograd / FD ratio
       on knife_pos. Bug had ratio amplifying ~40× per iter (5.3e9 at 6
       iters). Fix expected: ratio < 10× at num_iter=6.

  G4 — Test E equivalent (n=1 FD parity):
       At num_iter=6 (production default), compute FD parity for each
       axis of knife_pos. Bug had rel_err 30 億% on x, 442 億% on z.
       Fix expected: rel_err < 5%.

Skips G5 (Test J kernel-isolation) — diagnostic-only, redundant if
G3/G4 pass.
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

from path4_poc_v1_dual_purpose import (
    snap_solver_state, reset_solver_state,
)


@wp.kernel
def loss_sum_pos_z_kernel(
    pos: wp.array(dtype=wp.vec3),
    interior_mask: wp.array(dtype=wp.int32),
    interior_count: float,
    loss_buf: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if interior_mask[i] == 1:
        wp.atomic_add(loss_buf, 0, pos[i][2] / interior_count)


def setup_env():
    from warp_sim.envs.debulk_env import DebulkEnv
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


def run_substep_with_loss(env, knife_pos_xyz, n_substeps, interior_mask_wp,
                           interior_count, requires_grad):
    """Set knife_pos, run substeps, return loss buffer (with tape if requires_grad)."""
    solver = env.solver

    if requires_grad:
        solver.knife_pos = wp.array(
            [wp.vec3(*knife_pos_xyz)], dtype=wp.vec3,
            device=solver.device, requires_grad=True,
        )
    else:
        solver.knife_pos = wp.array(
            [wp.vec3(*knife_pos_xyz)], dtype=wp.vec3,
            device=solver.device, requires_grad=False,
        )

    loss_buf = wp.zeros(1, dtype=wp.float32, device=solver.device,
                         requires_grad=requires_grad)

    if requires_grad:
        tape = wp.Tape()
        with tape:
            for _ in range(n_substeps):
                solver._substep()
            wp.launch(loss_sum_pos_z_kernel, dim=solver.N,
                      inputs=[solver.p["pos"], interior_mask_wp,
                              interior_count, loss_buf],
                      device=solver.device)
        return loss_buf, solver.knife_pos, tape
    else:
        for _ in range(n_substeps):
            solver._substep()
        wp.launch(loss_sum_pos_z_kernel, dim=solver.N,
                  inputs=[solver.p["pos"], interior_mask_wp,
                          interior_count, loss_buf],
                  device=solver.device)
        wp.synchronize()
        return float(loss_buf.numpy()[0])


def measure_grad_at_num_iter(env, num_iter, knife_pos_xyz, eps=5e-3):
    """For given num_iterations, compute (autograd, FD) on knife_pos.

    Returns: (autograd_grad: np.array(3,), fd_grad: np.array(3,)).
    """
    solver = env.solver
    solver.num_iterations = num_iter

    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)

    snap = snap_solver_state(env)

    # ---- Autograd ----
    reset_solver_state(env, snap)
    loss_buf, kp_array, tape = run_substep_with_loss(
        env, knife_pos_xyz, n_substeps=1,
        interior_mask_wp=interior_mask_wp, interior_count=interior_count,
        requires_grad=True,
    )
    tape.backward(loss=loss_buf)
    wp.synchronize()
    autograd_grad = kp_array.grad.numpy()[0].copy()

    # ---- FD per axis ----
    fd_grad = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        # +eps
        reset_solver_state(env, snap)
        kp_p = list(knife_pos_xyz); kp_p[axis] += eps
        loss_p = run_substep_with_loss(
            env, kp_p, n_substeps=1,
            interior_mask_wp=interior_mask_wp, interior_count=interior_count,
            requires_grad=False,
        )

        # -eps
        reset_solver_state(env, snap)
        kp_m = list(knife_pos_xyz); kp_m[axis] -= eps
        loss_m = run_substep_with_loss(
            env, kp_m, n_substeps=1,
            interior_mask_wp=interior_mask_wp, interior_count=interior_count,
            requires_grad=False,
        )

        fd_grad[axis] = (loss_p - loss_m) / (2 * eps)

    return autograd_grad, fd_grad


def gate_g3_g4(env, knife_pos_xyz=(0.85, 0.5, 0.12)):
    print("=" * 78)
    print("Day 3 — G3 (Jacobi iter scan) + G4 (n=1 FD parity)")
    print("=" * 78)
    print(f"  knife_pos baseline: {knife_pos_xyz}, knife_radius=0.15, dt=0.005")
    print()

    print(f"  {'num_iter':>8} {'‖autograd‖':>15} {'‖FD‖':>15} {'ratio':>12} {'verdict':>10}")
    print(f"  {'-'*8} {'-'*15} {'-'*15} {'-'*12} {'-'*10}")

    # G3: scan num_iterations
    g3_results = []
    for num_iter in [0, 1, 2, 3, 6]:
        autograd, fd = measure_grad_at_num_iter(env, num_iter, knife_pos_xyz)
        ag_norm = float(np.linalg.norm(autograd))
        fd_norm = float(np.linalg.norm(fd))
        ratio = ag_norm / max(fd_norm, 1e-12)

        # Plan thresholds: 0=1.23, 1<2, 2<3, 3<5, 6<10
        threshold = {0: 2.0, 1: 2.0, 2: 3.0, 3: 5.0, 6: 10.0}[num_iter]
        verdict = "PASS" if ratio < threshold else "FAIL"

        g3_results.append((num_iter, ag_norm, fd_norm, ratio, verdict))
        print(f"  {num_iter:>8} {ag_norm:>15.4e} {fd_norm:>15.4e} "
              f"{ratio:>12.3e} {verdict:>10}")

    g3_pass = all(v[4] == "PASS" for v in g3_results)
    print()
    print(f"  G3 verdict: {'✅ PASS' if g3_pass else '❌ FAIL'}  "
          f"(plan threshold: ratio < 10× at num_iter=6)")

    # G4: detailed FD parity at num_iter=6 (production default)
    print()
    print("=" * 78)
    print("G4 — Per-axis knife_pos FD parity at num_iter=6")
    print("=" * 78)

    env.solver.num_iterations = 6
    autograd_6, fd_6 = measure_grad_at_num_iter(env, 6, knife_pos_xyz)

    print(f"  {'axis':>6} {'autograd':>15} {'FD':>15} {'rel_err':>12} {'verdict':>10}")
    print(f"  {'-'*6} {'-'*15} {'-'*15} {'-'*12} {'-'*10}")

    g4_results = []
    for axis, axis_name in enumerate(["x", "y", "z"]):
        ag = autograd_6[axis]
        fd = fd_6[axis]
        rel_err = abs(ag - fd) / max(abs(fd), 1e-12)
        # Plan: knife_pos rel_err < 5%
        verdict = "PASS" if rel_err < 0.05 else "FAIL"
        g4_results.append((axis_name, ag, fd, rel_err, verdict))
        print(f"  {axis_name:>6} {ag:>+15.4e} {fd:>+15.4e} "
              f"{rel_err*100:>11.2f}% {verdict:>10}")

    g4_pass = all(v[4] == "PASS" for v in g4_results)
    print()
    print(f"  G4 verdict: {'✅ PASS' if g4_pass else '❌ FAIL'}  "
          f"(plan threshold: rel_err < 5% per axis)")

    return g3_pass, g4_pass


def main():
    wp.init()
    env = setup_env()

    g3_pass, g4_pass = gate_g3_g4(env)

    print()
    print("=" * 78)
    print("Day 3 Summary")
    print("=" * 78)
    print(f"  G3 (Jacobi iter scan):           {'✅ PASS' if g3_pass else '❌ FAIL'}")
    print(f"  G4 (n=1 FD parity):              {'✅ PASS' if g4_pass else '❌ FAIL'}")
    print(f"  G5 (Test J kernel-isolation):    SKIPPED (redundant if G3/G4 pass)")

    if g3_pass and g4_pass:
        print()
        print("  ✅ Day 3 PASS — Layer A SSA fix verified end-to-end.")
        print("  → knife_pos backward catastrophic 5.3e9× off → now matches FD.")
        print("  → Proceed to Day 4 (Stage B regression + Sanity 1-4 checks).")
        return 0
    else:
        print()
        print("  ❌ Day 3 FAIL — STOP, debug per plan §5 Decision Points:")
        if not g3_pass:
            print("    - G3 fail: scan shows residual amplification per iter.")
            print("      Likely root cause: scratch buffers (deltas/delta_counts) need")
            print("      cycle-scoping too. See plan Risk R1 (Müller adjoint) + B/D warnings.")
        if not g4_pass:
            print("    - G4 fail at num_iter=6: end-to-end residual.")
            print("      Per plan: if rel_err > 50% → R1 (Müller adjoint interaction)")
            print("                if rel_err 5-50% → likely deltas/delta_counts cycle-scope needed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
