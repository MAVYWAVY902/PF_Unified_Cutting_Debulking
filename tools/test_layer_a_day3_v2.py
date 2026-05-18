#!/usr/bin/env python3
"""Day 3 v2 — Layer A isolated FD parity test (cutting DISABLED).

Mirrors Test I H1 setup exactly, but uses production solver._substep()
so we test the actual SSA dispatch we wrote.

  Setup mirroring Test I H1:
    - knife_active = 1            (Layer A collision active)
    - cutting disabled via step_count=1 + cutting_update_interval=2
      (1 % 2 != 0, cutting block never fires)
    - loss = mean(pos[interior].z)
    - knife at (0.85, 0.5, 0.12)
    - num_iter scan {0, 1, 2, 3, 6}

  Two modes tested:
    - LEGACY (use_ssa_soft_removal=False): should reproduce Test I's
      ratio ≈ 5.3e9 at num_iter=6, confirming production legacy path
      has the same bug as differentiable_substep_isolated helper.
    - SSA   (use_ssa_soft_removal=True):  should give ratio < 10× at
      num_iter=6, confirming our Layer A fix works.

  PASS condition: legacy reproduces ~5e9 AND SSA fixes it to <10×.
  Either fail = bug.
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

from path4_poc_v1_dual_purpose import snap_solver_state, reset_solver_state


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


def setup_env(use_ssa: bool):
    """Build env in given mode, configured to disable cutting block.
    knife_active=1 keeps Layer A (collision) active. step_count=1 +
    cutting_update_interval=2 → 1%2 != 0 → cutting block does NOT fire."""
    from warp_sim.envs.debulk_env import DebulkEnv
    cfg_path = REPO_ROOT / "warp_sim" / "config" / "debulk_cap_armA.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["obs_mode"] = "baseline"
    cfg["task_type"] = "capsule_only"
    cfg["use_ssa_soft_removal"] = use_ssa

    env = DebulkEnv(cfg)
    env.reset()                                    # step_count=0, knife_active=0
    # Mirror Test I H1: knife active but cutting block disabled
    env.solver.knife_active = 1
    env.solver.knife_radius = 0.15
    env.solver.aspiration_threshold = 0.05
    env.solver.alpha_min = 3.0
    env.solver.soft_removal_sharpness = 20.0
    env.solver.cutting_update_interval = 2          # ← key: combined with step_count=1
    env.solver.step_count = 1                        # ← key: 1%2=1 ≠ 0, cutting OFF
    if use_ssa:
        env.solver._alloc_ssa_buffers()
    return env


def run_substep_with_loss(env, knife_pos_xyz, requires_grad,
                            interior_mask_wp, interior_count):
    """Set knife_pos as wp.array(requires_grad), run 1 substep, return
    (loss_buf, knife_pos_array, tape_or_None)."""
    solver = env.solver
    solver.knife_pos = wp.array(
        [wp.vec3(*knife_pos_xyz)], dtype=wp.vec3,
        device=solver.device, requires_grad=requires_grad,
    )
    loss_buf = wp.zeros(1, dtype=wp.float32, device=solver.device,
                         requires_grad=requires_grad)
    if requires_grad:
        tape = wp.Tape()
        with tape:
            solver._substep()
            wp.launch(loss_sum_pos_z_kernel, dim=solver.N,
                      inputs=[solver.p["pos"], interior_mask_wp,
                              interior_count, loss_buf], device=solver.device)
        return loss_buf, solver.knife_pos, tape
    else:
        solver._substep()
        wp.launch(loss_sum_pos_z_kernel, dim=solver.N,
                  inputs=[solver.p["pos"], interior_mask_wp,
                          interior_count, loss_buf], device=solver.device)
        wp.synchronize()
        return float(loss_buf.numpy()[0])


def measure_grad(env, num_iter, knife_pos_xyz=(0.85, 0.5, 0.12), eps=5e-3):
    """Returns (autograd_grad: np.array(3,), fd_grad: np.array(3,))."""
    solver = env.solver
    solver.num_iterations = num_iter

    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)

    snap = snap_solver_state(env)

    # Autograd
    reset_solver_state(env, snap)
    solver.step_count = 1   # snap captures step_count, but ensure 1 (defensive)
    loss_buf, kp_arr, tape = run_substep_with_loss(
        env, knife_pos_xyz, True, interior_mask_wp, interior_count)
    tape.backward(loss=loss_buf)
    wp.synchronize()
    autograd = kp_arr.grad.numpy()[0].copy()

    # FD per axis
    fd = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        reset_solver_state(env, snap); solver.step_count = 1
        kp_p = list(knife_pos_xyz); kp_p[axis] += eps
        loss_p = run_substep_with_loss(env, kp_p, False,
                                          interior_mask_wp, interior_count)
        reset_solver_state(env, snap); solver.step_count = 1
        kp_m = list(knife_pos_xyz); kp_m[axis] -= eps
        loss_m = run_substep_with_loss(env, kp_m, False,
                                          interior_mask_wp, interior_count)
        fd[axis] = (loss_p - loss_m) / (2 * eps)

    return autograd, fd


def scan_mode(env, mode_name):
    print(f"\n{mode_name}")
    print(f"  {'num_iter':>8}   {'‖autograd‖':>15}   {'‖FD‖':>15}   {'ratio':>15}")
    results = []
    for num_iter in [0, 1, 2, 3, 6]:
        ag, fd = measure_grad(env, num_iter)
        ag_norm = float(np.linalg.norm(ag))
        fd_norm = float(np.linalg.norm(fd))
        ratio = ag_norm / max(fd_norm, 1e-12)
        results.append((num_iter, ag_norm, fd_norm, ratio))
        print(f"  {num_iter:>8}   {ag_norm:>15.4e}   {fd_norm:>15.4e}   {ratio:>15.3e}")
    return results


def main():
    wp.init()
    print("=" * 78)
    print("Day 3 v2 — Layer A isolated FD parity (cutting disabled)")
    print("Setup mirrors Test I H1: knife active, cutting OFF, loss=mean(pos.z)")
    print("=" * 78)

    # Phase 1: legacy mode (no fix) — should reproduce Test I's 5.3e9
    env_legacy = setup_env(use_ssa=False)
    results_legacy = scan_mode(env_legacy, "[1] LEGACY MODE (use_ssa_soft_removal=False)")

    # Phase 2: SSA mode (with fix) — expect <10× at num_iter=6
    env_ssa = setup_env(use_ssa=True)
    results_ssa = scan_mode(env_ssa, "[2] SSA MODE (use_ssa_soft_removal=True, with our fix)")

    # Verdict
    print()
    print("=" * 78)
    print("Verdict")
    print("=" * 78)

    # Find num_iter=6 row
    legacy_6 = next(r for r in results_legacy if r[0] == 6)
    ssa_6    = next(r for r in results_ssa    if r[0] == 6)

    print(f"  At num_iter=6:")
    print(f"    Legacy ratio: {legacy_6[3]:.3e}")
    print(f"    SSA    ratio: {ssa_6[3]:.3e}")
    print(f"    Test I memory baseline: ~5.3e9")

    legacy_matches = 1e8 < legacy_6[3] < 1e11
    ssa_pass       = ssa_6[3] < 10.0

    print()
    if legacy_matches:
        print(f"  ✅ Legacy mode reproduces Test I bug (ratio={legacy_6[3]:.2e} ∈ [1e8, 1e11])")
    else:
        print(f"  ⚠️ Legacy mode ratio {legacy_6[3]:.2e} doesn't match Test I's ~5e9.")
        print(f"     Possible reasons: cutting wasn't truly disabled, env state diff,")
        print(f"     or production solver differs from differentiable_substep_isolated.")

    if ssa_pass:
        print(f"  ✅ SSA mode (our fix) ratio {ssa_6[3]:.2e} < 10× — Layer A fix WORKS")
    else:
        print(f"  ❌ SSA mode ratio {ssa_6[3]:.2e} >= 10× — Layer A fix has issues")

    if legacy_matches and ssa_pass:
        print()
        print(f"  ✅ Day 3 v2 PASS — fix verified on Layer A only.")
        print(f"  → Next: Day 3 v3 (cutting ENABLED, full pipeline)")
        return 0
    else:
        print()
        print(f"  ❌ Day 3 v2 FAIL — see analysis above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
