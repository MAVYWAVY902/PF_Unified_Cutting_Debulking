#!/usr/bin/env python3
"""Path 4 Falsification Test G — knife active but FAR from tumor.

QUESTION: Is the n=1 catastrophic knife_pos gradient due to:
  (a) wp.where backward computing the UNTAKEN p_knife branch with bad
      gradient even when no particle is inside knife (Pattern A from
      reference_warp_autograd_gotchas.md), OR
  (b) actual collision firing causing R-W-R amplification

Setup: knife at (10, 10, 10) — way outside any particle. With knife_active=1:
  - collision_kernel_ssa runs the wp.where
  - Forward: ALL particles take p_floor branch (no one inside knife)
  - Backward: wp.where computes BOTH branches' gradients
  - update_edge_damage_kernel: ALL edges early-return (dist > 1.3 * radius)
  - Cutting block effectively does nothing in forward

Expected loss: ~1.0 (no cutting happens)
Expected gradients (CORRECT autograd): all zero (params don't enter loss)

If knife_pos grad ≠ 0 → wp.where backward in collision_kernel is computing
spurious gradient through untaken p_knife branch. Pattern A confirmed.

If all grads = 0 → bug appears only when collision actually fires; bug is
in collision-firing path (likely R-W-R on pred_pos amplification).
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
    differentiable_substep, make_param_dict, snap_solver_state,
    reset_solver_state, loss_mean_alpha_at_interior_kernel,
)
from warp_sim.envs.debulk_env import DebulkEnv


def main():
    wp.init()
    print("=" * 72)
    print("  Test G — knife_active=1 BUT knife far from tumor (untaken branches)")
    print("=" * 72)

    cfg_path = REPO_ROOT / "warp_sim" / "config" / "debulk_cap_armA.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["obs_mode"] = "baseline"
    cfg["task_type"] = "capsule_only"
    cfg["use_ssa_soft_removal"] = True

    env = DebulkEnv(cfg)
    env.reset()
    env.solver.cutting_update_interval = 1
    env.solver.knife_active = 1   # ★ knife ON
    env.solver.knife_radius = 0.15
    env.solver.aspiration_threshold = 0.05
    env.solver.alpha_min = 3.0
    env.solver.soft_removal_sharpness = 20.0
    env.solver._alloc_ssa_buffers()

    # ★ Knife FAR (10, 10, 10) — no particle is anywhere close
    knife_pos_np = (10.0, 10.0, 10.0)

    print(f"  knife_active = {env.solver.knife_active}  (knife ON)")
    print(f"  knife_pos    = {knife_pos_np}  (FAR from tumor)")
    print(f"  Tumor centroid (0.5, 0.5, 0.118) — distance to knife ≈ 16.4")
    print(f"  Forward expectation: wp.where ALL takes p_floor branch")
    print(f"  Cutting block: ALL edges early-return (gate)")
    print()

    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)
    n_substeps = 1

    # ---- Autograd ----
    params = make_param_dict(20.0, 3.0, 0.05, 0.15, requires_grad=True,
                             device=env.device)
    knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                         device=env.device, requires_grad=True)

    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device,
                        requires_grad=True)
    dt_sub = env.solver.dt / float(env.solver.substeps)
    with wp.Tape() as tape:
        differentiable_substep(env.solver, params, knife_pos, dt_sub)
        wp.launch(loss_mean_alpha_at_interior_kernel, dim=env.solver.N,
                  inputs=[env.solver._current_alpha(), interior_mask_wp,
                          interior_count, loss_buf],
                  device=env.device)
    tape.backward(loss=loss_buf)
    wp.synchronize()

    ag_loss = float(loss_buf.numpy()[0])
    ag_grads = {
        "knife_pos_x": float(knife_pos.grad.numpy()[0][0]),
        "knife_pos_y": float(knife_pos.grad.numpy()[0][1]),
        "knife_pos_z": float(knife_pos.grad.numpy()[0][2]),
        "knife_radius": float(params["knife_radius"].grad.numpy()[0]),
        "alpha_min": float(params["alpha_min"].grad.numpy()[0]),
        "sharpness": float(params["sharpness"].grad.numpy()[0]),
        "threshold": float(params["threshold"].grad.numpy()[0]),
    }

    print(f"  Loss: {ag_loss:.6e}  (should be ~1.0, no cutting)")
    print()
    print(f"  {'param':<20} {'autograd grad':>20} {'verdict':>15}")
    print(f"  {'-'*20} {'-'*20} {'-'*15}")

    n_zero = 0
    n_spurious = 0
    spurious_max = 0.0
    for k, v in ag_grads.items():
        is_zero = abs(v) < 1e-9
        verdict = "✓ ZERO" if is_zero else "✗ SPURIOUS"
        if is_zero:
            n_zero += 1
        else:
            n_spurious += 1
            spurious_max = max(spurious_max, abs(v))
        print(f"  {k:<20} {v:>+20.6e} {verdict:>15}")

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    if n_spurious == 0:
        print("  ✅ All gradients = 0 (knife too far, no collision fired)")
        print("  → wp.where untaken-branch backward is CLEAN")
        print("  → The catastrophic gradient happens ONLY when collision actually fires")
        print("  → Bug location: pred_pos R-W-R amplification when collision_delta")
        print("    is non-zero (and cluster_solve subsequently reads pred_pos)")
    else:
        print(f"  ❌ {n_spurious}/7 gradients SPURIOUS even with knife far away")
        print(f"  Max spurious magnitude: {spurious_max:.3e}")
        print()
        print("  → Pattern A CONFIRMED: wp.where backward computes p_knife branch's")
        print("    gradient even when forward never took it. Particles far from knife")
        print("    contribute spurious gradient via knife_radius/safe_dist terms.")
        print()
        print("  Spurious gradients:")
        for k, v in sorted(ag_grads.items(), key=lambda x: -abs(x[1]))[:5]:
            if abs(v) > 1e-9:
                print(f"    {k:<20} {v:+.4e}")


if __name__ == "__main__":
    main()
