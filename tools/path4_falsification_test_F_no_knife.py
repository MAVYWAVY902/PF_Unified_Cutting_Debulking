#!/usr/bin/env python3
"""Path 4 Falsification Test F — knife_active=0.

QUESTION: Is the n=1 catastrophic gradient on knife_pos due to:
  (a) Layer A backward bug (independent of cutting), OR
  (b) Cutting-driven bias (only when knife actively interacts)

When knife_active=0:
  - Layer A: collision_kernel_ssa skips knife branch (just floor)
            → knife_pos / knife_radius are NOT READ by any kernel
  - Layer B: cutting_block doesn't fire (gated by knife_active==1)
            → alpha_min/sharpness/threshold are NOT READ by any kernel

Expected with knife_active=0:
  - All params' gradients should be EXACTLY 0
  - FD should agree (loss(p+ε) == loss(p-ε) since p doesn't affect anything)

If autograd gives nonzero gradient on any param → spurious gradient = bug
If autograd gives 0 on all params → backward is at least correct in this trivial case
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
    print("  Test F — knife_active=0 (cutting OFF, knife doesn't touch anything)")
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
    env.solver.knife_active = 0   # ★ KNIFE OFF
    env.solver.knife_radius = 0.15
    env.solver.aspiration_threshold = 0.05
    env.solver.alpha_min = 3.0
    env.solver.soft_removal_sharpness = 20.0
    env.solver._alloc_ssa_buffers()

    print(f"  knife_active = {env.solver.knife_active}  (knife OFF)")
    print(f"  Expected: ALL gradients should be 0 (params not read)")
    print()

    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)

    knife_pos_np = (0.85, 0.5, 0.12)
    n_substeps = 1

    state_snap = snap_solver_state(env)

    # ---- Autograd at n=1, knife_active=0 ----
    print("[1] Autograd backward (knife OFF, n=1)...")
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

    print(f"  Loss: {ag_loss:.6e}  (should be 1.0 since no cutting)")
    print()
    print(f"  {'param':<20} {'autograd grad':>20} {'expected':>10} {'verdict':>15}")
    print(f"  {'-'*20} {'-'*20} {'-'*10} {'-'*15}")

    n_zero = 0
    n_spurious = 0
    for k, v in ag_grads.items():
        is_zero = abs(v) < 1e-9
        verdict = "✓ ZERO" if is_zero else "✗ SPURIOUS"
        if is_zero:
            n_zero += 1
        else:
            n_spurious += 1
        print(f"  {k:<20} {v:>+20.6e} {0.0:>10.0f} {verdict:>15}")

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    if n_spurious == 0:
        print("  ✅ ALL gradients are 0 (as expected when knife OFF)")
        print("  → Layer A backward is at least correct in trivial case")
        print("  → The catastrophic n=1 gradient WITH cutting is CUTTING-DRIVEN")
        print("  → Bug location: cutting block backward path or its interaction")
        print("    with Layer A pred_pos updates")
    else:
        print(f"  ❌ {n_spurious}/7 gradients are NONZERO when params don't enter computation")
        print("  → SPURIOUS GRADIENTS — autograd backward has a bug INDEPENDENT")
        print("    of cutting block. The bug is in Layer A physics path itself.")
        print()
        print("  Top spurious magnitudes:")
        sorted_grads = sorted(ag_grads.items(), key=lambda x: -abs(x[1]))
        for k, v in sorted_grads[:5]:
            if abs(v) > 1e-9:
                print(f"    {k:<20} {v:+.4e}")


if __name__ == "__main__":
    main()
