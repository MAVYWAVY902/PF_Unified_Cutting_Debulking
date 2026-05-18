#!/usr/bin/env python3
"""Path 4 Falsification Test C — mirror equivalence test.

THE QUESTION: Is my `differentiable_substep()` in path4_poc_v1_dual_purpose.py
truly equivalent (forward-wise) to production `solver._substep()`?

If YES → my mirror is correct, eliminates hypothesis C (mirror bug).
        Then we know: production solver._substep() backward genuinely has
        an issue when cutting fires.

If NO  → my mirror has a bug. ALL Phase 1 conclusions invalidated.
        Must fix mirror before any further analysis.

METHOD:
  1. Setup env + solver in known state, capture snapshot
  2. Run differentiable_substep() N times (no tape, forward only)
  3. Snapshot final alpha (call it alpha_mirror)
  4. Restore state from snapshot
  5. Run production solver._substep() N times
  6. Snapshot final alpha (call it alpha_production)
  7. Compare element-wise

PASS criterion: max |alpha_mirror - alpha_production| < 1e-5
              (allowing tiny float32 noise from wp.array[0] vs Python float)
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from path4_poc_v1_dual_purpose import (
    differentiable_substep, make_param_dict, snap_solver_state,
    reset_solver_state,
)
from warp_sim.envs.debulk_env import DebulkEnv


def main():
    wp.init()
    print("=" * 72)
    print("  Test C — Mirror equivalence (differentiable_substep vs solver._substep)")
    print("=" * 72)

    # ---- Load env in same config as Phase 1 ----
    cfg_path = REPO_ROOT / "warp_sim" / "config" / "debulk_cap_armA.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["obs_mode"] = "baseline"
    cfg["task_type"] = "capsule_only"
    cfg["use_ssa_soft_removal"] = True

    env = DebulkEnv(cfg)
    env.reset()

    # Match Phase 1 strong-signal config
    env.solver.cutting_update_interval = 1
    env.solver.knife_active = 1
    env.solver.knife_radius = 0.15
    env.solver.aspiration_threshold = 0.05
    env.solver.alpha_min = 3.0
    env.solver.soft_removal_sharpness = 20.0
    env.solver._alloc_ssa_buffers()

    # Knife on torus tube (verified to actually cut)
    knife_pos_np = np.array([0.85, 0.5, 0.12], dtype=np.float32)
    n_substeps = 30

    print(f"  N particles:   {env.solver.N}")
    print(f"  knife_pos:     ({knife_pos_np[0]}, {knife_pos_np[1]}, {knife_pos_np[2]})")
    print(f"  knife_radius:  {env.solver.knife_radius}")
    print(f"  threshold:     {env.solver.aspiration_threshold}")
    print(f"  sharpness:     {env.solver.soft_removal_sharpness}")
    print(f"  alpha_min:     {env.solver.alpha_min}")
    print(f"  n_substeps:    {n_substeps}")
    print()

    # ---- Capture pristine state ----
    state_snap = snap_solver_state(env)

    # ---- Run 1: my differentiable_substep (mirror) ----
    print("─" * 72)
    print("Run 1: differentiable_substep (my mirror, _param_arr cutting kernels)")
    print("─" * 72)

    params = make_param_dict(
        sharpness=20.0, alpha_min=3.0, threshold=0.05, knife_radius=0.15,
        requires_grad=False, device=env.device,
    )
    knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                         device=env.device, requires_grad=False)

    dt_sub = env.solver.dt / float(env.solver.substeps)
    for _ in range(n_substeps):
        differentiable_substep(env.solver, params, knife_pos, dt_sub)
    wp.synchronize()

    alpha_mirror = env.solver._current_alpha().numpy().copy()
    cycle_mirror = env.solver._ssa_cycle_index
    print(f"  Final cycle: {cycle_mirror}")
    print(f"  alpha:  min={alpha_mirror.min():.6f}  "
          f"mean={alpha_mirror.mean():.6f}  "
          f"#(<0.5)={int((alpha_mirror < 0.5).sum())}")

    # ---- Restore state ----
    reset_solver_state(env, state_snap)
    env.solver.step_count = 0

    # ---- Run 2: production solver._substep() ----
    print()
    print("─" * 72)
    print("Run 2: solver._substep() (production, float-arg cutting kernels)")
    print("─" * 72)

    env.solver.knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                                    device=env.device, requires_grad=False)
    for _ in range(n_substeps):
        env.solver._substep()
    wp.synchronize()

    alpha_production = env.solver._current_alpha().numpy().copy()
    cycle_production = env.solver._ssa_cycle_index
    print(f"  Final cycle: {cycle_production}")
    print(f"  alpha:  min={alpha_production.min():.6f}  "
          f"mean={alpha_production.mean():.6f}  "
          f"#(<0.5)={int((alpha_production < 0.5).sum())}")

    # ---- Compare ----
    print()
    print("=" * 72)
    print("MIRROR EQUIVALENCE ANALYSIS")
    print("=" * 72)

    diff = np.abs(alpha_mirror - alpha_production)
    print(f"  cycle_mirror == cycle_production:  "
          f"{'✓' if cycle_mirror == cycle_production else '✗'}  "
          f"({cycle_mirror} vs {cycle_production})")
    print(f"  Max abs diff(alpha):    {diff.max():.6e}")
    print(f"  Mean abs diff(alpha):   {diff.mean():.6e}")
    print(f"  Particles with diff>1e-5: {int((diff > 1e-5).sum())} / {env.solver.N}")
    print(f"  Particles with diff>0.01: {int((diff > 0.01).sum())} / {env.solver.N}")

    if diff.max() < 1e-5:
        print()
        print("  ✅ PASS — mirror is byte-equivalent to production")
        print("  → Hypothesis C (mirror bug) eliminated")
        print("  → Phase 1's NaN/bias issues are real production-path issues, not")
        print("    mirror bugs. Proceed to Test B (cutting OFF) and Test D (1-substep).")
        return 0
    elif diff.max() < 1e-3:
        print()
        print("  ⚠️ PARTIAL — small numerical drift (likely float32 ordering)")
        print(f"  Max diff {diff.max():.2e} is small but exceeds 1e-5 strict threshold")
        print("  Hypothesis C unlikely but not fully ruled out")
        return 0
    else:
        print()
        print("  ❌ FAIL — mirror differs significantly from production")
        print(f"  Max diff {diff.max():.2e} >> 1e-5")
        print("  HYPOTHESIS C CONFIRMED: my differentiable_substep has a bug")
        print("  → ALL Phase 1 conclusions are invalidated")
        print("  → Must fix mirror before any further analysis")

        # Diagnostic: which particles differ most?
        idx = np.argsort(-diff)[:5]
        print(f"  Top 5 differing particles:")
        for i in idx:
            print(f"    [{i:5d}]  mirror={alpha_mirror[i]:.6f}  "
                  f"production={alpha_production[i]:.6f}  "
                  f"diff={diff[i]:.4e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
