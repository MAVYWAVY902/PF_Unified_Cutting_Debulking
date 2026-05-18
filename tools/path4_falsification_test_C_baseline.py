#!/usr/bin/env python3
"""Control test: production solver._substep() twice, same initial state.

Measures: baseline non-determinism from GPU atomic_add ordering.

If diff is similar to mirror-vs-production diff → atomic_add non-determinism
explains it (mirror is fine).

If diff is much smaller than mirror-vs-production diff → there's a real
mirror bug.
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
    snap_solver_state, reset_solver_state,
)
from warp_sim.envs.debulk_env import DebulkEnv


def main():
    wp.init()
    print("=" * 72)
    print("  Control: production solver._substep() × 2 (atomic_add non-determinism)")
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
    env.solver.knife_active = 1
    env.solver.knife_radius = 0.15
    env.solver.aspiration_threshold = 0.05
    env.solver.alpha_min = 3.0
    env.solver.soft_removal_sharpness = 20.0
    env.solver._alloc_ssa_buffers()

    knife_pos_np = np.array([0.85, 0.5, 0.12], dtype=np.float32)
    n_substeps = 30

    state_snap = snap_solver_state(env)

    # Run A
    env.solver.knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                                    device=env.device)
    for _ in range(n_substeps):
        env.solver._substep()
    wp.synchronize()
    alpha_A = env.solver._current_alpha().numpy().copy()

    # Reset & Run B
    reset_solver_state(env, state_snap)
    env.solver.step_count = 0
    env.solver.knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                                    device=env.device)
    for _ in range(n_substeps):
        env.solver._substep()
    wp.synchronize()
    alpha_B = env.solver._current_alpha().numpy().copy()

    diff = np.abs(alpha_A - alpha_B)
    print(f"  Run A:   min={alpha_A.min():.6f}  mean={alpha_A.mean():.6f}")
    print(f"  Run B:   min={alpha_B.min():.6f}  mean={alpha_B.mean():.6f}")
    print()
    print(f"  Max abs diff:    {diff.max():.6e}")
    print(f"  Mean abs diff:   {diff.mean():.6e}")
    print(f"  Particles diff>1e-5:  {int((diff > 1e-5).sum())} / {env.solver.N}")
    print(f"  Particles diff>0.01:  {int((diff > 0.01).sum())} / {env.solver.N}")

    print()
    if diff.max() < 1e-6:
        print("  ✅ Production is byte-deterministic across runs.")
        print("  → If mirror diff is much larger, mirror has a real bug.")
    else:
        print("  ⚠️ Production has non-determinism across runs (atomic_add ordering).")
        print("  → If mirror-vs-production diff is similar magnitude, mirror is fine.")
        print(f"  → Compare to Test C result (max 7.31e-01, mean 1.10e-03)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
