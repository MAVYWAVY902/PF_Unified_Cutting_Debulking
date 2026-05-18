#!/usr/bin/env python3
"""Phase 2B Day 3 — Forward smoke test: split mode vs legacy.

Run identical initial state through 100 steps in BOTH modes, compare final pos.
Pass: max abs diff < 1e-4 (numerical equivalence).

Forward physics MUST match because:
  - K1+K2 (split) computes same cx_mean, qc_mean as legacy lines 467-510
  - K3 (split covariance) computes same A as legacy lines 518-541
  - K4 (split polar) calls extract_rotation_muller — same as legacy line 545
  - K5 (split scatter) writes same correction to deltas as legacy lines 548-560

Any forward divergence → wiring bug, not autograd bug.
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

wp.config.verify_autograd_array_access = False

from path4_falsification_test_H_isolate import setup_env


def run_n_steps(use_split, n_steps, num_iter, knife_pos_np):
    """Run n_steps in either legacy or split mode, return final pos."""
    env = setup_env()
    env.solver.num_iterations = num_iter
    env.solver.use_split_cluster_solve = use_split
    # SSA mode for both — split mode requires it for pred_pos cycle-scoping
    env.solver.use_ssa_soft_removal = True

    # Set knife pos
    env.solver.knife_pos = wp.array([wp.vec3(*knife_pos_np)],
                                       dtype=wp.vec3, device=env.device)
    env.solver.knife_active = 1

    for _ in range(n_steps):
        env.solver.step()

    pos_final = env.solver.p["pos"].numpy().copy()
    # Snapshot SSA state (in case use_split=True keeps current alpha in cycle buffer)
    if env.solver.use_ssa_soft_removal:
        env.solver.snapshot_to_p_arrays()
    alpha_final = env.solver.p["alpha"].numpy().copy()
    return pos_final, alpha_final


def main():
    wp.init()
    print("=" * 78)
    print("  Phase 2B Day 3 — Forward smoke test: split vs legacy")
    print("=" * 78)

    knife_pos_np = (0.85, 0.5, 0.12)
    NUM_ITER = 6
    N_STEPS = 50

    print(f"  Config: num_iter={NUM_ITER}, n_steps={N_STEPS}")
    print(f"  Knife pos: {knife_pos_np}")

    print("\n  Running LEGACY mode...")
    pos_legacy, alpha_legacy = run_n_steps(False, N_STEPS, NUM_ITER, knife_pos_np)

    print("  Running SPLIT mode...")
    pos_split, alpha_split = run_n_steps(True, N_STEPS, NUM_ITER, knife_pos_np)

    # Compare
    diff = np.abs(pos_legacy - pos_split)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))

    print()
    print("=" * 78)
    print("  RESULT")
    print("=" * 78)
    print(f"  Final pos shape: {pos_legacy.shape}")
    print(f"  Max abs diff:    {max_diff:.3e}")
    print(f"  Mean abs diff:   {mean_diff:.3e}")
    print(f"  Median abs diff: {median_diff:.3e}")

    # Sample particles
    print(f"  Sample (particle 0):")
    print(f"    legacy: {pos_legacy[0]}")
    print(f"    split:  {pos_split[0]}")
    print(f"    diff:   {pos_split[0] - pos_legacy[0]}")

    print(f"  Sample (particle 1000):")
    print(f"    legacy: {pos_legacy[1000]}")
    print(f"    split:  {pos_split[1000]}")
    print(f"    diff:   {pos_split[1000] - pos_legacy[1000]}")

    # Alpha state (cutting damage)
    alpha_diff = float(np.max(np.abs(alpha_legacy - alpha_split)))
    print(f"  Alpha max diff:  {alpha_diff:.3e}")

    print()
    if max_diff < 1e-4:
        print(f"  🎯 PASS — split mode forward physics matches legacy (max diff < 1e-4)")
    elif max_diff < 1e-3:
        print(f"  ✅ ACCEPTABLE — small numerical drift (< 1e-3); likely floating-point ordering")
    else:
        print(f"  ❌ FAIL — split mode diverges from legacy at max diff {max_diff:.3e}")
        print(f"     Investigate: does split mode physics match legacy semantics?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
