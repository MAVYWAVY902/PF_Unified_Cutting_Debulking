#!/usr/bin/env python3
"""Phase 2B Day 3 minimal — 1-step no-cutting forward parity.

If split mode and legacy give same per-substep physics, the 5-kernel split is
mathematically equivalent. Diff of 50-step + cutting test was likely from
floating-point ordering compounded by cutting amplification.

This minimal test isolates the kernel-only equivalence:
  - 1 step (no compounding)
  - knife_active = 0 (no cutting)
  - Sweep num_iter [1, 3, 6] to see if Jacobi-iter count affects diff
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


def run_one_step_at_iter(use_split, num_iter, knife_pos_np):
    env = setup_env()
    env.solver.num_iterations = num_iter
    env.solver.use_split_cluster_solve = use_split
    env.solver.use_ssa_soft_removal = True
    env.solver.knife_pos = wp.array([wp.vec3(*knife_pos_np)],
                                       dtype=wp.vec3, device=env.device)
    # NO cutting — knife collision active but no edge damage
    env.solver.knife_active = 1
    env.solver.cutting_update_interval = 100000  # effectively disabled

    env.solver.step()

    pos_final = env.solver.p["pos"].numpy().copy()
    return pos_final


def main():
    wp.init()
    print("=" * 78)
    print("  Phase 2B Day 3 minimal — 1-step no-cutting forward parity")
    print("=" * 78)

    knife_pos_np = (0.85, 0.5, 0.12)

    print(f"  Knife pos: {knife_pos_np} (collision active, NO cutting)")
    print()
    print(f"  {'num_iter':>10} {'max diff':>12} {'mean diff':>12} {'verdict':>10}")

    for num_iter in (1, 3, 6):
        pos_legacy = run_one_step_at_iter(False, num_iter, knife_pos_np)
        pos_split = run_one_step_at_iter(True, num_iter, knife_pos_np)

        diff = np.abs(pos_legacy - pos_split)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        verdict = "✓ match" if max_diff < 1e-5 else (
            "≈ small drift" if max_diff < 1e-3 else "✗ FAIL")

        print(f"  {num_iter:>10} {max_diff:>12.3e} {mean_diff:>12.3e} {verdict:>10}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
