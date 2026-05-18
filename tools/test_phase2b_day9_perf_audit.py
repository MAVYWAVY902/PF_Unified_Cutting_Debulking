#!/usr/bin/env python3
"""Phase 2B Day 9-10 — Memory + perf audit.

Profile split mode vs legacy mode:
  - Forward runtime per step (100 steps timed)
  - GPU memory delta during 1 step + tape recording

Targets:
  - Split mode runtime ≤ 2× legacy (extra kernel launches, but each simpler)
  - Memory ≤ 4 GB peak on RTX 2070 (8 GB total, leave headroom)
"""
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

wp.config.verify_autograd_array_access = False

from path4_falsification_test_H_isolate import setup_env


def profile_mode(use_split, n_steps=100, num_iter=6, knife_pos_np=(0.85, 0.5, 0.12)):
    env = setup_env()
    env.solver.num_iterations = num_iter
    env.solver.use_split_cluster_solve = use_split
    env.solver.use_ssa_soft_removal = True
    env.solver.knife_pos = wp.array([wp.vec3(*knife_pos_np)],
                                       dtype=wp.vec3, device=env.device)
    env.solver.knife_active = 1
    env.solver.cutting_update_interval = 50

    # Warmup
    for _ in range(3):
        env.solver.step()
    wp.synchronize()

    # GPU memory before steps
    free_before, total = wp.get_device("cuda:0").free_memory, wp.get_device("cuda:0").total_memory
    used_before = (total - free_before) / 1024 / 1024  # MB

    # Time n_steps
    t0 = time.time()
    for _ in range(n_steps):
        env.solver.step()
    wp.synchronize()
    elapsed = time.time() - t0

    free_after = wp.get_device("cuda:0").free_memory
    used_after = (total - free_after) / 1024 / 1024
    delta_mem = used_after - used_before

    return dict(
        elapsed_s=elapsed,
        ms_per_step=elapsed / n_steps * 1000,
        used_before_mb=used_before,
        used_after_mb=used_after,
        delta_mem_mb=delta_mem,
    )


def main():
    wp.init()
    print("=" * 78)
    print("  Phase 2B Day 9-10 — Memory + perf audit")
    print("=" * 78)

    print("\n  Profiling LEGACY mode...")
    r_legacy = profile_mode(False)
    print(f"    {r_legacy['ms_per_step']:.2f} ms/step ({r_legacy['elapsed_s']:.1f} s for 100 steps)")
    print(f"    GPU mem: {r_legacy['used_before_mb']:.0f} → {r_legacy['used_after_mb']:.0f} MB (delta {r_legacy['delta_mem_mb']:+.1f})")

    print("\n  Profiling SPLIT mode...")
    r_split = profile_mode(True)
    print(f"    {r_split['ms_per_step']:.2f} ms/step ({r_split['elapsed_s']:.1f} s for 100 steps)")
    print(f"    GPU mem: {r_split['used_before_mb']:.0f} → {r_split['used_after_mb']:.0f} MB (delta {r_split['delta_mem_mb']:+.1f})")

    print()
    print("=" * 78)
    print("  RESULT")
    print("=" * 78)
    speedup = r_legacy['ms_per_step'] / r_split['ms_per_step']
    overhead = r_split['ms_per_step'] / r_legacy['ms_per_step']
    print(f"  Runtime:  legacy {r_legacy['ms_per_step']:.2f} ms/step  vs  split {r_split['ms_per_step']:.2f} ms/step")
    print(f"            split overhead = {overhead:.2f}× legacy")
    print(f"  Memory:   legacy delta {r_legacy['delta_mem_mb']:+.1f} MB  vs  split delta {r_split['delta_mem_mb']:+.1f} MB")
    print()

    # Targets
    if overhead <= 2.0:
        print(f"  ✅ Runtime PASS (split {overhead:.2f}× legacy ≤ 2× target)")
    else:
        print(f"  ⚠️ Runtime ABOVE target: split {overhead:.2f}× legacy > 2×")
    if r_split['used_after_mb'] < 4096:
        print(f"  ✅ Memory PASS (split peak {r_split['used_after_mb']:.0f} MB < 4 GB target)")
    else:
        print(f"  ⚠️ Memory above target: split peak {r_split['used_after_mb']:.0f} MB ≥ 4 GB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
