#!/usr/bin/env python3
"""Stage B Day 4 Phase 2: SSA-mode regression runner.

Wraps warp_sim/test_capsule_task.py and forces every DebulkEnv into SSA mode
(use_ssa_soft_removal=True) by monkey-patching the _load_cfg helper.

This avoids polluting the test file. If 38/38 still pass with SSA on, then
Stage B forward physics is verified-equivalent end-to-end. If something fails,
we'll know exactly which subsystem (m3_4 L2 tape interaction is the prime
suspect — its snapshot/restore wasn't designed for cycle-scoped buffer lists).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the test module then patch its _load_cfg before running.
from warp_sim import test_capsule_task as t

_orig_load_cfg = t._load_cfg


def _patched_load_cfg(obs_mode="baseline", **overrides):
    cfg = _orig_load_cfg(obs_mode=obs_mode, **overrides)
    cfg["use_ssa_soft_removal"] = True
    return cfg


t._load_cfg = _patched_load_cfg

if __name__ == "__main__":
    print("=" * 72)
    print("  Stage B Day 4 Phase 2 — SSA-mode regression (use_ssa_soft_removal=True)")
    print("=" * 72)
    t.main()
