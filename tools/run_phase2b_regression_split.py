#!/usr/bin/env python3
"""Phase 2B Day 5-6: dual-mode regression runner.

Forces every DebulkEnv into split mode (use_split_cluster_solve=True) by
monkey-patching _load_cfg helper.

Pass criterion: same pass rate as Stage B SSA baseline. If any test that
passed in legacy/SSA mode FAILS in split mode, we have a wiring bug.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warp_sim import test_capsule_task as t

_orig_load_cfg = t._load_cfg


def _patched_load_cfg(obs_mode="baseline", **overrides):
    cfg = _orig_load_cfg(obs_mode=obs_mode, **overrides)
    cfg["use_ssa_soft_removal"] = True            # SSA mode required for split mode
    cfg["use_split_cluster_solve"] = True         # Phase 2B 5-kernel split
    return cfg


t._load_cfg = _patched_load_cfg

if __name__ == "__main__":
    print("=" * 72)
    print("  Phase 2B Day 5-6 — Dual-mode regression (split + SSA)")
    print("=" * 72)
    t.main()
