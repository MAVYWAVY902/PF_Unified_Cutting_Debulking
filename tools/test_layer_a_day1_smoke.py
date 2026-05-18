#!/usr/bin/env python3
"""Day 1 exit gate smoke test for Layer A SSA refactor.

Verifies:
  (1) Imports work — apply_shape_delta_ssa_kernel + apply_collision_delta_ssa_kernel
  (2) legacy mode (use_ssa_soft_removal=False) runs 1 substep without crash
  (3) SSA   mode (use_ssa_soft_removal=True)  runs 1 substep without crash
  (4) Both modes produce non-NaN pred_pos / pos / vel after 1 substep

Day 1 does NOT verify gradient correctness — that's Day 2's G1/G2.
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

wp.config.verify_autograd_array_access = False  # Day 2 will turn this on

from warp_sim.envs.debulk_env import DebulkEnv


def setup_env(use_ssa: bool):
    cfg_path = REPO_ROOT / "warp_sim" / "config" / "debulk_cap_armA.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["obs_mode"] = "baseline"
    cfg["task_type"] = "capsule_only"
    cfg["use_ssa_soft_removal"] = use_ssa

    env = DebulkEnv(cfg)
    env.reset()
    env.solver.knife_active = 1
    env.solver.knife_radius = 0.15
    env.solver.set_knife_pos(0.85, 0.5, 0.12)  # on torus tube
    return env


def run_one_substep(env):
    env.solver._substep()
    pred_pos = env.solver.p["pred_pos"].numpy()
    pos      = env.solver.p["pos"].numpy()
    vel      = env.solver.p["vel"].numpy()
    return pred_pos, pos, vel


def check_finite(name, arr):
    if not np.all(np.isfinite(arr)):
        n_nan = int(np.sum(~np.isfinite(arr)))
        raise RuntimeError(f"{name}: {n_nan}/{arr.size} non-finite values")
    return arr.shape, float(np.linalg.norm(arr))


def main():
    print("=" * 60)
    print("Day 1 Smoke Test — Layer A SSA refactor")
    print("=" * 60)

    print("\n[1/3] Legacy mode (use_ssa_soft_removal=False)...")
    env_legacy = setup_env(use_ssa=False)
    pred_pos_L, pos_L, vel_L = run_one_substep(env_legacy)
    for name, arr in [("pred_pos", pred_pos_L), ("pos", pos_L), ("vel", vel_L)]:
        shape, norm = check_finite(name, arr)
        print(f"  {name:8s} shape={shape}  ‖x‖={norm:.6f}")

    print("\n[2/3] SSA mode (use_ssa_soft_removal=True)...")
    env_ssa = setup_env(use_ssa=True)
    pred_pos_S, pos_S, vel_S = run_one_substep(env_ssa)
    for name, arr in [("pred_pos", pred_pos_S), ("pos", pos_S), ("vel", vel_S)]:
        shape, norm = check_finite(name, arr)
        print(f"  {name:8s} shape={shape}  ‖x‖={norm:.6f}")

    print("\n[3/3] Cross-mode forward output diff (should be ~GPU noise)...")
    diff_pred = np.linalg.norm(pred_pos_L - pred_pos_S)
    diff_pos  = np.linalg.norm(pos_L - pos_S)
    diff_vel  = np.linalg.norm(vel_L - vel_S)
    print(f"  ‖pred_pos_L - pred_pos_S‖ = {diff_pred:.6e}")
    print(f"  ‖pos_L - pos_S‖           = {diff_pos:.6e}")
    print(f"  ‖vel_L - vel_S‖           = {diff_vel:.6e}")
    print("  (Both modes do same physics. Diff should be ~atomic_add nondet noise.)")

    print("\nDay 1 exit gate: PASS")
    print("  (1) Imports OK")
    print("  (2) Legacy mode 1-substep no crash")
    print("  (3) SSA mode 1-substep no crash")
    print("  (4) All outputs finite")


if __name__ == "__main__":
    main()
