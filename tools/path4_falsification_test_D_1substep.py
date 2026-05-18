#!/usr/bin/env python3
"""Path 4 Falsification Test D — 1-substep backward.

QUESTION: Is the NaN gradient (seen with 30 substeps + cutting) due to
multi-substep accumulation, or already present at 1 substep?

If 1 substep gives NaN → single-substep backward is broken (kernel-level issue)
If 1 substep gives finite + sensible → multi-substep accumulation explodes NaN
                                       (long-horizon stability issue)

This isolates "per-substep correctness" from "horizon accumulation".
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

wp.config.verify_autograd_array_access = True

from path4_poc_v1_dual_purpose import (
    differentiable_substep, make_param_dict, make_knife_pos,
    snap_solver_state, reset_solver_state,
    loss_mean_alpha_at_interior_kernel,
)
from warp_sim.envs.debulk_env import DebulkEnv


def main():
    wp.init()
    print("=" * 72)
    print("  Test D — 1-substep backward (isolate per-substep vs accumulation)")
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

    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)

    knife_pos_np = (0.85, 0.5, 0.12)

    # ---- Test for n_substeps in [1, 3, 10, 30] ----
    for n_substeps in [1, 3, 10, 30]:
        print()
        print("─" * 72)
        print(f"  n_substeps = {n_substeps}")
        print("─" * 72)

        # Reset state for each test
        env.reset()
        env.solver.cutting_update_interval = 1
        env.solver.knife_active = 1
        env.solver.knife_radius = 0.15
        env.solver.aspiration_threshold = 0.05
        env.solver.alpha_min = 3.0
        env.solver.soft_removal_sharpness = 20.0
        env.solver._alloc_ssa_buffers()

        params = make_param_dict(
            sharpness=20.0, alpha_min=3.0, threshold=0.05, knife_radius=0.15,
            requires_grad=True, device=env.device,
        )
        knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                             device=env.device, requires_grad=True)

        loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device,
                            requires_grad=True)
        dt_sub = env.solver.dt / float(env.solver.substeps)

        try:
            with wp.Tape() as tape:
                for _ in range(n_substeps):
                    differentiable_substep(env.solver, params, knife_pos, dt_sub)
                final_alpha = env.solver._current_alpha()
                wp.launch(loss_mean_alpha_at_interior_kernel, dim=env.solver.N,
                          inputs=[final_alpha, interior_mask_wp,
                                  interior_count, loss_buf],
                          device=env.device)

            tape.backward(loss=loss_buf)
            wp.synchronize()

            loss_val = float(loss_buf.numpy()[0])
            alpha_np = final_alpha.numpy()
            n_dead = int((alpha_np < 0.5).sum())

            grads = {
                "knife_pos": tuple(float(x) for x in knife_pos.grad.numpy()[0])
                              if knife_pos.grad is not None else None,
                "knife_radius": float(params["knife_radius"].grad.numpy()[0])
                                 if params["knife_radius"].grad is not None else None,
                "alpha_min": float(params["alpha_min"].grad.numpy()[0])
                              if params["alpha_min"].grad is not None else None,
                "sharpness": float(params["sharpness"].grad.numpy()[0])
                              if params["sharpness"].grad is not None else None,
                "threshold": float(params["threshold"].grad.numpy()[0])
                              if params["threshold"].grad is not None else None,
            }

            print(f"  Forward loss: {loss_val:.6e}  (#dead={n_dead})")
            print(f"  Gradients:")
            all_finite = True
            any_nonzero = False
            for k, v in grads.items():
                if v is None:
                    print(f"    {k:<15}  ❌ None")
                    all_finite = False
                elif isinstance(v, tuple):
                    finite = all(np.isfinite(c) for c in v)
                    nonzero = any(abs(c) > 1e-12 for c in v)
                    print(f"    {k:<15}  ({v[0]:+.3e}, {v[1]:+.3e}, {v[2]:+.3e})  "
                          f"{'finite' if finite else 'NaN/inf'}  "
                          f"{'nonzero' if nonzero else 'all-zero'}")
                    if not finite:
                        all_finite = False
                    if nonzero:
                        any_nonzero = True
                else:
                    finite = np.isfinite(v)
                    nonzero = abs(v) > 1e-12
                    print(f"    {k:<15}  {v:+.3e}  "
                          f"{'finite' if finite else 'NaN/inf'}  "
                          f"{'nonzero' if nonzero else 'zero'}")
                    if not finite:
                        all_finite = False
                    if nonzero:
                        any_nonzero = True

            if all_finite and any_nonzero:
                verdict = "✅ FINITE + INFORMATIVE"
            elif all_finite:
                verdict = "⚠️ FINITE BUT ALL ZERO (silent zero?)"
            else:
                verdict = "❌ NaN/inf"
            print(f"  Verdict: {verdict}")

        except Exception as e:
            print(f"  ❌ EXCEPTION: {type(e).__name__}: {e}")

    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print("  - If 1-substep is finite + nonzero, but 30-substep is NaN:")
    print("    → multi-substep gradient accumulation EXPLODES")
    print("    → likely numerical, fix with shorter horizon or gradient clipping")
    print("  - If 1-substep is already NaN:")
    print("    → per-substep backward broken (kernel-level R-W-R or numerical)")
    print("    → must fix at kernel level")
    print("  - If 1-substep is finite but ZERO (silent zero):")
    print("    → cutting-block-only path differs from full pipeline")
    print("    → suggests Layer A backward chain is broken at the entry")


if __name__ == "__main__":
    main()
