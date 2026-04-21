"""Does finite-difference gradient give non-zero when Warp autograd fails?

FD uses pure forward passes (no tape), so immune to Warp adjoint bugs.
Tries multiple eps values to find threshold where damage change is measurable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pf_shape_matching_warp"))

import warp as wp
wp.init()
import yaml, numpy as np

from warp_sim.envs.debulk_env import DebulkEnv


def main():
    cfg = yaml.safe_load(open(
        os.path.join(os.path.dirname(__file__), "config", "debulk_cap_armA.yaml")))
    cfg['obs_mode'] = 'finite_diff'  # uses _compute_tactile_finite_diff
    env = DebulkEnv(cfg)
    env.reset()
    # Move knife to interior where contact definitely happens
    for _ in range(3):
        env.step(np.array([0.5, -0.5, 0.5], dtype=np.float32))

    print(f"\n=== FD GRADIENT SWEEP ===", flush=True)
    print(f"  knife_pos: {env.solver.knife_pos.numpy().reshape(3)}", flush=True)

    # Try different eps values
    for eps in [0.001, 0.01, 0.1, 0.3, 1.0]:
        env._fd_epsilon = eps
        g = env._compute_tactile_finite_diff()
        print(f"  eps={eps:6.3f}  g={g}  |g|={np.linalg.norm(g):.6e}", flush=True)

    print(f"\n=== ALSO TRY: knife deep in interior ===", flush=True)
    env.solver.knife_pos = wp.array([wp.vec3(0.5, 0.25, 0.5)], dtype=wp.vec3,
                                     device=env.device, requires_grad=True)
    for eps in [0.01, 0.1, 0.3, 1.0]:
        env._fd_epsilon = eps
        g = env._compute_tactile_finite_diff()
        print(f"  eps={eps:6.3f}  g={g}  |g|={np.linalg.norm(g):.6e}", flush=True)


if __name__ == "__main__":
    main()
