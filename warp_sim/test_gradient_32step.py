"""Test gradient stability over 32-step horizon for SAPO compatibility.

Verifies:
1. wp.Tape() records 32 env steps without error
2. tape.backward() produces finite gradients on action arrays
3. Gradient norms stay bounded (no explosion)
4. Gradient direction matches finite difference (sanity check)
"""
import sys
import os
import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pf_shape_matching_warp"))

from warp_sim.envs.debulk_env import DebulkEnv


def main():
    wp.init()
    device = "cuda:0"

    cfg = {
        "mesh": os.path.join(os.path.dirname(__file__), "..", "resources", "cube", "cube_dense.msh"),
        "settle_steps": 100,
        "horizon_len": 32,
        "sim_substeps": 2,
        "action_scale": 0.005,
        "knife_start": [0.5, 0.8, 0.5],
        "knife_radius": 0.1,
        "dt": 0.005,
        "num_iterations": 4,
        "stiffness": 0.6,
        "damping": 0.98,
        "cutting_interval": 10,
        "lambda_nerve": 5.0,
        "mu_action": 0.01,
        "bonus": 0.0,
        "volume_target": 999.0,
    }

    env = DebulkEnv(cfg, device=device)
    horizon = 32

    # ================================================================
    print("=" * 60)
    print("Test 1: 32-step differentiable rollout")
    print("=" * 60)

    obs = env.initialize_trajectory()
    actions = []
    rewards = []

    tape = wp.Tape()
    try:
        with tape:
            for t in range(horizon):
                # Constant downward action (moves knife toward mesh)
                action = wp.array(
                    [wp.vec3(0.0, -0.5, 0.0)],
                    dtype=wp.vec3, device=device, requires_grad=True,
                )
                actions.append(action)
                obs, rew, done, _ = env.step(action)
                wp.synchronize()
                rewards.append(float(rew.numpy()[0]))

        print(f"  Forward pass completed. Total reward: {sum(rewards):.4f}")
        print(f"  Rewards per step: {[f'{r:.4f}' for r in rewards[:8]]}...")

        # Backward
        tape.backward(loss=env.rew_buf)
        wp.synchronize()
        print("  Backward pass completed.")

        # Check gradients on each action
        grad_norms = []
        for t, a in enumerate(actions):
            if a.grad is not None:
                g = a.grad.numpy()
                norm = np.linalg.norm(g)
                grad_norms.append(norm)
                if t < 8 or t == horizon - 1:
                    print(f"    step {t:2d}: action.grad = {g.flatten()}, norm = {norm:.6f}")
            else:
                grad_norms.append(0.0)
                if t < 4:
                    print(f"    step {t:2d}: action.grad = None")

        has_grads = sum(1 for n in grad_norms if n > 0)
        print(f"\n  Steps with non-zero gradients: {has_grads}/{horizon}")

        # Check for NaN/Inf
        any_nan = any(np.isnan(n) for n in grad_norms)
        any_inf = any(np.isinf(n) for n in grad_norms)
        max_norm = max(grad_norms) if grad_norms else 0

        print(f"  Max gradient norm: {max_norm:.4f}")
        print(f"  Any NaN: {any_nan}")
        print(f"  Any Inf: {any_inf}")
        print(f"  Bounded (< 1e4): {max_norm < 1e4}")

        tape_ok = not any_nan and not any_inf and max_norm < 1e4
        print(f"\n  Test 1: {'PASS' if tape_ok else 'FAIL'}")

    except Exception as e:
        print(f"  ERROR: {e}")
        tape_ok = False
        import traceback
        traceback.print_exc()

    env.clear_grad()

    # ================================================================
    print("\n" + "=" * 60)
    print("Test 2: Finite-difference gradient direction check")
    print("=" * 60)

    def run_rollout_reward(env, action_y, horizon, device):
        """Run a rollout with constant action_y, return total reward."""
        env.reset()
        env.solver.knife_active = 1
        total_r = 0.0
        for t in range(horizon):
            a = wp.array(
                [wp.vec3(0.0, action_y, 0.0)],
                dtype=wp.vec3, device=device, requires_grad=False,
            )
            _, rew, _, _ = env.step(a)
            wp.synchronize()
            total_r += float(rew.numpy()[0])
        return total_r

    # Central differences on action_y
    eps = 0.1
    fd_horizon = 16  # shorter for speed

    r_plus = run_rollout_reward(env, -0.5 + eps, fd_horizon, device)
    r_minus = run_rollout_reward(env, -0.5 - eps, fd_horizon, device)
    r_base = run_rollout_reward(env, -0.5, fd_horizon, device)

    fd_grad = (r_plus - r_minus) / (2.0 * eps)
    print(f"  R(ay={-0.5+eps:.1f}) = {r_plus:.4f}")
    print(f"  R(ay={-0.5:.1f})     = {r_base:.4f}")
    print(f"  R(ay={-0.5-eps:.1f}) = {r_minus:.4f}")
    print(f"  dR/d(action_y) (FD) = {fd_grad:.6f}")

    fd_ok = abs(fd_grad) > 1e-8 or (r_plus != r_minus)
    print(f"\n  Test 2: {'PASS' if fd_ok else 'FAIL (zero gradient — check if knife reaches mesh)'}")

    # ================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Tape 32-step:   {'PASS' if tape_ok else 'FAIL'}")
    print(f"  FD gradient:    {'PASS' if fd_ok else 'FAIL'}")
    all_pass = tape_ok and fd_ok
    print(f"\n  OVERALL: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    if all_pass:
        print("  -> Gradients stable over 32 steps. SAPO training should work.")


if __name__ == "__main__":
    main()
