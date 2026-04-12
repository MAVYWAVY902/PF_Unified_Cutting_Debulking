"""Smoke test for DebulkEnv (Rewarped-style): verify interface, shapes, reset."""
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
        "action_scale": 0.01,
        "knife_start": [0.5, 0.8, 0.5],
        "knife_radius": 0.1,
        "dt": 0.005,
        "num_iterations": 4,
        "stiffness": 0.6,
        "damping": 0.98,
        "cutting_interval": 10,
        "lambda_nerve": 5.0,
        "episode_length": 300,
    }

    print("=" * 50)
    print("Test 1: DebulkEnv initialization")
    print("=" * 50)
    env = DebulkEnv(cfg, device=device)
    print(f"  num_obs={env.num_obs}, num_act={env.num_act}, N_SAMPLE={env.N_SAMPLE}")
    assert env.num_obs == env.N_SAMPLE * 4 + 3, f"Expected {env.N_SAMPLE*4+3}, got {env.num_obs}"
    assert env.num_act == 3
    print("  PASS")

    print("\n" + "=" * 50)
    print("Test 2: initialize_trajectory() returns dict obs")
    print("=" * 50)
    obs_dict = env.initialize_trajectory()
    wp.synchronize()

    assert "particle_q" in obs_dict, "Missing particle_q"
    assert "particle_alpha" in obs_dict, "Missing particle_alpha"
    assert "knife_pos" in obs_dict, "Missing knife_pos"

    pq = obs_dict["particle_q"].numpy()
    pa = obs_dict["particle_alpha"].numpy()
    kp = obs_dict["knife_pos"].numpy()
    print(f"  particle_q shape: {pq.shape} (expect ({env.N_SAMPLE}, 3))")
    print(f"  particle_alpha shape: {pa.shape} (expect ({env.N_SAMPLE},))")
    print(f"  knife_pos shape: {kp.shape}")
    print(f"  alpha range: [{pa.min():.3f}, {pa.max():.3f}]")
    assert pq.shape == (env.N_SAMPLE, 3)
    assert pa.shape == (env.N_SAMPLE,)

    # Test flat conversion
    flat = env.get_obs_flat_np(obs_dict)
    print(f"  flat obs shape: {flat.shape} (expect ({env.num_obs},))")
    assert flat.shape == (env.num_obs,)
    print("  PASS")

    print("\n" + "=" * 50)
    print("Test 3: step() loop (32 steps)")
    print("=" * 50)
    rewards = []
    for t in range(32):
        action_np = np.random.randn(3).astype(np.float32) * 0.1
        action_wp = wp.array(
            [wp.vec3(action_np[0], action_np[1], action_np[2])],
            dtype=wp.vec3, device=device, requires_grad=True,
        )
        obs_dict, rew, done, _ = env.step(action_wp)
        wp.synchronize()
        r = float(rew.numpy()[0])
        rewards.append(r)
        if t % 8 == 0:
            flat = env.get_obs_flat_np(obs_dict)
            print(f"  step {t:2d}: reward={r:+.4f}, flat_obs[:5]={flat[:5]}")

    print(f"  Total reward: {sum(rewards):.4f}")
    print("  PASS (no crash)")

    print("\n" + "=" * 50)
    print("Test 4: reset() restores state")
    print("=" * 50)
    pos_after = env.solver.p["pos"].numpy().copy()
    env.reset()
    wp.synchronize()
    pos_reset = env.solver.p["pos"].numpy()
    diff = np.max(np.abs(pos_reset - env._snap_pos.numpy()))
    print(f"  Max pos diff after reset: {diff:.6f} (should be ~0)")
    assert diff < 1e-5
    print("  PASS")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
