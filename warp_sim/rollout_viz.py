"""Rollout a trained policy and visualize particle deformation + tool trajectory.

Outputs:
  1. Per-frame PNG images in rollout_frames/
  2. MP4 video (if ffmpeg available)
  3. Trajectory data (knife positions, rewards) as .npz

Usage:
    python warp_sim/rollout_viz.py [--model ppo_vec_debulk_final] [--frames 128]
"""
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pf_shape_matching_warp"))

import warp as wp
wp.init()

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from stable_baselines3 import PPO
from warp_sim.envs.debulk_env import DebulkEnv


def rollout_and_save(model_path, cfg, num_steps=128, out_dir="rollout_frames"):
    """Run policy, collect data, render frames."""

    os.makedirs(out_dir, exist_ok=True)

    env = DebulkEnv(cfg)
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    wp.synchronize()

    # Collect trajectory data
    knife_positions = []
    rewards = []
    all_pos = []
    all_alpha = []

    # Initial state
    pos0 = env.solver.p["pos"].numpy().copy()
    rest_pos = env.rest_pos.numpy().copy()
    N = env.solver.N

    print(f"Rolling out {num_steps} steps...")

    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        wp.synchronize()

        # Collect data
        kp = env.solver.knife_pos.numpy().reshape(3)
        pos = env.solver.p["pos"].numpy().copy()
        alpha = env.solver.p["alpha"].numpy().copy()

        knife_positions.append(kp.copy())
        rewards.append(float(r))
        all_pos.append(pos)
        all_alpha.append(alpha)

        if step % 10 == 0:
            print(f"  step {step:3d}: reward={r:.4f}, knife=({kp[0]:.3f},{kp[1]:.3f},{kp[2]:.3f})")

        if term or trunc:
            break

    knife_positions = np.array(knife_positions)
    rewards = np.array(rewards)

    print(f"\nTotal reward: {rewards.sum():.2f}")
    print(f"Knife travel: {np.linalg.norm(np.diff(knife_positions, axis=0), axis=1).sum():.4f}")

    # Save trajectory data
    np.savez(os.path.join(out_dir, "trajectory.npz"),
             knife_positions=knife_positions,
             rewards=rewards,
             rest_pos=rest_pos[:, :3] if rest_pos.ndim > 1 else rest_pos)

    # Render frames
    print(f"\nRendering {len(all_pos)} frames to {out_dir}/...")

    # Compute bounds from initial positions
    pos_min = rest_pos.reshape(-1, 3).min(axis=0) - 0.1
    pos_max = rest_pos.reshape(-1, 3).max(axis=0) + 0.1

    # Subsample particles for visualization (too many = slow)
    viz_stride = max(1, N // 2000)
    viz_indices = np.arange(0, N, viz_stride)

    # Get nerve mask for coloring
    nerve_mask = env.nerve_mask.numpy() if hasattr(env, 'nerve_mask') else np.zeros(N)

    for frame_idx in range(len(all_pos)):
        pos = all_pos[frame_idx].reshape(-1, 3)
        alpha = all_alpha[frame_idx]
        kp = knife_positions[frame_idx]

        # Subsample
        pos_sub = pos[viz_indices]
        alpha_sub = alpha[viz_indices]
        nerve_sub = nerve_mask[viz_indices]

        fig = plt.figure(figsize=(12, 5))

        # --- Left: 3D scatter of particles ---
        ax1 = fig.add_subplot(121, projection='3d')

        # Color by: nerve=red, tumor(alive)=blue, tumor(damaged)=orange
        colors = np.zeros((len(pos_sub), 4))
        for i in range(len(pos_sub)):
            if nerve_sub[i] == 1:
                colors[i] = [1.0, 0.2, 0.2, 0.8]  # red = nerve
            elif alpha_sub[i] < 0.5:
                colors[i] = [1.0, 0.6, 0.0, 0.5]  # orange = damaged
            else:
                colors[i] = [0.2, 0.5, 1.0, 0.6]  # blue = healthy tumor

        ax1.scatter(pos_sub[:, 0], pos_sub[:, 2], pos_sub[:, 1],
                    c=colors, s=3, depthshade=True)

        # Draw knife as sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        kr = cfg.get("knife_radius", 0.15)
        sx = kp[0] + kr * np.outer(np.cos(u), np.sin(v))
        sy = kp[2] + kr * np.outer(np.sin(u), np.sin(v))
        sz = kp[1] + kr * np.outer(np.ones_like(u), np.cos(v))
        ax1.plot_surface(sx, sy, sz, color='green', alpha=0.3)

        # Draw knife trajectory so far
        if frame_idx > 0:
            traj = knife_positions[:frame_idx + 1]
            ax1.plot(traj[:, 0], traj[:, 2], traj[:, 1],
                     'g-', linewidth=2, label='tool path')

        ax1.set_xlim(pos_min[0], pos_max[0])
        ax1.set_ylim(pos_min[2], pos_max[2])
        ax1.set_zlim(pos_min[1], pos_max[1])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')
        ax1.set_zlabel('Y')
        ax1.set_title(f'Step {frame_idx} | Reward: {rewards[:frame_idx+1].sum():.2f}')
        ax1.view_init(elev=25, azim=45 + frame_idx * 0.5)

        # --- Right: reward curve ---
        ax2 = fig.add_subplot(122)
        ax2.plot(rewards[:frame_idx + 1], 'b-', linewidth=1.5)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlim(0, num_steps)
        ax2.set_ylim(min(rewards.min() - 0.01, -0.1), rewards.max() + 0.01)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.set_title('Per-step Reward')

        # Add text info
        fig.text(0.5, 0.02,
                 f'Knife: ({kp[0]:.3f}, {kp[1]:.3f}, {kp[2]:.3f}) | '
                 f'Cumulative: {rewards[:frame_idx+1].sum():.2f}',
                 ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"frame_{frame_idx:04d}.png"), dpi=100)
        plt.close()

        if frame_idx % 20 == 0:
            print(f"  rendered frame {frame_idx}/{len(all_pos)}")

    print(f"\nFrames saved to {out_dir}/")

    # Try to make video with ffmpeg
    try:
        import subprocess
        video_path = os.path.join(out_dir, "rollout.mp4")
        cmd = (f"ffmpeg -y -framerate 15 -i {out_dir}/frame_%04d.png "
               f"-c:v libx264 -pix_fmt yuv420p {video_path}")
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        print(f"Video saved: {video_path}")
    except Exception as e:
        print(f"ffmpeg not available, frames only: {e}")
        print(f"To make video manually: ffmpeg -framerate 15 -i {out_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p rollout.mp4")


def main():
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_debulk_final")
    parser.add_argument("--config", default="warp_sim/config/debulk_ppo.yaml")
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--outdir", default="rollout_frames")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["episode_length"] = args.frames

    rollout_and_save(args.model, cfg, num_steps=args.frames, out_dir=args.outdir)


if __name__ == "__main__":
    main()
