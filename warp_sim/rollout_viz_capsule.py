"""Rollout a trained Arm-C (or any arm) policy on the capsule-preservation task
and visualize intracapsular debulking behavior.

Renders 3D + side views per frame:
  • Capsule particles (boundary_mask)  — red (alive) / faded gray (ablated)
  • Interior particles                 — blue (alive) / faded gray (ablated)
  • Knife position + trail             — black sphere + trailing line
  • HUD: episode step, interior_done_frac, capsule_breach_frac, cumulative reward

Ideal visualization signature of "intracapsular debulking + capsule preservation":
  • Shell outline (red) stays largely intact throughout the episode
  • Interior (blue) progressively fades to gray from inside outward
  • Knife enters via one breach point, stays within the shell
  • Final cavity is hollow (no blue left) with shell-like outline visible

Outputs:
  rollout_frames_capsule/frame_NNNN.png   (per-frame 3D + side scatter + HUD)
  rollout_frames_capsule/rollout.mp4      (stitched via ffmpeg if available)
  rollout_frames_capsule/summary.json     (per-step metrics)

Usage (Arm C seed 0 example):
  python warp_sim/rollout_viz_capsule.py \\
      --model    checkpoints/cap_armC_diffM_seed0/cap_armC_diffM_seed0_final.zip \\
      --vecnorm  checkpoints/cap_armC_diffM_seed0/cap_armC_diffM_seed0_final_vecnormalize.pkl \\
      --cfg      warp_sim/config/debulk_cap_armC.yaml \\
      --frames   256 \\
      --out-dir  rollout_frames_capsule_armC
"""
import os
import sys
import argparse
import json
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pf_shape_matching_warp"))

import warp as wp
wp.init()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

import yaml

ALPHA_ALIVE_THRESHOLD = 0.5


def _collect_frame_state(env):
    pos = env.solver.p["pos"].numpy()                    # (N, 3)
    alpha = env.solver.p["alpha"].numpy()                # (N,)
    knife = env.solver.knife_pos.numpy().reshape(3)
    boundary_mask = env._boundary_mask_np.astype(bool)
    interior_mask = env._interior_mask_np.astype(bool)
    return {
        "pos": pos, "alpha": alpha, "knife": knife,
        "boundary_mask": boundary_mask, "interior_mask": interior_mask,
        "capsule_breach_frac": float(env._capsule_breached_frac),
        "interior_done_frac":  float(env._interior_done_frac),
    }


def _render_frame(state, knife_trail, frame_idx, step_t, cum_reward, out_path,
                  cavity_zoom=False):
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.08)

    pos = state["pos"]
    alpha = state["alpha"]
    alive = alpha > ALPHA_ALIVE_THRESHOLD
    bm = state["boundary_mask"]
    im = state["interior_mask"]

    cap_alive = bm & alive
    cap_dead  = bm & ~alive
    int_alive = im & alive
    int_dead  = im & ~alive

    # --- Subplot 1: 3D scatter ---
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    # Capsule alive (red, larger)
    ax3d.scatter(pos[cap_alive, 0], pos[cap_alive, 1], pos[cap_alive, 2],
                 c="crimson", s=4, alpha=0.55, label=f"capsule ({int(cap_alive.sum())})",
                 depthshade=False)
    # Interior alive (blue, smaller)
    ax3d.scatter(pos[int_alive, 0], pos[int_alive, 1], pos[int_alive, 2],
                 c="royalblue", s=3, alpha=0.22, label=f"interior ({int(int_alive.sum())})",
                 depthshade=False)
    # Dead particles (translucent gray, smallest) — only if exist, for context
    if cap_dead.any():
        ax3d.scatter(pos[cap_dead, 0], pos[cap_dead, 1], pos[cap_dead, 2],
                     c="lightcoral", s=2, alpha=0.12,
                     label=f"capsule ablated ({int(cap_dead.sum())})",
                     depthshade=False)
    if int_dead.any():
        ax3d.scatter(pos[int_dead, 0], pos[int_dead, 1], pos[int_dead, 2],
                     c="lightgray", s=2, alpha=0.10,
                     label=f"interior ablated ({int(int_dead.sum())})",
                     depthshade=False)
    # Knife
    ax3d.scatter([state["knife"][0]], [state["knife"][1]], [state["knife"][2]],
                 c="black", s=120, marker="o", label="knife", depthshade=False)
    if len(knife_trail) > 1:
        trail = np.asarray(knife_trail)
        ax3d.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                  c="black", linewidth=1.0, alpha=0.6)

    ax3d.set_xlim(0, 1); ax3d.set_ylim(0, 0.55); ax3d.set_zlim(0, 1)
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title(f"3D  |  frame {frame_idx:04d}, sim step {step_t}", fontsize=10)
    ax3d.legend(loc="upper right", fontsize=7, framealpha=0.85)
    # View from front-upper corner for clarity
    ax3d.view_init(elev=22, azim=-65)

    # --- Subplot 2: XZ projection (top-down through y) ---
    ax2d = fig.add_subplot(gs[0, 1])
    ax2d.scatter(pos[cap_alive, 0], pos[cap_alive, 2],
                 c="crimson", s=4, alpha=0.55)
    ax2d.scatter(pos[int_alive, 0], pos[int_alive, 2],
                 c="royalblue", s=3, alpha=0.22)
    if cap_dead.any():
        ax2d.scatter(pos[cap_dead, 0], pos[cap_dead, 2],
                     c="lightcoral", s=2, alpha=0.18)
    if int_dead.any():
        ax2d.scatter(pos[int_dead, 0], pos[int_dead, 2],
                     c="lightgray", s=2, alpha=0.18)
    ax2d.scatter([state["knife"][0]], [state["knife"][2]],
                 c="black", s=120, marker="o")
    if len(knife_trail) > 1:
        trail = np.asarray(knife_trail)
        ax2d.plot(trail[:, 0], trail[:, 2], c="black", linewidth=1.0, alpha=0.6)

    ax2d.set_xlim(0, 1); ax2d.set_ylim(0, 1)
    ax2d.set_xlabel("x"); ax2d.set_ylabel("z")
    ax2d.set_aspect("equal")
    ax2d.set_title("Top-down view (XZ)", fontsize=10)
    ax2d.grid(alpha=0.2)

    # HUD overlay on 2D subplot
    hud = (
        f"step: {step_t}\n"
        f"interior debulked: {state['interior_done_frac']*100:6.2f}%\n"
        f"capsule breach:     {state['capsule_breach_frac']*100:6.2f}%\n"
        f"cumulative reward:  {cum_reward:7.2f}"
    )
    ax2d.text(0.02, 0.98, hud, transform=ax2d.transAxes,
              fontsize=9, va="top", ha="left",
              family="monospace",
              bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray",
                        boxstyle="round,pad=0.4"))

    plt.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def run_rollout(model_path, vecnorm_path, cfg_path, frames, out_dir,
                device="cuda:0", render_every=1):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    from warp_sim.envs.debulk_env import DebulkEnv

    cfg = yaml.safe_load(open(cfg_path))
    env_fn = lambda: DebulkEnv(cfg, device=device)
    env = DummyVecEnv([env_fn])
    env = VecMonitor(env)
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    underlying = env.envs[0] if hasattr(env, "envs") else env.venv.envs[0]

    model = PPO.load(model_path, env=env, device="cpu")
    os.makedirs(out_dir, exist_ok=True)

    obs = env.reset()
    knife_trail = [underlying.solver.knife_pos.numpy().reshape(3).tolist()]
    summary = []
    cum_reward = 0.0

    for t in range(frames):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info_list = env.step(action)
        cum_reward += float(reward[0])

        # Collect state BEFORE auto-reset wipes it. If done=True fired this step,
        # VecEnv already reset the underlying env's particle alpha / knife — but
        # info dict carries the true end-of-episode scalars.
        if done[0]:
            print(f"Episode ended at step {t}: freezing final frame on pre-reset metrics from info dict")
            info = info_list[0]
            state = _collect_frame_state(underlying)
            state["interior_done_frac"]  = float(info.get("interior_done_frac", state["interior_done_frac"]))
            state["capsule_breach_frac"] = float(info.get("capsule_breached_frac", state["capsule_breach_frac"]))
        else:
            state = _collect_frame_state(underlying)
        knife_trail.append(state["knife"].tolist())

        if t % render_every == 0:
            out_path = os.path.join(out_dir, f"frame_{t:04d}.png")
            _render_frame(state, knife_trail, frame_idx=t, step_t=t,
                          cum_reward=cum_reward, out_path=out_path)

        summary.append({
            "step": t,
            "knife_pos": state["knife"].tolist(),
            "reward": float(reward[0]),
            "cumulative_reward": round(cum_reward, 3),
            "interior_done_frac": state["interior_done_frac"],
            "capsule_breach_frac": state["capsule_breach_frac"],
            "alive_capsule": int((state["boundary_mask"] & (state["alpha"] > ALPHA_ALIVE_THRESHOLD)).sum()),
            "alive_interior": int((state["interior_mask"] & (state["alpha"] > ALPHA_ALIVE_THRESHOLD)).sum()),
        })

        if done[0]:
            # Render ONE final frame even if rendering interval missed it
            if t % render_every != 0:
                out_path = os.path.join(out_dir, f"frame_{t:04d}.png")
                _render_frame(state, knife_trail, frame_idx=t, step_t=t,
                              cum_reward=cum_reward, out_path=out_path)
            break

    # Final summary to JSON
    final = summary[-1] if summary else None
    meta = {
        "frames_rendered": t + 1,
        "final_interior_done_frac": final["interior_done_frac"] if final else None,
        "final_capsule_breach_frac": final["capsule_breach_frac"] if final else None,
        "cumulative_reward": round(cum_reward, 3),
        "model_path": model_path,
        "cfg_path": cfg_path,
        "terminated_early": bool(done[0]) if 'done' in locals() else False,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"meta": meta, "per_step": summary}, f, indent=2)
    print(f"\n=== Rollout summary ===")
    print(f"  frames rendered:       {meta['frames_rendered']}")
    print(f"  final interior cov:    {meta['final_interior_done_frac']*100:.2f}%")
    print(f"  final capsule breach:  {meta['final_capsule_breach_frac']*100:.2f}%")
    print(f"  cumulative reward:     {meta['cumulative_reward']}")

    # Stitch MP4 via ffmpeg (if available)
    mp4_path = os.path.join(out_dir, "rollout.mp4")
    try:
        cmd = [
            "ffmpeg", "-y", "-framerate", "20",
            "-pattern_type", "glob", "-i", os.path.join(out_dir, "frame_*.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
            mp4_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  MP4 video:             {mp4_path}")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  (ffmpeg failed — PNGs only at {out_dir}/)")

    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True, help="PPO .zip checkpoint path")
    p.add_argument("--vecnorm",  default=None,  help="VecNormalize .pkl path")
    p.add_argument("--cfg",      required=True, help="arm YAML config")
    p.add_argument("--frames",   type=int, default=256)
    p.add_argument("--render-every", type=int, default=1,
                   help="render every Nth frame (>1 = skip frames, faster)")
    p.add_argument("--out-dir",  default="rollout_frames_capsule")
    p.add_argument("--device",   default="cuda:0")
    args = p.parse_args()

    run_rollout(args.model, args.vecnorm, args.cfg, args.frames,
                args.out_dir, device=args.device, render_every=args.render_every)


if __name__ == "__main__":
    main()
