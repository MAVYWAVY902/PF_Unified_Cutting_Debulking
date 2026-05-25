"""GIF showing Config A (current, with zombies) vs Config B (anti-zombie debulk).

Side-by-side animation, T=10, 3D perspective.
Demonstrates the dramatic improvement from threshold 0.5→0.3 + sharpness 20→100.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))

import torch
import warp as wp
wp.init()

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_load_state, solver_save_state  # noqa
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa
from tools.day6f_v0_multistep_synth import make_knife_trajectory  # noqa


def run_forward_all_frames(solver, rest_state, theta_3d, stiff_pp_np,
                              knife_traj, knife_radius, iters, damping, device):
    solver.num_iterations = iters
    solver.damping = damping
    theta_torch = torch.tensor(theta_3d, dtype=torch.float32, device=device, requires_grad=False)
    stiff_pp_torch = torch.tensor(stiff_pp_np, dtype=torch.float32, device=device, requires_grad=False)
    state_in = rest_state
    frames = []
    with torch.no_grad():
        for t, kp in enumerate(knife_traj):
            a, p, dv = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, kp, knife_radius, stiff_pp_torch)
            frames.append((p.cpu().numpy().copy(), a.cpu().numpy().copy()))
            state_in = solver_save_state(solver)
    return frames


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    N = solver.N
    pos_rest = mesh_info["vertices"].copy()
    device = "cuda:0"
    T = 10
    knife_traj = make_knife_trajectory(T)
    stiff_uniform = np.full(N, 0.5, dtype=np.float32)

    # Config A: current Step 2 baseline (with zombies)
    print("  Sim Config A (baseline w/ zombies) ...")
    frames_A = run_forward_all_frames(
        solver, rest_state, np.array([3.0, 20.0, 0.5], dtype=np.float32),
        stiff_uniform, knife_traj, 0.08, iters=2, damping=0.99, device=device)
    solver_load_state(solver, rest_state)

    # Config B: anti-zombie + true debulk
    print("  Sim Config B (anti-zombie, clean debulk) ...")
    frames_B = run_forward_all_frames(
        solver, rest_state, np.array([3.0, 100.0, 0.3], dtype=np.float32),
        stiff_uniform, knife_traj, 0.10, iters=2, damping=0.99, device=device)
    solver_load_state(solver, rest_state)

    # Subsample for visual cleanliness
    rng = np.random.default_rng(13)
    sub = rng.choice(N, size=2800, replace=False)

    # Global color scale (use Config B's p97 — A's flyers would dominate)
    all_disp_B = np.concatenate(
        [np.linalg.norm(p - pos_rest, axis=1) * 1000 for p, _ in frames_B])
    vmax = float(np.percentile(all_disp_B, 97))
    print(f"  Color scale vmax (B's p97): {vmax:.2f} mm")

    # Build figure: 2 panels side by side
    fig = plt.figure(figsize=(15, 7))
    ax_A = fig.add_subplot(1, 2, 1, projection='3d')
    ax_B = fig.add_subplot(1, 2, 2, projection='3d')
    sup_title = fig.suptitle("", fontsize=13, fontweight='bold')

    AXES_LIM = (-0.10, 0.40)  # extended to show flyers in A

    def render_panel(ax, frames, kt_idx, label, kr, theta_str):
        ax.cla()
        pos, alpha = frames[kt_idx]
        kp = knife_traj[kt_idx]
        disp = np.linalg.norm(pos - pos_rest, axis=1) * 1000
        cut = alpha < 0.5
        keep = ~cut
        plot_keep = sub[keep[sub]]
        plot_cut = sub[cut[sub]]
        # Particles
        ax.scatter(pos[plot_keep, 0], pos[plot_keep, 1], pos[plot_keep, 2],
                    c=disp[plot_keep], cmap='hot', s=4, alpha=0.55,
                    vmin=0, vmax=vmax)
        if len(plot_cut) > 0:
            ax.scatter(pos[plot_cut, 0], pos[plot_cut, 1], pos[plot_cut, 2],
                        c='red', marker='x', s=10, linewidths=1.0, alpha=0.85)
        # Knife ball
        ax.scatter([kp[0]], [kp[1]], [kp[2]], c='cyan', s=50, alpha=0.85,
                    marker='o', edgecolors='black', linewidths=0.7)
        # Knife wireframe sphere
        u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:6j]
        sx = kp[0] + kr * np.cos(u) * np.sin(v)
        sy = kp[1] + kr * np.sin(u) * np.sin(v)
        sz = kp[2] + kr * np.cos(v)
        ax.plot_wireframe(sx, sy, sz, color='cyan', alpha=0.3, linewidth=0.5)
        # Stats
        n_dead = int((alpha < 0.01).sum())
        n_zomb = int(((alpha >= 0.01) & (alpha < 0.5)).sum())
        n_fly  = int((disp > 50).sum())
        ax.set_title(f"{label}\n{theta_str}\n"
                     f"dead={n_dead}  zombie={n_zomb}  flyers(>50mm)={n_fly}\n"
                     f"max disp={disp.max():.0f}mm",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_xlim(*AXES_LIM); ax.set_ylim(*AXES_LIM); ax.set_zlim(*AXES_LIM)
        ax.view_init(elev=22, azim=-58)

    def update(frame_idx):
        render_panel(ax_A, frames_A, frame_idx,
                     "Config A — BASELINE", 0.08,
                     "theta=(3, 20, 0.5)  ← current Warp default")
        render_panel(ax_B, frames_B, frame_idx,
                     "Config B — ANTI-ZOMBIE", 0.10,
                     "theta=(3, 100, 0.3)  ← true debulk")
        sup_title.set_text(
            f"Warp PBD soft removal — A (sigmoid w/ zombies) vs B (anti-zombie, true debulk)\n"
            f"Frame {frame_idx+1}/{T}  |  knife sweeps x∈[0.12, 0.18] at y=0.30 z=0.15\n"
            f"red X = cut particles, cyan = knife")
        return []

    frame_indices = list(range(T)) + [T - 1] * 3
    print(f"  Building animation ({T} frames + holds) ...")
    anim = FuncAnimation(fig, update, frames=frame_indices,
                            interval=600, blit=False, repeat=True)
    out_path = REPO / "tools" / "config_a_vs_b_animation.gif"
    print(f"  Saving GIF to {out_path} ...")
    anim.save(str(out_path), writer=PillowWriter(fps=2), dpi=85)
    print(f"  Done. Size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
