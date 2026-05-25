"""Generate 3D cube deformation GIF — frame-by-frame across T=10 trajectory.

3 PBD configs side-by-side: rigid / medium / compliant.
Each frame shows cube particles colored by per-particle displacement from rest.
Cut particles (alpha<0.5) marked red.
Knife shown as small cyan dot at current trajectory position.

Output: t10_cube_animation.gif
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


def run_forward_collect_all_frames(solver, rest_state, theta_3d, stiff_pp_np,
                                       knife_traj, knife_radius, num_iterations,
                                       damping, device):
    """Return list of (pos, alpha) one per frame."""
    solver.num_iterations = num_iterations
    solver.damping = damping
    theta_torch = torch.tensor(theta_3d, dtype=torch.float32, device=device, requires_grad=False)
    stiff_pp_torch = torch.tensor(stiff_pp_np, dtype=torch.float32, device=device, requires_grad=False)
    state_in = rest_state
    frames = []
    with torch.no_grad():
        for t, kp in enumerate(knife_traj):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, kp, knife_radius, stiff_pp_torch)
            frames.append((pos_t.cpu().numpy().copy(),
                            alpha_t.cpu().numpy().copy()))
            state_in = solver_save_state(solver)
    return frames


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    N = solver.N
    pos_rest = mesh_info["vertices"].copy()
    device = "cuda:0"
    knife_radius = 0.08
    T = 10
    knife_traj = make_knife_trajectory(T)
    theta_3d = np.array([3.0, 20.0, 0.5], dtype=np.float32)
    stiff_uniform = np.full(N, 0.5, dtype=np.float32)

    configs = [
        ("Rigid (iters=6)",     6, 0.98),
        ("Medium (iters=4)",    4, 0.99),
        ("Compliant (iters=2)", 2, 0.99),
    ]

    all_frames = {}
    for label, iters, damp in configs:
        print(f"  Sim {label} ...")
        frames = run_forward_collect_all_frames(
            solver, rest_state, theta_3d, stiff_uniform, knife_traj,
            knife_radius, iters, damp, device)
        solver_load_state(solver, rest_state)
        all_frames[label] = frames

    # Subsample particles for clean rendering
    rng = np.random.default_rng(13)
    sub = rng.choice(N, size=2500, replace=False)

    # Global color scale
    all_disp = []
    for label in [c[0] for c in configs]:
        for pos, _ in all_frames[label]:
            all_disp.append(np.linalg.norm(pos - pos_rest, axis=1) * 1000)
    vmax = float(np.percentile(np.concatenate(all_disp), 97))
    print(f"  Global disp vmax (p97): {vmax:.2f} mm")

    # ============================================================
    # Build figure
    # ============================================================
    fig = plt.figure(figsize=(18, 6.5))
    axes = []
    for i, (label, _, _) in enumerate(configs):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.set_xlim(0, 0.3); ax.set_ylim(0, 0.32); ax.set_zlim(0, 0.3)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=24, azim=-55)
        axes.append(ax)

    sup_title = fig.suptitle("", fontsize=14, fontweight='bold')

    def update(frame_idx):
        for ax, (label, iters, damp) in zip(axes, configs):
            ax.cla()
            pos, alpha = all_frames[label][frame_idx]
            disp = np.linalg.norm(pos - pos_rest, axis=1) * 1000  # mm
            kp = knife_traj[frame_idx]
            cut = alpha < 0.5
            keep = ~cut
            plot_keep = sub[keep[sub]]
            plot_cut = sub[cut[sub]]
            # Cube particles
            ax.scatter(pos[plot_keep, 0], pos[plot_keep, 1], pos[plot_keep, 2],
                        c=disp[plot_keep], cmap='hot', s=4, alpha=0.55,
                        vmin=0, vmax=vmax)
            # Cut particles in red
            if len(plot_cut) > 0:
                ax.scatter(pos[plot_cut, 0], pos[plot_cut, 1], pos[plot_cut, 2],
                            c='red', marker='x', s=14, linewidths=1.0, alpha=0.85)
            # Knife dot (SMALLER per user feedback)
            ax.scatter([kp[0]], [kp[1]], [kp[2]], c='cyan', s=50,
                        alpha=0.85, marker='o', edgecolors='black', linewidths=0.7)
            # Also visualize knife radius as wireframe sphere (smaller wireframe)
            u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:6j]
            sx = kp[0] + knife_radius * np.cos(u) * np.sin(v)
            sy = kp[1] + knife_radius * np.sin(u) * np.sin(v)
            sz = kp[2] + knife_radius * np.cos(v)
            ax.plot_wireframe(sx, sy, sz, color='cyan', alpha=0.25, linewidth=0.4)
            # Statistics in title
            n_cut = int(cut.sum())
            ax.set_title(f"{label}\n"
                         f"frame {frame_idx+1}/{T} | cut={n_cut} | "
                         f"mean disp={disp.mean():.2f}mm | max={disp.max():.0f}mm",
                         fontsize=10, fontweight='bold')
            ax.set_xlim(0, 0.3); ax.set_ylim(0, 0.32); ax.set_zlim(0, 0.3)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            ax.view_init(elev=24, azim=-55)
        sup_title.set_text(
            f"T={T} cube deformation animation — frame {frame_idx+1}/{T}\n"
            f"Knife (cyan ball) sweeps x ∈ [0.12, 0.18] at y=0.30, z=0.15  |  "
            f"red X = cut particles, color = displacement magnitude")
        return []

    print(f"  Building animation with {T} frames + holds ...")
    # Hold last frame for 1.5s by repeating it
    frame_indices = list(range(T)) + [T - 1] * 3  # pause on last frame
    anim = FuncAnimation(fig, update, frames=frame_indices,
                            interval=600, blit=False, repeat=True)
    out_path = REPO / "tools" / "t10_cube_animation.gif"
    print(f"  Saving GIF to {out_path} ...")
    anim.save(str(out_path), writer=PillowWriter(fps=2), dpi=80)
    print(f"  Done. Size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
