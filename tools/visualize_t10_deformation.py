"""Detailed T=10 deformation rendering — forward-only, no sys-ID.

3 configs (rigid → medium → compliant), 3 views per config:
   - Top view (x-y at knife z): see knife cut path + cavity
   - Side view (x-z at knife y): see depth/sag of deformation
   - 3D perspective with displacement arrows on outer surface

Particles colored by displacement magnitude (mm).
Cut particles (alpha < 0.5) marked separately as red X.
Knife sweep shown as cyan dots/line.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def run_forward_collect(solver, rest_state, theta_3d, stiff_pp_np, knife_traj,
                          knife_radius, num_iterations, damping, device):
    """Run T-frame forward; return final pos + final alpha."""
    solver.num_iterations = num_iterations
    solver.damping = damping
    theta_torch = torch.tensor(theta_3d, dtype=torch.float32, device=device, requires_grad=False)
    stiff_pp_torch = torch.tensor(stiff_pp_np, dtype=torch.float32, device=device, requires_grad=False)
    state_in = rest_state
    final_pos = None
    final_alpha = None
    with torch.no_grad():
        for t, kp in enumerate(knife_traj):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, kp, knife_radius, stiff_pp_torch)
            final_pos = pos_t.cpu().numpy().copy()
            final_alpha = alpha_t.cpu().numpy().copy()
            state_in = solver_save_state(solver)
    return final_pos, final_alpha


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
        ("Rigid PBD\niters=6, damping=0.98",     6, 0.98),
        ("Medium PBD\niters=4, damping=0.99",    4, 0.99),
        ("Compliant PBD\niters=2, damping=0.99", 2, 0.99),
    ]

    results = {}
    for label, iters, damp in configs:
        print(f"  Running {label.replace(chr(10), ' ')} (T={T}) ...")
        final_pos, final_alpha = run_forward_collect(
            solver, rest_state, theta_3d, stiff_uniform,
            knife_traj, knife_radius, iters, damp, device)
        solver_load_state(solver, rest_state)
        results[label] = (final_pos, final_alpha)

    knife_x_pts = [kp[0] for kp in knife_traj]
    knife_y = knife_traj[0][1]
    knife_z = knife_traj[0][2]

    # Compute global color scale across all configs for fair comparison
    all_disp_mm = []
    for label in [c[0] for c in configs]:
        final_pos, _ = results[label]
        disp = np.linalg.norm(final_pos - pos_rest, axis=1) * 1000
        all_disp_mm.append(disp)
    # Cap at 95th percentile for clear visualization (max is too extreme)
    vmax = float(np.percentile(np.concatenate(all_disp_mm), 95))
    print(f"\n  Global disp vmax (p95): {vmax:.2f} mm")

    # Slice definitions
    z_thick = 0.04  # ±2cm around knife_z (for top view)
    y_thick = 0.06  # ±3cm around knife_y (for side view)
    slice_z = np.abs(pos_rest[:, 2] - knife_z) < z_thick / 2
    slice_y = np.abs(pos_rest[:, 1] - knife_y) < y_thick / 2
    print(f"  Top-view slice (|z-{knife_z}|<{z_thick/2}): {slice_z.sum()} particles")
    print(f"  Side-view slice (|y-{knife_y}|<{y_thick/2}): {slice_y.sum()} particles")

    fig = plt.figure(figsize=(20, 17))

    # ============================================================
    # Row 1: top view (x-y plane at knife z) — see knife sweep + cavity
    # ============================================================
    for col, (label, iters, damp) in enumerate(configs):
        ax = fig.add_subplot(3, 3, col + 1)
        final_pos, final_alpha = results[label]
        disp = np.linalg.norm(final_pos - pos_rest, axis=1) * 1000
        # In slice
        idx = np.where(slice_z)[0]
        cut_mask = (final_alpha < 0.5) & slice_z
        keep_mask = (final_alpha >= 0.5) & slice_z

        # Draw kept particles colored by displacement
        sc = ax.scatter(final_pos[keep_mask, 0], final_pos[keep_mask, 1],
                          c=disp[keep_mask], cmap='hot', s=12, alpha=0.85,
                          vmin=0, vmax=vmax)
        # Draw cut particles as red X
        if cut_mask.sum() > 0:
            ax.scatter(final_pos[cut_mask, 0], final_pos[cut_mask, 1],
                        c='red', marker='x', s=20, linewidths=1.2,
                        label=f'cut ({int(cut_mask.sum())} pts)')
        # Knife sweep line
        ax.plot([min(knife_x_pts), max(knife_x_pts)], [knife_y, knife_y],
                 'cyan', lw=2, alpha=0.5, label='knife path')
        for kx in knife_x_pts:
            knife_circ = plt.Circle((kx, knife_y), knife_radius,
                                      fill=False, color='cyan', lw=0.8, alpha=0.25)
            ax.add_patch(knife_circ)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        slice_disp = disp[keep_mask]
        ax.set_title(f"TOP view  |  {label}\n"
                     f"cut={int(cut_mask.sum())}  mean={slice_disp.mean():.1f}mm  "
                     f"p95={np.percentile(slice_disp, 95):.1f}mm",
                     fontsize=10, fontweight='bold')
        ax.set_aspect("equal")
        ax.set_xlim(0, 0.32); ax.set_ylim(0, 0.34)
        ax.legend(loc='lower right', fontsize=7)
        if col == 2:
            plt.colorbar(sc, ax=ax, label='|disp| (mm, cap p95)', shrink=0.85)

    # ============================================================
    # Row 2: side view (x-z plane at knife y) — see depth + sag
    # ============================================================
    for col, (label, iters, damp) in enumerate(configs):
        ax = fig.add_subplot(3, 3, 3 + col + 1)
        final_pos, final_alpha = results[label]
        disp = np.linalg.norm(final_pos - pos_rest, axis=1) * 1000
        cut_mask = (final_alpha < 0.5) & slice_y
        keep_mask = (final_alpha >= 0.5) & slice_y
        sc = ax.scatter(final_pos[keep_mask, 0], final_pos[keep_mask, 2],
                          c=disp[keep_mask], cmap='hot', s=10, alpha=0.85,
                          vmin=0, vmax=vmax)
        if cut_mask.sum() > 0:
            ax.scatter(final_pos[cut_mask, 0], final_pos[cut_mask, 2],
                        c='red', marker='x', s=20, linewidths=1.2)
        # Knife dots in x-z plane (y projection)
        for kx in knife_x_pts:
            ax.plot([kx], [knife_z], 'co', markersize=8, alpha=0.5,
                     markeredgecolor='black')
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m, up)")
        slice_disp = disp[keep_mask]
        ax.set_title(f"SIDE view  |  {label}\n"
                     f"cut={int(cut_mask.sum())}  mean={slice_disp.mean():.1f}mm  "
                     f"p95={np.percentile(slice_disp, 95):.1f}mm",
                     fontsize=10, fontweight='bold')
        ax.set_aspect("equal")
        ax.set_xlim(0, 0.32); ax.set_ylim(-0.02, 0.32)
        if col == 2:
            plt.colorbar(sc, ax=ax, label='|disp| (mm, cap p95)', shrink=0.85)

    # ============================================================
    # Row 3: 3D perspective with displacement arrows on top surface
    # ============================================================
    # Subset of particles for clean 3D view
    rng = np.random.default_rng(13)
    sub = rng.choice(N, size=3500, replace=False)
    for col, (label, iters, damp) in enumerate(configs):
        ax = fig.add_subplot(3, 3, 6 + col + 1, projection='3d')
        final_pos, final_alpha = results[label]
        disp = np.linalg.norm(final_pos - pos_rest, axis=1) * 1000
        cut_mask = final_alpha < 0.5
        keep_mask = final_alpha >= 0.5
        # Particles in 3D
        plot_keep = sub[keep_mask[sub]]
        plot_cut = sub[cut_mask[sub]]
        sc = ax.scatter(final_pos[plot_keep, 0],
                          final_pos[plot_keep, 1],
                          final_pos[plot_keep, 2],
                          c=disp[plot_keep], cmap='hot', s=4, alpha=0.55,
                          vmin=0, vmax=vmax)
        if len(plot_cut) > 0:
            ax.scatter(final_pos[plot_cut, 0],
                        final_pos[plot_cut, 1],
                        final_pos[plot_cut, 2],
                        c='red', marker='x', s=14, linewidths=1.1, alpha=0.9)
        # Knife trajectory
        for kx in knife_x_pts:
            ax.scatter([kx], [knife_y], [knife_z], c='cyan', s=200,
                        alpha=0.7, marker='o', edgecolors='black', linewidths=1.0)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"3D perspective  |  {label}\n"
                     f"global mean disp = {disp.mean():.2f}mm  "
                     f"max = {disp.max():.0f}mm",
                     fontsize=10, fontweight='bold')
        ax.set_xlim(0, 0.3); ax.set_ylim(0, 0.32); ax.set_zlim(0, 0.3)
        ax.view_init(elev=24, azim=-55)
        if col == 2:
            plt.colorbar(sc, ax=ax, label='|disp| (mm, cap p95)',
                          shrink=0.55, pad=0.1)

    fig.suptitle(
        f"T={T} cube deformation under 3 PBD configs (forward-only, no sys-ID)\n"
        f"Rows: TOP view (x-y at knife z) | SIDE view (x-z at knife y) | 3D perspective\n"
        f"Particles colored by |displacement|. Red X = cut particles (alpha<0.5). Cyan = knife sweep.",
        fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout()
    out_path = REPO / "tools" / "t10_deformation_compare.png"
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
