"""Forward sim viz — compare 3 configs to find best 'soft cube' setting.

Config A (current Step 2 baseline): iters=2, damping=0.99, stiff=0.5, theta=(3,20,0.5)
Config B (soft cube — anti-zombie):  iters=2, damping=0.99, stiff=0.5, theta=(3,100,0.3)
Config C (very soft):                iters=1, damping=0.999, stiff=0.1, theta=(3,100,0.3)

T=10 frames. Knife r=0.10 (slightly larger).
Compare:
   - 3D cube perspective with displacement color
   - Cavity / dead / zombie particle counts
   - Side-view cross-section
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


def run_forward_collect_final(solver, rest_state, theta_3d, stiff_pp_np,
                                knife_traj, knife_radius, iters, damping, device):
    solver.num_iterations = iters
    solver.damping = damping
    theta_torch = torch.tensor(theta_3d, dtype=torch.float32, device=device, requires_grad=False)
    stiff_pp_torch = torch.tensor(stiff_pp_np, dtype=torch.float32, device=device, requires_grad=False)
    state_in = rest_state
    pos_final = alpha_final = None
    with torch.no_grad():
        for t, kp in enumerate(knife_traj):
            a, p, dv = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, kp, knife_radius, stiff_pp_torch)
            pos_final = p.cpu().numpy().copy()
            alpha_final = a.cpu().numpy().copy()
            state_in = solver_save_state(solver)
    return pos_final, alpha_final


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    N = solver.N
    pos_rest = mesh_info["vertices"].copy()
    device = "cuda:0"
    T = 10
    knife_traj = make_knife_trajectory(T)

    # ========== 3 configs ==========
    configs = [
        # (label, iters, damping, stiff_value, theta_3d, knife_radius)
        ("A: baseline\niters=2, dmp=0.99\nstiff=0.5\nthr=0.5 shp=20\n(current Step 2)",
            2, 0.99, 0.5, (3.0, 20.0, 0.5), 0.08),
        ("B: anti-zombie\niters=2, dmp=0.99\nstiff=0.5\nthr=0.3 shp=100\n(true debulk)",
            2, 0.99, 0.5, (3.0, 100.0, 0.3), 0.10),
        ("C: very soft\niters=1, dmp=0.999\nstiff=0.1\nthr=0.3 shp=100\n(jello cube)",
            1, 0.999, 0.1, (3.0, 100.0, 0.3), 0.10),
    ]

    results = {}
    for label, iters, damp, stf, theta, kr in configs:
        clean = label.replace(chr(10), ' | ')
        print(f"  Running [{clean}] ...")
        stiff_uniform = np.full(N, stf, dtype=np.float32)
        pos_f, alpha_f = run_forward_collect_final(
            solver, rest_state, np.array(theta, dtype=np.float32), stiff_uniform,
            knife_traj, kr, iters, damp, device)
        solver_load_state(solver, rest_state)
        disp = np.linalg.norm(pos_f - pos_rest, axis=1) * 1000
        n_dead = int((alpha_f < 0.01).sum())
        n_zombie = int(((alpha_f >= 0.01) & (alpha_f < 0.5)).sum())
        n_alive = int((alpha_f >= 0.5).sum())
        n_flyers = int((disp > 50).sum())  # > 5cm = flying
        results[label] = dict(
            pos=pos_f, alpha=alpha_f, disp=disp, kr=kr,
            n_dead=n_dead, n_zombie=n_zombie, n_alive=n_alive, n_flyers=n_flyers)
        print(f"    dead={n_dead}  zombie={n_zombie}  flyers(>50mm)={n_flyers}  "
              f"mean disp={disp.mean():.2f}mm  max={disp.max():.0f}mm")

    knife_x_pts = [kp[0] for kp in knife_traj]
    knife_y = knife_traj[0][1]
    knife_z = knife_traj[0][2]
    # Use a slice through knife z plane
    z_thick = 0.06
    slice_mask = np.abs(pos_rest[:, 2] - knife_z) < z_thick / 2

    # Global p97 displacement scale across configs (for fair color comparison)
    all_disp = np.concatenate([results[c[0]]['disp'] for c in configs])
    vmax = float(np.percentile(all_disp, 97))
    print(f"\n  Global vmax (p97): {vmax:.2f} mm")

    # ========== Build figure: 3 rows × 3 cols ==========
    # Row 1: 3D perspective
    # Row 2: side view x-z (vertical slice at knife y)
    # Row 3: top view x-y (slice at knife z)
    fig = plt.figure(figsize=(20, 17))

    rng = np.random.default_rng(13)
    sub = rng.choice(N, size=3500, replace=False)

    side_slice = np.abs(pos_rest[:, 1] - knife_y) < 0.05
    top_slice = np.abs(pos_rest[:, 2] - knife_z) < z_thick / 2

    # Row 1: 3D perspective
    for col, (label, iters, damp, stf, theta, kr) in enumerate(configs):
        ax = fig.add_subplot(3, 3, col + 1, projection='3d')
        r = results[label]
        pos = r['pos']
        alpha = r['alpha']
        disp = r['disp']
        cut_mask = alpha < 0.5
        plot_keep = sub[~cut_mask[sub]]
        plot_cut = sub[cut_mask[sub]]
        sc = ax.scatter(pos[plot_keep, 0], pos[plot_keep, 1], pos[plot_keep, 2],
                          c=disp[plot_keep], cmap='hot', s=4, alpha=0.55,
                          vmin=0, vmax=vmax)
        if len(plot_cut) > 0:
            ax.scatter(pos[plot_cut, 0], pos[plot_cut, 1], pos[plot_cut, 2],
                        c='red', marker='x', s=12, linewidths=1.0, alpha=0.85)
        for kx in knife_x_pts:
            ax.scatter([kx], [knife_y], [knife_z], c='cyan', s=40,
                        alpha=0.8, marker='o', edgecolors='black', linewidths=0.6)
        ax.set_title(f"{label}\n"
                     f"dead={r['n_dead']} zombie={r['n_zombie']} flyers={r['n_flyers']}\n"
                     f"mean disp={disp.mean():.1f}mm  max={disp.max():.0f}mm",
                     fontsize=9, fontweight='bold')
        ax.set_xlim(-0.05, 0.35); ax.set_ylim(-0.05, 0.35); ax.set_zlim(-0.05, 0.35)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=24, azim=-55)
        if col == 2:
            plt.colorbar(sc, ax=ax, label='|disp| mm', shrink=0.6, pad=0.1)

    # Row 2: side view (x-z plane at knife y)
    for col, (label, iters, damp, stf, theta, kr) in enumerate(configs):
        ax = fig.add_subplot(3, 3, 3 + col + 1)
        r = results[label]
        pos = r['pos']; alpha = r['alpha']; disp = r['disp']
        cut_mask = (alpha < 0.5) & side_slice
        keep_mask = (alpha >= 0.5) & side_slice
        sc = ax.scatter(pos[keep_mask, 0], pos[keep_mask, 2],
                          c=disp[keep_mask], cmap='hot', s=10, alpha=0.85,
                          vmin=0, vmax=vmax)
        if cut_mask.sum() > 0:
            ax.scatter(pos[cut_mask, 0], pos[cut_mask, 2],
                        c='red', marker='x', s=18, linewidths=1.0)
        for kx in knife_x_pts:
            ax.plot([kx], [knife_z], 'co', markersize=6, alpha=0.6, markeredgecolor='black')
        ax.set_xlabel("x"); ax.set_ylabel("z (up)")
        ax.set_title(f"SIDE view  |  {label.split(chr(10))[0]}", fontsize=10, fontweight='bold')
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 0.35); ax.set_ylim(-0.05, 0.35)
        if col == 2:
            plt.colorbar(sc, ax=ax, label='|disp| mm', shrink=0.85)

    # Row 3: top view (x-y plane at knife z)
    for col, (label, iters, damp, stf, theta, kr) in enumerate(configs):
        ax = fig.add_subplot(3, 3, 6 + col + 1)
        r = results[label]
        pos = r['pos']; alpha = r['alpha']; disp = r['disp']
        cut_mask = (alpha < 0.5) & top_slice
        keep_mask = (alpha >= 0.5) & top_slice
        sc = ax.scatter(pos[keep_mask, 0], pos[keep_mask, 1],
                          c=disp[keep_mask], cmap='hot', s=10, alpha=0.85,
                          vmin=0, vmax=vmax)
        if cut_mask.sum() > 0:
            ax.scatter(pos[cut_mask, 0], pos[cut_mask, 1],
                        c='red', marker='x', s=18, linewidths=1.0)
        ax.plot([min(knife_x_pts), max(knife_x_pts)], [knife_y, knife_y],
                 'cyan', lw=2, alpha=0.5)
        for kx in knife_x_pts:
            knife_circ = plt.Circle((kx, knife_y), r['kr'],
                                      fill=False, color='cyan', lw=0.8, alpha=0.4)
            ax.add_patch(knife_circ)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"TOP view  |  {label.split(chr(10))[0]}", fontsize=10, fontweight='bold')
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 0.35); ax.set_ylim(-0.05, 0.35)
        if col == 2:
            plt.colorbar(sc, ax=ax, label='|disp| mm', shrink=0.85)

    fig.suptitle("Soft cube sweep — 3 configs forward sim (T=10)\n"
                 "Row 1: 3D perspective | Row 2: SIDE x-z at knife y | Row 3: TOP x-y at knife z\n"
                 "Goal: find config with fewer flyers / more true debulk / larger collapse",
                 fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout()
    out_path = REPO / "tools" / "soft_cube_sweep.png"
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
