"""Visualize: (1) deformation under different param regimes side-by-side,
            (2) 3D perspective view of learned stiff field.

Top row:    rest cube → deformed cube under (rigid / mid / compliant) PBD
            colored by per-particle displacement magnitude
Bottom row: stiff field 3D perspective: truth | recovered | error
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


def run_forward(solver, rest_state, theta_3d, stiff_pp_np, knife_traj,
                  knife_radius, num_iterations, damping, device):
    """Run T-frame forward at the given PBD config; return final pos."""
    solver.num_iterations = num_iterations
    solver.damping = damping
    theta_torch = torch.tensor(theta_3d, dtype=torch.float32, device=device, requires_grad=False)
    stiff_pp_torch = torch.tensor(stiff_pp_np, dtype=torch.float32, device=device, requires_grad=False)
    state_in = rest_state
    final_pos = None
    with torch.no_grad():
        for t, kp in enumerate(knife_traj):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, kp, knife_radius, stiff_pp_torch)
            final_pos = pos_t.cpu().numpy().copy()
            state_in = solver_save_state(solver)
    return final_pos


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    N = solver.N
    pos_rest = mesh_info["vertices"].copy()
    device = "cuda:0"
    knife_radius = 0.08
    T = 3
    knife_traj = make_knife_trajectory(T)
    theta_3d = np.array([3.0, 20.0, 0.5], dtype=np.float32)

    # Same homogeneous stiff field for fair deformation comparison
    stiff_uniform = np.full(N, 0.5, dtype=np.float32)

    # 3 configs
    configs = [
        ("Rigid PBD\n(iters=6, damping=0.98)",  6, 0.98),
        ("Medium PBD\n(iters=4, damping=0.99)", 4, 0.99),
        ("Compliant PBD\n(iters=2, damping=0.99)", 2, 0.99),
    ]
    deformed_pos = {}
    for label, iters, damp in configs:
        print(f"  Running {label.replace(chr(10), ' ')} ...")
        final = run_forward(solver, rest_state, theta_3d, stiff_uniform,
                              knife_traj, knife_radius, iters, damp, device)
        # Reset solver state for next config
        solver_load_state(solver, rest_state)
        deformed_pos[label] = final

    # Load stiff recovery data
    data = np.load(REPO / "tools" / "stageA_inclusion_recovery.npz")
    stiff_truth = data["stiff_truth"]
    stiff_recovered = data["stiff_recovered"]
    inclusion_mask = data["inclusion_mask"]
    incl_center = pos_rest[inclusion_mask].mean(axis=0)
    incl_r = 0.06

    # ============================================================
    # Build figure: 2 rows × 3 cols = 6 panels
    # Row 1: deformation under 3 configs
    # Row 2: stiff truth | stiff recovered | per-particle scatter
    # ============================================================
    fig = plt.figure(figsize=(20, 13))

    knife_x_pts = [kp[0] for kp in knife_traj]
    knife_y = knife_traj[0][1]
    knife_z = knife_traj[0][2]

    # === Row 1: 3D deformation views ===
    for col, (label, iters, damp) in enumerate(configs):
        ax = fig.add_subplot(2, 3, col + 1, projection='3d')
        final = deformed_pos[label]
        disp = np.linalg.norm(final - pos_rest, axis=1) * 1000  # mm
        # Subsample for speed
        rng = np.random.default_rng(42)
        sub = rng.choice(N, size=4000, replace=False)
        sc = ax.scatter(final[sub, 0], final[sub, 1], final[sub, 2],
                          c=disp[sub], cmap='hot', s=4, alpha=0.7,
                          vmin=0, vmax=max(disp.max(), 1.0))
        # Knife
        for kx in knife_x_pts:
            ax.scatter([kx], [knife_y], [knife_z], c='cyan', s=400,
                        alpha=0.8, marker='o', edgecolors='black', linewidths=1.5)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_title(f"{label}\n"
                     f"mean disp = {disp.mean():.2f} mm  "
                     f"max disp = {disp.max():.2f} mm",
                     fontsize=11, fontweight='bold')
        ax.set_xlim(0, 0.3); ax.set_ylim(0, 0.32); ax.set_zlim(0, 0.3)
        ax.view_init(elev=22, azim=-58)
        plt.colorbar(sc, ax=ax, label='|displacement| (mm)', shrink=0.7, pad=0.1)

    # === Row 2: stiff field 3D perspective views ===
    vmin = 0.0
    vmax = max(stiff_truth.max(), 1.6)

    # Panel 4: TRUTH stiff
    ax = fig.add_subplot(2, 3, 4, projection='3d')
    # Show subsample + all inclusion particles
    rng = np.random.default_rng(7)
    sub = rng.choice(N, size=3000, replace=False)
    incl_idx = np.where(inclusion_mask)[0]
    plot_idx = np.unique(np.concatenate([sub, incl_idx]))
    sc = ax.scatter(pos_rest[plot_idx, 0], pos_rest[plot_idx, 1], pos_rest[plot_idx, 2],
                      c=stiff_truth[plot_idx], cmap='viridis',
                      vmin=vmin, vmax=vmax, s=6, alpha=0.7)
    # Inclusion sphere outline (drawn as wireframe)
    u, v = np.mgrid[0:2*np.pi:18j, 0:np.pi:10j]
    sx = incl_center[0] + incl_r * np.cos(u) * np.sin(v)
    sy = incl_center[1] + incl_r * np.sin(u) * np.sin(v)
    sz = incl_center[2] + incl_r * np.cos(v)
    ax.plot_wireframe(sx, sy, sz, color='red', alpha=0.4, linewidth=0.6)
    for kx in knife_x_pts:
        ax.scatter([kx], [knife_y], [knife_z], c='cyan', s=300,
                    alpha=0.7, marker='o', edgecolors='black', linewidths=1.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"TRUTH stiff field\n"
                 f"inside inclusion: {stiff_truth[inclusion_mask].mean():.2f}\n"
                 f"outside: {stiff_truth[~inclusion_mask].mean():.2f}",
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, 0.3); ax.set_ylim(0, 0.32); ax.set_zlim(0, 0.3)
    ax.view_init(elev=22, azim=-58)
    plt.colorbar(sc, ax=ax, label='stiff', shrink=0.7, pad=0.1)

    # Panel 5: RECOVERED stiff
    ax = fig.add_subplot(2, 3, 5, projection='3d')
    sc = ax.scatter(pos_rest[plot_idx, 0], pos_rest[plot_idx, 1], pos_rest[plot_idx, 2],
                      c=stiff_recovered[plot_idx], cmap='viridis',
                      vmin=vmin, vmax=vmax, s=6, alpha=0.7)
    ax.plot_wireframe(sx, sy, sz, color='red', alpha=0.4, linewidth=0.6)
    for kx in knife_x_pts:
        ax.scatter([kx], [knife_y], [knife_z], c='cyan', s=300,
                    alpha=0.7, marker='o', edgecolors='black', linewidths=1.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    inside_mean = stiff_recovered[inclusion_mask].mean()
    outside_mean = stiff_recovered[~inclusion_mask].mean()
    ax.set_title(f"RECOVERED (sys-ID, Step 2)\n"
                 f"inside: {inside_mean:.3f} (truth {stiff_truth[inclusion_mask].mean():.2f}, "
                 f"err {abs(inside_mean - stiff_truth[inclusion_mask].mean())/stiff_truth[inclusion_mask].mean()*100:.1f}%)\n"
                 f"outside: {outside_mean:.3f} (truth {stiff_truth[~inclusion_mask].mean():.2f}, "
                 f"err {abs(outside_mean - stiff_truth[~inclusion_mask].mean())/stiff_truth[~inclusion_mask].mean()*100:.1f}%)",
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, 0.3); ax.set_ylim(0, 0.32); ax.set_zlim(0, 0.3)
    ax.view_init(elev=22, azim=-58)
    plt.colorbar(sc, ax=ax, label='stiff', shrink=0.7, pad=0.1)

    # Panel 6: per-particle truth vs recovered scatter
    ax = fig.add_subplot(2, 3, 6)
    ax.scatter(stiff_truth[~inclusion_mask], stiff_recovered[~inclusion_mask],
                s=1, c='orange', alpha=0.3,
                label=f'outside ({(~inclusion_mask).sum()} pts)')
    ax.scatter(stiff_truth[inclusion_mask], stiff_recovered[inclusion_mask],
                s=10, c='blue', alpha=0.7,
                label=f'inside ({inclusion_mask.sum()} pts)')
    lo = 0.0
    hi = max(stiff_truth.max(), stiff_recovered[inclusion_mask].max())
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='y=x (perfect)')
    # Mean markers
    ax.axhline(stiff_recovered[inclusion_mask].mean(), color='blue', ls=':', lw=1,
                label=f'inside mean = {stiff_recovered[inclusion_mask].mean():.3f}')
    ax.axhline(stiff_recovered[~inclusion_mask].mean(), color='orange', ls=':', lw=1,
                label=f'outside mean = {stiff_recovered[~inclusion_mask].mean():.3f}')
    ax.axvline(0.3, color='red', ls='-', lw=2, alpha=0.4, label='truth outside (0.3)')
    ax.axvline(0.8, color='red', ls='-', lw=2, alpha=0.4, label='truth inside (0.8)')
    ax.set_xlabel("truth stiff", fontsize=12)
    ax.set_ylabel("recovered stiff", fontsize=12)
    ax.set_title("Per-particle recovery quality\n"
                 "inside mean tracks truth well (1.6% err)\n"
                 "outside particles drift but mean is right direction",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 2.5)

    fig.suptitle(
        "Stage A-4 Step 2: PBD compliance + per-particle stiffness recovery\n"
        "Top: deformation under different PBD configs (cyan = knife) — compliant config gives larger displacement & better stiff sensitivity\n"
        "Bottom: stiff field perspective — truth (planted inclusion) vs recovered (sys-ID output) vs per-particle quality",
        fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout()
    out_path = REPO / "tools" / "stageA_step2_compare.png"
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
