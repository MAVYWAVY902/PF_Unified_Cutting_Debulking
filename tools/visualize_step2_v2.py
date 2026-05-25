"""V2 — clearer comparison.

Top row: 2D slice through knife z-plane showing displacement field.
         Only show cut-zone particles (within 8cm of knife).
         Larger dots, displacement vectors as arrows.
Bottom row: 3D perspective of recovered stiff field, multiple angles.
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
    T = 10  # user request: longer trajectory for clearer deformation differences
    knife_traj = make_knife_trajectory(T)
    theta_3d = np.array([3.0, 20.0, 0.5], dtype=np.float32)
    stiff_uniform = np.full(N, 0.5, dtype=np.float32)

    configs = [
        ("Rigid PBD\niters=6, stiff=0.5\n(snap rate 99.998%)",  6, 0.98),
        ("Medium PBD\niters=4, stiff=0.5\n(snap rate 99.6%)",   4, 0.99),
        ("Compliant PBD\niters=2, stiff=0.5\n(snap rate 93.75%)", 2, 0.99),
    ]
    deformed_pos = {}
    for label, iters, damp in configs:
        print(f"  Running {label.replace(chr(10), ' ').replace('  ', ' ')} ...")
        final = run_forward(solver, rest_state, theta_3d, stiff_uniform,
                              knife_traj, knife_radius, iters, damp, device)
        solver_load_state(solver, rest_state)
        deformed_pos[label] = final

    # Load stiff recovery data
    data = np.load(REPO / "tools" / "stageA_inclusion_recovery.npz")
    stiff_truth = data["stiff_truth"]
    stiff_recovered = data["stiff_recovered"]
    inclusion_mask = data["inclusion_mask"]
    incl_center = pos_rest[inclusion_mask].mean(axis=0)
    incl_r = 0.06

    knife_x_pts = [kp[0] for kp in knife_traj]
    knife_y = knife_traj[0][1]
    knife_z = knife_traj[0][2]
    knife_center = np.array([np.mean(knife_x_pts), knife_y, knife_z])

    # === Compute slice mask: particles in z slice near knife z ===
    z_thick = 0.04  # ±2cm of knife z
    slice_mask = np.abs(pos_rest[:, 2] - knife_z) < z_thick / 2
    print(f"\n  Slice z={knife_z}±{z_thick/2}: {slice_mask.sum()} particles")

    fig = plt.figure(figsize=(20, 13))

    # === Row 1: 2D slice displacement comparison ===
    # Use SAME color scale for direct comparison
    all_disp = []
    for _, _, _ in configs: pass
    for label, _, _ in configs:
        final = deformed_pos[label]
        disp = np.linalg.norm(final - pos_rest, axis=1) * 1000  # mm
        all_disp.append(disp[slice_mask])
    vmax_disp = max(d.max() for d in all_disp)
    # Cap at 50mm for visibility (95th percentile is more useful than max)
    vmax_disp_show = min(vmax_disp, 50.0)

    for col, (label, iters, damp) in enumerate(configs):
        ax = fig.add_subplot(2, 3, col + 1)
        final = deformed_pos[label]
        disp = np.linalg.norm(final - pos_rest, axis=1) * 1000  # mm
        # Plot rest positions colored by displacement
        pos_s = pos_rest[slice_mask]
        disp_s = disp[slice_mask]
        sc = ax.scatter(pos_s[:, 0], pos_s[:, 1], c=disp_s,
                          cmap='hot', s=12, alpha=0.85,
                          vmin=0, vmax=vmax_disp_show)
        # Knife sweep line + sphere outlines
        ax.plot([min(knife_x_pts), max(knife_x_pts)], [knife_y, knife_y],
                 'cyan', lw=3, alpha=0.7)
        for kx in knife_x_pts:
            circle = plt.Circle((kx, knife_y), knife_radius,
                                 fill=False, color='cyan', lw=1.5, alpha=0.5)
            ax.add_patch(circle)
        # Statistics
        mean_d = disp_s.mean()
        p95_d  = np.percentile(disp_s, 95)
        max_d  = disp_s.max()
        ax.set_title(f"{label}\n"
                     f"slice mean={mean_d:.2f}mm  p95={p95_d:.2f}mm  max={max_d:.2f}mm",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        ax.set_xlim(0, 0.32); ax.set_ylim(0, 0.34)
        plt.colorbar(sc, ax=ax, label='|displacement| (mm, capped 50)', shrink=0.85)

    # === Row 2: stiff field (truth | recovered) + per-particle ===

    # Use same inclusion z-slice
    pos_s = pos_rest[slice_mask]
    truth_s = stiff_truth[slice_mask]
    rec_s = stiff_recovered[slice_mask]
    incl_s = inclusion_mask[slice_mask]
    vmin = 0.0
    vmax = 1.5

    # Panel 4: Truth stiff 2D slice
    ax = fig.add_subplot(2, 3, 4)
    sc = ax.scatter(pos_s[:, 0], pos_s[:, 1], c=truth_s, cmap='viridis',
                      vmin=vmin, vmax=vmax, s=12, alpha=0.85)
    circle = plt.Circle((incl_center[0], incl_center[1]), incl_r,
                          fill=False, edgecolor='red', lw=2.5, ls='--')
    ax.add_patch(circle)
    ax.plot([min(knife_x_pts), max(knife_x_pts)], [knife_y, knife_y],
             'cyan', lw=3, alpha=0.7)
    for kx in knife_x_pts:
        knife_circ = plt.Circle((kx, knife_y), knife_radius,
                                  fill=False, color='cyan', lw=1, alpha=0.4)
        ax.add_patch(knife_circ)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"TRUTH stiff (z-slice)\n"
                 f"inside inclusion: {stiff_truth[inclusion_mask].mean():.2f}, "
                 f"outside: {stiff_truth[~inclusion_mask].mean():.2f}",
                 fontsize=11, fontweight='bold')
    ax.set_aspect("equal"); ax.set_xlim(0, 0.32); ax.set_ylim(0, 0.34)
    plt.colorbar(sc, ax=ax, label='stiff', shrink=0.85)

    # Panel 5: Recovered stiff 2D slice
    ax = fig.add_subplot(2, 3, 5)
    sc = ax.scatter(pos_s[:, 0], pos_s[:, 1], c=rec_s, cmap='viridis',
                      vmin=vmin, vmax=vmax, s=12, alpha=0.85)
    circle = plt.Circle((incl_center[0], incl_center[1]), incl_r,
                          fill=False, edgecolor='red', lw=2.5, ls='--',
                          label='true inclusion')
    ax.add_patch(circle)
    ax.plot([min(knife_x_pts), max(knife_x_pts)], [knife_y, knife_y],
             'cyan', lw=3, alpha=0.7, label='knife')
    for kx in knife_x_pts:
        knife_circ = plt.Circle((kx, knife_y), knife_radius,
                                  fill=False, color='cyan', lw=1, alpha=0.4)
        ax.add_patch(knife_circ)
    in_mean = stiff_recovered[inclusion_mask].mean()
    out_mean = stiff_recovered[~inclusion_mask].mean()
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"RECOVERED stiff (sys-ID output)\n"
                 f"inside: {in_mean:.3f} (truth 0.80, err {abs(in_mean-0.8)/0.8*100:.1f}%)\n"
                 f"outside: {out_mean:.3f} (truth 0.30, err {abs(out_mean-0.3)/0.3*100:.0f}%)",
                 fontsize=11, fontweight='bold')
    ax.set_aspect("equal"); ax.set_xlim(0, 0.32); ax.set_ylim(0, 0.34)
    ax.legend(loc='lower right', fontsize=8)
    plt.colorbar(sc, ax=ax, label='stiff', shrink=0.85)

    # Panel 6: 3D perspective view of recovered stiff
    ax = fig.add_subplot(2, 3, 6, projection='3d')
    rng = np.random.default_rng(7)
    sub = rng.choice(N, size=3000, replace=False)
    incl_idx = np.where(inclusion_mask)[0]
    plot_idx = np.unique(np.concatenate([sub, incl_idx]))
    sc = ax.scatter(pos_rest[plot_idx, 0], pos_rest[plot_idx, 1], pos_rest[plot_idx, 2],
                      c=stiff_recovered[plot_idx], cmap='viridis',
                      vmin=vmin, vmax=vmax, s=5, alpha=0.5)
    u, v = np.mgrid[0:2*np.pi:18j, 0:np.pi:10j]
    sx = incl_center[0] + incl_r * np.cos(u) * np.sin(v)
    sy = incl_center[1] + incl_r * np.sin(u) * np.sin(v)
    sz = incl_center[2] + incl_r * np.cos(v)
    ax.plot_wireframe(sx, sy, sz, color='red', alpha=0.5, linewidth=0.7)
    for kx in knife_x_pts:
        ax.scatter([kx], [knife_y], [knife_z], c='cyan', s=400,
                    alpha=0.8, marker='o', edgecolors='black', linewidths=1.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("RECOVERED stiff — 3D perspective\n"
                 "(red ball = true inclusion location\n"
                 "cyan dots = knife sweep)",
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, 0.3); ax.set_ylim(0, 0.32); ax.set_zlim(0, 0.3)
    ax.view_init(elev=28, azim=-50)

    fig.suptitle(
        "Stage A-4 Step 2: PBD compliance + per-particle stiffness recovery (z-slice through knife plane)\n"
        "Top: deformation field — compliant config has more spread / higher p95 displacement in cut zone\n"
        "Bottom: stiff field — truth inclusion at red dashed circle; recovered shows inclusion is found (slice + 3D views)",
        fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout()
    out_path = REPO / "tools" / "stageA_step2_compare_v2.png"
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
