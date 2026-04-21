"""Interactive 3D visualization of the training setup.

Shows: tumor particles, nerve particles, fixed particles, knife position/size.
Uses PyVista for interactive 3D viewing.

Usage:
    python warp_sim/visualize_setup.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pf_shape_matching_warp"))

import warp as wp
wp.init()

from warp_sim.envs.debulk_env import DebulkEnv

cfg = {
    "mesh": "resources/cube/cube_dense.msh",
    "settle_steps": 200,
    "sim_substeps": 2,
    "action_scale": 0.01,
    "knife_start": [0.95, 0.7, 0.5],
    "knife_radius": 0.1,
    "episode_length": 128,
    "lambda_nerve": 10.0,
    "dt": 0.005,
    "num_iterations": 4,
    "stiffness": 0.6,
    "damping": 0.98,
    "cutting_interval": 1,
}

env = DebulkEnv(cfg)
obs, _ = env.reset()
wp.synchronize()

# Get particle data
pos = env.solver.p["pos"].numpy().reshape(-1, 3)
alpha = env.solver.p["alpha"].numpy()
fixed = env.solver.p["fixed"].numpy()
nerve = env.nerve_mask.numpy()
N = len(pos)

# Classify particles
tumor_mask = (fixed == 0) & (nerve == 0)
nerve_mask = nerve == 1
fixed_mask = fixed == 1

print(f"Total particles: {N}")
print(f"  Tumor: {tumor_mask.sum()} ({tumor_mask.sum()*100//N}%)")
print(f"  Nerve: {nerve_mask.sum()} ({nerve_mask.sum()*100//N}%)")
print(f"  Fixed: {fixed_mask.sum()} ({fixed_mask.sum()*100//N}%)")
print(f"  Mesh bounds: x=[{pos[:,0].min():.2f}, {pos[:,0].max():.2f}]")
print(f"               y=[{pos[:,1].min():.2f}, {pos[:,1].max():.2f}]")
print(f"               z=[{pos[:,2].min():.2f}, {pos[:,2].max():.2f}]")

knife_pos = np.array(cfg["knife_start"])
knife_r = cfg["knife_radius"]
print(f"\n  Knife start: {knife_pos}")
print(f"  Knife radius: {knife_r}")

# Try PyVista first, fall back to matplotlib
try:
    import pyvista as pv

    print("\nUsing PyVista (interactive 3D)...")

    pl = pv.Plotter(window_size=[1200, 800])
    pl.set_background("white")

    # Tumor particles (blue)
    if tumor_mask.sum() > 0:
        tumor_pts = pos[tumor_mask]
        # Subsample if too many
        stride = max(1, len(tumor_pts) // 3000)
        tumor_sub = tumor_pts[::stride]
        tumor_cloud = pv.PolyData(tumor_sub)
        pl.add_mesh(tumor_cloud, color="dodgerblue", point_size=4,
                    render_points_as_spheres=True, opacity=0.6, label="Tumor")

    # Nerve particles (red)
    if nerve_mask.sum() > 0:
        nerve_pts = pos[nerve_mask]
        stride = max(1, len(nerve_pts) // 2000)
        nerve_sub = nerve_pts[::stride]
        nerve_cloud = pv.PolyData(nerve_sub)
        pl.add_mesh(nerve_cloud, color="red", point_size=5,
                    render_points_as_spheres=True, opacity=0.8, label="Nerve")

    # Fixed particles (gray)
    if fixed_mask.sum() > 0:
        fixed_pts = pos[fixed_mask]
        stride = max(1, len(fixed_pts) // 1000)
        fixed_sub = fixed_pts[::stride]
        fixed_cloud = pv.PolyData(fixed_sub)
        pl.add_mesh(fixed_cloud, color="gray", point_size=3,
                    render_points_as_spheres=True, opacity=0.3, label="Fixed (ground)")

    # Knife sphere (green, transparent)
    knife_sphere = pv.Sphere(radius=knife_r, center=knife_pos)
    pl.add_mesh(knife_sphere, color="green", opacity=0.4, label="Knife (CUSA tool)")

    # Knife center marker
    knife_pt = pv.PolyData(knife_pos.reshape(1, 3))
    pl.add_mesh(knife_pt, color="darkgreen", point_size=12,
                render_points_as_spheres=True, label="Knife center")

    # Bounding box
    bounds = [pos[:, 0].min() - 0.1, pos[:, 0].max() + 0.1,
              pos[:, 1].min() - 0.1, pos[:, 1].max() + 0.1,
              pos[:, 2].min() - 0.1, pos[:, 2].max() + 0.1]
    pl.add_bounding_box(color="black", line_width=1)

    # Axes and labels
    pl.add_axes()
    pl.add_legend(loc="upper right", size=(0.2, 0.2))
    pl.add_title("P(F) Debulking Setup\n"
                 f"Knife at ({knife_pos[0]:.2f}, {knife_pos[1]:.2f}, {knife_pos[2]:.2f}), "
                 f"radius={knife_r}", font_size=12)

    # Add text annotations
    pl.add_text(f"Tumor: {tumor_mask.sum()} particles (blue)\n"
                f"Nerve: {nerve_mask.sum()} particles (red)\n"
                f"Fixed: {fixed_mask.sum()} particles (gray)",
                position="lower_left", font_size=9, color="black")

    # Show nerve boundary line
    nerve_y_max = pos[nerve_mask, 1].max() if nerve_mask.sum() > 0 else 0
    print(f"  Nerve top boundary: y={nerve_y_max:.3f}")

    # Add a horizontal plane at nerve boundary
    plane_center = [pos[:, 0].mean(), nerve_y_max, pos[:, 2].mean()]
    plane_size = pos[:, 0].max() - pos[:, 0].min()
    nerve_plane = pv.Plane(center=plane_center, direction=(0, 1, 0),
                           i_size=plane_size * 1.2, j_size=plane_size * 1.2)
    pl.add_mesh(nerve_plane, color="red", opacity=0.15, label="Nerve boundary")

    print("\nOpening interactive viewer...")
    print("  - Rotate: left click + drag")
    print("  - Zoom: scroll wheel")
    print("  - Pan: middle click + drag")
    print("  - Close: q or close window")

    pl.show()

except ImportError:
    print("\nPyVista not available. Using matplotlib (non-interactive)...")
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Subsample
    stride = max(1, N // 3000)

    if tumor_mask.sum() > 0:
        t = pos[tumor_mask][::stride]
        ax.scatter(t[:, 0], t[:, 2], t[:, 1], c="dodgerblue", s=2, alpha=0.5, label="Tumor")
    if nerve_mask.sum() > 0:
        n = pos[nerve_mask][::stride]
        ax.scatter(n[:, 0], n[:, 2], n[:, 1], c="red", s=4, alpha=0.7, label="Nerve")
    if fixed_mask.sum() > 0:
        f = pos[fixed_mask][::stride]
        ax.scatter(f[:, 0], f[:, 2], f[:, 1], c="gray", s=1, alpha=0.3, label="Fixed")

    # Knife sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    sx = knife_pos[0] + knife_r * np.outer(np.cos(u), np.sin(v))
    sy = knife_pos[2] + knife_r * np.outer(np.sin(u), np.sin(v))
    sz = knife_pos[1] + knife_r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(sx, sy, sz, color="green", alpha=0.3)
    ax.scatter(*knife_pos[[0, 2, 1]], c="darkgreen", s=100, marker="*", label="Knife center")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (up)")
    ax.legend(loc="upper left")
    ax.set_title(f"P(F) Debulking Setup\nKnife at {knife_pos}, r={knife_r}")

    plt.tight_layout()
    plt.savefig("setup_visualization.png", dpi=150)
    print("Saved: setup_visualization.png")
    plt.show()
