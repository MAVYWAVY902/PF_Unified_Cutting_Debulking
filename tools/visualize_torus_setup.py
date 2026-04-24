#!/usr/bin/env python3
"""Render torus mesh overview — top view, side view, 3D, capsule shading.

Output: resources/torus/torus_overview.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pf_shape_matching_warp"))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from pf_shape_matching_warp.mesh.loader import load_tet_mesh

V, T = load_tet_mesh("resources/torus/torus.msh")
bm = np.load("resources/torus/boundary_mask.npy")

# Proposed initial knife position: just outside torus ring at right side
knife_init = np.array([0.5 + 0.35 + 0.12 + 0.03, 0.25, 0.5])  # outside ring equator (+x side)
# Alternative: at workspace corner (current cuboid default)
knife_corner = np.array([1.0, 0.5, 0.0])

fig = plt.figure(figsize=(16, 5))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.1)

# --- 1: Top-down XZ view ---
ax0 = fig.add_subplot(gs[0, 0])
ax0.scatter(V[~bm, 0], V[~bm, 2], c="royalblue", s=1.5, alpha=0.30,
            label=f"interior ({(~bm).sum()})")
ax0.scatter(V[bm, 0], V[bm, 2], c="crimson", s=2, alpha=0.55,
            label=f"capsule ({bm.sum()})")
ax0.scatter(knife_init[0], knife_init[2], c="black", s=180, marker="o",
            edgecolor="yellow", linewidth=2, label="knife start (proposed)")
ax0.scatter(knife_corner[0], knife_corner[2], c="gray", s=120, marker="X",
            label="cuboid corner-start (old)")
knife_r = 0.05
circ = plt.Circle((knife_init[0], knife_init[2]), knife_r, fill=False,
                  edgecolor="black", linestyle="--", linewidth=1.2, alpha=0.6)
ax0.add_patch(circ)
ax0.set_aspect("equal")
ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
ax0.set_xlabel("x"); ax0.set_ylabel("z")
ax0.set_title("Top-down (XZ)  — ring visible", fontsize=11)
ax0.legend(loc="upper left", fontsize=7, framealpha=0.9)
ax0.grid(alpha=0.2)

# --- 2: Side view XY (shows flat-laying torus thickness) ---
ax1 = fig.add_subplot(gs[0, 1])
ax1.scatter(V[~bm, 0], V[~bm, 1], c="royalblue", s=1.5, alpha=0.30)
ax1.scatter(V[bm, 0], V[bm, 1], c="crimson", s=2, alpha=0.55)
ax1.scatter(knife_init[0], knife_init[1], c="black", s=180, marker="o",
            edgecolor="yellow", linewidth=2)
circ2 = plt.Circle((knife_init[0], knife_init[1]), knife_r, fill=False,
                   edgecolor="black", linestyle="--", linewidth=1.2, alpha=0.6)
ax1.add_patch(circ2)
ax1.set_aspect("equal")
ax1.set_xlim(0, 1); ax1.set_ylim(0, 0.55)
ax1.set_xlabel("x"); ax1.set_ylabel("y (gravity ↓)")
ax1.set_title("Side (XY) — tube thickness in y", fontsize=11)
ax1.grid(alpha=0.2)

# --- 3: 3D perspective ---
ax2 = fig.add_subplot(gs[0, 2], projection="3d")
subset = np.random.choice(np.where(~bm)[0], min(2000, (~bm).sum()), replace=False)
ax2.scatter(V[subset, 0], V[subset, 1], V[subset, 2],
            c="royalblue", s=2, alpha=0.15, depthshade=False)
ax2.scatter(V[bm, 0], V[bm, 1], V[bm, 2],
            c="crimson", s=3, alpha=0.55, depthshade=False)
ax2.scatter([knife_init[0]], [knife_init[1]], [knife_init[2]],
            c="black", s=120, depthshade=False, edgecolor="yellow", linewidth=2)
ax2.set_xlim(0, 1); ax2.set_ylim(0, 0.55); ax2.set_zlim(0, 1)
ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
ax2.set_title("3D — doughnut laying flat", fontsize=11)
ax2.view_init(elev=28, azim=-55)

# --- 4: Cross-section (plane y=0.25, z ≈ center); shows tube cross-section ---
ax3 = fig.add_subplot(gs[0, 3])
plane_mask = (V[:, 2] > 0.49) & (V[:, 2] < 0.51)  # ~z slice through center
Vp = V[plane_mask]
bmp = bm[plane_mask]
ax3.scatter(Vp[~bmp, 0], Vp[~bmp, 1], c="royalblue", s=8, alpha=0.55,
            label=f"interior (slice)")
ax3.scatter(Vp[bmp, 0], Vp[bmp, 1], c="crimson", s=10, alpha=0.85,
            label=f"capsule (slice)")
# Draw ideal centerline path at this slice (two tube cross-sections)
theta = np.linspace(0, 2*np.pi, 64)
for cx_ring in [0.5 - 0.35, 0.5 + 0.35]:
    ax3.plot(cx_ring + 0.12 * np.cos(theta), 0.25 + 0.12 * np.sin(theta),
             "k--", linewidth=1, alpha=0.4, label="tube wall" if cx_ring > 0.5 else None)
    ax3.scatter(cx_ring, 0.25, c="green", s=40, marker="+",
                label="centerline target" if cx_ring > 0.5 else None)
ax3.set_aspect("equal")
ax3.set_xlim(0, 1); ax3.set_ylim(0, 0.55)
ax3.set_xlabel("x"); ax3.set_ylabel("y")
ax3.set_title("z-slice: tube cross-sections\n(green = centerline target)", fontsize=10)
ax3.legend(loc="upper left", fontsize=7, framealpha=0.9)
ax3.grid(alpha=0.2)

fig.suptitle(
    f"Torus tumor mesh — R={0.35}, r={0.12}  |  "
    f"{len(V)} verts ({bm.sum()} capsule + {(~bm).sum()} interior = "
    f"{bm.sum()/len(V)*100:.0f}% / {(~bm).sum()/len(V)*100:.0f}%)  |  "
    f"knife_r=0.05 → tube clearance={0.12 - 0.05:.3f}",
    fontsize=11, y=1.02,
)

out = "resources/torus/torus_overview.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Wrote {out}")
