"""Slow raster cut in warp, side-by-side: cluster split OFF vs ON. One combined GIF.
Knife slowly rasters a mid-depth slab (gravity off during cut), then gravity. Reads
alpha as FIRST return of WarpFullForwardOp.apply.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster
from tools.pointnetpp.warp_cluster_split import split_clusters_inplace

cfg = MESH_CONFIGS[0]; device = "cuda:0"; ta = cfg["top_axis"]; kr = cfg["knife_radius_abs"]
CUT_FRAC = 0.55; N_LINES = 8; T_LINE = 35; SETTLE = 150; SPLIT_EVERY = 5
TAU = 0.30; CAP_EVERY = 4


def run(do_split):
    solver, rest_state, mi = setup_solver_with_rest(str(REPO / cfg["path"]), rings=2, fix_threshold_fraction=0.20)
    N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
    pos0 = mi["vertices"].copy(); bb0, bb1 = pos0.min(0), pos0.max(0); sp = bb1 - bb0
    fr = (pos0[:, ta] - bb0[ta]) / sp[ta]
    lat = [a for a in range(3) if a != ta]
    roof = (fr > CUT_FRAC + 0.06); base = (fr < CUT_FRAC - 0.06)
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos0[:, 2].min() - 6.0)
    sw = torch.tensor(np.full(N, 0.5, np.float32), device=device)
    twt = torch.tensor(np.full(N, TAU, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device)
    far = pos0.max(0) + 100.0
    traj = make_raster(bb0, bb1, ta, lat[0], lat[1], CUT_FRAC, N_LINES, T_LINE)
    knife_seq = [np.asarray(p, np.float32) for p in traj] + [far.astype(np.float32)] * SETTLE
    solver.reset_to_rest(pos0); st = rest_state
    caps = []
    with torch.no_grad():
        for i, kp in enumerate(knife_seq):
            cutting = i < len(traj)
            solver.gravity = wp.vec3(0., 0., 0.) if cutting else wp.vec3(0., -9.81, 0.)
            radius = kr if cutting else 0.0
            aw, pw, _ = WarpFullForwardOp.apply(tht, solver, st, kp, radius, sw, twt)
            if do_split and i % SPLIT_EVERY == 0:
                split_clusters_inplace(solver, drop_small=True)
            st = solver_save_state(solver)
            if i % CAP_EVERY == 0:
                caps.append((pw.cpu().numpy().copy(), (aw.cpu().numpy() > 0.5)))
    return caps, (bb0, bb1, roof, base)


capsOFF, geo = run(False)
capsON, _ = run(True)
bb0, bb1, roof, base = geo
nF = min(len(capsOFF), len(capsON))

fig = plt.figure(figsize=(10, 5))
axes = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]


def draw(i):
    # Map the GRAVITY/top axis (ta) to matplotlib's vertical (z) slot so "up" on screen =
    # up in the sim. Lateral axes -> screen x/y. Previously P[:,2] (a lateral axis) was the
    # vertical slot, so the fall (along axis ta) was drawn into screen-depth -> looked sideways.
    la0, la1 = [a for a in range(3) if a != ta]
    for ax, caps, lab in zip(axes, [capsOFF, capsON], ["split OFF", "split ON"]):
        ax.clear()
        pf, alive = caps[i]
        for mask, col, a in [(base & alive, "steelblue", 0.3), (roof & alive, "crimson", 0.7)]:
            P = pf[mask]; ax.scatter(P[:, la0], P[:, la1], P[:, ta], s=2, c=col, alpha=a, linewidths=0)
        ax.set_title(f"warp  {lab}", fontsize=12)
        ax.view_init(elev=10, azim=-70); ax.set_box_aspect([1, 1, 1.5])
        ax.set_xlim(bb0[la0], bb1[la0]); ax.set_ylim(bb0[la1], bb1[la1])
        ax.set_zlim(bb0[ta] - 0.7, bb1[ta])     # extend DOWN to show the roof fall
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle(f"slow raster cut @ fr={CUT_FRAC} + gravity (screen up = axis {ta})   frame {i}/{nF}", fontsize=11)


anim = FuncAnimation(fig, draw, frames=nF, interval=80)
out = REPO / "tools/pointnetpp/viz_out/raster_split_2panel.gif"
anim.save(str(out), writer=PillowWriter(fps=14))
print(f"saved {out}  ({nF} frames)", flush=True)
