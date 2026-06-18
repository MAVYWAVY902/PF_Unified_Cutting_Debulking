"""Clean-cut self-collision demo: a controlled horizontal slab removed (= one clean flat
cut, severs top from base) + gravity, with cluster SPLIT always ON (top is a real free
fragment). Two panels isolate SELF-COLLISION:
   left  = SPLIT, no self-collision  -> top falls straight THROUGH the base (warp had no
           particle-particle contact).
   right = SPLIT + self-collision    -> top LANDS on the base and rests (soft-penalty
           hash-grid contact, differentiable, sweet-spot k=4/it=6/d=0.8x mean-edge).
Renders a 2-panel GIF with the gravity/top axis drawn as screen-up.
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
from tools.pointnetpp.warp_cluster_split import split_clusters_inplace

cfg = MESH_CONFIGS[0]; device = "cuda:0"; ta = cfg["top_axis"]
T_EL = 2; BAND_C = 0.55; STEPS = 180; SPLIT_EVERY = 5; CAP_EVERY = 3
SC_STIFF = 4.0; SC_ITERS = 6; SC_DIST_MULT = 0.8     # sweet spot from the sweep


def run(sc_on):
    solver, rest_state, mi = setup_solver_with_rest(str(REPO / cfg["path"]), rings=2, fix_threshold_fraction=0.20)
    N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
    pos0 = mi["vertices"].copy()
    bb0, bb1 = pos0.min(0), pos0.max(0); sp = bb1 - bb0
    fr = (pos0[:, ta] - bb0[ta]) / sp[ta]
    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    mean_el = float(np.linalg.norm(pos0[ea] - pos0[eb], axis=1).mean())
    if sc_on:
        solver.use_self_collision = True
        solver.self_collision_dist = SC_DIST_MULT * mean_el
        solver.self_collision_stiffness = SC_STIFF
        solver.self_collision_iters = SC_ITERS
    Tf = T_EL * mean_el / sp[ta]
    removed = (fr >= BAND_C - Tf / 2) & (fr < BAND_C + Tf / 2)
    roof = (fr >= BAND_C + Tf / 2) & (~solver.p["fixed"].numpy().astype(bool))
    base = fr < (BAND_C - Tf / 2)
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos0[:, 2].min() - 5.0)
    sw = torch.tensor(np.full(N, 0.5, np.float32), device=device)
    twt = torch.tensor(np.full(N, 0.30, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device)
    far = pos0.max(0) + 100.0
    solver.reset_to_rest(pos0)
    st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rest_state.items()}
    st["alpha"][removed] = 0.0; st["particle_damage"][removed] = 10.0
    roof_top0 = pos0[roof, ta].mean()
    caps = []
    with torch.no_grad():
        for step in range(STEPS):
            _a, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, 0.0, sw, twt)
            if step % SPLIT_EVERY == 0:
                split_clusters_inplace(solver, drop_small=False)
            st = solver_save_state(solver)
            st["alpha"][removed] = 0.0; st["particle_damage"][removed] = 10.0
            if step % CAP_EVERY == 0:
                caps.append(pw.cpu().numpy().copy())
    drop = (roof_top0 - caps[-1][roof, ta].mean()) * 1000
    return caps, (bb0, bb1, removed, roof, base), drop


capsOFF, geo, dropOFF = run(False)
capsON, _, dropON = run(True)
bb0, bb1, removed, roof, base = geo
nF = min(len(capsOFF), len(capsON))
la0, la1 = [a for a in range(3) if a != ta]

fig = plt.figure(figsize=(11, 6))
axes = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]


def draw(i):
    panels = [(capsOFF, "SPLIT, NO self-collision", dropOFF),
              (capsON, "SPLIT + self-collision", dropON)]
    for ax, (caps, lab, dr) in zip(axes, panels):
        ax.clear()
        pf = caps[i]
        for mask, col, a in [(base & ~removed, "steelblue", 0.30), (roof & ~removed, "crimson", 0.7)]:
            P = pf[mask]; ax.scatter(P[:, la0], P[:, la1], P[:, ta], s=2, c=col, alpha=a, linewidths=0)
        ax.set_title(f"{lab}\ntop drop = {dr:.0f}mm", fontsize=11)
        ax.view_init(elev=10, azim=-70); ax.set_box_aspect([1, 1, 1.6])
        ax.set_xlim(bb0[la0], bb1[la0]); ax.set_ylim(bb0[la1], bb1[la1])
        ax.set_zlim(bb0[ta] - 0.8, bb1[ta])     # shared range: shows through-fall vs landing
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle(f"Severed TOP (red) + gravity: does self-collision catch it on the base (blue)?  "
                 f"frame {i}/{nF}", fontsize=11)


anim = FuncAnimation(fig, draw, frames=nF, interval=80)
out = REPO / "tools/pointnetpp/viz_out/split_selfcollision.gif"
anim.save(str(out), writer=PillowWriter(fps=12))
print(f"saved {out}  ({nF} frames)  dropOFF(through)={dropOFF:.0f}mm  dropON(land)={dropON:.0f}mm", flush=True)
