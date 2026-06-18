"""Animated perspective GIF of debulk: progressively remove a mid slab, then gravity.
SPLIT env (0/1) toggles warp cluster split. Shows: OFF=top floats, ON=top falls.
Output: viz_out/debulk_split{ON,OFF}.gif
"""
import sys, os
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

SPLIT = os.environ.get("SPLIT", "0") == "1"
cfg = MESH_CONFIGS[0]; device = "cuda:0"; ta = cfg["top_axis"]
T_EL = 3; BAND_C = 0.55; REMOVE_FRAMES = 35; FALL_FRAMES = 120; SPLIT_EVERY = 5

solver, rest_state, mi = setup_solver_with_rest(str(REPO / cfg["path"]), rings=2, fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
pos0 = mi["vertices"].copy(); bb0, bb1 = pos0.min(0), pos0.max(0); sp = bb1 - bb0
fr = (pos0[:, ta] - bb0[ta]) / sp[ta]
lat = [a for a in range(3) if a != ta]; sweep = lat[0]
ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
mel = float(np.linalg.norm(pos0[ea] - pos0[eb], axis=1).mean())
Tf = T_EL * mel / sp[ta]
band = (fr >= BAND_C - Tf / 2) & (fr < BAND_C + Tf / 2)
roof = (fr >= BAND_C + Tf / 2) & (~solver.p["fixed"].numpy().astype(bool))
base = fr < (BAND_C - Tf / 2)
xn = (pos0[:, sweep] - bb0[sweep]) / sp[sweep]      # for progressive left->right removal
solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos0[:, 2].min() - 8.0)
sw = torch.tensor(np.full(N, 0.5, np.float32), device=device)
twt = torch.tensor(np.full(N, 0.30, np.float32), device=device)
tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device)
far = pos0.max(0) + 100.0

solver.reset_to_rest(pos0)
st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rest_state.items()}
frames = []
with torch.no_grad():
    for f in range(REMOVE_FRAMES + FALL_FRAMES):
        # progressive removal: band columns with xn < progress get removed
        if f < REMOVE_FRAMES:
            prog = (f + 1) / REMOVE_FRAMES
            rem_now = band & (xn < prog)
        st["alpha"][rem_now] = 0.0; st["particle_damage"][rem_now] = 10.0
        aw, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, 0.0, sw, twt)  # (alpha, pos, dv)
        if SPLIT and f % SPLIT_EVERY == 0:
            split_clusters_inplace(solver, drop_small=False)
        st = solver_save_state(solver)
        st["alpha"][rem_now] = 0.0; st["particle_damage"][rem_now] = 10.0
        pf = pw.cpu().numpy(); alive = aw.cpu().numpy() > 0.5
        frames.append((pf.copy(), alive.copy()))

fig = plt.figure(figsize=(5, 5)); axx = fig.add_subplot(111, projection="3d")
title = "warp  split " + ("ON (cut volume FALLS)" if SPLIT else "OFF (floats)")


def draw(i):
    axx.clear()
    pf, alive = frames[i]
    for mask, col, a in [(base & alive, "steelblue", 0.3), (roof & alive, "crimson", 0.6)]:
        P = pf[mask]; axx.scatter(P[:, 0], P[:, 1], P[:, 2], s=2, c=col, alpha=a, linewidths=0)
    axx.set_title(f"{title}\nframe {i}", fontsize=11)
    axx.view_init(elev=8, azim=-70); axx.set_box_aspect([1, 1, 1])
    lim = [[bb0[k], bb1[k]] for k in range(3)]
    lim[ta] = [bb0[ta] - 0.35, bb1[ta]]          # extend DOWN along the gravity/top axis for the fall
    axx.set_xlim(*lim[0]); axx.set_ylim(*lim[1]); axx.set_zlim(*lim[2])
    axx.set_xticks([]); axx.set_yticks([]); axx.set_zticks([])


anim = FuncAnimation(fig, draw, frames=len(frames), interval=60)
out = REPO / f"tools/pointnetpp/viz_out/debulk_split{'ON' if SPLIT else 'OFF'}.gif"
anim.save(str(out), writer=PillowWriter(fps=18))
print(f"saved {out}", flush=True)
