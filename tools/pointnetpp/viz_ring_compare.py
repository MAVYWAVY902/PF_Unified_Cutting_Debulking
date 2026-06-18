"""Viz-only: rings=2 (current, floats) vs rings=1 (smaller clusters, separates) on the
SAME thin sub-surface slab cut + gravity. Renders final alive-particle clouds side by side.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS

cfg = MESH_CONFIGS[0]; device = "cuda:0"; top_axis = cfg["top_axis"]
T_EL = 2          # slab thickness in edge-lengths (thin cut, like one raster)
BAND_C = 0.55     # slab center (above anchor)
STEPS = 150


def run(rings):
    solver, rest_state, mi = setup_solver_with_rest(str(REPO / cfg["path"]), rings=rings, fix_threshold_fraction=0.20)
    N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
    pos = mi["vertices"].copy()
    bb0, bb1 = pos.min(0), pos.max(0); sp = bb1 - bb0
    fr = (pos[:, top_axis] - bb0[top_axis]) / sp[top_axis]
    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    mean_el = float(np.linalg.norm(pos[ea] - pos[eb], axis=1).mean())
    Tf = T_EL * mean_el / sp[top_axis]
    removed = (fr >= BAND_C - Tf / 2) & (fr < BAND_C + Tf / 2)
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos[:, 2].min() - 5.0)
    far = pos.max(0) + 100.0
    sw = torch.tensor(np.full(N, 0.5, np.float32), device=device)
    twt = torch.tensor(np.full(N, 0.30, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device)
    solver.reset_to_rest(pos); st = rest_state
    for _ in range(STEPS):
        d = solver.p["damage"].numpy(); d[removed] = 10.0; solver.p["damage"].assign(d)
        a = solver.p["alpha"].numpy(); a[removed] = 0.0; solver.p["alpha"].assign(a)
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, 0.0, sw, twt); st = solver_save_state(solver)
    pf = pw.cpu().numpy()
    roof = (fr >= BAND_C + Tf / 2) & (~solver.p["fixed"].numpy().astype(bool))
    drop = (pos[roof, top_axis].mean() - pf[roof, top_axis].mean()) * 1000
    return pos, pf, removed, roof, drop, mean_el


fig = plt.figure(figsize=(13, 6))
for i, rings in enumerate([2, 1]):
    pos, pf, removed, roof, drop, mel = run(rings)
    alive = ~removed
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    P = pf[alive]
    col = np.where(roof[alive], "crimson", "steelblue")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=2, c=col, alpha=0.5, linewidths=0)
    ax.set_title(f"rings={rings}  (cluster diam ~{3 if rings==2 else 1.5:.1f} edge)\n"
                 f"roof drop = {drop:.1f} mm", fontsize=12)
    ax.view_init(elev=8, azim=-70)
    ax.set_box_aspect([1, 1, 1]); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
fig.suptitle(f"Thin sub-surface slab removed (~{T_EL} edge) + gravity, {STEPS} steps  "
             f"(red=roof above cut, blue=base; removed hidden)", fontsize=12)
plt.tight_layout()
out = REPO / "tools/pointnetpp/viz_out/ring_compare.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved {out}", flush=True)
