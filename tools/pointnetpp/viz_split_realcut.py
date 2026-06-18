"""Real knife cut + cluster split visual comparison. Knife rasters a horizontal slab
through the cube (real damage->removal), severing the top. Compare NO-split (top bridges
/ floats) vs SPLIT (top decouples + falls). Renders final frame side by side.
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
from tools.pointnetpp.test_group_generalize import make_raster
from tools.pointnetpp.warp_cluster_split import split_clusters_inplace

cfg = MESH_CONFIGS[0]; device = "cuda:0"; top_axis = cfg["top_axis"]
kr = cfg["knife_radius_abs"]
N_LINES = 16; T_LINE = 20; SETTLE = 150; SPLIT_EVERY = 5
TAU = 0.30
CUT_FRACS = [0.50]              # single slab; gravity OFF during cut => no over-cut feedback


def run(do_split):
    solver, rest_state, mi = setup_solver_with_rest(str(REPO / cfg["path"]), rings=2, fix_threshold_fraction=0.20)
    N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
    pos0 = mi["vertices"].copy()
    bb0, bb1 = pos0.min(0), pos0.max(0); sp = bb1 - bb0
    fr = (pos0[:, top_axis] - bb0[top_axis]) / sp[top_axis]
    lateral = [a for a in range(3) if a != top_axis]
    solver.floor_y = float(pos0[:, 2].min() - 5.0)
    sw = torch.tensor(np.full(N, 0.5, np.float32), device=device)
    twt = torch.tensor(np.full(N, TAU, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device)
    far = pos0.max(0) + 100.0

    traj = []
    for cfrac in CUT_FRACS:
        traj += make_raster(bb0, bb1, top_axis, lateral[0], lateral[1], cfrac, N_LINES, T_LINE)
    knife_seq = [np.asarray(p, np.float32) for p in traj] + [far.astype(np.float32)] * SETTLE

    solver.reset_to_rest(pos0)
    st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rest_state.items()}
    cmid = np.mean(CUT_FRACS)
    roof = fr > (max(CUT_FRACS) + 0.06); base = fr < (min(CUT_FRACS) - 0.06)
    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    roof_top0 = pos0[roof, top_axis].mean()
    nC = solver.num_clusters
    with torch.no_grad():
        for i, kp in enumerate(knife_seq):
            cutting = i < len(traj)
            # gravity OFF during cut (no over-cut feedback from sagging), ON during settle (top falls)
            solver.gravity = wp.vec3(0.0, 0.0, 0.0) if cutting else wp.vec3(0.0, -9.81, 0.0)
            _, pw, aw = WarpFullForwardOp.apply(tht, solver, st, kp, kr, sw, twt)
            if do_split and i % SPLIT_EVERY == 0:
                info = split_clusters_inplace(solver, drop_small=True); nC = info["C_after"]
            st = solver_save_state(solver)
    pf = pw.cpu().numpy(); alpha = aw.cpu().numpy()
    alive = alpha > 0.5
    roof_drop = (roof_top0 - pf[roof & alive, top_axis].mean()) * 1000
    # diagnostics: removed count + is roof disconnected from base?
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    keep = alive[ea] & alive[eb]
    A = csr_matrix((np.ones(keep.sum() * 2, np.int8),
                    (np.concatenate([ea[keep], eb[keep]]), np.concatenate([eb[keep], ea[keep]]))),
                   shape=(N, N))
    _, lab = connected_components(A, directed=False)
    severed = (np.intersect1d(np.unique(lab[roof & alive]), np.unique(lab[base & alive])).size == 0)
    print(f"  [{'SPLIT' if do_split else 'NO-split'}] removed={int((~alive).sum())}  "
          f"roof-base SEVERED={severed}  roof_drop={roof_drop:.0f}mm  C={nC}", flush=True)
    return pos0, pf, alive, roof, base, roof_drop, nC


fig = plt.figure(figsize=(13, 6))
for i, ds in enumerate([False, True]):
    pos0, pf, alive, roof, base, drop, nC = run(ds)
    sh = alive & ~roof & ~base   # cut-zone survivors
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    for mask, col, a in [(base & alive, "steelblue", 0.35), (sh, "gray", 0.4), (roof & alive, "crimson", 0.6)]:
        P = pf[mask]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=2, c=col, alpha=a, linewidths=0)
    ax.set_title(f"{'SPLIT' if ds else 'NO-split'}   roof drop={drop:.0f}mm   #clusters={nC}", fontsize=12)
    ax.view_init(elev=8, azim=-70); ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
fig.suptitle(f"Real knife cut (raster slab @ fr={CUT_FRACS}, gravity off during cut) + drop   "
             f"(red=top above cut, blue=base, removed hidden)", fontsize=12)
plt.tight_layout()
out = REPO / "tools/pointnetpp/viz_out/split_realcut.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved {out}", flush=True)
