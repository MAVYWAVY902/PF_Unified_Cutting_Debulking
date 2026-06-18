"""Prove the bridging + show split fixes it. Slow deep debulk; count BRIDGING clusters
(alive members spanning >=2 connected components of the alive graph). Then run WITH
split + settle frames, render 2-panel (OFF floats | ON falls). alpha = FIRST return.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster
from tools.pointnetpp.warp_cluster_split import split_clusters_inplace

TAU_HARD = 0.55; TAU_SOFT = 0.15
OUT = REPO / "tools/pointnetpp/viz_out"
LAYERS = [0.90, 0.82, 0.74]; T_LINE = 35; N_LINES = 10; SETTLE = 70; SPLIT_EVERY = 5; CAP = 4


def bridging_count(solver, ea, eb):
    alpha = solver.p["alpha"].numpy(); N = solver.N
    alive = alpha > 0.5
    keep = alive[ea] & alive[eb]
    A = csr_matrix((np.ones(keep.sum() * 2, np.int8),
                    (np.concatenate([ea[keep], eb[keep]]), np.concatenate([eb[keep], ea[keep]]))), shape=(N, N))
    n_comp, lab = connected_components(A, directed=False)
    comp_sz = np.bincount(lab, weights=alive.astype(np.int64), minlength=n_comp)
    nclumps = int((comp_sz >= 20).sum())
    starts = solver.c["starts"].numpy(); sizes = solver.c["sizes"].numpy(); pids = solver.c["particle_ids"].numpy()
    bridging = 0
    for cid in range(solver.num_clusters):
        s0 = starts[cid]; n = sizes[cid]
        if n < 2: continue
        m = pids[s0:s0 + n]; am = m[alive[m]]
        if len(am) >= 2 and len(np.unique(lab[am])) >= 2:
            bridging += 1
    return bridging, nclumps, int(alive.sum())


def run(do_split):
    cfg = MESH_CONFIGS[0]; device = "cuda:0"
    solver, rest_state, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
    N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
    pos_rest = mi["vertices"]
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos_rest[:, 2].min() - 6.0)
    kr = cfg["knife_radius_abs"]; ta = cfg["top_axis"]
    solver.reset_to_rest(pos_rest); far = pos_rest.max(0) + 10.0
    sw = torch.tensor(np.full(N, 0.5, np.float32), device=device)
    twt = torch.tensor(np.full(N, 0.30, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device); st = rest_state
    with torch.no_grad():
        for _ in range(60):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, twt); st = solver_save_state(solver)
    pos_rest = pw.cpu().numpy().copy(); rest_state = st
    bb0, bb1 = pos_rest.min(0), pos_rest.max(0); sp = bb1 - bb0
    lat = [a for a in range(3) if a != ta]; sweep, latx = lat[0], lat[1]
    xn = (pos_rest[:, sweep] - bb0[sweep]) / sp[sweep]; zn = (pos_rest[:, latx] - bb0[latx]) / sp[latx]
    hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0)
    tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
    tau_t = torch.tensor(tau_truth, device=device); stiff_t = torch.tensor(np.full(N, 0.5, np.float32), device=device)
    theta = torch.tensor([3., 20., 0.3], dtype=torch.float32, device=device)
    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    traj = []
    for yf in LAYERS:
        traj += make_raster(bb0, bb1, ta, sweep, latx, yf, N_LINES, T_LINE)
    knife_seq = traj + [far.astype(np.float32)] * SETTLE
    solver.reset_to_rest(pos_rest); st = rest_state
    POS = []; AL = []
    with torch.no_grad():
        for i, knife in enumerate(knife_seq):
            radius = kr if i < len(traj) else 0.0
            a, p, _ = WarpFullForwardOp.apply(theta, solver, st, knife, radius, stiff_t, tau_t)
            if do_split and i % SPLIT_EVERY == 0:
                split_clusters_inplace(solver, drop_small=True)
            st = solver_save_state(solver)
            if i % CAP == 0:
                POS.append(p.cpu().numpy().copy()); AL.append((a.cpu().numpy() > 0.5).copy())
    br = bridging_count(solver, ea, eb)
    return POS, AL, br, (bb0, bb1, sweep, latx, ta, tau_truth)


print("=== Part 1: NO-split — bridging proof ===", flush=True)
posOFF, alOFF, infoOFF, geo = run(False)
print(f"  NO-split final: bridging_clusters={infoOFF[0]}  clumps(>=20)={infoOFF[1]}  alive={infoOFF[2]}", flush=True)
print("=== Part 2: SPLIT ===", flush=True)
posON, alON, infoON, _ = run(True)
print(f"  SPLIT final:    bridging_clusters={infoON[0]}  clumps(>=20)={infoON[1]}  alive={infoON[2]}", flush=True)
print(f"  => bridging {infoOFF[0]} -> {infoON[0]}  (split severs the cross-gap coupling)", flush=True)

bb0, bb1, sweep, latx, ta, tau_truth = geo
PX, PY, PZ = sweep, latx, ta
xl = (bb0[PX], bb1[PX]); yl = (bb0[PY], bb1[PY]); zl = (bb0[PZ] - 0.45, bb1[PZ])
nF = min(len(posOFF), len(posON))
fig = plt.figure(figsize=(11, 5)); axes = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]


def draw(i):
    for ax, POS, AL, lab in zip(axes, [posOFF, posON], [alOFF, alON], ["split OFF (bridged, floats)", "split ON (severed, falls)"]):
        ax.clear()
        p = POS[i]; alive = AL[i]
        ax.scatter(p[alive, PX], p[alive, PY], p[alive, PZ], c=tau_truth[alive], cmap="coolwarm",
                   vmin=TAU_SOFT, vmax=TAU_HARD, s=4, alpha=0.85, edgecolors="none")
        ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(elev=18, azim=-60)
        try:
            ax.set_box_aspect((xl[1]-xl[0], yl[1]-yl[0], zl[1]-zl[0]))
        except Exception:
            pass
        ax.set_title(f"warp  {lab}", fontsize=11)
    fig.suptitle(f"slow deep debulk + settle  frame {i}/{nF}  (red=hard τ, blue=soft τ)", fontsize=11)
    return []


anim = FuncAnimation(fig, draw, frames=nF, blit=False)
gif = OUT / "split_proof_2panel.gif"; t0 = time.time()
anim.save(str(gif), writer=PillowWriter(fps=10)); plt.close(fig)
print(f"-> saved {gif}  ({nF} frames, {time.time()-t0:.0f}s)", flush=True)
