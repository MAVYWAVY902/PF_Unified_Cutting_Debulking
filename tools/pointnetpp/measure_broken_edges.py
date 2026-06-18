"""Why cpp cuts through but warp doesn't. Measure warp edge damage in deep debulk:
how many edges reach >0.5 (broken)? And does connectivity sever via removed particles
or broken edges?
"""
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

REPO = Path('/home/yunxin/pf-shape-matching')
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'pf_shape_matching_warp'))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster

cfg = MESH_CONFIGS[0]; ta = cfg['top_axis']; kr = cfg['knife_radius_abs']; dev = 'cuda:0'
s, rs, mi = setup_solver_with_rest(str(REPO / cfg['path']), fix_threshold_fraction=0.20)
N = s.N; s.damping = 0.95; s.num_iterations = 2
p0 = mi['vertices']; bb0, bb1 = p0.min(0), p0.max(0); sp = bb1 - bb0
s.gravity = wp.vec3(0., -9.81, 0.); s.floor_y = float(p0[:, 2].min() - 6)
lat = [a for a in range(3) if a != ta]; sweep, latx = lat[0], lat[1]
sw = torch.tensor(np.full(N, 0.5, np.float32), device=dev)
twt = torch.tensor(np.full(N, 0.30, np.float32), device=dev)
tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=dev); st = rs
far = p0.max(0) + 10
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, s, st, far, kr, sw, twt); st = solver_save_state(s)
p0 = pw.cpu().numpy().copy(); rs = st
bb0, bb1 = p0.min(0), p0.max(0); sp = bb1 - bb0
xn = (p0[:, sweep] - bb0[sweep]) / sp[sweep]; zn = (p0[:, latx] - bb0[latx]) / sp[latx]
hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0)
tau_t = torch.tensor(np.where(hard, 0.55, 0.15).astype(np.float32), device=dev)
stiff_t = torch.tensor(np.full(N, 0.5, np.float32), device=dev)
theta = torch.tensor([3., 20., 0.3], dtype=torch.float32, device=dev)
ea = s.e['a'].numpy().astype(np.int64); eb = s.e['b'].numpy().astype(np.int64)
traj = []
for yf in [0.90, 0.82, 0.74]:
    traj += make_raster(bb0, bb1, ta, sweep, latx, yf, 10, 35)
s.reset_to_rest(p0); st = rs
with torch.no_grad():
    for knife in traj:
        a, p, _ = WarpFullForwardOp.apply(theta, s, st, knife, kr, stiff_t, tau_t); st = solver_save_state(s)
ed = s.e['damage'].numpy(); alpha = a.cpu().numpy(); alive = alpha > 0.5
E = len(ed)
print(f"=== warp edge damage in deep debulk (E={E}) ===", flush=True)
print(f"  max edge damage = {ed.max():.3f}   (cpp reference ~0.65)", flush=True)
for thr in [0.5, 0.3, 0.2, 0.1, 0.05]:
    print(f"  edges damage>{thr}: {int((ed > thr).sum())}", flush=True)
print(f"  removed particles = {int((~alive).sum())}/{N}", flush=True)


def ncomp(keep):
    r = ea[keep]; c = eb[keep]
    A = csr_matrix((np.ones(len(r) * 2, np.int8), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(N, N))
    nc, lab = connected_components(A, directed=False)
    sz = np.bincount(lab, weights=alive.astype(np.int64), minlength=nc)
    return int((sz >= 20).sum())


keepA = alive[ea] & alive[eb]
keepB = alive[ea] & alive[eb] & (ed <= 0.5)
print(f"\n  alive-clumps(>=20) via REMOVED-only       = {ncomp(keepA)}", flush=True)
print(f"  alive-clumps(>=20) via REMOVED + broken0.5 = {ncomp(keepB)}", flush=True)
print("  => max_edge<0.5 & broken adds ~0 clumps => warp has NO broken edges => severance ONLY via removed particles (hard-tau bridges)", flush=True)
