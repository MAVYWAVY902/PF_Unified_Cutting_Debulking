"""Discriminator: does num_iter=8 blow up WITHOUT cutting (knife far, alpha static=1)?
 - blows without cut  => pure Jacobi-over-overlapping-clusters instability (stiff too high
                         for the cluster overlap; needs under-relaxation), cut-independent.
 - stable without cut, blows with cut => the alpha CHANGE during cutting drives it.
Run 180 frames at num_iter 2 & 8, (a) knife FAR (no cut), (b) knife raster cut.
"""
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster

cfg = MESH_CONFIGS[0]; dev = "cuda:0"; TAU = 0.30; THETA = np.array([3., 20., 0.3], np.float32)
solver, rest_state, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.substeps = 1; solver.num_iterations = 2
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
tht = torch.tensor(THETA, device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)
solver.reset_to_rest(pr0); FAR = pr0.max(0) + 10.; st = rest_state
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, FAR, kr, swt(0.5), swt(TAU)); st = solver_save_state(solver)
prs = pw.cpu().numpy().copy(); REST = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
bb0, bb1 = prs.min(0), prs.max(0); sp = bb1 - bb0; center = 0.5 * (bb0 + bb1); diag = float(np.linalg.norm(sp))
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
cut_traj = make_raster(bb0, bb1, top, sweep, la, 0.93, 6, 30)
far_traj = [FAR.astype(np.float32)] * len(cut_traj)
print(f"N={N} diag={diag:.3f}m  traj={len(cut_traj)} frames")


def run(ni, traj):
    solver.num_iterations = ni; solver.collision_in_loop = True
    solver.reset_to_rest(prs); st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in REST.items()}
    sw, twt = swt(0.5), swt(TAU); peak = 0.0
    with torch.no_grad():
        for kp in traj:
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, kp, kr, sw, twt)
            st = solver_save_state(solver); p = st["pos"]
            if not np.isfinite(p).all():
                peak = float("inf"); break
            peak = max(peak, float(np.abs(p - center).max()))
    return peak


print(f"\n  {'num_iter':>8s} {'NO-cut(far)':>14s} {'WITH-cut':>14s}")
for ni in (2, 8):
    pf = run(ni, far_traj); pc = run(ni, cut_traj)
    fs = f"{pf:.3f}m" if pf < 1e4 else f"{pf:.1e}m"
    cs = f"{pc:.3f}m" if pc < 1e4 else f"{pc:.1e}m"
    print(f"  {ni:8d} {fs:>14s} {cs:>14s}", flush=True)
solver.num_iterations = 2
