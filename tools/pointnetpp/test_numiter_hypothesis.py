"""Decisive test of the 'contact iteration diverges' hypothesis: vary num_iterations
(Jacobi shape-matching passes per substep) on the SAME canonical shallow cut.
If iteration-divergence is the cause:
   more iters -> earlier/worse blow-up; num_iter=1 markedly more stable.
If it's something else: onset roughly invariant to num_iter.

Canonical shallow raster y_frac=0.93, 6x30=180 frames, stiff 0.5 & 0.7.
Report onset frame, distance & removed% at onset, peak max|pos-c|.
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
N = solver.N; solver.damping = 0.95; solver.substeps = 1
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]; dt = solver.dt
tht = torch.tensor(THETA, device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)
# settle with the production default num_iter=2
solver.num_iterations = 2
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.; st = rest_state
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, swt(0.5), swt(TAU)); st = solver_save_state(solver)
prs = pw.cpu().numpy().copy(); REST = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
bb0, bb1 = prs.min(0), prs.max(0); sp = bb1 - bb0; center = 0.5 * (bb0 + bb1); diag = float(np.linalg.norm(sp))
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
traj = make_raster(bb0, bb1, top, sweep, la, 0.93, 6, 30)
kd = np.array([0.0] + [float(np.linalg.norm(traj[i] - traj[i-1])) for i in range(1, len(traj))]); cumdist = np.cumsum(kd)
print(f"N={N} diag={diag:.3f}m  traj={len(traj)} frames  (settle done at num_iter=2)")


def run(stiff, ni):
    solver.num_iterations = ni
    solver.reset_to_rest(prs); st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in REST.items()}
    sw, twt = swt(stiff), swt(TAU); onset = None; peak = 0.0; rem = None
    with torch.no_grad():
        for step, kp in enumerate(traj):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, kp.astype(np.float32), kr, sw, twt)
            st = solver_save_state(solver)
            p = st["pos"]
            if not np.isfinite(p).all():
                onset = onset if onset is not None else step; peak = float("inf"); break
            mp = float(np.abs(p - center).max()); peak = max(peak, mp)
            if onset is None and mp > 2 * diag:
                onset = step; rem = float((solver.p["alpha"].numpy() < 0.5).mean()) * 100
    return onset, (cumdist[onset] if onset is not None else None), rem, peak


for stiff in (0.5, 0.7):
    print(f"\n=== stiff {stiff} ===")
    print(f"  {'num_iter':>8s} {'onset':>8s} {'dist@onset':>11s} {'rem%@onset':>10s} {'peak|pos|':>12s}")
    for ni in (1, 2, 4, 8):
        onset, dist, rem, peak = run(stiff, ni)
        on = "NEVER" if onset is None else f"f{onset}"
        ds = "-" if dist is None else f"{dist*1000:.0f}mm"
        rs = "-" if rem is None else f"{rem:.1f}%"
        pk = f"{peak:.2f}m" if peak < 1e4 else f"{peak:.1e}m"
        print(f"  {ni:8d} {on:>8s} {ds:>11s} {rs:>10s} {pk:>12s}", flush=True)
solver.num_iterations = 2
