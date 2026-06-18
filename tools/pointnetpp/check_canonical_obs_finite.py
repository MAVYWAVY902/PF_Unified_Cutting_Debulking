"""Does the ACTUAL production obs path (gen_obs_per_particle_stiff) blow up at the
canonical calibration settings? Replicate test_ls.py / test_group_generalize EXACTLY
(tau PATTERN truth 0.55/0.15, stiff 0.5, raster y_frac 0.93 & 0.88, 10x30). Report
per-frame max|pos| and max|v| of the OBS trajectory. If finite -> my standalone harness
differs from production; if it blows -> calibration runs on garbage.
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
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster

TAU_HARD = 0.55; TAU_SOFT = 0.15; STIFF = 0.5
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest_state, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
pr = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
solver.reset_to_rest(pr); far = pr.max(0) + 10.
sw = torch.tensor(np.full(N, STIFF, np.float32), device=dev); twt = torch.tensor(np.full(N, 0.30, np.float32), device=dev)
tht = torch.tensor([3., 20., 0.3], dtype=torch.float32, device=dev); st = rest_state
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, twt); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0; center = 0.5 * (bb0 + bb1); diag = float(np.linalg.norm(bs))
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
DIAG = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0)
tau_truth = np.where(DIAG, TAU_HARD, TAU_SOFT).astype(np.float32)
stiff_u = np.full(N, STIFF, np.float32); theta_truth = np.array([3., 20., 0.3], np.float32)
print(f"N={N} diag={diag:.3f}m  alpha_vel_scale={solver.alpha_vel_scale}")

for yf in (0.93, 0.88):
    traj = make_raster(bb0, bb1, top, sweep, la, yf, 10, 30)
    op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta_truth, stiff_u, traj, kr, dev, tau_truth_np=tau_truth)
    mp = [float(np.abs(p - center).max()) for p in op]
    finite = all(np.isfinite(p).all() for p in op)
    peak = max(mp); last = mp[-1]
    # alive-only on last frame
    print(f"  y_frac={yf}: frames={len(op)}  finite={finite}  peak max|pos-c|={peak:.3f}m ({peak/diag:.2f}x diag)  last={last:.3f}m")
    print(f"          per-frame max|pos| sample: {[round(m,2) for m in mp[::30]]}")
