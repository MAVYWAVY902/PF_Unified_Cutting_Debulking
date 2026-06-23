"""FIX-CHECK: num_iter=1 self-supports terribly (diag_water_settle: 35% droop). The ONLY reason we're
stuck at num_iter=1 is the CUTTING blow-up. So: does num_iter=6 + muller_stiff_remap stay STABLE through a
real raster cut AND keep the body firm? Settle 60 (no knife) → raster cut. Report:
  sag(mm) after settle (firmness)  |  max|disp|(mm) during cut (blow-up if huge)  |  NaN?
Configs span the tension: video baseline vs firm-but-raw vs firm+remap.
"""
import sys, os, io, contextlib, numpy as np, torch
from pathlib import Path
_DEVNULL = open(os.devnull, "w")   # swallow warp's per-launch R-W-R stdout spam during stepping
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
wp.config.verify_autograd_array_access = False   # silence per-launch R-W-R warnings (forward-only diag)
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster

cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.enable_cutting = True; solver.tau_soften = False
pr0 = mi["vertices"]; top = cfg["top_axis"]; kr = cfg["knife_radius_abs"]
solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
top0 = pr0[:, top].max(); far = pr0.max(0) + 10.0
tht = torch.tensor([3., 20., 0.3], device=dev)
print(f"N={N}  settle 60 (no knife) → raster cut. gravity=-9.81 damping={solver.damping}")
print(f"\n{'num_iter':>8s} {'k':>5s} {'remap':>6s}  {'sag_settle(mm)':>14s} {'maxdisp_cut(mm)':>15s} {'NaN':>5s}  verdict")

CONFIGS = [(1, 0.5, False), (1, 0.9, False), (6, 0.9, False), (2, 0.9, True), (6, 0.9, True)]


def run(num_iter, k, remap):
    solver.num_iterations = num_iter; solver.muller_stiff_remap = remap
    solver.reset_to_rest(pr0); st = rest0
    sw = torch.tensor(np.full(N, k, np.float32), device=dev); tw = torch.tensor(np.full(N, 0.3, np.float32), device=dev)
    with torch.no_grad(), contextlib.redirect_stdout(_DEVNULL):
        for _ in range(60):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, tw); st = solver_save_state(solver)
    pr = pw.cpu().numpy(); sag = (top0 - pr[:, top].max()) * 1e3
    bb0, bb1 = pr.min(0), pr.max(0)
    y = bb1[top] - 4.0 * 1e-3 + kr
    base = make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)
    one = [kp.copy().astype(np.float32) for kp in base]
    for kp in one: kp[top] = np.float32(y)
    traj = one * 3
    maxd = 0.0; nan = False
    with torch.no_grad(), contextlib.redirect_stdout(_DEVNULL):
        for kp in traj:
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, kp, kr, sw, tw); st = solver_save_state(solver)
            p = pw.cpu().numpy()
            if not np.isfinite(p).all(): nan = True; break
            d = float(np.linalg.norm(p - pr, axis=1).max() * 1e3); maxd = max(maxd, d)
    return sag, maxd, nan


for ni, k, rm in CONFIGS:
    sag, maxd, nan = run(ni, k, rm)
    blow = nan or maxd > 1000
    firm = sag < 20
    v = ("BLOW-UP" if blow else ("FIRM+STABLE ✓" if firm else "stable but soft"))
    print(f"{ni:>8d} {k:>5.1f} {str(rm):>6s}  {sag:>14.1f} {maxd:>15.1f} {str(nan):>5s}  {v}", flush=True)
print("\nKEY: want a row with small sag_settle (firm, not water) AND finite/small maxdisp_cut (cut doesn't blow).")
print("If num_iter=6+remap is FIRM+STABLE → video fix = num_iter=6 + muller_stiff_remap (firm body, stable cut).")
