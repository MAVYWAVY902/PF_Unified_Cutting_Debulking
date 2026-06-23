"""Replicate the firm video's obs-gen EXACTLY (settle 140 → gen_obs over cut traj) and print the cube's
sag at each cut frame. If sag is flat ~7.9mm → cube is static, the cut just deforms it. If sag keeps
growing during the cut → it's still settling when debulk starts (the bug the user sees) → find why.
"""
import sys, os, contextlib, numpy as np, torch
from pathlib import Path
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
_DN = open(os.devnull, "w")
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster

TAU_HARD, TAU_SOFT, STIFF, NUM_ITER, REACH_MM, R = 0.55, 0.15, 0.9, 6, 4.0, 3
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = NUM_ITER; solver.enable_cutting = True; solver.muller_stiff_remap = False
pr0 = mi["vertices"]; top = cfg["top_axis"]; kr = cfg["knife_radius_abs"]
solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
top0 = pr0[:, top].max(); far = pr0.max(0) + 10.0
tht = torch.tensor([3., 20., 0.3], device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)

# --- settle 140 (exactly like the video) ---
solver.reset_to_rest(pr0); st = rest0
with torch.no_grad(), contextlib.redirect_stdout(_DN):
    for _ in range(140):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, swt(STIFF), swt(0.3)); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
sag_settle = (top0 - pr[:, top].max()) * 1e3
print(f"sag after 140-settle = {sag_settle:.2f} mm (should be ~7.9, static)")

# --- build cut traj (exactly like the video) ---
bb0, bb1 = pr.min(0), pr.max(0)
xn = (pr[:, sweep] - bb0[sweep]) / (bb1[sweep] - bb0[sweep]); zn = (pr[:, la] - bb0[la]) / (bb1[la] - bb0[la])
hard = ((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2
tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
y_center = bb1[top] - REACH_MM * 1e-3 + kr
base = make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)
one = [kp.copy().astype(np.float32) for kp in base]
for kp in one: kp[top] = np.float32(y_center)
traj = one * R

# --- gen_obs exactly like the video, then read sag per obs frame ---
solver.num_iterations = NUM_ITER; solver.tau_soften = True
with contextlib.redirect_stdout(_DN):
    op, oa = gen_obs_per_particle_stiff(solver, rest_state, np.array([3., 20., 0.3], np.float32),
                                        np.full(N, STIFF, np.float32), traj, kr, dev, tau_truth_np=tau_truth)
op = np.asarray(op)
print(f"\n{'cutframe':>8s} {'sag(mm)':>8s} {'Δfrom settle':>12s}")
for f in list(range(0, 20)) + [30, 50, 80, 120, len(op) - 1]:
    s = (top0 - op[f][:, top].max()) * 1e3
    print(f"{f:>8d} {s:>8.2f} {s - sag_settle:>12.2f}", flush=True)
print("\nKEY: if sag jumps up in the first ~10 cut frames then keeps drifting → cube re-settles at cut start (bug).")
print("If sag stays ~7.9 flat → cube is static; what looks like 'sinking' is the knife pushing it down locally.")
