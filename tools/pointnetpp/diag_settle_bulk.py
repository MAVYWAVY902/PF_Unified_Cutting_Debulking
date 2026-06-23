"""DECISIVE settle test. User insists the cube keeps sinking under GRAVITY (no cut) for ~400 frames.
My earlier test used MAX-TOP sag (could be held by a few edge/pinned particles while the BULK sinks).
Here: NO knife at all, pure gravity, 500 frames. Report the BULK, not just the top:
  meanY_drop = how far the AVERAGE particle height fell from rest (mm)
  comY_drop  = centre-of-mass height drop (mm)
  maxmove    = worst single-particle move this frame (mm/frame) — equilibrium when ~0
If meanY/comY keep dropping past frame 140 → cube genuinely still settling → I was wrong (max-top hid it).
If flat after ~140 → settling done early. num_iter=6, k=0.9, damping=0.95 (the firm-video setting).
"""
import sys, os, contextlib, numpy as np, torch
from pathlib import Path
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
_DN = open(os.devnull, "w")
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS

cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.enable_cutting = True; solver.tau_soften = False; solver.muller_stiff_remap = False
solver.num_iterations = 6; solver.damping = 0.95
pr0 = mi["vertices"]; top = cfg["top_axis"]; kr = cfg["knife_radius_abs"]
solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
far = pr0.max(0) + 10.0
tht = torch.tensor([3., 20., 0.3], device=dev)
sw = torch.tensor(np.full(N, 0.9, np.float32), device=dev); tw = torch.tensor(np.full(N, 0.3, np.float32), device=dev)
meanY0 = pr0[:, top].mean(); comY0 = pr0[:, top].mean()   # uniform mass → com == mean
print(f"N={N} num_iter=6 k=0.9 damping=0.95  PURE GRAVITY, NO KNIFE, 500 frames")
print(f"\n{'frame':>5s} {'meanY_drop(mm)':>14s} {'comY_drop(mm)':>13s} {'maxmove(mm/f)':>13s}")

solver.reset_to_rest(pr0); st = rest0; prev = pr0.copy(); rows = []
with torch.no_grad(), contextlib.redirect_stdout(_DN):
    for f in range(1, 501):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, tw); st = solver_save_state(solver)
        p = pw.cpu().numpy()
        mv = float(np.linalg.norm(p - prev, axis=1).max() * 1e3); prev = p
        if f % 20 == 0 or f in (1, 5, 10, 140):
            rows.append((f, (meanY0 - p[:, top].mean()) * 1e3, (comY0 - p[:, top].mean()) * 1e3, mv))
for f, md, cd, mv in rows:
    print(f"{f:>5d} {md:>14.2f} {cd:>13.2f} {mv:>13.4f}", flush=True)
print("\nKEY: if meanY_drop is still GROWING at frame 140/200/300 → bulk still sinking (user right).")
print("If meanY_drop plateaus by ~140 → settled early; the video's later sink is NOT free-gravity settling.")
