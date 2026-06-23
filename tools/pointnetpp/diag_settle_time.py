"""How many frames until the cube is truly STILL at num_iter=6, k=0.9 (the firm-video setting)?
Settle under gravity, no knife. Print per-frame max-move (mm) over the body — when it drops below ~0.05mm
the cube has stopped and it's safe to start debulking. Reports the settle frame.
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
top0 = pr0[:, top].max(); far = pr0.max(0) + 10.0
tht = torch.tensor([3., 20., 0.3], device=dev)
sw = torch.tensor(np.full(N, 0.9, np.float32), device=dev); tw = torch.tensor(np.full(N, 0.3, np.float32), device=dev)
print(f"N={N} num_iter=6 k=0.9 damping=0.95  settle-to-still probe")
print(f"\n{'frame':>5s} {'sag(mm)':>8s} {'maxmove(mm/frame)':>18s}")

solver.reset_to_rest(pr0); st = rest0; prev = pr0.copy(); settled_at = None
with torch.no_grad(), contextlib.redirect_stdout(_DN):
    rows = []
    for f in range(1, 401):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, tw); st = solver_save_state(solver)
        p = pw.cpu().numpy()
        mv = float(np.linalg.norm(p - prev, axis=1).max() * 1e3)   # mm moved this frame (worst particle)
        sag = (top0 - p[:, top].max()) * 1e3
        prev = p
        if settled_at is None and mv < 0.05 and f > 20:
            settled_at = f
        if f % 20 == 0 or f in (1, 5, 10):
            rows.append((f, sag, mv))
for f, sag, mv in rows:
    print(f"{f:>5d} {sag:>8.1f} {mv:>18.4f}", flush=True)
print(f"\nSETTLED (maxmove<0.05mm/frame) at frame: {settled_at}")
print("→ video should settle this many frames (or until still) BEFORE starting the debulk trajectory.")
