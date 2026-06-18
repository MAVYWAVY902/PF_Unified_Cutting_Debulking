"""Does STIFF=0.9 + num_iter=3 (no muller) blow up during cutting? Confirms blow-up = eff-stiffness×num_iter.
Settle under gravity, run the light-graze cut, track max |displacement from settled rest| per window + NaN.
Configs: k=0.9 n=1 (baseline stable) / n=2 / n=3 (no muller) / n=3 +muller (eff back to 0.9, should be stable).
"""
import sys, numpy as np, torch
from pathlib import Path
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster

STIFF, REACH_MM, R = 0.9, 4.0, 3
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.enable_cutting = True
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
thetat = torch.tensor([3., 20., 0.3], device=dev)
sw = torch.tensor(np.full(N, STIFF, np.float32), device=dev); tw = torch.tensor(np.full(N, 0.3, np.float32), device=dev)
print(f"N={N} blow-up check STIFF={STIFF}  (cutting, light-graze)")

# settle once at num_iter=1 (shared rest)
solver.num_iterations = 1; solver.muller_stiff_remap = False
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.; st = rest0
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(thetat, solver, st, far, kr, sw, tw); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
y_center = bb1[top] - REACH_MM * 1e-3 + kr
base = make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)
one = [(kp.copy().astype(np.float32)) for kp in base]
for kp in one: kp[top] = np.float32(y_center)
traj = one * R
diag = bs.max()  # cube diagonal scale for "runaway" reference


def run(n_iter, muller):
    solver.num_iterations = n_iter; solver.muller_stiff_remap = muller
    solver.reset_to_rest(pr); st = rest_state
    maxdisp = 0.0; blew_at = -1
    with torch.no_grad():
        for t, kp in enumerate(traj):
            _, pw, _ = WarpFullForwardOp.apply(thetat, solver, st, kp, kr, sw, tw)
            st = solver_save_state(solver)
            cur = pw.cpu().numpy()
            if not np.isfinite(cur).all():
                blew_at = t; maxdisp = float("inf"); break
            d = np.linalg.norm(cur - pr, axis=1).max() * 1e3  # mm from settled rest
            maxdisp = max(maxdisp, d)
            if d > diag * 1e3 * 3 and blew_at < 0:  # >3× cube = runaway
                blew_at = t
    return maxdisp, blew_at


print(f"\n{'config':>22s}  {'eff-stiff':>9s}  {'max disp(mm)':>12s}  {'blew at frame':>13s}")
for n, mul in [(1, False), (2, False), (3, False), (3, True)]:
    eff = STIFF if mul else (1.0 - (1.0 - STIFF) ** n)
    md, ba = run(n, mul)
    tag = f"k={STIFF} n={n}{' +muller' if mul else ''}"
    md_s = "inf(NaN)" if md == float("inf") else f"{md:.1f}"
    ba_s = "—(stable)" if ba < 0 else str(ba)
    print(f"{tag:>22s}  {eff:>9.3f}  {md_s:>12s}  {ba_s:>13s}", flush=True)
print(f"\n(cube ~{bs.max()*1e3:.0f}mm; runaway = >3×cube or NaN)")
print("predict: n=1 stable; n=2/3 (no muller) blow (eff 0.99/0.999); n=3+muller stable (eff back to 0.9)")
