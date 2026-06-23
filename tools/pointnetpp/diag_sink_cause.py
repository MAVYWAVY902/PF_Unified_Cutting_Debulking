"""ISOLATE the cause of the during-video sink. From the SAME 140-settled state, run the SAME cut
trajectory three ways and measure the BULK (mean-Y) sink:
  (a) NO knife (parked far)         → pure gravity from settled state (expect flat: settling done)
  (b) knife on traj, REMOVAL ON     → the video's setting (debulking)
  (c) knife on traj, REMOVAL OFF    → knife present, collides/PUSHES, but removes nothing (contact only)
If (c) also sinks → the sink is the knife PUSHING the cube (contact), NOT debulking → user is right.
If (c) flat but (b) sinks → it IS material removal. If all flat → sink is elsewhere (rendering/mapping).
num_iter=6, k=0.9, damping=0.95. Reports mean-Y drop at several cut frames + worst-particle move.
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
from tools.pointnetpp.test_group_generalize import make_raster

cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 6; solver.muller_stiff_remap = False; solver.tau_soften = False
pr0 = mi["vertices"]; top = cfg["top_axis"]; kr = cfg["knife_radius_abs"]
solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
far = pr0.max(0) + 10.0
tht = torch.tensor([3., 20., 0.3], device=dev)
sw = torch.tensor(np.full(N, 0.9, np.float32), device=dev); tw = torch.tensor(np.full(N, 0.3, np.float32), device=dev)

# settle 140 ONCE, reuse for all three
solver.enable_cutting = True
solver.reset_to_rest(pr0); st = rest0
with torch.no_grad(), contextlib.redirect_stdout(_DN):
    for _ in range(140):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, tw); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
meanY_settled = pr[:, top].mean()
bb0, bb1 = pr.min(0), pr.max(0)
y_center = bb1[top] - 4.0 * 1e-3 + kr
base = make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)
one = [kp.copy().astype(np.float32) for kp in base]
for kp in one: kp[top] = np.float32(y_center)
cut_traj = one * 3
n = len(cut_traj)
print(f"N={N}  settled meanY drop-from-rest = {(pr0[:,top].mean()-meanY_settled)*1e3:.2f}mm; cut traj = {n} frames")
print(f"\n{'mode':>22s}  " + " ".join(f"f{f}".rjust(8) for f in (1, 20, 50, 100, 150, n - 1)) + "   (meanY sink mm from settled)")


def run(mode):
    if mode == "no_knife":
        solver.enable_cutting = True; traj = [far.astype(np.float32)] * n
    elif mode == "removal_ON":
        solver.enable_cutting = True; traj = cut_traj
    else:  # contact_only
        solver.enable_cutting = False; traj = cut_traj
    st_l = rest_state; sink = {}
    with torch.no_grad(), contextlib.redirect_stdout(_DN):
        for f in range(1, n + 1):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st_l, traj[f - 1], kr, sw, tw); st_l = solver_save_state(solver)
            if f in (1, 20, 50, 100, 150, n - 1):
                p = pw.cpu().numpy(); sink[f] = (meanY_settled - p[:, top].mean()) * 1e3
    solver.enable_cutting = True
    return sink


for mode in ("no_knife", "removal_ON", "contact_only"):
    s = run(mode)
    print(f"{mode:>22s}  " + " ".join(f"{s[f]:8.2f}" for f in (1, 20, 50, 100, 150, n - 1)), flush=True)
print("\nKEY: positive = bulk sank further below the settled state. no_knife≈0 confirms gravity done.")
print("contact_only sinks like removal_ON → knife PUSH (contact), not debulking. Both flat → sink is elsewhere.")
