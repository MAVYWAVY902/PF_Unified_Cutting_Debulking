"""WHY does the visible cube top drop below the settled line? (user is right: it drops ~14mm).
From the 140-settled state, run the SAME cut traj three ways and measure the VISIBLE top (alpha>0.5):
  visible_top_drop = settled_top - max top of NON-removed particles (mm)  → how far below the red line
  outer_ring_drop  = settled_top - mean top of NON-removed particles in the OUTER lateral ring (mm)
                     (the silhouette edge, away from the central cut) → is the WHOLE top sagging or just centre?
Modes: no_knife (gravity only) | contact_only (knife pushes, no removal) | removal_ON (debulk).
This assigns the cause: if removal_ON drops but contact_only/no_knife stay flat → it's material removal.
If contact_only also drops → knife push. If outer_ring drops too → whole top sags (not just a pit).
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

cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 6; solver.muller_stiff_remap = False; solver.tau_soften = False
pr0 = mi["vertices"]; top = cfg["top_axis"]; kr = cfg["knife_radius_abs"]
solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
far = pr0.max(0) + 10.0
tht = torch.tensor([3., 20., 0.3], device=dev)
sw = torch.tensor(np.full(N, 0.9, np.float32), device=dev); tw = torch.tensor(np.full(N, 0.3, np.float32), device=dev)

solver.enable_cutting = True; solver.reset_to_rest(pr0); st = rest0
with torch.no_grad(), contextlib.redirect_stdout(_DN):
    for _ in range(140):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, tw); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
settled_top = pr[:, top].max()
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
outer = (np.maximum(np.abs(xn - 0.5), np.abs(zn - 0.5)) > 0.40)   # outer lateral ring (silhouette edge)
y = bb1[top] - 4.0 * 1e-3 + kr
base = make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)
one = [kp.copy().astype(np.float32) for kp in base]
for kp in one: kp[top] = np.float32(y)
cut_traj = one * 3; n = len(cut_traj)
print(f"N={N} settled_top={settled_top*1e3:.1f}mm  outer-ring particles={int(outer.sum())}")
print(f"\n{'mode':>14s} {'measure':>16s}  " + " ".join(f"f{f}".rjust(7) for f in (1, 30, 80, 150, n - 1)))


def run(mode):
    if mode == "no_knife":
        solver.enable_cutting = True; traj = [far.astype(np.float32)] * n
    elif mode == "contact":
        solver.enable_cutting = False; traj = cut_traj
    else:
        solver.enable_cutting = True; traj = cut_traj
    st_l = rest_state; vt = {}; ot = {}
    with torch.no_grad(), contextlib.redirect_stdout(_DN):
        for f in range(1, n + 1):
            a, pw, _ = WarpFullForwardOp.apply(tht, solver, st_l, traj[f - 1], kr, sw, tw); st_l = solver_save_state(solver)
            if f in (1, 30, 80, 150, n - 1):
                p = pw.cpu().numpy(); al = a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
                vis = al > 0.5
                vt[f] = (settled_top - p[vis, top].max()) * 1e3
                om = outer & vis
                ot[f] = (settled_top - p[om, top].mean()) * 1e3 if om.any() else float("nan")
    solver.enable_cutting = True
    return vt, ot


for mode in ("no_knife", "contact", "removal_ON"):
    vt, ot = run(mode)
    print(f"{mode:>14s} {'visible_top_drop':>16s}  " + " ".join(f"{vt[f]:7.1f}" for f in (1, 30, 80, 150, n - 1)), flush=True)
    print(f"{'':>14s} {'outer_ring_drop':>16s}  " + " ".join(f"{ot[f]:7.1f}" for f in (1, 30, 80, 150, n - 1)), flush=True)
print("\n(mm below the settled red line). outer_ring_drop large → WHOLE top sags. only visible_top large → central pit.")
