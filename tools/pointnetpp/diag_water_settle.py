"""CORE-ISSUE DIAGNOSTIC: is the body a soft SOLID (sags to equilibrium, self-supporting)
or WATER (keeps flowing, no self-support)? Settle under gravity, NO knife, track over frames:
  - sag    = drop of the top surface (mm) from rest  → how much it droops
  - still   = body still moving at end? (mean |vel| mm/s)  → equilibrium reached or flowing
  - spread = lateral footprint growth (mm)  → water spreads, solid holds footprint
Sweep num_iterations (1 = blow-up-forced video setting, 2, 6 = design) × stiffness k.
A self-supporting solid: sag bounded, vel→0 (settles), spread≈0. Water: sag grows, vel stays high, spread up.
"""
import sys, numpy as np, torch
from pathlib import Path
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS

cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.enable_cutting = True; solver.muller_stiff_remap = False
pr0 = mi["vertices"]; top = cfg["top_axis"]; kr = cfg["knife_radius_abs"]
solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
lat = [a for a in range(3) if a != top]
top0 = pr0[:, top].max(); foot0 = (pr0[:, lat].max(0) - pr0[:, lat].min(0))
far = pr0.max(0) + 10.0
tht = torch.tensor([3., 20., 0.3], device=dev)
print(f"N={N} top_axis={top} rest top={top0*1e3:.1f}mm  SETTLE 80 frames, NO knife (far). gravity=-9.81 damping={solver.damping}")
print(f"\n{'num_iter':>8s} {'k':>5s}  {'sag@40':>8s} {'sag@80':>8s}  {'still(|v|mm/s)':>14s} {'spread mm':>10s}  verdict")


def settle(num_iter, k):
    solver.num_iterations = num_iter; solver.tau_soften = False
    solver.reset_to_rest(pr0); st = rest0
    sw = torch.tensor(np.full(N, k, np.float32), device=dev); tw = torch.tensor(np.full(N, 0.3, np.float32), device=dev)
    sag = {}; prev = pr0.copy()
    with torch.no_grad():
        for f in range(1, 81):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, tw); st = solver_save_state(solver)
            p = pw.cpu().numpy()
            if f in (40, 80):
                sag[f] = (top0 - p[:, top].max()) * 1e3                 # top surface drop (mm)
                if f == 80:
                    vel = np.linalg.norm(p - prev, axis=1) / solver.dt * 1e3   # mm/s, last-frame
                    foot = p[:, lat].max(0) - p[:, lat].min(0)
                    spread = float((foot - foot0).max()) * 1e3
                    still = float(vel.mean())
            prev = p
    return sag[40], sag[80], still, spread


for ni in (1, 2, 6):
    for k in (0.5, 0.9):
        s40, s80, still, spread = settle(ni, k)
        # verdict: settled solid if still vel low AND sag stops growing; water if vel high / sag grows
        growing = s80 - s40
        v = "SOLID (settled)" if (still < 5 and growing < 2) else ("flowing" if still > 20 else "soft, still creeping")
        print(f"{ni:>8d} {k:>5.1f}  {s40:>8.1f} {s80:>8.1f}  {still:>14.1f} {spread:>10.1f}  {v}", flush=True)
print("\nKEY: video used num_iter=1,k=0.5. If that row = 'flowing'/high-vel/big-sag while num_iter=6 = 'SOLID' →")
print("core issue = num_iter=1 (blow-up-forced) kills shape-matching rigidity (stiffness builds across iterations).")
