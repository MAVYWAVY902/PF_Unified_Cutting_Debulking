"""Is the HARD region actually cut in the current test? Generate DIAG obs at TRUTH (hard τ=0.55,
soft τ=0.15), grazing y=0.93 raster, and measure α (removal mask) in hard vs soft regions over
the trajectory. If hard α stays ≈1 (intact, no cavity) -> confirms 'hard never fractured -> no
τ signal'. Report mean α + removed-fraction per region, on surface and overall.
"""
import sys, numpy as np, torch
from pathlib import Path
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster

TAU_HARD, TAU_SOFT, STIFF = 0.55, 0.15, 0.5
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 2; solver.enable_cutting = True
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
tht = torch.tensor([3., 20., 0.3], device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.; st = rest0
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, swt(STIFF), swt(0.3)); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
surf = pr[:, top] > (bb1[top] - 0.10 * bs[top])
xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0)        # DIAG stripes (hard τ=0.55)
tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
traj = make_raster(bb0, bb1, top, sweep, la, 0.93, 8, 25)
solver.reset_to_rest(pr)
op, oa = gen_obs_per_particle_stiff(solver, rest_state, np.array([3., 20., 0.3], np.float32),
                                    np.full(N, STIFF, np.float32), traj, kr, dev, tau_truth_np=tau_truth)
a_final = oa[-1]   # final-frame alpha (removal mask) at TRUTH
print(f"N={N} DIAG truth (hard τ=0.55 / soft τ=0.15), grazing y=0.93 raster — α after full cut")


def stats(mask, lbl):
    a = a_final[mask]
    print(f"  {lbl:18s} n={mask.sum():6d}  mean_α={a.mean():.3f}  "
          f"removed(α<0.5)={100*(a<0.5).mean():4.1f}%  intact(α>0.9)={100*(a>0.9).mean():4.1f}%  "
          f"band(0.1<α<0.9)={100*((a>0.1)&(a<0.9)).mean():4.1f}%")


print("\n--- ALL particles ---")
stats(hard, "HARD (τ=0.55)"); stats(~hard, "SOFT (τ=0.15)")
print("\n--- SURFACE only (what the camera sees) ---")
stats(hard & surf, "HARD surf"); stats(~hard & surf, "SOFT surf")
print("\nif HARD intact≈100% / removed≈0% / band≈0% -> hard never fractured -> τ_hard unobservable (confirms theory)")
