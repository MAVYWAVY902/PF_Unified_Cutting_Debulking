"""The RIGHT light-load knob = knife HEIGHT (closest-approach distance -> partial P(F)), not radius.
Sweep knife height from penetrating to grazing-above-surface; few reps. Find a height where damage
lands BETWEEN the thresholds: soft (τ=0.15) fractures, hard (τ=0.55) HOLDS (frac_soft >> frac_hard).
That 'force ~35' regime makes the SURFACE removal distinguish τ -> strong corr-free signal. DIAG truth.
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
hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0)
tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
surf_top = bb1[top]
print(f"N={N} surf={int(surf.sum())} surf_top={surf_top:.3f} kr={kr:.3f} span_top={bs[top]:.3f}")
print("knife HEIGHT sweep (reach = how far knife sphere reaches below surface). DIAG truth, R=3 reps")


def run(reach_mm, R=3, n_lines=6, t_line=12):
    # knife center placed so its lowest point reaches `reach` below the surface top.
    # reach>0 = penetrates; reach<0 = grazes above (sphere bottom above surface -> barely/none).
    reach = reach_mm * 1e-3
    y_center = surf_top - reach + kr            # sphere bottom = y_center - kr = surf_top - reach
    one = []
    for f in range(n_lines * t_line):
        # rebuild a raster but override the top-axis coord to y_center
        pass
    base = make_raster(bb0, bb1, top, sweep, la, 0.5, n_lines, t_line)  # y_frac placeholder; override top coord
    one = []
    for kp in base:
        q = kp.copy(); q[top] = np.float32(y_center); one.append(q)
    traj = one * R
    solver.reset_to_rest(pr)
    op, oa = gen_obs_per_particle_stiff(solver, rest_state, np.array([3., 20., 0.3], np.float32),
                                        np.full(N, STIFF, np.float32), traj, kr, dev, tau_truth_np=tau_truth)
    A = np.stack(oa, 0); frac = (A < 0.5).any(0)
    frh = 100 * (frac & hard & surf).sum() / max((hard & surf).sum(), 1)
    frs = 100 * (frac & ~hard & surf).sum() / max((~hard & surf).sum(), 1)
    return frh, frs


print(f"\n{'reach(mm)':>9s} {'frac_hard':>9s} {'frac_soft':>9s} {'separation(s-h)':>15s}")
for reach in (20.0, 10.0, 5.0, 2.0, 0.0, -5.0, -10.0):
    frh, frs = run(reach)
    print(f"{reach:9.1f} {frh:8.1f}% {frs:8.1f}% {frs-frh:14.1f}", flush=True)
print("\nwant: some height with frac_soft HIGH, frac_hard LOW (big separation) = load landed between τ values")
print("      -> surface removal distinguishes hard vs soft -> high-τ becomes observable")
