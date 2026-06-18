"""TRUE light-load + onset-REP: shrink the cutting knife_radius so each grazing pass does only
PARTIAL damage -> takes MANY reps to fracture -> onset spreads across reps and encodes τ cleanly.
Measure onset in REPS (removes within-rep scan-order noise). Sweep cut-radius. DIAG truth.
Want: soft fractures early (low onset-rep, high frac), hard late/never (high onset-rep, low frac)
-> strong corr(onset-rep, τ) and/or strong frac separation = load-to-failure τ readout.
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
kr0 = cfg["knife_radius_abs"]; top = cfg["top_axis"]
tht = torch.tensor([3., 20., 0.3], device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.; st = rest0
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr0, swt(STIFF), swt(0.3)); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
surf = pr[:, top] > (bb1[top] - 0.10 * bs[top])
xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0)
tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
print(f"N={N} surf={int(surf.sum())} mean_edge~{np.linalg.norm(pr[solver.e['a'].numpy().astype(int)]-pr[solver.e['b'].numpy().astype(int)],axis=1).mean()*1000:.1f}mm")
print("light-load onset-REP (DIAG truth); sweep CUT radius (smaller=lighter per pass), R=20 reps")


def run(kr_cut, R=20, n_lines=6, t_line=12, y_frac=0.93):
    one = make_raster(bb0, bb1, top, sweep, la, y_frac, n_lines, t_line)
    fpr = len(one)                                # frames per rep
    traj = one * R
    solver.reset_to_rest(pr)
    op, oa = gen_obs_per_particle_stiff(solver, rest_state, np.array([3., 20., 0.3], np.float32),
                                        np.full(N, STIFF, np.float32), traj, kr_cut, dev, tau_truth_np=tau_truth)
    A = np.stack(oa, 0)                            # (T,N)
    T = A.shape[0]
    frac = (A < 0.5).any(0)
    onset_f = np.where(frac, (A < 0.5).argmax(0), T)
    onset_rep = onset_f / fpr                      # in REPS (removes within-rep scan noise)
    sf = surf
    frh = 100 * (frac & hard & sf).sum() / max((hard & sf).sum(), 1)
    frs = 100 * (frac & ~hard & sf).sum() / max((~hard & sf).sum(), 1)
    orh = onset_rep[hard & sf & frac]; ors = onset_rep[~hard & sf & frac]
    fm = sf & frac
    corr = float(np.corrcoef(onset_rep[fm], tau_truth[fm])[0, 1]) if fm.sum() > 10 else float("nan")
    return R, fpr, frh, frs, (orh.mean() if len(orh) else np.nan), (ors.mean() if len(ors) else np.nan), corr


print(f"\n{'kr_cut':>7s} {'frac_hard':>9s} {'frac_soft':>9s} {'onsetREP_h':>10s} {'onsetREP_s':>10s} {'corr(rep,τ)':>11s}")
for krc in (0.030, 0.015, 0.008, 0.005):
    R, fpr, frh, frs, orh, ors, c = run(krc)
    print(f"{krc:7.3f} {frh:8.1f}% {frs:8.1f}% {orh:10.2f} {ors:10.2f} {c:11.3f}", flush=True)
print("\nwant: smaller kr_cut -> soft fractures (high frac_soft, low onsetREP_s), hard holds/late")
print("      (frac_soft >> frac_hard) OR (onsetREP_h >> onsetREP_s, corr up) = strong τ readout")
