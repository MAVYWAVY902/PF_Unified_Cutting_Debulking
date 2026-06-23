"""②-b HETEROGENEOUS cohesive law (the REAL risk the uniform test missed).

The uniform law-mismatch test "passed" because the SAME softening law applied everywhere → the
hard-vs-soft contrast was preserved. But real tissue is NOT uniform: tumor ≠ normal tissue, and
different tissues may soften DIFFERENTLY. This test puts a DIFFERENT softening law on the hard
region vs the soft region, then calibrates with the uniform-linear law. If the method holds → robust
to heterogeneous laws; if it breaks → the uniform-test "pass" was hollow for the real (inclusion) case.

Worst case "confuse": hard region EARLY-softens (p=0.5, deforms MORE → looks softer) while soft region
LATE-softens (p=2, deforms LESS → looks harder) — maximally inverts the hard/soft contrast.
obs = heterogeneous law; calib = uniform linear (p=1). occ (α, law-agnostic) vs both_bal ((b)).
3 seeds — and we report std so we can see if differences exceed the noise floor.
"""
import sys, numpy as np, torch
from pathlib import Path
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff
from tools.pointnetpp.hierarchy import build_hierarchy
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.calibrate_online import calibrate_online
from tools.pointnetpp.surface_projection import column_grid_assignment
from tools.pointnetpp.test_group_generalize import make_raster

TAU_HARD, TAU_SOFT, STIFF, REACH_MM, R = 0.55, 0.15, 0.5, 4.0, 3
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 1; solver.muller_stiff_remap = False; solver.enable_cutting = True
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
theta = np.array([3., 20., 0.3], np.float32); thetat = torch.tensor(theta, device=dev)
sw = torch.tensor(np.full(N, STIFF, np.float32), device=dev); tw = torch.tensor(np.full(N, 0.3, np.float32), device=dev)
print(f"N={N} ②-b HETEROGENEOUS law (real risk: tumor≠normal soften differently)  STIFF={STIFF}")

solver.tau_soften = True; solver.tau_soften_exp = 1.0; solver.tau_soften_exp_field = None
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.; st = rest0
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(thetat, solver, st, far, kr, sw, tw); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
surf = pr[:, top] > (bb1[top] - 0.10 * bs[top]); surf_idx = torch.tensor(np.where(surf)[0], dtype=torch.long, device=dev)
col_id, _ = column_grid_assignment(pr, surf, top, grid_n=8)
ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
eat = torch.tensor(ea, dtype=torch.long, device=dev); ebt = torch.tensor(eb, dtype=torch.long, device=dev)
ert = torch.tensor(np.linalg.norm(pr[ea] - pr[eb], axis=1), dtype=torch.float32, device=dev)
H = build_hierarchy(torch.from_numpy(pr).to(dev, torch.float32), M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)
y_center = bb1[top] - REACH_MM * 1e-3 + kr
base = make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)
one = [(kp.copy().astype(np.float32)) for kp in base]
for kp in one: kp[top] = np.float32(y_center)
traj = one * R

CONFIGS = {"occ": (False, 0.0, 1.0), "cham_on": (True, 1.0, 0.0), "both_bal": (True, 1.0, 5.0e-4)}
HET = {                       # (p_hard, p_soft) of the obs-generating law
    "uniform_p1":  (1.0, 1.0),    # baseline (self-consistent)
    "confuse":     (0.5, 2.0),    # hard early-soften (looks soft) + soft late-soften (looks hard) → worst
}


def recover(pattern, hetname, cfgname, seed):
    ts, lc, loc = CONFIGS[cfgname]; p_hard, p_soft = HET[hetname]
    rng = np.random.default_rng(seed)
    xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
    hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0) if pattern == "DIAG" \
        else (((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2)
    tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
    tau_init = rng.uniform(0.12, 0.58, N).astype(np.float32)
    ctx = dict(solver=solver, rest_state=rest_state, N=N, device=dev, knife_radius=kr, pos_rest=pr,
               theta_3d_np=theta, stiff_init_np=np.full(N, STIFF, np.float32), tau_init_np=tau_init,
               tau_bound=(0.05, 0.95), surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt, edges_rest_t=ert, H=H,
               group_col_id=col_id, surf_top_axis=top)
    # ---- obs = "real tissue" with HETEROGENEOUS law (p_hard on inclusion, p_soft outside) ----
    solver.num_iterations = 1; solver.enable_cutting = True; solver.tau_soften = True
    solver.tau_soften_exp_field = np.where(hard, p_hard, p_soft).astype(np.float32)
    solver.reset_to_rest(pr)
    op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta, np.full(N, STIFF, np.float32), traj, kr, dev, tau_truth_np=tau_truth)
    multi = [(traj, [torch.tensor(p, device=dev) for p in op], [torch.tensor(a, device=dev) for a in oa])]
    # ---- calib with uniform linear law (p=1) ----
    solver.tau_soften = ts; solver.tau_soften_exp_field = None; solver.tau_soften_exp = 1.0
    _, tau, _ = calibrate_online(f"Group+{cfgname}", ctx, multi, N_passes=4, lr=0.03, lr_tau=0.5,
                                 lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=loc,
                                 return_best=True, line_search=False, verbose=False,
                                 chamfer_loss=True, lambda_chamfer=lc, cav_occupancy=True,
                                 cav_occ_pointcloud=True, cav_occ_meannorm=True)
    solver.tau_soften = False; solver.tau_soften_exp_field = None; solver.tau_soften_exp = 1.0
    return float(tau[hard].mean()), float(tau[~hard].mean())


SEEDS = (0, 1, 2)
print(f"\n{'pattern':7s} {'config':>9s} {'law(h/s)':>11s}  {'τ_hard→0.55':>12s} {'τ_soft→0.15':>12s} {'abs_err':>8s}  {'Δvs uni':>8s}")
for pattern in ("DIAG", "SPOT"):
    for cfgname in CONFIGS:
        base_err = None
        for hetname in HET:
            TH, TS = [], []
            for s in SEEDS:
                h, so = recover(pattern, hetname, cfgname, s); TH.append(h); TS.append(so)
            ae = abs(np.mean(TH) - TAU_HARD) + abs(np.mean(TS) - TAU_SOFT)
            if hetname == "uniform_p1": base_err = ae
            dv = "" if base_err is None else f"{ae - base_err:+.3f}"
            ph, ps = HET[hetname]
            print(f"{pattern:7s} {cfgname:>9s} {f'{ph}/{ps}':>11s}  {np.mean(TH):.3f}±{np.std(TH):.3f}  {np.mean(TS):.3f}±{np.std(TS):.3f}  {ae:>8.3f}  {dv:>8s}", flush=True)
print("\nKEY: 'confuse' (0.5/2.0) maximally inverts hard/soft. If both_bal HOLDS (Δ small, < noise std) → robust to")
print("heterogeneous law (the real tumor≠normal case). If both_bal BREAKS (hard collapses) → the uniform pass was hollow.")
