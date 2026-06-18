"""OPT/VAL trajectory split (DDBot-style generalization test).

DDBot calibrates physics on an OPTIMIZATION trajectory, then validates the calibrated
parameters PREDICT a *held-out VALIDATION trajectory* it never saw. We do the analog for
the per-particle τ FIELD:

  1. calibrate τ on the OPT trajectory (light-graze raster, sweep along axis `sweep`)
  2. WITHOUT re-optimizing, run forward with the recovered τ on a *different* VAL trajectory
     (orthogonal raster, sweep along `la`) and compare the PREDICTED cavity to the
     ground-truth cavity (true τ on the same val traj).
  3. baseline = the random INIT τ on the val traj. Calibrated should predict the held-out
     cavity much better than init → proves we recovered the FIELD, not the path.

Metric: surface per-particle final-frame alpha error (removed/kept), occupancy MSE. 3 seeds.
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

TAU_HARD, TAU_SOFT, STIFF = 0.55, 0.15, 0.5
REACH_MM, R = 4.0, 3
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 2; solver.enable_cutting = True
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
print(f"N={N} OPT/VAL split (DDBot-style): calibrate on raster-A, predict held-out raster-B  reach={REACH_MM}mm R={R}")

# settle once (shared)
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.
sw = torch.tensor(np.full(N, STIFF, np.float32), device=dev); twt = torch.tensor(np.full(N, 0.3, np.float32), device=dev)
tht = torch.tensor([3., 20., 0.3], dtype=torch.float32, device=dev); st = rest0
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, twt); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
surf = pr[:, top] > (bb1[top] - 0.10 * bs[top]); surf_idx = torch.tensor(np.where(surf)[0], dtype=torch.long, device=dev)
surf_ids = np.where(surf)[0]
col_id, _ = column_grid_assignment(pr, surf, top, grid_n=8)
ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
eat = torch.tensor(ea, dtype=torch.long, device=dev); ebt = torch.tensor(eb, dtype=torch.long, device=dev)
ert = torch.tensor(np.linalg.norm(pr[ea] - pr[eb], axis=1), dtype=torch.float32, device=dev)
H = build_hierarchy(torch.from_numpy(pr).to(dev, torch.float32), M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)
theta_truth = np.array([3., 20., 0.3], np.float32)
y_center = bb1[top] - REACH_MM * 1e-3 + kr


def light_raster(ax_sweep, ax_cross):
    base = make_raster(bb0, bb1, top, ax_sweep, ax_cross, 0.5, 6, 12)
    one = [(kp.copy().astype(np.float32)) for kp in base]
    for kp in one: kp[top] = np.float32(y_center)
    return one * R


# OPT raster sweeps along `sweep`; VAL raster sweeps along `la` (orthogonal path, same surface)
TRAJ_OPT = light_raster(sweep, la)
TRAJ_VAL = light_raster(la, sweep)


def final_alpha(tau_np, traj):
    solver.reset_to_rest(pr)
    _, oa = gen_obs_per_particle_stiff(solver, rest_state, theta_truth, np.full(N, STIFF, np.float32), traj, kr, dev, tau_truth_np=tau_np)
    return oa[-1]                       # (N,) final-frame removal mask


def trial(pattern, seed):
    rng = np.random.default_rng(seed)
    xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
    hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0) if pattern == "DIAG" \
        else (((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2)
    tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
    tau_init = rng.uniform(0.12, 0.58, N).astype(np.float32)
    # ---- calibrate τ on OPT traj ----
    ctx = dict(solver=solver, rest_state=rest_state, N=N, device=dev, knife_radius=kr, pos_rest=pr,
               theta_3d_np=theta_truth, stiff_init_np=np.full(N, STIFF, np.float32), tau_init_np=tau_init,
               tau_bound=(0.05, 0.95), surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt, edges_rest_t=ert, H=H,
               group_col_id=col_id, surf_top_axis=top)
    solver.reset_to_rest(pr)
    op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta_truth, np.full(N, STIFF, np.float32), TRAJ_OPT, kr, dev, tau_truth_np=tau_truth)
    multi = [(TRAJ_OPT, [torch.tensor(p, device=dev) for p in op], [torch.tensor(a, device=dev) for a in oa])]
    _, tau_rec, _ = calibrate_online("Group+optval", ctx, multi, N_passes=4, lr=0.03, lr_tau=0.5,
                                     lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=1.0,
                                     return_best=True, line_search=False, verbose=False,
                                     chamfer_loss=True, lambda_chamfer=0.0, cav_occupancy=True,
                                     cav_occ_pointcloud=True, cav_occ_meannorm=True)
    # ---- field contrast (in-sample quality) ----
    contrast = (tau_rec[hard].mean() - tau_rec[~hard].mean()) / (TAU_HARD - TAU_SOFT)
    # ---- predict HELD-OUT VAL traj ----
    # Metric is SPATIAL STRUCTURE of the held-out cavity, not bulk alpha: light graze removes a
    # thin layer almost everywhere, so a global alpha-mean error washes out the τ-sensitive band.
    # removal r = 1-alpha. (a) does the held-out cavity show hard-less-removed structure (contrast)?
    # (b) does the predicted removal PATTERN correlate with GT over the surface?
    r_gt = 1.0 - final_alpha(tau_truth, TRAJ_VAL)[surf_ids]
    r_pred = 1.0 - final_alpha(tau_rec, TRAJ_VAL)[surf_ids]
    r_base = 1.0 - final_alpha(tau_init, TRAJ_VAL)[surf_ids]
    h = hard[surf_ids]
    # held-out cavity contrast = soft removed MORE than hard (positive = correct structure)
    vc_gt = float(r_gt[~h].mean() - r_gt[h].mean())
    vc_pred = float(r_pred[~h].mean() - r_pred[h].mean())
    vc_base = float(r_base[~h].mean() - r_base[h].mean())

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float((a * b).sum() / d)
    cor_pred = corr(r_pred, r_gt); cor_base = corr(r_base, r_gt)
    return contrast, vc_gt, vc_pred, vc_base, cor_pred, cor_base


print(f"\n{'pattern':7s}  {'in-contrast':>11s}  || held-out VAL traj:  {'cav-contrast: GT':>16s} {'cal':>6s} {'init':>6s}   {'removal-corr: cal':>17s} {'init':>6s}")
for pattern in ("DIAG", "SPOT"):
    IC, VG, VP, VB, CP, CB = [], [], [], [], [], []
    for s in (0, 1, 2):
        ic, vg, vp, vb, cp, cb = trial(pattern, s)
        IC.append(ic); VG.append(vg); VP.append(vp); VB.append(vb); CP.append(cp); CB.append(cb)
    print(f"{pattern:7s}  {np.mean(IC)*100:9.0f}%  ||                    "
          f"{np.mean(VG):16.3f} {np.mean(VP):6.3f} {np.mean(VB):6.3f}   "
          f"{np.mean(CP):17.3f} {np.mean(CB):6.3f}", flush=True)
print("\nwant: cal cav-contrast ≈ GT (init ≈ 0) AND removal-corr(cal) >> removal-corr(init)")
print("      → calibrated τ predicts the HELD-OUT cavity's spatial structure (field, not path)")
