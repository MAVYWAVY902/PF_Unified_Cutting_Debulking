"""DECISIVE test for (b): does τ-coupled softening + a POSITION (chamfer) loss recover τ where
plain chamfer FAILED before (because position was τ-blind)?

3 configs, all num_iter=1 (isolate tau_soften):
  occ      : occupancy-only, tau_soften OFF  — current working incumbent (DIAG~68%, SPOT overshoot)
  cham_off : chamfer position loss, tau_soften OFF  — the old FALSIFIED config (position τ-blind)
  cham_on  : chamfer position loss, tau_soften ON   — (b): does waking position resurrect chamfer?

Read: cham_on should show contrast >> cham_off (b woke the channel chamfer reads). SPOT τ_hard:
does (b)+chamfer pull the 0.85 overshoot toward 0.55? Compare all to occ incumbent.
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

TAU_HARD, TAU_SOFT, STIFF, REACH_MM, R = 0.55, 0.15, 0.9, 4.0, 3   # STIFF 0.5→0.9: firmer (less gravity sag)
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 1; solver.enable_cutting = True
solver.muller_stiff_remap = False   # clean num_iter=1 regime: eff-stiffness = k = STIFF
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
print(f"N={N} (b) recovery STIFF={STIFF}: occ / cham_on / both_bal  num_iter=1  reach={REACH_MM}mm")

solver.reset_to_rest(pr0); far = pr0.max(0) + 10.
sw = torch.tensor(np.full(N, STIFF, np.float32), device=dev); tw = torch.tensor(np.full(N, 0.3, np.float32), device=dev)
tht = torch.tensor([3., 20., 0.3], dtype=torch.float32, device=dev); st = rest0
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, tw); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
surf = pr[:, top] > (bb1[top] - 0.10 * bs[top]); surf_idx = torch.tensor(np.where(surf)[0], dtype=torch.long, device=dev)
col_id, _ = column_grid_assignment(pr, surf, top, grid_n=8)
ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
eat = torch.tensor(ea, dtype=torch.long, device=dev); ebt = torch.tensor(eb, dtype=torch.long, device=dev)
ert = torch.tensor(np.linalg.norm(pr[ea] - pr[eb], axis=1), dtype=torch.float32, device=dev)
H = build_hierarchy(torch.from_numpy(pr).to(dev, torch.float32), M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)
theta_truth = np.array([3., 20., 0.3], np.float32)
y_center = bb1[top] - REACH_MM * 1e-3 + kr
base = make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)
one = [(kp.copy().astype(np.float32)) for kp in base]
for kp in one: kp[top] = np.float32(y_center)
traj = one * R

CONFIGS = {  # name: (tau_soften, chamfer_loss, lambda_chamfer, cav_occupancy, lambda_outer_cav)
    "occ":      (False, True, 0.0, True, 1.0),
    "cham_on":  (True,  True, 1.0, False, 0.0),
    "both_bal": (True,  True, 1.0, True, 5.0e-4),   # SCALE-BALANCED position+occupancy
}


def recover(pattern, cfgname, seed):
    ts, cl, lc, occ, loc = CONFIGS[cfgname]
    rng = np.random.default_rng(seed)
    solver.num_iterations = 1; solver.muller_stiff_remap = False; solver.enable_cutting = True; solver.tau_soften = ts
    xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
    hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0) if pattern == "DIAG" \
        else (((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2)
    tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
    tau_init = rng.uniform(0.12, 0.58, N).astype(np.float32)
    ctx = dict(solver=solver, rest_state=rest_state, N=N, device=dev, knife_radius=kr, pos_rest=pr,
               theta_3d_np=theta_truth, stiff_init_np=np.full(N, STIFF, np.float32), tau_init_np=tau_init,
               tau_bound=(0.05, 0.95), surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt, edges_rest_t=ert, H=H,
               group_col_id=col_id, surf_top_axis=top)
    solver.reset_to_rest(pr)
    op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta_truth, np.full(N, STIFF, np.float32), traj, kr, dev, tau_truth_np=tau_truth)
    multi = [(traj, [torch.tensor(p, device=dev) for p in op], [torch.tensor(a, device=dev) for a in oa])]
    _, tau, _ = calibrate_online(f"Group+{cfgname}", ctx, multi, N_passes=4, lr=0.03, lr_tau=0.5,
                                 lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=loc,
                                 return_best=True, line_search=False, verbose=False,
                                 chamfer_loss=cl, lambda_chamfer=lc, cav_occupancy=occ,
                                 cav_occ_pointcloud=occ, cav_occ_meannorm=occ)
    solver.tau_soften = False
    return float(tau[hard].mean()), float(tau[~hard].mean())


SEEDS = (0, 1, 2)
print(f"\n{'pattern':7s} {'config':>9s}  {'τ_hard→0.55':>12s} {'τ_soft→0.15':>12s} {'contrast':>9s}")
for pattern in ("DIAG", "SPOT"):
    for cfgname in CONFIGS:
        TH, TS = [], []
        for s in SEEDS:
            h, so = recover(pattern, cfgname, s); TH.append(h); TS.append(so)
        c = (np.mean(TH) - np.mean(TS)) / (TAU_HARD - TAU_SOFT) * 100
        print(f"{pattern:7s} {cfgname:>9s}  {np.mean(TH):.3f}±{np.std(TH):.3f}  {np.mean(TS):.3f}±{np.std(TS):.3f}  {c:7.0f}%", flush=True)
print("\nwant: cham_on contrast >> cham_off (b resurrects chamfer); SPOT cham_on τ_hard closer to 0.55 than occ's overshoot")
