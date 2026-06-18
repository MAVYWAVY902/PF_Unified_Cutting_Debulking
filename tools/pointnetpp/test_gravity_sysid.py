"""Does reducing gravity (9.81 → 1.5) affect sysID? Re-settle per gravity (different sag → different
geometry), then recover τ. occ & cham_on are single-loss (no λ-balance confound) = clean gravity test;
both_bal is the production combo (λ kept at 5e-4, approximate). STIFF=0.9, num_iter=1, 3 seeds.

Answer shape: if cham_on/occ recovery at G=1.5 >= G=9.81 → reducing gravity is a win (firmer demo +
no sysID loss). If clearly worse → (b)/occupancy was using gravity-driven sag, there's a tradeoff.
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

TAU_HARD, TAU_SOFT, STIFF, REACH_MM, R = 0.55, 0.15, 0.9, 4.0, 3
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 1; solver.muller_stiff_remap = False; solver.enable_cutting = True
pr0 = mi["vertices"]; solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
theta = np.array([3., 20., 0.3], np.float32); thetat = torch.tensor(theta, device=dev)
sw = torch.tensor(np.full(N, STIFF, np.float32), device=dev)
print(f"N={N} GRAVITY sysID test STIFF={STIFF} num_iter=1")

CONFIGS = {  # (tau_soften, chamfer_loss, lambda_chamfer, cav_occupancy, lambda_outer_cav)
    "occ":      (False, True, 0.0, True, 1.0),
    "cham_on":  (True,  True, 1.0, False, 0.0),
    "both_bal": (True,  True, 1.0, True, 5.0e-4),
}


def settle_geometry(G):
    solver.gravity = wp.vec3(0., -float(G), 0.)
    solver.reset_to_rest(pr0); far = pr0.max(0) + 10.
    tw = torch.tensor(np.full(N, 0.3, np.float32), device=dev); st = rest0
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
    sag = (pr[surf_idx.cpu().numpy()][:, top] - pr0[surf_idx.cpu().numpy()][:, top]).mean() * 1e3
    return dict(pr=pr, rest_state=rest_state, bb0=bb0, bb1=bb1, bs=bs, sweep=sweep, la=la,
                surf_idx=surf_idx, col_id=col_id, eat=eat, ebt=ebt, ert=ert, H=H, sag=sag)


def recover(g, pattern, cfgname, seed):
    ts, cl, lc, occ, loc = CONFIGS[cfgname]
    rng = np.random.default_rng(seed)
    solver.gravity = wp.vec3(0., -float(g["_G"]), 0.); solver.tau_soften = ts
    solver.num_iterations = 1; solver.muller_stiff_remap = False; solver.enable_cutting = True
    pr = g["pr"]; sweep, la = g["sweep"], g["la"]; bb0, bb1, bs = g["bb0"], g["bb1"], g["bs"]
    xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
    hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0) if pattern == "DIAG" \
        else (((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2)
    tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
    tau_init = rng.uniform(0.12, 0.58, N).astype(np.float32)
    y_center = bb1[top] - REACH_MM * 1e-3 + kr
    base = make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)
    one = [(kp.copy().astype(np.float32)) for kp in base]
    for kp in one: kp[top] = np.float32(y_center)
    traj = one * R
    ctx = dict(solver=solver, rest_state=g["rest_state"], N=N, device=dev, knife_radius=kr, pos_rest=pr,
               theta_3d_np=theta, stiff_init_np=np.full(N, STIFF, np.float32), tau_init_np=tau_init,
               tau_bound=(0.05, 0.95), surface_idx=g["surf_idx"], edges_a_t=g["eat"], edges_b_t=g["ebt"],
               edges_rest_t=g["ert"], H=g["H"], group_col_id=g["col_id"], surf_top_axis=top)
    solver.reset_to_rest(pr)
    op, oa = gen_obs_per_particle_stiff(solver, g["rest_state"], theta, np.full(N, STIFF, np.float32), traj, kr, dev, tau_truth_np=tau_truth)
    multi = [(traj, [torch.tensor(p, device=dev) for p in op], [torch.tensor(a, device=dev) for a in oa])]
    _, tau, _ = calibrate_online(f"Group+{cfgname}", ctx, multi, N_passes=4, lr=0.03, lr_tau=0.5,
                                 lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=loc,
                                 return_best=True, line_search=False, verbose=False,
                                 chamfer_loss=cl, lambda_chamfer=lc, cav_occupancy=occ,
                                 cav_occ_pointcloud=occ, cav_occ_meannorm=occ)
    solver.tau_soften = False
    return float(tau[hard].mean()), float(tau[~hard].mean())


SEEDS = (0, 1, 2)
print(f"\n{'G':>5s} {'pattern':7s} {'config':>9s}  {'τ_hard→0.55':>12s} {'τ_soft→0.15':>12s} {'abs_err':>8s}")
for G in (9.81, 1.5):
    geom = settle_geometry(G); geom["_G"] = G
    print(f"  -- gravity={G}  surface sag={geom['sag']:+.1f} mm --", flush=True)
    for pattern in ("DIAG", "SPOT"):
        for cfgname in CONFIGS:
            TH, TS = [], []
            for s in SEEDS:
                h, so = recover(geom, pattern, cfgname, s); TH.append(h); TS.append(so)
            ae = abs(np.mean(TH) - TAU_HARD) + abs(np.mean(TS) - TAU_SOFT)
            print(f"{G:>5.2f} {pattern:7s} {cfgname:>9s}  {np.mean(TH):.3f}±{np.std(TH):.3f}  {np.mean(TS):.3f}±{np.std(TS):.3f}  {ae:>8.3f}", flush=True)
print("\nwant: at G=1.5, abs_err <= G=9.81 (esp. cham_on/occ, single-loss) → reducing gravity doesn't hurt sysID")
