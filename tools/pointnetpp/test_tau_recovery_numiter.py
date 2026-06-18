"""Does num_iter=1 (the most-stable forward fix) preserve GROUP tau recovery vs the
current num_iter=2 baseline? DIAG + SPOT patterns, gate-ON physics, canonical settings.
settle + obs-gen + calibration all at the matching num_iter (fair comparison).
"""
import sys, time, numpy as np, torch
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

TAU_HARD = 0.55; TAU_SOFT = 0.15; TAU_INIT = 0.30; STIFF = 0.5
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest_state0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
print(f"N={N} (tau recovery: num_iter 2 baseline vs 1)")


def recovery(ni, pattern_name):
    solver.num_iterations = ni; solver.muller_stiff_remap = False; solver.tw_gate = 0.0
    solver.alpha_vel_scale = 0.0; solver.collision_in_loop = True
    # settle at this num_iter
    solver.reset_to_rest(pr0); far = pr0.max(0) + 10.
    sw = torch.tensor(np.full(N, STIFF, np.float32), device=dev); twt = torch.tensor(np.full(N, TAU_INIT, np.float32), device=dev)
    tht = torch.tensor([3., 20., 0.3], dtype=torch.float32, device=dev); st = rest_state0
    with torch.no_grad():
        for _ in range(60):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, twt); st = solver_save_state(solver)
    pr = pw.cpu().numpy().copy(); rest_state = st
    bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
    lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
    surf = pr[:, top] > (bb1[top] - 0.10 * bs[top]); surf_idx = torch.tensor(np.where(surf)[0], dtype=torch.long, device=dev)
    col_id, _ = column_grid_assignment(pr, surf, top, grid_n=8)
    xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
    if pattern_name == "DIAG":
        hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0)
    else:
        hard = (((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2)
    tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
    trajA = make_raster(bb0, bb1, top, sweep, la, 0.93, 10, 30)
    trajB = make_raster(bb0, bb1, top, sweep, la, 0.88, 10, 30)
    stiff_u = np.full(N, STIFF, np.float32); theta_truth = np.array([3., 20., 0.3], np.float32)
    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    eat = torch.tensor(ea, dtype=torch.long, device=dev); ebt = torch.tensor(eb, dtype=torch.long, device=dev)
    ert = torch.tensor(np.linalg.norm(pr[ea] - pr[eb], axis=1), dtype=torch.float32, device=dev)
    xr = torch.from_numpy(pr).to(dev, torch.float32)
    H = build_hierarchy(xr, M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)
    ctx = dict(solver=solver, rest_state=rest_state, N=N, device=dev, knife_radius=kr, pos_rest=pr,
               theta_3d_np=theta_truth, stiff_init_np=np.full(N, STIFF, np.float32), tau_init_np=np.full(N, TAU_INIT, np.float32),
               tau_bound=(0.05, 0.95), surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt, edges_rest_t=ert, H=H, group_col_id=col_id)
    multi = []
    for traj in (trajA, trajB):
        solver.reset_to_rest(pr)
        op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta_truth, stiff_u, traj, kr, dev, tau_truth_np=tau_truth)
        multi.append((traj, [torch.tensor(p, device=dev) for p in op], [torch.tensor(a, device=dev) for a in oa]))
    t0 = time.time()
    _, tau, _ = calibrate_online("Group+L_pos_partial", ctx, multi, N_passes=4, lr=0.03, lr_tau=0.5,
                                 lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=1.0,
                                 return_best=True, line_search=False, verbose=False)
    rec = (tau[hard].mean() - tau[~hard].mean()) / (TAU_HARD - TAU_SOFT) * 100
    return float(rec), time.time() - t0


print(f"\n{'pattern':6s}  {'num_iter=2 (base)':>20s}  {'num_iter=1 (fix)':>18s}")
for pname in ("DIAG", "SPOT"):
    r2, t2 = recovery(2, pname)
    r1, t1 = recovery(1, pname)
    print(f"{pname:6s}  {r2:6.1f}% ({t2:.0f}s)        {r1:6.1f}% ({t1:.0f}s)", flush=True)
solver.num_iterations = 2
