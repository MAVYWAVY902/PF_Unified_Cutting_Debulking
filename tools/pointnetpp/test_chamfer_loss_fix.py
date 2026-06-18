"""Does chamfer-as-LOSS (soft, bidirectional, free-run state) fix the corr-free STATE channel
that chamfer_state (hard NN-snap state overwrite) broke (SPOT 71→-48)? Grazing y=0.93, GROUP.
Configs (SPOT+DIAG, 2 seeds):
  baseline        : corr-using (teacher-force + per-particle L_cav)            ref 71/16
  chamfer_state   : hard NN-snap state overwrite (current corr-free, broken)
  chamfer_loss    : NEW soft bidirectional Chamfer L_gap + FREE-RUN + per-particle L_cav
  chamfer_loss+occ: full corr-free (chamfer_loss + cav_occupancy)
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

TAU_HARD, TAU_SOFT, TAU_INIT, STIFF = 0.55, 0.15, 0.30, 0.5
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 2; solver.enable_cutting = True
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
print(f"N={N} chamfer-as-loss fix (grazing y=0.93, GROUP)")


def recovery(pattern, kw, seed):
    solver.num_iterations = 2; solver.enable_cutting = True
    solver.reset_to_rest(pr0); far = pr0.max(0) + 10.
    sw = torch.tensor(np.full(N, STIFF, np.float32), device=dev); twt = torch.tensor(np.full(N, TAU_INIT, np.float32), device=dev)
    tht = torch.tensor([3., 20., 0.3], dtype=torch.float32, device=dev); st = rest0
    with torch.no_grad():
        for _ in range(60):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, twt); st = solver_save_state(solver)
    pr = pw.cpu().numpy().copy(); rest_state = st
    bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
    lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
    surf = pr[:, top] > (bb1[top] - 0.10 * bs[top]); surf_idx = torch.tensor(np.where(surf)[0], dtype=torch.long, device=dev)
    col_id, _ = column_grid_assignment(pr, surf, top, grid_n=8)
    xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
    hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0) if pattern == "DIAG" \
        else (((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2)
    tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
    traj = make_raster(bb0, bb1, top, sweep, la, 0.93, 8, 25)
    theta_truth = np.array([3., 20., 0.3], np.float32)
    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    eat = torch.tensor(ea, dtype=torch.long, device=dev); ebt = torch.tensor(eb, dtype=torch.long, device=dev)
    ert = torch.tensor(np.linalg.norm(pr[ea] - pr[eb], axis=1), dtype=torch.float32, device=dev)
    H = build_hierarchy(torch.from_numpy(pr).to(dev, torch.float32),
                        M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)
    ctx = dict(solver=solver, rest_state=rest_state, N=N, device=dev, knife_radius=kr, pos_rest=pr,
               theta_3d_np=theta_truth, stiff_init_np=np.full(N, STIFF, np.float32), tau_init_np=np.full(N, TAU_INIT, np.float32),
               tau_bound=(0.05, 0.95), surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt, edges_rest_t=ert, H=H, group_col_id=col_id,
               surf_top_axis=top)
    solver.reset_to_rest(pr)
    op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta_truth, np.full(N, STIFF, np.float32), traj, kr, dev, tau_truth_np=tau_truth)
    multi = [(traj, [torch.tensor(p, device=dev) for p in op], [torch.tensor(a, device=dev) for a in oa])]
    _, tau, _ = calibrate_online("Group+chamferfix", ctx, multi, N_passes=4, lr=0.03, lr_tau=0.5,
                                 lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=1.0,
                                 return_best=True, line_search=False, verbose=False, **kw)
    return (tau[hard].mean() - tau[~hard].mean()) / (TAU_HARD - TAU_SOFT) * 100


CONFIGS = [
    ("baseline", {}),
    ("chamfer_state", dict(chamfer_state=True)),
    ("chamfer_loss", dict(chamfer_loss=True)),
    ("chamfer_loss+occ", dict(chamfer_loss=True, cav_occupancy=True, cav_grid_n=12)),
]
print(f"\n{'pattern':7s} {'config':>18s}  {'2 seeds':>16s}  mean")
for pname in ("SPOT", "DIAG"):
    for name, kw in CONFIGS:
        rs = [float(recovery(pname, kw, s)) for s in (0, 1)]
        print(f"{pname:7s} {name:>18s}  {str([round(r,1) for r in rs]):>16s}  {np.mean(rs):6.1f}%", flush=True)
print("\nwant: chamfer_loss recovers ≈ baseline (not sign-flipped like chamfer_state) → state-channel fixed")
