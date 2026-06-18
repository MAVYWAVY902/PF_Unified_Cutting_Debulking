"""both_on diagnosis: is the combine-failure a true tug-of-war, or just L_gap vs L_cav SCALE imbalance?
Run both_on (DIAG) verbose, capture L_gap (chamfer, ~meters) and L_cav (occupancy, ~O(1)) magnitudes.
If they differ by orders of magnitude, the Adam combined-gradient is dominated by one → "combine failure"
may just be scale imbalance (fixable by rescaling lambda), not a fundamental tug-of-war.
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
N = solver.N; solver.damping = 0.95; solver.num_iterations = 1; solver.enable_cutting = True
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
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
xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0)
tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
solver.tau_soften = True
ctx = dict(solver=solver, rest_state=rest_state, N=N, device=dev, knife_radius=kr, pos_rest=pr,
           theta_3d_np=theta_truth, stiff_init_np=np.full(N, STIFF, np.float32),
           tau_init_np=np.random.default_rng(0).uniform(0.12, 0.58, N).astype(np.float32),
           tau_bound=(0.05, 0.95), surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt, edges_rest_t=ert, H=H,
           group_col_id=col_id, surf_top_axis=top)
solver.reset_to_rest(pr)
op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta_truth, np.full(N, STIFF, np.float32), traj, kr, dev, tau_truth_np=tau_truth)
multi = [(traj, [torch.tensor(p, device=dev) for p in op], [torch.tensor(a, device=dev) for a in oa])]
print("=== both_on DIAG, verbose — read L_gap (chamfer) vs L_cav (occupancy) magnitudes ===", flush=True)
calibrate_online("Group+scalecheck", ctx, multi, N_passes=1, lr=0.03, lr_tau=0.5,
                 lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=1.0,
                 return_best=True, line_search=False, verbose=True, log_every=40,
                 chamfer_loss=True, lambda_chamfer=1.0, cav_occupancy=True,
                 cav_occ_pointcloud=True, cav_occ_meannorm=True)
print("\nread the L_gap=... L_cav=... columns above: if they differ by >10×, both_on is scale-imbalanced")
