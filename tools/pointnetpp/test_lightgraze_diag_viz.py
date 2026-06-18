"""Verify DIAG corr-free recovery is a REAL stripe field (not an inflated scalar). Run the real
config (obs point-cloud occupancy + sim rest-cell + free-run), save the recovered τ field, and plot
top-down truth / recovered / error. Same scalar-vs-spatial check we did for SPOT.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0)           # DIAG stripes
tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
kr_=kr
_yc = bb1[top] - 4.0e-3 + kr
_base = make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)
traj=[]
for _kp in _base:
    _q=_kp.copy().astype(np.float32); _q[top]=np.float32(_yc); traj.append(_q)
traj = traj*3
theta_truth = np.array([3., 20., 0.3], np.float32)
ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
eat = torch.tensor(ea, dtype=torch.long, device=dev); ebt = torch.tensor(eb, dtype=torch.long, device=dev)
ert = torch.tensor(np.linalg.norm(pr[ea] - pr[eb], axis=1), dtype=torch.float32, device=dev)
H = build_hierarchy(torch.from_numpy(pr).to(dev, torch.float32), M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)
ctx = dict(solver=solver, rest_state=rest_state, N=N, device=dev, knife_radius=kr, pos_rest=pr,
           theta_3d_np=theta_truth, stiff_init_np=np.full(N, STIFF, np.float32), tau_init_np=np.full(N, TAU_INIT, np.float32),
           tau_bound=(0.05, 0.95), surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt, edges_rest_t=ert, H=H, group_col_id=col_id,
           surf_top_axis=top)
solver.reset_to_rest(pr)
op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta_truth, np.full(N, STIFF, np.float32), traj, kr, dev, tau_truth_np=tau_truth)
multi = [(traj, [torch.tensor(p, device=dev) for p in op], [torch.tensor(a, device=dev) for a in oa])]
# REAL config: obs point-cloud occupancy + sim rest-cell + free-run
_, tau, _ = calibrate_online("Group+diagviz", ctx, multi, N_passes=4, lr=0.03, lr_tau=0.5,
                             lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=1.0,
                             return_best=True, line_search=False, verbose=False,
                             chamfer_loss=True, lambda_chamfer=0.0, cav_occupancy=True, cav_occ_pointcloud=True)
rec = (tau[hard].mean() - tau[~hard].mean()) / (TAU_HARD - TAU_SOFT) * 100
print(f"N={N} DIAG corr-free (obs point-cloud + sim rest + free-run)  contrast={rec:.1f}%  "
      f"hard τ={tau[hard].mean():.3f} soft τ={tau[~hard].mean():.3f} (truth 0.55/0.15)")

m = surf; x = pr[m, sweep]; y = pr[m, la]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, vals, title, cmap, vlo, vhi in [
    (axes[0], tau_truth[m], "TRUTH τ (DIAG stripes (LIGHT-GRAZE)", "viridis", 0.10, 0.60),
    (axes[1], tau[m], "RECOVERED τ (corr-free obs point-cloud)", "viridis", 0.10, 0.60),
    (axes[2], (tau - tau_truth)[m], "ERROR (recovered - truth)", "coolwarm", -0.3, 0.3),
]:
    sc = ax.scatter(x, y, c=vals, s=7, cmap=cmap, vmin=vlo, vmax=vhi, linewidths=0)
    ax.set_title(title, fontsize=11); ax.set_aspect("equal"); ax.set_xlabel(f"axis {sweep}"); ax.set_ylabel(f"axis {la}")
    plt.colorbar(sc, ax=ax, fraction=0.046)
fig.suptitle(f"DIAG τ corr-free recovery — contrast {rec:.1f}%  hard {tau[hard].mean():.3f} / soft {tau[~hard].mean():.3f}", fontsize=12)
fig.tight_layout()
out = REPO / "tools/pointnetpp/viz_out/lightgraze_diag_tau.png"
fig.savefig(out, dpi=120, bbox_inches="tight"); print(f"saved {out}")
