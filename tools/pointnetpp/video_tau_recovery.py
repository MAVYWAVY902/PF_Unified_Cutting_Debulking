"""Row of 3 animated panels: GROUND-TRUTH τ | RANDOM sim init τ | RECONSTRUCTED τ (evolving as the
raster optimizes). For DIAG and SPOT. Uses the validated real-path config (light-graze excitation +
corr-free point-cloud occupancy + free-run + sampling-invariant mean-norm). Top-down surface view,
colored by τ; knife position overlaid on the reconstruction panel.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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
tht = torch.tensor([3., 20., 0.3], device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.; st = rest0
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, swt(STIFF), swt(0.3)); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
surf = pr[:, top] > (bb1[top] - 0.10 * bs[top]); surf_idx = torch.tensor(np.where(surf)[0], dtype=torch.long, device=dev)
col_id, _ = column_grid_assignment(pr, surf, top, grid_n=8)
ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
eat = torch.tensor(ea, dtype=torch.long, device=dev); ebt = torch.tensor(eb, dtype=torch.long, device=dev)
ert = torch.tensor(np.linalg.norm(pr[ea] - pr[eb], axis=1), dtype=torch.float32, device=dev)
H = build_hierarchy(torch.from_numpy(pr).to(dev, torch.float32), M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)


def run(pattern):
    rng = np.random.default_rng(0)
    xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
    hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0) if pattern == "DIAG" \
        else (((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2)
    tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
    tau_init = rng.uniform(0.12, 0.58, N).astype(np.float32)            # RANDOM init
    y_center = bb1[top] - REACH_MM * 1e-3 + kr
    base = make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)
    one = [(kp.copy().astype(np.float32)) for kp in base]
    for kp in one: kp[top] = np.float32(y_center)
    traj = one * R; fpr = len(one)
    theta_truth = np.array([3., 20., 0.3], np.float32)
    ctx = dict(solver=solver, rest_state=rest_state, N=N, device=dev, knife_radius=kr, pos_rest=pr,
               theta_3d_np=theta_truth, stiff_init_np=np.full(N, STIFF, np.float32), tau_init_np=tau_init,
               tau_bound=(0.05, 0.95), surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt, edges_rest_t=ert, H=H,
               group_col_id=col_id, surf_top_axis=top)
    solver.reset_to_rest(pr)
    op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta_truth, np.full(N, STIFF, np.float32), traj, kr, dev, tau_truth_np=tau_truth)
    multi = [(traj, [torch.tensor(p, device=dev) for p in op], [torch.tensor(a, device=dev) for a in oa])]
    snaps = []
    calibrate_online("Group+video", ctx, multi, N_passes=4, lr=0.03, lr_tau=0.5, lr_decay_per_pass=0.6,
                     lambda_hist=0.0, lambda_outer_cav=1.0, return_best=True, line_search=False, verbose=False,
                     chamfer_loss=True, lambda_chamfer=0.0, cav_occupancy=True, cav_occ_pointcloud=True,
                     cav_occ_meannorm=True, snapshot_every=17, snapshot_out=snaps)   # 17 coprime with t_line=12 (no aliasing)
    return tau_truth, tau_init, snaps, hard, traj, fpr, op, oa


def make_video(pattern):
    tau_truth, tau_init, snaps, hard, traj, fpr, op, oa = run(pattern)
    m = surf; x = pr[m, sweep]; y = pr[m, la]
    surf_ids = np.where(surf)[0]
    op = np.asarray(op); oa = np.asarray(oa)        # (T, N, 3), (T, N)
    steps = [s[0] for s in snaps]; taus = [s[1] for s in snaps]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.7))
    vmin, vmax = 0.10, 0.60
    for ax in axes:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    sc0 = axes[0].scatter(x, y, c=tau_truth[m], s=7, cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0)
    axes[0].set_title("GROUND TRUTH τ", fontsize=12)
    axes[1].scatter(x, y, c=tau_init[m], s=7, cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0)
    axes[1].set_title("RANDOM sim init τ", fontsize=12)
    sc2 = axes[2].scatter(x, y, c=taus[0][m], s=7, cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0)
    # faint full raster path (one rep) so the boustrophedon sweep is visible
    rpath = np.array([[kp[sweep], kp[la]] for kp in traj[:fpr]])
    axes[2].plot(rpath[:, 0], rpath[:, 1], "-", color="white", lw=0.8, alpha=0.5)
    ktrail, = axes[2].plot([], [], "r-", lw=1.5, alpha=0.6)        # recent sweep trail
    kmark, = axes[2].plot([], [], "ro", ms=11, mfc="none", mew=2.5)
    # panel 4: obs point cloud the camera sees — faint full footprint (background) + present points (foreground).
    # holes appear where the knife has removed surface points (alpha<=0.5 → absent from cloud).
    axes[3].scatter(x, y, c="0.85", s=7, linewidths=0)            # static ghost of full surface
    f0 = steps[0] % len(traj); pres0 = oa[f0][surf_ids] > 0.5
    sc3 = axes[3].scatter(op[f0][surf_ids][pres0][:, sweep], op[f0][surf_ids][pres0][:, la],
                          c="#1f77b4", s=7, linewidths=0)
    axes[3].set_xlim(*axes[0].get_xlim()); axes[3].set_ylim(*axes[0].get_ylim())
    axes[3].set_title("OBS point cloud (camera)\ncavity = holes where cut", fontsize=11)
    plt.colorbar(sc0, ax=axes, fraction=0.018, pad=0.01)

    def upd(i):
        tau = taus[i]
        sc2.set_array(tau[m])
        c = (tau[hard].mean() - tau[~hard].mean()) / (TAU_HARD - TAU_SOFT) * 100
        axes[2].set_title(f"RECONSTRUCTED τ — step {steps[i]}\nhard {tau[hard].mean():.2f} / soft {tau[~hard].mean():.2f} (contrast {c:.0f}%)", fontsize=11)
        cur = steps[i] % len(traj)
        kp = traj[cur]; kmark.set_data([kp[sweep]], [kp[la]])
        lo = max(0, cur - 17); seg = traj[lo:cur + 1]
        ktrail.set_data([q[sweep] for q in seg], [q[la] for q in seg])
        pres = oa[cur][surf_ids] > 0.5
        sc3.set_offsets(np.column_stack([op[cur][surf_ids][pres][:, sweep], op[cur][surf_ids][pres][:, la]]))
        return sc2, kmark, ktrail, sc3

    fig.suptitle(f"{pattern} — corr-free τ reconstruction (light-graze raster, real-D405 pipeline)", fontsize=13)
    anim = FuncAnimation(fig, upd, frames=len(taus), interval=180)
    outd = REPO / "tools/pointnetpp/viz_out"; outd.mkdir(exist_ok=True)
    out = outd / f"tau_recovery_{pattern}.gif"
    anim.save(str(out), writer=PillowWriter(fps=6)); plt.close(fig)
    print(f"saved {out}  ({len(taus)} frames, final contrast {((taus[-1][hard].mean()-taus[-1][~hard].mean())/(TAU_HARD-TAU_SOFT)*100):.0f}%)", flush=True)


for p in ("DIAG", "SPOT"):
    make_video(p)
