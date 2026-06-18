"""5-panel τ-recovery video for the (b) mechanism (τ-coupled cohesive softening + balanced
position+occupancy loss, num_iter=1, tau_soften ON). Panels:
  1 GROUND TRUTH τ | 2 RANDOM init τ | 3 RECONSTRUCTED τ (both_bal, evolving) |
  4 OBS point-cloud cavity | 5 PRE-FRACTURE DEFORMATION (|disp from rest|, present particles)

Panel 5 is the (b) signature the old (occupancy-only, tau_soften OFF) videos could NOT show:
with τ in the position channel, SOFT regions deform MORE before fracture, HARD regions hold firm.
Title of panel 5 reports soft-mean vs hard-mean displacement = the woken position channel, quantified.
Outputs tau_recovery_b_{pattern}.gif (old occupancy gifs kept for side-by-side comparison).
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
N = solver.N; solver.damping = 0.95; solver.num_iterations = 1; solver.enable_cutting = True   # (b): num_iter=1
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
    solver.num_iterations = 1; solver.tau_soften = True            # (b) ON
    xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
    hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0) if pattern == "DIAG" \
        else (((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2)
    tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
    tau_init = rng.uniform(0.12, 0.58, N).astype(np.float32)
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
    # both_bal config: chamfer position loss (lambda 1) + scale-balanced occupancy (5e-4), (b) ON
    calibrate_online("Group+video_b", ctx, multi, N_passes=4, lr=0.03, lr_tau=0.5, lr_decay_per_pass=0.6,
                     lambda_hist=0.0, lambda_outer_cav=5.0e-4, return_best=True, line_search=False, verbose=False,
                     chamfer_loss=True, lambda_chamfer=1.0, cav_occupancy=True, cav_occ_pointcloud=True,
                     cav_occ_meannorm=True, snapshot_every=17, snapshot_out=snaps)
    solver.tau_soften = False
    return tau_truth, tau_init, snaps, hard, traj, fpr, op, oa


def make_video(pattern):
    tau_truth, tau_init, snaps, hard, traj, fpr, op, oa = run(pattern)
    m = surf; x = pr[m, sweep]; y = pr[m, la]
    surf_ids = np.where(surf)[0]
    op = np.asarray(op); oa = np.asarray(oa)
    steps = [s[0] for s in snaps]; taus = [s[1] for s in snaps]
    hard_s = hard[surf_ids]
    # pre-fracture deformation magnitude (mm) per surface particle per frame: |op - rest|
    disp_all = np.linalg.norm(op[:, surf_ids, :] - pr[surf_ids][None], axis=2) * 1e3   # (T, Ns) in mm
    dvmax = float(np.percentile(disp_all, 97))                                          # colormap cap

    fig, axes = plt.subplots(1, 5, figsize=(25, 4.7))
    vmin, vmax = 0.10, 0.60
    for ax in axes:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    sc0 = axes[0].scatter(x, y, c=tau_truth[m], s=7, cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0)
    axes[0].set_title("GROUND TRUTH τ", fontsize=12)
    axes[1].scatter(x, y, c=tau_init[m], s=7, cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0)
    axes[1].set_title("RANDOM sim init τ", fontsize=12)
    sc2 = axes[2].scatter(x, y, c=taus[0][m], s=7, cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0)
    rpath = np.array([[kp[sweep], kp[la]] for kp in traj[:fpr]])
    axes[2].plot(rpath[:, 0], rpath[:, 1], "-", color="white", lw=0.8, alpha=0.5)
    ktrail, = axes[2].plot([], [], "r-", lw=1.5, alpha=0.6)
    kmark, = axes[2].plot([], [], "ro", ms=11, mfc="none", mew=2.5)
    plt.colorbar(sc0, ax=axes[:3], fraction=0.012, pad=0.01)
    # panel 4: obs point cloud cavity
    axes[3].scatter(x, y, c="0.85", s=7, linewidths=0)
    f0 = steps[0] % len(traj); pres0 = oa[f0][surf_ids] > 0.5
    sc3 = axes[3].scatter(op[f0][surf_ids][pres0][:, sweep], op[f0][surf_ids][pres0][:, la], c="#1f77b4", s=7, linewidths=0)
    axes[3].set_xlim(*axes[0].get_xlim()); axes[3].set_ylim(*axes[0].get_ylim())
    axes[3].set_title("OBS point cloud (camera)\ncavity = holes where cut", fontsize=11)
    # panel 5: pre-fracture deformation magnitude (the (b) signature)
    axes[4].scatter(x, y, c="0.9", s=7, linewidths=0)
    sc4 = axes[4].scatter(op[f0][surf_ids][pres0][:, sweep], op[f0][surf_ids][pres0][:, la],
                          c=disp_all[f0][pres0], s=7, cmap="magma", vmin=0, vmax=dvmax, linewidths=0)
    axes[4].set_xlim(*axes[0].get_xlim()); axes[4].set_ylim(*axes[0].get_ylim())
    plt.colorbar(sc4, ax=axes[4], fraction=0.025, pad=0.01, label="|disp| (mm)")

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
        sc4.set_offsets(np.column_stack([op[cur][surf_ids][pres][:, sweep], op[cur][surf_ids][pres][:, la]]))
        sc4.set_array(disp_all[cur][pres])
        # quantify (b): soft present particles deform MORE than hard present ones
        ph = pres & hard_s; ps = pres & (~hard_s)
        ds = disp_all[cur][ps].mean() if ps.any() else 0.0
        dh = disp_all[cur][ph].mean() if ph.any() else 0.0
        axes[4].set_title(f"PRE-FRACTURE DEFORMATION (b)\nsoft {ds:.2f}mm vs hard {dh:.2f}mm", fontsize=11)
        return sc2, kmark, ktrail, sc3, sc4

    fig.suptitle(f"{pattern} — (b) τ-coupled softening: τ now lives in the position channel (panel 5)", fontsize=13)
    anim = FuncAnimation(fig, upd, frames=len(taus), interval=180)
    outd = REPO / "tools/pointnetpp/viz_out"; outd.mkdir(exist_ok=True)
    out = outd / f"tau_recovery_b_{pattern}.gif"
    anim.save(str(out), writer=PillowWriter(fps=6)); plt.close(fig)
    print(f"saved {out}  ({len(taus)} frames, final contrast {((taus[-1][hard].mean()-taus[-1][~hard].mean())/(TAU_HARD-TAU_SOFT)*100):.0f}%)", flush=True)


for p in ("DIAG", "SPOT"):
    make_video(p)
