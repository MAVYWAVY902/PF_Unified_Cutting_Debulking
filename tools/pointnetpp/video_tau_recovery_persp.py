"""5-panel τ-recovery video for the (b) mechanism (τ-coupled cohesive softening + balanced
position+occupancy loss, num_iter=1, tau_soften ON). Panels:
  1 GROUND TRUTH τ | 2 RANDOM init τ | 3 RECONSTRUCTED τ (both_bal, evolving) |
  4 OBS point-cloud cavity | 5 PRE-FRACTURE DEFORMATION (|disp from rest|, present particles)

Panel 5 is the (b) signature the old (occupancy-only, tau_soften OFF) videos could NOT show:
with τ in the position channel, SOFT regions deform MORE before fracture, HARD regions hold firm.
Title of panel 5 reports soft-mean vs hard-mean displacement = the woken position channel, quantified.
Outputs tau_recovery_b09_{pattern}.gif (old occupancy gifs kept for side-by-side comparison).
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables projection='3d')
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

TAU_HARD, TAU_SOFT, STIFF = 0.55, 0.15, 0.9   # 0.9 + num_iter=6: firm self-supporting solid (diag_water_*: 3% sag, stable cut)
NUM_ITER = 6                                   # was 1 (softest, forced by blow-up myth) → 6 is firm AND stable on shallow graze
REACH_MM, R = 4.0, 3
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = NUM_ITER; solver.enable_cutting = True   # firm self-support
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
tht = torch.tensor([3., 20., 0.3], device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.; st = rest0
# settle 140 frames: the cube UNDER-DAMPED-BOUNCES (sag 0→10.9→7.0→7.9mm), dead-still only by ~f120
# (diag_settle_time.py). 60 caught it mid-bounce → it kept sinking during the cut. 140 = fully static first.
with torch.no_grad():
    for _ in range(140):
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
    solver.num_iterations = NUM_ITER; solver.tau_soften = True     # (b) ON, firm
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
    op = np.asarray(op); oa = np.asarray(oa)
    steps = [s[0] for s in snaps]; taus = [s[1] for s in snaps]
    # subsample for 3D speed (fixed indices, whole body so cut + sag are visible in perspective)
    rng = np.random.default_rng(1)
    sub = rng.choice(N, size=min(8000, N), replace=False)
    disp_all = np.linalg.norm(op[:, sub, :] - pr[sub][None], axis=2) * 1e3   # (T, |sub|) mm from settled rest
    dvmax = float(np.percentile(disp_all, 97))
    # plot coords: horizontal = lateral (X=sweep, Z=la), VERTICAL = top axis (Y, the gravity/depth axis)
    PX, PY, PZ = sweep, la, top      # PZ (height) = Y so gravity is "down" and sag is visible
    vmin, vmax = 0.10, 0.60
    elev, azim = 22, -62
    ghost = pr[sub]                  # static full-body footprint for the camera panel (holes = removed)

    fig = plt.figure(figsize=(33, 6.5))
    titles = ["GROUND TRUTH τ (hard vs soft)", "RANDOM init τ (what sim starts from)",
              "RECONSTRUCTED τ (evolving)", "OBS point cloud (camera)\ncavity = holes where cut",
              "DEFORMATION |disp| mm (sag + cut)"]
    axs = [fig.add_subplot(1, 5, k + 1, projection="3d") for k in range(5)]

    def style(ax, t):
        ax.set_title(t, fontsize=11)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_xlim(pr[:, PX].min(), pr[:, PX].max())
        ax.set_ylim(pr[:, PY].min(), pr[:, PY].max())
        ax.set_zlim(pr[:, PZ].min(), pr[:, PZ].max() + 0.04)   # headroom for knife
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim)
        ax.set_zlabel("Y (gravity ↓ / depth)", fontsize=8)

    f0 = steps[0] % len(traj); pres0 = oa[f0][sub] > 0.5
    P0 = op[f0][sub]
    sc_gt = axs[0].scatter(P0[pres0, PX], P0[pres0, PY], P0[pres0, PZ], c=tau_truth[sub][pres0], cmap="viridis", vmin=vmin, vmax=vmax, s=4, depthshade=True, linewidths=0)
    sc_in = axs[1].scatter(P0[pres0, PX], P0[pres0, PY], P0[pres0, PZ], c=tau_init[sub][pres0], cmap="viridis", vmin=vmin, vmax=vmax, s=4, depthshade=True, linewidths=0)
    sc_rc = axs[2].scatter(P0[pres0, PX], P0[pres0, PY], P0[pres0, PZ], c=taus[0][sub][pres0], cmap="viridis", vmin=vmin, vmax=vmax, s=4, depthshade=True, linewidths=0)
    axs[3].scatter(ghost[:, PX], ghost[:, PY], ghost[:, PZ], c="0.85", s=3, depthshade=False, linewidths=0)   # static ghost → holes pop
    sc_cam = axs[3].scatter(P0[pres0, PX], P0[pres0, PY], P0[pres0, PZ], c="#1f77b4", s=4, depthshade=True, linewidths=0)
    sc_df = axs[4].scatter(P0[pres0, PX], P0[pres0, PY], P0[pres0, PZ], c=disp_all[f0][pres0], cmap="magma", vmin=0, vmax=dvmax, s=4, depthshade=True, linewidths=0)
    km = [a.scatter([traj[f0][PX]], [traj[f0][PY]], [traj[f0][PZ]], c="red", s=80, marker="o") for a in axs]
    for a, t in zip(axs, titles): style(a, t)
    plt.colorbar(sc_gt, ax=axs[2], fraction=0.025, pad=0.10, label="τ (toughness)")
    plt.colorbar(sc_df, ax=axs[4], fraction=0.025, pad=0.10, label="|disp| (mm)")

    def upd(i):
        cur = steps[i] % len(traj); pres = oa[cur][sub] > 0.5
        P = op[cur][sub]
        xs, ys, zs = P[pres, PX], P[pres, PY], P[pres, PZ]
        for sc, c in ((sc_gt, tau_truth[sub][pres]), (sc_in, tau_init[sub][pres]),
                      (sc_rc, taus[i][sub][pres]), (sc_df, disp_all[cur][pres])):
            sc._offsets3d = (xs, ys, zs); sc.set_array(c)
        sc_cam._offsets3d = (xs, ys, zs)
        for k in km:
            k._offsets3d = ([traj[cur][PX]], [traj[cur][PY]], [traj[cur][PZ]])
        tau = taus[i]
        ct = (tau[hard].mean() - tau[~hard].mean()) / (TAU_HARD - TAU_SOFT) * 100
        ds = disp_all[cur][tau_truth[sub] < 0.3].mean(); dh = disp_all[cur][tau_truth[sub] > 0.3].mean()
        axs[2].set_title(f"RECONSTRUCTED τ — step {steps[i]}\nhard {tau[hard].mean():.2f} / soft {tau[~hard].mean():.2f} ({ct:.0f}%)", fontsize=10)
        axs[4].set_title(f"DEFORMATION |disp| mm\nsoft {ds:.1f} > hard {dh:.1f} (the (b) signal)", fontsize=10)
        return sc_gt, sc_in, sc_rc, sc_cam, sc_df

    fig.suptitle(f"{pattern} — PERSPECTIVE 5-panel, STIFF={STIFF} both_bal: Y(vertical)=gravity → sag + cut depth visible", fontsize=13)
    anim = FuncAnimation(fig, upd, frames=len(taus), interval=180)
    outd = REPO / "tools/pointnetpp/viz_out"; outd.mkdir(exist_ok=True)
    out = outd / f"tau_recovery_persp_firm_{pattern}.gif"
    anim.save(str(out), writer=PillowWriter(fps=6)); plt.close(fig)
    print(f"saved {out}  ({len(taus)} frames)", flush=True)


for p in ("DIAG", "SPOT"):
    make_video(p)
