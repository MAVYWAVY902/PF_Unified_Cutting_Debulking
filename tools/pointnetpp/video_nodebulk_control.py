"""NO-DEBULKING CONTROL video: same cube, same firm settings (num_iter=6, k=0.9), settle 140 frames,
then HOLD under gravity with the knife parked FAR AWAY (no cutting) for the same duration as the cut.
Side-by-side with tau_recovery_persp_firm_{DIAG,SPOT}.gif this isolates the question: is the 'sinking'
during the firm videos from incomplete settling, or from the debulk hollowing the cube out?
If THIS control stays flat ~7.9mm sag while the cut videos climb to ~21.7mm → the sink is debulk-induced.
3 panels: τ truth on the body | body (camera view, NO holes since no cut) | deformation |disp| + live sag.
Body is identical for DIAG/SPOT without a cut, so τ colouring is the only difference (one file per pattern).
"""
import sys, os, contextlib
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
_DN = open(os.devnull, "w")
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster

TAU_HARD, TAU_SOFT, STIFF, NUM_ITER, REACH_MM, R = 0.55, 0.15, 0.9, 6, 4.0, 3
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = NUM_ITER; solver.enable_cutting = True; solver.muller_stiff_remap = False
pr0 = mi["vertices"]; top = cfg["top_axis"]; kr = cfg["knife_radius_abs"]
solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
top0 = pr0[:, top].max(); far = pr0.max(0) + 10.0
tht = torch.tensor([3., 20., 0.3], device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)

# settle 140 (identical to the firm video)
solver.tau_soften = False
solver.reset_to_rest(pr0); st = rest0
with torch.no_grad(), contextlib.redirect_stdout(_DN):
    for _ in range(140):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, swt(STIFF), swt(0.3)); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
sag0 = (top0 - pr[:, top].max()) * 1e3
print(f"settled sag = {sag0:.2f} mm; holding {len(make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)) * R} frames with NO knife", flush=True)

# HOLD: knife parked far for the same number of frames as the cut trajectory (no cutting at all)
n_hold = len(make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)) * R
far_traj = [far.astype(np.float32) for _ in range(n_hold)]
solver.tau_soften = True
with contextlib.redirect_stdout(_DN):
    op, oa = gen_obs_per_particle_stiff(solver, rest_state, np.array([3., 20., 0.3], np.float32),
                                        np.full(N, STIFF, np.float32), far_traj, kr, dev,
                                        tau_truth_np=np.full(N, TAU_SOFT, np.float32))
op = np.asarray(op)


def make_video(pattern):
    xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
    hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0) if pattern == "DIAG" \
        else (((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2)
    tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
    rng = np.random.default_rng(1); sub = rng.choice(N, size=min(8000, N), replace=False)
    disp_all = np.linalg.norm(op[:, sub, :] - pr[sub][None], axis=2) * 1e3
    dvmax = max(1.0, float(np.percentile(disp_all, 97)))
    frames = list(range(0, len(op), max(1, len(op) // 51)))
    PX, PY, PZ = sweep, la, top
    vmin, vmax = 0.10, 0.60; elev, azim = 22, -62

    fig = plt.figure(figsize=(20, 6.5))
    titles = ["GROUND TRUTH τ (NO cut)", "BODY (camera) — NO holes, no cut", "DEFORMATION |disp| mm + sag"]
    axs = [fig.add_subplot(1, 3, k + 1, projection="3d") for k in range(3)]

    def style(ax, t):
        ax.set_title(t, fontsize=11); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_xlim(pr[:, PX].min(), pr[:, PX].max()); ax.set_ylim(pr[:, PY].min(), pr[:, PY].max())
        ax.set_zlim(pr[:, PZ].min(), pr[:, PZ].max() + 0.04); ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim); ax.set_zlabel("Y (gravity ↓)", fontsize=8)

    f0 = frames[0]; P0 = op[f0][sub]
    sc_gt = axs[0].scatter(P0[:, PX], P0[:, PY], P0[:, PZ], c=tau_truth[sub], cmap="viridis", vmin=vmin, vmax=vmax, s=4, depthshade=True, linewidths=0)
    sc_bd = axs[1].scatter(P0[:, PX], P0[:, PY], P0[:, PZ], c="#1f77b4", s=4, depthshade=True, linewidths=0)
    sc_df = axs[2].scatter(P0[:, PX], P0[:, PY], P0[:, PZ], c=disp_all[f0], cmap="magma", vmin=0, vmax=dvmax, s=4, depthshade=True, linewidths=0)
    for a, t in zip(axs, titles): style(a, t)
    plt.colorbar(sc_df, ax=axs[2], fraction=0.025, pad=0.10, label="|disp| (mm)")

    def upd(i):
        cur = frames[i]; P = op[cur][sub]
        for sc, c in ((sc_gt, tau_truth[sub]), (sc_df, disp_all[cur])):
            sc._offsets3d = (P[:, PX], P[:, PY], P[:, PZ]); sc.set_array(c)
        sc_bd._offsets3d = (P[:, PX], P[:, PY], P[:, PZ])
        sag = (top0 - op[cur][:, top].max()) * 1e3
        axs[2].set_title(f"DEFORMATION |disp| mm — hold frame {cur}\nsag {sag:.1f} mm (NO cut → should stay flat)", fontsize=10)
        return sc_gt, sc_bd, sc_df

    fig.suptitle(f"{pattern} — NO-DEBULK CONTROL, firm (num_iter=6,k=0.9), settle 140 then HOLD (no knife)", fontsize=13)
    anim = FuncAnimation(fig, upd, frames=len(frames), interval=180)
    outd = REPO / "tools/pointnetpp/viz_out"; outd.mkdir(exist_ok=True)
    out = outd / f"tau_recovery_persp_nodebulk_{pattern}.gif"
    anim.save(str(out), writer=PillowWriter(fps=6)); plt.close(fig)
    sag_end = (top0 - op[-1][:, top].max()) * 1e3
    print(f"saved {out}  ({len(frames)} frames)  sag: start {sag0:.1f} → end {sag_end:.1f} mm", flush=True)


for p in ("DIAG", "SPOT"):
    make_video(p)
