"""Convergence triptych+ video: top-down tau map evolving over optimization.

4 panels: Truth | sim-init | GROUP | per-particle
View    : top-down — each pixel = MEAN recovered tau over that lateral column
          (along depth). This exposes the headline gap: per-particle's surface
          moves but interior stays at init -> column-mean stays washed out;
          GROUP's interior inherits -> the pattern lights up.
Time    : optimization step (snapshots captured inside calibrate_online).
Color   : tau (soft=blue -> hard=red).

One GIF per lateral pattern (L/R, diagonal stripes, lateral spot).
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[2]
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

TAU_HARD = 0.55; TAU_SOFT = 0.15; TAU_INIT = 0.30; STIFF_UNIFORM = 0.5
GN = 22                       # render grid resolution (lateral)
OUT = REPO / "tools/pointnetpp/viz_out"; OUT.mkdir(exist_ok=True)


def col_mean_image(tau, xn, zn, gn=GN):
    """Per lateral cell, mean tau over all particles (all depths) -> (gn,gn) image."""
    ix = np.clip((xn * gn).astype(int), 0, gn - 1)
    iz = np.clip((zn * gn).astype(int), 0, gn - 1)
    flat = ix * gn + iz
    img = np.full(gn * gn, np.nan, np.float32)
    sums = np.bincount(flat, weights=tau, minlength=gn * gn)
    cnts = np.bincount(flat, minlength=gn * gn)
    nz = cnts > 0
    img[nz] = sums[nz] / cnts[nz]
    return img.reshape(gn, gn)


def main():
    cfg = MESH_CONFIGS[0]; device = "cuda:0"
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]),
                                                            fix_threshold_fraction=0.20)
    N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos_rest[:, 2].min() - 1.0)
    kr = cfg["knife_radius_abs"]; top_axis = cfg["top_axis"]

    solver.reset_to_rest(pos_rest); far = pos_rest.max(0) + 10.0
    sw = torch.tensor(np.full(N, STIFF_UNIFORM, np.float32), device=device)
    twt = torch.tensor(np.full(N, TAU_INIT, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device); st = rest_state
    with torch.no_grad():
        for _ in range(60):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, twt); st = solver_save_state(solver)
    pos_rest = pw.cpu().numpy().copy(); mesh_info["vertices"] = pos_rest; rest_state = st
    bb0, bb1 = pos_rest.min(0), pos_rest.max(0); sp = bb1 - bb0
    lateral = [a for a in range(3) if a != top_axis]
    sweep_axis, lat_axis = lateral[0], lateral[1]
    xn = (pos_rest[:, sweep_axis] - bb0[sweep_axis]) / sp[sweep_axis]
    zn = (pos_rest[:, lat_axis] - bb0[lat_axis]) / sp[lat_axis]

    PATTERNS = {
        "LR_HALVES":   xn < 0.5,
        "DIAG_STRIPES": (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0),
        "LATERAL_SPOT": ((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2,
    }

    surf = pos_rest[:, top_axis] > (bb1[top_axis] - 0.10 * sp[top_axis])
    surf_idx = torch.tensor(np.where(surf)[0], dtype=torch.long, device=device)
    col_id, _ = column_grid_assignment(pos_rest, surf, top_axis, grid_n=8)

    trajA = make_raster(bb0, bb1, top_axis, sweep_axis, lat_axis, 0.93, 10, 30)
    trajB = make_raster(bb0, bb1, top_axis, sweep_axis, lat_axis, 0.88, 10, 30)

    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    eat = torch.tensor(ea, dtype=torch.long, device=device); ebt = torch.tensor(eb, dtype=torch.long, device=device)
    erl = np.linalg.norm(pos_rest[ea] - pos_rest[eb], axis=1)
    ert = torch.tensor(erl, dtype=torch.float32, device=device)
    xr = torch.from_numpy(pos_rest).to(device, torch.float32)
    H = build_hierarchy(xr, M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)
    stiff_init = np.full(N, STIFF_UNIFORM, np.float32); tau_init = np.full(N, TAU_INIT, np.float32)
    stiff_uniform = np.full(N, STIFF_UNIFORM, np.float32)
    theta_truth = np.array([3., 20., 0.3], np.float32)
    ctx = dict(solver=solver, rest_state=rest_state, N=N, device=device, knife_radius=kr,
               pos_rest=pos_rest, theta_3d_np=theta_truth, stiff_init_np=stiff_init, tau_init_np=tau_init,
               tau_bound=(0.05, 0.95), surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt,
               edges_rest_t=ert, H=H, group_col_id=col_id)

    init_img = col_mean_image(np.full(N, TAU_INIT, np.float32), xn, zn)

    for pname, hard in PATTERNS.items():
        tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
        truth_img = col_mean_image(tau_truth, xn, zn)
        multi = []
        for traj in (trajA, trajB):
            solver.reset_to_rest(pos_rest)
            op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta_truth, stiff_uniform,
                                                traj, kr, device, tau_truth_np=tau_truth)
            multi.append((traj, [torch.tensor(p, device=device) for p in op],
                          [torch.tensor(a, device=device) for a in oa]))

        snaps = {}
        for cond, key in [("Group+L_pos_partial", "GROUP"), ("L_pos_partial", "per-particle")]:
            out = []
            t0 = time.time()
            calibrate_online(cond, ctx, multi, N_passes=4, lr=0.03, lr_tau=0.5,
                             lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=1.0,
                             return_best=False, snapshot_every=80, snapshot_out=out,
                             log_every=99999, verbose=False)
            snaps[key] = out
            print(f"  [{pname}] {key}: {len(out)} snapshots, {time.time()-t0:.0f}s", flush=True)

        # align frame counts
        nf = min(len(snaps["GROUP"]), len(snaps["per-particle"]))
        g_imgs = [col_mean_image(snaps["GROUP"][i][1], xn, zn) for i in range(nf)]
        p_imgs = [col_mean_image(snaps["per-particle"][i][1], xn, zn) for i in range(nf)]
        steps = [snaps["GROUP"][i][0] for i in range(nf)]

        fig, ax = plt.subplots(1, 4, figsize=(16, 4.3))
        kw = dict(origin="lower", vmin=TAU_SOFT, vmax=TAU_HARD, cmap="coolwarm")
        ax[0].imshow(truth_img, **kw); ax[0].set_title("Truth τ")
        ax[1].imshow(init_img, **kw); ax[1].set_title("sim init (τ=0.30)")
        im2 = ax[2].imshow(g_imgs[0], **kw); t2 = ax[2].set_title("GROUP")
        im3 = ax[3].imshow(p_imgs[0], **kw); t3 = ax[3].set_title("per-particle")
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        cb = fig.colorbar(im2, ax=ax, fraction=0.025, pad=0.01); cb.set_label("τ (toughness)")
        sup = fig.suptitle("", fontsize=13)

        def update(f):
            im2.set_data(g_imgs[f]); im3.set_data(p_imgs[f])
            gc = (np.nanmean(g_imgs[f][truth_img > 0.5 * (TAU_HARD + TAU_SOFT)]) -
                  np.nanmean(g_imgs[f][truth_img < 0.5 * (TAU_HARD + TAU_SOFT)]))
            pc = (np.nanmean(p_imgs[f][truth_img > 0.5 * (TAU_HARD + TAU_SOFT)]) -
                  np.nanmean(p_imgs[f][truth_img < 0.5 * (TAU_HARD + TAU_SOFT)]))
            t2.set_text(f"GROUP  (contrast {gc/(TAU_HARD-TAU_SOFT)*100:.0f}%)")
            t3.set_text(f"per-particle  (contrast {pc/(TAU_HARD-TAU_SOFT)*100:.0f}%)")
            sup.set_text(f"{pname}  —  top-down column-mean τ  —  opt step {steps[f]}")
            return im2, im3, t2, t3, sup

        anim = FuncAnimation(fig, update, frames=nf, blit=False)
        gif = OUT / f"converge_{pname}.gif"
        anim.save(str(gif), writer=PillowWriter(fps=4)); plt.close(fig)
        print(f"  -> saved {gif}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
