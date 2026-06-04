"""Surface-inclusion experiment — inclusion located AT THE SURFACE where the
knife actually cuts, not buried in the interior.

Previous setup: incl_center_rel = (0.5, 0.80, 0.5) → buried interior.
Surface obs (top 10% slab) couldn't see interior tau → math under-determined.

This setup:
   cube/tumor: incl_center_rel = (0.5, 0.95, 0.5)  — Y axis top
   torus:      incl_center_rel = (0.75, 0.50, 0.95) — Z axis top

Inclusion overlaps with the top 10% surface slab AND with the knife
trajectory plane → knife actually passes through inclusion → cavity
shape differs based on tau → tau is observable from surface obs.

For each mesh × medium inclusion:
   1. Calibrate Liang + Hier under corrected physics + partial obs
   2. Report stiff_in, stiff_IoU, tau_in, tau_IoU
   3. Surface particles inside-vs-outside inclusion counts
   4. Sim-real surface overlap on held-out T_B
   5. Binary tau classification PNG (TP/FN/FP per method)
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))

import warp as wp  # noqa: E402
wp.init()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.stageA_inclusion_recovery import build_inclusion_truth, gen_obs_per_particle_stiff  # noqa: E402
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec  # noqa: E402
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402
from tools.pointnetpp.eval_hier_phantom_realistic import (  # noqa: E402
    predict_on_traj, surface_overlap_metrics, param_iou,
)


# SURFACE inclusion centers (overlap with top 10% slab)
SURFACE_INCL_CFG = {
    # name: (incl_center_rel, incl_radius)
    "M1_cube":  ((0.5, 0.95, 0.5), 0.07),     # y top-axis, near top
    "M2_tumor": ((0.5, 0.95, 0.5), 0.005),    # y top-axis
    "M3_torus": ((0.75, 0.50, 0.95), 0.08),   # z top-axis
}


def run_one(cfg, device, T_calib=10, T_pred=10, N_epochs=15):
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.95
    solver.num_iterations = 2
    solver.gravity = wp.vec3(0.0, 0.0, 0.0)
    pos_rest = mesh_info["vertices"]
    solver.floor_y = float(pos_rest[:, 2].min() - 0.1 * (pos_rest[:, 2].max() - pos_rest[:, 2].min()))

    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    top_axis = cfg["top_axis"]

    # Override: surface inclusion
    incl_rel, incl_radius = SURFACE_INCL_CFG[cfg["name"]]
    incl_center = bbox_min + np.array(incl_rel) * bbox_size

    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=cfg["stiff_in"], stiff_out=cfg["stiff_out"],
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    n_incl = int(inclusion_mask.sum())
    if n_incl == 0:
        print(f"  WARN: no particles inside inclusion at surface, skip")
        return None

    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx_np = np.where(surface_mask)[0]
    n_surface = int(surface_mask.sum())
    incl_at_surface = int((inclusion_mask & surface_mask).sum())
    print(f"  N={N}  n_incl={n_incl} ({n_incl/N*100:.2f}%)  n_surface={n_surface}  "
          f"incl∩surface={incl_at_surface} ({incl_at_surface/n_incl*100:.1f}% of inclusion on surface)", flush=True)

    theta_truth_np = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    traj_calib = make_traj_from_spec(cfg["traj_specs"]["A"], bbox_min, bbox_max, T_calib)
    traj_B = make_traj_from_spec(cfg["traj_specs"]["B"], bbox_min, bbox_max, T_pred)

    solver.reset_to_rest(pos_rest)
    obs_pos_calib, obs_alpha_calib = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_truth, traj_calib, knife_radius, device,
        tau_truth_np=tau_truth)
    solver.reset_to_rest(pos_rest)
    obs_pos_B, obs_alpha_B = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_truth, traj_B, knife_radius, device,
        tau_truth_np=tau_truth)

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos_calib]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha_calib]
    surface_idx = torch.tensor(surface_idx_np, dtype=torch.long, device=device)

    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edges_a_t = torch.tensor(edges_a_np, dtype=torch.long, device=device)
    edges_b_t = torch.tensor(edges_b_np, dtype=torch.long, device=device)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)

    x_rest = torch.from_numpy(pos_rest).to(device, dtype=torch.float32)
    M1 = min(512, max(32, N // 20))
    M2 = min(64, max(8, N // 200))
    H_hier = build_hierarchy(x_rest, M1=M1, M2=M2, k_neighbors=3)

    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)

    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=traj_calib, knife_radius=knife_radius,
        pos_rest=pos_rest, theta_truth=theta_truth_np, theta_3d_np=theta_truth_np,
        stiff_init_np=stiff_init, stiff_truth=stiff_truth,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        inclusion_mask=inclusion_mask,
        surface_idx=surface_idx, surface_mask=surface_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H_hier,
    )

    results = {}
    for method in ["Liang_inner", "Hier+Liang_inner"]:
        t0 = time.time()
        stiff, tau, _ = calibrate(method, ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                          lambda_smooth=0.0, lambda_E=1e-5)
        dt = time.time() - t0
        stiff_in_mean = float(stiff[inclusion_mask].mean()) if n_incl > 0 else 0.0
        stiff_out_mean = float(stiff[~inclusion_mask].mean())
        tau_in_mean = float(tau[inclusion_mask].mean()) if n_incl > 0 else 0.0
        tau_out_mean = float(tau[~inclusion_mask].mean())
        stiff_iou = param_iou(stiff, stiff_truth, threshold=0.5, direction=">")
        tau_iou = param_iou(tau, tau_truth, threshold=0.3, direction="<")
        # Precision/recall on tau<0.3
        pred_inside = tau < 0.3
        truth_inside = tau_truth < 0.3
        tp = int((pred_inside & truth_inside).sum())
        fn = int((~pred_inside & truth_inside).sum())
        fp = int((pred_inside & ~truth_inside).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        pred_pos, pred_alpha = predict_on_traj(stiff, tau, traj_B, ctx)
        sr = surface_overlap_metrics(pred_pos, obs_pos_B, pred_alpha, obs_alpha_B, surface_idx_np)
        results[method] = dict(
            stiff_in_mean=stiff_in_mean, stiff_out_mean=stiff_out_mean,
            tau_in_mean=tau_in_mean, tau_out_mean=tau_out_mean,
            stiff_iou=stiff_iou, tau_iou=tau_iou,
            tau_TP=tp, tau_FN=fn, tau_FP=fp,
            tau_precision=precision, tau_recall=recall,
            pos_mm=sr["pos_mm"], cav_iou=sr["cav_iou"], alpha_mse=sr["alpha_mse"],
            stiff=stiff, tau=tau, runtime=dt,
        )
        print(f"  {method:<18}  stiff_in={stiff_in_mean:.3f} (truth {cfg['stiff_in']:.2f}) iou={stiff_iou:.3f}  "
               f"tau_in={tau_in_mean:.3f} iou={tau_iou:.3f} P={precision:.2f} R={recall:.2f}  "
               f"simreal_B pos={sr['pos_mm']:.3f}mm cav_IoU={sr['cav_iou']:.3f}  ({dt:.0f}s)", flush=True)

    return dict(cfg=cfg, n_incl=n_incl, N=N, n_surface=n_surface,
                 incl_at_surface=incl_at_surface,
                 inclusion_mask=inclusion_mask, surface_mask=surface_mask,
                 incl_center=incl_center, incl_radius=incl_radius,
                 pos_rest=pos_rest, tau_truth=tau_truth, stiff_truth=stiff_truth,
                 results=results)


def render_binary_tau(data, mesh_name, out_path):
    pos = data["pos_rest"]
    incl_center = data["incl_center"]; incl_r = data["incl_radius"]
    top_axis = data["cfg"]["top_axis"]
    bbox_min = pos.min(0); bbox_max = pos.max(0)
    other = [i for i in range(3) if i != top_axis]
    ax_h, ax_v = other[0], other[1]
    tau_GT = data["tau_truth"]
    tau_L = data["results"]["Liang_inner"]["tau"]
    tau_H = data["results"]["Hier+Liang_inner"]["tau"]
    threshold = 0.3
    truth_in = tau_GT < threshold

    slice_val = incl_center[top_axis]
    slice_tol = incl_r * 1.5
    slab = np.abs(pos[:, top_axis] - slice_val) < slice_tol

    fig = plt.figure(figsize=(20, 12))
    # Row 1: cross-section TP/FN/FP
    panels = [("Truth", tau_GT < threshold, None),
              ("Liang vs Truth", tau_L < threshold, "Liang"),
              ("Hier+Liang vs Truth", tau_H < threshold, "Hier")]
    for j, (label, pred, mname) in enumerate(panels):
        ax = fig.add_subplot(2, 3, j + 1)
        slab_pos = pos[slab]; p = pred[slab]; t = truth_in[slab]
        if mname is None:
            ax.scatter(slab_pos[~p, ax_h], slab_pos[~p, ax_v], c="lightblue",
                        s=20, edgecolors="none", alpha=0.5, label=f"τ≥0.3 ({(~p).sum()})")
            ax.scatter(slab_pos[p, ax_h], slab_pos[p, ax_v], c="red",
                        s=30, edgecolors="darkred", linewidth=0.5,
                        label=f"τ<0.3 ({p.sum()})")
        else:
            tp = p & t; fn = ~p & t; fp = p & ~t; tn = ~p & ~t
            ax.scatter(slab_pos[tn, ax_h], slab_pos[tn, ax_v], c="lightgray",
                        s=15, edgecolors="none", alpha=0.4, label=f"TN ({tn.sum()})")
            ax.scatter(slab_pos[tp, ax_h], slab_pos[tp, ax_v], c="green",
                        s=28, edgecolors="none", label=f"TP ({tp.sum()})")
            ax.scatter(slab_pos[fn, ax_h], slab_pos[fn, ax_v], c="red",
                        s=30, edgecolors="darkred", linewidth=0.5, label=f"FN ({fn.sum()})")
            ax.scatter(slab_pos[fp, ax_h], slab_pos[fp, ax_v], c="orange",
                        s=30, edgecolors="darkorange", linewidth=0.5, label=f"FP ({fp.sum()})")
            union = (p | t).sum(); iou = 0.0 if union == 0 else int((p & t).sum()) / union
            ax.text(0.5, 1.05, f"IoU = {iou:.3f}", transform=ax.transAxes,
                     ha="center", fontsize=11, fontweight="bold",
                     bbox=dict(boxstyle="round", fc="yellow", alpha=0.7))
        ax.add_patch(Circle((incl_center[ax_h], incl_center[ax_v]), incl_r,
                              fill=False, color="green", linewidth=2.0, linestyle="--"))
        ax.set_xlim(bbox_min[ax_h], bbox_max[ax_h])
        ax.set_ylim(bbox_min[ax_v], bbox_max[ax_v])
        ax.set_xlabel(["x", "y", "z"][ax_h])
        ax.set_ylabel(["x", "y", "z"][ax_v])
        ax.set_aspect("equal")
        ax.set_title(f"{label}\nslab through {['x','y','z'][top_axis]}={slice_val:.3f}",
                       fontsize=10)
        ax.legend(loc="lower right", fontsize=8)

    # Row 2: histograms
    for j, (tau, name) in enumerate([(tau_GT, "GT"), (tau_L, "Liang"), (tau_H, "Hier")]):
        ax = fig.add_subplot(2, 3, 4 + j)
        in_t = tau[data["inclusion_mask"]]
        out_t = tau[~data["inclusion_mask"]]
        all_min = min(in_t.min(), out_t.min()); all_max = max(in_t.max(), out_t.max())
        rng = max(all_max - all_min, 1e-4)
        bins = np.linspace(all_min - rng * 0.05, all_max + rng * 0.05, 60)
        ax.hist(out_t, bins=bins, alpha=0.6, color="lightblue",
                 label=f"outside (truth τ=0.40)", density=True)
        ax.hist(in_t, bins=bins, alpha=0.6, color="red",
                 label=f"inside (truth τ=0.20)", density=True)
        ax.axvline(0.3, color="black", linestyle="--", linewidth=1.5, label="τ=0.3 threshold")
        ax.set_xlabel("τ value"); ax.set_ylabel("density")
        ax.set_title(f"{name}  in_mean={in_t.mean():.4f}  out_mean={out_t.mean():.4f}  "
                       f"in_std={in_t.std():.5f}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"{mesh_name}  ·  SURFACE INCLUSION  ·  binary tau classification + histograms\n"
                  f"Inclusion overlaps surface (top 10%) where knife actually cuts",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    device = "cuda:0"
    all_res = []
    t0 = time.time()
    for cfg in MESH_CONFIGS:
        print(f"\n{'='*80}\n{cfg['name']}  SURFACE INCLUSION\n{'='*80}", flush=True)
        try:
            data = run_one(cfg, device, T_calib=10, T_pred=10, N_epochs=15)
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}"); traceback.print_exc(); continue
        if data is None: continue
        all_res.append(data)
        png_path = REPO / "tools" / "pointnetpp" / f"surface_incl_binary_{cfg['name']}.png"
        render_binary_tau(data, cfg["name"], png_path)
        print(f"  saved {png_path}", flush=True)

    # Summary
    print(f"\n{'='*100}")
    print(f"SURFACE INCLUSION RESULTS — hier vs Liang under realistic phantom setup")
    print(f"{'='*100}")
    print(f"{'mesh':<10} {'method':<18} "
          f"{'stiff_in':>9} {'st_iou':>7} {'tau_in':>8} {'tau_iou':>8} "
          f"{'P':>5} {'R':>5} {'pos_mm':>8} {'cav_IoU':>8}")
    print("-" * 100)
    for data in all_res:
        for m in ["Liang_inner", "Hier+Liang_inner"]:
            r = data["results"][m]
            print(f"{data['cfg']['name']:<10} {m:<18} "
                   f"{r['stiff_in_mean']:>9.3f} {r['stiff_iou']:>7.3f} "
                   f"{r['tau_in_mean']:>8.3f} {r['tau_iou']:>8.3f} "
                   f"{r['tau_precision']:>5.2f} {r['tau_recall']:>5.2f} "
                   f"{r['pos_mm']:>8.3f} {r['cav_iou']:>8.3f}")
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
