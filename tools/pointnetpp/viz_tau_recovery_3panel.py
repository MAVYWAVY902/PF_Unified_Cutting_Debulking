"""3D perspective videos comparing GT | Liang | Hier on tau recovery.

For each mesh (cube, tumor, torus):
   1. Calibrate Liang_inner and Hier+Liang_inner under partial obs + corrected physics
   2. Forward T_B (held-out) with 3 param sets:
        - GT: stiff_truth, tau_truth
        - Liang-only: calibrated stiff_L, tau_L
        - Hier: calibrated stiff_H, tau_H
   3. Render 3-panel 3D perspective animation comparing cavity formation
   4. Save 3-panel GIF

Uses MEDIUM inclusion per mesh (balanced regime where hier tau win is clear).
Viewing angle slowly rotates for parallax depth.
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
from PIL import Image  # noqa: E402

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.stageA_inclusion_recovery import build_inclusion_truth, gen_obs_per_particle_stiff  # noqa: E402
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec  # noqa: E402
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402
from tools.pointnetpp.eval_hier_phantom_realistic import predict_on_traj  # noqa: E402


# Medium inclusion radii per mesh (balanced regime per 3-mesh sweep)
MEDIUM_INCL = {
    "M1_cube":  0.07,    # 4.2% volume
    "M2_tumor": 0.005,   # 7.2% volume
    "M3_torus": 0.08,    # 1.2% volume
}


def render_3panel_frame(pos_T, alpha_T, pos_L, alpha_L, pos_H, alpha_H,
                          knife_pos, top_axis, bbox_min, bbox_max,
                          incl_center, incl_radius, surface_mask,
                          t, T_total, mesh_name, tau_iou_L, tau_iou_H, out_path):
    """6-panel 3D perspective.
    Row 1: GT | Liang | Hier — surface particles, alive=light gray small, dead=red big.
    Row 2: GT | Liang-vs-GT diff | Hier-vs-GT diff — cavity classification.
    Only SURFACE particles rendered (= what D405 camera would see).
    """
    fig = plt.figure(figsize=(20, 12))
    azim = 30.0 + (t / max(T_total - 1, 1)) * 60.0
    elev = 22.0
    size_box = (bbox_max - bbox_min).max()
    cx = 0.5 * (bbox_min + bbox_max)

    def setup_3d_axes(ax):
        ax.set_xlim(cx[0] - 0.55 * size_box, cx[0] + 0.55 * size_box)
        ax.set_ylim(cx[1] - 0.55 * size_box, cx[1] + 0.55 * size_box)
        ax.set_zlim(cx[2] - 0.55 * size_box, cx[2] + 0.55 * size_box)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    def draw_knife_and_incl(ax):
        ax.scatter([knife_pos[0]], [knife_pos[1]], [knife_pos[2]], marker="X",
                    s=240, color="black", edgecolor="white", linewidth=1.6, zorder=10)
        u = np.linspace(0, 2 * np.pi, 18)
        v = np.linspace(0, np.pi, 9)
        sx = incl_center[0] + incl_radius * np.outer(np.cos(u), np.sin(v))
        sy = incl_center[1] + incl_radius * np.outer(np.sin(u), np.sin(v))
        sz = incl_center[2] + incl_radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(sx, sy, sz, color="green", linewidth=0.6, alpha=0.5)

    # Row 1: surface particles colored by alpha (saturated red for dead, light gray for alive)
    row1 = [
        (pos_T, alpha_T, "GT (truth)"),
        (pos_L, alpha_L, f"Liang  τ_IoU={tau_iou_L:.3f}"),
        (pos_H, alpha_H, f"Hier+Liang  τ_IoU={tau_iou_H:.3f}"),
    ]
    for j, (pos, alpha, label) in enumerate(row1):
        ax = fig.add_subplot(2, 3, j + 1, projection="3d")
        sm = surface_mask
        a = alpha[sm]
        p = pos[sm]
        alive = a > 0.5
        dead = ~alive
        # Alive — small light gray, low alpha
        ax.scatter(p[alive, 0], p[alive, 1], p[alive, 2],
                    c="lightgray", s=8, edgecolors="none", alpha=0.25, depthshade=True)
        # Dead — large red, saturated
        ax.scatter(p[dead, 0], p[dead, 1], p[dead, 2],
                    c="red", s=35, edgecolors="darkred", linewidth=0.4,
                    alpha=0.95, depthshade=True)
        draw_knife_and_incl(ax)
        setup_3d_axes(ax)
        cav = float(dead.mean())
        n_dead = int(dead.sum())
        ax.set_title(f"{label}  ·  surface cav={cav*100:.1f}%  ({n_dead} dead)", fontsize=10)

    # Row 2: GT same | Liang-vs-GT diff | Hier-vs-GT diff
    truth_dead = alpha_T[surface_mask] < 0.5
    p_surf = pos_T[surface_mask]  # use GT positions as common reference
    # GT panel again on left
    ax = fig.add_subplot(2, 3, 4, projection="3d")
    ax.scatter(p_surf[~truth_dead, 0], p_surf[~truth_dead, 1], p_surf[~truth_dead, 2],
                c="lightgray", s=8, edgecolors="none", alpha=0.25)
    ax.scatter(p_surf[truth_dead, 0], p_surf[truth_dead, 1], p_surf[truth_dead, 2],
                c="red", s=35, edgecolors="darkred", linewidth=0.4, alpha=0.95)
    draw_knife_and_incl(ax)
    setup_3d_axes(ax)
    ax.set_title(f"GT cavity (reference) — {int(truth_dead.sum())} dead surface", fontsize=10)

    # Liang & Hier diff
    diffs = [(alpha_L, "Liang"), (alpha_H, "Hier+Liang")]
    for j, (alpha_pred, label) in enumerate(diffs):
        ax = fig.add_subplot(2, 3, 5 + j, projection="3d")
        pred_dead = alpha_pred[surface_mask] < 0.5
        tp = truth_dead & pred_dead
        fn = truth_dead & ~pred_dead   # truth has cavity, pred missed
        fp = ~truth_dead & pred_dead   # pred has cavity, truth doesn't
        tn = ~truth_dead & ~pred_dead
        ax.scatter(p_surf[tn, 0], p_surf[tn, 1], p_surf[tn, 2],
                    c="lightgray", s=6, edgecolors="none", alpha=0.20)
        ax.scatter(p_surf[tp, 0], p_surf[tp, 1], p_surf[tp, 2],
                    c="green", s=28, edgecolors="none", alpha=0.9)
        ax.scatter(p_surf[fn, 0], p_surf[fn, 1], p_surf[fn, 2],
                    c="red", s=35, edgecolors="darkred", linewidth=0.4, alpha=0.95)
        ax.scatter(p_surf[fp, 0], p_surf[fp, 1], p_surf[fp, 2],
                    c="orange", s=35, edgecolors="darkorange", linewidth=0.4, alpha=0.95)
        draw_knife_and_incl(ax)
        setup_3d_axes(ax)
        union = (tp | fn | fp).sum()
        iou = 0.0 if union == 0 else tp.sum() / union
        ax.set_title(f"{label} vs GT  ·  cav IoU={iou:.3f}\n"
                       f"TP={int(tp.sum())} (green) FN={int(fn.sum())} (red) FP={int(fp.sum())} (orange)",
                       fontsize=9)

    fig.suptitle(f"{mesh_name}  ·  frame {t+1}/{T_total}  ·  Surface-only view (D405-style) — TAU recovery comparison",
                  fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=85, bbox_inches="tight")
    plt.close(fig)


def param_iou_arr(pred, truth, threshold, direction):
    if direction == ">":
        pm = pred > threshold; tm = truth > threshold
    else:
        pm = pred < threshold; tm = truth < threshold
    inter = int((pm & tm).sum()); union = int((pm | tm).sum())
    return 1.0 if union == 0 else inter / union


def run_one_mesh(cfg, device, T_calib=10, T_pred=30, N_epochs=15):
    print(f"\n{'='*80}\nMESH: {cfg['name']}\n{'='*80}", flush=True)
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.95
    solver.num_iterations = 2
    solver.gravity = wp.vec3(0.0, 0.0, 0.0)
    pos_rest = mesh_info["vertices"]
    solver.floor_y = float(pos_rest[:, 2].min() - 0.1 * (pos_rest[:, 2].max() - pos_rest[:, 2].min()))

    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size
    incl_radius = MEDIUM_INCL[cfg["name"]]
    top_axis = cfg["top_axis"]

    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=cfg["stiff_in"], stiff_out=cfg["stiff_out"],
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    n_incl = int(inclusion_mask.sum())
    print(f"  incl r={incl_radius}  n_incl={n_incl}/{N} ({n_incl/N*100:.2f}%)", flush=True)

    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx_np = np.where(surface_mask)[0]

    theta_truth_np = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    traj_calib = make_traj_from_spec(cfg["traj_specs"]["A"], bbox_min, bbox_max, T_calib)
    # Long held-out T_B at 80% bbox sweep for richer cavity
    spec_B = dict(cfg["traj_specs"]["B"])
    spec_B["sweep_range"] = (0.10, 0.90)
    traj_B = make_traj_from_spec(spec_B, bbox_min, bbox_max, T_pred)

    solver.reset_to_rest(pos_rest)
    obs_pos_calib, obs_alpha_calib = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_truth, traj_calib, knife_radius, device,
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

    print(f"  Calibrate Liang ...", flush=True)
    t0 = time.time()
    stiff_L, tau_L, _ = calibrate("Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                      lambda_smooth=0.0, lambda_E=1e-5)
    print(f"    done {time.time()-t0:.0f}s", flush=True)
    print(f"  Calibrate Hier ...", flush=True)
    t0 = time.time()
    stiff_H, tau_H, _ = calibrate("Hier+Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                      lambda_smooth=0.0, lambda_E=1e-5)
    print(f"    done {time.time()-t0:.0f}s", flush=True)

    tau_iou_L = param_iou_arr(tau_L, tau_truth, threshold=0.3, direction="<")
    tau_iou_H = param_iou_arr(tau_H, tau_truth, threshold=0.3, direction="<")
    print(f"  tau_iou Liang={tau_iou_L:.3f}  Hier={tau_iou_H:.3f}  ratio={tau_iou_H/max(tau_iou_L,1e-6):.1f}x", flush=True)

    print(f"  Forward T_B ({T_pred} frames) for GT/Liang/Hier ...", flush=True)
    solver.reset_to_rest(pos_rest)
    truth_pos, truth_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_truth, traj_B, knife_radius, device,
        tau_truth_np=tau_truth)
    pred_L_pos, pred_L_alpha = predict_on_traj(stiff_L, tau_L, traj_B, ctx)
    pred_H_pos, pred_H_alpha = predict_on_traj(stiff_H, tau_H, traj_B, ctx)

    print(f"  Render {T_pred} frames ...", flush=True)
    out_dir = REPO / "tools" / "pointnetpp" / f"viz_tau_v2_{cfg['name']}"
    out_dir.mkdir(exist_ok=True)
    for t in range(T_pred):
        render_3panel_frame(
            truth_pos[t], truth_alpha[t],
            pred_L_pos[t], pred_L_alpha[t],
            pred_H_pos[t], pred_H_alpha[t],
            traj_B[t], top_axis, bbox_min, bbox_max,
            incl_center, incl_radius, surface_mask,
            t, T_pred, cfg["name"], tau_iou_L, tau_iou_H,
            out_dir / f"frame_{t:02d}.png")
    frames = [Image.open(out_dir / f"frame_{t:02d}.png") for t in range(T_pred)]
    gif_path = REPO / "tools" / "pointnetpp" / f"viz_tau_recovery_v2_{cfg['name']}.gif"
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                      duration=200, loop=0)
    print(f"  saved {gif_path}", flush=True)


def main():
    device = "cuda:0"
    t0 = time.time()
    for cfg in MESH_CONFIGS:
        try:
            run_one_mesh(cfg, device, T_calib=10, T_pred=30, N_epochs=15)
        except Exception as e:
            import traceback
            print(f"  {cfg['name']} FAILED: {e}")
            traceback.print_exc()
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
