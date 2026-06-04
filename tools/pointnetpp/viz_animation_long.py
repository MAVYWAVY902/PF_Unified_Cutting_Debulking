"""Long-horizon visual comparison — T=30 (3x calibration horizon).

Tests if hier vs Liang differences become visually apparent at longer horizon.
Calibration done on T=10 (original), then forward prediction extended to T=30
(knife sweeps 3x further distance).

This is the FAIR visual test:
   - Longer cut accumulation → cavity 5-10% (vs 1% at T=10)
   - 3 views per frame (top-down surgical + xz slice + per-frame stats)
   - 30 frames @ 200ms = 6 sec dynamic GIF
   - Calibration extrapolated to UNSEEN knife positions

If hier vs Liang differences visible at T=30 → hier IS contribution
If still indistinguishable → hier confirmed not contribution
"""

import sys
import time
import math
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))

import warp as wp  # noqa: E402
wp.init()
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402

from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff,
)
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, predict_on_traj  # noqa: E402
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402


def make_extended_traj_x_sweep(bbox_min, bbox_max, T_ext, y_frac=1.0, z_frac=0.5,
                                  x_start_frac=0.10, x_end_frac=0.90):
    """Long x sweep: knife traverses 80% of cube x dim over T_ext frames at top y."""
    xs = np.linspace(bbox_min[0] + x_start_frac * (bbox_max[0] - bbox_min[0]),
                       bbox_min[0] + x_end_frac * (bbox_max[0] - bbox_min[0]),
                       T_ext)
    y = bbox_min[1] + y_frac * (bbox_max[1] - bbox_min[1])
    z = bbox_min[2] + z_frac * (bbox_max[2] - bbox_min[2])
    return [np.array([xs[t], y, z], dtype=np.float32) for t in range(T_ext)]


def render_2view_frame(positions_list, alpha_list, labels, knife_pos,
                       bbox_min, bbox_max, incl_center, incl_radius,
                       frame_t, T_total, cavity_per_frame, out_path):
    """Render 2x3 grid per frame: top row top-down view, bottom row xz slice."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Top-down view (xz plane, top y layer)
    top_y_threshold = bbox_max[1] - 0.10 * (bbox_max[1] - bbox_min[1])
    for j, (pos, alpha, label) in enumerate(zip(positions_list, alpha_list, labels)):
        ax = axes[0, j]
        mask = pos[:, 1] > top_y_threshold
        sc = ax.scatter(pos[mask, 0], pos[mask, 2],
                        c=alpha[mask], cmap="RdYlBu", s=15, vmin=0, vmax=1,
                        edgecolors="none")
        ax.add_patch(Circle((incl_center[0], incl_center[2]), incl_radius,
                              fill=False, color="green", linewidth=2.5,
                              linestyle="--"))
        ax.scatter([knife_pos[0]], [knife_pos[2]], marker="X", s=250,
                   color="black", zorder=5, edgecolor="white", linewidth=1.5)
        ax.set_xlim(bbox_min[0] - 0.02, bbox_max[0] + 0.02)
        ax.set_ylim(bbox_min[2] - 0.02, bbox_max[2] + 0.02)
        ax.set_xlabel("x"); ax.set_ylabel("z")
        ax.set_title(f"{label} (TOP-DOWN view, y>0.27)", fontsize=10)
        ax.set_aspect("equal")

    # xz slice at inclusion y
    slice_y = float(incl_center[1])
    slice_tol = 0.025
    for j, (pos, alpha, label) in enumerate(zip(positions_list, alpha_list, labels)):
        ax = axes[1, j]
        mask = np.abs(pos[:, 1] - slice_y) < slice_tol
        sc = ax.scatter(pos[mask, 0], pos[mask, 2],
                        c=alpha[mask], cmap="RdYlBu", s=15, vmin=0, vmax=1,
                        edgecolors="none")
        ax.add_patch(Circle((incl_center[0], incl_center[2]), incl_radius,
                              fill=False, color="green", linewidth=2.5,
                              linestyle="--"))
        ax.scatter([knife_pos[0]], [knife_pos[2]], marker="X", s=250,
                   color="black", zorder=5, edgecolor="white", linewidth=1.5)
        ax.set_xlim(bbox_min[0] - 0.02, bbox_max[0] + 0.02)
        ax.set_ylim(bbox_min[2] - 0.02, bbox_max[2] + 0.02)
        ax.set_xlabel("x"); ax.set_ylabel("z")
        ax.set_title(f"{label} (xz slice at inclusion y)", fontsize=10)
        ax.set_aspect("equal")

    cav_t = ", ".join([f"{c[frame_t]*100:.1f}%" for c in cavity_per_frame])
    fig.suptitle(f"Frame {frame_t+1}/{T_total}  ·  knife at x={knife_pos[0]:.3f}  ·  "
                  f"cavity (Truth/Liang/Hier) = {cav_t}", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=85, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = MESH_CONFIGS[0]
    device = "cuda:0"
    T_calib = 10
    T_extended = 30
    N_epochs = 60
    lambda_E = 1e-5
    stiff_in_truth = 0.8

    print(f"\n=== Long-horizon visual test ===")
    print(f"  Calibration: T={T_calib}  (original setup)")
    print(f"  Prediction:  T={T_extended}  (3x longer, x sweep 10%-90%)")
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size
    incl_radius = cfg["incl_radius_abs"]
    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=stiff_in_truth, stiff_out=0.3,
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)
    top_axis = cfg["top_axis"]
    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    print(f"  N={N}  inclusion {int(inclusion_mask.sum())}/{N}")

    # === Calibrate on T_calib=10 (same as our standard setup) ===
    theta_truth = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    from tools.pointnetpp.multi_mesh_test import make_traj_from_spec
    traj_calib = make_traj_from_spec(cfg["traj_specs"]["A"], bbox_min, bbox_max, T_calib)
    solver.reset_to_rest(pos_rest)
    obs_calib_pos, obs_calib_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth, stiff_truth, traj_calib, knife_radius, device,
        tau_truth_np=tau_truth)
    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_calib_pos]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_calib_alpha]
    surface_idx = torch.tensor(np.where(surface_mask)[0], dtype=torch.long, device=device)
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64), dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64), dtype=torch.long, device=device)
    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)
    x_rest = torch.from_numpy(pos_rest).to(device, dtype=torch.float32)
    H = build_hierarchy(x_rest, M1=512, M2=64, k_neighbors=3)
    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=traj_calib, knife_radius=knife_radius,
        pos_rest=pos_rest, theta_truth=theta_truth, theta_3d_np=theta_truth,
        stiff_init_np=stiff_init, stiff_truth=stiff_truth,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        inclusion_mask=inclusion_mask,
        surface_idx=surface_idx, surface_mask=surface_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H,
    )

    print(f"\n[Calibrate Liang on T={T_calib}]", flush=True)
    stiff_L, tau_L, _ = calibrate("Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                     lambda_smooth=0.0, lambda_E=lambda_E)
    print(f"  Liang stiff_in_int = {stiff_L[inclusion_mask & ~surface_mask].mean():.3f}")
    print(f"\n[Calibrate Hier+Liang on T={T_calib}]", flush=True)
    stiff_H, tau_H, _ = calibrate("Hier+Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                     lambda_smooth=0.0, lambda_E=lambda_E)
    print(f"  Hier stiff_in_int = {stiff_H[inclusion_mask & ~surface_mask].mean():.3f}")

    # === Build extended trajectory: T=30, sweeping x 10%-90% of bbox at top y ===
    print(f"\n[Build extended T={T_extended} trajectory]")
    traj_long = make_extended_traj_x_sweep(bbox_min, bbox_max, T_extended,
                                             y_frac=1.0, z_frac=0.5,
                                             x_start_frac=0.10, x_end_frac=0.90)
    print(f"  Knife x: {traj_long[0][0]:.3f} → {traj_long[-1][0]:.3f}  (full sweep)")

    # === Generate truth + Liang + Hier predictions on T=30 ===
    print(f"\n[Forward T={T_extended} for truth + Liang + Hier]")
    solver.reset_to_rest(pos_rest)
    truth_pos, truth_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth, stiff_truth, traj_long, knife_radius, device,
        tau_truth_np=tau_truth)
    print(f"  Truth final cavity: {(truth_alpha[-1] < 0.5).mean()*100:.2f}%")
    pred_L_pos, pred_L_alpha = predict_on_traj(stiff_L, tau_L, traj_long, ctx)
    pred_H_pos, pred_H_alpha = predict_on_traj(stiff_H, tau_H, traj_long, ctx)
    print(f"  Liang final cavity: {(pred_L_alpha[-1] < 0.5).mean()*100:.2f}%")
    print(f"  Hier final cavity:  {(pred_H_alpha[-1] < 0.5).mean()*100:.2f}%")

    cavity_truth = [float((a < 0.5).mean()) for a in truth_alpha]
    cavity_L = [float((a < 0.5).mean()) for a in pred_L_alpha]
    cavity_H = [float((a < 0.5).mean()) for a in pred_H_alpha]

    # === Render 30 frames with 2-view layout ===
    print(f"\n[Render {T_extended} frames 2x3 grid]")
    out_dir = REPO / "tools" / "pointnetpp" / "viz_anim_long"
    out_dir.mkdir(exist_ok=True)
    for t in range(T_extended):
        out_path = out_dir / f"frame_{t:02d}.png"
        render_2view_frame(
            [truth_pos[t], pred_L_pos[t], pred_H_pos[t]],
            [truth_alpha[t], pred_L_alpha[t], pred_H_alpha[t]],
            ["Truth", "Liang", "Hier+Liang"],
            knife_pos=traj_long[t],
            bbox_min=bbox_min, bbox_max=bbox_max,
            incl_center=incl_center, incl_radius=incl_radius,
            frame_t=t, T_total=T_extended,
            cavity_per_frame=[cavity_truth, cavity_L, cavity_H],
            out_path=out_path,
        )
        if t % 5 == 0:
            print(f"  rendered frame {t}/{T_extended}", flush=True)

    # === Stitch GIF ===
    print(f"\n[Stitch GIF]")
    from PIL import Image
    frames = [Image.open(out_dir / f"frame_{t:02d}.png") for t in range(T_extended)]
    gif_path = REPO / "tools" / "pointnetpp" / "viz_anim_long.gif"
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                      duration=250, loop=0)
    print(f"  saved {gif_path}")

    # === Cavity curve summary ===
    print(f"\n[Cavity progression]")
    print(f"  Frame    Truth     Liang     Hier      Liang-Truth   Hier-Truth")
    for t in [0, 5, 10, 15, 20, 25, T_extended - 1]:
        print(f"  {t:5d}   {cavity_truth[t]*100:5.2f}%   {cavity_L[t]*100:5.2f}%   "
              f"{cavity_H[t]*100:5.2f}%   {(cavity_L[t]-cavity_truth[t])*100:+10.2f}%  "
              f"{(cavity_H[t]-cavity_truth[t])*100:+10.2f}%")


if __name__ == "__main__":
    main()
