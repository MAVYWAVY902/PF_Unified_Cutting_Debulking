"""Multi-style visualization — expose hier vs Liang differences via 4 viz approaches.

The 3-panel side-by-side scatter plots failed to show hier vs Liang differences
even though numerical diff is ~1mm. This script tries 4 different viz styles
that may expose subtle particle position shifts:

   Style A: OVERLAY — Truth + Liang + Hier in SAME panel, different colors
              (eye can detect offsets directly)
   Style B: DIFF MAP — color particles by |pred_L - pred_H| magnitude
              (directly highlights divergence regions)
   Style C: CAVITY BOUNDARY — extract cavity outline, overlay 3 contours
              (shows which method's cut shape matches truth best)
   Style D: DISPLACEMENT ARROWS — for sample particles, draw rest→final arrows
              (shows direction + magnitude of motion difference)

Uses cached stiff/pred if available (saves 5 min calibration time).
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
from PIL import Image  # noqa: E402

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402

from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff,
)
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, predict_on_traj  # noqa: E402
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402
from tools.pointnetpp.viz_animation_long import make_extended_traj_x_sweep  # noqa: E402


def style_A_overlay(pos_T, alpha_T, pos_L, alpha_L, pos_H, alpha_H,
                    bbox_min, bbox_max, incl_center, incl_radius,
                    title, out_path, slice_axis=1):
    """Single panel: all 3 methods overlaid with different colors.
    Show alpha < 0.5 (cut) particles in 3 colors; intact particles in gray."""
    slice_y = float(incl_center[slice_axis])
    slice_tol = 0.025
    others = [i for i in [0, 1, 2] if i != slice_axis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Top-down view (y > 0.27)
    top_y = bbox_max[1] - 0.10 * (bbox_max[1] - bbox_min[1])
    mask_top = pos_T[:, 1] > top_y

    # xz slice
    mask_slice = np.abs(pos_T[:, slice_axis] - slice_y) < slice_tol

    for ax, mask, view in [(axes[0], mask_top, "Top-down (surgical)"),
                             (axes[1], mask_slice, "xz slice (inclusion plane)")]:
        # All intact particles in light gray as reference
        ax.scatter(pos_T[mask, others[0]], pos_T[mask, others[1]],
                   c="lightgray", s=10, alpha=0.3, label="Truth (intact)")
        # Cut particles per method (alpha < 0.5)
        cut_T = mask & (alpha_T < 0.5)
        cut_L = mask & (alpha_L < 0.5)
        cut_H = mask & (alpha_H < 0.5)
        ax.scatter(pos_T[cut_T, others[0]], pos_T[cut_T, others[1]],
                   c="black", s=40, marker="o", alpha=0.7, edgecolors="none",
                   label=f"Truth cut ({cut_T.sum()})")
        ax.scatter(pos_L[cut_L, others[0]], pos_L[cut_L, others[1]],
                   c="blue", s=25, marker="^", alpha=0.7, edgecolors="none",
                   label=f"Liang cut ({cut_L.sum()})")
        ax.scatter(pos_H[cut_H, others[0]], pos_H[cut_H, others[1]],
                   c="red", s=12, marker="s", alpha=0.7, edgecolors="none",
                   label=f"Hier cut ({cut_H.sum()})")
        ax.add_patch(Circle((incl_center[others[0]], incl_center[others[1]]),
                              incl_radius, fill=False, color="green", linewidth=2,
                              linestyle="--", label="inclusion"))
        ax.set_xlim(bbox_min[others[0]] - 0.02, bbox_max[others[0]] + 0.02)
        ax.set_ylim(bbox_min[others[1]] - 0.02, bbox_max[others[1]] + 0.02)
        ax.set_xlabel(f"axis {others[0]}"); ax.set_ylabel(f"axis {others[1]}")
        ax.set_title(view, fontsize=11)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"STYLE A: 3-method OVERLAY at final frame\n{title}", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


def style_B_diffmap(pos_T, pos_L, pos_H, pos_rest,
                    bbox_min, bbox_max, incl_center, incl_radius,
                    title, out_path):
    """2x2 panel: per-particle position diff colored as heatmap.
       (a) |pred_L - pos_truth|  (Liang prediction error)
       (b) |pred_H - pos_truth|  (Hier prediction error)
       (c) |pred_L - pred_H|     (Direct method divergence)
       (d) Δ_err = |pred_H - truth| - |pred_L - truth|  (Hier minus Liang prediction error)
    """
    diff_L_T = np.linalg.norm(pos_L - pos_T, axis=1)
    diff_H_T = np.linalg.norm(pos_H - pos_T, axis=1)
    diff_L_H = np.linalg.norm(pos_L - pos_H, axis=1)
    delta_err = diff_H_T - diff_L_T

    # Top-down view of all particles colored by metric
    top_y = bbox_max[1] - 0.10 * (bbox_max[1] - bbox_min[1])
    mask = pos_rest[:, 1] > top_y

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    panels = [
        (axes[0, 0], diff_L_T, "(a) |Liang - Truth|",   "viridis", 0, diff_L_T.max() if mask.sum() else 1),
        (axes[0, 1], diff_H_T, "(b) |Hier - Truth|",    "viridis", 0, diff_H_T.max() if mask.sum() else 1),
        (axes[1, 0], diff_L_H, "(c) |Liang - Hier|",     "magma",   0, diff_L_H.max() if mask.sum() else 1),
        (axes[1, 1], delta_err, "(d) Δerr = |H-T| - |L-T|\n(blue=Hier better, red=Liang better)",
         "seismic", -np.max(np.abs(delta_err)), np.max(np.abs(delta_err))),
    ]
    for ax, vals, label, cmap, vmin, vmax in panels:
        sc = ax.scatter(pos_rest[mask, 0], pos_rest[mask, 2],
                        c=vals[mask], cmap=cmap, vmin=vmin, vmax=vmax,
                        s=20, edgecolors="none")
        ax.add_patch(Circle((incl_center[0], incl_center[2]), incl_radius,
                              fill=False, color="green", linewidth=2, linestyle="--"))
        ax.set_xlim(bbox_min[0] - 0.02, bbox_max[0] + 0.02)
        ax.set_ylim(bbox_min[2] - 0.02, bbox_max[2] + 0.02)
        ax.set_xlabel("x"); ax.set_ylabel("z")
        ax.set_title(label, fontsize=11)
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.7, label="m")
    fig.suptitle(f"STYLE B: DIFF MAP — per-particle position differences (top-down view)\n{title}",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


def style_C_cavity_boundary(pos_T, alpha_T, pos_L, alpha_L, pos_H, alpha_H,
                            pos_rest, bbox_min, bbox_max, incl_center, incl_radius,
                            title, out_path):
    """Cavity boundary contour overlay — show where cut edges are for 3 methods.
       Compute density of cut particles, draw contour at alpha=0.5 threshold."""
    top_y = bbox_max[1] - 0.10 * (bbox_max[1] - bbox_min[1])
    mask = pos_rest[:, 1] > top_y

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # All intact particles light gray
    ax.scatter(pos_rest[mask, 0], pos_rest[mask, 2], c="lightgray", s=8, alpha=0.3, edgecolors="none")
    # Cut particles per method
    for (alpha, pos, color, label, marker, size) in [
        (alpha_T, pos_T, "black", "Truth cut", "o", 50),
        (alpha_L, pos_L, "tab:blue", "Liang cut", "x", 60),
        (alpha_H, pos_H, "tab:red", "Hier cut", "+", 60),
    ]:
        cut_mask = mask & (alpha < 0.5)
        ax.scatter(pos[cut_mask, 0], pos[cut_mask, 2],
                   c=color, marker=marker, s=size, alpha=0.7, linewidths=1.5,
                   label=f"{label} ({cut_mask.sum()})")
    ax.add_patch(Circle((incl_center[0], incl_center[2]), incl_radius,
                          fill=False, color="green", linewidth=2.5, linestyle="--",
                          label="inclusion"))
    ax.set_xlim(bbox_min[0] - 0.02, bbox_max[0] + 0.02)
    ax.set_ylim(bbox_min[2] - 0.02, bbox_max[2] + 0.02)
    ax.set_xlabel("x"); ax.set_ylabel("z")
    ax.set_title(f"STYLE C: CUT PARTICLE OVERLAY (top-down view, y > 0.27)\n{title}", fontsize=12)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


def style_D_displacement_arrows(pos_T, pos_L, pos_H, pos_rest,
                                bbox_min, bbox_max, incl_center, incl_radius,
                                title, out_path, n_sample=200):
    """Subsample particles, draw displacement arrows from rest to final pos.
       3 colors for 3 methods, see arrow patterns differ."""
    top_y = bbox_max[1] - 0.10 * (bbox_max[1] - bbox_min[1])
    mask = pos_rest[:, 1] > top_y
    idx = np.where(mask)[0]
    if len(idx) > n_sample:
        np.random.seed(0)
        idx = np.random.choice(idx, n_sample, replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, pos, color, label in [
        (axes[0], pos_T, "black", "Truth"),
        (axes[1], pos_L, "tab:blue", "Liang"),
        (axes[2], pos_H, "tab:red", "Hier+Liang"),
    ]:
        ax.scatter(pos_rest[mask, 0], pos_rest[mask, 2], c="lightgray", s=5, alpha=0.2)
        dx = pos[idx, 0] - pos_rest[idx, 0]
        dz = pos[idx, 2] - pos_rest[idx, 2]
        ax.quiver(pos_rest[idx, 0], pos_rest[idx, 2], dx, dz,
                  angles="xy", scale_units="xy", scale=1, color=color, alpha=0.7,
                  width=0.003, headwidth=4)
        ax.add_patch(Circle((incl_center[0], incl_center[2]), incl_radius,
                              fill=False, color="green", linewidth=2, linestyle="--"))
        ax.set_xlim(bbox_min[0] - 0.02, bbox_max[0] + 0.02)
        ax.set_ylim(bbox_min[2] - 0.02, bbox_max[2] + 0.02)
        ax.set_xlabel("x"); ax.set_ylabel("z")
        ax.set_title(f"{label} displacement arrows ({n_sample} samples)", fontsize=11)
        ax.set_aspect("equal")
    fig.suptitle(f"STYLE D: DISPLACEMENT ARROWS (top-down view sample)\n{title}", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = MESH_CONFIGS[0]
    device = "cuda:0"
    T_calib = 10
    T_extended = 30
    N_epochs = 60
    lambda_E = 1e-5
    NUM_ITER = 16
    INCL_RADIUS = 0.10  # 11% inclusion

    # Cache file for calibrated stiff
    cache_path = REPO / "tools" / "pointnetpp" / "viz_multi_style_cache.npz"

    print(f"\n=== Multi-style viz test (inclusion ~11%, num_iter={NUM_ITER}) ===")
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = NUM_ITER
    pos_rest = mesh_info["vertices"]
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size

    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=0.8, stiff_out=0.3,
        center_override=tuple(incl_center), radius_override=INCL_RADIUS)
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)
    top_axis = cfg["top_axis"]
    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    print(f"  N={N}, inclusion {int(inclusion_mask.sum())}/{N}")

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

    # Cache stiff fields
    if cache_path.exists():
        print(f"\n[Loading cached stiff from {cache_path}]")
        d = np.load(cache_path)
        stiff_L = d["stiff_L"]
        tau_L = d["tau_L"]
        stiff_H = d["stiff_H"]
        tau_H = d["tau_H"]
    else:
        print(f"\n[Calibrate Liang]", flush=True)
        stiff_L, tau_L, _ = calibrate("Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                         lambda_smooth=0.0, lambda_E=lambda_E)
        print(f"\n[Calibrate Hier]", flush=True)
        stiff_H, tau_H, _ = calibrate("Hier+Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                         lambda_smooth=0.0, lambda_E=lambda_E)
        np.savez(cache_path, stiff_L=stiff_L, tau_L=tau_L, stiff_H=stiff_H, tau_H=tau_H)
        print(f"  cached to {cache_path}")
    print(f"  Liang stiff_in_int = {stiff_L[inclusion_mask & ~surface_mask].mean():.3f}")
    print(f"  Hier  stiff_in_int = {stiff_H[inclusion_mask & ~surface_mask].mean():.3f}")

    # Forward T=30
    traj_long = make_extended_traj_x_sweep(bbox_min, bbox_max, T_extended,
                                             y_frac=1.0, z_frac=0.5,
                                             x_start_frac=0.10, x_end_frac=0.90)
    print(f"\n[Forward T={T_extended}]", flush=True)
    solver.reset_to_rest(pos_rest)
    truth_pos, truth_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth, stiff_truth, traj_long, knife_radius, device,
        tau_truth_np=tau_truth)
    pred_L_pos, pred_L_alpha = predict_on_traj(stiff_L, tau_L, traj_long, ctx)
    pred_H_pos, pred_H_alpha = predict_on_traj(stiff_H, tau_H, traj_long, ctx)

    # Final frame
    t = T_extended - 1
    title = f"Frame {t+1}/{T_extended}, inclusion {int(inclusion_mask.sum())/N*100:.1f}% mesh, num_iter={NUM_ITER}"

    print(f"\n[Render multi-style viz at frame {t}]")
    out_dir = REPO / "tools" / "pointnetpp" / "viz_multi_style"
    out_dir.mkdir(exist_ok=True)

    style_A_overlay(truth_pos[t], truth_alpha[t],
                    pred_L_pos[t], pred_L_alpha[t],
                    pred_H_pos[t], pred_H_alpha[t],
                    bbox_min, bbox_max, incl_center, INCL_RADIUS,
                    title, out_dir / "style_A_overlay.png")
    print(f"  saved style A")

    style_B_diffmap(truth_pos[t], pred_L_pos[t], pred_H_pos[t], pos_rest,
                    bbox_min, bbox_max, incl_center, INCL_RADIUS,
                    title, out_dir / "style_B_diffmap.png")
    print(f"  saved style B")

    style_C_cavity_boundary(truth_pos[t], truth_alpha[t],
                            pred_L_pos[t], pred_L_alpha[t],
                            pred_H_pos[t], pred_H_alpha[t],
                            pos_rest, bbox_min, bbox_max, incl_center, INCL_RADIUS,
                            title, out_dir / "style_C_cavity.png")
    print(f"  saved style C")

    style_D_displacement_arrows(truth_pos[t], pred_L_pos[t], pred_H_pos[t], pos_rest,
                                bbox_min, bbox_max, incl_center, INCL_RADIUS,
                                title, out_dir / "style_D_arrows.png")
    print(f"  saved style D")
    print(f"\nAll styles in {out_dir}/")


if __name__ == "__main__":
    main()
