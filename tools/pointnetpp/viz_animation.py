"""Visualize sim animations side-by-side: Truth vs Liang vs Hier.

Generate per-frame visualization showing how each calibrated sim's deformation
+ cut pattern looks compared to truth obs. This is the paradigm-correct test:
if sim videos look like obs video, calibration is good.

Setup:
   M1_cube buried inclusion, truth stiff_in=0.8
   Partial obs (top 10%, simulate D405)
   λE=1e-5, 60 epochs
   Render T_B (held-out cross-traj) — true visual test

Output:
   - T_A animation (training fit sanity)
   - T_B animation (cross-traj prediction visual test)
   - Both as GIF (3 panels side-by-side per frame)
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
from tools.pointnetpp.multi_mesh_test import (  # noqa: E402
    MESH_CONFIGS, make_traj_from_spec, predict_on_traj,
)
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402


def render_frame_3panel(positions_list, alpha_list, labels, knife_pos,
                         pos_rest, bbox_min, bbox_max, incl_center, incl_radius,
                         frame_t, T_total, out_path,
                         slice_axis=1, slice_tol=0.025):
    """Render one frame as 3-panel xz-slice visualization."""
    slice_val = float(incl_center[slice_axis])
    bbox_size = bbox_max - bbox_min
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, pos, alpha, label in zip(axes, positions_list, alpha_list, labels):
        # Particles in slice near inclusion plane
        mask = np.abs(pos[:, slice_axis] - slice_val) < slice_tol
        others = [i for i in [0, 1, 2] if i != slice_axis]
        # Color by alpha (red = cut, blue = intact)
        sc = ax.scatter(pos[mask, others[0]], pos[mask, others[1]],
                        c=alpha[mask], cmap="RdYlBu", s=20, vmin=0, vmax=1,
                        edgecolors="none")
        # Mark inclusion outline
        ax.add_patch(Circle((incl_center[others[0]], incl_center[others[1]]),
                              incl_radius, fill=False, color="green", linewidth=2,
                              linestyle="--", label="true inclusion"))
        # Mark knife position
        ax.scatter([knife_pos[others[0]]], [knife_pos[others[1]]],
                   marker="X", s=200, color="black", label="knife", zorder=5)
        ax.set_xlim(bbox_min[others[0]] - 0.02, bbox_max[others[0]] + 0.02)
        ax.set_ylim(bbox_min[others[1]] - 0.02, bbox_max[others[1]] + 0.02)
        ax.set_xlabel(f"axis {['x','y','z'][others[0]]}")
        ax.set_ylabel(f"axis {['x','y','z'][others[1]]}")
        ax.set_title(label, fontsize=12)
        ax.set_aspect("equal")
        if label == labels[0]:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Frame {frame_t+1}/{T_total}  (color: red=cut, blue=intact, green=truth inclusion)",
                 fontsize=13)
    fig.colorbar(sc, ax=axes, shrink=0.6, label="alpha (1=intact, 0=cut)")
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = MESH_CONFIGS[0]
    device = "cuda:0"
    T = 10
    N_epochs = 60
    lambda_E = 1e-5
    stiff_in_truth = 0.8

    print(f"\n=== Visual animation: truth vs Liang vs Hier ===")
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
    print(f"  N={N}, inclusion {int(inclusion_mask.sum())}, surface {int(surface_mask.sum())}")

    theta_truth = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    truth = {}
    for tr in ["A", "B"]:
        traj = make_traj_from_spec(cfg["traj_specs"][tr], bbox_min, bbox_max, T)
        solver.reset_to_rest(pos_rest)
        obs_pos, obs_alpha = gen_obs_per_particle_stiff(
            solver, rest_state, theta_truth, stiff_truth, traj, knife_radius, device,
            tau_truth_np=tau_truth)
        truth[tr] = dict(traj=traj, pos=obs_pos, alpha=obs_alpha)

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in truth["A"]["pos"]]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in truth["A"]["alpha"]]
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
        knife_traj_A=truth["A"]["traj"], knife_radius=knife_radius,
        pos_rest=pos_rest, theta_truth=theta_truth, theta_3d_np=theta_truth,
        stiff_init_np=stiff_init, stiff_truth=stiff_truth,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        inclusion_mask=inclusion_mask,
        surface_idx=surface_idx, surface_mask=surface_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H,
    )

    # Calibrate both methods
    print(f"\n[Calibrate Liang]")
    stiff_L, tau_L, _ = calibrate("Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                     lambda_smooth=0.0, lambda_E=lambda_E)
    print(f"  Liang stiff_in_int = {stiff_L[inclusion_mask & ~surface_mask].mean():.3f}")
    print(f"\n[Calibrate Hier+Liang]")
    stiff_H, tau_H, _ = calibrate("Hier+Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                     lambda_smooth=0.0, lambda_E=lambda_E)
    print(f"  Hier stiff_in_int = {stiff_H[inclusion_mask & ~surface_mask].mean():.3f}")

    # Forward sim on T_B (held-out) for both
    print(f"\n[Forward T_B with calibrated params]")
    pred_L_pos, pred_L_alpha = predict_on_traj(stiff_L, tau_L, truth["B"]["traj"], ctx)
    pred_H_pos, pred_H_alpha = predict_on_traj(stiff_H, tau_H, truth["B"]["traj"], ctx)

    # Render T_B animation (10 frames × 3 panels = 30 plots → 10 frame images)
    out_dir = REPO / "tools" / "pointnetpp" / "viz_anim_T_B"
    out_dir.mkdir(exist_ok=True)
    print(f"\n[Render T_B animation 10 frames]")
    for t in range(T):
        out_path = out_dir / f"frame_{t:02d}.png"
        render_frame_3panel(
            [truth["B"]["pos"][t], pred_L_pos[t], pred_H_pos[t]],
            [truth["B"]["alpha"][t], pred_L_alpha[t], pred_H_alpha[t]],
            ["Truth (real obs)", "Liang calibrated", "Hier+Liang calibrated"],
            knife_pos=truth["B"]["traj"][t],
            pos_rest=pos_rest, bbox_min=bbox_min, bbox_max=bbox_max,
            incl_center=incl_center, incl_radius=incl_radius,
            frame_t=t, T_total=T, out_path=out_path,
        )
        print(f"  saved frame {t}")

    # Stitch into GIF
    print(f"\n[Stitch into GIF]")
    try:
        from PIL import Image
        frames = [Image.open(out_dir / f"frame_{t:02d}.png") for t in range(T)]
        gif_path = REPO / "tools" / "pointnetpp" / "viz_anim_T_B.gif"
        frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                          duration=500, loop=0)
        print(f"  saved {gif_path}")
    except ImportError:
        print("  PIL not available, GIF skipped; PNG sequence in viz_anim_T_B/")

    # Also render T_A (training fit) — fewer frames for sanity
    print(f"\n[Forward T_A with calibrated params for sanity]")
    pred_LA_pos, pred_LA_alpha = predict_on_traj(stiff_L, tau_L, truth["A"]["traj"], ctx)
    pred_HA_pos, pred_HA_alpha = predict_on_traj(stiff_H, tau_H, truth["A"]["traj"], ctx)
    out_dir_A = REPO / "tools" / "pointnetpp" / "viz_anim_T_A"
    out_dir_A.mkdir(exist_ok=True)
    for t in [0, T // 2, T - 1]:  # only 3 key frames for T_A
        out_path = out_dir_A / f"frame_{t:02d}.png"
        render_frame_3panel(
            [truth["A"]["pos"][t], pred_LA_pos[t], pred_HA_pos[t]],
            [truth["A"]["alpha"][t], pred_LA_alpha[t], pred_HA_alpha[t]],
            ["Truth T_A", "Liang T_A", "Hier T_A"],
            knife_pos=truth["A"]["traj"][t],
            pos_rest=pos_rest, bbox_min=bbox_min, bbox_max=bbox_max,
            incl_center=incl_center, incl_radius=incl_radius,
            frame_t=t, T_total=T, out_path=out_path,
        )

    print(f"\n✓ Done. Visualization in tools/pointnetpp/viz_anim_T_B/ + viz_anim_T_B.gif")


if __name__ == "__main__":
    main()
