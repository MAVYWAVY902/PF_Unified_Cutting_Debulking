"""Phantom-realistic surface-only video viz: Truth | Liang | Hier on T_B.

For each mesh (cube/tumor/torus):
  1. Calibrate Liang_inner + Hier+Liang_inner (surface-masked L_pos+L_cav)
  2. Forward sim on held-out T_B
  3. Render top-down surface view (= what D405 camera would see)
     - 3 columns: Truth | Liang | Hier
     - 1 row per t in T_B
     - Color: alpha (red=dead/cavity, blue=alive)
     - Knife X marker + inclusion green dashed circle
  4. Save GIF per mesh

Sim-real overlap on surface is the criterion. If hier truly produces
sim closer to real (truth) surface obs, the Hier column should match
Truth column more closely than the Liang column.
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
from PIL import Image  # noqa: E402

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff,
)
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec  # noqa: E402
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402


def predict_on_traj(stiff_pp, tau_pp, knife_traj, ctx):
    device = ctx["device"]
    theta_torch = torch.tensor(ctx["theta_3d_np"], dtype=torch.float32, device=device,
                                  requires_grad=False)
    stiff_torch = torch.tensor(stiff_pp, dtype=torch.float32, device=device, requires_grad=False)
    tau_torch = torch.tensor(tau_pp, dtype=torch.float32, device=device, requires_grad=False)
    solver = ctx["solver"]
    rest_state = ctx["rest_state"]
    knife_radius = ctx["knife_radius"]
    pos_rest = ctx["pos_rest"]
    solver.reset_to_rest(pos_rest)
    state_in = rest_state
    pred_pos, pred_alpha = [], []
    with torch.no_grad():
        for t, knife_pos in enumerate(knife_traj):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, knife_pos, knife_radius,
                stiff_torch, tau_torch,
            )
            pred_pos.append(pos_t.cpu().numpy().copy())
            pred_alpha.append(alpha_t.cpu().numpy().copy())
            state_in = solver_save_state(solver)
    return pred_pos, pred_alpha


def render_frame(positions_list, alpha_list, labels, knife_pos,
                  top_axis, incl_center, incl_radius,
                  bbox_min, bbox_max, frame_t, T_total,
                  surface_mask, mesh_name, out_path):
    """3-column top-down surface view (alpha colormap)."""
    # Top-down view axes are the two NON-top axes
    other_axes = [i for i in range(3) if i != top_axis]
    ax_h, ax_v = other_axes[0], other_axes[1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for j, (pos, alpha, label) in enumerate(zip(positions_list, alpha_list, labels)):
        ax = axes[j]
        # Only surface particles (already filtered by mask)
        sm = surface_mask
        sc = ax.scatter(pos[sm, ax_h], pos[sm, ax_v],
                          c=alpha[sm], cmap="RdBu", s=18, vmin=0, vmax=1,
                          edgecolors="none")
        # Inclusion projection circle (project to viewing plane)
        ax.add_patch(Circle((incl_center[ax_h], incl_center[ax_v]), incl_radius,
                              fill=False, color="green", linewidth=2.5, linestyle="--"))
        # Knife
        ax.scatter([knife_pos[ax_h]], [knife_pos[ax_v]], marker="X", s=300,
                    color="black", zorder=5, edgecolor="white", linewidth=2.0)
        # Compute surface cavity %
        cav = float((alpha[sm] < 0.5).mean())
        ax.set_xlim(bbox_min[ax_h] - 0.02 * (bbox_max[ax_h] - bbox_min[ax_h]),
                     bbox_max[ax_h] + 0.02 * (bbox_max[ax_h] - bbox_min[ax_h]))
        ax.set_ylim(bbox_min[ax_v] - 0.02 * (bbox_max[ax_v] - bbox_min[ax_v]),
                     bbox_max[ax_v] + 0.02 * (bbox_max[ax_v] - bbox_min[ax_v]))
        ax.set_xlabel(["x", "y", "z"][ax_h])
        ax.set_ylabel(["x", "y", "z"][ax_v])
        ax.set_title(f"{label}  ·  surface cav={cav*100:.2f}%", fontsize=11)
        ax.set_aspect("equal")
    fig.suptitle(f"{mesh_name}  ·  T_B frame {frame_t+1}/{T_total}  (surface-only view, D405-style)",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=85, bbox_inches="tight")
    plt.close(fig)


def render_overlap_frame(truth_alpha, pred_L_alpha, pred_H_alpha,
                           positions_truth, knife_pos,
                           top_axis, incl_center, incl_radius,
                           bbox_min, bbox_max, frame_t, T_total,
                           surface_mask, mesh_name, out_path):
    """2-column overlap view: Liang-vs-Truth | Hier-vs-Truth.
    Color: green=TP (both predict cavity), red=FN (truth has cavity, pred doesn't),
           orange=FP (pred has cavity, truth doesn't), gray=both alive.
    Uses TRUTH positions for the scatter (only cavity classification differs).
    """
    other_axes = [i for i in range(3) if i != top_axis]
    ax_h, ax_v = other_axes[0], other_axes[1]
    sm = surface_mask

    truth_dead = truth_alpha[sm] < 0.5
    pos = positions_truth[sm]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for j, (pred_alpha, label) in enumerate([(pred_L_alpha, "Liang vs Truth"),
                                                 (pred_H_alpha, "Hier vs Truth")]):
        pred_dead = pred_alpha[sm] < 0.5
        tp = truth_dead & pred_dead
        fn = truth_dead & ~pred_dead
        fp = ~truth_dead & pred_dead
        tn = ~truth_dead & ~pred_dead
        ax = axes[j]
        ax.scatter(pos[tn, ax_h], pos[tn, ax_v], c="lightgray", s=10, edgecolors="none", alpha=0.4, label=f"TN ({tn.sum()})")
        ax.scatter(pos[tp, ax_h], pos[tp, ax_v], c="green",     s=22, edgecolors="none", label=f"TP ({tp.sum()})")
        ax.scatter(pos[fn, ax_h], pos[fn, ax_v], c="red",       s=22, edgecolors="none", label=f"FN ({fn.sum()})")
        ax.scatter(pos[fp, ax_h], pos[fp, ax_v], c="orange",    s=22, edgecolors="none", label=f"FP ({fp.sum()})")
        ax.add_patch(Circle((incl_center[ax_h], incl_center[ax_v]), incl_radius,
                              fill=False, color="green", linewidth=2.0, linestyle="--", alpha=0.6))
        ax.scatter([knife_pos[ax_h]], [knife_pos[ax_v]], marker="X", s=300,
                    color="black", zorder=5, edgecolor="white", linewidth=2.0)
        ax.set_xlim(bbox_min[ax_h] - 0.02 * (bbox_max[ax_h] - bbox_min[ax_h]),
                     bbox_max[ax_h] + 0.02 * (bbox_max[ax_h] - bbox_min[ax_h]))
        ax.set_ylim(bbox_min[ax_v] - 0.02 * (bbox_max[ax_v] - bbox_min[ax_v]),
                     bbox_max[ax_v] + 0.02 * (bbox_max[ax_v] - bbox_min[ax_v]))
        ax.set_xlabel(["x", "y", "z"][ax_h])
        ax.set_ylabel(["x", "y", "z"][ax_v])
        union = (tp | fn | fp).sum()
        iou = 0.0 if union == 0 else tp.sum() / union
        ax.set_title(f"{label}  ·  surface cav IoU = {iou:.3f}", fontsize=11)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"{mesh_name}  ·  T_B frame {frame_t+1}/{T_total}  ·  surface cavity classification",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=85, bbox_inches="tight")
    plt.close(fig)


def run_one_mesh(cfg, device, T_calib=10, T_pred=10, N_epochs=15):
    print(f"\n{'='*78}\nMESH: {cfg['name']}\n{'='*78}", flush=True)
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = 2
    # 2026-05-30 Plan C: gravity=0 to isolate hier verdict from axis-mismatch
    # body-force drift (especially visible on small-bbox tumor).
    solver.gravity = wp.vec3(0.0, 0.0, 0.0)
    pos_rest = mesh_info["vertices"]
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size
    incl_radius = cfg["incl_radius_abs"]
    top_axis = cfg["top_axis"]
    print(f"  N={N}  bbox={bbox_size}  top_axis={['x','y','z'][top_axis]}", flush=True)

    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=cfg["stiff_in"], stiff_out=cfg["stiff_out"],
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    n_incl = int(inclusion_mask.sum())
    if n_incl == 0 or n_incl == N:
        return

    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx_np = np.where(surface_mask)[0]
    print(f"  inclusion {n_incl}/{N} ({n_incl/N*100:.2f}%)  surface {len(surface_idx_np)}/{N}", flush=True)

    theta_truth_np = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    traj_calib = make_traj_from_spec(cfg["traj_specs"]["A"], bbox_min, bbox_max, T_calib)
    # Long held-out trajectory: 80% of bbox swept along non-top axis crossing inclusion
    # Use traj_spec "B" axis but extend sweep range to (0.10, 0.90)
    spec_B = dict(cfg["traj_specs"]["B"])
    spec_B["sweep_range"] = (0.10, 0.90)
    traj_B = make_traj_from_spec(spec_B, bbox_min, bbox_max, T_pred)

    print(f"  Gen truth obs on T_A + T_B ...", flush=True)
    solver.reset_to_rest(pos_rest)
    obs_calib_pos, obs_calib_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_truth, traj_calib, knife_radius, device,
        tau_truth_np=tau_truth)
    solver.reset_to_rest(pos_rest)
    obs_B_pos, obs_B_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_truth, traj_B, knife_radius, device,
        tau_truth_np=tau_truth)

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_calib_pos]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_calib_alpha]
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
    H = build_hierarchy(x_rest, M1=M1, M2=M2, k_neighbors=3)

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
        H=H,
    )

    print(f"  Calibrate Liang ...", flush=True)
    t0 = time.time()
    stiff_L, tau_L, _ = calibrate("Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                      lambda_smooth=0.0, lambda_E=1e-5)
    print(f"    done {time.time()-t0:.0f}s", flush=True)

    print(f"  Calibrate Hier+Liang ...", flush=True)
    t0 = time.time()
    stiff_H, tau_H, _ = calibrate("Hier+Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                      lambda_smooth=0.0, lambda_E=1e-5)
    print(f"    done {time.time()-t0:.0f}s", flush=True)

    print(f"  Predict on T_B ...", flush=True)
    pred_L_pos, pred_L_alpha = predict_on_traj(stiff_L, tau_L, traj_B, ctx)
    pred_H_pos, pred_H_alpha = predict_on_traj(stiff_H, tau_H, traj_B, ctx)

    out_dir = REPO / "tools" / "pointnetpp" / f"viz_phantom_g0_{cfg['name']}"
    out_dir.mkdir(exist_ok=True)
    print(f"  Render {T_pred} frames ...", flush=True)
    for t in range(T_pred):
        render_frame(
            [obs_B_pos[t], pred_L_pos[t], pred_H_pos[t]],
            [obs_B_alpha[t], pred_L_alpha[t], pred_H_alpha[t]],
            ["Truth (real obs)", "Liang", "Hier"],
            traj_B[t], top_axis, incl_center, incl_radius, bbox_min, bbox_max,
            t, T_pred, surface_mask, cfg["name"], out_dir / f"alpha_{t:02d}.png")
        render_overlap_frame(
            obs_B_alpha[t], pred_L_alpha[t], pred_H_alpha[t],
            obs_B_pos[t], traj_B[t], top_axis, incl_center, incl_radius,
            bbox_min, bbox_max, t, T_pred, surface_mask, cfg["name"],
            out_dir / f"overlap_{t:02d}.png")
        if t % 2 == 0:
            print(f"    frame {t}", flush=True)

    # Build GIFs
    for prefix, gif_name in [("alpha", f"viz_phantom_g0_{cfg['name']}_alpha.gif"),
                                ("overlap", f"viz_phantom_g0_{cfg['name']}_overlap.gif")]:
        frames = [Image.open(out_dir / f"{prefix}_{t:02d}.png") for t in range(T_pred)]
        gif_path = REPO / "tools" / "pointnetpp" / gif_name
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
            print(f"  ⚠ {cfg['name']} FAILED: {e}")
            traceback.print_exc()
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
