"""Re-render Phase 2 v4 cube with DEVIATION HEATMAP — color each surface
particle by its L2 distance from fake-real position.

Cavity-binary rendering can't show sub-pixel 0.1mm improvements. Heatmap
makes sub-pixel deviations VISIBLE via colormap intensity changes.

3-panel: fake-real (reference) | sim_init deviation | sim_Hier deviation.
Same colormap range so calibration improvement = color change.
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
from matplotlib.colors import LogNorm  # noqa: E402
from PIL import Image  # noqa: E402

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff  # noqa: E402
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec  # noqa: E402
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402
from tools.pointnetpp.eval_hier_phantom_realistic import predict_on_traj  # noqa: E402

TAU_LEFT = 0.15; TAU_RIGHT = 0.55
TAU_INIT = 0.30
STIFF_UNIFORM = 0.5


def build_lr_tau(pos_rest, tau_left, tau_right):
    x_mid = 0.5 * (pos_rest[:, 0].min() + pos_rest[:, 0].max())
    left_mask = pos_rest[:, 0] < x_mid
    tau = np.where(left_mask, tau_left, tau_right).astype(np.float32)
    return tau, left_mask, x_mid


def render_4panel_heatmap(truth_pos, init_pos, liang_pos, hier_pos, knife_pos,
                            bbox_min, bbox_max, x_mid,
                            t, T, mesh_name, surface_mask,
                            vmax_mm, out_path):
    """Render 4 panels:
       1. fake-real: reference
       2. sim_init: deviation heatmap
       3. sim_Liang: deviation heatmap
       4. sim_Hier: deviation heatmap
    Same LogNorm vmin/vmax so color directly comparable.
    """
    fig = plt.figure(figsize=(20, 6))
    azim = 30.0 + (t / max(T - 1, 1)) * 60.0
    elev = 22.0
    size_box = (bbox_max - bbox_min).max()
    cx = 0.5 * (bbox_min + bbox_max)
    sm = surface_mask
    p_truth = truth_pos[sm]

    # Panel 1: fake-real reference
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.scatter(p_truth[:, 0], p_truth[:, 1], p_truth[:, 2],
                c="steelblue", s=10, edgecolors="none", alpha=0.6, depthshade=True)
    yv = np.linspace(bbox_min[1], bbox_max[1], 5)
    zv = np.linspace(bbox_min[2], bbox_max[2], 5)
    Y, Z = np.meshgrid(yv, zv)
    X = np.full_like(Y, x_mid)
    ax.plot_wireframe(X, Y, Z, color="blue", linewidth=0.4, alpha=0.4)
    ax.scatter([knife_pos[0]], [knife_pos[1]], [knife_pos[2]], marker="X",
                s=160, color="black", edgecolor="white", linewidth=1.4, zorder=10)
    ax.set_xlim(cx[0] - 0.55 * size_box, cx[0] + 0.55 * size_box)
    ax.set_ylim(cx[1] - 0.55 * size_box, cx[1] + 0.55 * size_box)
    ax.set_zlim(cx[2] - 0.55 * size_box, cx[2] + 0.55 * size_box)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("fake-real (reference)", fontsize=10)

    # Panels 2 + 3 + 4: heatmap of deviation
    for j, (pos, label) in enumerate([(init_pos, "sim_init"),
                                          (liang_pos, "sim_Liang"),
                                          (hier_pos, "sim_Hier")]):
        ax = fig.add_subplot(1, 4, 2 + j, projection="3d")
        p_sim = pos[sm]
        dev = np.linalg.norm(p_sim - p_truth, axis=1) * 1000.0  # mm
        # LOG colormap — compress high-dev range, expand low-dev range
        # so sub-mm differences in moderately-affected particles become visible.
        # vmin=0.05mm (≈ smallest meaningful deviation), vmax=vmax_mm.
        dev_clipped = np.clip(dev, 0.05, None)  # avoid log(0)
        sc = ax.scatter(p_sim[:, 0], p_sim[:, 1], p_sim[:, 2],
                          c=dev_clipped, cmap="hot_r", s=12,
                          norm=LogNorm(vmin=0.05, vmax=vmax_mm),
                          edgecolors="none", alpha=0.85, depthshade=True)
        ax.plot_wireframe(X, Y, Z, color="blue", linewidth=0.4, alpha=0.4)
        ax.scatter([knife_pos[0]], [knife_pos[1]], [knife_pos[2]], marker="X",
                    s=160, color="black", edgecolor="white", linewidth=1.4, zorder=10)
        ax.set_xlim(cx[0] - 0.55 * size_box, cx[0] + 0.55 * size_box)
        ax.set_ylim(cx[1] - 0.55 * size_box, cx[1] + 0.55 * size_box)
        ax.set_zlim(cx[2] - 0.55 * size_box, cx[2] + 0.55 * size_box)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        mean_dev = float(dev.mean())
        max_dev = float(dev.max())
        ax.set_title(f"{label}  mean dev={mean_dev:.3f}mm  max={max_dev:.2f}mm",
                       fontsize=10)
        plt.colorbar(sc, ax=ax, fraction=0.04, label="deviation (mm)")

    fig.suptitle(f"{mesh_name}  ·  frame {t+1}/{T}  ·  Surface deviation heatmap  ·  "
                  f"darker red = larger sim-real gap",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=80, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = MESH_CONFIGS[0]   # M1_cube
    device = "cuda:0"
    T_calib = 15; T_held = 30; N_epochs = 100

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

    tau_truth, lr_mask, x_mid = build_lr_tau(pos_rest, TAU_LEFT, TAU_RIGHT)
    stiff_uniform = np.full(N, STIFF_UNIFORM, dtype=np.float32)

    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx_np = np.where(surface_mask)[0]

    spec_calib = dict(cfg["traj_specs"]["A"])
    spec_calib["sweep_range"] = (0.10, 0.90)
    traj_calib = make_traj_from_spec(spec_calib, bbox_min, bbox_max, T_calib)
    spec_held = dict(cfg["traj_specs"]["A"])
    spec_held["sweep_range"] = (0.10, 0.90)
    fixed_held = dict(spec_held["fixed"])
    other_axes = [i for i in range(3) if i != spec_held["sweep_axis"]]
    for ax_idx in other_axes:
        if ax_idx != top_axis:
            old_frac = fixed_held[ax_idx]
            fixed_held[ax_idx] = max(0.20, min(0.80, old_frac - 0.25))
            break
    spec_held["fixed"] = fixed_held
    traj_held = make_traj_from_spec(spec_held, bbox_min, bbox_max, T_held)

    theta_truth_np = np.array([3.0, 20.0, 0.3], dtype=np.float32)

    solver.reset_to_rest(pos_rest)
    obs_pos_calib, obs_alpha_calib = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_uniform, traj_calib,
        knife_radius, device, tau_truth_np=tau_truth)
    solver.reset_to_rest(pos_rest)
    obs_pos_held, obs_alpha_held = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_uniform, traj_held,
        knife_radius, device, tau_truth_np=tau_truth)

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
    stiff_init = np.full(N, STIFF_UNIFORM, dtype=np.float32)
    tau_init = np.full(N, TAU_INIT, dtype=np.float32)

    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=traj_calib, knife_radius=knife_radius,
        pos_rest=pos_rest, theta_truth=theta_truth_np, theta_3d_np=theta_truth_np,
        stiff_init_np=stiff_init, stiff_truth=stiff_uniform,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        inclusion_mask=lr_mask,
        surface_idx=surface_idx, surface_mask=surface_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H_hier,
    )

    print("Forward sim_init ...", flush=True)
    init_pos, init_alpha = predict_on_traj(stiff_init, tau_init, traj_held, ctx)
    print("Calibrate Liang (lr_tau=0.5, N=100) ...", flush=True)
    _, tau_L, _ = calibrate("Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.5,
                                lambda_smooth=0.0, lambda_E=1e-5)
    print(f"  tau_L_left={tau_L[lr_mask].mean():.3f} (truth 0.15)  "
           f"tau_L_right={tau_L[~lr_mask].mean():.3f} (truth 0.55)", flush=True)
    print("Calibrate Hier (lr_tau=0.5, N=100) ...", flush=True)
    _, tau_H, _ = calibrate("Hier+Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.5,
                                lambda_smooth=0.0, lambda_E=1e-5)
    print(f"  tau_H_left={tau_H[lr_mask].mean():.3f} (truth 0.15)  "
           f"tau_H_right={tau_H[~lr_mask].mean():.3f} (truth 0.55)", flush=True)
    print("Forward sim_Liang ...", flush=True)
    liang_pos, _ = predict_on_traj(stiff_init, tau_L, traj_held, ctx)
    print("Forward sim_Hier ...", flush=True)
    hier_pos, hier_alpha = predict_on_traj(stiff_init, tau_H, traj_held, ctx)

    # Auto vmax: 99th percentile of init deviation over all frames
    all_dev = []
    for t in range(T_held):
        d = np.linalg.norm(init_pos[t][surface_mask] - obs_pos_held[t][surface_mask], axis=1) * 1000
        all_dev.append(d)
    vmax_mm = float(np.percentile(np.concatenate(all_dev), 99))
    print(f"  vmax for colormap = {vmax_mm:.3f} mm (99th percentile of sim_init deviation)")

    out_dir = REPO / "tools" / "pointnetpp" / "phase2_v4_heatmap_4panel_cube"
    out_dir.mkdir(exist_ok=True)
    for t in range(T_held):
        render_4panel_heatmap(
            obs_pos_held[t], init_pos[t], liang_pos[t], hier_pos[t],
            traj_held[t], bbox_min, bbox_max, x_mid,
            t, T_held, cfg["name"], surface_mask, vmax_mm,
            out_dir / f"frame_{t:02d}.png")
    frames = [Image.open(out_dir / f"frame_{t:02d}.png") for t in range(T_held)]
    gif_path = REPO / "tools" / "pointnetpp" / "phase2_v4_cube_heatmap_4panel.gif"
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                      duration=250, loop=0)
    print(f"saved {gif_path}")


if __name__ == "__main__":
    main()
