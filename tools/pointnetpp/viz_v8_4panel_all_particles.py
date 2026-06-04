"""4-panel comparison video, ALL PARTICLES (not just surface).
Panels: fake-real | sim_init | v8 Liang | v8 Hier (held-out trajectory).
v8 calibration: multi-trajectory, lr scheduler, best-checkpoint, λ=1.
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


def render_4panel_all(panels_data, knife_pos, bbox_min, bbox_max, x_mid,
                        t, T, mesh_name, out_path, subsample_idx):
    """4-panel side-by-side, ALL particles (subsampled if too many)."""
    fig = plt.figure(figsize=(20, 5.5))
    azim = 30.0 + (t / max(T - 1, 1)) * 60.0
    elev = 22.0
    size_box = (bbox_max - bbox_min).max()
    cx = 0.5 * (bbox_min + bbox_max)
    for j, (pos, alpha, label) in enumerate(panels_data):
        ax = fig.add_subplot(1, 4, j + 1, projection="3d")
        p = pos[subsample_idx]
        a = alpha[subsample_idx]
        alive = a > 0.5
        dead = ~alive
        # Alive — color by Y-depth so cube structure visible
        # Use coolwarm: low Y (bottom) blue, high Y (top, near knife) red-ish
        y_vals = p[alive, 1]
        ax.scatter(p[alive, 0], p[alive, 1], p[alive, 2],
                    c=y_vals, cmap="Blues", s=6, edgecolors="none",
                    alpha=0.35, depthshade=True, vmin=y_vals.min(), vmax=y_vals.max())
        # Dead — large red, saturated (cavity)
        ax.scatter(p[dead, 0], p[dead, 1], p[dead, 2],
                    c="red", s=20, edgecolors="darkred", linewidth=0.3,
                    alpha=1.0, depthshade=True)
        # Left/right boundary plane
        yv = np.linspace(bbox_min[1], bbox_max[1], 4)
        zv = np.linspace(bbox_min[2], bbox_max[2], 4)
        Y, Z = np.meshgrid(yv, zv)
        X = np.full_like(Y, x_mid)
        ax.plot_wireframe(X, Y, Z, color="blue", linewidth=0.3, alpha=0.3)
        # Knife
        ax.scatter([knife_pos[0]], [knife_pos[1]], [knife_pos[2]], marker="X",
                    s=160, color="black", edgecolor="white", linewidth=1.2, zorder=10)
        ax.set_xlim(cx[0] - 0.55 * size_box, cx[0] + 0.55 * size_box)
        ax.set_ylim(cx[1] - 0.55 * size_box, cx[1] + 0.55 * size_box)
        ax.set_zlim(cx[2] - 0.55 * size_box, cx[2] + 0.55 * size_box)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x", fontsize=8); ax.set_ylabel("y", fontsize=8); ax.set_zlabel("z", fontsize=8)
        cav_all = float(dead.mean())
        ax.set_title(f"{label}  ·  cav={cav_all*100:.1f}%", fontsize=9)
    fig.suptitle(f"{mesh_name}  ·  frame {t+1}/{T}  ·  ALL particles (interior+surface)  ·  v8 multi-traj",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=72, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = MESH_CONFIGS[0]   # M1_cube
    device = "cuda:0"
    T_calib = 15; T_held = 30; N_epochs = 100

    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.95
    solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    # v9: per-mesh gravity. Cube top=Y → Y-down gravity, floor far below in Z
    if cfg["top_axis"] == 1:
        solver.gravity = wp.vec3(0.0, -9.81, 0.0)
        solver.floor_y = float(pos_rest[:, 2].min() - 1.0)
    else:
        solver.gravity = wp.vec3(0.0, 0.0, -9.81)
        solver.floor_y = float(pos_rest[:, 2].min() - 0.1 * (pos_rest[:, 2].max() - pos_rest[:, 2].min()))

    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    top_axis = cfg["top_axis"]

    tau_truth, lr_mask, x_mid = build_lr_tau(pos_rest, TAU_LEFT, TAU_RIGHT)
    stiff_uniform = np.full(N, STIFF_UNIFORM, dtype=np.float32)

    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx_np = np.where(surface_mask)[0]

    # Trajectories: T_calib1 (z=0.5), T_calib2 (z=0.7), T_held (z=0.25)
    spec_calib = dict(cfg["traj_specs"]["A"])
    spec_calib["sweep_range"] = (0.10, 0.90)
    traj_calib = make_traj_from_spec(spec_calib, bbox_min, bbox_max, T_calib)
    spec_calib2 = dict(cfg["traj_specs"]["A"])
    spec_calib2["sweep_range"] = (0.10, 0.90)
    fixed_calib2 = dict(spec_calib2["fixed"])
    other_axes_c = [i for i in range(3) if i != spec_calib2["sweep_axis"]]
    for ax_idx in other_axes_c:
        if ax_idx != top_axis:
            old_frac = fixed_calib2[ax_idx]
            fixed_calib2[ax_idx] = max(0.20, min(0.80, old_frac + 0.20))
            break
    spec_calib2["fixed"] = fixed_calib2
    traj_calib2 = make_traj_from_spec(spec_calib2, bbox_min, bbox_max, T_calib)

    spec_held = dict(cfg["traj_specs"]["A"])
    spec_held["sweep_range"] = (0.10, 0.90)
    fixed_held = dict(spec_held["fixed"])
    for ax_idx in other_axes_c:
        if ax_idx != top_axis:
            old_frac = fixed_held[ax_idx]
            fixed_held[ax_idx] = max(0.20, min(0.80, old_frac - 0.25))
            break
    spec_held["fixed"] = fixed_held
    traj_held = make_traj_from_spec(spec_held, bbox_min, bbox_max, T_held)

    theta_truth_np = np.array([3.0, 20.0, 0.3], dtype=np.float32)

    print("Gen fake-real obs (calib1, calib2, held) ...", flush=True)
    solver.reset_to_rest(pos_rest)
    obs_pos_calib, obs_alpha_calib = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_uniform, traj_calib,
        knife_radius, device, tau_truth_np=tau_truth)
    solver.reset_to_rest(pos_rest)
    obs_pos_calib2, obs_alpha_calib2 = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_uniform, traj_calib2,
        knife_radius, device, tau_truth_np=tau_truth)
    solver.reset_to_rest(pos_rest)
    obs_pos_held, obs_alpha_held = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_uniform, traj_held,
        knife_radius, device, tau_truth_np=tau_truth)

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos_calib]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha_calib]
    obs_pos_C_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos_calib2]
    obs_alpha_C_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha_calib2]
    multi_traj = [(traj_calib, obs_pos_A_t, obs_alpha_A_t),
                  (traj_calib2, obs_pos_C_t, obs_alpha_C_t)]
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

    print("Forward sim_init on held-out ...", flush=True)
    init_pos, init_alpha = predict_on_traj(stiff_init, tau_init, traj_held, ctx)

    # v8 Liang
    print("Calibrate v8 Liang ...", flush=True)
    _, tau_L, _ = calibrate("L_pos_partial", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.5,
                                lambda_smooth=0.0, lambda_E=1e-5, lambda_outer_cav=1.0,
                                lr_schedule=[(30, 0.2), (60, 0.2)],
                                return_best=True, multi_traj_obs=multi_traj)
    print(f"  tau_L: left={tau_L[lr_mask].mean():.3f} right={tau_L[~lr_mask].mean():.3f}", flush=True)
    # v8 Hier
    print("Calibrate v8 Hier ...", flush=True)
    _, tau_H, _ = calibrate("Hier+L_pos_partial", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.5,
                                lambda_smooth=0.0, lambda_E=1e-5, lambda_outer_cav=1.0,
                                lr_schedule=[(30, 0.2), (60, 0.2)],
                                return_best=True, multi_traj_obs=multi_traj)
    print(f"  tau_H: left={tau_H[lr_mask].mean():.3f} right={tau_H[~lr_mask].mean():.3f}", flush=True)

    print("Forward sim_Liang on held-out ...", flush=True)
    L_pos, L_alpha = predict_on_traj(stiff_init, tau_L, traj_held, ctx)
    print("Forward sim_Hier on held-out ...", flush=True)
    H_pos, H_alpha = predict_on_traj(stiff_init, tau_H, traj_held, ctx)

    # Subsample for rendering (10000 of 21745)
    rng = np.random.RandomState(0)
    subsample_idx = rng.choice(N, 10000, replace=False)

    # Render 4-panel
    out_dir = REPO / "tools" / "pointnetpp" / "phase2_v9_all_4panel"
    out_dir.mkdir(exist_ok=True)
    print(f"Render {T_held} frames ...", flush=True)
    for t in range(T_held):
        panels = [
            (obs_pos_held[t], obs_alpha_held[t], "fake-real (truth)"),
            (init_pos[t], init_alpha[t], "sim_init (τ=0.30)"),
            (L_pos[t], L_alpha[t], "v8 Liang"),
            (H_pos[t], H_alpha[t], "v8 Hier"),
        ]
        render_4panel_all(panels, traj_held[t], bbox_min, bbox_max, x_mid,
                            t, T_held, cfg["name"],
                            out_dir / f"frame_{t:02d}.png", subsample_idx)
    frames = [Image.open(out_dir / f"frame_{t:02d}.png") for t in range(T_held)]
    gif_path = REPO / "tools" / "pointnetpp" / "phase2_v9_all_particles_4panel.gif"
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                      duration=250, loop=0)
    print(f"saved {gif_path}", flush=True)


if __name__ == "__main__":
    main()
