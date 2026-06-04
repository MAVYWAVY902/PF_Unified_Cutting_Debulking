"""v11 cube-on-ground experiment + visualization.

Setup (phantom-realistic):
   - Cube placed on ground, bottom 20% Y fixed (anchor to floor)
   - Y-down gravity, 60-frame warm-up settle (mesh equilibrates)
   - Knife sweeps over TOP surface (T_held = 30 frames)
   - 4-panel: fake-real | sim_init | v11 Liang | v11 Hier

Rendering optimized for deformation visibility:
   - ALL particles rendered (not just surface)
   - SIDE-VIEW camera (angle to see Y sag + Z cavity)
   - Alive: Y-depth Blues colormap (cube structure visible)
   - Dead: red saturated (cavity emphasized)
   - Ground reference line at Y_min (where bottom anchored)
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


def render_4panel(panels_data, knife_pos, bbox_min, bbox_max, x_mid, y_floor,
                    t, T, mesh_name, out_path, subsample_idx):
    """4-panel fixed perspective view, Z-axis up (matplotlib default).

    Physical Y (gravity) → mapped to display Z (vertical).
    Physical Z → mapped to display Y (depth).
    """
    fig = plt.figure(figsize=(20, 6))
    azim = -60.0   # fixed
    elev = 18.0    # fixed
    size_box = (bbox_max - bbox_min).max()
    cx = 0.5 * (bbox_min + bbox_max)
    for j, (pos, alpha, label) in enumerate(panels_data):
        ax = fig.add_subplot(1, 4, j + 1, projection="3d")
        p = pos[subsample_idx]
        a = alpha[subsample_idx]
        alive = a > 0.5
        dead = ~alive
        # Alive — colored by physical Y (vertical) → mapped to display Z
        y_vals = p[alive, 1]
        ax.scatter(p[alive, 0], p[alive, 2], p[alive, 1],
                    c=y_vals, cmap="Blues", s=8, edgecolors="none",
                    alpha=0.45, depthshade=True,
                    vmin=bbox_min[1], vmax=bbox_max[1])
        ax.scatter(p[dead, 0], p[dead, 2], p[dead, 1],
                    c="red", s=22, edgecolors="darkred", linewidth=0.3,
                    alpha=1.0, depthshade=True)
        # Ground plane reference at physical Y = y_floor → display Z = y_floor
        xx = np.linspace(bbox_min[0], bbox_max[0], 6)
        zz = np.linspace(bbox_min[2], bbox_max[2], 6)
        XX, ZZ = np.meshgrid(xx, zz)
        YY = np.full_like(XX, y_floor)
        ax.plot_wireframe(XX, ZZ, YY, color="brown", linewidth=0.5, alpha=0.6)
        # Left/right boundary plane at X = x_mid (in physical Y,Z)
        yv = np.linspace(bbox_min[1], bbox_max[1], 4)
        zv = np.linspace(bbox_min[2], bbox_max[2], 4)
        Y, Z = np.meshgrid(yv, zv)
        X_bdy = np.full_like(Y, x_mid)
        ax.plot_wireframe(X_bdy, Z, Y, color="blue", linewidth=0.3, alpha=0.3)
        # Knife
        ax.scatter([knife_pos[0]], [knife_pos[2]], [knife_pos[1]], marker="X",
                    s=180, color="black", edgecolor="white", linewidth=1.5, zorder=10)
        ax.set_xlim(cx[0] - 0.55 * size_box, cx[0] + 0.55 * size_box)
        ax.set_ylim(cx[2] - 0.55 * size_box, cx[2] + 0.55 * size_box)
        ax.set_zlim(bbox_min[1] - 0.05 * size_box, bbox_max[1] + 0.15 * size_box)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x", fontsize=8); ax.set_ylabel("z", fontsize=8); ax.set_zlabel("y (up)", fontsize=8)
        cav = float(dead.mean())
        ax.set_title(f"{label}  cav={cav*100:.1f}%", fontsize=9)
    fig.suptitle(f"{mesh_name}  ·  frame {t+1}/{T}  ·  Cube on ground, gravity ON (Y-axis up), top-knife debulk",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=75, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = MESH_CONFIGS[0]   # M1_cube
    device = "cuda:0"
    T_calib = 15; T_held = 30; N_epochs = 100

    # v10/v11 physics: warm-up + 20% anchor
    solver, rest_state, mesh_info = setup_solver_with_rest(
        str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
    N = solver.N
    solver.damping = 0.95
    solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    # Y-down gravity for cube
    solver.gravity = wp.vec3(0.0, -9.81, 0.0)
    solver.floor_y = float(pos_rest[:, 2].min() - 1.0)  # Z-floor far away (no interference)
    y_floor = float(pos_rest[:, 1].min())   # for viz reference

    # Warm-up settle
    knife_radius = cfg["knife_radius_abs"]
    print(f"Warm-up settle (60 frames, gravity only) ...", flush=True)
    solver.reset_to_rest(pos_rest)
    far_knife = pos_rest.max(0) + 10.0
    stiff_warm = np.full(N, STIFF_UNIFORM, dtype=np.float32)
    tau_warm = np.full(N, TAU_INIT, dtype=np.float32)
    theta_warm = np.array([3.0, 20.0, 0.3], dtype=np.float32)
    theta_warm_t = torch.tensor(theta_warm, dtype=torch.float32, device=device)
    stiff_warm_t = torch.tensor(stiff_warm, dtype=torch.float32, device=device)
    tau_warm_t = torch.tensor(tau_warm, dtype=torch.float32, device=device)
    state_warm = rest_state
    with torch.no_grad():
        for _ in range(60):
            alpha_w, pos_w, dv_w = WarpFullForwardOp.apply(
                theta_warm_t, solver, state_warm, far_knife, knife_radius,
                stiff_warm_t, tau_warm_t)
            state_warm = solver_save_state(solver)
    pos_settled = pos_w.cpu().numpy().copy()
    sag = float(np.linalg.norm(pos_settled - pos_rest, axis=1).max() * 1000)
    sag_mean = float(np.linalg.norm(pos_settled - pos_rest, axis=1).mean() * 1000)
    print(f"  Warm-up done: max sag={sag:.2f}mm  mean sag={sag_mean:.3f}mm", flush=True)
    pos_rest = pos_settled
    mesh_info["vertices"] = pos_settled
    rest_state = state_warm

    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    top_axis = cfg["top_axis"]

    tau_truth, lr_mask, x_mid = build_lr_tau(pos_rest, TAU_LEFT, TAU_RIGHT)
    stiff_uniform = np.full(N, STIFF_UNIFORM, dtype=np.float32)

    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx_np = np.where(surface_mask)[0]

    # Trajectories
    spec_calib = dict(cfg["traj_specs"]["A"])
    spec_calib["sweep_range"] = (0.10, 0.90)
    traj_calib = make_traj_from_spec(spec_calib, bbox_min, bbox_max, T_calib)
    spec_calib2 = dict(cfg["traj_specs"]["A"])
    spec_calib2["sweep_range"] = (0.10, 0.90)
    fixed_calib2 = dict(spec_calib2["fixed"])
    other_axes_c = [i for i in range(3) if i != spec_calib2["sweep_axis"]]
    for ax_idx in other_axes_c:
        if ax_idx != top_axis:
            fixed_calib2[ax_idx] = max(0.20, min(0.80, fixed_calib2[ax_idx] + 0.20))
            break
    spec_calib2["fixed"] = fixed_calib2
    traj_calib2 = make_traj_from_spec(spec_calib2, bbox_min, bbox_max, T_calib)
    spec_held = dict(cfg["traj_specs"]["A"])
    spec_held["sweep_range"] = (0.10, 0.90)
    fixed_held = dict(spec_held["fixed"])
    for ax_idx in other_axes_c:
        if ax_idx != top_axis:
            fixed_held[ax_idx] = max(0.20, min(0.80, fixed_held[ax_idx] - 0.25))
            break
    spec_held["fixed"] = fixed_held
    traj_held = make_traj_from_spec(spec_held, bbox_min, bbox_max, T_held)

    theta_truth_np = np.array([3.0, 20.0, 0.3], dtype=np.float32)

    print("Gen fake-real obs ...", flush=True)
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

    cache_path = REPO / "tools" / "pointnetpp" / "phase2_v11_cube_cache.npz"
    if cache_path.exists():
        print(f"Load cache {cache_path.name}", flush=True)
        d = np.load(str(cache_path))
        init_pos = list(d["init_pos"]); init_alpha = list(d["init_alpha"])
        L_pos = list(d["L_pos"]); L_alpha = list(d["L_alpha"])
        H_pos = list(d["H_pos"]); H_alpha = list(d["H_alpha"])
        obs_pos_held = list(d["obs_pos"]); obs_alpha_held = list(d["obs_alpha"])
        tau_L = d["tau_L"]; tau_H = d["tau_H"]
        print(f"  tau_L: left={tau_L[lr_mask].mean():.3f} right={tau_L[~lr_mask].mean():.3f}", flush=True)
        print(f"  tau_H: left={tau_H[lr_mask].mean():.3f} right={tau_H[~lr_mask].mean():.3f}", flush=True)
    else:
        print("Forward sim_init held-out ...", flush=True)
        init_pos, init_alpha = predict_on_traj(stiff_init, tau_init, traj_held, ctx)
        print("Calibrate Liang ...", flush=True)
        _, tau_L, _ = calibrate("L_pos_partial", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.5,
                                    lambda_smooth=0.0, lambda_E=1e-5, lambda_outer_cav=1.0,
                                    lr_schedule=[(30, 0.2), (60, 0.2)],
                                    return_best=True, multi_traj_obs=multi_traj)
        print(f"  tau_L: left={tau_L[lr_mask].mean():.3f} right={tau_L[~lr_mask].mean():.3f}", flush=True)
        print("Calibrate Hier ...", flush=True)
        _, tau_H, _ = calibrate("Hier+L_pos_partial", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.5,
                                    lambda_smooth=0.0, lambda_E=1e-5, lambda_outer_cav=1.0,
                                    lr_schedule=[(30, 0.2), (60, 0.2)],
                                    return_best=True, multi_traj_obs=multi_traj)
        print(f"  tau_H: left={tau_H[lr_mask].mean():.3f} right={tau_H[~lr_mask].mean():.3f}", flush=True)
        print("Forward sim_Liang ...", flush=True)
        L_pos, L_alpha = predict_on_traj(stiff_init, tau_L, traj_held, ctx)
        print("Forward sim_Hier ...", flush=True)
        H_pos, H_alpha = predict_on_traj(stiff_init, tau_H, traj_held, ctx)
        np.savez_compressed(str(cache_path),
            init_pos=np.stack(init_pos), init_alpha=np.stack(init_alpha),
            L_pos=np.stack(L_pos), L_alpha=np.stack(L_alpha),
            H_pos=np.stack(H_pos), H_alpha=np.stack(H_alpha),
            obs_pos=np.stack(obs_pos_held), obs_alpha=np.stack(obs_alpha_held),
            tau_L=tau_L, tau_H=tau_H)
        print(f"saved cache {cache_path.name}", flush=True)

    # Subsample for rendering
    rng = np.random.RandomState(0)
    subsample_idx = rng.choice(N, 10000, replace=False)

    out_dir = REPO / "tools" / "pointnetpp" / "phase2_v11_cube_on_ground"
    out_dir.mkdir(exist_ok=True)
    print(f"Render {T_held} frames ...", flush=True)
    for t in range(T_held):
        panels = [
            (obs_pos_held[t], obs_alpha_held[t], "fake-real (truth)"),
            (init_pos[t], init_alpha[t], "sim_init"),
            (L_pos[t], L_alpha[t], "v11 Liang"),
            (H_pos[t], H_alpha[t], "v11 Hier"),
        ]
        render_4panel(panels, traj_held[t], bbox_min, bbox_max, x_mid, y_floor,
                       t, T_held, cfg["name"],
                       out_dir / f"frame_{t:02d}.png", subsample_idx)
    frames = [Image.open(out_dir / f"frame_{t:02d}.png") for t in range(T_held)]
    gif_path = REPO / "tools" / "pointnetpp" / "phase2_v11_cube_on_ground.gif"
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                      duration=250, loop=0)
    print(f"saved {gif_path}", flush=True)


if __name__ == "__main__":
    main()
