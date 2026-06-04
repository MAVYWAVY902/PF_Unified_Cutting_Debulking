"""Generate 5 side-by-side comparison GIFs:
   1. fake-real vs sim_init
   2. fake-real vs v5 Liang  (λ_outer_cav = 10)
   3. fake-real vs v5 Hier
   4. fake-real vs v6 Liang  (λ_outer_cav = 0.01)
   5. fake-real vs v6 Hier
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


def render_2panel_3d(p1, a1, lbl1, p2, a2, lbl2,
                       knife_pos, bbox_min, bbox_max, x_mid,
                       t, T, mesh_name, surface_mask, suptitle, out_path):
    fig = plt.figure(figsize=(10, 5))
    azim = 30.0 + (t / max(T - 1, 1)) * 60.0
    elev = 22.0
    size_box = (bbox_max - bbox_min).max()
    cx = 0.5 * (bbox_min + bbox_max)
    for j, (pos, alpha, label) in enumerate([(p1, a1, lbl1), (p2, a2, lbl2)]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        sm = surface_mask
        a = alpha[sm]; p = pos[sm]
        alive = a > 0.5; dead = ~alive
        ax.scatter(p[alive, 0], p[alive, 1], p[alive, 2],
                    c="lightgray", s=6, edgecolors="none", alpha=0.20, depthshade=True)
        ax.scatter(p[dead, 0], p[dead, 1], p[dead, 2],
                    c="red", s=22, edgecolors="darkred", linewidth=0.3,
                    alpha=0.95, depthshade=True)
        yv = np.linspace(bbox_min[1], bbox_max[1], 5)
        zv = np.linspace(bbox_min[2], bbox_max[2], 5)
        Y, Z = np.meshgrid(yv, zv)
        X = np.full_like(Y, x_mid)
        ax.plot_wireframe(X, Y, Z, color="blue", linewidth=0.3, alpha=0.3)
        ax.scatter([knife_pos[0]], [knife_pos[1]], [knife_pos[2]], marker="X",
                    s=140, color="black", edgecolor="white", linewidth=1.2, zorder=10)
        ax.set_xlim(cx[0] - 0.55 * size_box, cx[0] + 0.55 * size_box)
        ax.set_ylim(cx[1] - 0.55 * size_box, cx[1] + 0.55 * size_box)
        ax.set_zlim(cx[2] - 0.55 * size_box, cx[2] + 0.55 * size_box)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        cav = float(dead.mean())
        ax.set_title(f"{label}  cav={cav*100:.1f}%", fontsize=9)
    fig.suptitle(f"{mesh_name}  ·  frame {t+1}/{T}  ·  {suptitle}", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=70, bbox_inches="tight")
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

    # v5 (λ=10) — Liang + Hier
    print("Calibrate v5 Liang (λ=10) ...", flush=True)
    _, tau_v5_L, _ = calibrate("L_pos_partial", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.5,
                                     lambda_smooth=0.0, lambda_E=1e-5, lambda_outer_cav=10.0)
    print(f"  tau_v5_L_left={tau_v5_L[lr_mask].mean():.3f}  tau_v5_L_right={tau_v5_L[~lr_mask].mean():.3f}")
    print("Calibrate v5 Hier (λ=10) ...", flush=True)
    _, tau_v5_H, _ = calibrate("Hier+L_pos_partial", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.5,
                                     lambda_smooth=0.0, lambda_E=1e-5, lambda_outer_cav=10.0)
    print(f"  tau_v5_H_left={tau_v5_H[lr_mask].mean():.3f}  tau_v5_H_right={tau_v5_H[~lr_mask].mean():.3f}")

    # v6 (λ=0.01) — Liang + Hier
    print("Calibrate v6 Liang (λ=0.01) ...", flush=True)
    _, tau_v6_L, _ = calibrate("L_pos_partial", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.5,
                                     lambda_smooth=0.0, lambda_E=1e-5, lambda_outer_cav=0.01)
    print(f"  tau_v6_L_left={tau_v6_L[lr_mask].mean():.3f}  tau_v6_L_right={tau_v6_L[~lr_mask].mean():.3f}")
    print("Calibrate v6 Hier (λ=0.01) ...", flush=True)
    _, tau_v6_H, _ = calibrate("Hier+L_pos_partial", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.5,
                                     lambda_smooth=0.0, lambda_E=1e-5, lambda_outer_cav=0.01)
    print(f"  tau_v6_H_left={tau_v6_H[lr_mask].mean():.3f}  tau_v6_H_right={tau_v6_H[~lr_mask].mean():.3f}")

    # Forward all
    print("Forward v5_L ...", flush=True)
    v5L_pos, v5L_alpha = predict_on_traj(stiff_init, tau_v5_L, traj_held, ctx)
    print("Forward v5_H ...", flush=True)
    v5H_pos, v5H_alpha = predict_on_traj(stiff_init, tau_v5_H, traj_held, ctx)
    print("Forward v6_L ...", flush=True)
    v6L_pos, v6L_alpha = predict_on_traj(stiff_init, tau_v6_L, traj_held, ctx)
    print("Forward v6_H ...", flush=True)
    v6H_pos, v6H_alpha = predict_on_traj(stiff_init, tau_v6_H, traj_held, ctx)

    # Render 5 comparison GIFs
    out_root = REPO / "tools" / "pointnetpp"
    comparisons = [
        (init_pos, init_alpha, "sim_init (τ=0.30)", "v5v6_cube_realVSinit", "fake-real vs sim_init"),
        (v5L_pos, v5L_alpha, "v5 Liang (λ=10)", "v5v6_cube_realVS_v5L", "fake-real vs v5 Liang (no inner GD)"),
        (v5H_pos, v5H_alpha, "v5 Hier (λ=10)", "v5v6_cube_realVS_v5H", "fake-real vs v5 Hier (no inner GD)"),
        (v6L_pos, v6L_alpha, "v6 Liang (λ=0.01)", "v5v6_cube_realVS_v6L", "fake-real vs v6 Liang (no inner GD)"),
        (v6H_pos, v6H_alpha, "v6 Hier (λ=0.01)", "v5v6_cube_realVS_v6H", "fake-real vs v6 Hier (no inner GD)"),
    ]
    for sim_pos, sim_alpha, label, stub, suptitle in comparisons:
        print(f"Rendering {stub} ...", flush=True)
        out_dir = out_root / f"{stub}_frames"
        out_dir.mkdir(exist_ok=True)
        for t in range(T_held):
            render_2panel_3d(
                obs_pos_held[t], obs_alpha_held[t], "fake-real",
                sim_pos[t], sim_alpha[t], label,
                traj_held[t], bbox_min, bbox_max, x_mid,
                t, T_held, cfg["name"], surface_mask, suptitle,
                out_dir / f"frame_{t:02d}.png")
        frames = [Image.open(out_dir / f"frame_{t:02d}.png") for t in range(T_held)]
        gif_path = out_root / f"{stub}.gif"
        frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                          duration=250, loop=0)
        print(f"  saved {gif_path}")


if __name__ == "__main__":
    main()
