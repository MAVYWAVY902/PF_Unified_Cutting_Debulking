"""Visualize the TAU PARAMETER FIELD itself (not cavity).

The cavity comparison doesn't show tau recovery because knife geometry,
not tau field, determines cavity. Tau IoU measures the per-particle tau
prediction quality — that's a PARAMETER, not an observation.

For each mesh (cube, tumor, torus), render the per-particle tau field as
3-panel 3D scatter: GT | Liang | Hier. Particles colored by tau value:
   - Red = low tau (inclusion-like, tau < 0.3)
   - Blue = high tau (healthy tissue, tau > 0.3)
   - White = threshold

Two views per mesh:
   1. 3D rotating animation showing full field
   2. Cross-section through inclusion center showing interior recovery

If hier truly recovers tau, hier panel should show red blob co-located
with green inclusion sphere wireframe. Liang panel should be mostly uniform.
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
from tools.stageA_inclusion_recovery import build_inclusion_truth, gen_obs_per_particle_stiff  # noqa: E402
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec  # noqa: E402
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402


MEDIUM_INCL = {
    "M1_cube":  0.07,
    "M2_tumor": 0.005,
    "M3_torus": 0.08,
}


def calibrate_for_mesh(cfg, incl_radius, device, T_calib=10, N_epochs=15):
    """Run Liang + Hier calibration, return tau_truth, tau_L, tau_H + mesh info."""
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
    top_axis = cfg["top_axis"]

    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=cfg["stiff_in"], stiff_out=cfg["stiff_out"],
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)

    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx_np = np.where(surface_mask)[0]

    theta_truth_np = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    traj_calib = make_traj_from_spec(cfg["traj_specs"]["A"], bbox_min, bbox_max, T_calib)

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
    _, tau_L, _ = calibrate("Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                lambda_smooth=0.0, lambda_E=1e-5)
    print(f"  Calibrate Hier ...", flush=True)
    _, tau_H, _ = calibrate("Hier+Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                lambda_smooth=0.0, lambda_E=1e-5)
    return dict(
        tau_truth=tau_truth, tau_L=tau_L, tau_H=tau_H,
        pos_rest=pos_rest, inclusion_mask=inclusion_mask,
        bbox_min=bbox_min, bbox_max=bbox_max, top_axis=top_axis,
        incl_center=incl_center, incl_radius=incl_radius,
    )


def render_tau_field_static(data, mesh_name, out_path):
    """Static 6-panel: row 1 = 3D scatter colored by tau; row 2 = cross-section through inclusion center."""
    pos = data["pos_rest"]
    incl_center = data["incl_center"]; incl_r = data["incl_radius"]
    bbox_min = data["bbox_min"]; bbox_max = data["bbox_max"]
    top_axis = data["top_axis"]
    other = [i for i in range(3) if i != top_axis]
    ax_h, ax_v = other[0], other[1]

    taus = [
        (data["tau_truth"], "GT (truth)"),
        (data["tau_L"],     "Liang only"),
        (data["tau_H"],     "Hier+Liang"),
    ]

    fig = plt.figure(figsize=(20, 12))
    # Row 1: 3D scatter, color by tau (inverted RdBu so red=low tau=inclusion)
    for j, (tau, label) in enumerate(taus):
        ax = fig.add_subplot(2, 3, j + 1, projection="3d")
        idx = np.arange(len(pos))
        if len(pos) > 6000:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(pos), 6000, replace=False)
        sc = ax.scatter(pos[idx, 0], pos[idx, 1], pos[idx, 2],
                          c=tau[idx], cmap="RdBu", s=10, vmin=0.15, vmax=0.45,
                          edgecolors="none", alpha=0.6, depthshade=True)
        # Inclusion ground-truth wireframe
        u = np.linspace(0, 2 * np.pi, 18)
        v = np.linspace(0, np.pi, 9)
        sx = incl_center[0] + incl_r * np.outer(np.cos(u), np.sin(v))
        sy = incl_center[1] + incl_r * np.outer(np.sin(u), np.sin(v))
        sz = incl_center[2] + incl_r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(sx, sy, sz, color="green", linewidth=0.8, alpha=0.6)
        size = (bbox_max - bbox_min).max()
        cx = 0.5 * (bbox_min + bbox_max)
        ax.set_xlim(cx[0] - 0.55 * size, cx[0] + 0.55 * size)
        ax.set_ylim(cx[1] - 0.55 * size, cx[1] + 0.55 * size)
        ax.set_zlim(cx[2] - 0.55 * size, cx[2] + 0.55 * size)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=22, azim=35)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        tau_in_int = float(tau[data["inclusion_mask"]].mean())
        tau_out_int = float(tau[~data["inclusion_mask"]].mean())
        ax.set_title(f"{label}\nτ_in={tau_in_int:.3f}  τ_out={tau_out_int:.3f}  (truth: 0.20/0.40)",
                       fontsize=10)
        plt.colorbar(sc, ax=ax, fraction=0.04, label="τ")

    # Row 2: 2D cross-section slice through inclusion center (perpendicular to top_axis)
    slice_axis = top_axis
    slice_val = incl_center[slice_axis]
    slice_tol = incl_r * 1.2  # slab containing inclusion
    mask_slab = np.abs(pos[:, slice_axis] - slice_val) < slice_tol
    for j, (tau, label) in enumerate(taus):
        ax = fig.add_subplot(2, 3, 4 + j)
        sc = ax.scatter(pos[mask_slab, ax_h], pos[mask_slab, ax_v],
                          c=tau[mask_slab], cmap="RdBu", s=18, vmin=0.15, vmax=0.45,
                          edgecolors="none")
        ax.add_patch(Circle((incl_center[ax_h], incl_center[ax_v]), incl_r,
                              fill=False, color="green", linewidth=2.0, linestyle="--"))
        ax.set_xlim(bbox_min[ax_h] - 0.02 * (bbox_max[ax_h] - bbox_min[ax_h]),
                     bbox_max[ax_h] + 0.02 * (bbox_max[ax_h] - bbox_min[ax_h]))
        ax.set_ylim(bbox_min[ax_v] - 0.02 * (bbox_max[ax_v] - bbox_min[ax_v]),
                     bbox_max[ax_v] + 0.02 * (bbox_max[ax_v] - bbox_min[ax_v]))
        ax.set_xlabel(["x", "y", "z"][ax_h])
        ax.set_ylabel(["x", "y", "z"][ax_v])
        ax.set_aspect("equal")
        ax.set_title(f"{label} cross-section at {['x','y','z'][slice_axis]}={slice_val:.3f}",
                       fontsize=10)
        plt.colorbar(sc, ax=ax, fraction=0.04, label="τ")

    fig.suptitle(f"{mesh_name}  ·  TAU PARAMETER FIELD recovered  "
                  f"(red=inclusion-like τ<0.3, blue=healthy τ>0.3)\n"
                  f"Green wireframe/dashed = true inclusion location",
                  fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


def render_tau_field_rotating(data, mesh_name, out_path, T_frames=24):
    """Rotating 3D animation showing tau field from multiple angles."""
    pos = data["pos_rest"]
    incl_center = data["incl_center"]; incl_r = data["incl_radius"]
    bbox_min = data["bbox_min"]; bbox_max = data["bbox_max"]

    taus = [
        (data["tau_truth"], "GT (truth)"),
        (data["tau_L"],     "Liang only"),
        (data["tau_H"],     "Hier+Liang"),
    ]

    out_dir = Path(str(out_path) + "_frames")
    out_dir.mkdir(exist_ok=True)

    for t in range(T_frames):
        azim = (t / T_frames) * 360.0
        fig = plt.figure(figsize=(18, 6.5))
        for j, (tau, label) in enumerate(taus):
            ax = fig.add_subplot(1, 3, j + 1, projection="3d")
            idx = np.arange(len(pos))
            if len(pos) > 6000:
                rng = np.random.RandomState(0)
                idx = rng.choice(len(pos), 6000, replace=False)
            sc = ax.scatter(pos[idx, 0], pos[idx, 1], pos[idx, 2],
                              c=tau[idx], cmap="RdBu", s=10, vmin=0.15, vmax=0.45,
                              edgecolors="none", alpha=0.6, depthshade=True)
            u = np.linspace(0, 2 * np.pi, 18)
            v = np.linspace(0, np.pi, 9)
            sx = incl_center[0] + incl_r * np.outer(np.cos(u), np.sin(v))
            sy = incl_center[1] + incl_r * np.outer(np.sin(u), np.sin(v))
            sz = incl_center[2] + incl_r * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(sx, sy, sz, color="green", linewidth=0.8, alpha=0.6)
            size = (bbox_max - bbox_min).max()
            cx = 0.5 * (bbox_min + bbox_max)
            ax.set_xlim(cx[0] - 0.55 * size, cx[0] + 0.55 * size)
            ax.set_ylim(cx[1] - 0.55 * size, cx[1] + 0.55 * size)
            ax.set_zlim(cx[2] - 0.55 * size, cx[2] + 0.55 * size)
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=22, azim=azim)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            tau_in = float(tau[data["inclusion_mask"]].mean())
            ax.set_title(f"{label}\nτ_in={tau_in:.3f} (truth 0.20)", fontsize=10)
        fig.suptitle(f"{mesh_name} TAU field  ·  rotation {azim:.0f}°  "
                      f"·  GREEN ring = true inclusion location  ·  red dots = inclusion-like τ",
                      fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"frame_{t:02d}.png", dpi=85, bbox_inches="tight")
        plt.close(fig)

    frames = [Image.open(out_dir / f"frame_{t:02d}.png") for t in range(T_frames)]
    frames[0].save(str(out_path), save_all=True, append_images=frames[1:],
                      duration=180, loop=0)


def main():
    device = "cuda:0"
    t0 = time.time()
    for cfg in MESH_CONFIGS:
        print(f"\n{'='*80}\n{cfg['name']}\n{'='*80}", flush=True)
        data = calibrate_for_mesh(cfg, MEDIUM_INCL[cfg["name"]], device,
                                       T_calib=10, N_epochs=15)
        # Static cross-section
        static_path = REPO / "tools" / "pointnetpp" / f"tau_field_static_{cfg['name']}.png"
        render_tau_field_static(data, cfg["name"], static_path)
        print(f"  saved {static_path}", flush=True)
        # Rotating GIF
        rot_path = REPO / "tools" / "pointnetpp" / f"tau_field_rotating_{cfg['name']}.gif"
        render_tau_field_rotating(data, cfg["name"], rot_path, T_frames=24)
        print(f"  saved {rot_path}", flush=True)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
