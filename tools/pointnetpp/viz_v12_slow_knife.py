"""v12: slow knife + long horizon, verify deformation visibility.

Uses v11 cached tau (Liang + Hier) — no re-calibration needed.
Only re-runs FORWARD with:
   - T_held = 300 (10× longer than v11)
   - knife traverses 0.10 → 0.90 over 300 steps (vs 30 in v11)
   - effective knife speed ≈ 160 mm/s (10× slower than v11's 1600 mm/s,
       still faster than real ~30 mm/s but visible)
Total sim time = 300 × 5ms = 1.5 sec (vs v11's 0.15 sec).
"""
import sys
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
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec  # noqa: E402
from tools.pointnetpp.eval_hier_phantom_realistic import predict_on_traj  # noqa: E402
from tools.pointnetpp.viz_v11_cube_on_ground import render_4panel, build_lr_tau  # noqa: E402

TAU_LEFT = 0.15; TAU_RIGHT = 0.55
TAU_INIT = 0.30
STIFF_UNIFORM = 0.5
T_HELD = 300


def main():
    cfg = MESH_CONFIGS[0]
    device = "cuda:0"

    solver, rest_state, mesh_info = setup_solver_with_rest(
        str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
    N = solver.N
    solver.damping = 0.95
    solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    solver.gravity = wp.vec3(0.0, -9.81, 0.0)
    solver.floor_y = float(pos_rest[:, 2].min() - 1.0)
    y_floor = float(pos_rest[:, 1].min())

    knife_radius = cfg["knife_radius_abs"]
    # Warm-up settle
    print("Warm-up settle ...", flush=True)
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
            _, pos_w, _ = WarpFullForwardOp.apply(
                theta_warm_t, solver, state_warm, far_knife, knife_radius,
                stiff_warm_t, tau_warm_t)
            state_warm = solver_save_state(solver)
    pos_settled = pos_w.cpu().numpy().copy()
    pos_rest = pos_settled
    mesh_info["vertices"] = pos_settled
    rest_state = state_warm
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0)
    top_axis = cfg["top_axis"]

    tau_truth, lr_mask, x_mid = build_lr_tau(pos_rest, TAU_LEFT, TAU_RIGHT)
    stiff_uniform = np.full(N, STIFF_UNIFORM, dtype=np.float32)

    # Use SAME held-out spec as v11 (different from calib) BUT longer T
    spec_held = dict(cfg["traj_specs"]["A"])
    spec_held["sweep_range"] = (0.10, 0.90)
    fixed_held = dict(spec_held["fixed"])
    other_axes_c = [i for i in range(3) if i != spec_held["sweep_axis"]]
    for ax_idx in other_axes_c:
        if ax_idx != top_axis:
            fixed_held[ax_idx] = max(0.20, min(0.80, fixed_held[ax_idx] - 0.25))
            break
    spec_held["fixed"] = fixed_held
    traj_held = make_traj_from_spec(spec_held, bbox_min, bbox_max, T_HELD)
    # diagnostic
    bbox_size = bbox_max - bbox_min
    sweep_axis = spec_held["sweep_axis"]
    total_dist = (spec_held["sweep_range"][1] - spec_held["sweep_range"][0]) * bbox_size[sweep_axis] * 1000
    sim_sec = T_HELD * solver.dt
    knife_speed_mm_s = total_dist / sim_sec
    print(f"  knife traverses {total_dist:.1f}mm over {sim_sec:.2f}s = {knife_speed_mm_s:.1f} mm/s", flush=True)

    theta_truth_np = np.array([3.0, 20.0, 0.3], dtype=np.float32)

    # Load cached tau
    cache_path = REPO / "tools" / "pointnetpp" / "phase2_v11_cube_cache.npz"
    d = np.load(str(cache_path))
    tau_L = d["tau_L"]; tau_H = d["tau_H"]
    print(f"loaded cache: tau_L L={tau_L[lr_mask].mean():.3f} R={tau_L[~lr_mask].mean():.3f}", flush=True)
    print(f"               tau_H L={tau_H[lr_mask].mean():.3f} R={tau_H[~lr_mask].mean():.3f}", flush=True)

    print("Gen fake-real obs (T=300, slow knife) ...", flush=True)
    solver.reset_to_rest(pos_rest)
    obs_pos_held, obs_alpha_held = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_uniform, traj_held,
        knife_radius, device, tau_truth_np=tau_truth)

    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx = torch.tensor(np.where(surface_mask)[0], dtype=torch.long, device=device)
    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edges_a_t = torch.tensor(edges_a_np, dtype=torch.long, device=device)
    edges_b_t = torch.tensor(edges_b_np, dtype=torch.long, device=device)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)
    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=traj_held, knife_radius=knife_radius,
        pos_rest=pos_rest, theta_truth=theta_truth_np, theta_3d_np=theta_truth_np,
        stiff_init_np=np.full(N, STIFF_UNIFORM, dtype=np.float32),
        stiff_truth=stiff_uniform,
        tau_init_np=np.full(N, TAU_INIT, dtype=np.float32),
        tau_bound=(0.05, 0.95), inclusion_mask=lr_mask,
        surface_idx=surface_idx, surface_mask=surface_mask,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
    )

    stiff_init = np.full(N, STIFF_UNIFORM, dtype=np.float32)
    tau_init = np.full(N, TAU_INIT, dtype=np.float32)

    print("Forward sim_init ...", flush=True)
    init_pos, init_alpha = predict_on_traj(stiff_init, tau_init, traj_held, ctx)
    print("Forward sim_Liang ...", flush=True)
    L_pos, L_alpha = predict_on_traj(stiff_init, tau_L, traj_held, ctx)
    print("Forward sim_Hier ...", flush=True)
    H_pos, H_alpha = predict_on_traj(stiff_init, tau_H, traj_held, ctx)

    # Subsample for viz
    rng = np.random.RandomState(0)
    subsample_idx = rng.choice(N, 10000, replace=False)

    # Save cache
    cache12 = REPO / "tools" / "pointnetpp" / "phase2_v12_slow_knife_cache.npz"
    np.savez_compressed(str(cache12),
        init_pos=np.stack(init_pos), init_alpha=np.stack(init_alpha),
        L_pos=np.stack(L_pos), L_alpha=np.stack(L_alpha),
        H_pos=np.stack(H_pos), H_alpha=np.stack(H_alpha),
        obs_pos=np.stack(obs_pos_held), obs_alpha=np.stack(obs_alpha_held))

    # Render: keep every ~10th frame to limit GIF size
    out_dir = REPO / "tools" / "pointnetpp" / "phase2_v12_slow_knife"
    out_dir.mkdir(exist_ok=True)
    keep = list(range(0, T_HELD, 10))  # 30 frames in GIF
    print(f"Render {len(keep)} frames (every 10th of {T_HELD}) ...", flush=True)
    for k_idx, t in enumerate(keep):
        panels = [
            (obs_pos_held[t], obs_alpha_held[t], "fake-real (truth)"),
            (init_pos[t], init_alpha[t], "sim_init"),
            (L_pos[t], L_alpha[t], "v11 Liang τ"),
            (H_pos[t], H_alpha[t], "v11 Hier τ"),
        ]
        render_4panel(panels, traj_held[t], bbox_min, bbox_max, x_mid, y_floor,
                       k_idx, len(keep), f"{cfg['name']}  [slow-knife T=300, ~160 mm/s]",
                       out_dir / f"frame_{k_idx:02d}.png", subsample_idx)
    frames = [Image.open(out_dir / f"frame_{k_idx:02d}.png") for k_idx in range(len(keep))]
    gif_path = REPO / "tools" / "pointnetpp" / "phase2_v12_slow_knife.gif"
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                      duration=250, loop=0)
    print(f"saved {gif_path}", flush=True)

    # Quantitative deformation magnitude vs sim_init for diagnostic
    disp = np.linalg.norm(np.stack(obs_pos_held) - pos_rest, axis=-1) * 1000  # T,N in mm
    print(f"\nDeformation (fake-real vs rest):")
    print(f"  max disp over time: {disp.max(axis=1).max():.2f} mm")
    print(f"  mean disp at t=T/2: {disp[T_HELD // 2].mean():.3f} mm")
    print(f"  mean disp at t=T-1: {disp[-1].mean():.3f} mm")
    cav = (np.stack(obs_alpha_held) < 0.5).mean(axis=1)
    print(f"  cavity fraction final: {cav[-1] * 100:.2f}%")


if __name__ == "__main__":
    main()
