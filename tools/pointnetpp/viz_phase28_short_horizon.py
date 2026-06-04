"""Liang-style T=10 short-horizon evaluation of Phase 2.8 calibration.

Protocol (matches Liang 2023 Section III-D, Figure 5):
   Every T_la=10 frames, sim state is re-anchored to obs at that frame.
   Between anchors, sim free-rolls 10 steps with calibrated tau.
   This is the "MPC lookahead window" deployment metric.

Reports:
   - Per-window: mean Chamfer / cavity_IoU at the END of each 10-frame rollout
   - Aggregate over T=300 trajectory
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

from PIL import Image  # noqa: E402
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state, solver_load_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec  # noqa: E402
from tools.pointnetpp.viz_v11_cube_on_ground import render_4panel, build_lr_tau  # noqa: E402

TAU_LEFT = 0.15; TAU_RIGHT = 0.55
TAU_INIT = 0.30
STIFF_UNIFORM = 0.5
T_TRAJ = 300
T_LA = 10  # Liang's lookahead window


def short_horizon_predict(stiff_pp, tau_pp, knife_traj, obs_pos_t, obs_alpha_t, ctx, T_la):
    """Re-anchor every T_la frames to obs; free-rollout in between.
    Returns full T-length sequence (pos, alpha) corresponding to sim outputs.
    """
    device = ctx["device"]
    theta_torch = torch.tensor(ctx["theta_3d_np"], dtype=torch.float32, device=device,
                                 requires_grad=False)
    stiff_torch = torch.tensor(stiff_pp, dtype=torch.float32, device=device, requires_grad=False)
    tau_torch = torch.tensor(tau_pp, dtype=torch.float32, device=device, requires_grad=False)
    solver = ctx["solver"]; pos_rest = ctx["pos_rest"]
    knife_radius = ctx["knife_radius"]
    T = len(knife_traj)
    pred_pos, pred_alpha = [], []
    solver.reset_to_rest(pos_rest)
    state_in = ctx["rest_state"]
    with torch.no_grad():
        for t in range(T):
            # Re-anchor every T_la frames: replace sim state.pos with obs.pos
            if t > 0 and t % T_la == 0:
                state_in = solver_save_state(solver)
                state_in["pos"] = obs_pos_t[t - 1].cpu().numpy().astype(np.float32).copy()
                # also reset particle_damage / alpha from obs for sharper window
                state_in["alpha"] = obs_alpha_t[t - 1].cpu().numpy().astype(np.float32).copy()
            alpha_t, pos_t, _ = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, knife_traj[t], knife_radius,
                stiff_torch, tau_torch,
            )
            pred_pos.append(pos_t.cpu().numpy().copy())
            pred_alpha.append(alpha_t.cpu().numpy().copy())
            state_in = solver_save_state(solver)
    return pred_pos, pred_alpha


def main():
    cfg = MESH_CONFIGS[0]
    device = "cuda:0"

    solver, rest_state, mesh_info = setup_solver_with_rest(
        str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
    N = solver.N
    solver.damping = 0.95; solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    solver.gravity = wp.vec3(0.0, -9.81, 0.0)
    solver.floor_y = float(pos_rest[:, 2].min() - 1.0)
    y_floor = float(pos_rest[:, 1].min())
    knife_radius = cfg["knife_radius_abs"]

    print("Warm-up settle ...", flush=True)
    solver.reset_to_rest(pos_rest)
    far_knife = pos_rest.max(0) + 10.0
    stiff_warm_t = torch.tensor(np.full(N, STIFF_UNIFORM, dtype=np.float32),
                                  dtype=torch.float32, device=device)
    tau_warm_t = torch.tensor(np.full(N, TAU_INIT, dtype=np.float32),
                                dtype=torch.float32, device=device)
    theta_warm_t = torch.tensor(np.array([3.0, 20.0, 0.3], dtype=np.float32),
                                  dtype=torch.float32, device=device)
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
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    top_axis = cfg["top_axis"]

    tau_truth, lr_mask, x_mid = build_lr_tau(pos_rest, TAU_LEFT, TAU_RIGHT)
    stiff_uniform = np.full(N, STIFF_UNIFORM, dtype=np.float32)
    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold

    # Calibration traj A
    spec_A = dict(cfg["traj_specs"]["A"])
    spec_A["sweep_range"] = (0.10, 0.90)
    traj = make_traj_from_spec(spec_A, bbox_min, bbox_max, T_TRAJ)

    theta_truth = np.array([3.0, 20.0, 0.3], dtype=np.float32)
    print("Gen fake-real obs ...", flush=True)
    solver.reset_to_rest(pos_rest)
    obs_pos, obs_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth, stiff_uniform, traj,
        knife_radius, device, tau_truth_np=tau_truth)
    obs_pos_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]

    cache = REPO / "tools" / "pointnetpp" / "phase2_online_phase1_cache.npz"
    d = np.load(str(cache))
    tau_L = d["tau_L"]; tau_H = d["tau_H"]

    ctx = dict(solver=solver, rest_state=rest_state, N=N, device=device,
                  knife_radius=knife_radius, pos_rest=pos_rest, theta_3d_np=theta_truth)

    stiff_init = np.full(N, STIFF_UNIFORM, dtype=np.float32)
    tau_init = np.full(N, TAU_INIT, dtype=np.float32)
    print("Short-horizon predict sim_init ...", flush=True)
    init_pos, init_alpha = short_horizon_predict(
        stiff_init, tau_init, traj, obs_pos_t, obs_alpha_t, ctx, T_LA)
    print("Short-horizon predict Liang ...", flush=True)
    L_pos, L_alpha = short_horizon_predict(
        stiff_init, tau_L, traj, obs_pos_t, obs_alpha_t, ctx, T_LA)
    print("Short-horizon predict Hier ...", flush=True)
    H_pos, H_alpha = short_horizon_predict(
        stiff_init, tau_H, traj, obs_pos_t, obs_alpha_t, ctx, T_LA)

    # Metrics
    obs_pos_np = np.stack(obs_pos); obs_alpha_np = np.stack(obs_alpha)
    init_pos_np = np.stack(init_pos); init_alpha_np = np.stack(init_alpha)
    L_pos_np = np.stack(L_pos); L_alpha_np = np.stack(L_alpha)
    H_pos_np = np.stack(H_pos); H_alpha_np = np.stack(H_alpha)
    surf = surface_mask

    # Per-anchor-end metric: at t = T_LA, 2*T_LA, ..., evaluate
    anchor_ends = list(range(T_LA - 1, T_TRAJ, T_LA))
    print(f"\n=== Liang-style short-horizon metric (T_la={T_LA}, {len(anchor_ends)} anchor windows) ===")
    for name, p, a in [("sim_init", init_pos_np, init_alpha_np),
                          ("Liang",    L_pos_np,    L_alpha_np),
                          ("Hier",     H_pos_np,    H_alpha_np)]:
        pos_mm_list = []; iou_list = []
        for t_end in anchor_ends:
            pos_mm = np.linalg.norm(p[t_end, surf] - obs_pos_np[t_end, surf], axis=-1).mean() * 1000
            sim_cav = (a[t_end, surf] < 0.5)
            obs_cav = (obs_alpha_np[t_end, surf] < 0.5)
            inter = (sim_cav & obs_cav).sum(); union = (sim_cav | obs_cav).sum()
            iou = inter / max(union, 1)
            pos_mm_list.append(pos_mm); iou_list.append(iou)
        print(f"  {name:10s}: pos {np.mean(pos_mm_list):.3f}±{np.std(pos_mm_list):.3f} mm   "
                f"cav_IoU {np.mean(iou_list):.3f}±{np.std(iou_list):.3f}  (mean of {len(anchor_ends)} windows)")

    # Viz — 30 frames every 10th of 300
    rng = np.random.RandomState(0)
    subsample_idx = rng.choice(N, 10000, replace=False)
    out_dir = REPO / "tools" / "pointnetpp" / "phase28_short_horizon_viz"
    out_dir.mkdir(exist_ok=True)
    keep = list(range(0, T_TRAJ, 10))
    print(f"\nRender {len(keep)} frames ...", flush=True)
    for k_idx, t in enumerate(keep):
        panels = [
            (obs_pos[t], obs_alpha[t], "fake-real truth"),
            (init_pos[t], init_alpha[t], "sim_init"),
            (L_pos[t], L_alpha[t], "Liang Phase2.8"),
            (H_pos[t], H_alpha[t], "Hier Phase2.8"),
        ]
        render_4panel(panels, traj[t], bbox_min, bbox_max, x_mid, y_floor,
                       k_idx, len(keep),
                       f"{cfg['name']}  [Liang short-horizon T_la={T_LA}, calib traj A]",
                       out_dir / f"frame_{k_idx:02d}.png", subsample_idx)
    frames = [Image.open(out_dir / f"frame_{k_idx:02d}.png") for k_idx in range(len(keep))]
    gif_path = REPO / "tools" / "pointnetpp" / "phase28_short_horizon_viz.gif"
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                      duration=250, loop=0)
    print(f"saved {gif_path}", flush=True)


if __name__ == "__main__":
    main()
