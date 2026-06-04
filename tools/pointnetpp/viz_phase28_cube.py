"""4-panel viz of Phase 2.8 online calibration result on cube.

Loads tau_L / tau_H from phase2_online_phase1_cache.npz (Phase 2.8 output).
Forward 4 sims on T=300 slow-knife trajectory, render side-by-side.
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
T_HELD = 900   # slow-knife match


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
    knife_radius = cfg["knife_radius_abs"]   # Phase 2.8 default

    # warm-up
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
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    top_axis = cfg["top_axis"]

    tau_truth, lr_mask, x_mid = build_lr_tau(pos_rest, TAU_LEFT, TAU_RIGHT)
    stiff_uniform = np.full(N, STIFF_UNIFORM, dtype=np.float32)

    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx = torch.tensor(np.where(surface_mask)[0], dtype=torch.long, device=device)

    # ★ VIZ ON CALIBRATION TRAJ A (same as test_online_phase1_cube used)
    # This tests whether calibration ACTUALLY worked (model matches obs on calib traj).
    # Held-out test is generalization — separate question.
    spec_A = dict(cfg["traj_specs"]["A"])
    spec_A["sweep_range"] = (0.10, 0.90)
    traj_held = make_traj_from_spec(spec_A, bbox_min, bbox_max, T_HELD)

    theta_truth_np = np.array([3.0, 20.0, 0.3], dtype=np.float32)

    # Load Phase 2.8 tau
    cache = REPO / "tools" / "pointnetpp" / "phase2_online_phase1_cache.npz"
    d = np.load(str(cache))
    tau_L = d["tau_L"]; tau_H = d["tau_H"]
    print(f"loaded Phase 2.8 cache:", flush=True)
    print(f"  tau_L L={tau_L[lr_mask].mean():.3f} R={tau_L[~lr_mask].mean():.3f}  gap={tau_L[~lr_mask].mean() - tau_L[lr_mask].mean():.3f}")
    print(f"  tau_H L={tau_H[lr_mask].mean():.3f} R={tau_H[~lr_mask].mean():.3f}  gap={tau_H[~lr_mask].mean() - tau_H[lr_mask].mean():.3f}")

    print("Gen fake-real obs (T=300 slow knife) ...", flush=True)
    solver.reset_to_rest(pos_rest)
    obs_pos_held, obs_alpha_held = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_uniform, traj_held,
        knife_radius, device, tau_truth_np=tau_truth)

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
    print("Forward sim_Liang (Phase 2.8) ...", flush=True)
    L_pos_list, L_alpha_list = predict_on_traj(stiff_init, tau_L, traj_held, ctx)
    print("Forward sim_Hier (Phase 2.8) ...", flush=True)
    H_pos_list, H_alpha_list = predict_on_traj(stiff_init, tau_H, traj_held, ctx)

    # subsample for viz
    rng = np.random.RandomState(0)
    subsample_idx = rng.choice(N, 10000, replace=False)
    out_dir = REPO / "tools" / "pointnetpp" / "phase5_slowknife_viz"
    out_dir.mkdir(exist_ok=True)
    keep = list(range(0, T_HELD, 10))  # 30 frames
    print(f"Render {len(keep)} frames (every 10th of {T_HELD}) ...", flush=True)
    for k_idx, t in enumerate(keep):
        panels = [
            (obs_pos_held[t], obs_alpha_held[t], "fake-real truth"),
            (init_pos[t], init_alpha[t], "sim_init (uncalib)"),
            (L_pos_list[t], L_alpha_list[t], "Liang online τ"),
            (H_pos_list[t], H_alpha_list[t], "Hier online τ"),
        ]
        render_4panel(panels, traj_held[t], bbox_min, bbox_max, x_mid, y_floor,
                       k_idx, len(keep),
                       f"{cfg['name']}  [Phase 2.8 online on CALIB traj A, T=300]",
                       out_dir / f"frame_{k_idx:02d}.png", subsample_idx)
    frames = [Image.open(out_dir / f"frame_{k_idx:02d}.png") for k_idx in range(len(keep))]
    gif_path = REPO / "tools" / "pointnetpp" / "phase5_slowknife_viz.gif"
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                      duration=250, loop=0)
    print(f"saved {gif_path}", flush=True)

    # Quantitative — surface position + cavity IoU per method
    obs_pos_np = np.stack(obs_pos_held)
    obs_alpha_np = np.stack(obs_alpha_held)
    init_pos_np = np.stack(init_pos); init_alpha_np = np.stack(init_alpha)
    L_pos_np = np.stack(L_pos_list); L_alpha_np = np.stack(L_alpha_list)
    H_pos_np = np.stack(H_pos_list); H_alpha_np = np.stack(H_alpha_list)
    surf_np = surface_mask
    print("\n=== surface metrics over T=300 ===")
    for name, p, a in [("sim_init", init_pos_np, init_alpha_np),
                          ("Liang", L_pos_np, L_alpha_np),
                          ("Hier", H_pos_np, H_alpha_np)]:
        pos_mm = np.linalg.norm(p[:, surf_np] - obs_pos_np[:, surf_np], axis=-1).mean() * 1000
        # cavity IoU: both alpha < 0.5
        sim_cav = (a[:, surf_np] < 0.5)
        obs_cav = (obs_alpha_np[:, surf_np] < 0.5)
        inter = (sim_cav & obs_cav).sum()
        union = (sim_cav | obs_cav).sum()
        iou = inter / max(union, 1)
        print(f"  {name:10s}: surface_pos {pos_mm:.3f} mm   surface_cavity_IoU {iou:.3f}")


if __name__ == "__main__":
    main()
