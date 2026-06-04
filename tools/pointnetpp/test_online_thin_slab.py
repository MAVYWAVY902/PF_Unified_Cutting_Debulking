"""Thin-slab variant of test_online_phase1_cube.py.

Mesh: cuboid_h040 (1m × 0.4m × 1m, aspect 2.5:1) vs cube_tet (1:1:1).
Hypothesis: thinner geometry exposes bending modes → tau magnitude becomes
observable from surface positions (which couldn't see it in thick cube).

Setup parallel to v12/Phase 2.8 cube:
   - Y is thin direction (gravity axis, top-anchored knife)
   - L/R tau split on X axis
   - Knife sweeps along Z axis on top surface (Y=Y_max)
   - Bottom (Y=Y_min) 20% anchored
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

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff  # noqa: E402
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.calibrate_online import calibrate_online  # noqa: E402

TAU_LEFT = 0.15; TAU_RIGHT = 0.55
TAU_INIT = 0.30
STIFF_UNIFORM = 0.5
T_TRAJ = 300


def gpu_mem_mb():
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def main():
    device = "cuda:0"
    mesh_path = REPO / "resources" / "cuboid_flat" / "cuboid_h040.msh"

    solver, rest_state, mesh_info = setup_solver_with_rest(
        str(mesh_path), fix_threshold_fraction=0.20)
    N = solver.N
    solver.damping = 0.95; solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    print(f"Mesh: N={N}, extent={pos_rest.max(0) - pos_rest.min(0)}", flush=True)

    # Thin direction = Y (extent 0.4). Top = Y_max.
    top_axis = 1
    solver.gravity = wp.vec3(0.0, -9.81, 0.0)
    solver.floor_y = float(pos_rest[:, 2].min() - 1.0)
    knife_radius = 0.05   # 50mm, matches cube scale

    # ---- warm-up settle ----
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
    sag_max = np.linalg.norm(pos_settled - pos_rest, axis=1).max() * 1000
    print(f"  Warm-up done: max sag {sag_max:.2f} mm", flush=True)
    pos_rest = pos_settled
    mesh_info["vertices"] = pos_settled
    rest_state = state_warm
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0)
    bbox_size = bbox_max - bbox_min
    print(f"  Settled extent: {bbox_size}", flush=True)

    # ---- L/R tau truth on X axis ----
    x_mid = 0.5 * (pos_rest[:, 0].min() + pos_rest[:, 0].max())
    lr_mask = pos_rest[:, 0] < x_mid
    tau_truth = np.where(lr_mask, TAU_LEFT, TAU_RIGHT).astype(np.float32)
    stiff_uniform = np.full(N, STIFF_UNIFORM, dtype=np.float32)

    # ---- surface index (top 10% of Y) ----
    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx = torch.tensor(np.where(surface_mask)[0], dtype=torch.long, device=device)
    print(f"  Surface particles (top 10% Y): {surface_mask.sum()} / {N} ({100*surface_mask.mean():.1f}%)", flush=True)

    # ---- 2 calibration trajectories: knife sweeps along Z on top surface ----
    # Traj A: sweep Z 0.10→0.90, at X=0.50 (center), Y=Y_max
    # Traj B: sweep Z 0.10→0.90, at X=0.70 (offset), Y=Y_max
    def make_traj(x_frac, z_frac_range, T):
        x_v = bbox_min[0] + x_frac * bbox_size[0]
        y_v = bbox_max[1]
        z_s = np.linspace(
            bbox_min[2] + z_frac_range[0] * bbox_size[2],
            bbox_min[2] + z_frac_range[1] * bbox_size[2], T)
        return [np.array([x_v, y_v, z_v], dtype=np.float32) for z_v in z_s]

    traj_A = make_traj(0.50, (0.10, 0.90), T_TRAJ)
    traj_B = make_traj(0.70, (0.10, 0.90), T_TRAJ)

    theta_truth = np.array([3.0, 20.0, 0.3], dtype=np.float32)
    print(f"Gen fake-real obs (2 traj × T={T_TRAJ}) ...", flush=True)
    solver.reset_to_rest(pos_rest)
    obs_pos_A, obs_alpha_A = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth, stiff_uniform, traj_A,
        knife_radius, device, tau_truth_np=tau_truth)
    solver.reset_to_rest(pos_rest)
    obs_pos_B, obs_alpha_B = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth, stiff_uniform, traj_B,
        knife_radius, device, tau_truth_np=tau_truth)

    # Quick deformation diagnostic
    disp_A = np.linalg.norm(np.stack(obs_pos_A) - pos_rest, axis=-1) * 1000
    cav_A = (np.stack(obs_alpha_A) < 0.5).mean(axis=1)
    print(f"  Traj A: max disp {disp_A.max():.1f} mm, mean t=T/2 {disp_A[T_TRAJ//2].mean():.3f} mm")
    print(f"         cavity final {cav_A[-1]*100:.2f}%", flush=True)

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos_A]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha_A]
    obs_pos_B_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos_B]
    obs_alpha_B_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha_B]
    multi_traj = [(traj_A, obs_pos_A_t, obs_alpha_A_t),
                  (traj_B, obs_pos_B_t, obs_alpha_B_t)]

    x_rest = torch.from_numpy(pos_rest).to(device, dtype=torch.float32)
    M1 = min(512, max(32, N // 20))
    M2 = min(64, max(8, N // 200))
    H_hier = build_hierarchy(x_rest, M1=M1, M2=M2, k_neighbors=3)
    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edges_a_t = torch.tensor(edges_a_np, dtype=torch.long, device=device)
    edges_b_t = torch.tensor(edges_b_np, dtype=torch.long, device=device)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)

    stiff_init = np.full(N, STIFF_UNIFORM, dtype=np.float32)
    tau_init = np.full(N, TAU_INIT, dtype=np.float32)
    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_radius=knife_radius, pos_rest=pos_rest,
        theta_3d_np=theta_truth,
        stiff_init_np=stiff_init, tau_init_np=tau_init,
        tau_bound=(0.05, 0.95),
        surface_idx=surface_idx,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H_hier,
    )

    # ---- Phase 2.8 setup (no inner GD, simplified) ----
    LR = 0.03; LR_TAU = 0.5; N_PASSES = 4; LAMBDA_HIST = 1.0; LAMBDA_CAV = 1.0
    LR_DECAY = 0.5
    print(f"\n=== Liang online thin slab (Phase 2.8 setup) ===")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    stiff_L, tau_L, log_L = calibrate_online(
        "L_pos_partial", ctx, multi_traj, N_passes=N_PASSES,
        lr=LR, lr_tau=LR_TAU, lr_decay_per_pass=LR_DECAY,
        lambda_hist=LAMBDA_HIST,
        lambda_outer_cav=LAMBDA_CAV, log_every=200,
    )
    t_L = time.time() - t0
    mem_L = gpu_mem_mb()
    print(f"\n=== Hier online thin slab ===")
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    stiff_H, tau_H, log_H = calibrate_online(
        "Hier+L_pos_partial", ctx, multi_traj, N_passes=N_PASSES,
        lr=LR, lr_tau=LR_TAU, lr_decay_per_pass=LR_DECAY,
        lambda_hist=LAMBDA_HIST,
        lambda_outer_cav=LAMBDA_CAV, log_every=200,
    )
    t_H = time.time() - t1
    mem_H = gpu_mem_mb()

    print("\n" + "=" * 70)
    print("THIN-SLAB RESULTS  (cuboid_h040, aspect 2.5:1)")
    print("=" * 70)
    print(f"Liang online:")
    print(f"  time: {t_L:.1f}s   peak GPU: {mem_L:.0f} MB")
    print(f"  tau_L: L={tau_L[lr_mask].mean():.3f} R={tau_L[~lr_mask].mean():.3f}  "
            f"gap={tau_L[~lr_mask].mean() - tau_L[lr_mask].mean():.3f}  "
            f"mean={(tau_L[lr_mask].mean() + tau_L[~lr_mask].mean()) / 2:.3f}")
    print(f"Hier online:")
    print(f"  time: {t_H:.1f}s   peak GPU: {mem_H:.0f} MB")
    print(f"  tau_H: L={tau_H[lr_mask].mean():.3f} R={tau_H[~lr_mask].mean():.3f}  "
            f"gap={tau_H[~lr_mask].mean() - tau_H[lr_mask].mean():.3f}  "
            f"mean={(tau_H[lr_mask].mean() + tau_H[~lr_mask].mean()) / 2:.3f}")
    print(f"\nTruth: L=0.150 R=0.550  gap=0.400  mean=0.350")
    print(f"\nCube reference (Phase 2.8): Hier gap=0.055 (14% of truth)")

    cache = REPO / "tools" / "pointnetpp" / "phase_thin_slab_cache.npz"
    np.savez_compressed(str(cache),
        tau_L=tau_L, tau_H=tau_H, stiff_L=stiff_L, stiff_H=stiff_H,
        tau_truth=tau_truth, lr_mask=lr_mask, pos_rest=pos_rest)
    print(f"saved {cache}")


if __name__ == "__main__":
    main()
