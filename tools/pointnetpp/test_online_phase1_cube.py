"""Phase 1 smoke test: online calibration on v12 cube setup (T=300 slow knife).

Goals (lambda_hist=0 for now):
   1. Verify memory stays bounded (not O(T))
   2. tau gap recovery should match or exceed v11 epoch-based
   3. Per-frame Adam step rate sane (~1-3 step/s on RTX 2070)
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
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec  # noqa: E402
from tools.pointnetpp.calibrate_online import calibrate_online  # noqa: E402

TAU_LEFT = 0.15; TAU_RIGHT = 0.55
TAU_INIT = 0.30
STIFF_UNIFORM = 0.5
T_TRAJ = 300   # back to Phase 2.8 baseline; key change is teacher_force_every below


def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


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
    knife_radius = cfg["knife_radius_abs"]   # revert to Phase 2.8 default

    # ---- warm-up settle ----
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

    # ---- L/R tau truth ----
    x_mid = 0.5 * (pos_rest[:, 0].min() + pos_rest[:, 0].max())
    lr_mask = pos_rest[:, 0] < x_mid
    tau_truth = np.where(lr_mask, TAU_LEFT, TAU_RIGHT).astype(np.float32)
    stiff_truth = np.full(N, STIFF_UNIFORM, dtype=np.float32)

    # ---- surface index ----
    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx = torch.tensor(np.where(surface_mask)[0], dtype=torch.long, device=device)

    # ---- 2 calibration trajectories, T=300 each slow knife ----
    spec_A = dict(cfg["traj_specs"]["A"])
    spec_A["sweep_range"] = (0.10, 0.90)
    traj_A = make_traj_from_spec(spec_A, bbox_min, bbox_max, T_TRAJ)
    spec_B = dict(cfg["traj_specs"]["A"])
    spec_B["sweep_range"] = (0.10, 0.90)
    fixed_B = dict(spec_B["fixed"])
    other_axes = [i for i in range(3) if i != spec_B["sweep_axis"]]
    for ax in other_axes:
        if ax != top_axis:
            fixed_B[ax] = max(0.20, min(0.80, fixed_B[ax] + 0.20))
            break
    spec_B["fixed"] = fixed_B
    traj_B = make_traj_from_spec(spec_B, bbox_min, bbox_max, T_TRAJ)

    theta_truth = np.array([3.0, 20.0, 0.3], dtype=np.float32)
    print("Gen fake-real obs (2 traj × T=300) ...", flush=True)
    solver.reset_to_rest(pos_rest)
    obs_pos_A, obs_alpha_A = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth, stiff_truth, traj_A,
        knife_radius, device, tau_truth_np=tau_truth)
    solver.reset_to_rest(pos_rest)
    obs_pos_B, obs_alpha_B = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth, stiff_truth, traj_B,
        knife_radius, device, tau_truth_np=tau_truth)

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos_A]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha_A]
    obs_pos_B_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos_B]
    obs_alpha_B_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha_B]
    multi_traj = [(traj_A, obs_pos_A_t, obs_alpha_A_t),
                  (traj_B, obs_pos_B_t, obs_alpha_B_t)]

    # ---- hier ----
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

    # ---- Phase 6: Truncated BPTT (teacher_force_every=10) ----
    LR = 0.03; LR_TAU = 0.5; N_PASSES = 4; LAMBDA_HIST = 1.0; LAMBDA_CAV = 1.0
    LR_DECAY = 0.5
    USE_INNER_GD = False
    INNER_LAMBDA_E = 0.0
    TF_EVERY = 10   # ★ free-rollout window = 10 frames (recover accumulated cavity signal)
    print(f"Phase 6 (truncated BPTT, teacher_force_every={TF_EVERY}): "
            f"lambda_hist={LAMBDA_HIST}, lambda_cav={LAMBDA_CAV}")
    print(f"\n=== Liang online (hist={LAMBDA_HIST}, cav={LAMBDA_CAV}, lr={LR}, lr_tau={LR_TAU}) ===")
    mem0 = gpu_mem_mb()
    t0 = time.time()
    stiff_L, tau_L, log_L = calibrate_online(
        "L_pos_partial", ctx, multi_traj, N_passes=N_PASSES,
        lr=LR, lr_tau=LR_TAU, lr_decay_per_pass=LR_DECAY,
        lambda_hist=LAMBDA_HIST,
        lambda_outer_cav=LAMBDA_CAV,
        use_inner_gd=USE_INNER_GD,
        inner_lambda_E=INNER_LAMBDA_E,
        teacher_force_every=TF_EVERY, log_every=200,
    )
    t_L = time.time() - t0
    mem_L_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"\n=== Hier online (hist={LAMBDA_HIST}, cav={LAMBDA_CAV}) ===")
    torch.cuda.reset_peak_memory_stats()
    mem1 = gpu_mem_mb()
    t1 = time.time()
    stiff_H, tau_H, log_H = calibrate_online(
        "Hier+L_pos_partial", ctx, multi_traj, N_passes=N_PASSES,
        lr=LR, lr_tau=LR_TAU, lr_decay_per_pass=LR_DECAY,
        lambda_hist=LAMBDA_HIST,
        lambda_outer_cav=LAMBDA_CAV,
        use_inner_gd=USE_INNER_GD,
        inner_lambda_E=INNER_LAMBDA_E,
        teacher_force_every=TF_EVERY, log_every=200,
    )
    t_H = time.time() - t1
    mem_H_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    # ---- Report ----
    print("\n" + "=" * 70)
    print("PHASE 1 SMOKE TEST RESULTS")
    print("=" * 70)
    print(f"Frames per calibration: {2 * T_TRAJ} = 2 traj × T={T_TRAJ}")
    print(f"\nLiang online:")
    print(f"  time:        {t_L:.1f}s ({2 * T_TRAJ / t_L:.1f} frame/s)")
    print(f"  peak GPU:    {mem_L_peak:.0f} MB")
    print(f"  tau_L left:  {tau_L[lr_mask].mean():.3f}  right: {tau_L[~lr_mask].mean():.3f}  "
            f"gap: {tau_L[~lr_mask].mean() - tau_L[lr_mask].mean():.3f}")
    print(f"  loss start/end: {log_L[0]['L_gap']:.3e} -> {log_L[-1]['L_gap']:.3e}")
    print(f"\nHier online:")
    print(f"  time:        {t_H:.1f}s ({2 * T_TRAJ / t_H:.1f} frame/s)")
    print(f"  peak GPU:    {mem_H_peak:.0f} MB")
    print(f"  tau_H left:  {tau_H[lr_mask].mean():.3f}  right: {tau_H[~lr_mask].mean():.3f}  "
            f"gap: {tau_H[~lr_mask].mean() - tau_H[lr_mask].mean():.3f}")
    print(f"  loss start/end: {log_H[0]['L_gap']:.3e} -> {log_H[-1]['L_gap']:.3e}")
    print(f"\nReference v11 (epoch-based, T=15):")
    print(f"  Liang gap: 0.029  (recovered 7% of truth 0.40)")
    print(f"  Hier  gap: 0.080  (recovered 20% of truth 0.40)")

    # save cache
    cache = REPO / "tools" / "pointnetpp" / "phase2_online_phase1_cache.npz"
    np.savez_compressed(str(cache),
        tau_L=tau_L, tau_H=tau_H, stiff_L=stiff_L, stiff_H=stiff_H,
        tau_truth=tau_truth, lr_mask=lr_mask)
    print(f"\nsaved {cache}")


if __name__ == "__main__":
    main()
