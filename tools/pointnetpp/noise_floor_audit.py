"""Quantify GPU non-determinism noise floor of predict_on_trajectory.

Call predict_on_trajectory(truth_params) 5 times on the same trajectory and
measure the spread (std) of outputs. This gives the noise floor for ANY
comparison metric we use.

Then compare: hierarchical "win" Δ vs this noise floor.
Honest verdict:
   |Δ_signal| < 3 σ_noise   → claim is not distinguishable from noise
   |Δ_signal| > 5 σ_noise   → claim is meaningful

Metrics measured:
   1. mean per-particle pos diff (averaged over alive particles + T frames)
   2. alpha L1 (averaged over particles + T frames)
   3. max per-particle pos diff (worst case)
   4. cavity fraction (alive / total per frame, averaged)
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
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

from tools.day2_solver_wrapper import setup_solver_with_rest  # noqa: E402

from tools.stageA_inclusion_recovery import build_inclusion_truth  # noqa: E402
from tools.pointnetpp.cross_trajectory_test import (  # noqa: E402
    make_traj, predict_on_trajectory,
)


def per_traj_metrics(pred_pos, pred_alpha):
    """Stats per trajectory: total alpha sum, mean pos magnitude (for stable
    comparison across runs)."""
    T = len(pred_pos)
    total_alpha = sum(float(a.sum()) for a in pred_alpha)
    mean_pos = sum(float(p.mean()) for p in pred_pos) / T
    mean_alpha = total_alpha / (T * pred_alpha[0].size)
    return mean_pos, mean_alpha


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    print(f"\n=== Noise floor audit ===")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = 2

    device = "cuda:0"
    knife_radius = 0.08
    T = 10

    theta_truth = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    pos_rest = mesh_info["vertices"]
    stiff_truth, inclusion_mask, center, incl_r = build_inclusion_truth(
        pos_rest, stiff_in=0.8, stiff_out=0.3,
        center_override=(0.15, 0.25, 0.15),
        radius_override=0.06)
    tau_truth = np.where(inclusion_mask, 0.20, 0.40).astype(np.float32)

    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_radius=knife_radius, theta_3d_np=theta_truth,
        pos_rest=pos_rest,
    )

    # ==== Quantify noise: 5 repeats on T_A with truth params ====
    N_REPEATS = 5
    print(f"\n[Noise quantification] {N_REPEATS} repeats of predict(truth, T_A)")
    traj_A = make_traj("A", T=T)
    runs_pos = []
    runs_alpha = []
    for r in range(N_REPEATS):
        pred_pos, pred_alpha = predict_on_trajectory(stiff_truth, tau_truth, traj_A, ctx)
        runs_pos.append(pred_pos)
        runs_alpha.append(pred_alpha)
        if r > 0:
            # Diff vs run 0
            diffs_per_frame = []
            for t in range(T):
                d = np.linalg.norm(pred_pos[t] - runs_pos[0][t], axis=1)  # [N]
                diffs_per_frame.append(d)
            all_diffs = np.concatenate(diffs_per_frame)
            mean_d = all_diffs.mean()
            max_d = all_diffs.max()
            alpha_d = np.mean([np.abs(runs_alpha[0][t] - pred_alpha[t]).mean() for t in range(T)])
            print(f"  run {r} vs run 0: mean pos diff = {mean_d:.3e} m  "
                  f"max = {max_d:.3e} m   mean α diff = {alpha_d:.3e}")

    # Aggregate noise stats: compute pairwise diffs for all run pairs
    print(f"\n[Aggregate noise stats over 5 runs of same inputs]")
    pairwise_mean_pos = []
    pairwise_max_pos = []
    pairwise_mean_alpha = []
    for i in range(N_REPEATS):
        for j in range(i + 1, N_REPEATS):
            for t in range(T):
                d = np.linalg.norm(runs_pos[i][t] - runs_pos[j][t], axis=1)
                pairwise_mean_pos.append(d.mean())
                pairwise_max_pos.append(d.max())
                pairwise_mean_alpha.append(np.abs(runs_alpha[i][t] - runs_alpha[j][t]).mean())
    pairwise_mean_pos = np.array(pairwise_mean_pos)
    pairwise_max_pos = np.array(pairwise_max_pos)
    pairwise_mean_alpha = np.array(pairwise_mean_alpha)

    noise_mean_pos = pairwise_mean_pos.mean()
    noise_std_pos = pairwise_mean_pos.std()
    noise_max_pos = pairwise_max_pos.mean()
    noise_mean_alpha = pairwise_mean_alpha.mean()
    noise_std_alpha = pairwise_mean_alpha.std()

    print(f"  Position (mean per-particle per-frame):  noise μ = {noise_mean_pos:.3e} ± {noise_std_pos:.3e} m")
    print(f"  Position (max per-particle per-frame) :  noise μ = {noise_max_pos:.3e} m")
    print(f"  Alpha (mean per-particle per-frame)   :  noise μ = {noise_mean_alpha:.3e} ± {noise_std_alpha:.3e}")

    # ==== Compare to claimed signals from cross_traj_test ====
    print(f"\n{'='*72}\nSIGNAL vs NOISE comparison\n{'='*72}")
    signals = {
        "T_A pos (hier vs base)": ("pos", 0.000088 - 0.000070),
        "T_B pos (hier vs base)": ("pos", 0.000087 - 0.000074),
        "T_C pos (hier vs base)": ("pos", 0.000101 - 0.000085),
        "T_D pos (hier vs base)": ("pos", 0.000099 - 0.000085),
        "T_A α (hier vs base)":   ("alpha", 0.005826 - 0.005323),
        "T_B α (hier vs base)":   ("alpha", 0.005884 - 0.005425),
        "T_C α (hier vs base)":   ("alpha", 0.007199 - 0.006555),
        "T_D α (hier vs base)":   ("alpha", 0.007323 - 0.006725),
    }
    print(f"  {'metric':30s}  {'|Δ_signal|':>12s}  {'noise μ':>12s}  {'SNR':>6s}  verdict")
    for name, (kind, delta) in signals.items():
        ad = abs(delta)
        if kind == "pos":
            noise_mu = noise_mean_pos
            snr = ad / max(noise_mu, 1e-15)
        else:
            noise_mu = noise_mean_alpha
            snr = ad / max(noise_mu, 1e-15)
        if snr > 5:
            verdict = "✓ MEANINGFUL"
        elif snr > 3:
            verdict = "△ marginal"
        else:
            verdict = "✗ within noise"
        print(f"  {name:30s}  {ad:>12.3e}  {noise_mu:>12.3e}  {snr:>6.1f}× {verdict}")


if __name__ == "__main__":
    main()
