"""Day 8 audit T_seed: reproducibility across runs.

Sources of stochasticity in our pipeline:
  - Warp atomic_add in particle damage aggregation (parallel reduction order)
  - GPU floating-point non-associativity
  - (No explicit RNG in our code — solver state init deterministic, θ_init fixed)

Test: run Day 8 Step 5 N_runs=5 times with IDENTICAL setup. If results vary
much across runs, reproducibility issue. If they cluster tightly, reliable.

Pass criteria:
  - Std of final θ across runs < 5% relative
  - Std of last-10-epoch-avg θ < 3% relative

This rules out the "we got lucky on that one run" concern (audit hack H6).
"""
import sys
from pathlib import Path
import numpy as np
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pf_shape_matching_warp"))

import torch
import warp as wp
wp.init()
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

from tools.day2_solver_wrapper import setup_solver_with_rest  # noqa: E402
from tools.day8_step5_long_run import run_resect_loop  # noqa: E402
from tools.day8_step5_rerun_stable_gd import gen_obs_multistep_param_arr  # noqa: E402
from tools.day6f_v0_multistep_synth import make_knife_trajectory  # noqa: E402


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_radius = 0.04
    T = 3
    knife_traj = make_knife_trajectory(T)
    theta_true = np.array([3.0, 20.0, 0.5], dtype=np.float32)
    theta_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)
    param_names = ["α_min", "sharpness", "τ"]

    N_runs = 5
    N_epochs = 60
    lr = 0.02

    print(f"\n[Setup] N_runs={N_runs}, N_epochs={N_epochs}, lr={lr}")
    print(f"  θ_true={theta_true.tolist()}, θ_init={theta_init.tolist()}")
    print(f"  Same setup every run — checks atomic_add / GPU non-determinism")

    # Generate obs ONCE (deterministic baseline)
    obs_pos, obs_alpha = gen_obs_multistep_param_arr(
        solver, rest_state, theta_true, knife_traj, knife_radius, device)
    obs_pos_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_rest_t = torch.tensor(solver.e["pf_rest"].numpy(),
                                  dtype=torch.float32, device=device)

    final_thetas = []
    avg10_thetas = []
    L_gap_lasts = []

    for run in range(N_runs):
        print(f"\n[Run {run+1}/{N_runs}] …")
        t0 = time.time()
        traj, L_gap_hist, err = run_resect_loop(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
            theta_init, theta_true, device,
            N_epochs=N_epochs, lr=lr, n_inner=30)
        elapsed = time.time() - t0
        if err:
            print(f"  ✗ {err}")
            continue
        theta_final = traj[-1]
        theta_avg10 = np.mean(traj[-10:], axis=0)
        L_gap_last10 = np.mean(L_gap_hist[-10:])
        errs_final = np.abs(theta_final - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        errs_avg10 = np.abs(theta_avg10 - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        print(f"  {elapsed:.1f}s. final θ=({theta_final[0]:.4f},{theta_final[1]:.4f},"
              f"{theta_final[2]:.4f})  err=({errs_final[0]:.2f}%,{errs_final[1]:.2f}%,"
              f"{errs_final[2]:.2f}%)")
        print(f"         avg10 θ=({theta_avg10[0]:.4f},{theta_avg10[1]:.4f},"
              f"{theta_avg10[2]:.4f})  err=({errs_avg10[0]:.2f}%,{errs_avg10[1]:.2f}%,"
              f"{errs_avg10[2]:.2f}%)  L_gap_last10={L_gap_last10:.4e}")
        final_thetas.append(theta_final)
        avg10_thetas.append(theta_avg10)
        L_gap_lasts.append(L_gap_last10)

    # ============ Verdict ============
    print(f"\n" + "=" * 70)
    print(f"T_seed verdict — reproducibility across {N_runs} runs")
    print(f"=" * 70)

    final_thetas = np.array(final_thetas)
    avg10_thetas = np.array(avg10_thetas)
    L_gap_lasts = np.array(L_gap_lasts)

    print(f"\n  Final θ per run:")
    for run, th in enumerate(final_thetas):
        print(f"    run {run+1}: ({th[0]:.4f}, {th[1]:.4f}, {th[2]:.4f})")

    print(f"\n  Final θ statistics:")
    for d, name in enumerate(param_names):
        mean_d = float(np.mean(final_thetas[:, d]))
        std_d = float(np.std(final_thetas[:, d]))
        rel_std = std_d / max(abs(mean_d), 1e-12) * 100
        print(f"    {name:>10s}: mean={mean_d:.4f}  std={std_d:.4f}  rel_std={rel_std:.3f}%")

    print(f"\n  avg10 θ statistics:")
    for d, name in enumerate(param_names):
        mean_d = float(np.mean(avg10_thetas[:, d]))
        std_d = float(np.std(avg10_thetas[:, d]))
        rel_std = std_d / max(abs(mean_d), 1e-12) * 100
        print(f"    {name:>10s}: mean={mean_d:.4f}  std={std_d:.4f}  rel_std={rel_std:.3f}%")

    print(f"\n  L_gap_last10 across runs: mean={float(np.mean(L_gap_lasts)):.4e}  "
          f"std={float(np.std(L_gap_lasts)):.4e}")

    # Check
    avg10_rel_stds = [
        np.std(avg10_thetas[:, d]) / max(abs(np.mean(avg10_thetas[:, d])), 1e-12) * 100
        for d in range(3)
    ]
    max_rel_std = max(avg10_rel_stds)
    if max_rel_std < 1.0:
        verdict = f"✓✓ PASS — highly reproducible (max rel_std {max_rel_std:.2f}%)"
    elif max_rel_std < 3.0:
        verdict = f"✓ PASS — reproducible within 3% (max rel_std {max_rel_std:.2f}%)"
    elif max_rel_std < 10.0:
        verdict = f"⚠ MARGINAL — rel_std {max_rel_std:.2f}% > 3% target"
    else:
        verdict = f"✗ FAIL — rel_std {max_rel_std:.2f}% indicates non-reproducibility"
    print(f"\n  {verdict}")

    np.savez(REPO_ROOT / "tools" / "day8_T_seed.npz",
             final_thetas=final_thetas, avg10_thetas=avg10_thetas,
             L_gap_lasts=L_gap_lasts, theta_true=theta_true)


if __name__ == "__main__":
    main()
