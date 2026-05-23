"""Day 8 audit T_init: basin robustness — 5+ random θ_init points.

Day 7-8 results were on single θ_init = (5, 15, 0.6). Test concern: maybe this
init is in a "lucky basin"; other inits might not converge or converge wrong.

Test: 5+ random init points spanning the param ranges, run 100 epochs each
with same hyperparams. Check:
  - All converge toward θ_true?
  - Variance of final θ across inits?
  - Identifies basin of attraction vs need for warm-start

Init range:
  α_min ∈ [1.5, 8.0]    (truth 3.0)
  sharpness ∈ [8.0, 30.0]  (truth 20.0)
  τ ∈ [0.3, 0.8]        (truth 0.5)

5 inits picked deterministically across this box.

Pass criteria:
  - ≥4/5 inits have final err < 10% on all 3 dims (basin OK)
  - All 5 don't NaN
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
    param_names = ["α_min", "sharpness", "τ"]

    # 5 init points spanning the box
    inits = [
        np.array([5.0, 15.0, 0.6], dtype=np.float32),   # Day 8 baseline (mild +)
        np.array([1.5, 30.0, 0.3], dtype=np.float32),   # under-α, over-sharp, low-τ
        np.array([8.0, 8.0,  0.8], dtype=np.float32),   # over-α, under-sharp, high-τ
        np.array([2.0, 25.0, 0.45], dtype=np.float32),  # near-truth slight off
        np.array([6.0, 12.0, 0.7], dtype=np.float32),   # mid-range mixed
    ]
    init_labels = [
        "baseline",
        "α↓ s↑ τ↓",
        "α↑ s↓ τ↑",
        "near-truth",
        "mid-mixed",
    ]
    init_perts = [
        f"({(i[0]-theta_true[0])/theta_true[0]*100:+.0f}%, "
        f"{(i[1]-theta_true[1])/theta_true[1]*100:+.0f}%, "
        f"{(i[2]-theta_true[2])/theta_true[2]*100:+.0f}%)"
        for i in inits
    ]

    N_epochs = 100
    lr = 0.02

    print(f"\n[Setup] N_inits={len(inits)}, N_epochs={N_epochs}, lr={lr}")
    print(f"  θ_true = {theta_true.tolist()}")
    for i, (label, init, pert) in enumerate(zip(init_labels, inits, init_perts)):
        print(f"  init {i+1} ({label}): {init.tolist()}  pert={pert}")

    # Generate obs once
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

    results = []

    for i, (label, theta_init) in enumerate(zip(init_labels, inits)):
        init_err = np.abs(theta_init - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        print(f"\n[Init {i+1} '{label}'] θ_init={theta_init.tolist()}  "
              f"init err=({init_err[0]:.1f}%,{init_err[1]:.1f}%,{init_err[2]:.1f}%)")
        t0 = time.time()
        traj, L_gap_hist, err = run_resect_loop(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
            theta_init, theta_true, device,
            N_epochs=N_epochs, lr=lr, n_inner=30)
        elapsed = time.time() - t0
        if err:
            print(f"  ✗ {err} ({elapsed:.1f}s)")
            results.append(dict(label=label, init=theta_init, err=err, final=None))
            continue
        theta_final = traj[-1]
        theta_avg10 = np.mean(traj[-10:], axis=0)
        errs_final = np.abs(theta_final - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        errs_avg10 = np.abs(theta_avg10 - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        L_gap_last = float(L_gap_hist[-1])
        print(f"  {elapsed:.1f}s. final=({theta_final[0]:.4f},{theta_final[1]:.4f},"
              f"{theta_final[2]:.4f})  err=({errs_final[0]:.2f}%,{errs_final[1]:.2f}%,"
              f"{errs_final[2]:.2f}%)")
        print(f"         avg10=({theta_avg10[0]:.4f},{theta_avg10[1]:.4f},"
              f"{theta_avg10[2]:.4f})  err=({errs_avg10[0]:.2f}%,{errs_avg10[1]:.2f}%,"
              f"{errs_avg10[2]:.2f}%)  L_gap={L_gap_last:.4e}")
        results.append(dict(
            label=label, init=theta_init, final=theta_final,
            avg10=theta_avg10, errs_avg10=errs_avg10, L_gap=L_gap_last,
        ))

    # Verdict
    print(f"\n" + "=" * 70)
    print(f"T_init verdict — basin robustness across {len(inits)} init points")
    print(f"=" * 70)
    print(f"\n  Summary (avg-last-10-epoch err):")
    print(f"  {'init':>15s}  {'pert':>20s}  {'α_err':>7s}  {'s_err':>7s}  {'τ_err':>7s}  {'L_gap':>10s}")
    n_converged_strict = 0  # all dims < 10%
    n_converged_loose = 0   # all dims < 20%
    n_nan = 0
    for i, r in enumerate(results):
        if r.get('final') is None:
            print(f"  {r['label']:>15s}  {init_perts[i]:>20s}  --- NaN/abort ---")
            n_nan += 1
            continue
        errs = r['errs_avg10']
        status = "✓" if (errs < 10).all() else ("⚠" if (errs < 20).all() else "✗")
        print(f"  {r['label']:>15s}  {init_perts[i]:>20s}  "
              f"{errs[0]:>6.2f}%  {errs[1]:>6.2f}%  {errs[2]:>6.2f}%  "
              f"{r['L_gap']:.3e}  {status}")
        if (errs < 10).all():
            n_converged_strict += 1
        if (errs < 20).all():
            n_converged_loose += 1
    print()

    n_total = len(inits)
    if n_nan > 0:
        print(f"  {n_nan} runs crashed (NaN)")
    if n_converged_strict >= 4:
        print(f"  ✓✓ PASS — {n_converged_strict}/{n_total} inits converged tight (<10%)")
    elif n_converged_loose >= 4:
        print(f"  ✓ PASS — {n_converged_loose}/{n_total} inits converged loose (<20%)")
        print(f"     ({n_converged_strict}/{n_total} tight at 100 epochs — 200 epochs might tighten)")
    elif n_converged_loose >= 2:
        print(f"  ⚠ MARGINAL — only {n_converged_loose}/{n_total} converged within 20%")
        print(f"     Basin of attraction limited; warm-start may be needed")
    else:
        print(f"  ✗ FAIL — only {n_converged_loose}/{n_total} converged")
        print(f"     ReSECT not robust to arbitrary init; TPE warm-start required")

    np.savez(REPO_ROOT / "tools" / "day8_T_init.npz",
             inits=np.array(inits), theta_true=theta_true,
             results=results)


if __name__ == "__main__":
    main()
