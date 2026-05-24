"""CANONICAL ReSECT sys-ID entry (renamed from day9_fix_sharpness_validation 2026-05-23).

Pipeline (Layer 5, see project_day9_fix_sharpness_locked_2026_05_23.md):
   θ_optimize = (α_min, τ) 2D   [sharpness FROZEN, see SHARPNESS_FIXED below]
   Layers 1-4 (kernels, solver.step, WarpFullForwardOp, gd_inner_loop_stable)
   stay 3-element θ for compat. Freeze is a Layer-5-only hack:
     - theta.grad[1] = 0 each step
     - theta[1] snapped to SHARPNESS_FIXED post-Adam (defends float drift)

Replaces (DEPRECATED):
   tools/calibrate_stage2_autograd.py  — Stage B 2D (α, sharp), no τ, no ReSECT
   tools/day8_step5_long_run.py        — Day 8 free-sharp 200-ep baseline

For real phantom (Stage 2.2):
   SHARPNESS_FIXED ← from literature / pre-calibration. DO NOT use 20 blindly.
   Future Day 3b adds stiffness dim → θ becomes 3D (α_min, τ, stiffness),
   sharpness still frozen.

This file is the historical Day 9 validation harness. It runs two tests:
   Test 1: 200-ep long run × 3 lrs (best: lr=0.05 → α 0.48% / τ 0.25%)
   Test 2: 5-init basin sweep (3/5 converge — same count as free-sharp,
           but rescues init 2 (sharp_init=30, far from truth))

For Stage 2.2 production (TBD): adapt run_resect_loop_fix_sharp() to read real
phantom obs (pos, alpha or surface IoU) instead of synth gen_obs_multistep_param_arr.
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

from tools.day2_solver_wrapper import (  # noqa: E402
    setup_solver_with_rest, solver_save_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day8_stable_inner_loop import (  # noqa: E402
    gd_inner_loop_stable, SHARPNESS_INNER, TAU_INNER,
)
from tools.day8_step5_rerun_stable_gd import gen_obs_multistep_param_arr  # noqa: E402
from tools.day6f_v0_multistep_synth import (  # noqa: E402
    make_knife_trajectory, clamp_theta_inplace,
)


SHARPNESS_FIXED = 20.0  # truth value = literature value for synth test


def run_resect_loop_fix_sharp(solver, rest_state, knife_traj, knife_radius,
                                obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t,
                                edges_rest_t, theta_init, theta_true, device,
                                N_epochs, lr, n_inner=30):
    """ReSECT outer loop with sharpness FROZEN at SHARPNESS_FIXED.

    θ is 3-element internally (for WarpFullForwardOp), but grad on index 1
    is zeroed each step, and value snapped back to SHARPNESS_FIXED.
    Effective optimization is on (α_min, τ) only.
    """
    T = len(knife_traj)
    # Init with sharpness = SHARPNESS_FIXED (overriding theta_init[1])
    theta_init_fixed = theta_init.copy()
    theta_init_fixed[1] = SHARPNESS_FIXED
    theta_torch = torch.tensor(theta_init_fixed, dtype=torch.float32, device=device,
                                 requires_grad=True)
    opt = torch.optim.Adam([theta_torch], lr=lr)
    epoch_end_theta = [theta_torch.detach().cpu().numpy().copy()]
    L_gap_history = []

    for epoch in range(N_epochs):
        state = rest_state
        L_gaps_epoch = []
        for t in range(T):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state, knife_traj[t], knife_radius,
            )
            delta_x, delta_dv, info = gd_inner_loop_stable(
                pos_t, dv_t, obs_pos_t[t], obs_alpha_t[t],
                theta_torch, edges_a_t, edges_b_t, edges_rest_t,
                n_iter=n_inner, lr_pos=5e-3, lr_dv_factor=1.0,
                lambda_cav=1.0, lambda_E=0.0,
            )
            L_gap = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
            opt.zero_grad()
            L_gap.backward()
            if torch.isnan(theta_torch.grad).any():
                return None, None, "NaN"
            # KEY: zero sharpness gradient (freeze sharpness dim)
            theta_torch.grad[1] = 0.0
            opt.step()
            # Snap sharpness back exactly (Adam may have tiny float drift)
            with torch.no_grad():
                theta_torch[1] = SHARPNESS_FIXED
            clamp_theta_inplace(theta_torch)
            L_gaps_epoch.append(float(L_gap.item()))
            # Mode C inject — Day 3c: use Liang-style inner constants
            # (consistent with inner loop's sigmoid; outer θ does NOT enter)
            state = solver_save_state(solver)
            dx_np = delta_x.detach().cpu().numpy()
            ddv_np = delta_dv.detach().cpu().numpy()
            state["pos"] = state["pos"] + dx_np
            state["particle_damage"] = np.maximum(state["particle_damage"] + ddv_np, 0.0)
            new_dv = state["particle_damage"]
            state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(
                SHARPNESS_INNER * (new_dv - TAU_INNER), -50, 50)))
        epoch_end_theta.append(theta_torch.detach().cpu().numpy().copy())
        L_gap_history.append(np.mean(L_gaps_epoch))
    return np.array(epoch_end_theta), np.array(L_gap_history), None


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_radius = 0.04
    T = 3
    knife_traj = make_knife_trajectory(T)
    theta_true = np.array([3.0, 20.0, 0.5], dtype=np.float32)
    param_names = ["α_min", "sharpness(FIXED)", "τ"]

    print(f"\n[Setup] T={T}, θ_true={theta_true.tolist()}")
    print(f"        SHARPNESS_FIXED = {SHARPNESS_FIXED}")
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

    # ============ Test 1: Step 5 long run with fixed sharpness ============
    print(f"\n[Test 1] Step 5 long run with sharpness frozen={SHARPNESS_FIXED}, "
          f"baseline θ_init=(5,15,0.6)")
    theta_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)
    print(f"  N_epochs=200, lr=0.02, n_inner=30")

    for lr in [0.01, 0.02, 0.05]:
        print(f"\n  [lr={lr}]:")
        t0 = time.time()
        traj, L_gap_hist, err = run_resect_loop_fix_sharp(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
            theta_init, theta_true, device,
            N_epochs=200, lr=lr, n_inner=30)
        elapsed = time.time() - t0
        if err:
            print(f"    ✗ {err}")
            continue
        theta_avg10 = np.mean(traj[-10:], axis=0)
        errs = np.abs(theta_avg10 - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        print(f"    {elapsed:.1f}s. avg10 θ=({theta_avg10[0]:.4f}, "
              f"{theta_avg10[1]:.2f}, {theta_avg10[2]:.4f})")
        print(f"    err α_min={errs[0]:.2f}%  sharpness={errs[1]:.2f}%(forced 0)  τ={errs[2]:.2f}%")
        # Compare to Day 8 free-sharpness numbers:
        print(f"    Day 8 free-sharp (lr={lr}): "
              f"α={1.60 if lr==0.02 else 12.51 if lr==0.01 else 3.14}%, "
              f"sharp={1.16 if lr==0.02 else 7.14 if lr==0.01 else 1.58}%, "
              f"τ={0.41 if lr==0.02 else 4.17 if lr==0.01 else 0.55}%")

    # ============ Test 2: T_init multi-init with fixed sharpness ============
    print(f"\n[Test 2] T_init (5 inits, 100 ep) with sharpness frozen={SHARPNESS_FIXED}")
    inits = [
        np.array([5.0, 15.0, 0.6], dtype=np.float32),
        np.array([1.5, 30.0, 0.3], dtype=np.float32),
        np.array([8.0, 8.0,  0.8], dtype=np.float32),
        np.array([2.0, 25.0, 0.45], dtype=np.float32),
        np.array([6.0, 12.0, 0.7], dtype=np.float32),
    ]
    labels = ["baseline", "α↓ s↑ τ↓", "α↑ s↓ τ↑", "near-truth", "mid-mixed"]

    print(f"  Day 8 free-sharp result for comparison: 3/5 inits within 20%; "
          f"sharpness was slowest-converging dim")
    print(f"  Fixed sharpness should improve basin (no slow sharpness dim)")
    print()

    n_strict = 0
    n_loose = 0
    for i, (lbl, ti) in enumerate(zip(labels, inits)):
        ti_fixed = ti.copy(); ti_fixed[1] = SHARPNESS_FIXED  # snap initial sharpness
        init_err = np.abs(ti - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        t0 = time.time()
        traj, _, err = run_resect_loop_fix_sharp(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
            ti_fixed, theta_true, device,
            N_epochs=100, lr=0.02, n_inner=30)
        elapsed = time.time() - t0
        if err:
            print(f"  init {i+1} ({lbl}): {err}")
            continue
        ta = np.mean(traj[-10:], axis=0)
        errs = np.abs(ta - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        # Only check α_min and τ (sharpness frozen, doesn't count)
        relevant_errs = np.array([errs[0], errs[2]])
        sym = "✓" if (relevant_errs < 10).all() else ("⚠" if (relevant_errs < 20).all() else "✗")
        if (relevant_errs < 10).all(): n_strict += 1
        if (relevant_errs < 20).all(): n_loose += 1
        print(f"  init {i+1} ({lbl:>10s}) {elapsed:.0f}s: "
              f"init_err α={init_err[0]:.0f}% τ={init_err[2]:.0f}% → "
              f"final α={errs[0]:.2f}%  τ={errs[2]:.2f}%  {sym}")

    # ============ Verdict ============
    print(f"\n" + "=" * 70)
    print(f"Day 9 verdict — fixing sharpness=20")
    print(f"=" * 70)
    print(f"\n  T_init: {n_strict}/5 converged tight (<10%), {n_loose}/5 within 20%")
    print(f"  Day 8 free-sharp had: 3/5 within 20%")
    print(f"  → {'BETTER' if n_loose > 3 else 'SIMILAR' if n_loose == 3 else 'WORSE'} basin robustness")
    print(f"\n  Recommendation:")
    if n_loose >= 4:
        print(f"  ✓ Fix sharpness in Warp pipeline going forward")
        print(f"  ✓ θ = (α_min, τ) for sys-ID; sharpness preset from literature/pre-cal")
        print(f"  ✓ Re-run any subsequent experiments with fixed sharpness")
    elif n_loose >= 3:
        print(f"  ⚠ Comparable robustness; either choice defensible")
        print(f"  ⚠ Prefer fixed for simplicity (ICRA paper cleaner)")
    else:
        print(f"  ✗ Fixed sharpness degraded basin; keep sharpness in θ")


if __name__ == "__main__":
    main()
