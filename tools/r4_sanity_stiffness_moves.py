"""R4 sanity: does stiffness actually move in 4D calibrate_resect.py?

R4-v2 says L_α / L_pos / L_combined all have ~zero stiffness signal in our
cube+knife+1-substep setup. If true, then the actual pipeline's grad[3] should
be ≈ 0, Adam should not move stiffness, and final stiffness ≈ init.

If stiffness DOES move and recovers: R4-v2 is wrong / incomplete (cumulative
effects of T=3 + 30 inner iter give signal even though 1-step Hessian doesn't).

Setup:
  θ_true_4D = (3.0, 20.0, 0.5, 0.85)
  θ_init_4D = (5.0, 15.0, 0.6, 0.70)   ← stiffness 18% off from truth
  T=3, n_inner=30, lr=0.02, N_epochs=50 (short)
  sharpness frozen (Day 9 lock); stiffness FREE (3 of 4 dims optimized)

Logs per (epoch, t) step:
  - θ values (all 4)
  - ‖grad‖ per dim (especially [3] stiffness)
  - L_gap

Verdict:
  PASS (R4-v2 wrong): stiffness moves > 1% and toward truth
  FAIL (R4-v2 right): grad[3] ≈ 0 OR stiffness drifts away/random
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
from tools.day8_stable_inner_loop import gd_inner_loop_stable  # noqa: E402
from tools.day8_step5_rerun_stable_gd import gen_obs_multistep_param_arr  # noqa: E402
from tools.day6f_v0_multistep_synth import (  # noqa: E402
    make_knife_trajectory, clamp_theta_inplace,
)


SHARPNESS_FIXED = 20.0


def run_with_grad_logging(solver, rest_state, knife_traj, knife_radius,
                            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t,
                            edges_rest_t, theta_init_4d, theta_true_4d, device,
                            N_epochs, lr, n_inner=30):
    T = len(knife_traj)
    # Init: snap sharpness only; stiffness FREE
    theta_init_fixed = theta_init_4d.copy()
    theta_init_fixed[1] = SHARPNESS_FIXED
    theta_torch = torch.tensor(theta_init_fixed, dtype=torch.float32, device=device,
                                 requires_grad=True)
    opt = torch.optim.Adam([theta_torch], lr=lr)

    grad_log = []      # per-step grad[3] (stiffness)
    theta_log = []     # per-step θ[3] (stiffness)
    grad_full = []     # per-step full ‖grad‖ across all dims
    L_log = []

    theta_log.append(theta_torch.detach().cpu().numpy().copy())

    for epoch in range(N_epochs):
        state = rest_state
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
                return None, "NaN"
            # LOG per-step grad BEFORE Adam step
            g = theta_torch.grad.detach().cpu().numpy().copy()
            grad_log.append(g[3])  # stiffness grad
            grad_full.append(g.copy())
            L_log.append(float(L_gap.item()))
            # Freeze sharpness only (stiffness IS optimized)
            theta_torch.grad[1] = 0.0
            opt.step()
            with torch.no_grad():
                theta_torch[1] = SHARPNESS_FIXED  # snap
            clamp_theta_inplace(theta_torch)
            theta_log.append(theta_torch.detach().cpu().numpy().copy())
            # Mode C inject
            state = solver_save_state(solver)
            dx_np = delta_x.detach().cpu().numpy()
            ddv_np = delta_dv.detach().cpu().numpy()
            state["pos"] = state["pos"] + dx_np
            state["particle_damage"] = np.maximum(
                state["particle_damage"] + ddv_np, 0.0)
            s_val = float(theta_torch[1].item())
            tau_val = float(theta_torch[2].item())
            new_dv = state["particle_damage"]
            state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(
                s_val * (new_dv - tau_val), -50, 50)))
    return dict(
        theta_log=np.array(theta_log),
        grad_log=np.array(grad_log),
        grad_full=np.array(grad_full),
        L_log=np.array(L_log),
    ), None


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, _ = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_radius = 0.04
    T = 3
    knife_traj = make_knife_trajectory(T)
    theta_true = np.array([3.0, 20.0, 0.5, 0.85], dtype=np.float32)
    theta_init = np.array([5.0, 15.0, 0.6, 0.70], dtype=np.float32)
    print(f"\n[Setup]")
    print(f"  θ_true_4D = {theta_true.tolist()}")
    print(f"  θ_init_4D = {theta_init.tolist()}")
    print(f"  stiffness init off by {abs(0.70-0.85)/0.85*100:.1f}%")
    print(f"  N_epochs=50, lr=0.02, T=3 → 150 Adam steps")

    # Gen obs at 4D θ_true via param_arr path
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

    t0 = time.time()
    print(f"\n[Run] starting 4D recovery …")
    result, err = run_with_grad_logging(
        solver, rest_state, knife_traj, knife_radius,
        obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
        theta_init, theta_true, device,
        N_epochs=50, lr=0.02, n_inner=30)
    elapsed = time.time() - t0
    if err:
        print(f"✗ {err}")
        return
    print(f"  elapsed {elapsed:.1f}s, {len(result['L_log'])} Adam steps")

    theta_traj = result["theta_log"]   # (151, 4)
    g_stiff = result["grad_log"]       # (150,)
    g_full = result["grad_full"]       # (150, 4)
    L = result["L_log"]                # (150,)

    # ---- Per-dim diagnostics ----
    print(f"\n{'='*72}")
    print(f"Per-dim trajectory (θ_true = [3.0, 20.0, 0.5, 0.85])")
    print(f"{'='*72}")
    final_theta = theta_traj[-1]
    init_theta = theta_traj[0]
    avg10 = theta_traj[-10:].mean(axis=0)
    move_pct = np.abs(final_theta - init_theta) / np.maximum(np.abs(init_theta), 1e-12) * 100
    err_pct = np.abs(avg10 - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
    names = ["α_min", "sharp(FROZEN)", "τ", "stiffness"]
    print(f"\n  {'dim':<14s}  {'init':>8s}  {'final':>8s}  {'truth':>8s}  "
          f"{'moved':>10s}  {'err vs truth':>14s}")
    for i in range(4):
        print(f"  {names[i]:<14s}  {init_theta[i]:8.4f}  {final_theta[i]:8.4f}  "
              f"{theta_true[i]:8.4f}  {move_pct[i]:9.2f}%  {err_pct[i]:13.2f}%")

    # ---- Stiffness-specific deep dive ----
    print(f"\n{'='*72}")
    print(f"Stiffness gradient diagnostic (the key question)")
    print(f"{'='*72}")
    print(f"\n  ‖grad[stiff]‖ stats over 150 steps:")
    print(f"    mean: {np.abs(g_stiff).mean():.4e}")
    print(f"    max : {np.abs(g_stiff).max():.4e}")
    print(f"    min : {np.abs(g_stiff).min():.4e}")
    print(f"    std : {g_stiff.std():.4e}")
    print(f"    sign flips: {int(np.sum(np.diff(np.sign(g_stiff)) != 0))}")

    print(f"\n  Compare to other dims (mean ‖grad‖):")
    for i, n in enumerate(names):
        print(f"    {n:<14s}: {np.abs(g_full[:, i]).mean():.4e}")

    # Stiffness trajectory pattern
    stiff_traj = theta_traj[:, 3]
    print(f"\n  Stiffness trajectory (samples):")
    pts = np.linspace(0, len(stiff_traj) - 1, 11, dtype=int)
    for p in pts:
        marker = "←" if p == 0 else ""
        if p == len(stiff_traj) - 1:
            marker = "← final"
        print(f"    step {p:3d}: stiff = {stiff_traj[p]:.6f}  {marker}")

    # ---- Verdict ----
    print(f"\n{'='*72}")
    print(f"VERDICT")
    print(f"{'='*72}")
    stiff_moved_pct = move_pct[3]
    stiff_err_pct = err_pct[3]
    stiff_toward_truth = abs(final_theta[3] - theta_true[3]) < abs(init_theta[3] - theta_true[3])

    if stiff_moved_pct < 1.0:
        print(f"  ✗ Stiffness BARELY MOVED ({stiff_moved_pct:.2f}% from init)")
        print(f"    → R4-v2 CONFIRMED: pipeline's effective L_gap doesn't see stiffness signal.")
        print(f"    → Joint 4D recovery NOT viable. Must use Stage 2.1 indenter to fix stiffness.")
    elif stiff_toward_truth and stiff_err_pct < 10:
        print(f"  ✓ Stiffness RECOVERED ({stiff_err_pct:.2f}% err < 10%)")
        print(f"    → R4-v2 WRONG: cumulative T+inner effects give signal even though")
        print(f"      single-step Hessian doesn't show it. P1.2-6 IS viable.")
    elif stiff_toward_truth:
        print(f"  ⚠ Stiffness MOVED TOWARD TRUTH but not converged ({stiff_err_pct:.2f}% err)")
        print(f"    → Some signal exists. Try longer run / different lr.")
    else:
        print(f"  ✗ Stiffness DRIFTED AWAY from truth ({stiff_err_pct:.2f}% err)")
        print(f"    → Gradient direction WRONG. Possibly L_gap sign bug or saddle.")

    np.savez(str(REPO_ROOT / "tools" / "r4_sanity_stiff_traj.npz"),
              theta_log=theta_traj, grad_log=g_stiff, grad_full=g_full, L_log=L)
    print(f"\n  trajectory saved → tools/r4_sanity_stiff_traj.npz")


if __name__ == "__main__":
    main()
