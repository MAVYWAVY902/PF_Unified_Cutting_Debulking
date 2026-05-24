"""R3-B: diagnose all-high init failure — budget vs local minimum?

R3 showed init=(6, 30, 0.8, 1.3) failed at 50 ep with α 34%, τ 18%, stiff 23%.
Question: would 200 ep recover it (just budget), or does it plateau (local min)?

Also probe lr sensitivity (lr=0.02 vs lr=0.01 vs lr=0.05) — high init may
need different lr because gradient magnitudes differ from baseline.

Logs per-10-epoch snapshots of all 4 dims for the most informative case.
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

SHARPNESS_FIXED = 20.0


def run_4d_logged(solver, rest_state, knife_traj, knife_radius,
                    obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t,
                    edges_rest_t, theta_init_4d, device,
                    N_epochs=200, lr=0.02, n_inner=30):
    T = len(knife_traj)
    theta_init_fixed = theta_init_4d.copy()
    theta_init_fixed[1] = SHARPNESS_FIXED
    theta_torch = torch.tensor(theta_init_fixed, dtype=torch.float32, device=device,
                                 requires_grad=True)
    opt = torch.optim.Adam([theta_torch], lr=lr)

    epoch_thetas = [theta_torch.detach().cpu().numpy().copy()]
    L_per_epoch = []
    grad_norm_per_step = []

    for epoch in range(N_epochs):
        state = rest_state
        L_gaps_e = []
        for t in range(T):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state, knife_traj[t], knife_radius,
            )
            delta_x, delta_dv, _ = gd_inner_loop_stable(
                pos_t, dv_t, obs_pos_t[t], obs_alpha_t[t],
                theta_torch, edges_a_t, edges_b_t, edges_rest_t,
                n_iter=n_inner, lr_pos=0.5, lr_dv_factor=1.0,  # Day 3c+lr fix: Newton-optimal lr_pos
                lambda_cav=1.0, lambda_E=0.0,
            )
            L_gap = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
            opt.zero_grad()
            L_gap.backward()
            if torch.isnan(theta_torch.grad).any():
                return None, "NaN", None, None
            g_norm = float(torch.norm(theta_torch.grad).item())
            grad_norm_per_step.append(g_norm)
            theta_torch.grad[1] = 0.0
            opt.step()
            with torch.no_grad():
                theta_torch[1] = SHARPNESS_FIXED
            clamp_theta_inplace(theta_torch)
            state = solver_save_state(solver)
            dx_np = delta_x.detach().cpu().numpy()
            ddv_np = delta_dv.detach().cpu().numpy()
            state["pos"] = state["pos"] + dx_np
            state["particle_damage"] = np.maximum(
                state["particle_damage"] + ddv_np, 0.0)
            # Day 3c: inject uses Liang-style inner constants (not outer θ)
            new_dv = state["particle_damage"]
            state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(
                SHARPNESS_INNER * (new_dv - TAU_INNER), -50, 50)))
            L_gaps_e.append(float(L_gap.item()))
        epoch_thetas.append(theta_torch.detach().cpu().numpy().copy())
        L_per_epoch.append(np.mean(L_gaps_e))
    return np.array(epoch_thetas), None, np.array(L_per_epoch), np.array(grad_norm_per_step)


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, _ = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_radius = 0.04
    T = 3
    knife_traj = make_knife_trajectory(T)
    theta_true = np.array([3.0, 20.0, 0.5, 0.85], dtype=np.float32)
    theta_init_allhigh = np.array([6.0, 30.0, 0.8, 1.3], dtype=np.float32)

    print(f"\n[Setup] θ_true_4D = {theta_true.tolist()}")
    print(f"        init all-high = {theta_init_allhigh.tolist()}")
    print(f"        init err: α=100% τ=60% stiff=53%")

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

    # Run 200 ep with lr=0.02 (R3 default), and lr=0.05 (more aggressive)
    runs = [("lr=0.02", 0.02), ("lr=0.05", 0.05)]
    all_results = {}

    for label, lr in runs:
        print(f"\n{'='*72}")
        print(f"Run: {label}  N_epochs=200")
        print(f"{'='*72}")
        t0 = time.time()
        traj, err, L, gnorm = run_4d_logged(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
            theta_init_allhigh, device, N_epochs=200, lr=lr, n_inner=30)
        elapsed = time.time() - t0
        if err:
            print(f"  ✗ {err}")
            continue
        all_results[label] = dict(traj=traj, L=L, gnorm=gnorm)
        print(f"  elapsed {elapsed:.0f}s, 200 epochs × T=3 = 600 Adam steps")

        # Snapshot every 20 epochs
        print(f"\n  Trajectory (every 20 ep):")
        print(f"  {'ep':>4s}  {'α':>8s}  {'τ':>8s}  {'stiff':>8s}  "
              f"{'α err%':>7s}  {'τ err%':>7s}  {'stiff err%':>10s}  L_gap")
        for ep in [0, 10, 20, 40, 60, 80, 100, 140, 180, 199]:
            th = traj[ep + 1] if ep + 1 < len(traj) else traj[-1]
            errs = np.abs(th - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
            L_val = L[ep] if ep < len(L) else L[-1]
            print(f"  {ep:>4d}  {th[0]:8.4f}  {th[2]:8.4f}  {th[3]:8.4f}  "
                  f"{errs[0]:7.2f}  {errs[2]:7.2f}  {errs[3]:10.2f}  {L_val:.4e}")

        avg10 = traj[-10:].mean(axis=0)
        errs_final = np.abs(avg10 - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        free_errs = np.array([errs_final[0], errs_final[2], errs_final[3]])
        status = "✓ tight" if (free_errs < 10).all() else ("⚠ loose" if (free_errs < 20).all() else "✗ failed")
        print(f"\n  Final avg10: α={errs_final[0]:.2f}%, τ={errs_final[2]:.2f}%, "
              f"stiff={errs_final[3]:.2f}%  →  {status}")

        # Loss plateau check
        L_first10 = L[:10].mean()
        L_last10 = L[-10:].mean()
        L_drop = L_first10 / max(L_last10, 1e-30)
        print(f"  L_gap evolution: first 10 ep mean={L_first10:.4e}, "
              f"last 10 ep mean={L_last10:.4e}, drop={L_drop:.1f}×")

    # Verdict
    print(f"\n{'='*72}")
    print(f"R3-B VERDICT")
    print(f"{'='*72}")
    converged_any = False
    for label, lr in runs:
        if label not in all_results:
            continue
        r = all_results[label]
        avg10 = r["traj"][-10:].mean(axis=0)
        errs = np.abs(avg10 - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        free_errs = np.array([errs[0], errs[2], errs[3]])
        if (free_errs < 10).all():
            converged_any = True
            print(f"  ✓ {label} converged at 200 ep — R3 50 ep was BUDGET, not local min")
        elif (free_errs < 20).all():
            print(f"  ⚠ {label} loose at 200 ep — partial budget + maybe shallow basin")
        else:
            print(f"  ✗ {label} still failed at 200 ep — likely LOCAL MIN or unreachable")
    if converged_any:
        print(f"\n  All-high recovery IS achievable with enough epochs.")
        print(f"  R3 marginal status was budget-limited, not basin-limited.")
    else:
        print(f"\n  All-high may genuinely have a local min trap.")
        print(f"  Need: per-dim lr scaling, TPE warm-start, or different init strategy.")


if __name__ == "__main__":
    main()
