"""Day 8 Step 5 LONG RUN: 200 epochs to find true sharpness recovery plateau.

Day 8 60-epoch run gave sharp 10.4% err — better than Day 7's broken 22% but
still room to improve. Question: is 10% the real obs limit, or just slow
convergence?

This long run:
  - 200 epochs × T=3 = 600 Adam updates
  - Track θ trajectory every epoch
  - Look for plateau on each dim
  - If sharp keeps decreasing → just slow convergence (not obs limit)
  - If sharp plateaus near 5-10% → real obs limit

Also tests 3 different lrs to ensure plateau is lr-robust:
  - lr=0.02 (Day 8 baseline)
  - lr=0.01 (slower, more careful)
  - lr=0.05 (faster, more aggressive)
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
    setup_solver_with_rest, solver_load_state, solver_save_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day8_stable_inner_loop import gd_inner_loop_stable  # noqa: E402
from tools.day8_step5_rerun_stable_gd import (  # noqa: E402
    gen_obs_multistep_param_arr,
)
from tools.day6f_v0_multistep_synth import (  # noqa: E402
    make_knife_trajectory, clamp_theta_inplace,
)


def run_resect_loop(solver, rest_state, knife_traj, knife_radius,
                     obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
                     theta_init, theta_true, device,
                     N_epochs, lr, n_inner=30):
    """One full multi-epoch online ReSECT run with stable GD."""
    T = len(knife_traj)
    theta_torch = torch.tensor(theta_init, dtype=torch.float32, device=device,
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
            opt.step()
            clamp_theta_inplace(theta_torch)
            L_gaps_epoch.append(float(L_gap.item()))
            state = solver_save_state(solver)
            dx_np = delta_x.detach().cpu().numpy()
            ddv_np = delta_dv.detach().cpu().numpy()
            state["pos"] = state["pos"] + dx_np
            state["particle_damage"] = np.maximum(state["particle_damage"] + ddv_np, 0.0)
            s_val = float(theta_torch[1].item())
            tau_val = float(theta_torch[2].item())
            new_dv = state["particle_damage"]
            state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(s_val * (new_dv - tau_val), -50, 50)))
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
    theta_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)
    param_names = ["α_min", "sharpness", "τ"]

    print(f"\n[Setup] θ_true={theta_true.tolist()}  θ_init={theta_init.tolist()}")
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

    # Run 3 lrs in parallel for plateau verification
    runs = {}
    for lr in [0.01, 0.02, 0.05]:
        print(f"\n[Run lr={lr}] starting 200-epoch long run …")
        t0 = time.time()
        traj, L_gap_hist, err = run_resect_loop(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
            theta_init, theta_true, device,
            N_epochs=200, lr=lr, n_inner=30)
        elapsed = time.time() - t0
        if err:
            print(f"  ✗ {err}")
            continue
        print(f"  done in {elapsed:.1f}s")
        # Print at checkpoint epochs
        ckpt_epochs = [10, 30, 60, 100, 150, 199]
        for e in ckpt_epochs:
            th = traj[e+1]
            errs = np.abs(th - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
            print(f"  ep {e+1:>3d}: θ=({th[0]:.4f},{th[1]:.4f},{th[2]:.4f})  "
                  f"err=({errs[0]:5.2f}%,{errs[1]:5.2f}%,{errs[2]:5.2f}%)  "
                  f"L_gap={L_gap_hist[e]:.3e}")
        # Average over last 10 epochs for stable measure
        theta_avg10 = np.mean(traj[-10:], axis=0)
        errs_avg10 = np.abs(theta_avg10 - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        print(f"  avg10 last: θ=({theta_avg10[0]:.4f},{theta_avg10[1]:.4f},{theta_avg10[2]:.4f})  "
              f"err=({errs_avg10[0]:5.2f}%,{errs_avg10[1]:5.2f}%,{errs_avg10[2]:5.2f}%)")
        runs[lr] = dict(traj=traj, L_gap_hist=L_gap_hist, avg10_err=errs_avg10)

    # Verdict
    print(f"\n" + "=" * 70)
    print(f"Day 8 Step 5 LONG RUN — sharpness plateau analysis")
    print(f"=" * 70)
    print(f"\n  Final err per dim, avg last 10 epochs:")
    print(f"  {'lr':>6s}  {'α_min':>9s}  {'sharp':>9s}  {'τ':>9s}")
    for lr, r in runs.items():
        print(f"  {lr:>6.3f}  {r['avg10_err'][0]:>7.2f}%  {r['avg10_err'][1]:>7.2f}%  "
              f"{r['avg10_err'][2]:>7.2f}%")

    # Sharpness trend across epochs
    print(f"\n  Sharpness err trajectory (best run, lr={list(runs.keys())[0]}):")
    if runs:
        best_lr = min(runs.keys(), key=lambda k: runs[k]['avg10_err'][1])
        traj_best = runs[best_lr]['traj']
        for e in [10, 30, 60, 100, 150, 200]:
            err_s = abs(traj_best[e][1] - 20.0) / 20.0 * 100
            print(f"    epoch {e:>3d}: sharp={traj_best[e][1]:.4f}  err={err_s:5.2f}%")

    # Determine if plateau reached
    if runs:
        best_lr = min(runs.keys(), key=lambda k: runs[k]['avg10_err'][1])
        traj_best = runs[best_lr]['traj']
        sharp_traj = traj_best[:, 1]
        last50_var = np.std(sharp_traj[-50:])
        last50_mean = np.mean(sharp_traj[-50:])
        print(f"\n  Plateau analysis (best lr={best_lr}):")
        print(f"    sharp last 50 epochs:  mean={last50_mean:.4f}  std={last50_var:.4f}")
        if last50_var < 0.05:
            print(f"    → PLATEAU detected (std < 0.05)")
            print(f"    → sharp ~{last50_mean:.2f} is the convergence floor")
            print(f"    → err {abs(last50_mean-20)/20*100:.1f}% is real obs limit (with this 1-step T=3 obs)")
        else:
            print(f"    → still moving (std {last50_var:.4f}) — could go further with more epochs")

    np.savez(REPO_ROOT / "tools" / "day8_step5_long_run.npz",
             theta_true=theta_true, theta_init=theta_init,
             **{f"lr_{lr}_traj": r['traj'] for lr, r in runs.items()},
             **{f"lr_{lr}_lgap": r['L_gap_hist'] for lr, r in runs.items()})


if __name__ == "__main__":
    main()
