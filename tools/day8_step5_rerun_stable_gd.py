"""Day 8: re-run Day 7 Step 5 with stable GD inner loop.

Test if α_min 1.5% / τ 3.0% / sharp 22% recovery survives the lr_dv fix.
If yes: Day 7's claim was robust to inner loop instability (outer Adam saw
        correct ∇θ direction despite divergent Δ magnitudes).
If no:  Day 7 result was artifact of broken inner loop.

Uses:
  - param_arr obs path (same as sim, no path drift)
  - gd_inner_loop_stable (lr_dv = 1/s², theta-adaptive)
  - Same outer hyperparams as Day 7 Step 5 v2 (lr=0.02, N_epochs=60)
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
from tools.day6f_v0_multistep_synth import (  # noqa: E402
    make_knife_trajectory, clamp_theta_inplace,
)


def gen_obs_multistep_param_arr(solver, rest_state, theta_true, knife_traj,
                                   knife_radius, device):
    """obs via param_arr path (matches sim, no drift)."""
    theta_torch = torch.tensor(theta_true, dtype=torch.float32, device=device,
                                 requires_grad=False)
    obs_pos = []
    obs_alpha = []
    state_in = rest_state
    with torch.no_grad():
        for t, knife_pos in enumerate(knife_traj):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, knife_pos, knife_radius,
            )
            obs_pos.append(pos_t.cpu().numpy().copy())
            obs_alpha.append(alpha_t.cpu().numpy().copy())
            state_in = solver_save_state(solver)
    return obs_pos, obs_alpha


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

    N_epochs = 60
    lr = 0.02
    n_inner = 30

    print(f"\n[Setup] T={T}, N_epochs={N_epochs}, lr={lr}, n_inner={n_inner}")
    print(f"  θ_true={theta_true.tolist()}  θ_init={theta_init.tolist()}")
    print(f"  Inner loop: stable GD with lr_dv = 1/s² (adaptive)")
    print(f"  Obs path: param_arr (matches sim, no path drift)")

    # Generate obs via param_arr path
    print(f"\n[Setup] generating obs at θ_true (param_arr path) …")
    obs_pos, obs_alpha = gen_obs_multistep_param_arr(
        solver, rest_state, theta_true, knife_traj, knife_radius, device)
    obs_pos_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]
    for t in range(T):
        cf = float((obs_alpha[t] < 0.5).mean())
        print(f"  frame {t+1}: obs cavity_frac={cf:.4f}")

    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_rest_t = torch.tensor(solver.e["pf_rest"].numpy(),
                                  dtype=torch.float32, device=device)

    # Sanity: L_gap at θ_true via this stable inner loop
    print(f"\n[Sanity] L_gap at θ_true with stable GD inner loop …")
    theta_truth_t = torch.tensor(theta_true, dtype=torch.float32, device=device,
                                   requires_grad=False)
    L_gap_truth_per_frame = []
    state = rest_state
    with torch.no_grad():
        for t in range(T):
            alpha_st, pos_st, dv_st = WarpFullForwardOp.apply(
                theta_truth_t, solver, state, knife_traj[t], knife_radius,
            )
            state = solver_save_state(solver)
        # Re-compute L_gap per frame at truth
    state = rest_state
    for t in range(T):
        with torch.no_grad():
            alpha_st, pos_st, dv_st = WarpFullForwardOp.apply(
                theta_truth_t, solver, state, knife_traj[t], knife_radius,
            )
        delta_x_t, delta_dv_t, info_t = gd_inner_loop_stable(
            pos_st.detach(), dv_st.detach(), obs_pos_t[t], obs_alpha_t[t],
            theta_truth_t, edges_a_t, edges_b_t, edges_rest_t,
            n_iter=30, lr_pos=5e-3, lr_dv_factor=1.0,
            lambda_cav=1.0, lambda_E=0.0,
        )
        L_gap_t = float((delta_x_t**2).sum() + (delta_dv_t**2).sum())
        L_gap_truth_per_frame.append(L_gap_t)
        state = solver_save_state(solver)
    print(f"  L_gap(θ_true) per frame: " +
          "  ".join(f"{l:.4e}" for l in L_gap_truth_per_frame))
    print(f"  Sum: {sum(L_gap_truth_per_frame):.4e}")
    print(f"  (Day 7 Step 5 broken GD baseline noise: 0.84 sum, now {sum(L_gap_truth_per_frame):.4e})")

    # ============ Multi-epoch online ReSECT with stable GD ============
    print(f"\n[ReSECT] running {N_epochs} epochs of T={T} frames each …")
    theta_torch = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                 requires_grad=True)
    opt = torch.optim.Adam([theta_torch], lr=lr)
    trajectory = [theta_torch.detach().cpu().numpy().copy()]
    epoch_end_theta = [trajectory[0].copy()]
    L_gap_history = []

    print(f"  {'epoch':>5s}  {'L_gap_mean':>11s}  "
          f"{'α_min':>7s}  {'sharp':>7s}  {'τ':>7s}  "
          f"{'err α':>7s}  {'err s':>7s}  {'err τ':>7s}")

    t0 = time.time()
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
                print(f"  epoch {epoch} frame {t+1}: NaN — abort")
                return
            opt.step()
            clamp_theta_inplace(theta_torch)
            trajectory.append(theta_torch.detach().cpu().numpy().copy())
            L_gaps_epoch.append(float(L_gap.item()))

            # Mode C state injection
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

        if epoch < 5 or epoch % 5 == 4 or epoch == N_epochs - 1:
            th = theta_torch.detach().cpu().numpy()
            errs = np.abs(th - theta_true) / np.maximum(np.abs(theta_true), 1e-12)
            print(f"  {epoch:>5d}  {L_gap_history[-1]:>11.4e}  "
                  f"{th[0]:>7.4f}  {th[1]:>7.4f}  {th[2]:>7.4f}  "
                  f"{errs[0]*100:>6.1f}%  {errs[1]*100:>6.1f}%  {errs[2]*100:>6.1f}%")

    elapsed = time.time() - t0
    print(f"\n[Timing] {N_epochs} × {T} frames in {elapsed:.1f}s")

    # ============ Verdict ============
    theta_final = epoch_end_theta[-1]
    theta_avg_last5 = np.mean(epoch_end_theta[-5:], axis=0)
    rel_err_final = np.abs(theta_final - theta_true) / np.maximum(np.abs(theta_true), 1e-12)
    rel_err_avg5 = np.abs(theta_avg_last5 - theta_true) / np.maximum(np.abs(theta_true), 1e-12)

    print(f"\n" + "=" * 70)
    print(f"Day 8 verdict — Day 7 Step 5 RE-RUN with stable GD inner loop")
    print(f"=" * 70)

    print(f"\n  Day 7 Step 5 v2 (broken inner GD):")
    print(f"    α_min err: 1.5%   sharp err: 22.0%   τ err: 3.0%")
    print(f"    L_gap last5 mean: 0.28 (per frame, = baseline noise floor)")
    print(f"\n  Day 8 (stable inner GD, fixed lr_dv=1/s²):")
    for d, name in enumerate(param_names):
        print(f"    {name:>10s}: final={theta_final[d]:.4f}  avg5={theta_avg_last5[d]:.4f}  "
              f"err {rel_err_final[d]*100:.1f}% (avg5 {rel_err_avg5[d]*100:.1f}%)")
    L_gap_last5_mean = np.mean(L_gap_history[-5:])
    print(f"    L_gap last5 mean: {L_gap_last5_mean:.4e}")
    print(f"    L_gap(θ_true) per frame: {np.mean(L_gap_truth_per_frame):.4e}  (clean baseline)")

    np.savez(REPO_ROOT / "tools" / "day8_step5_rerun_trajectory.npz",
             trajectory=np.array(trajectory),
             epoch_end_theta=np.array(epoch_end_theta),
             L_gap_history=np.array(L_gap_history),
             L_gap_truth_per_frame=np.array(L_gap_truth_per_frame),
             theta_true=theta_true, theta_init=theta_init)


if __name__ == "__main__":
    main()
