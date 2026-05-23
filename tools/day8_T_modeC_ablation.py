"""Day 8 audit Mode C ablation: does state injection actually matter?

Mode C: between frames, inject inner-loop Δ into sim state
  state_{t+1}.pos    += Δ_x
  state_{t+1}.damage  = max(0, state.damage + Δ_dv)
  state_{t+1}.alpha   = σ(-s(damage_new - τ))

Architecture claim: Mode C is critical to ReSECT — without it, sim state
drifts from real (synth obs) and gradient quality degrades.

Test: run identical 100-epoch ReSECT with/without Mode C inject. Compare:
  - Final θ recovery
  - L_gap trajectory
  - Convergence rate

Pass criteria (validates Mode C):
  - WITHOUT Mode C: significantly worse recovery OR divergence
  - WITH Mode C: standard recovery

If both work equally well:
  → Mode C is decoration, paper §3 narrative needs revision
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
from tools.day8_step5_rerun_stable_gd import gen_obs_multistep_param_arr  # noqa: E402
from tools.day6f_v0_multistep_synth import (  # noqa: E402
    make_knife_trajectory, clamp_theta_inplace,
)


def run_resect_with_modeC(solver, rest_state, knife_traj, knife_radius,
                            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
                            theta_init, theta_true, device,
                            N_epochs, lr, n_inner=30,
                            mode_C_enabled=True):
    """ReSECT loop with mode_C_enabled toggle."""
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

            # State propagation: with or without Mode C injection
            state = solver_save_state(solver)
            if mode_C_enabled:
                dx_np = delta_x.detach().cpu().numpy()
                ddv_np = delta_dv.detach().cpu().numpy()
                state["pos"] = state["pos"] + dx_np
                state["particle_damage"] = np.maximum(state["particle_damage"] + ddv_np, 0.0)
                s_val = float(theta_torch[1].item())
                tau_val = float(theta_torch[2].item())
                new_dv = state["particle_damage"]
                state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(s_val * (new_dv - tau_val), -50, 50)))
            # else: state is raw sim state — no Δ injection

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

    N_epochs = 100
    lr = 0.02

    print(f"\n[Setup] θ_true={theta_true.tolist()}  θ_init={theta_init.tolist()}")
    print(f"  N_epochs={N_epochs}, lr={lr}, T={T}")

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

    # Run A: Mode C ENABLED (baseline)
    print(f"\n[A] Mode C ENABLED (state injection) …")
    t0 = time.time()
    traj_A, L_gap_A, err_A = run_resect_with_modeC(
        solver, rest_state, knife_traj, knife_radius,
        obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
        theta_init, theta_true, device,
        N_epochs=N_epochs, lr=lr, n_inner=30,
        mode_C_enabled=True)
    elapsed_A = time.time() - t0

    # Run B: Mode C DISABLED (sim-only state propagation)
    print(f"\n[B] Mode C DISABLED (no state injection) …")
    t0 = time.time()
    traj_B, L_gap_B, err_B = run_resect_with_modeC(
        solver, rest_state, knife_traj, knife_radius,
        obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
        theta_init, theta_true, device,
        N_epochs=N_epochs, lr=lr, n_inner=30,
        mode_C_enabled=False)
    elapsed_B = time.time() - t0

    print(f"\n  Run A (Mode C ON):  {elapsed_A:.1f}s")
    print(f"  Run B (Mode C OFF): {elapsed_B:.1f}s")

    # Verdict
    print(f"\n" + "=" * 70)
    print(f"Mode C ablation verdict")
    print(f"=" * 70)

    if err_A or err_B:
        print(f"  ✗ Run errors: A={err_A}  B={err_B}")
        return

    theta_A = np.mean(traj_A[-10:], axis=0)
    theta_B = np.mean(traj_B[-10:], axis=0)
    errs_A = np.abs(theta_A - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
    errs_B = np.abs(theta_B - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
    L_gap_A_last = float(np.mean(L_gap_A[-10:]))
    L_gap_B_last = float(np.mean(L_gap_B[-10:]))

    print(f"\n  avg10 last θ recovery:")
    print(f"    {'param':>10s}  {'truth':>7s}  {'A (ON)':>10s}  {'B (OFF)':>10s}  {'err A':>7s}  {'err B':>7s}")
    for d, name in enumerate(param_names):
        print(f"    {name:>10s}  {theta_true[d]:>7.4f}  {theta_A[d]:>10.4f}  {theta_B[d]:>10.4f}  "
              f"{errs_A[d]:>6.2f}%  {errs_B[d]:>6.2f}%")
    print(f"\n  L_gap last10 mean:  A={L_gap_A_last:.4e}  B={L_gap_B_last:.4e}  "
          f"ratio (B/A)={L_gap_B_last/max(L_gap_A_last,1e-12):.2f}×")

    # Per-frame L_gap evolution comparison
    print(f"\n  L_gap trajectory (sample every 10 epochs):")
    print(f"    {'epoch':>6s}  {'A (ON)':>12s}  {'B (OFF)':>12s}  {'ratio':>7s}")
    for e in [0, 10, 20, 50, 99]:
        if e < len(L_gap_A):
            ratio = L_gap_B[e] / max(L_gap_A[e], 1e-12)
            print(f"    {e:>6d}  {L_gap_A[e]:>12.4e}  {L_gap_B[e]:>12.4e}  {ratio:>6.2f}×")

    # Per-dim convergence speed
    print(f"\n  θ trajectory checkpoints:")
    for e in [10, 30, 60, 99]:
        if e < len(traj_A) - 1:
            tA = traj_A[e+1]; tB = traj_B[e+1]
            print(f"    ep {e+1:>3d}: A=({tA[0]:.3f},{tA[1]:.3f},{tA[2]:.3f})  "
                  f"B=({tB[0]:.3f},{tB[1]:.3f},{tB[2]:.3f})")

    # Conclusion
    max_err_A = float(np.max(errs_A))
    max_err_B = float(np.max(errs_B))
    diff_avg = float(np.mean(np.abs(theta_A - theta_B)))

    print(f"\n  Summary:")
    print(f"    max err A (ON):  {max_err_A:.2f}%")
    print(f"    max err B (OFF): {max_err_B:.2f}%")
    print(f"    avg |θ_A - θ_B|: {diff_avg:.4f}")

    if max_err_B < max_err_A * 1.2 and max_err_B < 15:
        print(f"\n  ⚠ Mode C provides NO significant benefit on this setup")
        print(f"     OFF (B) recovery comparable to ON (A) — paper §3 narrative needs revision")
        print(f"     'Mode C is critical' claim NOT supported")
    elif max_err_B > max_err_A * 2 or max_err_B > 30:
        print(f"\n  ✓ Mode C IS critical — OFF significantly worse")
        print(f"     Paper §3 narrative supported: state injection essential")
    else:
        print(f"\n  ⚠ Mode C helps but not dramatically (~{max_err_B/max_err_A:.1f}× difference)")
        print(f"     Paper §3 narrative requires nuance: 'Mode C improves convergence'")

    np.savez(REPO_ROOT / "tools" / "day8_T_modeC_ablation.npz",
             traj_A=traj_A, traj_B=traj_B,
             L_gap_A=L_gap_A, L_gap_B=L_gap_B,
             theta_true=theta_true, theta_init=theta_init)


if __name__ == "__main__":
    main()
