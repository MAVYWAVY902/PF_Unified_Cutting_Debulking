"""Day 7 Step 4: online ReSECT loop (T=3 frames, 1 Adam step per frame).

This is the FULL Liang Algorithm 1 pattern adapted for sys-ID:
  state = rest_state
  for t in 1..T:
      α_t, pos_t, dv_t = sim(θ, state, knife_t)         ← forward
      Δ_x_t, Δ_dv_t   = inner_loop(α_t, pos_t, dv_t, obs_t)  ← inner GD
      L_gap_t         = ‖Δ_x_t‖² + ‖Δ_dv_t‖²            ← outer loss
      L_gap_t.backward()                                  ← per-frame backward
      opt.step(θ)                                         ← per-frame θ update
      state ← save_state(solver)
      state.pos    += Δ_x_t                               ← Mode C inject
      state.damage  = max(0, state.damage + Δ_dv_t)       ← Mode C inject

Step 4 hypothesis: after T=3 frames (= 3 θ updates), θ should move TOWARD truth.

Step 4 is single-pass. Step 5 will do multi-epoch (cyclic) for full convergence.
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
from tools.day7_step1_lgap_landscape import simple_gd_inner_loop_relu  # noqa: E402
from tools.day6f_v0_multistep_synth import (  # noqa: E402
    make_knife_trajectory, gen_obs_multistep, clamp_theta_inplace,
)


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

    print(f"\n[Setup] T={T}  θ_true={theta_true.tolist()}  θ_init={theta_init.tolist()}")
    obs_pos, obs_alpha = gen_obs_multistep(
        solver, rest_state, theta_true, knife_traj, knife_radius)
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

    # ============ Online ReSECT loop ============
    print(f"\n[ReSECT] online loop, lr=0.10, n_inner=30, T={T}")
    theta_torch = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                 requires_grad=True)
    opt = torch.optim.Adam([theta_torch], lr=0.10)
    state = rest_state

    trajectory = [theta_torch.detach().cpu().numpy().copy()]
    L_gap_per_frame = []
    inner_red_per_frame = []
    cavity_corrected_per_frame = []

    print(f"\n  {'frame':>5s}  {'L_gap':>10s}  {'L_inner_red':>11s}  "
          f"{'α_min':>7s}  {'sharp':>7s}  {'τ':>7s}  "
          f"{'∇α':>10s}  {'∇s':>10s}  {'∇τ':>10s}")
    print(f"  {'init':>5s}  {'(start)':>10s}  {'':>11s}  "
          f"{theta_init[0]:>7.4f}  {theta_init[1]:>7.4f}  {theta_init[2]:>7.4f}")

    for t in range(T):
        # 1. Forward sim
        alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
            theta_torch, solver, state, knife_traj[t], knife_radius,
        )

        # 2. Inner loop (with create_graph=True → autograd back to θ)
        delta_x, delta_dv, info = simple_gd_inner_loop_relu(
            pos_t, dv_t, obs_pos_t[t], obs_alpha_t[t],
            theta_torch, edges_a_t, edges_b_t, edges_rest_t,
            n_iter=30, lr_pos=5e-3, lr_dv=0.1,
            lambda_cav=1.0, lambda_E=0.0,
        )

        # 3. L_gap = ‖Δ‖² (effective Δ_dv after clamp)
        # Note: for gradient flow, we use the un-clamped delta_dv in L_gap
        # so gradient flows through the inner loop unroll. The clamp only
        # applies to STATE INJECTION (next step).
        L_gap = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
        L_gap_val = float(L_gap.item())
        inner_red = info["L_init"] / max(info["L_final"], 1e-12)

        # 4. Per-frame Adam step
        opt.zero_grad()
        L_gap.backward()
        grad_vals = theta_torch.grad.detach().cpu().numpy().copy()
        if np.isnan(grad_vals).any():
            print(f"  frame {t+1}: NaN gradient — abort")
            return
        opt.step()
        clamp_theta_inplace(theta_torch)
        th = theta_torch.detach().cpu().numpy().copy()
        trajectory.append(th)
        L_gap_per_frame.append(L_gap_val)
        inner_red_per_frame.append(inner_red)

        # 5. Mode C: state injection
        state = solver_save_state(solver)
        dx_np = delta_x.detach().cpu().numpy()
        ddv_np = delta_dv.detach().cpu().numpy()
        state["pos"] = state["pos"] + dx_np
        # damage VALUE clamp ≥ 0
        state["particle_damage"] = np.maximum(state["particle_damage"] + ddv_np, 0.0)
        # Also update alpha from injected dv (consistent state)
        new_dv = state["particle_damage"]
        s_val = float(theta_torch[1].item())
        tau_val = float(theta_torch[2].item())
        state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(s_val * (new_dv - tau_val), -50, 50)))
        cf_corr = float((state["alpha"] < 0.5).mean())
        cavity_corrected_per_frame.append(cf_corr)

        print(f"  {t+1:>5d}  {L_gap_val:>10.4e}  {inner_red:>10.2f}×  "
              f"{th[0]:>7.4f}  {th[1]:>7.4f}  {th[2]:>7.4f}  "
              f"{grad_vals[0]:>+10.3e}  {grad_vals[1]:>+10.3e}  {grad_vals[2]:>+10.3e}")

    # ============ Verdict ============
    print("\n" + "=" * 70)
    print("Day 7 Step 4 verdict — single-pass online ReSECT")
    print("=" * 70)

    theta_final = trajectory[-1]
    rel_err_init = np.abs(theta_init - theta_true) / np.maximum(np.abs(theta_true), 1e-12)
    rel_err_final = np.abs(theta_final - theta_true) / np.maximum(np.abs(theta_true), 1e-12)
    moved = (rel_err_final < rel_err_init)

    print(f"\n  per-dim θ trajectory after T={T} frames (1 Adam step each):")
    for d, name in enumerate(param_names):
        print(f"    {name:>10s}: true={theta_true[d]:.4f}  init={theta_init[d]:.4f}  "
              f"final={theta_final[d]:.4f}  err {rel_err_init[d]*100:.1f}% → "
              f"{rel_err_final[d]*100:.1f}%  {'✓' if moved[d] else '✗'}")

    print(f"\n  L_gap per frame: " + "  ".join(f"{l:.3e}" for l in L_gap_per_frame))
    print(f"  Inner reductions: " + "  ".join(f"{r:.1f}×" for r in inner_red_per_frame))
    print(f"  Cavity_frac (corrected): " + "  ".join(f"{c:.4f}" for c in cavity_corrected_per_frame))
    print(f"  Cavity_frac (obs):       " +
          "  ".join(f"{float((obs_alpha[t]<0.5).mean()):.4f}" for t in range(T)))

    # Pass criteria
    n_moved = int(moved.sum())
    no_nan = not np.isnan(trajectory[-1]).any()
    print(f"\n  Checks:")
    print(f"    (1) No NaN in θ trajectory: {'✓' if no_nan else '✗'}")
    print(f"    (2) θ moved TOWARD truth on ≥2/3 dims: {n_moved}/3   "
          f"{'✓' if n_moved >= 2 else '✗'}")
    print(f"    (3) Inner loop converging ≥2× per frame:")
    inner_ok = all(r >= 2.0 for r in inner_red_per_frame)
    print(f"        per-frame inner reduce all ≥ 2×: {'✓' if inner_ok else '⚠'}")

    if no_nan and n_moved >= 2 and inner_ok:
        print(f"\n✓ Day 7 Step 4 PASS — online loop runs, θ moves toward truth")
    else:
        print(f"\n⚠ Day 7 Step 4 PARTIAL — check details above")

    # Save
    np.savez(REPO_ROOT / "tools" / "day7_step4_trajectory.npz",
             trajectory=np.array(trajectory),
             L_gap_per_frame=np.array(L_gap_per_frame),
             inner_red_per_frame=np.array(inner_red_per_frame),
             theta_true=theta_true, theta_init=theta_init)


if __name__ == "__main__":
    main()
