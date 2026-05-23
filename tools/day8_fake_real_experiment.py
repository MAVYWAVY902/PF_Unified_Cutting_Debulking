"""Day 8 fake real experiment: comprehensive Mode C vs Mode D under realistic
phantom-like distortions.

4 types of distortion (mimicking real phantom):
  1. Surface-only obs (1000 markers, 5% of N)              — camera tracking
  2. Gaussian noise (σ_pos = 0.5mm, σ_α = 0.05)            — sensor noise
  3. Registration offset (+1mm in x)                       — frame alignment
  4. Extra viscous damping post-step (vel *= 0.85)         — viscoelastic gap

Critical: distortion #4 creates a TRUE modeling gap — ReSECT sim has NO
post-step viscous damping, so sim(any θ) ≠ obs.

Test: 2 configs, Mode D vs Mode C, both with full distortions.

Expected outcome (paper-defining):
  - If Mode C significantly better: state injection critical under modeling gap
  - If similar: Mode C not needed even with gap
  - If Mode D better: Mode C harmful (very unlikely under gap)

This is the experiment that DETERMINES whether Mode C earns its place in
the architecture for ICRA paper §3 narrative.
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
from tools.day8_T_surface import compute_surface_mask, subsample_mask  # noqa: E402
from tools.day6f_v0_multistep_synth import (  # noqa: E402
    make_knife_trajectory, clamp_theta_inplace,
)


def gen_fake_real_obs(solver, rest_state, theta_true, knife_traj, knife_radius,
                       device,
                       reg_offset_x=0.001,      # 1mm registration error
                       viscous_factor=0.85,     # post-step vel damping
                       sigma_pos=5e-4,           # 0.5mm sensor noise
                       sigma_alpha=0.05,         # cavity uncertainty
                       seed=42):
    """Generate 'fake real' obs with 4 distortions.

    1. Registration offset: start cube shifted +1mm in x
    2. Viscous damping: after each step, vel *= viscous_factor (PBD doesn't model)
    3. Position noise: Gaussian σ=sigma_pos
    4. Alpha noise: Gaussian σ=sigma_alpha (clipped to [0,1])
    """
    # Shift initial state to simulate registration error
    offset_rest = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in rest_state.items()}
    offset_rest["pos"] = offset_rest["pos"] + np.array([reg_offset_x, 0, 0], dtype=np.float32)

    theta_torch = torch.tensor(theta_true, dtype=torch.float32, device=device,
                                 requires_grad=False)
    rng = np.random.default_rng(seed)

    obs_pos = []
    obs_alpha = []
    state_in = offset_rest
    with torch.no_grad():
        for t, knife_pos in enumerate(knife_traj):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, knife_pos, knife_radius,
            )
            # Save state to apply post-step modifications
            state_after = solver_save_state(solver)
            # Modeling gap: extra viscous damping (NOT in ReSECT sim)
            state_after["vel"] = state_after["vel"] * viscous_factor

            # Get clean pos, alpha
            pos_clean = pos_t.cpu().numpy()
            alpha_clean = alpha_t.cpu().numpy()
            # Add sensor noise
            noise_pos = rng.normal(0, sigma_pos, pos_clean.shape).astype(np.float32)
            noise_alpha = rng.normal(0, sigma_alpha, alpha_clean.shape).astype(np.float32)
            pos_noisy = pos_clean + noise_pos
            alpha_noisy = np.clip(alpha_clean + noise_alpha, 0.0, 1.0)
            obs_pos.append(pos_noisy)
            obs_alpha.append(alpha_noisy)
            state_in = state_after

    return obs_pos, obs_alpha


def gd_inner_loop_masked(
    pos_pred, dv_pred, pos_target, alpha_target, theta,
    obs_mask_t,
    n_iter=30, lr_pos=5e-3, lr_dv_factor=1.0,
    lambda_cav=1.0,
):
    """Inner loop with surface mask (re-import from day8_T_surface concept)."""
    N = pos_pred.shape[0]
    s = theta[1]; tau = theta[2]
    lr_dv = lr_dv_factor / (s.detach() ** 2)
    delta_x = torch.zeros((N, 3), dtype=pos_pred.dtype, device=pos_pred.device,
                            requires_grad=True)
    delta_dv = torch.zeros(N, dtype=pos_pred.dtype, device=pos_pred.device,
                              requires_grad=True)
    mask_3d = obs_mask_t.float().unsqueeze(-1)
    mask_1d = obs_mask_t.float()
    L_init = None
    for k in range(n_iter):
        pos_corr = pos_pred + delta_x
        dv_corr = torch.relu(dv_pred + delta_dv)
        alpha_corr = torch.sigmoid(-s * (dv_corr - tau))
        diff_pos = (pos_corr - pos_target) * mask_3d
        diff_alpha = (alpha_corr - alpha_target) * mask_1d
        L_pos = (diff_pos ** 2).sum()
        L_cav = (diff_alpha ** 2).sum()
        L_inner = L_pos + lambda_cav * L_cav
        if L_init is None:
            L_init = float(L_inner)
        grad_x, grad_dv = torch.autograd.grad(
            L_inner, (delta_x, delta_dv),
            create_graph=True, retain_graph=True,
        )
        delta_x = delta_x - lr_pos * grad_x
        delta_dv = delta_dv - lr_dv * grad_dv
    return delta_x, delta_dv, dict(L_init=L_init, L_final=float(L_inner))


def run_resect_fake_real(solver, rest_state, knife_traj, knife_radius,
                          obs_pos_t, obs_alpha_t, obs_mask_t,
                          theta_init, theta_true, device,
                          N_epochs, lr, mode_C_enabled, n_inner=30):
    """ReSECT with masked inner + toggleable Mode C."""
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
            delta_x, delta_dv, info = gd_inner_loop_masked(
                pos_t, dv_t, obs_pos_t[t], obs_alpha_t[t],
                theta_torch, obs_mask_t,
                n_iter=n_inner, lr_pos=5e-3, lr_dv_factor=1.0,
                lambda_cav=1.0,
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
            if mode_C_enabled:
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

    # Surface mask (1000 markers)
    full_surface = compute_surface_mask(mesh_info)
    surface_mask_np = subsample_mask(full_surface, 1000, rng_seed=42)
    n_obs = int(surface_mask_np.sum())
    obs_mask_t = torch.tensor(surface_mask_np, dtype=torch.bool, device=device)

    print(f"\n[Setup] θ_true={theta_true.tolist()}  θ_init={theta_init.tolist()}")
    print(f"  T={T}, N_obs_markers={n_obs} ({n_obs/solver.N*100:.1f}% of N)")
    print(f"  Distortions:")
    print(f"    1. Surface mask: {n_obs}/{solver.N} = {n_obs/solver.N*100:.1f}%")
    print(f"    2. Reg offset: +1mm x")
    print(f"    3. Viscous damping: vel *= 0.85 post-step (MODELING GAP)")
    print(f"    4. Sensor noise: σ_pos=0.5mm, σ_α=0.05")

    # Generate fake real obs
    print(f"\n[Setup] generating fake real obs …")
    obs_pos, obs_alpha = gen_fake_real_obs(
        solver, rest_state, theta_true, knife_traj, knife_radius, device,
        reg_offset_x=0.001, viscous_factor=0.85,
        sigma_pos=5e-4, sigma_alpha=0.05, seed=42)

    # Sanity print
    for t in range(T):
        cf = float((obs_alpha[t] < 0.5).mean())
        cf_surface = float((obs_alpha[t][surface_mask_np] < 0.5).mean())
        print(f"  frame {t+1}: cavity_frac={cf:.4f} (full), {cf_surface:.4f} (surface)")

    obs_pos_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]

    # Sanity: compute "best-possible" L_gap given the modeling gap
    # by running ReSECT sim at θ_true (no distortion) vs distorted obs
    print(f"\n[Sanity] gap at θ_true (clean sim vs fake real obs):")
    theta_truth_t = torch.tensor(theta_true, dtype=torch.float32, device=device,
                                   requires_grad=False)
    state = rest_state
    L_gap_truth_per_frame = []
    for t in range(T):
        with torch.no_grad():
            alpha_st, pos_st, dv_st = WarpFullForwardOp.apply(
                theta_truth_t, solver, state, knife_traj[t], knife_radius,
            )
        delta_x_t, delta_dv_t, _ = gd_inner_loop_masked(
            pos_st.detach(), dv_st.detach(), obs_pos_t[t], obs_alpha_t[t],
            theta_truth_t, obs_mask_t,
            n_iter=30, lr_pos=5e-3, lr_dv_factor=1.0, lambda_cav=1.0,
        )
        L_gap_t = float((delta_x_t**2).sum() + (delta_dv_t**2).sum())
        L_gap_truth_per_frame.append(L_gap_t)
        state = solver_save_state(solver)
    print(f"  L_gap(θ_true) per frame: " +
          "  ".join(f"{l:.4e}" for l in L_gap_truth_per_frame))
    print(f"  sum: {sum(L_gap_truth_per_frame):.4e}   ← non-zero = irreducible modeling gap")

    # Run configs
    N_epochs = 100
    lr = 0.02
    configs = [
        ("Mode D (no inject)",  False),
        ("Mode C (inject)",     True),
    ]
    results = {}

    for label, mode_C in configs:
        print(f"\n[{label}] running {N_epochs} epochs …")
        t0 = time.time()
        traj, L_gap_hist, err = run_resect_fake_real(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, obs_mask_t,
            theta_init, theta_true, device,
            N_epochs=N_epochs, lr=lr, mode_C_enabled=mode_C, n_inner=30)
        elapsed = time.time() - t0
        if err:
            print(f"  ✗ {err}")
            results[label] = None
            continue
        theta_avg10 = np.mean(traj[-10:], axis=0)
        errs_avg10 = np.abs(theta_avg10 - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        L_gap_last10 = float(np.mean(L_gap_hist[-10:]))
        print(f"  {elapsed:.1f}s. avg10 θ=({theta_avg10[0]:.4f},{theta_avg10[1]:.4f},"
              f"{theta_avg10[2]:.4f})  err=({errs_avg10[0]:.2f}%,{errs_avg10[1]:.2f}%,"
              f"{errs_avg10[2]:.2f}%)  L_gap_last10={L_gap_last10:.4e}")
        # Checkpoint print
        for e in [10, 30, 60, 99]:
            if e < len(traj) - 1:
                th = traj[e+1]
                errs_e = np.abs(th - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
                print(f"  ep {e+1:>3d}: ({th[0]:.4f},{th[1]:.4f},{th[2]:.4f})  "
                      f"err=({errs_e[0]:5.2f}%,{errs_e[1]:5.2f}%,{errs_e[2]:5.2f}%)  "
                      f"L_gap={L_gap_hist[e]:.3e}")
        results[label] = dict(traj=traj, L_gap_hist=L_gap_hist,
                              theta_avg10=theta_avg10, errs_avg10=errs_avg10,
                              L_gap_last10=L_gap_last10)

    # Verdict
    print(f"\n" + "=" * 70)
    print(f"Fake real experiment verdict — Mode C vs D under modeling gap")
    print(f"=" * 70)
    print(f"\n  Irreducible L_gap (at θ_true, due to modeling gap): "
          f"{sum(L_gap_truth_per_frame):.4e}")

    if all(results.values()):
        d_errs = results["Mode D (no inject)"]["errs_avg10"]
        c_errs = results["Mode C (inject)"]["errs_avg10"]
        d_L = results["Mode D (no inject)"]["L_gap_last10"]
        c_L = results["Mode C (inject)"]["L_gap_last10"]
        print(f"\n  Mode D (no inject): "
              f"α={d_errs[0]:.2f}%  s={d_errs[1]:.2f}%  τ={d_errs[2]:.2f}%  "
              f"L_gap={d_L:.3e}")
        print(f"  Mode C (inject):    "
              f"α={c_errs[0]:.2f}%  s={c_errs[1]:.2f}%  τ={c_errs[2]:.2f}%  "
              f"L_gap={c_L:.3e}")
        print(f"  Improvement (D - C, +ve = C better):  "
              f"α={d_errs[0]-c_errs[0]:+.2f}%  s={d_errs[1]-c_errs[1]:+.2f}%  "
              f"τ={d_errs[2]-c_errs[2]:+.2f}%")

        max_d = float(np.max(d_errs)); max_c = float(np.max(c_errs))
        if max_c < max_d * 0.5:
            print(f"\n  ✓✓ Mode C SIGNIFICANTLY BETTER (max err {max_c:.2f}% vs {max_d:.2f}%)")
            print(f"     Paper §3 supports: 'Mode C critical under modeling gap'")
        elif max_c < max_d * 0.9:
            print(f"\n  ✓ Mode C moderately better")
            print(f"     Paper §3 nuance: 'Mode C improves recovery under modeling gap'")
        elif max_c < max_d * 1.1:
            print(f"\n  ⚠ Mode C ≈ Mode D under modeling gap (no clear winner)")
            print(f"     Paper §3 should not claim Mode C critical")
        else:
            print(f"\n  ✗ Mode C WORSE under modeling gap (unexpected!)")
            print(f"     Mode C consistently harmful — drop from architecture")

    np.savez(REPO_ROOT / "tools" / "day8_fake_real_experiment.npz",
             theta_true=theta_true, theta_init=theta_init,
             L_gap_truth=np.array(L_gap_truth_per_frame),
             **{f"{k}_traj": v['traj'] for k, v in results.items() if v},
             **{f"{k}_lgap": v['L_gap_hist'] for k, v in results.items() if v})


if __name__ == "__main__":
    main()
