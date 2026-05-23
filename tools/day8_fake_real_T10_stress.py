"""Day 8 fake real T=10 stress test: extreme conditions to expose Mode C value.

Pushes T and modeling gap to extreme:
  T=10              (was 3 in original fake_real, now 9 chances for drift)
  viscous=0.5       (was 0.85, now 50% velocity damping per step — strong gap)
  Same sparse+noise+reg as before

Goal: differentiate Mode C from Mode D under conditions where their
anti-drift behavior should diverge meaningfully.

Hypothesis:
  T=3, viscous=0.85: similar (drift hasn't accumulated)
  T=10, viscous=0.5: Mode C significantly better (Mode D drift dominates)

If hypothesis confirmed: Mode C is essential for real long-T deployment.
If still similar: Mode C truly not needed, drop it.
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
from tools.day8_T_surface import compute_surface_mask, subsample_mask  # noqa: E402
from tools.day8_fake_real_experiment import (  # noqa: E402
    gen_fake_real_obs, gd_inner_loop_masked, run_resect_fake_real,
)


def make_knife_traj_T(T):
    """Knife sweep along x for T frames."""
    x_positions = np.linspace(0.12, 0.18, T)
    return [np.array([x, 0.30, 0.15], dtype=np.float32) for x in x_positions]


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_radius = 0.04
    T = 10                       # ← key change
    viscous_factor = 0.5         # ← key change (strong)
    knife_traj = make_knife_traj_T(T)
    theta_true = np.array([3.0, 20.0, 0.5], dtype=np.float32)
    theta_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)
    param_names = ["α_min", "sharpness", "τ"]

    # Surface mask (1000 markers)
    full_surface = compute_surface_mask(mesh_info)
    surface_mask_np = subsample_mask(full_surface, 1000, rng_seed=42)
    n_obs = int(surface_mask_np.sum())
    obs_mask_t = torch.tensor(surface_mask_np, dtype=torch.bool, device=device)

    print(f"\n[Stress test setup]")
    print(f"  T={T}  (9 between-frame state propagations, vs T=3 had 2)")
    print(f"  viscous_factor={viscous_factor}  (strong damping gap)")
    print(f"  knife sweep along x: 0.12 → 0.18 over {T} frames")
    print(f"  θ_init={theta_init.tolist()}  θ_true={theta_true.tolist()}")

    # Generate fake real obs with extreme distortions
    print(f"\n[Setup] generating T={T} fake real obs with viscous={viscous_factor} …")
    t0 = time.time()
    obs_pos, obs_alpha = gen_fake_real_obs(
        solver, rest_state, theta_true, knife_traj, knife_radius, device,
        reg_offset_x=0.001, viscous_factor=viscous_factor,
        sigma_pos=5e-4, sigma_alpha=0.05, seed=42)
    print(f"  obs gen done in {time.time()-t0:.1f}s")

    # Track cavity growth
    print(f"  Cavity growth across {T} frames:")
    for t in range(T):
        cf = float((obs_alpha[t] < 0.5).mean())
        cf_surf = float((obs_alpha[t][surface_mask_np] < 0.5).mean())
        print(f"    frame {t+1:>2d}: total {cf:.4f}  surface {cf_surf:.4f}")

    obs_pos_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]

    # Sanity at θ_true (irreducible gap)
    print(f"\n[Sanity] L_gap at θ_true (irreducible due to modeling gap) …")
    theta_truth_t = torch.tensor(theta_true, dtype=torch.float32, device=device,
                                   requires_grad=False)
    state = rest_state
    L_gap_truth = 0.0
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
        L_gap_truth += L_gap_t
        state = solver_save_state(solver)
    print(f"  Σ L_gap(θ_true) over {T} frames = {L_gap_truth:.4e}")
    print(f"  Per-frame avg = {L_gap_truth/T:.4e}")

    # Run configs
    N_epochs = 100
    lr = 0.02
    configs = [
        ("Mode D (no inject)", False),
        ("Mode C (inject)",    True),
    ]
    results = {}

    for label, mode_C in configs:
        print(f"\n[{label}] T={T}, {N_epochs} epochs …")
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
              f"{errs_avg10[2]:.2f}%)")
        print(f"  L_gap last10: {L_gap_last10:.4e}  (irreducible: {L_gap_truth:.4e})")
        # Checkpoints
        for e in [10, 30, 60, 99]:
            if e < len(traj) - 1:
                th = traj[e+1]
                errs_e = np.abs(th - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
                print(f"    ep {e+1:>3d}: ({th[0]:.4f},{th[1]:.4f},{th[2]:.4f})  "
                      f"err=({errs_e[0]:5.2f}%,{errs_e[1]:5.2f}%,{errs_e[2]:5.2f}%)  "
                      f"L_gap={L_gap_hist[e]:.3e}")
        results[label] = dict(traj=traj, L_gap_hist=L_gap_hist,
                              theta_avg10=theta_avg10, errs_avg10=errs_avg10,
                              L_gap_last10=L_gap_last10)

    # Verdict
    print(f"\n" + "=" * 70)
    print(f"T={T} stress verdict — Mode C value under EXTREME modeling gap")
    print(f"=" * 70)

    if all(results.values()):
        d_errs = results["Mode D (no inject)"]["errs_avg10"]
        c_errs = results["Mode C (inject)"]["errs_avg10"]
        d_L = results["Mode D (no inject)"]["L_gap_last10"]
        c_L = results["Mode C (inject)"]["L_gap_last10"]
        print(f"\n  T={T}, viscous={viscous_factor}, irreducible L_gap={L_gap_truth:.3e}")
        print(f"\n  Mode D (no inject): "
              f"α={d_errs[0]:.2f}%  s={d_errs[1]:.2f}%  τ={d_errs[2]:.2f}%  "
              f"L_gap={d_L:.3e}")
        print(f"  Mode C (inject):    "
              f"α={c_errs[0]:.2f}%  s={c_errs[1]:.2f}%  τ={c_errs[2]:.2f}%  "
              f"L_gap={c_L:.3e}")
        improvement = d_errs - c_errs
        print(f"  D - C (+=C better): "
              f"α={improvement[0]:+.2f}%  s={improvement[1]:+.2f}%  τ={improvement[2]:+.2f}%")

        max_d = float(np.max(d_errs)); max_c = float(np.max(c_errs))
        ratio = max_c / max(max_d, 1e-6)
        if ratio < 0.5:
            print(f"\n  ✓✓ Mode C SIGNIFICANTLY BETTER under stress "
                  f"({max_c:.1f}% vs {max_d:.1f}%, ratio {ratio:.2f})")
            print(f"     → Mode C earns its place for long-T / strong-gap scenarios")
            print(f"     → Paper §3 narrative: 'Mode C critical for T>10 or large gap'")
        elif ratio < 0.85:
            print(f"\n  ✓ Mode C moderately better under stress (ratio {ratio:.2f})")
            print(f"     → Paper §3 nuance: 'Mode C improves long-T recovery'")
        elif ratio < 1.15:
            print(f"\n  ⚠ Mode C ≈ Mode D even under stress (ratio {ratio:.2f})")
            print(f"     → Both modes handle this regime similarly")
            print(f"     → Paper §3: Mode C still not justified at this gap level")
        else:
            print(f"\n  ✗ Mode C WORSE even under stress (ratio {ratio:.2f})")
            print(f"     → Drop Mode C from architecture")

    np.savez(REPO_ROOT / "tools" / "day8_fake_real_T10_stress.npz",
             T=T, viscous_factor=viscous_factor,
             theta_true=theta_true, theta_init=theta_init,
             L_gap_truth=L_gap_truth,
             **{f"{k}_traj": v['traj'] for k, v in results.items() if v},
             **{f"{k}_lgap": v['L_gap_hist'] for k, v in results.items() if v})


if __name__ == "__main__":
    main()
