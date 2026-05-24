"""R3: 4D multi-init basin sweep — validates "reasonable" (robust to init).

5 inits covering close / baseline / far-low / far-high / extreme-stiff.
Each 50 epoch (~30 sec). Logs:
   - per-init final θ vs truth
   - convergence status per dim
   - basin success rate

Verdict:
   ≥ 4/5 converge < 10% per dim → robust basin, 4D sys-ID is reasonable
   2-3/5 → marginal, depends on warm-start
   ≤ 1/5 → ill-posed basin, 4D not reliable

Setup mirrors r4_sanity (sharpness frozen, stiffness FREE):
   θ_true = (3.0, 20.0, 0.5, 0.85)
   lr=0.02, n_inner=30, T=3, N_epochs=50
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


def run_4d_recovery(solver, rest_state, knife_traj, knife_radius,
                      obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t,
                      edges_rest_t, theta_init_4d, device,
                      N_epochs=50, lr=0.02, n_inner=30):
    T = len(knife_traj)
    theta_init_fixed = theta_init_4d.copy()
    theta_init_fixed[1] = SHARPNESS_FIXED
    theta_torch = torch.tensor(theta_init_fixed, dtype=torch.float32, device=device,
                                 requires_grad=True)
    opt = torch.optim.Adam([theta_torch], lr=lr)

    epoch_thetas = [theta_torch.detach().cpu().numpy().copy()]
    for epoch in range(N_epochs):
        state = rest_state
        for t in range(T):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state, knife_traj[t], knife_radius,
            )
            delta_x, delta_dv, _ = gd_inner_loop_stable(
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
        epoch_thetas.append(theta_torch.detach().cpu().numpy().copy())
    return np.array(epoch_thetas), None


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, _ = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_radius = 0.04
    T = 3
    knife_traj = make_knife_trajectory(T)
    theta_true = np.array([3.0, 20.0, 0.5, 0.85], dtype=np.float32)
    print(f"\n[Setup] θ_true_4D = {theta_true.tolist()}")
    print(f"        T={T}, N_epochs=50, lr=0.02")

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

    # 5 inits covering basin
    inits = [
        ("baseline",      np.array([5.0, 15.0, 0.60, 0.70], dtype=np.float32)),
        ("near-truth",    np.array([3.2, 22.0, 0.52, 0.82], dtype=np.float32)),
        ("all-low",       np.array([1.5, 10.0, 0.30, 0.40], dtype=np.float32)),
        ("all-high",      np.array([6.0, 30.0, 0.80, 1.30], dtype=np.float32)),
        ("stiff-extreme", np.array([4.0, 18.0, 0.55, 0.30], dtype=np.float32)),
    ]
    names_4d = ["α_min", "sharp(FROZEN)", "τ", "stiffness"]

    results = []
    for label, ti in inits:
        ti_err = np.abs(ti - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        ti_err_no_sharp = np.delete(ti_err, 1)
        print(f"\n[{label:>14s}] init=({ti[0]:.2f}, {ti[1]:.1f}, {ti[2]:.3f}, {ti[3]:.3f})  "
              f"init_err: α={ti_err[0]:.0f}% τ={ti_err[2]:.0f}% stiff={ti_err[3]:.0f}%")
        t0 = time.time()
        traj, err = run_4d_recovery(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
            ti, device, N_epochs=50, lr=0.02, n_inner=30)
        elapsed = time.time() - t0
        if err:
            print(f"   ✗ {err}")
            results.append(dict(label=label, status="error", err=err))
            continue
        avg10 = traj[-10:].mean(axis=0)
        errs = np.abs(avg10 - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        free_errs = np.array([errs[0], errs[2], errs[3]])  # α, τ, stiff
        all_under_10 = (free_errs < 10).all()
        all_under_20 = (free_errs < 20).all()
        status = "✓ tight (<10%)" if all_under_10 else ("⚠ loose (<20%)" if all_under_20 else "✗ failed")
        print(f"   {elapsed:.0f}s  final=({avg10[0]:.4f}, {avg10[1]:.1f}, {avg10[2]:.4f}, {avg10[3]:.4f})")
        print(f"   err: α={errs[0]:.2f}%  τ={errs[2]:.2f}%  stiff={errs[3]:.2f}%  {status}")
        results.append(dict(label=label, traj=traj, avg10=avg10, errs=errs,
                              all_under_10=all_under_10, all_under_20=all_under_20))

    # Summary table
    print(f"\n{'='*72}")
    print(f"BASIN SUMMARY (4D recovery, θ = α, τ, stiffness free)")
    print(f"{'='*72}")
    n_tight = sum(1 for r in results if r.get("all_under_10"))
    n_loose = sum(1 for r in results if r.get("all_under_20"))
    print(f"\n  {'init':<16s}  {'α err':>8s}  {'τ err':>8s}  {'stiff err':>10s}  status")
    for r in results:
        if "errs" not in r:
            print(f"  {r['label']:<16s}  (failed)")
            continue
        e = r["errs"]
        sym = "✓" if r["all_under_10"] else ("⚠" if r["all_under_20"] else "✗")
        print(f"  {r['label']:<16s}  {e[0]:7.2f}%  {e[2]:7.2f}%  {e[3]:9.2f}%   {sym}")

    print(f"\n  TIGHT (<10% all):  {n_tight}/5")
    print(f"  LOOSE (<20% all):  {n_loose}/5")

    # Verdict
    print(f"\n{'='*72}")
    print(f"R3 VERDICT")
    print(f"{'='*72}")
    if n_tight >= 4:
        print(f"  ✓ ROBUST BASIN ({n_tight}/5 tight)")
        print(f"    4D sys-ID is reasonable. P1.2-6 viable.")
        print(f"    → next: R5 noise robustness or P1.2 commit + paper write")
    elif n_tight >= 3:
        print(f"  ⚠ MARGINAL ({n_tight}/5 tight, {n_loose}/5 loose)")
        print(f"    Works from sensible inits but fragile from extreme.")
        print(f"    → Consider warm-start from Stage 2.1 indenter (gives close-to-truth init).")
    elif n_loose >= 3:
        print(f"  ⚠ WEAK ({n_tight}/5 tight, only {n_loose}/5 within 20%)")
        print(f"    Several inits fail tight criterion. Joint 4D fragile.")
    else:
        print(f"  ✗ FAILED BASIN ({n_tight}/5 tight, {n_loose}/5 loose)")
        print(f"    4D sys-ID unreliable. Need Stage 2.1 split or new observable.")

    out_npz = str(REPO_ROOT / "tools" / "r3_basin_results.npz")
    np.savez(out_npz, **{
        f"init_{i}_{r['label']}_traj": r.get("traj", np.array([]))
        for i, r in enumerate(results)
    })
    print(f"\n  trajectories saved → tools/r3_basin_results.npz")


if __name__ == "__main__":
    main()
