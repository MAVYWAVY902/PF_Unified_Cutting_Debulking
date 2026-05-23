"""Day 8 audit T_surface: CRITICAL portability test — sparse surface-only obs.

Real phantom: camera tracking gives only surface particles (~100-500 markers).
We can NOT observe interior alpha or position.

Test: mask obs to surface-only, see if ReSECT still recovers θ.

3 sparsity levels tested:
  Full surface ~29%  (all bbox-surface vertices, 6375/21745)
  Medium ~5%         (subsample 1000 markers, realistic camera mesh)
  Sparse ~1%         (200 markers, realistic OptiTrack markers)

Inner loop change: L_pos and L_cav both multiplied by surface_mask.
Interior particles have 0 gradient → Δ_x = Δ_dv = 0 there.
Mode C injection only touches surface (rest of Δ is naturally 0).

Pass criteria:
  - Full surface: < 5% err on all 3 params (relaxed from full obs)
  - Medium 5%: < 10% err
  - Sparse 1%: < 20% err (acceptable, if even this works it's strong)
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
from tools.day8_step5_rerun_stable_gd import gen_obs_multistep_param_arr  # noqa: E402
from tools.day6f_v0_multistep_synth import (  # noqa: E402
    make_knife_trajectory, clamp_theta_inplace,
)


def compute_surface_mask(mesh_info, eps=1e-3):
    """Identify particles on bbox surface."""
    v = mesh_info["vertices"]
    bmin = v.min(axis=0); bmax = v.max(axis=0)
    mask = ((np.abs(v[:, 0] - bmin[0]) < eps) | (np.abs(v[:, 0] - bmax[0]) < eps) |
            (np.abs(v[:, 1] - bmin[1]) < eps) | (np.abs(v[:, 1] - bmax[1]) < eps) |
            (np.abs(v[:, 2] - bmin[2]) < eps) | (np.abs(v[:, 2] - bmax[2]) < eps))
    return mask


def subsample_mask(full_mask, n_target, rng_seed=42):
    """Subsample True elements of mask to n_target count."""
    rng = np.random.default_rng(rng_seed)
    indices = np.where(full_mask)[0]
    if n_target >= len(indices):
        return full_mask.copy()
    selected = rng.choice(indices, size=n_target, replace=False)
    new_mask = np.zeros_like(full_mask)
    new_mask[selected] = True
    return new_mask


def gd_inner_loop_masked(
    pos_pred, dv_pred, pos_target, alpha_target, theta,
    obs_mask_t,
    n_iter=30, lr_pos=5e-3, lr_dv_factor=1.0,
    lambda_cav=1.0,
):
    """Inner loop with surface obs masking.

    obs_mask_t: (N,) bool tensor — True for observable particles.
    L_pos and L_cav contributions are zeroed for unobservable particles.
    Δ at unobservable particles stays 0 (no gradient → no update).
    """
    N = pos_pred.shape[0]
    s = theta[1]; tau = theta[2]
    lr_dv = lr_dv_factor / (s.detach() ** 2)

    delta_x = torch.zeros((N, 3), dtype=pos_pred.dtype, device=pos_pred.device,
                            requires_grad=True)
    delta_dv = torch.zeros(N, dtype=pos_pred.dtype, device=pos_pred.device,
                              requires_grad=True)

    mask_3d = obs_mask_t.float().unsqueeze(-1)  # (N, 1) for pos
    mask_1d = obs_mask_t.float()                 # (N,) for alpha

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


def run_resect_masked(solver, rest_state, knife_traj, knife_radius,
                       obs_pos_t, obs_alpha_t, obs_mask_t,
                       theta_init, theta_true, device,
                       N_epochs, lr, n_inner=30):
    """ReSECT loop with surface obs masking."""
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

    # Build masks
    full_surface_mask = compute_surface_mask(mesh_info)
    full_mask = np.ones(solver.N, dtype=bool)  # all-obs baseline

    sparsity_configs = [
        ("full",            full_mask,                                          solver.N),
        ("surface_all",     full_surface_mask,                                  int(full_surface_mask.sum())),
        ("surface_1000",    subsample_mask(full_surface_mask, 1000),            1000),
        ("surface_200",     subsample_mask(full_surface_mask, 200),             200),
    ]

    print(f"\n[Setup] θ_true={theta_true.tolist()}  θ_init={theta_init.tolist()}")
    print(f"  Total particles: {solver.N}")
    for label, mask, n in sparsity_configs:
        print(f"  {label:>15s}: {n} particles ({n/solver.N*100:.2f}%)")

    obs_pos, obs_alpha = gen_obs_multistep_param_arr(
        solver, rest_state, theta_true, knife_traj, knife_radius, device)
    obs_pos_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]

    N_epochs = 200
    lr = 0.02

    results = {}
    for label, mask, n in sparsity_configs:
        print(f"\n[{label}] {n} particles ({n/solver.N*100:.1f}%), {N_epochs} epochs, lr={lr}")
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
        t0 = time.time()
        traj, L_gap_hist, err = run_resect_masked(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, mask_t,
            theta_init, theta_true, device,
            N_epochs=N_epochs, lr=lr, n_inner=30)
        elapsed = time.time() - t0
        if err:
            print(f"  ✗ {err}")
            results[label] = dict(err=err)
            continue
        theta_avg10 = np.mean(traj[-10:], axis=0)
        errs_avg10 = np.abs(theta_avg10 - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        L_gap_last10 = float(np.mean(L_gap_hist[-10:]))
        print(f"  {elapsed:.1f}s. avg10 θ=({theta_avg10[0]:.4f},{theta_avg10[1]:.4f},"
              f"{theta_avg10[2]:.4f})  err=({errs_avg10[0]:.2f}%,{errs_avg10[1]:.2f}%,"
              f"{errs_avg10[2]:.2f}%)  L_gap={L_gap_last10:.4e}")
        results[label] = dict(theta_avg10=theta_avg10, errs_avg10=errs_avg10,
                              L_gap=L_gap_last10, n_obs=n)

    # Verdict
    print(f"\n" + "=" * 70)
    print(f"T_surface verdict — recovery vs obs sparsity")
    print(f"=" * 70)
    print(f"\n  {'config':>15s}  {'n_obs':>7s}  {'%N':>6s}  "
          f"{'α_err':>7s}  {'s_err':>7s}  {'τ_err':>7s}  {'L_gap':>10s}")
    for label, r in results.items():
        if r.get('err'):
            print(f"  {label:>15s}: {r['err']}")
            continue
        n_obs = r['n_obs']
        errs = r['errs_avg10']
        print(f"  {label:>15s}  {n_obs:>7d}  {n_obs/solver.N*100:>5.1f}%  "
              f"{errs[0]:>6.2f}%  {errs[1]:>6.2f}%  {errs[2]:>6.2f}%  "
              f"{r['L_gap']:.3e}")

    # Pass thresholds
    print(f"\n  Pass thresholds:")
    print(f"    full obs:        < 5% (regression check)")
    print(f"    surface_all 29%: < 5% (sparse but plenty)")
    print(f"    surface_1000 5%: < 10% (realistic camera tracking)")
    print(f"    surface_200 1%:  < 20% (challenging — OptiTrack-level)")

    # Save
    np.savez(REPO_ROOT / "tools" / "day8_T_surface.npz",
             theta_true=theta_true, theta_init=theta_init,
             **{label: r for label, r in results.items() if 'err' not in r})


if __name__ == "__main__":
    main()
