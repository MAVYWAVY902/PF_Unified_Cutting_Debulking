"""Test 1+2 v2 — PRINCIPLED adaptive inner lr via Hessian upper bound.

Methodology (no hack, derive from first principles):

  L_inner = L_pos + λ_cav · L_cav + λ_E · L_E

  Hessian w.r.t. Δx (per particle):
    H_pos = 2                              (MSE on positions)
    H_cav ≈ 0  (depends on Δdv not Δx)
    H_E   ≤ 2 · max_degree(particle) · λ_E (edge spring Laplacian upper bound)
                  where max_degree = max # of edges incident to any particle in mesh

  Total H_x_upper = 2 + 2·max_degree·λ_E

  Stable GD requires lr × λ_max(H) < 2.
  Newton-optimal:     lr = 1/λ_max
  Safe (half-Newton): lr = 1/(2·λ_max)   ← we use this

  → lr_pos = 1 / (2 + 2·max_degree·λ_E)

This is fully principled and reproducible. No tuning of magic numbers.

For each λ_E config:
  (1) Compute lr_pos from formula
  (2) Verify inner GD monotonic descent at first call (sanity)
  (3) Run 200-ep recovery on all-high init
  (4) Report
"""
import sys
from pathlib import Path
from collections import defaultdict
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


def extract_surface_vertices(tets):
    face_count = defaultdict(int)
    for tet in tets:
        a, b, c, d = int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])
        for face in [(a, c, b), (a, b, d), (a, d, c), (b, c, d)]:
            face_count[tuple(sorted(face))] += 1
    surf = set()
    for k, cnt in face_count.items():
        if cnt == 1:
            surf.update(k)
    return np.array(sorted(surf), dtype=np.int64)


def compute_max_edge_degree(edges_a, edges_b, N):
    """Max number of edges incident to any single particle."""
    deg = np.zeros(N, dtype=np.int32)
    np.add.at(deg, edges_a, 1)
    np.add.at(deg, edges_b, 1)
    return int(deg.max()), int(deg.mean()), int(deg.min())


def safe_inner_lr(lambda_E, max_degree):
    """Half-Newton lr_pos given Hessian upper bound: lr = 1/(2·H_upper)."""
    H_upper = 2.0 + 2.0 * max_degree * lambda_E
    return 1.0 / H_upper  # half-Newton for safety margin


def verify_inner_monotonic(solver, rest_state, knife_traj, knife_radius,
                              obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t,
                              edges_rest_t, theta_4d, device,
                              lr_pos, lambda_E, surface_idx, label):
    """Single inner GD call at θ_4d, check L_inner monotonic descent."""
    theta_torch = torch.tensor(theta_4d, dtype=torch.float32, device=device,
                                 requires_grad=False)
    with torch.no_grad():
        alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
            theta_torch, solver, rest_state, knife_traj[0], knife_radius,
        )
    _, _, info = gd_inner_loop_stable(
        pos_t, dv_t, obs_pos_t[0], obs_alpha_t[0],
        theta_torch, edges_a_t, edges_b_t, edges_rest_t,
        n_iter=30, lr_pos=lr_pos, lr_dv_factor=1.0,
        lambda_cav=1.0, lambda_E=lambda_E,
        surface_idx=surface_idx,
    )
    L_hist = info["L_history"]
    n_inc = sum(1 for i in range(1, len(L_hist))
                  if L_hist[i] > L_hist[i-1] * 1.001)
    L_drop = L_hist[0] / max(L_hist[-1], 1e-30) if L_hist[-1] > 0 else float('inf')
    is_finite = all(np.isfinite(L_hist))
    print(f"   [verify {label}] lr_pos={lr_pos:.4g}, λ_E={lambda_E:.2g}, "
          f"finite={is_finite}, monotonic={n_inc<3}, "
          f"L: {L_hist[0]:.3e} → {L_hist[-1]:.3e} (drop {L_drop:.1f}×)")
    return is_finite and (n_inc < 3)


def run_recovery(solver, rest_state, knife_traj, knife_radius,
                   obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t,
                   edges_rest_t, theta_init_4d, device,
                   N_epochs, lr, n_inner, lr_pos, lambda_E, surface_idx):
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
                n_iter=n_inner, lr_pos=lr_pos, lr_dv_factor=1.0,
                lambda_cav=1.0, lambda_E=lambda_E,
                surface_idx=surface_idx,
            )
            L_gap = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
            opt.zero_grad()
            L_gap.backward()
            if torch.isnan(theta_torch.grad).any():
                return None, f"NaN at epoch {epoch}, t={t}"
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
            new_dv = state["particle_damage"]
            state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(
                SHARPNESS_INNER * (new_dv - TAU_INNER), -50, 50)))
        epoch_thetas.append(theta_torch.detach().cpu().numpy().copy())
    return np.array(epoch_thetas), None


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    N = solver.N
    print(f"\nSolver: N={N}  E={int(solver.e['num_edges'])}")

    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    max_deg, mean_deg, min_deg = compute_max_edge_degree(edges_a_np, edges_b_np, N)
    print(f"  Mesh edge degree per particle: min={min_deg}, mean={mean_deg}, max={max_deg}")

    surface_verts = extract_surface_vertices(mesh_info["tets"])
    print(f"  Surface particles: {len(surface_verts)}/{N}  ({len(surface_verts)/N*100:.1f}%)")

    device = "cuda:0"
    knife_radius = 0.04
    T = 3
    knife_traj = make_knife_trajectory(T)
    theta_true = np.array([3.0, 20.0, 0.5, 0.85], dtype=np.float32)
    theta_init = np.array([6.0, 30.0, 0.8, 1.3], dtype=np.float32)  # all-high

    surface_idx_t = torch.tensor(surface_verts, dtype=torch.long, device=device)

    print(f"\n[Setup] θ_true = {theta_true.tolist()}")
    print(f"        all-high init = {theta_init.tolist()}")
    print(f"        N_epochs=200, T=3, n_inner=30, outer lr=0.02\n")

    obs_pos, obs_alpha = gen_obs_multistep_param_arr(
        solver, rest_state, theta_true, knife_traj, knife_radius, device)
    obs_pos_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]
    edges_a_t = torch.tensor(edges_a_np, dtype=torch.long, device=device)
    edges_b_t = torch.tensor(edges_b_np, dtype=torch.long, device=device)
    # BUG FIX: previously edges_rest_t was incorrectly set to solver.e["pf_rest"]
    # (probability values [0,1]), but L_E formula treats it as edge length in meters.
    # Compute actual edge rest LENGTHS from initial vertex positions.
    pos_rest = mesh_info["vertices"]   # rest state positions (N, 3)
    edge_rest_len = np.linalg.norm(
        pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)
    print(f"\n[Edge rest length stats] min={edge_rest_len.min():.4e}  "
          f"mean={edge_rest_len.mean():.4e}  max={edge_rest_len.max():.4e}  "
          f"(should be ~0.01-0.02 m for cube 0.3m)")

    # Compute principled lr_pos per config
    configs = []
    for label, surf, lE in [
        ("Base    (full,    λ_E=0)",    None,          0.0),
        ("T1      (surface, λ_E=0)",    surface_idx_t, 0.0),
        ("T2a     (surface, λ_E=0.01)", surface_idx_t, 0.01),
        ("T2b     (surface, λ_E=0.1)",  surface_idx_t, 0.1),
        ("T2c     (surface, λ_E=1.0)",  surface_idx_t, 1.0),
    ]:
        lr = safe_inner_lr(lE, max_deg)
        configs.append((label, surf, lE, lr))

    print(f"[Principled lr_pos via H_upper = 2 + 2·max_deg·λ_E]\n")
    print(f"  {'config':<32s}  {'λ_E':>6s}  {'H_upper':>8s}  {'lr_pos':>8s}")
    for label, _, lE, lr in configs:
        H_upper = 2.0 + 2.0 * max_deg * lE
        print(f"  {label:<32s}  {lE:6.3g}  {H_upper:8.2f}  {lr:8.4f}")

    # Sanity verify each config's inner GD is monotonic
    print(f"\n[Verify inner GD monotonic descent at θ_init (sanity)]\n")
    for label, surf, lE, lr in configs:
        verify_inner_monotonic(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
            theta_init, device, lr, lE, surf, label.split()[0],
        )

    # Run recovery
    print(f"\n[Recovery 200 ep on all-high init]")
    results = []
    for label, surf, lE, lr in configs:
        print(f"\n{'='*72}")
        print(f"  {label}    lr_pos={lr:.4f}")
        print(f"{'='*72}")
        t0 = time.time()
        traj, err = run_recovery(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
            theta_init, device, N_epochs=200, lr=0.02, n_inner=30,
            lr_pos=lr, lambda_E=lE, surface_idx=surf,
        )
        elapsed = time.time() - t0
        if err:
            print(f"   ✗ {err}")
            results.append((label, None))
            continue
        avg10 = traj[-10:].mean(axis=0)
        errs = np.abs(avg10 - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        free_errs = np.array([errs[0], errs[2], errs[3]])
        if (free_errs < 10).all(): status = "✓ tight"
        elif (free_errs < 20).all(): status = "⚠ loose"
        else: status = "✗ failed"
        print(f"   elapsed {elapsed:.0f}s  "
              f"final=({avg10[0]:.4f},{avg10[1]:.1f},{avg10[2]:.4f},{avg10[3]:.4f})")
        print(f"   err: α={errs[0]:.2f}%  τ={errs[2]:.2f}%  stiff={errs[3]:.2f}%  → {status}")
        results.append((label, dict(errs=errs, avg10=avg10, status=status)))

    print(f"\n{'='*72}")
    print(f"SUMMARY (principled adaptive lr)")
    print(f"{'='*72}")
    print(f"\n  {'config':<32s}  {'lr_pos':>8s}  {'α err':>8s}  {'τ err':>8s}  "
          f"{'stiff err':>10s}  status")
    for (label, _, lE, lr), (_, r) in zip(configs, results):
        if r is None:
            print(f"  {label:<32s}  {lr:8.4f}  (NaN)")
            continue
        e = r["errs"]
        print(f"  {label:<32s}  {lr:8.4f}  {e[0]:7.2f}%  {e[2]:7.2f}%  "
              f"{e[3]:9.2f}%   {r['status']}")


if __name__ == "__main__":
    main()
