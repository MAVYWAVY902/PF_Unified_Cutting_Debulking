"""Test 1+2: phantom-readiness with surface-only L_pos + L_E ON.

Goal: validate whether our pipeline (Day 3c fix) works under phantom-like
constraints — camera only observes surface, requires E term to propagate
information to interior particles (Liang's core mechanism).

Configs (all on all-high init = R3-B failure case, 200 ep, lr_pos=0.5):

  Base    (full obs, λ_E=0)        — current setup, known to fail (R3-B with fix)
  T1      (surface only, λ_E=0)    — expected to fail catastrophically (no E)
  T2a     (surface only, λ_E=0.1)  — small E
  T2b     (surface only, λ_E=1.0)  — Liang-ish
  T2c     (surface only, λ_E=10)   — strong E

Verdict:
  If T2 series rescues recovery → L_E truly enables Liang-style propagation
  If T2 series also fails → cube setup too rigid for surface→interior signal
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


def extract_surface_vertices(tets, N):
    """Extract surface vertex indices from tet mesh by finding boundary faces."""
    face_count = defaultdict(int)
    for tet in tets:
        a, b, c, d = int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])
        for face in [(a, c, b), (a, b, d), (a, d, c), (b, c, d)]:
            key = tuple(sorted(face))
            face_count[key] += 1
    surface_verts = set()
    for face_key, cnt in face_count.items():
        if cnt == 1:
            surface_verts.update(face_key)
    return np.array(sorted(surface_verts), dtype=np.int64)


def run_recovery(solver, rest_state, knife_traj, knife_radius,
                   obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t,
                   edges_rest_t, theta_init_4d, device,
                   N_epochs=200, lr=0.02, n_inner=30,
                   lambda_E=0.0, surface_idx=None):
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
                n_iter=n_inner, lr_pos=0.5, lr_dv_factor=1.0,
                lambda_cav=1.0, lambda_E=lambda_E,
                surface_idx=surface_idx,
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
            new_dv = state["particle_damage"]
            state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(
                SHARPNESS_INNER * (new_dv - TAU_INNER), -50, 50)))
        epoch_thetas.append(theta_torch.detach().cpu().numpy().copy())
    return np.array(epoch_thetas), None


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    # Extract surface particles
    tets = mesh_info["tets"]
    surface_verts = extract_surface_vertices(tets, solver.N)
    n_surface = len(surface_verts)
    n_interior = solver.N - n_surface
    print(f"  Surface particles: {n_surface}/{solver.N}  ({n_surface/solver.N*100:.1f}%)")
    print(f"  Interior particles: {n_interior}/{solver.N}  ({n_interior/solver.N*100:.1f}%)")

    device = "cuda:0"
    knife_radius = 0.04
    T = 3
    knife_traj = make_knife_trajectory(T)
    theta_true = np.array([3.0, 20.0, 0.5, 0.85], dtype=np.float32)
    theta_init = np.array([6.0, 30.0, 0.8, 1.3], dtype=np.float32)  # all-high

    surface_idx_t = torch.tensor(surface_verts, dtype=torch.long, device=device)

    print(f"\n[Setup] θ_true = {theta_true.tolist()}")
    print(f"        all-high init = {theta_init.tolist()}")
    print(f"        N_epochs=200, lr=0.02, T=3, lr_pos=0.5\n")

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

    configs = [
        ("Base    (full, λ_E=0)",     None,           0.0),
        ("T1      (surface, λ_E=0)",  surface_idx_t,  0.0),
        ("T2a     (surface, λ_E=0.1)", surface_idx_t, 0.1),
        ("T2b     (surface, λ_E=1.0)", surface_idx_t, 1.0),
        ("T2c     (surface, λ_E=10)",  surface_idx_t, 10.0),
    ]

    results = []
    for label, surf_idx, lam_E in configs:
        print(f"\n{'='*72}")
        print(f"  {label}")
        print(f"{'='*72}")
        t0 = time.time()
        traj, err = run_recovery(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
            theta_init, device, N_epochs=200, lr=0.02, n_inner=30,
            lambda_E=lam_E, surface_idx=surf_idx,
        )
        elapsed = time.time() - t0
        if err:
            print(f"   ✗ {err}")
            results.append((label, None))
            continue
        avg10 = traj[-10:].mean(axis=0)
        errs = np.abs(avg10 - theta_true) / np.maximum(np.abs(theta_true), 1e-12) * 100
        free_errs = np.array([errs[0], errs[2], errs[3]])
        status = "✓ tight" if (free_errs < 10).all() else ("⚠ loose" if (free_errs < 20).all() else "✗ failed")
        print(f"   elapsed {elapsed:.0f}s, final avg10 = "
              f"({avg10[0]:.4f}, {avg10[1]:.1f}, {avg10[2]:.4f}, {avg10[3]:.4f})")
        print(f"   err: α={errs[0]:.2f}%  τ={errs[2]:.2f}%  stiff={errs[3]:.2f}%  → {status}")
        results.append((label, dict(errs=errs, avg10=avg10, status=status)))

    # Summary
    print(f"\n{'='*72}")
    print(f"SUMMARY")
    print(f"{'='*72}")
    print(f"\n  {'config':<30s}  {'α err':>8s}  {'τ err':>8s}  {'stiff err':>10s}  status")
    for label, r in results:
        if r is None:
            print(f"  {label:<30s}  (NaN)")
            continue
        e = r["errs"]
        print(f"  {label:<30s}  {e[0]:7.2f}%  {e[2]:7.2f}%  {e[3]:9.2f}%   {r['status']}")

    # Verdict
    print(f"\n{'='*72}")
    print(f"TEST 1+2 VERDICT")
    print(f"{'='*72}")
    base_status = results[0][1]["status"] if results[0][1] else "✗"
    t1_status = results[1][1]["status"] if results[1][1] else "✗"
    t2_statuses = [r[1]["status"] if r[1] else "✗" for r in results[2:]]

    if "✓" in base_status:
        print(f"  Base (full obs) ✓ → R3-B baseline reproducible")
    else:
        print(f"  Base (full obs) ✗ → expected (Day 3c fix limitation)")

    if "✗" in t1_status:
        print(f"  T1 (surface, no E) ✗ → expected (no propagation to interior)")
    else:
        print(f"  T1 (surface, no E) survived?! → interior particles not needed?")

    if any("✓" in s for s in t2_statuses):
        print(f"  T2 (surface + E) ✓ → L_E SUCCESSFULLY propagates surface signal")
        print(f"    → pipeline phantom-ready when L_E enabled")
    elif any("⚠" in s for s in t2_statuses):
        print(f"  T2 (surface + E) marginal → L_E partially helps")
        print(f"    → tune λ_E or add more physics (volumetric / shape-matching)")
    else:
        print(f"  T2 (surface + E) ✗ → L_E (edge length only) insufficient")
        print(f"    → need stronger physics regularization (volumetric + shape-matching)")


if __name__ == "__main__":
    main()
