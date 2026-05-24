"""R4-v2: Hessian conditioning across 3 losses — identify the REAL diagnostic.

R4-v1 measured H of L_α only and concluded stiffness is rank-deficient (κ=3041).
But our actual pipeline optimizes L_gap ≈ L_pos + L_α (via inner loop),
so L_α-only Hessian was the wrong diagnostic.

Hypothesis (per user's deformation insight):
   stiffness has FIRST-order effect on positions (via shape-matching force),
   but only HIGH-order effect on α (5 transforms + sigmoid smoothing).
   Therefore L_α-only Hessian massively underestimates stiffness curvature.

This script measures Hessian for 3 losses at θ_true:
   1. L_α = (1/TN) Σ_t ‖α_t - α_t_target‖²        (R4-v1's loss)
   2. L_pos = (1/TN) Σ_t ‖pos_t - pos_t_target‖²   (direct deformation)
   3. L_combined = L_pos + L_α                      (effective L_gap surrogate)

Verdict:
   - If κ(L_pos) << κ(L_α): user's insight confirmed.
   - If κ(L_combined) < 100: actual pipeline well-conditioned, P1.2-6 viable.
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pf_shape_matching_warp"))

import warp as wp
import torch
wp.init()
print(f"Warp: {wp.__version__}  Torch: {torch.__version__}")

from tools.day2_solver_wrapper import (  # noqa: E402
    setup_solver_with_rest, solver_save_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day6f_v0_multistep_synth import make_knife_trajectory  # noqa: E402


PARAM_INDICES_3D = [0, 2, 3]
PARAM_NAMES_3D = ["α_min", "τ", "stiffness"]


def forward_collect(theta_4d_torch, solver, rest_state, knife_traj, knife_radius):
    """Run T-step forward, return (alphas, positions) lists of torch tensors."""
    T = len(knife_traj)
    state_in = rest_state
    alphas = []
    positions = []
    for t, knife_pos in enumerate(knife_traj):
        alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
            theta_4d_torch, solver, state_in, knife_pos, knife_radius,
        )
        alphas.append(alpha_t)
        positions.append(pos_t)
        state_in = solver_save_state(solver)
    return alphas, positions


def loss_at(theta_4d_torch, solver, rest_state, knife_traj, knife_radius,
            targets_alpha, targets_pos, kind):
    """Compute scalar loss of one of 3 kinds."""
    alphas, positions = forward_collect(theta_4d_torch, solver, rest_state,
                                          knife_traj, knife_radius)
    T = len(knife_traj)
    N = alphas[0].shape[0]
    L_alpha = sum(((a - ta) ** 2).sum() for a, ta in zip(alphas, targets_alpha)) / (T * N)
    L_pos   = sum(((p - tp) ** 2).sum() for p, tp in zip(positions, targets_pos)) / (T * N * 3)
    if kind == "alpha":
        return L_alpha
    if kind == "pos":
        return L_pos
    if kind == "combined":
        return L_pos + L_alpha
    raise ValueError(kind)


def gradient_at(theta_4d_np, solver, rest_state, knife_traj, knife_radius,
                 targets_alpha, targets_pos, kind, device):
    theta_torch = torch.tensor(theta_4d_np, dtype=torch.float32, device=device,
                                 requires_grad=True)
    L = loss_at(theta_torch, solver, rest_state, knife_traj, knife_radius,
                  targets_alpha, targets_pos, kind)
    L.backward()
    return theta_torch.grad.cpu().numpy().copy(), float(L.item())


def compute_hessian_3d(theta_true_4d, solver, rest_state, knife_traj, knife_radius,
                         targets_alpha, targets_pos, kind, device):
    """3×3 Hessian via central FD on autograd gradient."""
    eps_per_dim_4d = {0: 0.05, 2: 0.01, 3: 0.05}
    H = np.zeros((3, 3))
    for j_3d, j_4d in enumerate(PARAM_INDICES_3D):
        eps = eps_per_dim_4d[j_4d]
        theta_plus  = theta_true_4d.copy(); theta_plus[j_4d]  += eps
        theta_minus = theta_true_4d.copy(); theta_minus[j_4d] -= eps
        g_plus,  _ = gradient_at(theta_plus,  solver, rest_state, knife_traj,
                                    knife_radius, targets_alpha, targets_pos, kind, device)
        g_minus, _ = gradient_at(theta_minus, solver, rest_state, knife_traj,
                                    knife_radius, targets_alpha, targets_pos, kind, device)
        H[:, j_3d] = (g_plus[PARAM_INDICES_3D] - g_minus[PARAM_INDICES_3D]) / (2 * eps)
    return H


def analyze_H(H, kind):
    print(f"\n{'─'*72}")
    print(f"Loss = {kind}")
    print(f"{'─'*72}")
    print(f"  H matrix (3×3):")
    for i in range(3):
        row = "  ".join(f"{H[i,j]:+.4e}" for j in range(3))
        print(f"    [{row}]")
    H_sym = 0.5 * (H + H.T)
    asym = np.linalg.norm(H - H.T) / max(np.linalg.norm(H), 1e-20)
    print(f"  asymmetry = {asym*100:.2f}%")

    eigvals, eigvecs = np.linalg.eigh(H_sym)
    print(f"  Eigenvalues / eigenvectors (sorted ascending):")
    for i in range(3):
        v = eigvecs[:, i]
        comp = "  ".join(f"{PARAM_NAMES_3D[d]}={v[d]:+.3f}" for d in range(3))
        print(f"    λ_{i} = {eigvals[i]:+.4e}    [{comp}]")

    lam_max = abs(eigvals).max()
    lam_min = abs(eigvals).min()
    kappa = lam_max / max(lam_min, 1e-30)

    diag = np.abs(np.diag(H_sym))
    print(f"  Diagonal sensitivity ranking:")
    for d in np.argsort(diag)[::-1]:
        rel = diag[d] / max(diag.max(), 1e-30)
        print(f"    {PARAM_NAMES_3D[d]:<10s}: {diag[d]:.4e}  ({rel*100:.2f}% of max)")

    print(f"  → λ_max = {lam_max:.4e}, λ_min = {lam_min:.4e}, κ = {kappa:.2f}")
    return kappa, diag, eigvals, eigvecs


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, _ = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_radius = 0.04
    T = 3
    knife_traj = make_knife_trajectory(T)
    theta_true_4d = np.array([3.0, 20.0, 0.5, 0.85], dtype=np.float32)

    # Generate targets at θ_true
    print(f"\n[Setup] θ_true={theta_true_4d.tolist()}, T={T}")
    theta_torch_true = torch.tensor(theta_true_4d, dtype=torch.float32, device=device,
                                      requires_grad=False)
    with torch.no_grad():
        alphas_true, positions_true = forward_collect(theta_torch_true, solver,
                                                         rest_state, knife_traj, knife_radius)
    targets_alpha = [a.clone() for a in alphas_true]
    targets_pos = [p.clone() for p in positions_true]
    cavity_per_frame = [float((a < 0.5).float().mean()) for a in targets_alpha]
    print(f"  cavity per frame: {[f'{c:.4f}' for c in cavity_per_frame]}")

    # Compute Hessian for each loss
    results = {}
    for kind in ["alpha", "pos", "combined"]:
        print(f"\n{'='*72}")
        print(f"Computing Hessian for L_{kind} …")
        print(f"{'='*72}")
        H = compute_hessian_3d(theta_true_4d, solver, rest_state, knife_traj,
                                  knife_radius, targets_alpha, targets_pos, kind, device)
        kappa, diag, eigvals, eigvecs = analyze_H(H, f"L_{kind}")
        results[kind] = dict(H=H, kappa=kappa, diag=diag,
                              eigvals=eigvals, eigvecs=eigvecs)

    # Summary comparison
    print(f"\n{'='*72}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*72}")
    print(f"\n{'Loss':>12s}  {'κ':>12s}  {'τ diag':>14s}  {'α_min diag':>14s}  {'stiff diag':>14s}")
    for kind in ["alpha", "pos", "combined"]:
        r = results[kind]
        # Per-param diagonal (using PARAM_NAMES_3D order: α, τ, stiff)
        d = r["diag"]
        print(f"  L_{kind:<10s} {r['kappa']:12.2f}  {d[1]:14.4e}  {d[0]:14.4e}  {d[2]:14.4e}")

    # Diagnostic: did adding L_pos rescue stiffness conditioning?
    print(f"\n[Diagnostic]")
    stiff_diag_alpha = results["alpha"]["diag"][2]
    stiff_diag_pos = results["pos"]["diag"][2]
    stiff_diag_combined = results["combined"]["diag"][2]
    print(f"  stiffness diag in L_α:        {stiff_diag_alpha:.4e}")
    print(f"  stiffness diag in L_pos:      {stiff_diag_pos:.4e}  "
          f"({stiff_diag_pos / max(stiff_diag_alpha, 1e-30):.0f}× larger)")
    print(f"  stiffness diag in L_combined: {stiff_diag_combined:.4e}")

    kappa_alpha = results["alpha"]["kappa"]
    kappa_pos = results["pos"]["kappa"]
    kappa_combined = results["combined"]["kappa"]

    print(f"\n  κ(L_α)        = {kappa_alpha:.0f}")
    print(f"  κ(L_pos)      = {kappa_pos:.0f}")
    print(f"  κ(L_combined) = {kappa_combined:.0f}")

    # Verdict
    print(f"\n{'='*72}")
    print(f"R4-v2 VERDICT")
    print(f"{'='*72}")
    if kappa_combined < 100:
        print(f"  ✓ L_combined WELL-CONDITIONED (κ={kappa_combined:.0f} < 100)")
        print(f"    User's insight CONFIRMED: stiffness recovers via L_pos path.")
        print(f"    Our actual pipeline (L_gap ≈ L_combined) is well-posed for 3D sys-ID.")
        print(f"    → R4-v1's ill-conditioning was diagnostic artifact (wrong loss).")
        print(f"    → P1.2-6 (3D joint sys-ID) IS viable.")
    elif kappa_combined < 1000:
        print(f"  ⚠ L_combined MARGINAL (κ={kappa_combined:.0f})")
        print(f"    L_pos partially rescues stiffness but degeneracy remains.")
    else:
        print(f"  ✗ L_combined STILL ill-conditioned (κ={kappa_combined:.0f})")
        print(f"    L_pos alone not enough. Need force / cavity geometry observable.")


if __name__ == "__main__":
    main()
