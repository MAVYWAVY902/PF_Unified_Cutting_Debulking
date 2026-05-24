"""R4: Hessian conditioning of L_α(θ) at θ_true — 3-param identifiability test.

Question: can the 3 parameters (α_min, τ, stiffness) be uniquely identified
          from L_α observations alone?

Method: Compute 3×3 Hessian H = ∂²L_α/∂θ² at θ_true via central FD on the
        autograd gradient. The condition number κ(H) = λ_max / λ_min tells us:

  κ < 100   well-conditioned, recovery robust
  κ < 1000  marginal, recovery feasible but lr-sensitive
  κ < 1e4   ill-conditioned, recovery stuck along loss valley
  κ > 1e4   rank-deficient, MATHEMATICAL non-identifiability
            → need additional observable (force / cavity volume)

Loss: L_α = (1/T) · Σ_{t=0..T-1} ||α_t(θ) - α_t(θ_true)||² / N
      Direct, NO inner loop — this is the underlying sys-ID problem.
      T=3 matches calibrate_resect.py setup.

Setup: cube_tet.msh, knife sweep make_knife_trajectory(T=3), knife_radius=0.04.
       θ_true=(α=3.0, sharp=20, τ=0.5, stiff=0.85), sharpness frozen.
       FD ε per dim: α=0.05, τ=0.01, stiff=0.05 (per Day 3b ε sweep findings).
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


PARAM_INDICES_3D = [0, 2, 3]  # skip [1] sharpness (frozen)
PARAM_NAMES_3D = ["α_min", "τ", "stiffness"]


def loss_alpha_multistep(theta_4d_torch, solver, rest_state, knife_traj,
                          knife_radius, targets_alpha_torch):
    """L_α = (1/T) Σ_t ||α_t(θ) - α_t_target||² / N.

    Returns scalar torch tensor (differentiable wrt theta_4d_torch).
    """
    T = len(knife_traj)
    state_in = rest_state
    L_total = 0.0
    for t, knife_pos in enumerate(knife_traj):
        alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
            theta_4d_torch, solver, state_in, knife_pos, knife_radius,
        )
        N = alpha_t.shape[0]
        L_total = L_total + ((alpha_t - targets_alpha_torch[t]) ** 2).sum() / N
        state_in = solver_save_state(solver)
    return L_total / T


def gradient_at(theta_4d_np, solver, rest_state, knife_traj, knife_radius,
                 targets_alpha_torch, device):
    """Compute ∇_θ L_α at theta_4d_np. Returns 4-elem grad (incl sharpness)."""
    theta_torch = torch.tensor(theta_4d_np, dtype=torch.float32, device=device,
                                 requires_grad=True)
    L = loss_alpha_multistep(theta_torch, solver, rest_state, knife_traj,
                               knife_radius, targets_alpha_torch)
    L.backward()
    return theta_torch.grad.cpu().numpy().copy(), float(L.item())


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_radius = 0.04
    T = 3
    knife_traj = make_knife_trajectory(T)

    theta_true_4d = np.array([3.0, 20.0, 0.5, 0.85], dtype=np.float32)
    print(f"\n[Setup] θ_true_4d={theta_true_4d.tolist()}  T={T}  knife_r={knife_radius}")

    # ---- Generate α_true at θ_true (T frames) ----
    print(f"\n[Setup] Generating α_truth (T={T} frames) via WarpFullForwardOp …")
    theta_torch_true = torch.tensor(theta_true_4d, dtype=torch.float32, device=device,
                                      requires_grad=False)
    targets_alpha = []
    state_in = rest_state
    with torch.no_grad():
        for t, knife_pos in enumerate(knife_traj):
            alpha_t, _, _ = WarpFullForwardOp.apply(
                theta_torch_true, solver, state_in, knife_pos, knife_radius,
            )
            targets_alpha.append(alpha_t.clone())
            state_in = solver_save_state(solver)
    print(f"  truth cavity per frame: " +
          ", ".join(f"{float((a<0.5).float().mean()):.4f}" for a in targets_alpha))

    # ---- Verify L_α(θ_true) ≈ 0 (sanity) ----
    g_at_truth, L_at_truth = gradient_at(theta_true_4d, solver, rest_state,
                                            knife_traj, knife_radius, targets_alpha, device)
    print(f"\n[Sanity] L_α(θ_true) = {L_at_truth:.4e}  (should be ≈ machine precision)")
    g_at_truth_3d = g_at_truth[PARAM_INDICES_3D]
    print(f"          ‖∇L_α(θ_true)‖ on 3D = {np.linalg.norm(g_at_truth_3d):.4e}  (should be ≈ 0)")
    for i, n in enumerate(PARAM_NAMES_3D):
        print(f"            ∂L/∂{n:<10s} = {g_at_truth_3d[i]:+.4e}")

    # ---- Build 3×3 Hessian via FD on gradient ----
    # H[i, j] = (∂L/∂θ_i (θ + ε e_j) - ∂L/∂θ_i (θ - ε e_j)) / (2 ε)
    eps_per_dim_4d = {0: 0.05, 2: 0.01, 3: 0.05}  # α, τ, stiff (ε per dim)
    print(f"\n[Hessian] Computing 3×3 H via central FD on autograd gradient …")
    print(f"          FD ε: α=0.05, τ=0.01, stiff=0.05")

    H = np.zeros((3, 3))
    for j_3d, j_4d in enumerate(PARAM_INDICES_3D):
        eps = eps_per_dim_4d[j_4d]
        theta_plus  = theta_true_4d.copy(); theta_plus[j_4d]  += eps
        theta_minus = theta_true_4d.copy(); theta_minus[j_4d] -= eps
        g_plus,  _ = gradient_at(theta_plus,  solver, rest_state, knife_traj,
                                    knife_radius, targets_alpha, device)
        g_minus, _ = gradient_at(theta_minus, solver, rest_state, knife_traj,
                                    knife_radius, targets_alpha, device)
        H[:, j_3d] = (g_plus[PARAM_INDICES_3D] - g_minus[PARAM_INDICES_3D]) / (2 * eps)
        print(f"  col {PARAM_NAMES_3D[j_3d]:<10s}: H[:, {j_3d}] = "
              f"[{H[0, j_3d]:+.4e}, {H[1, j_3d]:+.4e}, {H[2, j_3d]:+.4e}]")

    # Symmetrize (H should be symmetric; deviation = FD noise)
    H_sym = 0.5 * (H + H.T)
    asym_err = np.linalg.norm(H - H.T) / max(np.linalg.norm(H), 1e-20)
    print(f"\n[Symmetry check] asymmetry ‖H - Hᵀ‖ / ‖H‖ = {asym_err*100:.2f}%")
    print(f"  (< 5% = clean; > 20% = FD noisy, results suspect)")

    # ---- Eigendecomposition ----
    print(f"\n[Eigendecomp of symmetric H]")
    eigvals, eigvecs = np.linalg.eigh(H_sym)
    for i in range(3):
        v = eigvecs[:, i]
        comp = "  ".join(f"{PARAM_NAMES_3D[d]}={v[d]:+.3f}" for d in range(3))
        print(f"  λ_{i} = {eigvals[i]:+.4e}    eigenvector: {comp}")

    # ---- Condition number ----
    lam_max = abs(eigvals).max()
    lam_min = abs(eigvals).min()
    kappa = lam_max / max(lam_min, 1e-30)
    print(f"\n[Conditioning]")
    print(f"  λ_max = {lam_max:.4e}")
    print(f"  λ_min = {lam_min:.4e}")
    print(f"  κ(H)  = {kappa:.2f}")

    # ---- Per-param ranking ----
    print(f"\n[Per-param diagonal sensitivity] (∂²L/∂θ_i² on diag)")
    diag = np.abs(np.diag(H_sym))
    rank = np.argsort(diag)[::-1]
    for rk_idx, dim in enumerate(rank):
        rel_to_max = diag[dim] / max(diag.max(), 1e-30)
        print(f"  rank {rk_idx+1}: {PARAM_NAMES_3D[dim]:<10s} diag={diag[dim]:.4e}  "
              f"({rel_to_max*100:.1f}% of max)")

    # ---- Verdict ----
    print(f"\n{'='*72}")
    print(f"R4 VERDICT — Hessian conditioning of L_α at θ_true")
    print(f"{'='*72}")
    if kappa < 100:
        print(f"  ✓ WELL-CONDITIONED (κ={kappa:.1f} < 100)")
        print(f"    3 parameters separately identifiable from L_α.")
        print(f"    → joint sys-ID should be robust. Proceed to R1-R3.")
    elif kappa < 1000:
        print(f"  ⚠ MARGINAL (κ={kappa:.1f}, 100-1000)")
        print(f"    Recovery feasible but lr-sensitive.")
        print(f"    Smallest-λ direction: {dict(zip(PARAM_NAMES_3D, eigvecs[:, 0]))}")
        print(f"    → R3 joint sys-ID may need per-dim lr or warm-start.")
    elif kappa < 1e4:
        print(f"  ✗ ILL-CONDITIONED (κ={kappa:.1f}, 1000-1e4)")
        print(f"    Loss valley along: {dict(zip(PARAM_NAMES_3D, eigvecs[:, 0]))}")
        print(f"    → joint optimizer will get stuck.")
        print(f"    Mitigation: staged sys-ID (fix some, free others) or new observable.")
    else:
        print(f"  ✗✗ RANK-DEFICIENT (κ={kappa:.1f}, > 1e4)")
        print(f"    L_α MATHEMATICALLY cannot uniquely identify all 3 params.")
        print(f"    Null direction: {dict(zip(PARAM_NAMES_3D, eigvecs[:, 0]))}")
        print(f"    → MUST add observable (force, cavity volume, surface IoU).")
        print(f"    → Current pipeline architecturally insufficient for 3-param sys-ID.")


if __name__ == "__main__":
    main()
