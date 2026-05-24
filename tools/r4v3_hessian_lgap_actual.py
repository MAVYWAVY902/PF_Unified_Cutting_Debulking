"""R4-v3: Hessian of the ACTUAL outer loss L_gap (through inner loop).

R4-v1 measured L_α Hessian, R4-v2 added L_pos and L_combined. BOTH measured
DIRECT losses, but our actual pipeline (calibrate_resect.py) optimizes:
   L_gap = ||Δx||² + ||Δdv||²
where (Δx, Δdv) come from gd_inner_loop_stable's 30-step GD on (L_pos + L_cav).

Because the inner loop minimizes L_pos + L_cav, ||Δx||² implicitly carries
position-gap signal — that's why R3-B succeeded for stiffness even though
R4-v2's direct L_pos showed weak signal.

This script measures the CORRECT Hessian of the actual pipeline loss.

Method:
  1. At θ_test, run T-step pipeline (with Mode C inject) accumulating ∇L_gap.
  2. FD this gradient at θ_true ± ε on each of 3 free dims.
  3. Eigendecompose → κ, eigenvectors.

Setup mirrors calibrate_resect.py exactly:
  T=3, n_inner=30, lr_pos=5e-3, lr_dv_factor=1.0, lambda_cav=1.0, lambda_E=0.0
  Sharpness frozen at 20; gradient on indices [0, 2, 3] = (α_min, τ, stiff).
"""
import sys
from pathlib import Path
import numpy as np

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
from tools.day6f_v0_multistep_synth import make_knife_trajectory  # noqa: E402


SHARPNESS_FIXED = 20.0
PARAM_INDICES_3D = [0, 2, 3]
PARAM_NAMES_3D = ["α_min", "τ", "stiffness"]


def lgap_grad_at(theta_4d_np, solver, rest_state, knife_traj, knife_radius,
                  obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
                  device, n_inner=30):
    """Compute ∇θ Σ_t L_gap_t at theta_4d_np via T-step pipeline + Mode C inject.

    Matches calibrate_resect.py per-frame backward + Mode C inject pattern.
    Returns 4-elem grad (incl frozen sharpness slot).
    """
    T = len(knife_traj)
    theta_torch = torch.tensor(theta_4d_np, dtype=torch.float32, device=device,
                                 requires_grad=True)
    # Pre-zero grad (defensive)
    if theta_torch.grad is None:
        theta_torch.grad = torch.zeros_like(theta_torch)
    else:
        theta_torch.grad.zero_()

    state = rest_state
    L_total = 0.0
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
        L_total += float(L_gap.item())
        # Per-frame backward — accumulates into theta_torch.grad
        L_gap.backward()
        # Mode C inject (matches calibrate_resect.py exactly)
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
    return theta_torch.grad.detach().cpu().numpy().copy(), L_total


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, _ = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_radius = 0.04
    T = 3
    knife_traj = make_knife_trajectory(T)
    theta_true_4d = np.array([3.0, 20.0, 0.5, 0.85], dtype=np.float32)
    print(f"\n[Setup] θ_true={theta_true_4d.tolist()}, T={T}, n_inner=30")

    # Generate obs at θ_true
    obs_pos, obs_alpha = gen_obs_multistep_param_arr(
        solver, rest_state, theta_true_4d, knife_traj, knife_radius, device)
    obs_pos_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_rest_t = torch.tensor(solver.e["pf_rest"].numpy(),
                                  dtype=torch.float32, device=device)

    # ---- Sanity: gradient at θ_true should be ≈ 0 ----
    print(f"\n[Sanity] ∇L_gap at θ_true (should be ≈ 0 since sim=obs)")
    g_at_truth, L_at_truth = lgap_grad_at(
        theta_true_4d, solver, rest_state, knife_traj, knife_radius,
        obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t, device)
    print(f"  L_gap_total at θ_true = {L_at_truth:.4e}")
    print(f"  ∇L_gap 4D = {g_at_truth}")
    print(f"  ‖∇‖_3D (excl sharp) = {np.linalg.norm(g_at_truth[PARAM_INDICES_3D]):.4e}")

    # ---- Hessian via central FD on gradient ----
    eps_per_dim_4d = {0: 0.05, 2: 0.01, 3: 0.05}
    print(f"\n[Hessian] Computing 3×3 H via central FD on accumulated L_gap gradient")
    print(f"          ε per dim: α=0.05, τ=0.01, stiff=0.05")
    print(f"          (6 gradient evals; each = T·forward + T·30-step inner GD + backward)")

    H = np.zeros((3, 3))
    for j_3d, j_4d in enumerate(PARAM_INDICES_3D):
        eps = eps_per_dim_4d[j_4d]
        theta_plus  = theta_true_4d.copy(); theta_plus[j_4d]  += eps
        theta_minus = theta_true_4d.copy(); theta_minus[j_4d] -= eps
        g_plus,  L_p = lgap_grad_at(theta_plus,  solver, rest_state, knife_traj,
                                       knife_radius, obs_pos_t, obs_alpha_t,
                                       edges_a_t, edges_b_t, edges_rest_t, device)
        g_minus, L_m = lgap_grad_at(theta_minus, solver, rest_state, knife_traj,
                                       knife_radius, obs_pos_t, obs_alpha_t,
                                       edges_a_t, edges_b_t, edges_rest_t, device)
        H[:, j_3d] = (g_plus[PARAM_INDICES_3D] - g_minus[PARAM_INDICES_3D]) / (2 * eps)
        print(f"  ε on {PARAM_NAMES_3D[j_3d]:<10s}: L+={L_p:.4e}  L-={L_m:.4e}")
        print(f"     H[:, {j_3d}] = [{H[0,j_3d]:+.4e}, {H[1,j_3d]:+.4e}, {H[2,j_3d]:+.4e}]")

    H_sym = 0.5 * (H + H.T)
    asym = np.linalg.norm(H - H.T) / max(np.linalg.norm(H), 1e-20)
    print(f"\n[Symmetry] ‖H - Hᵀ‖ / ‖H‖ = {asym*100:.2f}%")

    eigvals, eigvecs = np.linalg.eigh(H_sym)
    print(f"\n[Eigendecomp of symmetric H]")
    for i in range(3):
        v = eigvecs[:, i]
        comp = "  ".join(f"{PARAM_NAMES_3D[d]}={v[d]:+.3f}" for d in range(3))
        print(f"  λ_{i} = {eigvals[i]:+.4e}    [{comp}]")

    lam_max = abs(eigvals).max()
    lam_min = abs(eigvals).min()
    kappa = lam_max / max(lam_min, 1e-30)
    diag = np.abs(np.diag(H_sym))
    print(f"\n[Conditioning]")
    print(f"  λ_max = {lam_max:.4e}")
    print(f"  λ_min = {lam_min:.4e}")
    print(f"  κ(H_actual_Lgap) = {kappa:.2f}")
    print(f"\n[Diagonal per-param sensitivity]")
    for d in np.argsort(diag)[::-1]:
        rel = diag[d] / max(diag.max(), 1e-30)
        print(f"  {PARAM_NAMES_3D[d]:<10s}: {diag[d]:.4e}  ({rel*100:.2f}% of max)")

    print(f"\n{'='*72}")
    print(f"R4-v3 VERDICT — Hessian of ACTUAL pipeline loss")
    print(f"{'='*72}")
    print(f"  Compare:")
    print(f"    κ(L_α)        = 3041   (R4-v1, wrong loss)")
    print(f"    κ(L_pos)      = 6787   (R4-v2, wrong loss)")
    print(f"    κ(L_combined) = 2995   (R4-v2, dominated by L_α)")
    print(f"    κ(L_gap)      = {kappa:.0f}  ← THIS, actual pipeline")

    if kappa < 100:
        print(f"\n  ✓ WELL-CONDITIONED → R3-B success EXPLAINED by Hessian")
        print(f"    R4-v1/v2 ill-conditioning was diagnostic artifact.")
        print(f"    All 3 free params identifiable from L_gap.")
    elif kappa < 1000:
        print(f"\n  ⚠ MARGINAL but recoverable (matches R3-B 5/5 with 200 ep)")
    else:
        print(f"\n  ✗ Still ill-conditioned at L_gap level too")
        print(f"    R3-B success despite this = far-init dynamics + Adam normalization")


if __name__ == "__main__":
    main()
