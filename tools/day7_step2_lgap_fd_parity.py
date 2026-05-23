"""Day 7 Step 2: FD parity check on ∇L_gap.

After Step 1 verified L_gap landscape has minima at θ_true, Step 2 verifies
the GRADIENT is correctly propagated through the chain:

    θ → WarpFullForwardOp → (pos_sim, dv_sim)
      → simple_gd_inner_loop_relu (with create_graph=True) → (Δ_x, Δ_dv)
      → L_gap = ‖Δ_x‖² + ‖Δ_dv‖²
      → L_gap.backward() → θ.grad

vs FD ground truth: ∂L_gap/∂θ_d ≈ (L_gap(θ+ε_d) − L_gap(θ−ε_d)) / (2·ε_d)

Critical because:
  - Day 5 had a gradient bug (sharpness sign-flip) — we MUST confirm Day 7
    pipeline doesn't repeat that
  - Inner loop unroll with create_graph=True is autograd-heavy; could have
    silent NaN or wrong gradient through ReLU non-smoothness

Pass: rel_err < 10% on all 3 params, signs match FD.
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
    setup_solver_with_rest, solver_load_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day7_step1_lgap_landscape import (  # noqa: E402
    gen_obs_single_frame, simple_gd_inner_loop_relu,
)


def L_gap_at_autograd(theta_torch, solver, rest_state, knife_pos, knife_radius,
                        pos_target_t, alpha_target_t, edges_a_t, edges_b_t,
                        edges_rest_t, n_inner=30, lambda_E=0.0):
    """Build the FULL autograd chain θ → forward → inner loop → L_gap."""
    alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
        theta_torch, solver, rest_state, knife_pos, knife_radius,
    )
    # NOTE: pos_t, dv_t are torch tensors with autograd back to theta_torch
    # (via WarpFullForwardOp.backward). The inner loop uses these AS-IS with
    # create_graph=True so its Δ depends on (pos_t, dv_t, θ) → ultimately on θ.
    delta_x, delta_dv, info = simple_gd_inner_loop_relu(
        pos_t, dv_t, pos_target_t, alpha_target_t,
        theta_torch, edges_a_t, edges_b_t, edges_rest_t,
        n_iter=n_inner, lr_pos=5e-3, lr_dv=0.1,
        lambda_cav=1.0, lambda_E=lambda_E,
    )
    L_gap = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
    return L_gap, info


def L_gap_at_no_grad(theta_np, solver, rest_state, knife_pos, knife_radius,
                       pos_target_t, alpha_target_t, edges_a_t, edges_b_t,
                       edges_rest_t, device, n_inner=30, lambda_E=0.0):
    """Scalar L_gap at given theta (no autograd graph kept). For FD."""
    theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=device,
                                 requires_grad=False)
    with torch.no_grad():
        alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
            theta_torch, solver, rest_state, knife_pos, knife_radius,
        )
    delta_x, delta_dv, info = simple_gd_inner_loop_relu(
        pos_t.detach(), dv_t.detach(), pos_target_t, alpha_target_t,
        theta_torch, edges_a_t, edges_b_t, edges_rest_t,
        n_iter=n_inner, lr_pos=5e-3, lr_dv=0.1,
        lambda_cav=1.0, lambda_E=lambda_E,
    )
    L_gap = float(((delta_x ** 2).sum() + (delta_dv ** 2).sum()).item())
    return L_gap


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_pos = np.array([0.15, 0.30, 0.15], dtype=np.float32)
    knife_radius = 0.04
    theta_true = np.array([3.0, 20.0, 0.5], dtype=np.float32)
    theta_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)
    param_names = ["α_min", "sharpness", "τ"]

    print(f"\n[Setup] θ_true={theta_true.tolist()}  θ_init={theta_init.tolist()}")
    pos_target_np, alpha_target_np = gen_obs_single_frame(
        solver, rest_state, theta_true, knife_pos, knife_radius)
    pos_target_t = torch.tensor(pos_target_np, dtype=torch.float32, device=device)
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_rest_t = torch.tensor(solver.e["pf_rest"].numpy(),
                                  dtype=torch.float32, device=device)
    cavity_frac = float((alpha_target_np < 0.5).mean())
    print(f"[Setup] cavity_frac={cavity_frac:.4f}")

    # ---- Path A: autograd ∇L_gap ----
    print(f"\n[Path A] autograd ∇L_gap via simple_gd_inner_loop_relu (create_graph=True)")
    theta_torch = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                 requires_grad=True)
    L_gap_A, info_A = L_gap_at_autograd(
        theta_torch, solver, rest_state, knife_pos, knife_radius,
        pos_target_t, alpha_target_t, edges_a_t, edges_b_t, edges_rest_t,
        n_inner=30, lambda_E=0.0)
    print(f"  L_gap(θ_init) = {L_gap_A.item():.6e}")
    print(f"  L_inner: init={info_A['L_init']:.4e} → final={info_A['L_final']:.4e}  "
          f"(reduction {info_A['L_init']/max(info_A['L_final'],1e-12):.2f}×)")
    L_gap_A.backward()
    grad_A = theta_torch.grad.detach().cpu().numpy().copy()
    has_nan = bool(np.isnan(grad_A).any())
    print(f"  ∇L_gap = ({grad_A[0]:+.4e}, {grad_A[1]:+.4e}, {grad_A[2]:+.4e})")
    if has_nan:
        print(f"  ⚠ NaN in gradient — autograd chain broken")

    # ---- Path B: FD ground truth ----
    print(f"\n[Path B] FD ground truth on L_gap")
    L_0 = L_gap_at_no_grad(
        theta_init, solver, rest_state, knife_pos, knife_radius,
        pos_target_t, alpha_target_t, edges_a_t, edges_b_t, edges_rest_t,
        device, n_inner=30, lambda_E=0.0)
    print(f"  L_gap(θ_init) (FD path) = {L_0:.6e}  "
          f"(autograd: {L_gap_A.item():.6e}, rel_diff={abs(L_0-L_gap_A.item())/max(L_0,1e-12)*100:.3f}%)")

    eps_per_dim = [1e-2, 1e-1, 1e-2]
    grad_B = np.zeros(3)
    for d in range(3):
        eps = eps_per_dim[d]
        th_plus = theta_init.copy(); th_plus[d] += eps
        th_minus = theta_init.copy(); th_minus[d] -= eps
        L_p = L_gap_at_no_grad(
            th_plus, solver, rest_state, knife_pos, knife_radius,
            pos_target_t, alpha_target_t, edges_a_t, edges_b_t, edges_rest_t,
            device, n_inner=30, lambda_E=0.0)
        L_m = L_gap_at_no_grad(
            th_minus, solver, rest_state, knife_pos, knife_radius,
            pos_target_t, alpha_target_t, edges_a_t, edges_b_t, edges_rest_t,
            device, n_inner=30, lambda_E=0.0)
        grad_B[d] = (L_p - L_m) / (2 * eps)
        signal = (L_p - L_m) / max(L_0, 1e-12)
        print(f"  d={d} ({param_names[d]:>10s}): ε={eps:.0e}  "
              f"L(+ε)={L_p:.6e}  L(-ε)={L_m:.6e}  FD={grad_B[d]:+.4e}  "
              f"signal/L={signal:+.3e}")

    # ---- Parity check ----
    print("\n" + "=" * 70)
    print("Day 7 Step 2 — ∇L_gap parity: autograd ↔ FD")
    print("=" * 70)
    if has_nan:
        print("✗ NaN in autograd gradient — chain broken")
        return

    rel_err = np.abs(grad_A - grad_B) / np.maximum(np.abs(grad_B), 1e-12)
    sign_match = np.sign(grad_A) == np.sign(grad_B)
    cos_sim = np.dot(grad_A, grad_B) / (np.linalg.norm(grad_A) * np.linalg.norm(grad_B) + 1e-18)
    print(f"\n  per-dim:")
    for d, name in enumerate(param_names):
        ag = grad_A[d]; bg = grad_B[d]
        rel = rel_err[d] * 100
        sgn = "✓" if sign_match[d] else "✗"
        print(f"    {name:>10s}: autograd={ag:+.4e}  FD={bg:+.4e}  "
              f"rel_err={rel:.3f}%  sign {sgn}")

    print(f"\n  cos(autograd, FD) = {cos_sim:.6f}")
    print(f"  ‖autograd‖ / ‖FD‖ = {np.linalg.norm(grad_A)/np.linalg.norm(grad_B):.3f}")

    print()
    if sign_match.all() and (rel_err < 0.1).all():
        print(f"✓ Day 7 Step 2 PASS — ∇L_gap matches FD < 10% on all dims")
    elif sign_match.all() and (rel_err < 0.3).all():
        print(f"⚠ Day 7 Step 2 PARTIAL — signs match, magnitudes < 30% but not < 10%")
    elif sign_match.all():
        print(f"⚠ signs match but magnitudes way off")
    else:
        print(f"✗ Day 7 Step 2 FAIL — sign mismatch (autograd direction wrong)")
        print(f"   → Day 5's pathology might be returning; need to investigate")

    # Direction sanity vs θ_true
    print(f"\n[Sanity] direction toward θ_true:")
    correction_needed = theta_true - theta_init
    print(f"  needed Δθ to reach truth: ({correction_needed[0]:+.2f}, "
          f"{correction_needed[1]:+.2f}, {correction_needed[2]:+.2f})")
    print(f"  Adam moves -lr·grad, so -grad sign should match correction sign:")
    for d, name in enumerate(param_names):
        adam_dir = -np.sign(grad_A[d])
        need_dir = np.sign(correction_needed[d])
        agree = (adam_dir == need_dir)
        print(f"    {name:>10s}: -∇sign={adam_dir:+.0f}  needed={need_dir:+.0f}  "
              f"{'✓' if agree else '✗'}")


if __name__ == "__main__":
    main()
