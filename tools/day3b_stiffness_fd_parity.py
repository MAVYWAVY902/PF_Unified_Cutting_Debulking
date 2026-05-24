"""Day 3b (2026-05-23): FD parity test for stiffness gradient.

Validates apply_kernel_ssa_param_arr's autograd ∂L/∂stiffness against central
finite-difference ground truth. Methodology mirrors Day 6b (alpha_min, sharpness,
threshold parity) but exercises a different code path — stiffness enters EVERY
Jacobi iter × EVERY substep × T outer steps, so the backward graph through
stiffness is much longer than through cohesive params.

Pass criterion (Gate A, no hack):
   rel_err < 5%  on each of (α_min, sharpness, τ, stiffness)
   sign match on all 4 dims

Looser than Day 6b's 2% because:
   - stiffness backward chain depth ≈ T × substeps × num_iter ≈ 3 × 8 × 6 = 144 launches
     vs cohesive params' ≈ T × 1 = 3 launches
   - Atomic_add accumulation in apply_kernel_ssa_param_arr → atomic-order non-det
     adds tiny FP noise on top.

If FAIL → STOP. Do not hand-tune. Diagnose:
   - Gradient explode/vanish? Check magnitude vs Day 6b's grad.
   - Atomic non-det? Re-run multiple times, compute std.
   - Müller polar non-diff region? Check svd_3x3 / quat_from_matrix.
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
    setup_solver_with_rest, solver_load_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402


def loss_at(solver, rest_state, theta_np, knife_pos, knife_radius, alpha_target_np):
    """Eval loss = (1/N)·Σ(α - α_target)² at given θ via legacy float path.

    theta_np = (α_min, sharpness, τ, stiffness)
    """
    solver.set_theta_wp(None)  # ensure legacy path
    solver.alpha_min = float(theta_np[0])
    solver.soft_removal_sharpness = float(theta_np[1])
    solver.aspiration_threshold = float(theta_np[2])
    solver.stiffness = float(theta_np[3])
    solver.knife_radius = knife_radius
    solver.knife_active = 1
    solver.knife_pos = wp.array(np.array([knife_pos], dtype=np.float32),
                                  dtype=wp.vec3, device=solver.device,
                                  requires_grad=False)
    solver_load_state(solver, rest_state)
    solver.step()
    if solver.use_ssa_soft_removal:
        ci = solver._ssa_cycle_index
        alpha = solver._ssa_alpha_buffers[ci].numpy()
    else:
        alpha = solver.p["alpha"].numpy()
    N = solver.N
    return float(np.sum((alpha - alpha_target_np) ** 2)) / float(N)


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, _ = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_pos = np.array([0.15, 0.30, 0.15], dtype=np.float32)
    knife_radius = 0.04
    theta_true = np.array([3.0, 20.0, 0.5, 0.85], dtype=np.float32)
    theta_init = np.array([5.0, 15.0, 0.6, 0.70], dtype=np.float32)
    param_names = ["α_min", "sharpness", "τ", "stiffness"]

    # ---- Generate α_target at θ_true ----
    print(f"\nGenerating α_target at θ_true={theta_true.tolist()} (1 step)…")
    solver.set_theta_wp(None)
    solver.alpha_min = float(theta_true[0])
    solver.soft_removal_sharpness = float(theta_true[1])
    solver.aspiration_threshold = float(theta_true[2])
    solver.stiffness = float(theta_true[3])
    solver.knife_radius = knife_radius
    solver.knife_active = 1
    solver.knife_pos = wp.array(np.array([knife_pos], dtype=np.float32),
                                  dtype=wp.vec3, device=solver.device, requires_grad=False)
    solver_load_state(solver, rest_state)
    solver.step()
    if solver.use_ssa_soft_removal:
        ci = solver._ssa_cycle_index
        alpha_target_np = solver._ssa_alpha_buffers[ci].numpy().copy()
    else:
        alpha_target_np = solver.p["alpha"].numpy().copy()
    print(f"  target cavity: {float((alpha_target_np < 0.5).mean()):.4f}")

    # ---- Path A: 4D autograd via WarpFullForwardOp ----
    print(f"\n[Path A] Autograd through 4D WarpFullForwardOp (stiffness in θ)")
    theta_torch = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                 requires_grad=True)
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)
    alpha_out, _, _ = WarpFullForwardOp.apply(
        theta_torch, solver, rest_state, knife_pos, knife_radius,
    )
    N = solver.N
    loss_torch = ((alpha_out - alpha_target_t) ** 2).sum() / float(N)
    print(f"  loss @ θ_init = {loss_torch.item():.6e}")
    loss_torch.backward()
    grad_A = theta_torch.grad.cpu().numpy().copy()
    for d, name in enumerate(param_names):
        print(f"  grad_A[{name}] = {grad_A[d]:+.6e}")

    # ---- Path B: central FD on all 4 dims ----
    print(f"\n[Path B] Central FD ground truth (legacy float path)")
    # ε tuned per dim. stiffness sensitivity is tiny (~6e-5) so very small ε
    # falls into FD noise floor (see day3b_stiffness_diagnose.py ε sweep).
    # ε=5e-2 gives the cleanest FD baseline (rel_err 1.68% to autograd).
    eps_per_dim = [1e-2, 1e-1, 1e-2, 5e-2]
    grad_B = np.zeros(4)
    for d in range(4):
        eps = eps_per_dim[d]
        th_plus = theta_init.copy(); th_plus[d] += eps
        th_minus = theta_init.copy(); th_minus[d] -= eps
        L_p = loss_at(solver, rest_state, th_plus, knife_pos, knife_radius, alpha_target_np)
        L_m = loss_at(solver, rest_state, th_minus, knife_pos, knife_radius, alpha_target_np)
        grad_B[d] = (L_p - L_m) / (2 * eps)
        print(f"  grad_B[{param_names[d]:>10s}] = {grad_B[d]:+.6e}  (ε={eps:g})")

    # ---- Parity check ----
    print("\n" + "=" * 72)
    print("Day 3b parity: 4D autograd ↔ FD")
    print("=" * 72)
    rel_err = np.abs(grad_A - grad_B) / np.maximum(np.abs(grad_B), 1e-12)
    sign_match = np.sign(grad_A) == np.sign(grad_B)
    for d, name in enumerate(param_names):
        ag = grad_A[d]; bg = grad_B[d]
        rel = rel_err[d] * 100
        sgn = "✓" if sign_match[d] else "✗"
        print(f"  {name:>10s}: A={ag:+.4e}  B={bg:+.4e}  rel_err={rel:6.2f}%  sign {sgn}")

    pass_strict = bool(sign_match.all() and (rel_err < 0.05).all())   # < 5% Gate A
    pass_loose = bool(sign_match.all() and (rel_err < 0.20).all())    # < 20% degraded

    print()
    if pass_strict:
        print(f"✓ Day 3b GATE A PASS — all 4 dims < 5% rel_err  (stiffness backward CORRECT)")
    elif pass_loose:
        print(f"⚠ Day 3b PARTIAL — signs match, < 20% but not < 5%. Investigate before P1.2-6.")
    else:
        print(f"✗ Day 3b FAIL — STOP. Do not proceed to sys-ID. Diagnose:")
        print(f"     1. Atomic non-determinism? Re-run, check std.")
        print(f"     2. Backward magnitude vs cohesive grads?")
        print(f"     3. Possible Müller polar non-diff region?")


if __name__ == "__main__":
    main()
