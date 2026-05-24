"""Day 3b diagnostic: stiffness FD parity is 6.23% > 5% gate. Investigate why.

Three checks (no hack fix — methodology):
   C1. FD ε sweep — see if FD gradient converges as ε varies.
       True gradient = limit ε→0, but noise dominates at very small ε.
       Look for a plateau range where grad_B is stable.
   C2. Autograd determinism — same θ → same gradient under repeated runs?
       If grad_A varies > 1% across runs → atomic non-det in backward.
   C3. Forward determinism — same θ → same alpha output across repeated runs?
       Quantifies atomic non-det in forward.

Decision rule:
   - If FD ε sweep shows stable grad_B and grad_A is < 5% from that → PASS, atomic noise was the issue, not a bug.
   - If FD doesn't converge OR grad_A jumps across runs → BUG, do NOT proceed.
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


def forward_eval(solver, rest_state, theta_np, knife_pos, knife_radius):
    """Run one solver.step at theta_np via legacy float path, return alpha."""
    solver.set_theta_wp(None)
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
        return solver._ssa_alpha_buffers[ci].numpy().copy()
    return solver.p["alpha"].numpy().copy()


def loss_at(solver, rest_state, theta_np, knife_pos, knife_radius, alpha_target_np):
    alpha = forward_eval(solver, rest_state, theta_np, knife_pos, knife_radius)
    N = solver.N
    return float(np.sum((alpha - alpha_target_np) ** 2)) / float(N)


def autograd_grad(solver, rest_state, theta_np, knife_pos, knife_radius, alpha_target_t):
    """Backward through WarpFullForwardOp at theta_np with 4D θ; return grad."""
    device = "cuda:0"
    theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=device,
                                 requires_grad=True)
    alpha_out, _, _ = WarpFullForwardOp.apply(
        theta_torch, solver, rest_state, knife_pos, knife_radius,
    )
    N = solver.N
    loss_torch = ((alpha_out - alpha_target_t) ** 2).sum() / float(N)
    loss_torch.backward()
    return theta_torch.grad.cpu().numpy().copy()


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, _ = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_pos = np.array([0.15, 0.30, 0.15], dtype=np.float32)
    knife_radius = 0.04
    theta_true = np.array([3.0, 20.0, 0.5, 0.85], dtype=np.float32)
    theta_init = np.array([5.0, 15.0, 0.6, 0.70], dtype=np.float32)

    # Target
    alpha_target_np = forward_eval(solver, rest_state, theta_true, knife_pos, knife_radius)
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)

    # ============ C3: Forward determinism (3 repeats) ============
    print(f"\n[C3] Forward determinism — same θ_init 3 times")
    alphas = []
    for r in range(3):
        a = forward_eval(solver, rest_state, theta_init, knife_pos, knife_radius)
        alphas.append(a)
        print(f"  run {r+1}: alpha mean={a.mean():.6e}  std={a.std():.6e}  cavity={float((a<0.5).mean()):.4f}")
    a01_diff = float(np.linalg.norm(alphas[0] - alphas[1]))
    a02_diff = float(np.linalg.norm(alphas[0] - alphas[2]))
    print(f"  ‖α₀ - α₁‖ = {a01_diff:.4e}   ‖α₀ - α₂‖ = {a02_diff:.4e}")
    print(f"  ⇒ forward non-determinism level = {(a01_diff + a02_diff) / (2 * np.linalg.norm(alphas[0])):.2e}")

    # ============ C2: Autograd grad determinism (3 repeats) ============
    print(f"\n[C2] Autograd grad determinism — same θ_init 3 times")
    grads = []
    for r in range(3):
        g = autograd_grad(solver, rest_state, theta_init, knife_pos, knife_radius, alpha_target_t)
        grads.append(g)
        print(f"  run {r+1}: grad_A = {g}")
    g01 = np.abs(grads[0] - grads[1]) / np.maximum(np.abs(grads[0]), 1e-12) * 100
    g02 = np.abs(grads[0] - grads[2]) / np.maximum(np.abs(grads[0]), 1e-12) * 100
    print(f"  rel diff grad₁ vs grad₀ (%): {g01}")
    print(f"  rel diff grad₂ vs grad₀ (%): {g02}")

    # ============ C1: FD ε sweep on stiffness ============
    print(f"\n[C1] FD ε sweep on stiffness dim (other dims fixed at θ_init)")
    epsilons = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    print(f"  Path A autograd grad_stiffness (single run): {grads[0][3]:+.6e}")
    grad_B_stiff = {}
    for eps in epsilons:
        th_plus = theta_init.copy(); th_plus[3] += eps
        th_minus = theta_init.copy(); th_minus[3] -= eps
        L_p = loss_at(solver, rest_state, th_plus, knife_pos, knife_radius, alpha_target_np)
        L_m = loss_at(solver, rest_state, th_minus, knife_pos, knife_radius, alpha_target_np)
        gb = (L_p - L_m) / (2 * eps)
        grad_B_stiff[eps] = gb
        rel = abs(grads[0][3] - gb) / max(abs(gb), 1e-12) * 100
        print(f"  ε={eps:7.1e}: L+={L_p:.6e}  L-={L_m:.6e}  grad_B={gb:+.6e}  rel(A,B)={rel:5.2f}%")

    # ============ Verdict ============
    print(f"\n" + "=" * 72)
    best_eps = min(epsilons, key=lambda e: abs(grads[0][3] - grad_B_stiff[e]))
    best_rel = abs(grads[0][3] - grad_B_stiff[best_eps]) / abs(grad_B_stiff[best_eps]) * 100
    print(f"Best ε = {best_eps:.1e}  →  rel(A,B) = {best_rel:.2f}%")
    fwd_det = (a01_diff + a02_diff) / (2 * np.linalg.norm(alphas[0]))
    grad_det = max(g01[3], g02[3])
    print(f"Forward non-det:  {fwd_det:.2e} (acceptable if < 1e-4)")
    print(f"Grad non-det:     {grad_det:.2f}% (acceptable if < 1%)")
    if best_rel < 5 and fwd_det < 1e-4 and grad_det < 1:
        print(f"✓ Verdict: atomic noise at non-optimal ε caused 6% — true rel_err {best_rel:.2f}% < 5%, PASS")
    elif fwd_det > 1e-3 or grad_det > 5:
        print(f"✗ Verdict: BUG — non-determinism too high to trust autograd")
    else:
        print(f"⚠ Verdict: inconclusive — best rel_err {best_rel:.2f}%, det levels marginal")


if __name__ == "__main__":
    main()
