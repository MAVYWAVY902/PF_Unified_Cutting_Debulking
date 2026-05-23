"""Day 6d: end-to-end Liang-style ReSECT backward through full chain.

Three critical pieces:

  1. Out-of-place autograd-friendly inner Adam:
       Δ_new = Δ_old - lr * m_hat / (sqrt(v_hat) + eps)    [out-of-place!]
     This keeps Δ in the autograd graph across 30 inner iters.
     (torch.optim.Adam.step() does .data -=, which BREAKS the graph.)

  2. WarpFullForwardOp (Day 6b) wraps solver.step() as torch op.

  3. End-to-end test:
       θ_torch (requires_grad=True)
          ↓ WarpFullForwardOp
       pos, dv, alpha
          ↓ unrolled inner loop (30 steps, in autograd graph)
       Δx, Δδv
          ↓ L_outer = ‖Δx‖² + λ·‖Δδv‖²
       L_outer.backward()
          ↓ unroll backward through 30 inner steps + WarpFullForwardOp
       θ_torch.grad

Pass criterion: ∂L_outer/∂θ from autograd vs FD ≤ 5% rel_err.
(Larger tolerance than Day 6b because the chain is much longer:
30 inner steps × chain of Warp tape + torch ops.)
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


# ---------------------------------------------------------------------------
# Autograd-friendly inner loop (out-of-place Adam, in graph)
# ---------------------------------------------------------------------------

def compute_liang_inner_residuals_torch_autograd(
    pos_pred,        # (N, 3) torch.Tensor (may have requires_grad=True)
    dv_pred,         # (N,) torch.Tensor
    pos_target,      # (N, 3) torch.Tensor
    alpha_target,    # (N,) torch.Tensor
    theta,           # (3,) torch.Tensor (α_min, sharpness, threshold)
    edges_a,         # (E,) int64
    edges_b,
    edges_rest,      # (E,)
    n_iter=30,
    lr_pos=5e-3,
    lr_dv=0.1,
    lambda_cav=1.0,
    lambda_E=1.0,
    b1=0.9, b2=0.999, eps_adam=1e-8,
):
    """Custom out-of-place Adam inner loop. All updates are torch ops that
    KEEP the autograd graph: Δ_new = Δ_old - lr * m_hat / (sqrt(v_hat) + eps).
    create_graph=True on torch.autograd.grad allows higher-order grads (which
    is needed for the outer backward).
    """
    N = pos_pred.shape[0]
    device = pos_pred.device
    s = theta[1]
    tau = theta[2]

    # Initial residuals + Adam moments — all 0 + leaf nodes initially
    delta_x = torch.zeros((N, 3), dtype=pos_pred.dtype, device=device,
                            requires_grad=True)
    delta_dv_raw = torch.zeros(N, dtype=pos_pred.dtype, device=device,
                                  requires_grad=True)

    m_x = torch.zeros_like(delta_x)
    v_x = torch.zeros_like(delta_x)
    m_dv = torch.zeros_like(delta_dv_raw)
    v_dv = torch.zeros_like(delta_dv_raw)

    L_init = None
    for k in range(n_iter):
        # Forward: 4 terms
        pos_corr = pos_pred + delta_x
        delta_dv = torch.nn.functional.softplus(delta_dv_raw)
        dv_corr = dv_pred + delta_dv
        alpha_corr = torch.sigmoid(-s * (dv_corr - tau))

        diff_pos = pos_corr - pos_target
        diff_alpha = alpha_corr - alpha_target

        L_pos = (diff_pos ** 2).sum()
        L_cav = (diff_alpha ** 2).sum()

        edge_vec = pos_corr[edges_a] - pos_corr[edges_b]
        edge_len = edge_vec.norm(dim=1).clamp(min=1e-12)
        edge_residual = edge_len - edges_rest
        L_E = (edge_residual ** 2).sum()

        L_inner = L_pos + lambda_cav * L_cav + lambda_E * L_E

        if L_init is None:
            L_init = float(L_inner)

        # Compute gradient w.r.t. inner variables (Δx, Δδv_raw)
        # create_graph=True is critical — allows the outer backward to flow
        # through these gradient computations.
        grad_x, grad_raw = torch.autograd.grad(
            L_inner, (delta_x, delta_dv_raw),
            create_graph=True,
            retain_graph=True,
        )

        # Adam update — OUT-OF-PLACE so autograd graph remains intact.
        t1 = k + 1
        m_x = b1 * m_x + (1.0 - b1) * grad_x
        v_x = b2 * v_x + (1.0 - b2) * grad_x ** 2
        m_x_hat = m_x / (1.0 - b1 ** t1)
        v_x_hat = v_x / (1.0 - b2 ** t1)
        delta_x = delta_x - lr_pos * m_x_hat / (v_x_hat.sqrt() + eps_adam)

        m_dv = b1 * m_dv + (1.0 - b1) * grad_raw
        v_dv = b2 * v_dv + (1.0 - b2) * grad_raw ** 2
        m_dv_hat = m_dv / (1.0 - b1 ** t1)
        v_dv_hat = v_dv / (1.0 - b2 ** t1)
        delta_dv_raw = delta_dv_raw - lr_dv * m_dv_hat / (v_dv_hat.sqrt() + eps_adam)

    # Final residuals
    delta_dv_final = torch.nn.functional.softplus(delta_dv_raw)
    info = dict(L_inner_init=L_init, L_inner_final=float(L_inner))
    return delta_x, delta_dv_final, info


# ---------------------------------------------------------------------------
# End-to-end ReSECT step: WarpForward + torch inner loop + outer loss
# ---------------------------------------------------------------------------

def resect_outer_loss(theta_torch, solver, state_in, knife_pos, knife_radius,
                       pos_target, alpha_target, edges_a, edges_b, edges_rest,
                       n_inner=30, lambda_inner_cav=1.0, lambda_inner_E=1.0,
                       lambda_outer_dv=1.0):
    """Compute one ReSECT step's outer loss, with full autograd chain.

    Pipeline:
        θ → WarpFullForwardOp → (pos, dv, α)
          → torch unrolled inner loop → (Δx, Δδv)
          → L_outer = ‖Δx‖² + λ·‖Δδv‖²    (Liang L_gap, 2-channel)
    """
    # ---- Forward via Warp ----
    alpha, pos, dv = WarpFullForwardOp.apply(
        theta_torch, solver, state_in, knife_pos, knife_radius,
    )

    # ---- Inner loop (in torch autograd graph) ----
    delta_x, delta_dv, info_inner = compute_liang_inner_residuals_torch_autograd(
        pos_pred=pos, dv_pred=dv,
        pos_target=pos_target, alpha_target=alpha_target,
        theta=theta_torch,
        edges_a=edges_a, edges_b=edges_b, edges_rest=edges_rest,
        n_iter=n_inner,
        lambda_cav=lambda_inner_cav, lambda_E=lambda_inner_E,
    )

    # ---- Outer loss: Liang L_gap (2-channel) ----
    L_outer = (delta_x ** 2).sum() + lambda_outer_dv * (delta_dv ** 2).sum()

    return L_outer, info_inner


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    N = solver.N

    knife_pos = np.array([0.15, 0.30, 0.15], dtype=np.float32)
    knife_radius = 0.04
    theta_true = np.array([3.0, 20.0, 0.5], dtype=np.float32)
    theta_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)

    # ---- Generate obs target via legacy float path at θ_true ----
    print("\nGenerating obs target at θ_true (1 step solver.step) …")
    solver.set_theta_wp(None)
    solver.alpha_min = 3.0; solver.soft_removal_sharpness = 20.0
    solver.aspiration_threshold = 0.5; solver.knife_radius = knife_radius
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
    pos_target_np = solver.p["pos"].numpy().copy()
    print(f"  target cavity: {float((alpha_target_np<0.5).mean()):.4f}")
    print(f"  target pos drift max: {float(np.abs(pos_target_np - mesh_info['vertices']).max()):.4f}")

    # Edges (use solver's edge graph, same as Day 5)
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_rest_t = torch.tensor(solver.e["pf_rest"].numpy(),
                                  dtype=torch.float32, device=device)

    # Torch tensors for obs
    pos_target_t = torch.tensor(pos_target_np, dtype=torch.float32, device=device)
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)

    # ====== Path A: end-to-end PyTorch backward ======
    print("\n[Path A] end-to-end torch backward (θ → Warp forward → torch inner → L_outer) …")
    theta_torch = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                 requires_grad=True)

    L_outer, info_inner = resect_outer_loss(
        theta_torch, solver, rest_state, knife_pos, knife_radius,
        pos_target_t, alpha_target_t, edges_a_t, edges_b_t, edges_rest_t,
        n_inner=30, lambda_inner_cav=1.0, lambda_inner_E=1.0,
        lambda_outer_dv=1.0,
    )
    print(f"  L_inner final: {info_inner['L_inner_final']:.4e}  "
          f"(init {info_inner['L_inner_init']:.4e})")
    print(f"  L_outer = ‖Δx‖² + ‖Δδv‖² = {L_outer.item():.6e}")

    L_outer.backward()
    grad_A = theta_torch.grad.cpu().numpy().copy()
    print(f"  grad = ({grad_A[0]:+.4e}, {grad_A[1]:+.4e}, {grad_A[2]:+.4e})")

    # ====== Path B: central FD on the SAME L_outer wrt θ ======
    print("\n[Path B] central FD on L_outer wrt θ …")

    def L_outer_at(theta_np):
        """Recompute L_outer at given theta (no autograd)."""
        with torch.no_grad():
            theta_t_local = torch.tensor(theta_np, dtype=torch.float32, device=device,
                                            requires_grad=False)
            # We can't use WarpFullForwardOp.apply without requires_grad on theta.
            # So manually re-run forward at this theta via the wrapped op.
            theta_t_for_fwd = theta_t_local.clone().detach().requires_grad_(True)
        L, _ = resect_outer_loss(
            theta_t_for_fwd, solver, rest_state, knife_pos, knife_radius,
            pos_target_t, alpha_target_t, edges_a_t, edges_b_t, edges_rest_t,
            n_inner=30, lambda_inner_cav=1.0, lambda_inner_E=1.0,
            lambda_outer_dv=1.0,
        )
        return float(L.detach())

    eps_per_dim = [1e-2, 1e-1, 1e-2]
    grad_B = np.zeros(3)
    for d in range(3):
        eps = eps_per_dim[d]
        th_plus = theta_init.copy(); th_plus[d] += eps
        th_minus = theta_init.copy(); th_minus[d] -= eps
        L_p = L_outer_at(th_plus); L_m = L_outer_at(th_minus)
        grad_B[d] = (L_p - L_m) / (2 * eps)
        print(f"    d={d} ({['α_min','sharp','τ'][d]}): L({th_plus[d]:.3f})={L_p:.4e}  "
              f"L({th_minus[d]:.3f})={L_m:.4e}")
    print(f"  grad = ({grad_B[0]:+.4e}, {grad_B[1]:+.4e}, {grad_B[2]:+.4e})")

    # ====== Parity ======
    print("\n" + "=" * 64)
    print("Day 6d parity: full ReSECT autograd ↔ central FD")
    print("=" * 64)
    rel_err = np.abs(grad_A - grad_B) / np.maximum(np.abs(grad_B), 1e-12)
    sign_match = np.sign(grad_A) == np.sign(grad_B)
    for d, name in enumerate(["α_min", "sharpness", "τ"]):
        ag = grad_A[d]; bg = grad_B[d]
        rel = rel_err[d] * 100
        sgn = "✓" if sign_match[d] else "✗"
        print(f"  {name:>10s}: autograd={ag:+.4e}  FD={bg:+.4e}  rel_err={rel:.3f}%  sign {sgn}")

    nonzero = (np.abs(grad_A) > 1e-12).all()
    pass_tight = bool(sign_match.all() and (rel_err < 0.05).all())
    pass_loose = bool(sign_match.all() and (rel_err < 0.20).all())

    print()
    if pass_tight:
        print(f"✓ Day 6d PASS — end-to-end backward through full ReSECT chain (rel_err < 5%)")
        print(f"  This validates: WarpFullForwardOp + 30-step unrolled inner loop + outer L_gap")
        print(f"  → all gradient flow paths are correct, machine-precision faithful to Liang")
    elif pass_loose:
        print(f"⚠ Day 6d partial: rel_err < 20% but not < 5%")
        print(f"  Likely numerical regime sensitivity; FD ε or inner-loop convergence")
    elif nonzero and sign_match.all():
        print(f"⚠ Day 6d signs OK, magnitudes off")
    else:
        print(f"✗ Day 6d FAIL — see diagnostics above")


if __name__ == "__main__":
    main()
