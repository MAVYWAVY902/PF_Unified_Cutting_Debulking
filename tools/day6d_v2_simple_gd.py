"""Day 6d v2: simpler diagnostic — simple GD inner loop (like Liang's actual pattern).

Day 6d v1 failed because:
  - 30-step Adam-in-graph with create_graph=True compounds m, v as autograd
    tensors; chain depth ≈ 30, float underflow → NaN in sharpness/τ gradients
  - FD reused the same solver across 6 evals; wp.Tape stale references caused
    R-W-R warnings + corrupt FD readings

This v2 fixes both:
  - Inner: SIMPLE GD (Liang's exact pattern: Δ ← Δ - lr·grad), no moments
  - FD: use legacy float path (no wp.Tape) so no R-W-R pollution
  - Fewer iterations (10) to keep graph manageable
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


def simple_gd_inner_loop(
    pos_pred, dv_pred, pos_target, alpha_target, theta,
    edges_a, edges_b, edges_rest,
    n_iter=10, lr_pos=5e-3, lr_dv=0.1,
    lambda_cav=1.0, lambda_E=1.0,
):
    """Liang Eq.6 inner loop with SIMPLE gradient descent (not Adam).

    Δ ← Δ - lr·∇_Δ L_inner   [out-of-place, in autograd graph]

    Liang paper uses exactly this — "30 gradient descent steps with lr=50".
    No moments, no adaptive scaling. Simplest possible autograd-friendly pattern.
    """
    N = pos_pred.shape[0]
    s = theta[1]; tau = theta[2]

    delta_x = torch.zeros((N, 3), dtype=pos_pred.dtype, device=pos_pred.device,
                            requires_grad=True)
    delta_dv_raw = torch.zeros(N, dtype=pos_pred.dtype, device=pos_pred.device,
                                  requires_grad=True)

    L_init = None
    for k in range(n_iter):
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
        L_E = ((edge_len - edges_rest) ** 2).sum()

        L_inner = L_pos + lambda_cav * L_cav + lambda_E * L_E
        if L_init is None:
            L_init = float(L_inner)

        grad_x, grad_raw = torch.autograd.grad(
            L_inner, (delta_x, delta_dv_raw),
            create_graph=True, retain_graph=True,
        )

        # Simple GD — single op, minimal graph depth
        delta_x = delta_x - lr_pos * grad_x
        delta_dv_raw = delta_dv_raw - lr_dv * grad_raw

    delta_dv_final = torch.nn.functional.softplus(delta_dv_raw)
    return delta_x, delta_dv_final, dict(L_init=L_init, L_final=float(L_inner))


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_pos = np.array([0.15, 0.30, 0.15], dtype=np.float32)
    knife_radius = 0.04
    theta_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)

    # ---- Generate obs (legacy float path, no tape) ----
    print("\nGenerating obs target via legacy float path …")
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

    # Torch tensors (constants for outer loop)
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_rest_t = torch.tensor(solver.e["pf_rest"].numpy(),
                                  dtype=torch.float32, device=device)
    pos_target_t = torch.tensor(pos_target_np, dtype=torch.float32, device=device)
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)

    def run_chain(theta_torch, n_inner=10):
        """θ → Warp forward → torch inner GD → L_outer"""
        alpha, pos, dv = WarpFullForwardOp.apply(
            theta_torch, solver, rest_state, knife_pos, knife_radius,
        )
        delta_x, delta_dv, info_inner = simple_gd_inner_loop(
            pos_pred=pos, dv_pred=dv,
            pos_target=pos_target_t, alpha_target=alpha_target_t,
            theta=theta_torch,
            edges_a=edges_a_t, edges_b=edges_b_t, edges_rest=edges_rest_t,
            n_iter=n_inner, lr_pos=5e-3, lr_dv=0.1,
            lambda_cav=1.0, lambda_E=1.0,
        )
        L_outer = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
        return L_outer, info_inner

    # ====== Path A: autograd through full chain ======
    print("\n[Path A] end-to-end autograd backward (n_inner=10, simple GD) …")
    theta_torch = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                 requires_grad=True)
    L_outer, info_inner = run_chain(theta_torch, n_inner=10)
    print(f"  L_inner: {info_inner['L_init']:.4e} → {info_inner['L_final']:.4e}")
    print(f"  L_outer = {L_outer.item():.6e}")
    L_outer.backward()
    grad_A = theta_torch.grad.cpu().numpy().copy()
    print(f"  grad = ({grad_A[0]:+.4e}, {grad_A[1]:+.4e}, {grad_A[2]:+.4e})")
    has_nan = bool(np.isnan(grad_A).any())
    if has_nan:
        print(f"  ⚠ NaN in gradient — autograd chain still broken")

    # ====== Path B: pure FD (legacy float, no wp.Tape) ======
    print("\n[Path B] FD ground truth (legacy float path, no tape pollution) …")

    def L_outer_at_no_tape(theta_np, n_inner=10):
        """Forward via legacy float path; inner loop in torch without create_graph."""
        # Forward (no tape)
        solver.set_theta_wp(None)
        solver.alpha_min = float(theta_np[0])
        solver.soft_removal_sharpness = float(theta_np[1])
        solver.aspiration_threshold = float(theta_np[2])
        solver.knife_radius = knife_radius
        solver.knife_active = 1
        solver.knife_pos = wp.array(np.array([knife_pos], dtype=np.float32),
                                      dtype=wp.vec3, device=solver.device,
                                      requires_grad=False)
        solver_load_state(solver, rest_state)
        solver.step()

        if solver.use_ssa_soft_removal:
            ci = solver._ssa_cycle_index
            alpha_np = solver._ssa_alpha_buffers[ci].numpy()
            dv_np = solver._ssa_particle_damage_buffers[ci].numpy()
        else:
            alpha_np = solver.p["alpha"].numpy()
            dv_np = solver.p["damage"].numpy()
        pos_np = solver.p["pos"].numpy()

        # Inner loop in torch (no create_graph, no autograd needed for FD)
        pos_t = torch.tensor(pos_np, dtype=torch.float32, device=device)
        dv_t = torch.tensor(dv_np, dtype=torch.float32, device=device)
        theta_t = torch.tensor(theta_np, dtype=torch.float32, device=device)

        N = pos_t.shape[0]; s = theta_t[1]; tau = theta_t[2]
        delta_x = torch.zeros((N, 3), dtype=torch.float32, device=device,
                                requires_grad=True)
        delta_dv_raw = torch.zeros(N, dtype=torch.float32, device=device,
                                      requires_grad=True)

        for k in range(n_inner):
            pos_corr = pos_t + delta_x
            delta_dv = torch.nn.functional.softplus(delta_dv_raw)
            dv_corr = dv_t + delta_dv
            alpha_corr = torch.sigmoid(-s * (dv_corr - tau))
            L_pos = ((pos_corr - pos_target_t) ** 2).sum()
            L_cav = ((alpha_corr - alpha_target_t) ** 2).sum()
            edge_vec = pos_corr[edges_a_t] - pos_corr[edges_b_t]
            edge_len = edge_vec.norm(dim=1).clamp(min=1e-12)
            L_E = ((edge_len - edges_rest_t) ** 2).sum()
            L_inner = L_pos + L_cav + L_E
            grad_x, grad_raw = torch.autograd.grad(
                L_inner, (delta_x, delta_dv_raw),
                retain_graph=False,
            )
            with torch.no_grad():
                delta_x = (delta_x - 5e-3 * grad_x).requires_grad_(True)
                delta_dv_raw = (delta_dv_raw - 0.1 * grad_raw).requires_grad_(True)

        delta_dv_final = torch.nn.functional.softplus(delta_dv_raw)
        return float((delta_x ** 2).sum() + (delta_dv_final ** 2).sum())

    eps_per_dim = [1e-2, 1e-1, 1e-2]
    grad_B = np.zeros(3)
    for d in range(3):
        eps = eps_per_dim[d]
        th_plus = theta_init.copy(); th_plus[d] += eps
        th_minus = theta_init.copy(); th_minus[d] -= eps
        L_p = L_outer_at_no_tape(th_plus, n_inner=10)
        L_m = L_outer_at_no_tape(th_minus, n_inner=10)
        grad_B[d] = (L_p - L_m) / (2 * eps)
        print(f"  d={d} ({['α_min','sharp','τ'][d]}): "
              f"L(+ε)={L_p:.4e}  L(-ε)={L_m:.4e}  grad={grad_B[d]:+.4e}")

    # ====== Parity ======
    print("\n" + "=" * 64)
    print("Day 6d v2 parity: simple-GD autograd ↔ FD")
    print("=" * 64)
    if has_nan:
        print("  Path A has NaN — chain still broken")
    else:
        rel_err = np.abs(grad_A - grad_B) / np.maximum(np.abs(grad_B), 1e-12)
        sign_match = np.sign(grad_A) == np.sign(grad_B)
        for d, name in enumerate(["α_min", "sharpness", "τ"]):
            print(f"  {name:>10s}: autograd={grad_A[d]:+.4e}  "
                  f"FD={grad_B[d]:+.4e}  rel_err={rel_err[d]*100:.3f}%  "
                  f"sign {'✓' if sign_match[d] else '✗'}")
        pass_tight = bool(sign_match.all() and (rel_err < 0.05).all())
        if pass_tight:
            print(f"\n✓ Day 6d v2 PASS — simple GD autograd matches FD < 5%")
        elif sign_match.all() and (rel_err < 0.2).all():
            print(f"\n⚠ Day 6d v2 partial — signs OK, magnitudes < 20% but not < 5%")
        else:
            print(f"\n✗ Day 6d v2 FAIL")


if __name__ == "__main__":
    main()
