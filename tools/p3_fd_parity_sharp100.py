"""P3: Stage A FD parity for stiff_pp at theta=(3, 100, 0.3).

Baseline (Stage A): theta=(3, 20, 0.5), top-20 median rel_err 0.32%.

At sharp=100, sigmoid 5× steeper near transition. ε=0.05 may straddle the
transition and introduce bias. We sweep ε ∈ {0.05, 0.02, 0.01, 0.005} to find
the bias-noise sweet spot (Nesterov-Spokoiny U-curve).

Pass criterion: at least one ε gives top-20 median rel_err < 5%.
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))

import torch
import warp as wp
wp.init()

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_load_state, solver_save_state  # noqa
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa


def loss_at(theta_np, stiff_pp_np, solver, rest_state, knife_pos, knife_radius, alpha_target_np):
    solver_load_state(solver, rest_state)
    theta_torch = torch.tensor(theta_np, dtype=torch.float32, device="cuda:0", requires_grad=False)
    stiff_torch = torch.tensor(stiff_pp_np, dtype=torch.float32, device="cuda:0", requires_grad=False)
    alpha_out, _, _ = WarpFullForwardOp.apply(
        theta_torch, solver, rest_state, knife_pos, knife_radius, stiff_torch)
    N = alpha_out.shape[0]
    target = torch.tensor(alpha_target_np, dtype=torch.float32, device="cuda:0")
    return float(((alpha_out - target) ** 2).sum().item()) / float(N)


def autograd_grad(theta_np, stiff_pp_np, solver, rest_state, knife_pos, knife_radius,
                    alpha_target_t, device):
    solver_load_state(solver, rest_state)
    theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=device, requires_grad=True)
    stiff_pp_torch = torch.tensor(stiff_pp_np, dtype=torch.float32, device=device, requires_grad=True)
    alpha_out, _, _ = WarpFullForwardOp.apply(
        theta_torch, solver, rest_state, knife_pos, knife_radius, stiff_pp_torch)
    N = alpha_out.shape[0]
    loss = ((alpha_out - alpha_target_t) ** 2).sum() / float(N)
    loss.backward()
    return (stiff_pp_torch.grad.cpu().numpy().copy(), float(loss.item()))


def fd_parity(theta_np, stiff_eval_np, alpha_target_np, alpha_target_t,
                solver, rest_state, knife_pos, knife_radius, device,
                eps_sweep=(0.05, 0.02, 0.01, 0.005), K=20):
    # Autograd reference
    g_stiff, L_init = autograd_grad(theta_np, stiff_eval_np, solver, rest_state,
                                       knife_pos, knife_radius, alpha_target_t, device)
    top_idx = np.argsort(np.abs(g_stiff))[::-1][:K]
    print(f"  Loss(eval point) = {L_init:.6e}")
    print(f"  Top-{K} |g_autograd|: max {abs(g_stiff[top_idx[0]]):.3e}  "
          f"median {np.median(np.abs(g_stiff[top_idx])):.3e}")

    results = {}
    for eps in eps_sweep:
        rel_errs = []
        sign_ok = 0
        for idx in top_idx:
            sp = stiff_eval_np.copy(); sp[idx] += eps
            sm = stiff_eval_np.copy(); sm[idx] -= eps
            Lp = loss_at(theta_np, sp, solver, rest_state, knife_pos, knife_radius, alpha_target_np)
            Lm = loss_at(theta_np, sm, solver, rest_state, knife_pos, knife_radius, alpha_target_np)
            g_FD = (Lp - Lm) / (2 * eps)
            g_A = g_stiff[idx]
            rel = abs(g_A - g_FD) / max(abs(g_FD), 1e-12) * 100
            if (np.sign(g_A) == np.sign(g_FD)) or (abs(g_A) < 1e-12 and abs(g_FD) < 1e-12):
                sign_ok += 1
            rel_errs.append(rel)
        median = float(np.median(rel_errs))
        results[eps] = (median, sign_ok)
        print(f"  ε={eps:>6.4f}: median rel_err = {median:6.2f}%  sign match = {sign_ok}/{K}")
    return results


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    N = solver.N
    print(f"\nSolver N={N}")
    solver.num_iterations = 2
    solver.damping = 0.99
    device = "cuda:0"
    knife_radius = 0.08
    knife_pos = np.array([0.15, 0.30, 0.15], dtype=np.float32)

    # Compute alpha_target at truth stiff=0.85 (off-eval), theta=(3,100,0.3)
    theta_B = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    stiff_true = np.full(N, 0.85, dtype=np.float32)
    stiff_eval = np.full(N, 0.70, dtype=np.float32)  # off-truth for non-zero grad

    # Generate target
    _ = loss_at(theta_B, stiff_true, solver, rest_state, knife_pos, knife_radius,
                  np.zeros(N, dtype=np.float32))
    # Save alpha buffer as target
    if solver.use_ssa_soft_removal:
        ci = solver._ssa_cycle_index
        alpha_target_np = solver._ssa_alpha_buffers[ci].numpy().copy()
    else:
        alpha_target_np = solver.p["alpha"].numpy().copy()
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)

    # ========== FD parity test at theta_B ==========
    print(f"\n{'='*72}")
    print(f"P3: FD parity for stiff_pp at theta=(3, 100, 0.3)")
    print(f"{'='*72}")
    print(f"  eval point: stiff=0.70 (truth 0.85)")
    print(f"  ε sweep: [0.05, 0.02, 0.01, 0.005]")
    results = fd_parity(theta_B, stiff_eval, alpha_target_np, alpha_target_t,
                          solver, rest_state, knife_pos, knife_radius, device)

    # ========== Verdict ==========
    best_eps = min(results, key=lambda e: results[e][0])
    best_median, best_sign = results[best_eps]
    print(f"\n{'='*72}")
    print(f"VERDICT")
    print(f"{'='*72}")
    print(f"  Best ε = {best_eps}  →  median {best_median:.2f}%  sign {best_sign}/20")
    print(f"  Stage A baseline at theta=(3,20,0.5) ε=0.05: median 0.32%, sign 20/20")
    if best_median < 5.0:
        print(f"\n  ✓ P3 PASS — autograd correct at new theta (best ε={best_eps})")
        print(f"     Proceed to P4 (Step 2 inclusion recovery at new theta)")
    elif best_median < 20:
        print(f"\n  ⚠ P3 MARGINAL — best {best_median:.1f}% is non-zero but tolerable")
    else:
        print(f"\n  ✗ P3 FAIL — gradient may have issues at sharp=100")


if __name__ == "__main__":
    main()
