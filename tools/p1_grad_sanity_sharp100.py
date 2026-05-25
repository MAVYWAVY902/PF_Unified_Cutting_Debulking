"""P1: Forward grad sanity at theta=(3, 100, 0.3) vs (3, 20, 0.5) baseline.

Run 1 forward + backward at each theta. Compare:
   - finite-ness of all grads (alpha_min, sharpness, threshold, stiff_pp top-K)
   - magnitude ratio sharp_100 / sharp_20 — predicted ~5×
   - max grad / mean grad ratio (gradient cliff detection)

If sharp=100 gives < 10× larger grad and no NaN → safe to proceed to P2.
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
from tools.day6f_v0_multistep_synth import make_knife_trajectory  # noqa


def grad_at_theta(solver, rest_state, theta_np, knife_radius, knife_pos,
                    stiff_pp_np, alpha_target_np, device):
    """Run 1 forward + backward at given theta. Return all 4 grads."""
    solver_load_state(solver, rest_state)
    theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=device, requires_grad=True)
    stiff_pp_torch = torch.tensor(stiff_pp_np, dtype=torch.float32, device=device, requires_grad=True)
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)

    alpha_out, _, _ = WarpFullForwardOp.apply(
        theta_torch, solver, rest_state, knife_pos, knife_radius, stiff_pp_torch)
    N = alpha_out.shape[0]
    loss = ((alpha_out - alpha_target_t) ** 2).sum() / float(N)
    loss.backward()

    g_theta = theta_torch.grad.cpu().numpy().copy()
    g_stiff = stiff_pp_torch.grad.cpu().numpy().copy()
    return g_theta, g_stiff, float(loss.item())


def report(label, g_theta, g_stiff, loss):
    print(f"\n  {label}")
    print(f"  loss = {loss:.6e}")
    print(f"  grad alpha_min : {g_theta[0]:+.4e}")
    print(f"  grad sharpness : {g_theta[1]:+.4e}")
    print(f"  grad threshold : {g_theta[2]:+.4e}")
    sm, smin, smax = g_stiff.mean(), g_stiff.min(), g_stiff.max()
    abs_g = np.abs(g_stiff)
    nz = abs_g[abs_g > 1e-12]
    print(f"  grad stiff_pp  : mean {sm:+.4e}  range [{smin:+.4e}, {smax:+.4e}]")
    if len(nz) > 0:
        print(f"                   |g| nz: median {np.median(nz):.4e}  "
              f"max {abs_g.max():.4e}  ({len(nz)}/{len(g_stiff)} non-zero)")
    nan_count = int(np.isnan(g_stiff).sum()) + int(np.isnan(g_theta).sum())
    inf_count = int(np.isinf(g_stiff).sum()) + int(np.isinf(g_theta).sum())
    print(f"  finite check   : NaN={nan_count}  Inf={inf_count}")
    return abs_g.max(), float(np.median(nz)) if len(nz) > 0 else 0.0


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
    stiff_pp = np.full(N, 0.5, dtype=np.float32)

    # Generate target: at theta_A truth, run forward to get alpha_target_A
    theta_A = np.array([3.0, 20.0, 0.5], dtype=np.float32)
    theta_B = np.array([3.0, 100.0, 0.3], dtype=np.float32)

    # Use same target = zeros (just compute grad direction, not realistic loss)
    alpha_target = np.zeros(N, dtype=np.float32)

    # ========== Theta A (baseline) ==========
    g_th_A, g_st_A, loss_A = grad_at_theta(
        solver, rest_state, theta_A, knife_radius, knife_pos,
        stiff_pp, alpha_target, device)
    max_A, med_A = report("===== Config A: theta=(3, 20, 0.5) =====",
                              g_th_A, g_st_A, loss_A)
    solver_load_state(solver, rest_state)

    # ========== Theta B (Config B) ==========
    g_th_B, g_st_B, loss_B = grad_at_theta(
        solver, rest_state, theta_B, knife_radius, knife_pos,
        stiff_pp, alpha_target, device)
    max_B, med_B = report("===== Config B: theta=(3, 100, 0.3) =====",
                              g_th_B, g_st_B, loss_B)

    # ========== Comparison ==========
    print(f"\n{'='*70}")
    print(f"COMPARISON (B vs A)")
    print(f"{'='*70}")
    for i, name in enumerate(["alpha_min", "sharpness", "threshold"]):
        a, b = abs(g_th_A[i]), abs(g_th_B[i])
        if a > 1e-12:
            print(f"  grad {name:>10}: |A| = {a:.3e}  |B| = {b:.3e}  ratio B/A = {b/a:.2f}×")
        else:
            print(f"  grad {name:>10}: |A| = {a:.3e}  |B| = {b:.3e}  (A near-zero)")
    print(f"  stiff median |g|: A = {med_A:.3e}  B = {med_B:.3e}  ratio B/A = {med_B/med_A:.2f}×" if med_A > 0 else "")
    print(f"  stiff max    |g|: A = {max_A:.3e}  B = {max_B:.3e}  ratio B/A = {max_B/max_A:.2f}×")

    # ========== Verdict ==========
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")
    stiff_ratio = max_B / max_A
    has_nan = bool(np.isnan(g_st_B).any() or np.isnan(g_th_B).any())
    has_inf = bool(np.isinf(g_st_B).any() or np.isinf(g_th_B).any())
    print(f"  Stiff max grad ratio B/A = {stiff_ratio:.2f}×  (predict ~5×)")
    print(f"  B has NaN: {has_nan},   has Inf: {has_inf}")
    if has_nan or has_inf:
        print(f"\n  ✗ P1 FAIL — numerical issues, try sharp=50 or 70 instead")
    elif stiff_ratio > 15:
        print(f"\n  ⚠ P1 BORDERLINE — gradient {stiff_ratio:.1f}× larger may need lr/{stiff_ratio:.0f}")
    else:
        print(f"\n  ✓ P1 PASS — grad scaling within expected range, proceed to P2")
        print(f"     New Adam lr should be roughly old_lr / {stiff_ratio:.1f}")


if __name__ == "__main__":
    main()
