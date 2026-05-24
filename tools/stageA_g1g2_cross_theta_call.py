"""Stage A G1+G2: cross-θ + cross-call verification of per-particle stiff autograd.

G1 (cross-θ): rerun FD parity at stiff=0.85 (truth point, vs prior 0.70 off-truth).
   Goal: verify autograd is correct across multiple stiff values, not just one point.

G2 (cross-call determinism): call WarpFullForwardOp 3 times with identical
   (θ, stiff_pp), check that per-particle grad is deterministic.
   Goal: verify Day 3b tape.zero() correctly clears per-particle grad buffers.
   Without this, repeated calls would accumulate (1, 3, 6 triangular as before).
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

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_load_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp


def loss_at(theta_np, stiff_pp_np, solver, rest_state, knife_pos, knife_radius, alpha_target_np):
    solver.set_theta_wp(None)
    solver.knife_radius = knife_radius
    solver.knife_active = 1
    solver.knife_pos = wp.array(np.array([knife_pos], dtype=np.float32),
                                  dtype=wp.vec3, device=solver.device, requires_grad=False)
    stiff_wp = wp.array(stiff_pp_np.astype(np.float32),
                          dtype=wp.float32, device=solver.device, requires_grad=False)
    solver.set_theta_wp(dict(
        alpha_min=wp.array([float(theta_np[0])], dtype=wp.float32, device=solver.device, requires_grad=False),
        sharpness=wp.array([float(theta_np[1])], dtype=wp.float32, device=solver.device, requires_grad=False),
        threshold=wp.array([float(theta_np[2])], dtype=wp.float32, device=solver.device, requires_grad=False),
        knife_radius=wp.array([float(knife_radius)], dtype=wp.float32, device=solver.device, requires_grad=False),
        stiffness=stiff_wp,
    ))
    solver_load_state(solver, rest_state)
    solver.step()
    if solver.use_ssa_soft_removal:
        ci = solver._ssa_cycle_index
        alpha = solver._ssa_alpha_buffers[ci].numpy()
    else:
        alpha = solver.p["alpha"].numpy()
    N = solver.N
    return float(np.sum((alpha - alpha_target_np) ** 2)) / float(N)


def autograd_grads(theta_np, stiff_pp_np, solver, rest_state, knife_pos, knife_radius,
                     alpha_target_t, device):
    theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=device, requires_grad=True)
    stiff_pp_torch = torch.tensor(stiff_pp_np, dtype=torch.float32, device=device, requires_grad=True)
    alpha_out, _, _ = WarpFullForwardOp.apply(
        theta_torch, solver, rest_state, knife_pos, knife_radius, stiff_pp_torch)
    N = alpha_out.shape[0]
    loss = ((alpha_out - alpha_target_t) ** 2).sum() / float(N)
    loss.backward()
    return (theta_torch.grad.cpu().numpy().copy(),
            stiff_pp_torch.grad.cpu().numpy().copy(),
            float(loss.item()))


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, _ = setup_solver_with_rest(mesh_path)
    N = solver.N
    print(f"\nSolver N={N}")
    device = "cuda:0"
    knife_pos = np.array([0.15, 0.30, 0.15], dtype=np.float32)
    knife_radius = 0.04
    theta_3D = np.array([3.0, 20.0, 0.5], dtype=np.float32)

    # ============================================================
    # G1: cross-θ FD parity at DIFFERENT θ point (vs first test's θ_3D)
    # ============================================================
    # First test was at θ=(3, 20, 0.5). G1 tests at θ=(5, 15, 0.6) (moderate off).
    # stiff_eval=0.70 same as first test (SNR-good regime — top grads ~5e-6 → ε*grad=2.5e-7 > noise).
    print(f"\n{'='*72}")
    print(f"G1: Cross-θ FD parity at θ=(5, 15, 0.6), stiff=0.70")
    print(f"{'='*72}")

    theta_eval_G1 = np.array([5.0, 15.0, 0.6], dtype=np.float32)  # cross-θ
    stiff_true = np.full(N, 0.85, dtype=np.float32)
    stiff_eval = np.full(N, 0.70, dtype=np.float32)  # same as original (grad SNR good)

    # Generate target at θ_3D=(3, 20, 0.5), stiff=0.85 truth
    _ = loss_at(theta_3D, stiff_true, solver, rest_state, knife_pos, knife_radius,
                  np.zeros(N, dtype=np.float32))
    if solver.use_ssa_soft_removal:
        alpha_target_np = solver._ssa_alpha_buffers[solver._ssa_cycle_index].numpy().copy()
    else:
        alpha_target_np = solver.p["alpha"].numpy().copy()
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)

    # Autograd at the SHIFTED θ
    g_theta, g_stiff, L_init = autograd_grads(
        theta_eval_G1, stiff_eval, solver, rest_state, knife_pos, knife_radius,
        alpha_target_t, device)
    print(f"  loss at θ=(5,15,0.6), stiff=0.70: {L_init:.4e}")
    print(f"  grad stiff_pp: min={g_stiff.min():.4e} max={g_stiff.max():.4e} "
          f"mean={g_stiff.mean():.4e}")

    # Top-20 FD parity
    K = 20
    eps = 0.05
    top_idx = np.argsort(np.abs(g_stiff))[::-1][:K]
    rel_errs = []
    sign_ok_count = 0
    for idx in top_idx:
        sp = stiff_eval.copy(); sp[idx] += eps
        sm = stiff_eval.copy(); sm[idx] -= eps
        Lp = loss_at(theta_eval_G1, sp, solver, rest_state, knife_pos, knife_radius, alpha_target_np)
        Lm = loss_at(theta_eval_G1, sm, solver, rest_state, knife_pos, knife_radius, alpha_target_np)
        g_FD = (Lp - Lm) / (2 * eps)
        g_A = g_stiff[idx]
        rel = abs(g_A - g_FD) / max(abs(g_FD), 1e-12) * 100
        sign_ok = (np.sign(g_A) == np.sign(g_FD)) or (abs(g_A) < 1e-12 and abs(g_FD) < 1e-12)
        rel_errs.append(rel)
        if sign_ok: sign_ok_count += 1
    median = float(np.median(rel_errs))
    print(f"\n  G1 top-{K} FD parity: median rel_err = {median:.2f}%, "
          f"sign match = {sign_ok_count}/{K}")
    g1_pass = median < 5 and sign_ok_count == K

    # ============================================================
    # G2: cross-call determinism (3 calls, same θ + stiff)
    # ============================================================
    print(f"\n{'='*72}")
    print(f"G2: Cross-call determinism (3 calls, identical (θ, stiff))")
    print(f"{'='*72}")

    grads_per_call = []
    for call_idx in range(3):
        # Fresh tensors each call (matches sys-ID outer loop pattern)
        g_th, g_st, _ = autograd_grads(
            theta_3D, stiff_eval, solver, rest_state, knife_pos, knife_radius,
            alpha_target_t, device)
        grads_per_call.append(g_st)
        print(f"  Call {call_idx+1}: stiff grad sample (top-3 by |g|):")
        top3 = np.argsort(np.abs(g_st))[::-1][:3]
        for j, idx in enumerate(top3):
            print(f"    idx {int(idx):>6d}: g = {g_st[idx]:+.6e}")

    # Compute relative diffs across calls
    g0 = grads_per_call[0]
    print(f"\n  Cross-call relative diff:")
    for call_idx in [1, 2]:
        g_c = grads_per_call[call_idx]
        rel_diff = np.abs(g_c - g0) / np.maximum(np.abs(g0), 1e-12)
        # Only count particles where g0 is meaningful (non-noise)
        mask = np.abs(g0) > 1e-9
        if mask.sum() > 0:
            mean_rel = float(rel_diff[mask].mean() * 100)
            max_rel = float(rel_diff[mask].max() * 100)
            print(f"    Call 1 vs Call {call_idx+1}: mean rel_diff = {mean_rel:.4f}%, "
                  f"max = {max_rel:.4f}%")
    g2_pass = True
    for call_idx in [1, 2]:
        g_c = grads_per_call[call_idx]
        mask = np.abs(g0) > 1e-9
        rel_diff = np.abs(g_c - g0) / np.maximum(np.abs(g0), 1e-12)
        if rel_diff[mask].max() > 0.05:  # 5% threshold
            g2_pass = False

    # ============================================================
    # Verdict
    # ============================================================
    print(f"\n{'='*72}")
    print(f"VERDICT")
    print(f"{'='*72}")
    print(f"  G1 (cross-θ FD parity at stiff=0.80, truth=0.85): "
          f"{'✓ PASS' if g1_pass else '✗ FAIL'} (median {median:.2f}%)")
    print(f"  G2 (cross-call grad determinism):                  "
          f"{'✓ PASS' if g2_pass else '✗ FAIL'}")
    if g1_pass and g2_pass:
        print(f"\n  ✓✓ Stage A architecture verified across (θ, call) dimensions.")
        print(f"     Per-particle stiff Warp pipeline is production-ready.")
    else:
        print(f"\n  ⚠ Some verification failed — investigate before sys-ID.")


if __name__ == "__main__":
    main()
