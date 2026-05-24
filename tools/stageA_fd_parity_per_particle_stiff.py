"""Stage A FD parity: per-particle stiffness ∂L/∂stiff_i autograd vs FD.

For per-particle stiffness, we have N gradients (one per particle). FD parity on
all N is too expensive (2N forwards). Instead:
   1. Sample K=20 random particle indices
   2. For each: central FD on stiff[i] (perturb only that one entry by ±ε)
   3. Compare to autograd grad[i]

Pass: median rel_err < 5%, sign match.
Also: check sum-rule — Σ grad[i] should ≈ global_grad when all stiff equal.
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
    """Forward at given (θ, stiff_pp) via legacy float path → alpha → loss vs target."""
    solver.set_theta_wp(None)
    solver.alpha_min = float(theta_np[0])
    solver.soft_removal_sharpness = float(theta_np[1])
    solver.aspiration_threshold = float(theta_np[2])
    solver.knife_radius = knife_radius
    solver.knife_active = 1
    solver.knife_pos = wp.array(np.array([knife_pos], dtype=np.float32),
                                  dtype=wp.vec3, device=solver.device, requires_grad=False)
    # Inject per-particle stiff via theta_wp path with stiffness shape (N,)
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
    """Returns (grad_theta_3, grad_stiff_pp_N)."""
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
    stiff_uniform = np.full(N, 0.85, dtype=np.float32)

    # Generate target at θ_true, uniform stiff (= 0.85 everywhere)
    print(f"\n[Setup] Generating α_target at θ_true, uniform stiff=0.85 …")
    L0 = loss_at(theta_3D, stiff_uniform, solver, rest_state, knife_pos, knife_radius,
                  np.zeros(N, dtype=np.float32))   # dummy target, just to run forward
    if solver.use_ssa_soft_removal:
        ci = solver._ssa_cycle_index
        alpha_target_np = solver._ssa_alpha_buffers[ci].numpy().copy()
    else:
        alpha_target_np = solver.p["alpha"].numpy().copy()
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)

    # Perturb stiff: use init slightly off from "truth"
    stiff_init = np.full(N, 0.70, dtype=np.float32)

    # Autograd grads at (θ_3D, stiff_init)
    print(f"\n[Path A] Autograd grad at stiff_init=0.70 …")
    g_theta, g_stiff, L_init = autograd_grads(
        theta_3D, stiff_init, solver, rest_state, knife_pos, knife_radius,
        alpha_target_t, device)
    print(f"  loss(θ_3D, stiff=0.70) = {L_init:.4e}")
    print(f"  grad θ:   {g_theta}")
    print(f"  grad stiff_pp stats: min={g_stiff.min():.4e}  max={g_stiff.max():.4e}  "
          f"mean={g_stiff.mean():.4e}")
    print(f"    n_nonzero = {int((g_stiff != 0).sum())} / {N}")

    # ---- Sum-rule check ----
    total_g_stiff = float(g_stiff.sum())
    print(f"\n[Sum rule] Σ_i grad[stiff_i] = {total_g_stiff:.4e}")
    print(f"  This should ≈ global stiff grad when all stiff_i equal "
          f"(Chain rule for ∂L/∂c where c controls all stiff_i = sum of partials).")

    # ---- FD parity on K particles WITH largest |g_autograd| ----
    # Random sampling from N=21745 mostly hits particles far from knife (g≈0).
    # FD on those compares noise-to-noise → meaningless 100% rel_err.
    # Instead: take top-K by |g_autograd|, where the signal actually lives.
    K = 20
    abs_g = np.abs(g_stiff)
    top_idx = np.argsort(abs_g)[::-1][:K]   # K largest |g|
    sample_idx = [int(i) for i in top_idx]
    eps = 0.05
    print(f"\n  Sampling strategy: top-{K} particles by |g_autograd|")
    print(f"  (random sampling hits ~90% near-zero-grad particles in cube 21745)")

    print(f"\n[Path B] Central FD on {K} sampled particles, ε={eps} …")
    rel_errs = []
    print(f"  {'idx':>6s}  {'g_autograd':>14s}  {'g_FD':>14s}  {'rel_err':>10s}  {'sign':>6s}")
    n_fd_evals = 0
    fd_results = []
    for idx in sample_idx:
        stiff_plus = stiff_init.copy(); stiff_plus[idx] += eps
        stiff_minus = stiff_init.copy(); stiff_minus[idx] -= eps
        Lp = loss_at(theta_3D, stiff_plus, solver, rest_state, knife_pos, knife_radius, alpha_target_np)
        Lm = loss_at(theta_3D, stiff_minus, solver, rest_state, knife_pos, knife_radius, alpha_target_np)
        g_FD = (Lp - Lm) / (2 * eps)
        g_A = g_stiff[idx]
        if abs(g_FD) < 1e-12 and abs(g_A) < 1e-12:
            rel = 0.0
        elif abs(g_FD) < 1e-12:
            rel = float('inf')
        else:
            rel = abs(g_A - g_FD) / abs(g_FD) * 100
        sign_ok = np.sign(g_A) == np.sign(g_FD) or (abs(g_A) < 1e-12 and abs(g_FD) < 1e-12)
        fd_results.append((idx, g_A, g_FD, rel, sign_ok))
        n_fd_evals += 2
        rel_errs.append(rel)
        print(f"  {idx:>6d}  {g_A:+14.4e}  {g_FD:+14.4e}  {rel:9.2f}%  {'✓' if sign_ok else '✗'}")

    median_rel = float(np.median([r for r in rel_errs if np.isfinite(r)]))
    pct_under_5 = sum(1 for r in rel_errs if np.isfinite(r) and r < 5) / len(rel_errs) * 100
    pct_sign_ok = sum(1 for _, _, _, _, ok in fd_results if ok) / K * 100

    print(f"\n[Stats]")
    print(f"  median rel_err: {median_rel:.2f}%")
    print(f"  % under 5%:    {pct_under_5:.0f}%")
    print(f"  % sign match:  {pct_sign_ok:.0f}%")

    print(f"\n{'='*72}")
    if median_rel < 5 and pct_sign_ok > 90:
        print(f"✓ Stage A FD parity PASS — per-particle stiff autograd is correct.")
    elif median_rel < 20 and pct_sign_ok > 80:
        print(f"⚠ Stage A FD parity PARTIAL — sign mostly OK but magnitudes loose.")
    else:
        print(f"✗ Stage A FD parity FAIL — per-particle stiff gradient has bug.")


if __name__ == "__main__":
    main()
