"""Day 6f v2: rigorous validation of v0 vs v1 gradient via FD ground truth.

The question: is v1 (keep-state-in-graph) LEGITIMATELY giving better gradient,
or is it computing a spurious value that just happens to land closer in 30 Adam
steps for 1 init point?

Mathematical setup:
  L_total(θ) = Σ_{t=1..T} L_t(θ; chained_state(θ))
  where chained_state evolves: s_0 = rest, s_t = solver.step(θ, s_{t-1}, knife_t)

  v0 gradient: Σ ∂L_t/∂θ |_{s_{t-1} treated as constant}
  v1 gradient: d(L_total)/dθ  (FULL chain, state propagated through autograd)
  FD gradient: numerical d(L_total)/dθ  ← ground truth

Tests:
  (1) FD parity — which of {v0, v1} matches FD?
  (2) Cosine similarity — is v1 just scaled v0, or different direction?

This validates whether v1 has any TRUE info gain or is artifact.
"""
import sys
from pathlib import Path
import numpy as np
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pf_shape_matching_warp"))

import torch
import warp as wp
wp.init()
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

from tools.day2_solver_wrapper import (  # noqa: E402
    setup_solver_with_rest, solver_load_state, solver_save_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day6f_v1_multistep_keep_graph import (  # noqa: E402
    MultiFrameSolverOp, make_knife_trajectory, gen_obs_multistep,
)


def L_total_continuous_FD(solver, rest_state, theta_np, knife_traj, knife_radius,
                             obs_pos_t, obs_alpha_t, lambda_cav, device):
    """L_total via CONTINUOUS state (v1's objective) — uses MultiFrameSolverOp no_grad.

    This matches v1's forward path exactly: single setup, T sequential solver.step()s,
    no save/load between frames. NO autograd (just forward to extract scalar).
    """
    theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=device,
                                 requires_grad=False)
    T = len(knife_traj)
    with torch.no_grad():
        alpha_T, pos_T, dv_T = MultiFrameSolverOp.apply(
            theta_torch, solver, rest_state, knife_traj, knife_radius, T,
        )
        L_tot = 0.0
        for t in range(T):
            L_pos = ((pos_T[t] - obs_pos_t[t]) ** 2).sum()
            L_cav = ((alpha_T[t] - obs_alpha_t[t]) ** 2).sum()
            L_tot += float((L_pos + lambda_cav * L_cav).item())
    # Clear theta_wp after no_grad forward to avoid stale state for next call
    solver.set_theta_wp(None)
    return L_tot


def L_total_chained_FD(solver, rest_state, theta_np, knife_traj, knife_radius,
                         obs_pos, obs_alpha, lambda_cav):
    """Compute L_total via legacy float path (no autograd) — for FD only."""
    state_in = rest_state
    L_tot = 0.0
    T = len(knife_traj)
    for t in range(T):
        solver.set_theta_wp(None)
        solver.alpha_min = float(theta_np[0])
        solver.soft_removal_sharpness = float(theta_np[1])
        solver.aspiration_threshold = float(theta_np[2])
        solver.knife_radius = knife_radius
        solver.knife_active = 1
        solver.knife_pos = wp.array(np.array([knife_traj[t]], dtype=np.float32),
                                      dtype=wp.vec3, device=solver.device,
                                      requires_grad=False)
        solver_load_state(solver, state_in)
        solver.step()
        if solver.use_ssa_soft_removal:
            ci = solver._ssa_cycle_index
            alpha_np = solver._ssa_alpha_buffers[ci].numpy()
            dv_np = solver._ssa_particle_damage_buffers[ci].numpy()
        else:
            alpha_np = solver.p["alpha"].numpy()
            dv_np = solver.p["damage"].numpy()
        pos_np = solver.p["pos"].numpy()
        L_pos = float(np.sum((pos_np - obs_pos[t]) ** 2))
        L_cav = float(np.sum((alpha_np - obs_alpha[t]) ** 2))
        L_tot += L_pos + lambda_cav * L_cav
        state_in = solver_save_state(solver)
    return L_tot


def compute_v0_gradient(solver, rest_state, theta_torch, knife_traj, knife_radius,
                          obs_pos_t, obs_alpha_t, lambda_cav):
    if theta_torch.grad is not None:
        theta_torch.grad.zero_()
    state_in = rest_state
    L_frames = []
    T = len(knife_traj)
    for t in range(T):
        alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
            theta_torch, solver, state_in, knife_traj[t], knife_radius,
        )
        L_t = ((pos_t - obs_pos_t[t]) ** 2).sum() + lambda_cav * ((alpha_t - obs_alpha_t[t]) ** 2).sum()
        L_t.backward()
        L_frames.append(float(L_t.item()))
        state_in = solver_save_state(solver)
    grad = theta_torch.grad.detach().cpu().numpy().copy()
    return grad, sum(L_frames)


def compute_v1_gradient(solver, rest_state, theta_torch, knife_traj, knife_radius,
                          obs_pos_t, obs_alpha_t, lambda_cav):
    if theta_torch.grad is not None:
        theta_torch.grad.zero_()
    T = len(knife_traj)
    alpha_T, pos_T, dv_T = MultiFrameSolverOp.apply(
        theta_torch, solver, rest_state, knife_traj, knife_radius, T,
    )
    L_frames = []
    for t in range(T):
        L_t = ((pos_T[t] - obs_pos_t[t]) ** 2).sum() + lambda_cav * ((alpha_T[t] - obs_alpha_t[t]) ** 2).sum()
        L_frames.append(L_t)
    L_total = sum(L_frames)
    L_total.backward()
    grad = theta_torch.grad.detach().cpu().numpy().copy()
    return grad, float(L_total.item())


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_radius = 0.04
    T = 3
    knife_traj = make_knife_trajectory(T)
    theta_true = np.array([3.0, 20.0, 0.5], dtype=np.float32)
    theta_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)
    param_names = ["α_min", "sharpness", "τ"]

    print(f"\n[Setup] T={T}, θ_init={theta_init.tolist()}")

    print(f"\n[Setup] generating obs at θ_true …")
    obs_pos, obs_alpha = gen_obs_multistep(
        solver, rest_state, theta_true, knife_traj, knife_radius)
    obs_pos_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]

    # Probe λ_cav
    theta_probe = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                 requires_grad=False)
    state_in = rest_state
    L_pos_per_frame, L_cav_per_frame = [], []
    with torch.no_grad():
        for t in range(T):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_probe, solver, state_in, knife_traj[t], knife_radius,
            )
            L_pos_per_frame.append(float(((pos_t - obs_pos_t[t]) ** 2).sum()))
            L_cav_per_frame.append(float(((alpha_t - obs_alpha_t[t]) ** 2).sum()))
            state_in = solver_save_state(solver)
    lambda_cav = max(sum(L_pos_per_frame) / max(sum(L_cav_per_frame), 1e-12), 1e-6)
    print(f"[Probe] λ_cav={lambda_cav:.4e}")

    # =========================================================================
    # Test 1: FD ground truth
    # =========================================================================
    print(f"\n[Test 1] FD ground truth on chained L_total at θ_init")
    L_0 = L_total_chained_FD(
        solver, rest_state, theta_init, knife_traj, knife_radius,
        obs_pos, obs_alpha, lambda_cav)
    print(f"  L_total(θ_init) = {L_0:.6e}")

    eps_per_dim = [1e-2, 1e-1, 1e-2]
    grad_FD_B = np.zeros(3)
    for d in range(3):
        eps = eps_per_dim[d]
        th_plus = theta_init.copy(); th_plus[d] += eps
        th_minus = theta_init.copy(); th_minus[d] -= eps
        L_p = L_total_chained_FD(
            solver, rest_state, th_plus, knife_traj, knife_radius,
            obs_pos, obs_alpha, lambda_cav)
        L_m = L_total_chained_FD(
            solver, rest_state, th_minus, knife_traj, knife_radius,
            obs_pos, obs_alpha, lambda_cav)
        grad_FD_B[d] = (L_p - L_m) / (2 * eps)
        signal = (L_p - L_m) / max(L_0, 1e-12)
        print(f"  d={d} ({param_names[d]:>10s}): ε={eps:.0e}  "
              f"L_B(+ε)={L_p:.6e}  L_B(-ε)={L_m:.6e}  FD_B={grad_FD_B[d]:+.4e}  "
              f"signal/L₀={signal:+.3e}")

    # FD_A: continuous-state objective (v1's path)
    print(f"\n[Test 1b] FD ground truth on CONTINUOUS L_total (v1's objective)")
    L_0_A = L_total_continuous_FD(
        solver, rest_state, theta_init, knife_traj, knife_radius,
        obs_pos_t, obs_alpha_t, lambda_cav, device)
    print(f"  L_A(θ_init) = {L_0_A:.6e}")
    grad_FD_A = np.zeros(3)
    for d in range(3):
        eps = eps_per_dim[d]
        th_plus = theta_init.copy(); th_plus[d] += eps
        th_minus = theta_init.copy(); th_minus[d] -= eps
        L_p = L_total_continuous_FD(
            solver, rest_state, th_plus, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, lambda_cav, device)
        L_m = L_total_continuous_FD(
            solver, rest_state, th_minus, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, lambda_cav, device)
        grad_FD_A[d] = (L_p - L_m) / (2 * eps)
        print(f"  d={d} ({param_names[d]:>10s}): "
              f"L_A(+ε)={L_p:.6e}  L_A(-ε)={L_m:.6e}  FD_A={grad_FD_A[d]:+.4e}")

    # v0 gradient
    print(f"\n[Test 1] computing v0 gradient (per-frame backward, detached state)")
    theta_v0 = torch.tensor(theta_init, dtype=torch.float32, device=device,
                              requires_grad=True)
    grad_v0, L_v0 = compute_v0_gradient(
        solver, rest_state, theta_v0, knife_traj, knife_radius,
        obs_pos_t, obs_alpha_t, lambda_cav)
    print(f"  L_total = {L_v0:.6e}")
    print(f"  v0 grad = ({grad_v0[0]:+.4e}, {grad_v0[1]:+.4e}, {grad_v0[2]:+.4e})")

    # v1 gradient
    print(f"\n[Test 1] computing v1 gradient (keep-state-in-graph, single tape)")
    theta_v1 = torch.tensor(theta_init, dtype=torch.float32, device=device,
                              requires_grad=True)
    grad_v1, L_v1 = compute_v1_gradient(
        solver, rest_state, theta_v1, knife_traj, knife_radius,
        obs_pos_t, obs_alpha_t, lambda_cav)
    print(f"  L_total = {L_v1:.6e}")
    print(f"  v1 grad = ({grad_v1[0]:+.4e}, {grad_v1[1]:+.4e}, {grad_v1[2]:+.4e})")

    # =========================================================================
    # Test 1 verdict
    # =========================================================================
    print("\n" + "=" * 70)
    print("Test 1 verdict — which gradient matches FD ground truth?")
    print("=" * 70)
    print(f"\n  Forward L_total agreement:")
    print(f"    FD path: {L_0:.6e}")
    print(f"    v0 sum : {L_v0:.6e}   rel_diff={abs(L_v0-L_0)/L_0*100:.3f}%")
    print(f"    v1 sum : {L_v1:.6e}   rel_diff={abs(L_v1-L_0)/L_0*100:.3f}%")
    # (forward must agree — physics same)

    # Each gradient compared to ITS OWN objective's FD
    print(f"\n  Per-objective FD parity (apples-to-apples):")
    print(f"  {'dim':>10s}  {'FD_B':>13s}  {'v0 (B)':>13s}  {'v0_err':>8s}    "
          f"{'FD_A':>13s}  {'v1 (A)':>13s}  {'v1_err':>8s}")
    rel_err_v0_B = np.zeros(3); rel_err_v1_A = np.zeros(3)
    for d, name in enumerate(param_names):
        rel_err_v0_B[d] = abs(grad_v0[d] - grad_FD_B[d]) / max(abs(grad_FD_B[d]), 1e-12)
        rel_err_v1_A[d] = abs(grad_v1[d] - grad_FD_A[d]) / max(abs(grad_FD_A[d]), 1e-12)
        print(f"  {name:>10s}  {grad_FD_B[d]:>+13.4e}  {grad_v0[d]:>+13.4e}  "
              f"{rel_err_v0_B[d]*100:>7.2f}%    "
              f"{grad_FD_A[d]:>+13.4e}  {grad_v1[d]:>+13.4e}  {rel_err_v1_A[d]*100:>7.2f}%")
    print(f"\n  Mean rel_err vs OWN FD:  v0 vs FD_B={rel_err_v0_B.mean()*100:.2f}%   "
          f"v1 vs FD_A={rel_err_v1_A.mean()*100:.2f}%")
    # Verdicts
    v0_correct = rel_err_v0_B.mean() < 0.1
    v1_correct = rel_err_v1_A.mean() < 0.1
    if v0_correct and v1_correct:
        print(f"  ✓ BOTH gradients correct for their own objectives")
        print(f"     v0 differentiates L_B (save/load), v1 differentiates L_A (continuous)")
        print(f"     Difference is OBJECTIVE choice, not bug")
    elif v1_correct and not v0_correct:
        print(f"  ⚠ v1 correct (A), v0 approximates (B but with state-detach)")
    elif v0_correct and not v1_correct:
        print(f"  ✗ v0 correct, v1 has bug — keep-state-in-graph implementation broken")
    else:
        print(f"  ✗ Both deviate from their FD — both have issues")

    # Also show v1 vs FD_B (apples-to-oranges, but informative)
    rel_err_v1_B = np.abs(grad_v1 - grad_FD_B) / np.maximum(np.abs(grad_FD_B), 1e-12)
    rel_err_v0_A = np.abs(grad_v0 - grad_FD_A) / np.maximum(np.abs(grad_FD_A), 1e-12)
    print(f"\n  Cross-objective comparison (apples vs oranges):")
    print(f"    v0 vs FD_A (would v0 work for continuous obj?): {rel_err_v0_A.mean()*100:.2f}%")
    print(f"    v1 vs FD_B (would v1 work for snapshot obj?):   {rel_err_v1_B.mean()*100:.2f}%")

    # =========================================================================
    # Test 2: Cosine similarity & magnitude vs direction analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("Test 2 — magnitude vs direction analysis")
    print("=" * 70)
    cos_v0_v1 = np.dot(grad_v0, grad_v1) / (np.linalg.norm(grad_v0) * np.linalg.norm(grad_v1) + 1e-18)
    cos_v0_B = np.dot(grad_v0, grad_FD_B) / (np.linalg.norm(grad_v0) * np.linalg.norm(grad_FD_B) + 1e-18)
    cos_v1_A = np.dot(grad_v1, grad_FD_A) / (np.linalg.norm(grad_v1) * np.linalg.norm(grad_FD_A) + 1e-18)
    cos_FD_A_B = np.dot(grad_FD_A, grad_FD_B) / (np.linalg.norm(grad_FD_A) * np.linalg.norm(grad_FD_B) + 1e-18)

    print(f"\n  Cosine similarities:")
    print(f"    cos(v0, FD_B)   = {cos_v0_B:.6f}   (v0 direction within its objective)")
    print(f"    cos(v1, FD_A)   = {cos_v1_A:.6f}   (v1 direction within its objective)")
    print(f"    cos(FD_A, FD_B) = {cos_FD_A_B:.6f}   (do the two objectives have same gradient direction?)")
    print(f"    cos(v0, v1)     = {cos_v0_v1:.6f}")

    print(f"\n  Gradient norms:")
    print(f"    ‖v0‖   = {np.linalg.norm(grad_v0):.4e}")
    print(f"    ‖v1‖   = {np.linalg.norm(grad_v1):.4e}")
    print(f"    ‖FD_B‖ = {np.linalg.norm(grad_FD_B):.4e}")
    print(f"    ‖FD_A‖ = {np.linalg.norm(grad_FD_A):.4e}")

    # =========================================================================
    # Interpretation
    # =========================================================================
    print("\n" + "=" * 70)
    print("Interpretation: is v1 LEGITIMATELY better?")
    print("=" * 70)
    print(f"  v0 differentiates L_B (save/load snapshot) — FD parity: "
          f"{rel_err_v0_B.mean()*100:.2f}%  cos={cos_v0_B:.4f}")
    print(f"  v1 differentiates L_A (continuous) — FD parity: "
          f"{rel_err_v1_A.mean()*100:.2f}%  cos={cos_v1_A:.4f}")
    if v0_correct and v1_correct:
        # Both correct → real question: which OBJECTIVE matches our use case?
        if cos_FD_A_B > 0.99:
            print(f"\n  → FD_A and FD_B have nearly same direction (cos > 0.99)")
            print(f"    L_A ≈ L_B as objectives. v0 and v1 differ mainly in magnitude.")
            print(f"    The 'v1 better recovery' result is likely magnitude-driven, NOT info gain.")
        else:
            print(f"\n  → FD_A and FD_B have DIFFERENT directions (cos = {cos_FD_A_B:.4f})")
            print(f"    L_A and L_B are genuinely different objectives.")
            print(f"    Choose v0 if deploying online (snapshot-based real-time).")
            print(f"    Choose v1 if doing offline batch sys-ID (clean sim fitting).")
    elif v1_correct:
        print(f"\n  → v1 is correct for its objective; v0 has approximation error")
    elif v0_correct:
        print(f"\n  → v0 correct, v1 buggy. Need to debug MultiFrameSolverOp.")


if __name__ == "__main__":
    main()
