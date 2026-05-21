#!/usr/bin/env python3
"""
Online residual real-to-sim calibration POC (synth-to-synth).

Adapts Liang et al. 2023 (UCSD, arXiv:2309.11656) Algorithm 1 from the
non-cutting manipulation regime to the cutting/topology-change regime
with 4 cutting-specific modifications:

  Mod 1: Residual on damage state δ_v (irreversible) instead of positions x
         → softplus parameterization enforces Δδ_v ≥ 0
  Mod 2: Three-channel loss (cavity α + axial force F + physical consistency)
  Mod 3: L_traj (trajectory consistency) replaces L_hist (elastic equilibrium)
         since cutting damage is irreversible
  Mod 4: 4 global params (k, κ, s, τ) instead of per-particle k
         → no smoothness loss needed

Architecture (matches Liang Algorithm 1):
   for each frame t along knife trajectory u_t:
       1. Forward sim with current θ → predicted (δ_v^sim, α^sim, F^sim)
       2. Get observation (α^obs, F^obs) — synth from θ_true in this POC
       3. Inner loop: solve Δδ_v ≥ 0 such that α_corrected ≈ α^obs
       4. Outer loop: 1-step Adam on θ using L = L_gap + L_force + L_traj + L_prior

Synth-to-synth validation:
   target trajectory generated with θ_true; init with θ_0 ≠ θ_true;
   verify θ converges toward θ_true within 100 frames.

Reuses from tools/calibrate_stage2_autograd.py:
   setup_synthetic_system, make_params, run_ssa_full, mse_loss_kernel.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

# Diagnostic anchor — mandatory per Stage B engineering rules
wp.config.verify_autograd_array_access = True

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse existing infra
from tools.calibrate_stage2_autograd import (  # noqa: E402
    setup_synthetic_system,
    make_params,
    run_ssa_full,
    mse_loss_kernel,
    Adam2D,
)
from pf_shape_matching_warp.sim.pf_damage import (  # noqa: E402
    update_edge_damage_kernel_ssa_param_arr,
    apply_edge_damage_ssa_kernel,
    update_particle_damage_kernel,
    finalize_particle_damage_kernel_ssa,
    apply_particle_damage_ssa_kernel,
    update_soft_removal_kernel_ssa_param_arr,
)


# =============================================================================
# Helper: run_ssa returning both alpha and delta_v (mirrors run_ssa_full)
# =============================================================================

def run_ssa_full_with_state(params: dict, sys_dict: dict, n_cycles: int = 1):
    """Identical to run_ssa_full but returns BOTH alpha and particle_damage
    final wp.arrays (needed for residual mapping on damage state).

    Returns: (alpha_final_wp, delta_v_final_wp)
    """
    N, E = sys_dict["N"], sys_dict["E"]
    pos = wp.array(sys_dict["pos"], dtype=wp.vec3)
    edge_a = wp.array(sys_dict["edge_a"], dtype=wp.int32)
    edge_b = wp.array(sys_dict["edge_b"], dtype=wp.int32)
    pf_rest = wp.array(sys_dict["pf_rest"], dtype=wp.float32)
    knife_pos = wp.array(sys_dict["knife_pos"], dtype=wp.vec3)
    fixed = wp.array(np.zeros(N, dtype=np.int32), dtype=wp.int32)

    alpha_buffers = [
        wp.array(np.ones(N, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        for _ in range(n_cycles + 1)
    ]
    edge_damage_buffers = [
        wp.zeros(E, dtype=wp.float32, requires_grad=True)
        for _ in range(n_cycles + 1)
    ]
    particle_damage_buffers = [
        wp.zeros(N, dtype=wp.float32, requires_grad=True)
        for _ in range(n_cycles + 1)
    ]

    for cycle in range(n_cycles):
        # Edge damage
        edge_damage_delta = wp.zeros(E, dtype=wp.float32, requires_grad=True)
        wp.launch(update_edge_damage_kernel_ssa_param_arr, dim=E, inputs=[
            pos, alpha_buffers[cycle], edge_a, edge_b, pf_rest,
            edge_damage_buffers[cycle],
            knife_pos, params["knife_radius"], params["alpha_min"],
            edge_damage_delta,
        ])
        wp.launch(apply_edge_damage_ssa_kernel, dim=E, inputs=[
            edge_damage_buffers[cycle], edge_damage_delta,
            edge_damage_buffers[cycle + 1],
        ])
        # Particle damage
        damage_sum = wp.zeros(N, dtype=wp.float32, requires_grad=True)
        damage_count = wp.zeros(N, dtype=wp.float32, requires_grad=True)
        wp.launch(update_particle_damage_kernel, dim=E, inputs=[
            damage_sum, damage_count, alpha_buffers[cycle],
            edge_a, edge_b, edge_damage_buffers[cycle + 1],
        ])
        particle_damage_delta = wp.zeros(N, dtype=wp.float32, requires_grad=True)
        wp.launch(finalize_particle_damage_kernel_ssa, dim=N, inputs=[
            damage_sum, damage_count,
            particle_damage_buffers[cycle], particle_damage_delta,
        ])
        wp.launch(apply_particle_damage_ssa_kernel, dim=N, inputs=[
            particle_damage_buffers[cycle], particle_damage_delta,
            particle_damage_buffers[cycle + 1],
        ])
        # Soft removal
        wp.launch(update_soft_removal_kernel_ssa_param_arr, dim=N, inputs=[
            particle_damage_buffers[cycle + 1],
            alpha_buffers[cycle], fixed,
            params["threshold"], params["sharpness"],
            alpha_buffers[cycle + 1],
        ])

    return alpha_buffers[-1], particle_damage_buffers[-1]


# =============================================================================
# Adam optimizer extended to N-dimensional params
# =============================================================================

class AdamND:
    """N-dimensional Adam, generalization of Adam2D.

    Supports per-parameter learning rate via lr as np.array of shape (N,) or
    scalar.
    """

    def __init__(self, n_dims, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        if np.isscalar(lr):
            self.lr = np.full(n_dims, lr, dtype=np.float64)
        else:
            self.lr = np.asarray(lr, dtype=np.float64)
            assert self.lr.shape == (n_dims,), f"lr shape {self.lr.shape} != ({n_dims},)"
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = np.zeros(n_dims)
        self.v = np.zeros(n_dims)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        """Reset moments — useful when restarting from new init."""
        self.m = np.zeros_like(self.m)
        self.v = np.zeros_like(self.v)
        self.t = 0


# =============================================================================
# Inner loop: damage residual Δδ_v ≥ 0 (numpy / analytic)
# =============================================================================

def _softplus(x):
    """Numerically stable softplus."""
    return np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))


def _softplus_deriv(x):
    """Derivative of softplus = sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def compute_damage_residual(
    delta_v_sim,           # np.array(N,) — current sim damage state
    alpha_obs,             # np.array(N,) — observed cavity
    s, tau,                # current sigmoid params (float scalars)
    K_inner=30,
    lr_inner=0.05,
    verbose=False,
):
    """Inner loop: find Δδ_v ≥ 0 such that
        α_corrected = sigmoid(-s·(δ_v_sim + Δδ_v - τ))  ≈  α_obs

    Uses softplus parameterization: Δδ_v = softplus(δ_raw) ≥ 0
    enforces irreversibility (damage can only INCREASE).

    Analytic gradient (avoids autograd framework dependency at this scale):
        L = ½ Σ (sigmoid(-s·(δ_v + Δδ_v - τ)) - α_obs)²
        ∂L/∂Δδ_v = -(α_corr - α_obs) · α_corr · (1 - α_corr) · s
        ∂L/∂δ_raw = ∂L/∂Δδ_v · softplus'(δ_raw)

    Returns:
        delta_delta_v: np.array(N,) ≥ 0 (final residual)
        L_gap: float (final residual norm)
        L_inner_init: float (initial loss for comparison)
    """
    delta_v_sim = np.asarray(delta_v_sim, dtype=np.float64)
    alpha_obs = np.asarray(alpha_obs, dtype=np.float64)

    N = len(delta_v_sim)
    delta_raw = np.zeros(N, dtype=np.float64)  # Init Δδ_v = softplus(0) = log(2) ≈ 0.69

    # Adam state for this inner loop
    m = np.zeros(N); v = np.zeros(N)
    b1, b2, eps = 0.9, 0.999, 1e-8

    L_inner_init = None
    for k in range(K_inner):
        delta_delta_v = _softplus(delta_raw)
        delta_v_corrected = delta_v_sim + delta_delta_v
        alpha_corr = _sigmoid(-s * (delta_v_corrected - tau))

        # Loss: MSE on alpha
        diff = alpha_corr - alpha_obs
        L = 0.5 * float((diff ** 2).sum())
        if L_inner_init is None:
            L_inner_init = L

        # Analytic gradient ∂L/∂δ_raw
        # ∂α_corr/∂(δ_v + Δδ_v) = -s · α_corr · (1 - α_corr)
        # So ∂L/∂Δδ_v = diff · (-s · α_corr · (1 - α_corr))
        grad_delta = diff * (-s * alpha_corr * (1 - alpha_corr))
        # Chain through softplus: ∂Δδ_v/∂δ_raw = sigmoid(δ_raw)
        grad_raw = grad_delta * _softplus_deriv(delta_raw)

        # Adam update on delta_raw
        m = b1 * m + (1 - b1) * grad_raw
        v = b2 * v + (1 - b2) * grad_raw ** 2
        m_hat = m / (1 - b1 ** (k + 1))
        v_hat = v / (1 - b2 ** (k + 1))
        delta_raw = delta_raw - lr_inner * m_hat / (np.sqrt(v_hat) + eps)

        if verbose and (k % 5 == 0 or k == K_inner - 1):
            print(f"    inner k={k:2d}  L={L:.4e}  |Δδ|={np.linalg.norm(delta_delta_v):.4f}")

    delta_delta_v_final = _softplus(delta_raw)
    delta_v_corrected_final = delta_v_sim + delta_delta_v_final
    alpha_corr_final = _sigmoid(-s * (delta_v_corrected_final - tau))
    L_gap = 0.5 * float(((alpha_corr_final - alpha_obs) ** 2).sum())

    return delta_delta_v_final.astype(np.float32), L_gap, L_inner_init


# =============================================================================
# Synth 'real' trajectory generator
# =============================================================================

def make_knife_trajectory(n_frames: int, x_start: float = 0.4, x_end: float = 0.7):
    """Linear knife sweep along x-axis. y, z fixed at 0.5 (matches
    setup_synthetic_system 1D particle line).

    Returns: np.array shape (n_frames, 3)
    """
    xs = np.linspace(x_start, x_end, n_frames)
    traj = np.zeros((n_frames, 3), dtype=np.float32)
    traj[:, 0] = xs
    traj[:, 1] = 0.5
    traj[:, 2] = 0.5
    return traj


def synth_force_proxy(alpha_arr: np.ndarray, k_bulk: float, dt: float = 0.005,
                      mass: float = 1.0) -> float:
    """Approximate axial reaction force on knife from cavity-particle response.

    For synth POC: F ~ k_bulk × (1 - mean(alpha)) × (some unit factor).
    Real version would integrate shape_delta over knife-contact particles.
    """
    mean_damage = float((1.0 - alpha_arr).mean())
    return k_bulk * mean_damage * (mass / dt) * 0.001  # scale to ~Newton range


def generate_synth_real_trajectory(
    sys_dict,
    theta_true,            # (alpha_min, sharpness, threshold) np array shape (3,)
    knife_trajectory,      # (n_frames, 3) np array of knife positions
    k_bulk_true=1.0,
    n_cycles_per_frame=1,
):
    """Run run_ssa_full at each knife position to produce 'real' observations.

    Each frame, the knife at u_t produces a cavity α_t. We record:
      alpha_obs   — the synth ground-truth cavity
      force_obs   — synthetic axial force (proxy formula)
      knife_pos   — the u_t used

    Note: we re-run ssa from REST at each frame (n_cycles cycles), not
    accumulate damage across frames. This is a simplification for POC —
    real cutting would accumulate. Sufficient for testing the param-recovery
    loop because each frame is an independent (u_t, α_t) pair.

    Returns: list of dicts (one per frame).
    """
    frames = []
    for t, kp in enumerate(knife_trajectory):
        sys_t = dict(sys_dict)
        sys_t["knife_pos"] = np.array([kp], dtype=np.float32)

        params_true = make_params(
            alpha_min=float(theta_true[0]),
            sharpness=float(theta_true[1]),
            threshold=float(theta_true[2]),
            knife_radius=0.1,
            requires_grad=False,
        )
        alpha_t = run_ssa_full(params_true, sys_t, n_cycles=n_cycles_per_frame).numpy().copy()
        force_t = synth_force_proxy(alpha_t, k_bulk_true)

        frames.append(dict(
            t=t,
            knife_pos=kp.copy(),
            alpha_obs=alpha_t,
            force_obs=force_t,
        ))
    return frames


# =============================================================================
# Smoke test — verifies imports + Adam works
# =============================================================================

def smoke_test():
    """Quick verification that all imports work and Adam3D converges on
    a trivial quadratic loss. Should run in < 1s.
    """
    print("[smoke] testing imports + AdamND ...")
    wp.init()

    # Trivial quadratic: minimize ||θ - target||² over 3 params
    target = np.array([3.0, 20.0, 0.5])
    theta = np.array([5.0, 10.0, 0.3])
    opt = AdamND(n_dims=3, lr=np.array([0.1, 1.0, 0.01]))

    for step in range(200):
        grad = 2.0 * (theta - target)
        theta = opt.step(theta, grad)

    err = np.abs(theta - target)
    print(f"[smoke] AdamND converged: θ={theta}, |err|={err}")
    assert np.all(err < 1e-3), f"AdamND failed: err={err}"

    # Sanity check synth system + run_ssa_full still works
    sys_dict = setup_synthetic_system(N=20)
    params = make_params(alpha_min=3.0, sharpness=20.0, requires_grad=False)
    alpha = run_ssa_full(params, sys_dict, n_cycles=1).numpy()
    debulked = float((alpha < 0.5).mean())
    print(f"[smoke] run_ssa_full ran: alpha range=[{alpha.min():.3f}, {alpha.max():.3f}], "
          f"debulked frac={debulked:.3f}")

    print("[smoke] PASS")


def trajectory_test():
    """Phase 2 test: generate a 20-frame synth trajectory and verify
    cavity grows along the knife sweep.
    """
    print("\n[traj] generating 20-frame synth 'real' trajectory ...")
    wp.init()
    sys_dict = setup_synthetic_system(N=20)
    traj = make_knife_trajectory(n_frames=20, x_start=0.4, x_end=0.7)

    theta_true = np.array([3.0, 20.0, 0.5])  # alpha_min, sharpness, threshold
    frames = generate_synth_real_trajectory(
        sys_dict, theta_true, traj,
        k_bulk_true=1.0, n_cycles_per_frame=1,
    )

    # Sanity: as knife moves through particle line, cavity should appear
    # at different positions. Print debulked frac per frame.
    print(f"[traj] {len(frames)} frames generated")
    print(f"       knife_x: {traj[0,0]:.2f} → {traj[-1,0]:.2f}")
    debulked_per_frame = [(f["alpha_obs"] < 0.5).mean() for f in frames]
    force_per_frame = [f["force_obs"] for f in frames]

    for t in [0, 5, 10, 15, 19]:
        kx = frames[t]["knife_pos"][0]
        deb = debulked_per_frame[t]
        F = frames[t]["force_obs"]
        a_min = frames[t]["alpha_obs"].min()
        a_max = frames[t]["alpha_obs"].max()
        print(f"  frame {t:2d}: knife_x={kx:.3f}  debulked={deb:.3f}  "
              f"α=[{a_min:.3f}, {a_max:.3f}]  F={F:.4f}")

    # Check trajectory is non-trivial (cavity varies across frames)
    deb_arr = np.array(debulked_per_frame)
    assert deb_arr.max() > 0.01, "Trajectory generated no damage — knife too far"
    print(f"[traj] PASS  (debulked range: [{deb_arr.min():.3f}, {deb_arr.max():.3f}])")


def inner_loop_test():
    """Phase 3 test: given delta_v from WRONG params, run inner loop to find
    Δδ_v that recovers alpha_obs (from true params).

    Setup:
      - theta_true = (3.0, 20.0, 0.5)   → produces alpha_obs
      - theta_wrong = (5.0, 10.0, 0.5)  → produces delta_v_sim (wrong)
      - Inner loop solves Δδ_v ≥ 0 → check alpha_corrected ≈ alpha_obs
    """
    print("\n[inner] testing damage residual recovery ...")
    wp.init()
    sys_dict = setup_synthetic_system(N=20)
    # Use a knife position that produces partial cavity
    sys_dict["knife_pos"] = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)

    # 1. Generate alpha_obs from theta_true
    theta_true = np.array([3.0, 20.0, 0.5])
    params_true = make_params(
        alpha_min=float(theta_true[0]), sharpness=float(theta_true[1]),
        threshold=float(theta_true[2]), requires_grad=False,
    )
    alpha_obs_wp, delta_v_true_wp = run_ssa_full_with_state(params_true, sys_dict, n_cycles=1)
    alpha_obs = alpha_obs_wp.numpy()
    delta_v_true = delta_v_true_wp.numpy()

    # 2. Generate delta_v_sim from theta_wrong (smaller alpha_min → less damage)
    theta_wrong = np.array([5.0, 10.0, 0.5])
    params_wrong = make_params(
        alpha_min=float(theta_wrong[0]), sharpness=float(theta_wrong[1]),
        threshold=float(theta_wrong[2]), requires_grad=False,
    )
    alpha_sim_wp, delta_v_sim_wp = run_ssa_full_with_state(params_wrong, sys_dict, n_cycles=1)
    alpha_sim = alpha_sim_wp.numpy()
    delta_v_sim = delta_v_sim_wp.numpy()

    print(f"[inner] alpha_obs (true): min={alpha_obs.min():.3f}, max={alpha_obs.max():.3f}, "
          f"debulked frac={float((alpha_obs<0.5).mean()):.3f}")
    print(f"[inner] alpha_sim (wrong init): min={alpha_sim.min():.3f}, max={alpha_sim.max():.3f}, "
          f"debulked frac={float((alpha_sim<0.5).mean()):.3f}")
    L_initial = 0.5 * float(((alpha_sim - alpha_obs) ** 2).sum())
    print(f"[inner] initial cavity gap: L0 = {L_initial:.4e}")

    # 3. Run inner loop using the WRONG params' s, tau as the "current"
    s = float(theta_wrong[1]); tau = float(theta_wrong[2])
    delta_delta_v, L_gap, L_init = compute_damage_residual(
        delta_v_sim, alpha_obs, s, tau,
        K_inner=50, lr_inner=0.1, verbose=True,
    )

    # 4. Verify
    delta_v_corrected = delta_v_sim + delta_delta_v
    alpha_corrected = _sigmoid(-s * (delta_v_corrected - tau))
    L_residual = 0.5 * float(((alpha_corrected - alpha_obs) ** 2).sum())

    print(f"[inner] L_init = {L_init:.4e}")
    print(f"[inner] L_final residual = {L_residual:.4e}")
    print(f"[inner] reduction factor: {L_init / max(L_residual, 1e-12):.2e}×")
    print(f"[inner] |Δδ_v|_2 = {np.linalg.norm(delta_delta_v):.4f}")
    print(f"[inner] Δδ_v range: [{delta_delta_v.min():.3f}, {delta_delta_v.max():.3f}]")
    print(f"[inner] alpha_corrected: min={alpha_corrected.min():.3f}, max={alpha_corrected.max():.3f}")

    # Pass criterion: residual reduces by ≥ 10x AND alpha_corrected close to obs
    pass_ratio = (L_init / max(L_residual, 1e-12)) >= 10.0
    pass_irrev = float(delta_delta_v.min()) >= 0.0  # ≥ 0 ALWAYS (softplus guarantee)
    if pass_ratio and pass_irrev:
        print(f"[inner] PASS  (residual reduced {L_init/L_residual:.1f}× and Δδ_v ≥ 0)")
    else:
        print(f"[inner] FAIL  (ratio_pass={pass_ratio}, irrev_pass={pass_irrev})")
        return False
    return True


if __name__ == "__main__":
    import sys as _sys
    if "--smoke" in _sys.argv:
        smoke_test()
    elif "--traj" in _sys.argv:
        trajectory_test()
    elif "--inner" in _sys.argv:
        inner_loop_test()
    else:
        smoke_test()
        trajectory_test()
        inner_loop_test()
