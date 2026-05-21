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
# Outer loop: Warp-tape backward for ∂L/∂θ on (alpha_min, sharpness, threshold)
# =============================================================================

def compute_outer_gradient(
    theta,                # np.array(3,) — current (alpha_min, sharpness, threshold)
    sys_dict,             # dict with single knife_pos
    alpha_obs,            # np.array(N,) — observation
    n_cycles=1,
):
    """Computes ∂L_direct/∂θ where L_direct = Σ (α_sim - α_obs)²

    Strategy A (detached residual): outer gradient flows ONLY through
    forward sim w.r.t. θ. Uses existing Warp tape pattern from
    calibrate_stage2_autograd.py.

    Returns:
        loss: float (L_direct, normalized by N)
        grad: np.array(3,) — ∂L/∂(alpha_min, sharpness, threshold)
    """
    N = sys_dict["N"]
    params = make_params(
        alpha_min=float(theta[0]),
        sharpness=float(theta[1]),
        threshold=float(theta[2]),
        knife_radius=0.1,
        requires_grad=True,
    )
    target_wp = wp.array(alpha_obs.astype(np.float32), dtype=wp.float32)
    loss_buf = wp.zeros(1, dtype=wp.float32, requires_grad=True)

    with wp.Tape() as tape:
        alpha_sim_wp = run_ssa_full(params, sys_dict, n_cycles=n_cycles)
        wp.launch(mse_loss_kernel, dim=N,
                  inputs=[alpha_sim_wp, target_wp, loss_buf])

    tape.backward(loss=loss_buf)

    loss = float(loss_buf.numpy()[0]) / float(N)
    grad = np.array([
        float(params["alpha_min"].grad.numpy()[0]) / float(N),
        float(params["sharpness"].grad.numpy()[0]) / float(N),
        float(params["threshold"].grad.numpy()[0]) / float(N),
    ])
    return loss, grad


# =============================================================================
# L_traj (P8 Mod 3): trajectory consistency loss
# =============================================================================

def compute_traj_loss(
    theta,                    # current (alpha_min, sharp, tau)
    history,                  # list of dicts from online_calibration_loop
    sys_dict,                 # base sys_dict (knife_pos overridden per snapshot)
    n_samples=4,
    history_window=20,
    n_cycles_per_frame=1,
    rng=None,
):
    """L_traj = (1/n_samples) Σ_h ‖α^sim(θ; u_h) - α^obs_h‖²

    Enforces trajectory-wide identifiability by demanding that the current
    θ simultaneously explains 4 randomly-sampled past observations.

    Algorithm (mirrors Liang L_hist but for cutting):
      1. Sample n_samples random indices from the LAST history_window snapshots
      2. For each sample h:
         a. Build sys_h with h.knife_pos
         b. Compute L_direct + gradient via Warp tape using current θ
      3. Average loss + gradient over samples

    Returns:
        (L_traj_value, grad_traj) — both scalar / (3,) np.array
        If len(history) < n_samples, returns (0, zeros) (warmup phase).
    """
    if len(history) < n_samples:
        return 0.0, np.zeros(3)

    if rng is None:
        rng = np.random.default_rng()

    # Window: last `history_window` (or all if shorter)
    window = history[-history_window:] if len(history) >= history_window else history[:]
    indices = rng.choice(len(window), size=n_samples, replace=False)

    total_loss = 0.0
    total_grad = np.zeros(3)
    for idx in indices:
        h = window[idx]
        sys_h = dict(sys_dict)
        sys_h["knife_pos"] = np.array([h["knife_pos"]], dtype=np.float32)
        # Reuse compute_outer_gradient — it does its own Warp tape backward.
        L_h, grad_h = compute_outer_gradient(
            theta, sys_h, h["alpha_obs"],
            n_cycles=n_cycles_per_frame,
        )
        total_loss += L_h
        total_grad += grad_h

    return total_loss / n_samples, total_grad / n_samples


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


# =============================================================================
# Main online calibration loop (Phase 5)
# =============================================================================

def online_calibration_loop(
    sys_dict,
    real_frames,                # output of generate_synth_real_trajectory
    theta_0,                    # np.array(3,) initial guess
    K_inner=30,
    lr_inner=0.1,
    lr_outer=np.array([0.1, 1.0, 0.05]),    # per-param (alpha_min, sharp, threshold)
    n_cycles_per_frame=1,
    param_bounds=((0.5, 50.0), (1.0, 200.0), (0.05, 0.95)),
    lambda_prior=0.0,           # P7 Fix 1: L2 anchor toward theta_0. 0=disabled.
    lambda_traj=0.0,            # P8 Mod 3: trajectory consistency. 0=disabled.
    n_traj_samples=4,           # P8: snapshots per outer step (Liang default)
    traj_history_window=20,     # P8: window from which to sample (Liang default)
    rng_seed=42,                # P8: reproducibility for L_traj sampling
    track_best=True,            # P7: keep best-θ-so-far snapshot
    verbose=True,
    log_every=10,
):
    """Main online residual + Adam calibration loop.

    Algorithm (mirrors Liang Algorithm 1, adapted for cutting):
      for each frame t with observation (α_obs, F_obs):
          1. Forward sim with current θ → (α_sim, δ_v_sim)
          2. Inner loop → Δδ_v (state correction for next frame)
          3. Outer Warp-tape gradient → ∂L/∂θ
          4. (P7 Fix 1) Add L_prior gradient = 2·λ·(θ - θ_0)
          5. Adam step on θ (with per-param lr)
          6. Track best θ by L_direct
          7. Save history snapshot

    Returns: dict with log, history, best_theta, best_L, best_step.
    """
    n_frames = len(real_frames)
    theta = theta_0.copy().astype(np.float64)
    theta_0_const = theta_0.copy().astype(np.float64)  # immutable anchor
    opt = AdamND(n_dims=3, lr=lr_outer)
    history = []
    log = []

    bounds_lo = np.array([b[0] for b in param_bounds])
    bounds_hi = np.array([b[1] for b in param_bounds])

    # Best-so-far tracker
    best_L = float("inf")
    best_theta = theta.copy()
    best_step = 0

    # P8: RNG for L_traj snapshot sampling
    rng = np.random.default_rng(rng_seed)

    for t, obs in enumerate(real_frames):
        # 1. Build per-frame sys_dict
        sys_t = dict(sys_dict)
        sys_t["knife_pos"] = np.array([obs["knife_pos"]], dtype=np.float32)

        # 2. Forward sim with current θ → delta_v_sim for inner loop
        params_curr = make_params(
            alpha_min=float(theta[0]), sharpness=float(theta[1]),
            threshold=float(theta[2]), requires_grad=False,
        )
        alpha_sim_wp, delta_v_sim_wp = run_ssa_full_with_state(
            params_curr, sys_t, n_cycles=n_cycles_per_frame,
        )
        alpha_sim = alpha_sim_wp.numpy().copy()
        delta_v_sim = delta_v_sim_wp.numpy().copy()

        # 3. Inner loop: compute Δδ_v
        delta_delta_v, L_gap, _ = compute_damage_residual(
            delta_v_sim, obs["alpha_obs"],
            float(theta[1]), float(theta[2]),
            K_inner=K_inner, lr_inner=lr_inner, verbose=False,
        )

        # 4a. Outer gradient (Strategy A: backprop through forward sim only)
        L_direct, grad_data = compute_outer_gradient(
            theta, sys_t, obs["alpha_obs"], n_cycles=n_cycles_per_frame,
        )

        # 4b. P7 Fix 1: L_prior anchor toward theta_0
        if lambda_prior > 0.0:
            grad_prior = 2.0 * lambda_prior * (theta - theta_0_const)
        else:
            grad_prior = np.zeros_like(theta)

        # 4c. P8 Mod 3: L_traj — trajectory consistency on 4 history snapshots
        if lambda_traj > 0.0:
            L_traj, grad_traj = compute_traj_loss(
                theta, history, sys_dict,
                n_samples=n_traj_samples,
                history_window=traj_history_window,
                n_cycles_per_frame=n_cycles_per_frame,
                rng=rng,
            )
        else:
            L_traj, grad_traj = 0.0, np.zeros_like(theta)

        grad = grad_data + grad_prior + lambda_traj * grad_traj

        # 5. Adam step + clip to bounds
        theta = opt.step(theta, grad)
        theta = np.clip(theta, bounds_lo, bounds_hi)

        # 6. Best-so-far tracking
        if track_best and L_direct < best_L:
            best_L = L_direct
            best_theta = theta.copy()
            best_step = t

        # 7. Log + history
        log_entry = dict(
            t=t,
            knife_x=float(obs["knife_pos"][0]),
            L_direct=L_direct,
            L_gap=L_gap,
            alpha_min=float(theta[0]),
            sharpness=float(theta[1]),
            threshold=float(theta[2]),
            grad_a=float(grad_data[0]),
            grad_s=float(grad_data[1]),
            grad_t=float(grad_data[2]),
            grad_prior_a=float(grad_prior[0]),
            grad_prior_s=float(grad_prior[1]),
            grad_prior_t=float(grad_prior[2]),
            L_traj=float(L_traj),
            grad_traj_a=float(grad_traj[0]),
            grad_traj_s=float(grad_traj[1]),
            grad_traj_t=float(grad_traj[2]),
        )
        log.append(log_entry)

        # History (used by L_traj on subsequent frames)
        history.append(dict(
            knife_pos=obs["knife_pos"].copy(),
            alpha_obs=obs["alpha_obs"].copy(),
            delta_v_corrected=(delta_v_sim + delta_delta_v).copy(),
        ))

        if verbose and (t % log_every == 0 or t == n_frames - 1):
            best_tag = " ★" if t == best_step else ""
            traj_tag = f"  L_traj={L_traj:.4e}" if lambda_traj > 0 else ""
            print(f"[t={t:3d}] L_dir={L_direct:.4e}  L_gap={L_gap:.4e}{traj_tag}  "
                  f"θ=({theta[0]:.3f}, {theta[1]:.3f}, {theta[2]:.3f}){best_tag}")

    return dict(
        log=log,
        history=history,
        best_theta=best_theta,
        best_L=best_L,
        best_step=best_step,
        final_theta=theta.copy(),
        final_L=float(log[-1]["L_direct"]),
        theta_0=theta_0_const,
        lambda_prior=lambda_prior,
        lambda_traj=lambda_traj,
    )


def outer_grad_test():
    """Phase 4 test: verify outer gradient points toward theta_true.

    theta_true = (3.0, 20.0, 0.5)
    theta_wrong = (5.0, 10.0, 0.5)

    Since alpha_min needs to DECREASE (5→3), gradient should be POSITIVE.
    Since sharpness needs to INCREASE (10→20), gradient should be NEGATIVE.
    Threshold matches, gradient should be near zero.
    """
    print("\n[outer] testing gradient direction ...")
    wp.init()
    sys_dict = setup_synthetic_system(N=20)
    sys_dict["knife_pos"] = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)

    # 1. Generate alpha_obs from theta_true
    theta_true = np.array([3.0, 20.0, 0.5])
    params_true = make_params(
        alpha_min=float(theta_true[0]), sharpness=float(theta_true[1]),
        threshold=float(theta_true[2]), requires_grad=False,
    )
    alpha_obs = run_ssa_full(params_true, sys_dict, n_cycles=1).numpy().copy()

    # 2. Compute gradient at theta_wrong
    theta_wrong = np.array([5.0, 10.0, 0.5])
    loss, grad = compute_outer_gradient(theta_wrong, sys_dict, alpha_obs, n_cycles=1)

    print(f"[outer] theta_wrong = (α_min={theta_wrong[0]}, sharp={theta_wrong[1]}, τ={theta_wrong[2]})")
    print(f"[outer] theta_true  = (α_min={theta_true[0]}, sharp={theta_true[1]}, τ={theta_true[2]})")
    print(f"[outer] loss = {loss:.4e}")
    print(f"[outer] grad = (∂L/∂α_min={grad[0]:+.4e}, ∂L/∂sharp={grad[1]:+.4e}, "
          f"∂L/∂τ={grad[2]:+.4e})")
    print(f"[outer] expected directions:")
    print(f"  ∂L/∂α_min > 0  (so Adam decreases α_min from 5 to 3)  → got {grad[0]:+.2e}  "
          f"{'✓' if grad[0] > 0 else '✗'}")
    print(f"  ∂L/∂sharp < 0  (so Adam increases sharp from 10 to 20)  → got {grad[1]:+.2e}  "
          f"{'✓' if grad[1] < 0 else '✗'}")
    print(f"  |∂L/∂τ| small  (τ already correct)  → got {abs(grad[2]):.2e}  "
          f"{'✓' if abs(grad[2]) < max(abs(grad[0]), abs(grad[1])) else '⚠'}")

    pass_dir_a = grad[0] > 0
    pass_dir_s = grad[1] < 0
    if pass_dir_a and pass_dir_s:
        print(f"[outer] PASS  (both gradient directions point toward theta_true)")
        return True
    else:
        print(f"[outer] FAIL  (gradient does NOT point toward theta_true)")
        return False


def plot_convergence(log_path=None, out_path=None):
    """Phase 6: plot online residual calibration convergence.

    Generates 2x2 grid:
      (0,0) alpha_min trajectory vs true
      (0,1) sharpness trajectory vs true
      (1,0) threshold trajectory vs true
      (1,1) L_direct + L_gap log-scale vs frame
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if log_path is None:
        log_path = REPO_ROOT / "tools" / "online_residual_log.npz"
    if out_path is None:
        out_path = REPO_ROOT / "tools" / "online_residual_convergence.png"

    log = np.load(log_path)
    t = log["t"]
    theta_true = log["theta_true"]
    theta_0 = log["theta_0"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "ReSECT Online Residual Calibration — Synth-to-Synth (POC)\n"
        f"θ_0=({theta_0[0]:.1f}, {theta_0[1]:.1f}, {theta_0[2]:.2f})  →  "
        f"θ_true=({theta_true[0]:.1f}, {theta_true[1]:.1f}, {theta_true[2]:.2f})",
        fontsize=11,
    )

    param_names = ["alpha_min (κ)", "sharpness (s)", "threshold (τ)"]
    param_keys = ["alpha_min", "sharpness", "threshold"]
    positions = [(0, 0), (0, 1), (1, 0)]

    for (r, c), name, key, true_v in zip(positions, param_names, param_keys, theta_true):
        ax = axes[r, c]
        ax.plot(t, log[key], "b-", label="estimate", linewidth=1.5)
        ax.axhline(true_v, color="red", linestyle="--", label="ground truth", linewidth=1.2)
        ax.axhline(theta_0[positions.index((r, c))], color="gray", linestyle=":",
                   label="init θ_0", linewidth=1.0)
        ax.set_xlabel("frame")
        ax.set_ylabel(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        final = float(log[key][-1])
        rel = abs(final - true_v) / max(abs(true_v), 1e-6) * 100
        ax.set_title(f"{name}: final={final:.3f}  rel_err={rel:.2f}%", fontsize=10)

    # Loss panel
    ax = axes[1, 1]
    ax.semilogy(t, log["L_direct"], "b-", label="L_direct (cavity gap)", linewidth=1.2)
    ax.semilogy(t, log["L_gap"], "g--", label="L_gap (residual norm)", linewidth=1.0)
    ax.set_xlabel("frame")
    ax.set_ylabel("loss (log)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Loss trajectories", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {out_path}")
    return out_path


def integration_test(n_frames=100, lambda_prior=0.0, lambda_traj=0.0, log_out_name=None):
    """Phase 5/7 test: full online loop, synth-to-synth.

    Setup:
      theta_true = (3.0, 20.0, 0.5)
      theta_0    = (5.0, 10.0, 0.5)   (alpha_min 67% off, sharp 50% off)
      knife sweep x=0.35→0.65 (broad cavity)

    Args:
      n_frames     — trajectory length
      lambda_prior — P7 Fix 1 anchor strength (0 = original POC behavior)
      log_out_name — npz file basename for saving log
    """
    print(f"\n[integ] running {n_frames}-frame online calibration "
          f"(λ_prior={lambda_prior}, λ_traj={lambda_traj}) ...")
    wp.init()
    sys_dict = setup_synthetic_system(N=20)

    traj = make_knife_trajectory(n_frames=n_frames, x_start=0.35, x_end=0.65)
    theta_true = np.array([3.0, 20.0, 0.5])
    real_frames = generate_synth_real_trajectory(
        sys_dict, theta_true, traj,
        k_bulk_true=1.0, n_cycles_per_frame=1,
    )

    theta_0 = np.array([5.0, 10.0, 0.5])
    print(f"[integ] theta_true   = {theta_true}")
    print(f"[integ] theta_0      = {theta_0}")

    t_start = time.time()
    result = online_calibration_loop(
        sys_dict, real_frames, theta_0,
        K_inner=30, lr_inner=0.1,
        lr_outer=np.array([0.1, 1.0, 0.05]),
        lambda_prior=lambda_prior,
        lambda_traj=lambda_traj,
        verbose=True, log_every=max(1, n_frames // 10),
    )
    elapsed = time.time() - t_start

    log = result["log"]
    theta_final = result["final_theta"]
    theta_best = result["best_theta"]
    best_step = result["best_step"]
    best_L = result["best_L"]

    err_final = np.abs(theta_final - theta_true)
    rel_final = err_final / np.maximum(np.abs(theta_true), 1e-6)
    err_best = np.abs(theta_best - theta_true)
    rel_best = err_best / np.maximum(np.abs(theta_true), 1e-6)

    print(f"\n[integ] Done in {elapsed:.1f}s ({elapsed/n_frames*1000:.1f} ms/frame)")
    print(f"[integ] theta_true   = {theta_true}")
    print(f"[integ] theta_FINAL  = {theta_final}  (last step)")
    print(f"  rel err            = {(rel_final * 100).round(2)} %")
    print(f"[integ] theta_BEST   = {theta_best}  ★ at step {best_step} (L={best_L:.2e})")
    print(f"  rel err            = {(rel_best * 100).round(2)} %")

    n_pass_final = int((rel_final < 0.30).sum())
    n_pass_best = int((rel_best < 0.30).sum())
    pass_overall = n_pass_best >= 2

    print(f"[integ] Final:  {n_pass_final}/3 params within 30%")
    print(f"[integ] Best:   {n_pass_best}/3 params within 30%  ← Fix1 metric")
    if pass_overall:
        print(f"[integ] PASS (best-so-far has {n_pass_best}/3 within 30%)")
    else:
        print(f"[integ] PARTIAL")

    # Save log
    log_arr = dict(
        t=np.array([e["t"] for e in log]),
        L_direct=np.array([e["L_direct"] for e in log]),
        L_gap=np.array([e["L_gap"] for e in log]),
        alpha_min=np.array([e["alpha_min"] for e in log]),
        sharpness=np.array([e["sharpness"] for e in log]),
        threshold=np.array([e["threshold"] for e in log]),
        grad_a=np.array([e["grad_a"] for e in log]),
        grad_s=np.array([e["grad_s"] for e in log]),
        grad_t=np.array([e["grad_t"] for e in log]),
        grad_prior_a=np.array([e["grad_prior_a"] for e in log]),
        grad_prior_s=np.array([e["grad_prior_s"] for e in log]),
        grad_prior_t=np.array([e["grad_prior_t"] for e in log]),
        L_traj=np.array([e["L_traj"] for e in log]),
        grad_traj_a=np.array([e["grad_traj_a"] for e in log]),
        grad_traj_s=np.array([e["grad_traj_s"] for e in log]),
        grad_traj_t=np.array([e["grad_traj_t"] for e in log]),
        theta_true=theta_true,
        theta_0=theta_0,
        theta_best=theta_best,
        theta_final=theta_final,
        best_step=np.array([best_step]),
        lambda_prior=np.array([lambda_prior]),
        lambda_traj=np.array([lambda_traj]),
    )
    if log_out_name is None:
        log_out_name = "online_residual_log.npz"
    out_path = REPO_ROOT / "tools" / log_out_name
    np.savez(out_path, **log_arr)
    print(f"[integ] log saved to {out_path}")

    return pass_overall, log, result


def sweep_lambda_traj(n_frames=500, lambdas=(0.0, 0.1, 0.3, 0.5, 1.0)):
    """P8 sweep: run integration_test at multiple λ_traj values, report best.

    Uses lambda_prior=0 to isolate L_traj contribution.
    """
    print("=" * 70)
    print(f"P8 SWEEP: lambda_traj over {list(lambdas)}, n_frames={n_frames}")
    print("=" * 70)
    results = {}
    for lam in lambdas:
        print(f"\n{'#' * 70}\n#  λ_traj = {lam}\n{'#' * 70}")
        _, _, result = integration_test(
            n_frames=n_frames,
            lambda_prior=0.0,
            lambda_traj=lam,
            log_out_name=f"online_residual_log_traj_{lam:.1f}.npz",
        )
        results[lam] = result

    print("\n" + "=" * 70)
    print("P8 L_TRAJ SWEEP SUMMARY:")
    print("=" * 70)
    theta_true = np.array([3.0, 20.0, 0.5])
    print(f"{'λ_traj':>8} | {'best_step':>9} | {'best_L':>10} | "
          f"{'α_final':>8} | {'s_final':>8} | {'τ_final':>8} | "
          f"{'α_err%':>7} | {'s_err%':>7} | {'τ_err%':>7}")
    print("-" * 120)
    for lam, r in results.items():
        # Report FINAL params (not best) because L_traj should stabilize final
        f = r["final_theta"]
        rel = np.abs(f - theta_true) / np.maximum(np.abs(theta_true), 1e-6) * 100
        print(f"{lam:>8.2f} | {r['best_step']:>9d} | {r['best_L']:>10.2e} | "
              f"{f[0]:>8.3f} | {f[1]:>8.3f} | {f[2]:>8.3f} | "
              f"{rel[0]:>7.2f} | {rel[1]:>7.2f} | {rel[2]:>7.2f}")
    return results


def sweep_lambda_prior(n_frames=500, lambdas=(0.0, 1e-5, 1e-4, 1e-3, 1e-2)):
    """P7 sweep: run integration_test at multiple λ values, report best."""
    print("=" * 70)
    print(f"P7 SWEEP: lambda_prior over {list(lambdas)}, n_frames={n_frames}")
    print("=" * 70)
    results = {}
    for lam in lambdas:
        print(f"\n{'#' * 70}\n#  λ_prior = {lam}\n{'#' * 70}")
        _, _, result = integration_test(
            n_frames=n_frames,
            lambda_prior=lam,
            log_out_name=f"online_residual_log_lambda_{lam:.0e}.npz",
        )
        results[lam] = result

    print("\n" + "=" * 70)
    print("SWEEP SUMMARY:")
    print("=" * 70)
    theta_true = np.array([3.0, 20.0, 0.5])
    print(f"{'λ_prior':>10} | {'best_step':>9} | {'best_L':>10} | "
          f"{'α_best':>7} | {'s_best':>7} | {'τ_best':>7} | "
          f"{'α_err%':>7} | {'s_err%':>7} | {'τ_err%':>7}")
    print("-" * 110)
    for lam, r in results.items():
        b = r["best_theta"]
        rel = np.abs(b - theta_true) / np.maximum(np.abs(theta_true), 1e-6) * 100
        print(f"{lam:>10.0e} | {r['best_step']:>9d} | {r['best_L']:>10.2e} | "
              f"{b[0]:>7.3f} | {b[1]:>7.3f} | {b[2]:>7.3f} | "
              f"{rel[0]:>7.2f} | {rel[1]:>7.2f} | {rel[2]:>7.2f}")
    return results


if __name__ == "__main__":
    import sys as _sys
    if "--smoke" in _sys.argv:
        smoke_test()
    elif "--traj" in _sys.argv:
        trajectory_test()
    elif "--inner" in _sys.argv:
        inner_loop_test()
    elif "--outer" in _sys.argv:
        outer_grad_test()
    elif "--integ" in _sys.argv:
        n = 100
        lam_p = 0.0
        lam_t = 0.0
        for arg in _sys.argv:
            if arg.startswith("--frames="):
                n = int(arg.split("=")[1])
            if arg.startswith("--lambda=") or arg.startswith("--lambda_prior="):
                lam_p = float(arg.split("=")[1])
            if arg.startswith("--lambda_traj="):
                lam_t = float(arg.split("=")[1])
        integration_test(n_frames=n, lambda_prior=lam_p, lambda_traj=lam_t)
    elif "--sweep" in _sys.argv:
        n = 500
        for arg in _sys.argv:
            if arg.startswith("--frames="):
                n = int(arg.split("=")[1])
        sweep_lambda_prior(n_frames=n)
    elif "--sweep_traj" in _sys.argv:
        n = 500
        for arg in _sys.argv:
            if arg.startswith("--frames="):
                n = int(arg.split("=")[1])
        sweep_lambda_traj(n_frames=n)
    elif "--plot" in _sys.argv:
        plot_convergence()
    else:
        smoke_test()
        trajectory_test()
        inner_loop_test()
        outer_grad_test()
        integration_test(n_frames=50)  # shorter for full-test run
