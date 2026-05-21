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


if __name__ == "__main__":
    import sys as _sys
    if "--smoke" in _sys.argv:
        smoke_test()
    elif "--traj" in _sys.argv:
        trajectory_test()
    else:
        smoke_test()
        trajectory_test()
