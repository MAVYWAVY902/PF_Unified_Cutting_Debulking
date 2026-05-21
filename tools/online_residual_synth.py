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


if __name__ == "__main__":
    smoke_test()
