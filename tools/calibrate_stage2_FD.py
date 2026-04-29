#!/usr/bin/env python3
"""
Stage 2 cutting calibration — central-difference FD over (alpha_min, sharpness).

Why FD instead of Warp tape autograd:
    The production update_soft_removal_kernel does direct write `alpha[i] = ...`
    while alpha is also read upstream within the same Tape — read-write-read
    on a single buffer. POC poc_stage2_path_b_FAILED_keep_for_record.py
    reproduced the same silent-zero gradient that PlasticineLab (Huang 2021)
    reports as "detach-reattach failure" (§3.4). They fall back to FD/CMA-ES
    for those sub-problems; we follow the same precedent.

Why FD is fine here:
    Stage 2 has only 2 parameters: alpha_min, sharpness. Central diff costs
    4 sims per gradient step. Time budget for 50 steps:
        synthetic mode (N=20): ~20s total
        real-data mode (full mesh): ~1.7h total

Modes:
    --mode synth      Synthetic identifiability test (default).
                      Generate target alpha distribution from true params,
                      calibrate from wrong init, expect convergence.
    --mode real       Hook for real-world data. The loss_against_real()
                      function is left as a stub for the Stage 2.2
                      experimental pipeline to fill in.

Outputs:
    - Console log every step (params, loss, FD grad)
    - PNG plot of loss + param trajectories at end (if matplotlib available)
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pf_shape_matching_warp.sim.pf_damage import (
    update_edge_damage_kernel_ssa,
    apply_edge_damage_delta_kernel,
    update_particle_damage_kernel,
    finalize_particle_damage_kernel_ssa,
    apply_particle_damage_delta_kernel,
    update_soft_removal_kernel,
)


# =============================================================================
# Synthetic test system (small, fast — for skeleton validation)
# =============================================================================

def setup_synthetic_system(N: int = 20, knife_x: float = 0.5, seed: int = 42):
    """1D line of N particles near a knife at x=knife_x. Deterministic."""
    rng = np.random.default_rng(seed)
    pos_np = np.zeros((N, 3), dtype=np.float32)
    pos_np[:, 0] = np.linspace(0.3, 0.7, N)
    pos_np[:, 1] = 0.5
    pos_np[:, 2] = 0.5
    edge_a_np = np.arange(N - 1, dtype=np.int32)
    edge_b_np = np.arange(1, N, dtype=np.int32)
    pf_rest_np = np.full(N - 1, 0.5, dtype=np.float32)
    knife_pos_np = np.array([[knife_x, 0.5, 0.5]], dtype=np.float32)
    fixed_np = np.zeros(N, dtype=np.int32)

    return dict(
        pos=pos_np, edge_a=edge_a_np, edge_b=edge_b_np,
        pf_rest=pf_rest_np, knife_pos=knife_pos_np, fixed=fixed_np,
        N=N, E=N - 1,
    )


# =============================================================================
# Forward simulation — single cutting cycle pipeline
# =============================================================================

def run_cutting_forward(
    sys_dict: dict,
    alpha_min: float,
    sharpness: float,
    knife_radius: float = 0.1,
    threshold: float = 0.5,
    n_cycles: int = 3,
) -> np.ndarray:
    """Run cutting pipeline n_cycles times. Returns final alpha array (N,)."""
    N, E = sys_dict["N"], sys_dict["E"]

    pos = wp.array(sys_dict["pos"], dtype=wp.vec3)
    edge_a = wp.array(sys_dict["edge_a"], dtype=wp.int32)
    edge_b = wp.array(sys_dict["edge_b"], dtype=wp.int32)
    pf_rest = wp.array(sys_dict["pf_rest"], dtype=wp.float32)
    knife_pos = wp.array(sys_dict["knife_pos"], dtype=wp.vec3)
    fixed = wp.array(sys_dict["fixed"], dtype=wp.int32)

    edge_damage = wp.zeros(E, dtype=wp.float32)
    particle_damage = wp.zeros(N, dtype=wp.float32)
    alpha = wp.array(np.ones(N, dtype=np.float32), dtype=wp.float32)

    for _ in range(n_cycles):
        edge_damage_delta = wp.zeros(E, dtype=wp.float32)
        wp.launch(update_edge_damage_kernel_ssa, dim=E, inputs=[
            pos, alpha, edge_a, edge_b, pf_rest, edge_damage,
            knife_pos, knife_radius, alpha_min, edge_damage_delta,
        ])
        wp.launch(apply_edge_damage_delta_kernel, dim=E,
                  inputs=[edge_damage, edge_damage_delta])

        damage_sum = wp.zeros(N, dtype=wp.float32)
        damage_count = wp.zeros(N, dtype=wp.float32)
        wp.launch(update_particle_damage_kernel, dim=E, inputs=[
            damage_sum, damage_count, alpha, edge_a, edge_b, edge_damage,
        ])

        particle_damage_delta = wp.zeros(N, dtype=wp.float32)
        wp.launch(finalize_particle_damage_kernel_ssa, dim=N, inputs=[
            damage_sum, damage_count, particle_damage, particle_damage_delta,
        ])
        wp.launch(apply_particle_damage_delta_kernel, dim=N,
                  inputs=[particle_damage, particle_damage_delta])

        wp.launch(update_soft_removal_kernel, dim=N, inputs=[
            particle_damage, alpha, fixed, threshold, sharpness,
        ])

    wp.synchronize()
    return alpha.numpy().copy()


# =============================================================================
# Loss functions
# =============================================================================

def loss_synth(alpha: np.ndarray, target_alpha: np.ndarray) -> float:
    """MSE between current alpha distribution and target (identifiability test)."""
    return float(np.mean((alpha - target_alpha) ** 2))


def loss_against_real(alpha: np.ndarray, real_data: dict) -> float:
    """STUB for Stage 2.2 real-world calibration.

    To fill in:
      - real_data should contain measured cavity geometry (e.g. groove depth
        profile, cavity volume, debulked mask).
      - Convert sim alpha → comparable metric (e.g. debulked fraction =
        mean(alpha < 0.5), or fitted cavity volume).
      - Return MSE / IoU loss.
    """
    raise NotImplementedError(
        "loss_against_real() is a stub. Plug in Stage 2.2 measurement parser."
    )


# =============================================================================
# Central-diff FD gradient
# =============================================================================

def fd_grad(
    sys_dict: dict, params: np.ndarray, loss_fn,
    eps: Tuple[float, float] = (0.1, 0.5),
    fwd_kwargs: dict | None = None,
) -> Tuple[np.ndarray, float]:
    """Central diff. params = [alpha_min, sharpness]. Returns (grad, loss_at_center)."""
    fwd_kwargs = fwd_kwargs or {}
    grad = np.zeros_like(params)
    for i, eps_i in enumerate(eps):
        p_plus = params.copy(); p_plus[i] += eps_i
        p_minus = params.copy(); p_minus[i] -= eps_i
        a_plus = run_cutting_forward(sys_dict, float(p_plus[0]), float(p_plus[1]), **fwd_kwargs)
        a_minus = run_cutting_forward(sys_dict, float(p_minus[0]), float(p_minus[1]), **fwd_kwargs)
        grad[i] = (loss_fn(a_plus) - loss_fn(a_minus)) / (2.0 * eps_i)

    a_center = run_cutting_forward(sys_dict, float(params[0]), float(params[1]), **fwd_kwargs)
    return grad, loss_fn(a_center)


# =============================================================================
# Adam optimizer (2-D, manual)
# =============================================================================

class Adam2D:
    def __init__(self, lr: float = 0.1, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(2); self.v = np.zeros(2); self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# Main calibration loop
# =============================================================================

def calibrate(args):
    wp.init()

    # --- System setup ---
    sys_dict = setup_synthetic_system(N=args.n_particles)
    fwd_kwargs = dict(knife_radius=0.1, threshold=0.5, n_cycles=args.n_cycles)

    # --- Target generation (synth mode) or load (real mode) ---
    if args.mode == "synth":
        true_params = np.array([args.true_alpha_min, args.true_sharpness], dtype=np.float64)
        target_alpha = run_cutting_forward(
            sys_dict, float(true_params[0]), float(true_params[1]), **fwd_kwargs
        )
        debulked_frac = float(np.mean(target_alpha < 0.5))
        print(f"[synth] target generated with alpha_min={true_params[0]:.3f}, "
              f"sharpness={true_params[1]:.3f}")
        print(f"[synth] target debulked fraction (alpha<0.5): {debulked_frac:.3f}")

        def loss_fn(alpha): return loss_synth(alpha, target_alpha)

    elif args.mode == "real":
        raise NotImplementedError(
            "Real mode requires Stage 2.2 cavity measurement data. "
            "Fill in loss_against_real() and a data loader before using --mode real."
        )
    else:
        raise ValueError(f"unknown mode {args.mode}")

    # --- Init params (deliberately wrong in synth mode) ---
    params = np.array([args.init_alpha_min, args.init_sharpness], dtype=np.float64)
    print(f"[init]  alpha_min={params[0]:.3f}, sharpness={params[1]:.3f}")

    opt = Adam2D(lr=args.lr)
    history = []

    # --- Calibration loop ---
    t_start = time.time()
    for step in range(args.n_steps):
        grad, loss = fd_grad(sys_dict, params, loss_fn,
                             eps=(args.eps_alpha, args.eps_sharp),
                             fwd_kwargs=fwd_kwargs)
        params = opt.step(params, grad)
        # Hard bounds (physics: alpha_min > 0, sharpness > 0)
        params[0] = max(0.5, min(50.0, params[0]))
        params[1] = max(1.0, min(200.0, params[1]))

        history.append(dict(step=step, loss=loss, alpha_min=params[0],
                            sharpness=params[1], grad_a=grad[0], grad_s=grad[1]))
        if step % args.log_every == 0 or step == args.n_steps - 1:
            print(f"[step {step:3d}] loss={loss:.6e}  "
                  f"a_min={params[0]:.3f} (g={grad[0]:+.2e})  "
                  f"sharp={params[1]:.3f} (g={grad[1]:+.2e})")

    elapsed = time.time() - t_start
    print(f"\n[done] {args.n_steps} steps in {elapsed:.1f}s "
          f"({elapsed / args.n_steps:.2f}s/step, "
          f"{4 * args.n_steps} forward sims total + {args.n_steps} center evals)")

    # --- Report (synth mode) ---
    if args.mode == "synth":
        err_a = abs(params[0] - true_params[0])
        err_s = abs(params[1] - true_params[1])
        print(f"\n[synth result]")
        print(f"  true:       alpha_min={true_params[0]:.3f}, sharpness={true_params[1]:.3f}")
        print(f"  recovered:  alpha_min={params[0]:.3f}, sharpness={params[1]:.3f}")
        print(f"  abs error:  alpha_min={err_a:.3f}, sharpness={err_s:.3f}")
        # Loose pass criterion: within 20% of true value
        rel_a = err_a / true_params[0]
        rel_s = err_s / true_params[1]
        passed = rel_a < 0.20 and rel_s < 0.20
        print(f"  rel error:  alpha_min={rel_a*100:.1f}%, sharpness={rel_s*100:.1f}%")
        print(f"  identifiability: {'PASS' if passed else 'FAIL'}")

    # --- Optional plot ---
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            steps = [h["step"] for h in history]
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            axes[0].plot(steps, [h["loss"] for h in history]); axes[0].set_yscale("log")
            axes[0].set_title("loss"); axes[0].set_xlabel("step")
            axes[1].plot(steps, [h["alpha_min"] for h in history], label="estimate")
            if args.mode == "synth":
                axes[1].axhline(true_params[0], ls="--", color="r", label="true")
            axes[1].set_title("alpha_min"); axes[1].legend(); axes[1].set_xlabel("step")
            axes[2].plot(steps, [h["sharpness"] for h in history], label="estimate")
            if args.mode == "synth":
                axes[2].axhline(true_params[1], ls="--", color="r", label="true")
            axes[2].set_title("sharpness"); axes[2].legend(); axes[2].set_xlabel("step")
            out = REPO_ROOT / "tools" / f"calibrate_stage2_FD_{args.mode}.png"
            plt.tight_layout(); plt.savefig(out, dpi=120)
            print(f"[plot] saved to {out}")
        except ImportError:
            print("[plot] matplotlib not available, skipping")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["synth", "real"], default="synth")
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--log_every", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.1)

    # Synthetic-mode params
    p.add_argument("--n_particles", type=int, default=20)
    p.add_argument("--n_cycles", type=int, default=3,
                   help="cutting cycles per forward sim")
    p.add_argument("--true_alpha_min", type=float, default=3.0)
    p.add_argument("--true_sharpness", type=float, default=20.0)
    p.add_argument("--init_alpha_min", type=float, default=5.0)
    p.add_argument("--init_sharpness", type=float, default=10.0)

    # FD step sizes (parameter-scale-aware)
    p.add_argument("--eps_alpha", type=float, default=0.1)
    p.add_argument("--eps_sharp", type=float, default=0.5)

    p.add_argument("--plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    calibrate(parse_args())
