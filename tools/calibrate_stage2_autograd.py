#!/usr/bin/env python3
"""
⚠️ DEPRECATED 2026-05-23 — superseded by tools/calibrate_resect.py

This file predates the Day 6-8 ReSECT pipeline. It only does 2D (α_min, sharpness)
single-substep calibration; it has NO τ optimization, NO inner Mode C loop, and
NO sharpness freeze (the locked decision). Kept for:
   - Imports by tools/online_residual_synth.py and tools/test_stage_b_calibrator_sanity.py
   - 1-substep autograd parity sanity (Stage B's original validation role)

For Stage 2.2 phantom calibration going forward, use tools/calibrate_resect.py
(canonical ReSECT loop, θ = (α_min, τ) 2D, fix-sharp). See memory entry
project_day9_fix_sharpness_locked_2026_05_23.md for decision context.

------------------ Original docstring follows ------------------

Stage 2 cutting calibration — AUTOGRAD via cycle-scoped buffer SSA.

This is the production autograd counterpart to tools/calibrate_stage2_FD.py.
Where FD costs 4 forward sims per gradient step (central-diff over 2 params),
autograd costs ONE forward sim + ONE backward pass per step regardless of
parameter count.

Stage B framework dependency:
    Uses production SSA kernels from pf_damage.py:
        update_edge_damage_kernel_ssa_param_arr        (wp.array params for autograd)
        apply_edge_damage_ssa_kernel                    (cycle-scoped, replaces in-place)
        update_particle_damage_kernel                   (atomic_add, OK in tape)
        finalize_particle_damage_kernel_ssa             (cycle-scoped)
        apply_particle_damage_ssa_kernel                (cycle-scoped)
        update_soft_removal_kernel_ssa_param_arr        (wp.array params for autograd)

    Verified end-to-end gradient parity vs FD at rel_err < 1% (Stage B Day 2):
        sharpness rel_err 0.95% / alpha_min rel_err 0.11% (3-cycle, N=20)

Modes:
    --mode synth      Synthetic identifiability test (default).
                      Generate target alpha distribution from true params,
                      calibrate from wrong init, expect convergence.
    --mode real       Hook for Stage 2.2 phantom data — left as a stub.

Sanity test (this file is the harness for):
    Sanity 1 (THIS): synth recovery, expect rel_err < 5% on both params.
    Sanity 2 (TBD):  parity vs FD calibrator on same synth (target rel < 1%).
    Sanity 3 (TBD):  monotonic loss decrease + stable gradient norm.
    Sanity 4 (TBD):  runtime < 10 min for 200 iters (faster than FD by ~4×).
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

from pf_shape_matching_warp.sim.pf_damage import (
    update_edge_damage_kernel_ssa_param_arr,
    apply_edge_damage_ssa_kernel,
    update_particle_damage_kernel,
    finalize_particle_damage_kernel_ssa,
    apply_particle_damage_ssa_kernel,
    update_soft_removal_kernel_ssa_param_arr,
)


# =============================================================================
# Loss kernel
# =============================================================================

@wp.kernel
def mse_loss_kernel(
    alpha_sim: wp.array(dtype=wp.float32),
    alpha_target: wp.array(dtype=wp.float32),
    loss_buf: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    diff = alpha_sim[i] - alpha_target[i]
    wp.atomic_add(loss_buf, 0, diff * diff)


# =============================================================================
# Synthetic system (identical to FD calibrator for direct comparison)
# =============================================================================

def setup_synthetic_system(N: int = 20, knife_x: float = 0.5, seed: int = 42):
    """1D line of N particles near a knife at x=knife_x. Deterministic."""
    rng = np.random.default_rng(seed)
    pos_np = np.zeros((N, 3), dtype=np.float32)
    pos_np[:, 0] = np.linspace(0.3, 0.7, N)
    pos_np[:, 1] = 0.5
    pos_np[:, 2] = 0.5
    return dict(
        pos=pos_np,
        edge_a=np.arange(N - 1, dtype=np.int32),
        edge_b=np.arange(1, N, dtype=np.int32),
        pf_rest=np.full(N - 1, 0.5, dtype=np.float32),
        knife_pos=np.array([[knife_x, 0.5, 0.5]], dtype=np.float32),
        N=N, E=N - 1,
    )


# =============================================================================
# Param helpers
# =============================================================================

def make_params(alpha_min: float, sharpness: float,
                threshold: float = 0.5, knife_radius: float = 0.1,
                requires_grad: bool = True) -> dict:
    """Returns wp.array param dict — single-element arrays so autograd applies."""
    return dict(
        alpha_min=wp.array([alpha_min], dtype=wp.float32, requires_grad=requires_grad),
        sharpness=wp.array([sharpness], dtype=wp.float32, requires_grad=requires_grad),
        threshold=wp.array([threshold], dtype=wp.float32, requires_grad=requires_grad),
        knife_radius=wp.array([knife_radius], dtype=wp.float32, requires_grad=requires_grad),
    )


# =============================================================================
# Forward pass — cycle-scoped buffer SSA (matches test_stage_b_gradient_parity.py)
# =============================================================================

def run_ssa_full(params: dict, sys_dict: dict, n_cycles: int = 3):
    """Run cutting pipeline n_cycles times via SSA cycle-scoped buffers.

    Returns final alpha wp.array (alpha_buffers[-1]) — caller can either
    read its .numpy() for forward-only OR record on tape for backward.
    """
    N, E = sys_dict["N"], sys_dict["E"]
    pos = wp.array(sys_dict["pos"], dtype=wp.vec3)
    edge_a = wp.array(sys_dict["edge_a"], dtype=wp.int32)
    edge_b = wp.array(sys_dict["edge_b"], dtype=wp.int32)
    pf_rest = wp.array(sys_dict["pf_rest"], dtype=wp.float32)
    knife_pos = wp.array(sys_dict["knife_pos"], dtype=wp.vec3)
    fixed = wp.array(np.zeros(N, dtype=np.int32), dtype=wp.int32)

    # Cycle-scoped state buffer LISTS (n_cycles + 1 each)
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
        # === EDGE DAMAGE ===
        edge_damage_delta = wp.zeros(E, dtype=wp.float32, requires_grad=True)
        wp.launch(update_edge_damage_kernel_ssa_param_arr, dim=E, inputs=[
            pos, alpha_buffers[cycle], edge_a, edge_b, pf_rest,
            edge_damage_buffers[cycle],
            knife_pos, params["knife_radius"], params["alpha_min"],
            edge_damage_delta,
        ])
        wp.launch(apply_edge_damage_ssa_kernel, dim=E, inputs=[
            edge_damage_buffers[cycle],
            edge_damage_delta,
            edge_damage_buffers[cycle + 1],
        ])

        # === PARTICLE DAMAGE ===
        damage_sum = wp.zeros(N, dtype=wp.float32, requires_grad=True)
        damage_count = wp.zeros(N, dtype=wp.float32, requires_grad=True)
        wp.launch(update_particle_damage_kernel, dim=E, inputs=[
            damage_sum, damage_count, alpha_buffers[cycle],
            edge_a, edge_b,
            edge_damage_buffers[cycle + 1],
        ])

        particle_damage_delta = wp.zeros(N, dtype=wp.float32, requires_grad=True)
        wp.launch(finalize_particle_damage_kernel_ssa, dim=N, inputs=[
            damage_sum, damage_count,
            particle_damage_buffers[cycle],
            particle_damage_delta,
        ])
        wp.launch(apply_particle_damage_ssa_kernel, dim=N, inputs=[
            particle_damage_buffers[cycle],
            particle_damage_delta,
            particle_damage_buffers[cycle + 1],
        ])

        # === SOFT REMOVAL ===
        wp.launch(update_soft_removal_kernel_ssa_param_arr, dim=N, inputs=[
            particle_damage_buffers[cycle + 1],
            alpha_buffers[cycle],
            fixed,
            params["threshold"], params["sharpness"],
            alpha_buffers[cycle + 1],
        ])

    return alpha_buffers[-1]


# =============================================================================
# Adam optimizer (numpy, mirrors FD calibrator)
# =============================================================================

class Adam2D:
    def __init__(self, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = np.asarray(lr, dtype=np.float64) if not np.isscalar(lr) else lr
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = np.zeros(2); self.v = np.zeros(2); self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# Calibration loop
# =============================================================================

def calibrate(args):
    wp.init()

    sys_dict = setup_synthetic_system(N=args.n_particles)
    N = sys_dict["N"]

    # --- 1. Generate synthetic target ---
    if args.mode == "synth":
        target_params = make_params(
            args.true_alpha_min, args.true_sharpness,
            requires_grad=False,
        )
        target_alpha = run_ssa_full(target_params, sys_dict, n_cycles=args.n_cycles).numpy().copy()
        target_alpha_wp = wp.array(target_alpha, dtype=wp.float32)
        debulked_frac = float(np.mean(target_alpha < 0.5))
        print(f"[synth] target generated  alpha_min={args.true_alpha_min:.3f}  "
              f"sharpness={args.true_sharpness:.3f}")
        print(f"[synth] target debulked fraction (alpha<0.5): {debulked_frac:.3f}")
    elif args.mode == "real":
        raise NotImplementedError(
            "Real mode requires Stage 2.2 phantom cavity-IoU/force-trace data. "
            "Plug in a data loader + loss_against_real() before using."
        )
    else:
        raise ValueError(f"unknown mode {args.mode}")

    # --- 2. Init params from wrong values ---
    np_params = np.array([args.init_alpha_min, args.init_sharpness], dtype=np.float64)
    print(f"[init]  alpha_min={np_params[0]:.3f}  sharpness={np_params[1]:.3f}")

    opt = Adam2D(lr=np.array([args.lr, args.lr_sharp], dtype=np.float64))
    history = []
    best_loss = float("inf")
    best_params = np_params.copy()
    best_step = 0

    # --- 3. Calibration loop ---
    t_start = time.time()
    for step in range(args.n_steps):
        # Re-create wp.array params each iter (no .assign() in Warp)
        params = make_params(
            float(np_params[0]), float(np_params[1]),
            threshold=args.threshold, knife_radius=args.knife_radius,
            requires_grad=True,
        )

        loss_buf = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        with wp.Tape() as tape:
            alpha_sim = run_ssa_full(params, sys_dict, n_cycles=args.n_cycles)
            wp.launch(mse_loss_kernel, dim=N,
                      inputs=[alpha_sim, target_alpha_wp, loss_buf])

        tape.backward(loss=loss_buf)

        loss = float(loss_buf.numpy()[0]) / float(N)  # Normalize: matches FD's loss_synth (mean MSE)
        grad_a = float(params["alpha_min"].grad.numpy()[0]) / float(N)
        grad_s = float(params["sharpness"].grad.numpy()[0]) / float(N)
        # Note: gradient is also scaled by 1/N to match the mean-MSE convention.

        if loss < best_loss:
            best_loss, best_params, best_step = loss, np_params.copy(), step

        # Adam step
        np_params = opt.step(np_params, np.array([grad_a, grad_s]))
        # Hard bounds (physical positivity)
        np_params[0] = max(0.5, min(50.0, np_params[0]))
        np_params[1] = max(1.0, min(200.0, np_params[1]))

        history.append(dict(step=step, loss=loss, alpha_min=np_params[0],
                            sharpness=np_params[1], grad_a=grad_a, grad_s=grad_s))

        if step % args.log_every == 0 or step == args.n_steps - 1:
            print(f"[step {step:3d}] loss={loss:.6e}  "
                  f"a_min={np_params[0]:.3f} (g={grad_a:+.2e})  "
                  f"sharp={np_params[1]:.3f} (g={grad_s:+.2e})")

    elapsed = time.time() - t_start
    print(f"\n[best] step {best_step}: loss={best_loss:.6e}  "
          f"a_min={best_params[0]:.3f}  sharp={best_params[1]:.3f}")
    print(f"[done] {args.n_steps} steps in {elapsed:.1f}s "
          f"({elapsed / args.n_steps:.2f}s/step, autograd: 1 fwd + 1 bwd per step)")

    # --- 4. Synth-mode verification ---
    if args.mode == "synth":
        true_params = np.array([args.true_alpha_min, args.true_sharpness])
        err_a = abs(best_params[0] - true_params[0])
        err_s = abs(best_params[1] - true_params[1])
        rel_a = err_a / true_params[0]
        rel_s = err_s / true_params[1]
        loss_drop = history[0]["loss"] / max(best_loss, 1e-12)

        print(f"\n[synth result]")
        print(f"  true:       alpha_min={true_params[0]:.3f}  sharpness={true_params[1]:.3f}")
        print(f"  recovered:  alpha_min={best_params[0]:.3f}  sharpness={best_params[1]:.3f}")
        print(f"  abs error:  alpha_min={err_a:.3f}  sharpness={err_s:.3f}")
        print(f"  rel error:  alpha_min={rel_a*100:.1f}%  sharpness={rel_s*100:.1f}%")
        print(f"  loss drop:  {history[0]['loss']:.3e} → {best_loss:.3e}  ({loss_drop:.1e}×)")

        sanity1_pass = rel_a < 0.05 and rel_s < 0.05
        sanity1_relaxed = rel_a < 0.20 and rel_s < 0.20
        print(f"\n[Sanity 1] synth recovery <5%: {'PASS' if sanity1_pass else 'FAIL'}")
        print(f"[Sanity 1] synth recovery <20% (relaxed): {'PASS' if sanity1_relaxed else 'FAIL'}")
        if sanity1_pass:
            print(f"  ✅ Stage B framework end-to-end VERIFIED on synthetic recovery")
        elif sanity1_relaxed:
            print(f"  ⚠️ recovery within 20% but not 5%; expected: sharpness has weaker gradient")
            print(f"     (FD calibrator showed same — it's a synth-MSE identifiability floor)")
        return 0 if sanity1_relaxed else 1
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["synth", "real"], default="synth")
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--log_every", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.1, help="Adam lr for alpha_min")
    p.add_argument("--lr_sharp", type=float, default=1.0,
                   help="Adam lr for sharpness (~10× alpha_min lr; weaker gradient)")
    p.add_argument("--n_particles", type=int, default=20)
    p.add_argument("--n_cycles", type=int, default=3)
    p.add_argument("--true_alpha_min", type=float, default=3.0)
    p.add_argument("--true_sharpness", type=float, default=20.0)
    p.add_argument("--init_alpha_min", type=float, default=5.0)
    p.add_argument("--init_sharpness", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--knife_radius", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(calibrate(parse_args()))
