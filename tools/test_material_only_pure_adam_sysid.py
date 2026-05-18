#!/usr/bin/env python3
"""Material-only sys-ID — single Adam, no multi-restart, no Optuna.

Mirrors REAL phantom calibration workflow:
  - Knife trajectory: GIVEN (from robot encoder)
  - Initial mesh: GIVEN (from CT/known geometry)
  - UNKNOWN: stiffness only (1 dim)

Test setup:
  - Knife held at GT (0.85, 0.5, 0.12) — represents known robot trajectory
  - Stiffness varied: INITs from 0.20 to 0.95 (wide range)
  - Adam 100 iter, LR=0.05 (moderate, not aggressive)
  - Pass criterion: all INITs converge to <2% of GT (no local min issue)

If single Adam works from any INIT in [0.2, 0.95] — paper is clean
("standard pure Adam, no multi-restart needed").
"""
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

wp.config.verify_autograd_array_access = False

from path4_falsification_test_H_isolate import setup_env
from path4_poc_v1_dual_purpose import snap_solver_state, reset_solver_state
from test_layer_a_phase2b_e2e_stiffness_sysid import (
    differentiable_substep_e2e, loss_pos_mse_kernel,
)


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = 0.0; self.v = 0.0; self.t = 0

    def step(self, p, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def calibrate_stiffness(env, init_stiff, GT_KNIFE_POS, target_pos, state_snap,
                          dt_sub, interior_mask_wp, interior_count, inv_count,
                          num_iters=100, lr=0.05):
    """Calibrate stiffness only, knife_pos held at GT. Pure Adam, single run."""
    stiff_curr = float(init_stiff)
    optim = Adam(lr=lr)
    history = []

    for it in range(num_iters):
        reset_solver_state(env, state_snap)
        env.solver.step_count = 0
        knife_pos_arr = wp.array([wp.vec3(*GT_KNIFE_POS)], dtype=wp.vec3,
                                   device=env.device, requires_grad=False)
        stiff_arr = wp.array([stiff_curr], dtype=wp.float32, device=env.device,
                              requires_grad=True)
        loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)

        with wp.Tape() as tape:
            differentiable_substep_e2e(env.solver, knife_pos_arr, stiff_arr, dt_sub,
                                        enable_layer_a_collision=True)
            wp.launch(loss_pos_mse_kernel, dim=env.solver.N,
                      inputs=[env.solver.p["pos"], target_pos, interior_mask_wp,
                              inv_count, loss_buf], device=env.device)
        tape.backward(loss=loss_buf)
        wp.synchronize()

        loss_val = float(loss_buf.numpy()[0])
        grad_st = float(stiff_arr.grad.numpy()[0])
        history.append(dict(iter=it, stiff=stiff_curr, loss=loss_val, grad=grad_st))

        # Adam update with clip to physical range [0.05, 0.99]
        stiff_curr = float(optim.step(stiff_curr, grad_st))
        stiff_curr = float(np.clip(stiff_curr, 0.05, 0.99))

    return dict(final_stiff=stiff_curr,
                final_loss=history[-1]["loss"],
                init_loss=history[0]["loss"],
                history=history)


def main():
    wp.init()
    print("=" * 78)
    print("  MATERIAL-ONLY SYS-ID — Single Adam, No Multi-restart")
    print("=" * 78)
    print("  Setup: knife_pos held at GT (real-world: known from robot encoder)")
    print("         Optimize stiffness only (1 dim)")
    print("         Adam LR=0.05, 100 iter")
    print()

    env = setup_env()
    env.solver.num_iterations = 6
    env.solver.substeps = 1
    env.solver.use_split_cluster_solve = True
    env.solver.use_ssa_soft_removal = True
    env.solver.cutting_update_interval = 999999
    env.solver.knife_active = 1

    GT_KNIFE_POS = np.array([0.85, 0.5, 0.12])
    GT_STIFF = 0.85

    # Generate target at GT
    state_snap = snap_solver_state(env)
    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)
    inv_count = 1.0 / interior_count
    dt_sub = env.solver.dt / float(env.solver.substeps)

    env.solver.step_count = 0
    knife_pos_gt = wp.array([wp.vec3(*GT_KNIFE_POS)], dtype=wp.vec3, device=env.device)
    stiff_gt = wp.array([GT_STIFF], dtype=wp.float32, device=env.device)
    differentiable_substep_e2e(env.solver, knife_pos_gt, stiff_gt, dt_sub,
                                enable_layer_a_collision=True)
    target_pos_np = env.solver.p["pos"].numpy().copy()
    target_pos = wp.array(target_pos_np, dtype=wp.vec3, device=env.device)
    print(f"  GT: stiffness = {GT_STIFF}, knife_pos held at {GT_KNIFE_POS}")
    print()

    # Sweep INIT stiffness from far-low to far-high
    INITS = [0.20, 0.30, 0.50, 0.70, 0.95]
    print(f"  {'INIT':>6} {'final_stiff':>13} {'st_err':>10} {'init_loss':>12} "
          f"{'final_loss':>12} {'reduction':>10} {'time':>6} {'verdict':>10}")
    results = []
    for init_st in INITS:
        t0 = time.time()
        res = calibrate_stiffness(env, init_st, GT_KNIFE_POS, target_pos, state_snap,
                                    dt_sub, interior_mask_wp, interior_count, inv_count,
                                    num_iters=100, lr=0.05)
        t = time.time() - t0
        st_err = (res["final_stiff"] - GT_STIFF) / GT_STIFF * 100
        reduction = res["init_loss"] / max(res["final_loss"], 1e-20)
        verdict = "✅ <1%" if abs(st_err) < 1.0 else (
            "✅ <2%" if abs(st_err) < 2.0 else (
            "⚠️ <5%" if abs(st_err) < 5.0 else "❌ FAIL"))
        print(f"  {init_st:>6.2f} {res['final_stiff']:>13.4f} {st_err:>+9.2f}% "
              f"{res['init_loss']:>12.3e} {res['final_loss']:>12.3e} "
              f"{reduction:>10.0f}x {t:>5.1f}s {verdict:>10}")
        results.append((init_st, res, st_err))

    print()
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)
    all_pass = all(abs(st_err) < 2.0 for _, _, st_err in results)
    print(f"  All INITs converge to <2% of GT: {'✅ YES' if all_pass else '⚠️ NO'}")
    if all_pass:
        print(f"  → Pure Adam single run (no multi-restart) sufficient for material-only sys-ID")
        print(f"  → No Optuna needed, no fallback needed")
        print(f"  → Paper claim: 'pure Adam from any reasonable stiffness prior converges robustly'")
    else:
        far_inits_failed = [(i, e) for i, _, e in results if abs(e) >= 2.0]
        print(f"  Failed INITs: {far_inits_failed}")
        print(f"  → May need multi-restart or better INIT prior")

    return 0


if __name__ == "__main__":
    sys.exit(main())
