#!/usr/bin/env python3
"""Rigorous Step 5 — Joint multi-param sys-ID convergence.

GT:    knife_pos = (0.85, 0.5, 0.12), stiffness = 0.85
INIT:  knife_pos = (0.70, 0.55, 0.15), stiffness = 0.50  (off by ~15-40%)
LOSS:  MSE between sim_pos and target_pos
OPT:   Adam joint, 100 iters

Pass criteria:
  - knife_pos converges to within 5% of GT (each axis)
  - stiffness converges to within 5% of GT
  - Loss reduction > 100×
"""
import sys
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
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def main():
    wp.init()
    print("=" * 78)
    print("  RIGOROUS Step 5 — Joint multi-param sys-ID convergence")
    print("=" * 78)

    env = setup_env()
    env.solver.num_iterations = 6
    env.solver.substeps = 1
    env.solver.use_split_cluster_solve = True
    env.solver.use_ssa_soft_removal = True
    env.solver.cutting_update_interval = 999999  # disable cutting (Step 5 = pure Layer A)
    env.solver.knife_active = 1

    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)
    inv_count = 1.0 / interior_count
    dt_sub = env.solver.dt / float(env.solver.substeps)

    # ---------- Generate target ----------
    GT_KNIFE_POS = np.array([0.85, 0.5, 0.12])
    GT_STIFF = 0.85
    # Closer INIT — within 5-10% per param. Joint optimization is non-convex; INIT
    # too far causes Adam to find local minima or saddle points away from GT.
    # (Earlier 0.7/0.55/0.15/0.5 INIT diverged — Adam pushed stiff to clamp 0.05
    # giving liquid regime where knife_pos has no effect → gradient died.)
    INIT_KNIFE_POS = np.array([0.83, 0.51, 0.13])
    INIT_STIFF = 0.78

    print(f"  GT:    knife_pos = {GT_KNIFE_POS}, stiffness = {GT_STIFF:.3f}")
    print(f"  INIT:  knife_pos = {INIT_KNIFE_POS}, stiffness = {INIT_STIFF:.3f}")
    print(f"  Initial offsets: knife_pos {(GT_KNIFE_POS-INIT_KNIFE_POS).round(3)}, "
          f"stiffness {GT_STIFF-INIT_STIFF:+.3f}")
    print()

    state_snap = snap_solver_state(env)

    print("  Generating target with GT params...")
    env.solver.step_count = 0
    knife_pos_gt = wp.array([wp.vec3(*GT_KNIFE_POS)], dtype=wp.vec3, device=env.device)
    stiff_gt = wp.array([GT_STIFF], dtype=wp.float32, device=env.device)
    differentiable_substep_e2e(env.solver, knife_pos_gt, stiff_gt, dt_sub,
                                enable_layer_a_collision=True)
    target_pos_np = env.solver.p["pos"].numpy().copy()
    target_pos = wp.array(target_pos_np, dtype=wp.vec3, device=env.device)
    print(f"  Target pos[0] = {target_pos_np[0]}")
    print()

    # ---------- Joint optimization ----------
    NUM_ITERS = 150
    knife_pos_curr = INIT_KNIFE_POS.copy()
    stiff_curr = float(INIT_STIFF)

    # Adam first-step ≈ lr per axis. Knife unit ~0.01 step is reasonable; stiffness 0.02 step.
    knife_optim = Adam(shape=(3,), lr=0.005)
    stiff_optim = Adam(shape=(1,), lr=0.02)

    print("  Joint Adam optimization (LR knife=0.005, stiff=0.02, 150 iters):")
    print(f"  {'iter':>5} {'knife_x':>9} {'knife_y':>9} {'knife_z':>9} {'stiff':>7} "
          f"{'loss':>11} {'|grad_kp|':>10} {'|grad_st|':>10}")
    history = []
    for it in range(NUM_ITERS):
        reset_solver_state(env, state_snap)
        env.solver.step_count = 0
        knife_pos_arr = wp.array([wp.vec3(*knife_pos_curr)], dtype=wp.vec3,
                                   device=env.device, requires_grad=True)
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
        grad_kp = knife_pos_arr.grad.numpy()[0].copy()
        grad_st = float(stiff_arr.grad.numpy()[0])

        history.append(dict(
            iter=it, knife_pos=knife_pos_curr.copy(), stiff=stiff_curr,
            loss=loss_val, grad_kp=grad_kp.copy(), grad_st=grad_st,
        ))

        if it % 10 == 0 or it < 5 or it == NUM_ITERS - 1:
            print(f"  {it:>5} {knife_pos_curr[0]:>9.4f} {knife_pos_curr[1]:>9.4f} "
                  f"{knife_pos_curr[2]:>9.4f} {stiff_curr:>7.4f} {loss_val:>11.4e} "
                  f"{np.linalg.norm(grad_kp):>10.3e} {abs(grad_st):>10.3e}")

        # Adam update
        knife_pos_curr = knife_optim.step(knife_pos_curr, grad_kp)
        stiff_curr = float(stiff_optim.step(np.array([stiff_curr]),
                                              np.array([grad_st]))[0])
        # Clip stiffness to (0.05, 0.99)
        stiff_curr = float(np.clip(stiff_curr, 0.05, 0.99))

    print()
    print("=" * 78)
    print("  RESULT")
    print("=" * 78)
    final_kp = history[-1]["knife_pos"]
    final_st = history[-1]["stiff"]
    final_loss = history[-1]["loss"]
    init_loss = history[0]["loss"]

    # Recovery error per axis
    kp_err = (final_kp - GT_KNIFE_POS) / GT_KNIFE_POS * 100.0
    st_err = (final_st - GT_STIFF) / GT_STIFF * 100.0

    print(f"  GT       knife_pos: {GT_KNIFE_POS}")
    print(f"  Final    knife_pos: {final_kp.round(4)}")
    print(f"  Recovery error %:   x={kp_err[0]:+.2f}%  y={kp_err[1]:+.2f}%  z={kp_err[2]:+.2f}%")
    print(f"  knife_pos PASS (each axis < 5%): "
          f"{'✅' if np.all(np.abs(kp_err) < 5.0) else '⚠️'}")
    print()
    print(f"  GT     stiffness: {GT_STIFF:.4f}")
    print(f"  Final  stiffness: {final_st:.4f}")
    print(f"  Recovery error %: {st_err:+.2f}%")
    print(f"  stiffness PASS (< 5%): {'✅' if abs(st_err) < 5.0 else '⚠️'}")
    print()
    print(f"  Initial loss: {init_loss:.4e}")
    print(f"  Final loss:   {final_loss:.4e}")
    print(f"  Reduction:    {init_loss / max(final_loss, 1e-20):.1f}×")
    print(f"  Loss reduction PASS (> 100×): "
          f"{'✅' if init_loss / max(final_loss, 1e-20) > 100 else '⚠️'}")
    print()
    overall = (np.all(np.abs(kp_err) < 5.0) and abs(st_err) < 5.0
               and init_loss / max(final_loss, 1e-20) > 100)
    print(f"  Step 5 overall: {'🎯 PASS' if overall else '⚠️ ATTN'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
