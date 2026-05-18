#!/usr/bin/env python3
"""Pure Adam sys-ID rigorous comparison.

Tests if pure Adam (mirror DiSECt) can replace Optuna+Adam (SliceIt) for our
joint 4-dim sys-ID problem.

Modes:
  A: Single Adam, close INIT (≤10% off GT) — replicates Step 5 v3 (PASSED)
  B: Single Adam, far INIT (30% off GT) — replicates Step 5 v2 (FAILED, control)
  C: Multi-restart Adam (10 random INITs, take best) — pure-Adam alternative
                                                       to Optuna stage 1

GT:    knife_pos = (0.85, 0.5, 0.12), stiffness = 0.85
Loss:  MSE between sim_pos and target_pos (single substep)

Pass criteria:
  Each mode: recovery error <5% per param, loss reduction >100x
  Mode C compares to: did multi-restart Adam find GT basin without Optuna?
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


def setup_target(env, GT_KNIFE_POS, GT_STIFF):
    """Generate target_pos by running sim at GT params."""
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
    return target_pos, state_snap, dt_sub, interior_mask_wp, interior_count, inv_count


def run_adam(env, init_kp, init_st, target_pos, state_snap, dt_sub,
              interior_mask_wp, interior_count, inv_count,
              num_iters=80, knife_lr=0.005, stiff_lr=0.02, verbose=False):
    """Run Adam optimization from given INIT, return final params + loss trajectory."""
    knife_pos_curr = init_kp.copy()
    stiff_curr = float(init_st)
    knife_optim = Adam(shape=(3,), lr=knife_lr)
    stiff_optim = Adam(shape=(1,), lr=stiff_lr)
    history = []

    for it in range(num_iters):
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
        history.append(dict(iter=it, knife_pos=knife_pos_curr.copy(),
                              stiff=stiff_curr, loss=loss_val))

        # Adam update
        knife_pos_curr = knife_optim.step(knife_pos_curr, grad_kp)
        stiff_curr = float(stiff_optim.step(np.array([stiff_curr]),
                                              np.array([grad_st]))[0])
        stiff_curr = float(np.clip(stiff_curr, 0.05, 0.99))

    return dict(final_kp=knife_pos_curr, final_st=stiff_curr,
                final_loss=history[-1]["loss"],
                init_loss=history[0]["loss"],
                history=history)


def main():
    wp.init()
    print("=" * 78)
    print("  PURE ADAM SYS-ID: Multi-mode comparison")
    print("=" * 78)
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

    print("  Generating target with GT params...")
    target_pos, state_snap, dt_sub, interior_mask_wp, interior_count, inv_count = \
        setup_target(env, GT_KNIFE_POS, GT_STIFF)
    print(f"  GT: knife_pos={GT_KNIFE_POS}, stiffness={GT_STIFF}")
    print()

    # ============= Mode A: Close INIT =============
    print("=" * 78)
    print("  Mode A: Single Adam, CLOSE INIT (≤10% off GT)")
    print("=" * 78)
    init_kp_close = np.array([0.83, 0.51, 0.13])
    init_st_close = 0.78
    print(f"  INIT: knife_pos={init_kp_close}, stiff={init_st_close}")
    t0 = time.time()
    res_A = run_adam(env, init_kp_close, init_st_close, target_pos, state_snap,
                      dt_sub, interior_mask_wp, interior_count, inv_count,
                      num_iters=80)
    t_A = time.time() - t0

    kp_err_A = (res_A["final_kp"] - GT_KNIFE_POS) / GT_KNIFE_POS * 100
    st_err_A = (res_A["final_st"] - GT_STIFF) / GT_STIFF * 100
    A_pass = np.all(np.abs(kp_err_A) < 5.0) and abs(st_err_A) < 5.0
    print(f"  Final: kp={res_A['final_kp'].round(4)}, st={res_A['final_st']:.4f}")
    print(f"  Recovery err: kp({kp_err_A.round(2)}%), st({st_err_A:+.2f}%)")
    print(f"  Loss drop:   {res_A['init_loss']/max(res_A['final_loss'],1e-20):.0f}x")
    print(f"  Time:        {t_A:.1f}s")
    print(f"  Mode A: {'✅ PASS' if A_pass else '⚠️ ATTN'}")
    print()

    # ============= Mode B: Far INIT =============
    print("=" * 78)
    print("  Mode B: Single Adam, FAR INIT (30% off GT) — control")
    print("=" * 78)
    init_kp_far = np.array([0.70, 0.55, 0.15])
    init_st_far = 0.50
    print(f"  INIT: knife_pos={init_kp_far}, stiff={init_st_far}")
    t0 = time.time()
    res_B = run_adam(env, init_kp_far, init_st_far, target_pos, state_snap,
                      dt_sub, interior_mask_wp, interior_count, inv_count,
                      num_iters=80)
    t_B = time.time() - t0

    kp_err_B = (res_B["final_kp"] - GT_KNIFE_POS) / GT_KNIFE_POS * 100
    st_err_B = (res_B["final_st"] - GT_STIFF) / GT_STIFF * 100
    B_pass = np.all(np.abs(kp_err_B) < 5.0) and abs(st_err_B) < 5.0
    print(f"  Final: kp={res_B['final_kp'].round(4)}, st={res_B['final_st']:.4f}")
    print(f"  Recovery err: kp({kp_err_B.round(2)}%), st({st_err_B:+.2f}%)")
    print(f"  Loss drop:   {res_B['init_loss']/max(res_B['final_loss'],1e-20):.0f}x")
    print(f"  Time:        {t_B:.1f}s")
    print(f"  Mode B: {'✅ PASS' if B_pass else '⚠️ ATTN (stuck local min, expected)'}")
    print()

    # ============= Mode C: Multi-restart with B+C improvements =============
    print("=" * 78)
    print("  Mode C (improved): 10 restarts × 150 iter + MEDIAN-of-converged selection")
    print("=" * 78)
    NUM_RESTARTS = 10
    ITERS_PER_RESTART = 150              # B: matches Step 5 v3 (was 60)
    CONVERGED_LOSS_THRESHOLD = 1e-5      # restarts with final_loss < this = "converged"
    rng = np.random.default_rng(42)
    all_results = []
    print(f"  {'restart':>8} {'init_kp':>22} {'init_st':>8} {'final_loss':>12} "
          f"{'kp_err %':>15} {'st_err %':>10} {'class':>10}")
    t0 = time.time()
    for r in range(NUM_RESTARTS):
        init_kp = np.array([
            rng.uniform(0.65, 0.95),
            rng.uniform(0.40, 0.60),
            rng.uniform(0.08, 0.18),
        ])
        init_st = rng.uniform(0.30, 0.95)
        res = run_adam(env, init_kp, init_st, target_pos, state_snap,
                        dt_sub, interior_mask_wp, interior_count, inv_count,
                        num_iters=ITERS_PER_RESTART)
        res["init_kp"] = init_kp.copy()
        res["init_st"] = float(init_st)
        kp_err = (res["final_kp"] - GT_KNIFE_POS) / GT_KNIFE_POS * 100
        st_err = (res["final_st"] - GT_STIFF) / GT_STIFF * 100
        res["kp_err_pct"] = kp_err
        res["st_err_pct"] = st_err
        converged = res["final_loss"] < CONVERGED_LOSS_THRESHOLD
        cls = "✓ converged" if converged else "✗ stuck"
        all_results.append(res)
        print(f"  {r:>8} {str(init_kp.round(2)):>22} {init_st:>8.3f} "
              f"{res['final_loss']:>12.3e} {str(kp_err.round(1)):>15} "
              f"{st_err:>+10.2f} {cls:>10}")
    t_C = time.time() - t0

    converged = [r for r in all_results if r["final_loss"] < CONVERGED_LOSS_THRESHOLD]
    stuck = [r for r in all_results if r["final_loss"] >= CONVERGED_LOSS_THRESHOLD]
    print()
    print(f"  Converged: {len(converged)}/{NUM_RESTARTS}, Stuck: {len(stuck)}/{NUM_RESTARTS}")

    # Selection criterion 1 (old): min loss
    best_by_loss = min(all_results, key=lambda r: r["final_loss"])
    kp_err_loss = best_by_loss["kp_err_pct"]
    st_err_loss = best_by_loss["st_err_pct"]
    pass_by_loss = (np.all(np.abs(kp_err_loss) < 5.0)
                     and abs(st_err_loss) < 5.0)
    print(f"\n  ─── Selection criterion 1: best-by-loss ───")
    print(f"    Final: kp={best_by_loss['final_kp'].round(4)}, "
          f"st={best_by_loss['final_st']:.4f}")
    print(f"    Err:   kp({kp_err_loss.round(2)}%), st({st_err_loss:+.2f}%)")
    print(f"    Verdict: {'✅ PASS' if pass_by_loss else '⚠️ borderline'}")

    # Selection criterion 2 (new B+C): median of converged
    if len(converged) > 0:
        med_kp = np.median(np.array([r["final_kp"] for r in converged]), axis=0)
        med_st = float(np.median([r["final_st"] for r in converged]))
        kp_err_med = (med_kp - GT_KNIFE_POS) / GT_KNIFE_POS * 100
        st_err_med = (med_st - GT_STIFF) / GT_STIFF * 100
        pass_by_med = (np.all(np.abs(kp_err_med) < 5.0)
                        and abs(st_err_med) < 5.0)
        print(f"\n  ─── Selection criterion 2: MEDIAN-of-converged (B+C improved) ───")
        print(f"    Median: kp={med_kp.round(4)}, st={med_st:.4f}")
        print(f"    Err:    kp({kp_err_med.round(2)}%), st({st_err_med:+.2f}%)")
        print(f"    Verdict: {'✅ PASS' if pass_by_med else '⚠️ borderline'}")
    else:
        pass_by_med = False
        print(f"\n  ⚠️ NO converged restarts — cannot use median selection")

    print(f"\n  Total time: {t_C:.1f}s ({t_C/NUM_RESTARTS:.1f}s/restart)")
    C_pass = pass_by_med if len(converged) > 0 else pass_by_loss
    print(f"  Mode C overall: {'✅ PASS' if C_pass else '⚠️ ATTN'}")
    print()

    # ============= Final verdict =============
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"  Mode A (close INIT, single Adam):      {'✅ PASS' if A_pass else '⚠️'}  ({t_A:.0f}s)")
    print(f"  Mode B (far INIT, single Adam):        {'✅ PASS' if B_pass else '⚠️ FAIL'}  ({t_B:.0f}s) — stuck local min")
    print(f"  Mode C (multi-restart 10 random):      {'✅ PASS' if C_pass else '⚠️'}  ({t_C:.0f}s)")
    print()
    if C_pass:
        print(f"  🎯 Multi-restart Pure Adam IS sufficient — no Optuna needed!")
        print(f"     Cost: {t_C:.0f}s vs SliceIt Optuna+Adam ~10 min")
    else:
        print(f"  ⚠️ Multi-restart Pure Adam stuck at local min.")
        print(f"     SliceIt Optuna+Adam fallback recommended for hard cases.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
