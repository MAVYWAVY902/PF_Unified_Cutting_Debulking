#!/usr/bin/env python3
"""3D bulk multi-substep sys-ID — damping/friction observable via time horizon.

Previous single-substep test:
  - stiffness ✅ (5/5 INITs converge <1%)
  - damping  ❌ (gradient too weak in single substep)
  - friction ⚠️ (3/5 INITs converge)

Hypothesis: multi-substep simulation lets velocity develop + tangential motion
accumulate → damping/friction become observable.

Test:
  N=10 substeps, loss = MSE(sim_pos at all checkpoints, target_pos at same)
  Checkpoints: substeps {2, 5, 10} (3 observations spread across time)
  Pass: damping/friction recover within 10% (vs failed single-substep)
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
from test_3d_bulk_pure_adam_sysid import diff_substep_3d_bulk
from test_layer_a_phase2b_e2e_stiffness_sysid import loss_pos_mse_kernel


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = 0.0; self.v = 0.0; self.t = 0
    def step(self, p, grad):
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*grad
        self.v = self.b2*self.v + (1-self.b2)*grad*grad
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        return p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def gen_target_trajectory(env, GT_S, GT_D, GT_F, GT_KP, dt_sub, N=10,
                            checkpoints=(2, 5, 10)):
    """Run sim with GT params for N substeps, capture pos at checkpoints.
    Returns list of (substep_idx, target_pos_array)."""
    state_snap = snap_solver_state(env)
    env.solver.step_count = 0
    knife = wp.array([wp.vec3(*GT_KP)], dtype=wp.vec3, device=env.device)
    s = wp.array([GT_S], dtype=wp.float32, device=env.device)
    d = wp.array([GT_D], dtype=wp.float32, device=env.device)
    f = wp.array([GT_F], dtype=wp.float32, device=env.device)

    targets = {}
    for substep in range(1, N+1):
        diff_substep_3d_bulk(env.solver, knife, s, d, f, dt_sub)
        if substep in checkpoints:
            targets[substep] = wp.array(env.solver.p["pos"].numpy().copy(),
                                          dtype=wp.vec3, device=env.device)
    reset_solver_state(env, state_snap)
    env.solver.step_count = 0
    return targets, state_snap


def calibrate_3d_multi(env, init, GT, target_pos_dict, state_snap, dt_sub,
                        interior_mask_wp, interior_count, num_iters=80,
                        N_substeps=10, checkpoints=(2, 5, 10)):
    """3D Adam with multi-substep horizon. Loss accumulates pos MSE at checkpoints."""
    s_curr, d_curr, f_curr = init
    opt_s = Adam(lr=0.05); opt_d = Adam(lr=0.05); opt_f = Adam(lr=0.05)
    inv_count = 1.0 / interior_count
    GT_KP = np.array([0.85, 0.5, 0.12])

    history = []
    for it in range(num_iters):
        reset_solver_state(env, state_snap)
        env.solver.step_count = 0
        knife_pos_arr = wp.array([wp.vec3(*GT_KP)], dtype=wp.vec3,
                                   device=env.device, requires_grad=False)
        s_arr = wp.array([s_curr], dtype=wp.float32, device=env.device, requires_grad=True)
        d_arr = wp.array([d_curr], dtype=wp.float32, device=env.device, requires_grad=True)
        f_arr = wp.array([f_curr], dtype=wp.float32, device=env.device, requires_grad=True)
        loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)

        with wp.Tape() as tape:
            for substep in range(1, N_substeps+1):
                diff_substep_3d_bulk(env.solver, knife_pos_arr, s_arr, d_arr, f_arr, dt_sub)
                if substep in checkpoints:
                    # Accumulate loss at this checkpoint
                    wp.launch(loss_pos_mse_kernel, dim=env.solver.N,
                              inputs=[env.solver.p["pos"], target_pos_dict[substep],
                                      interior_mask_wp, inv_count, loss_buf],
                              device=env.device)
        tape.backward(loss=loss_buf)
        wp.synchronize()

        loss_val = float(loss_buf.numpy()[0])
        gs = float(s_arr.grad.numpy()[0])
        gd = float(d_arr.grad.numpy()[0])
        gf = float(f_arr.grad.numpy()[0])
        history.append((s_curr, d_curr, f_curr, loss_val))

        s_curr = float(np.clip(opt_s.step(s_curr, gs), 0.05, 0.99))
        d_curr = float(np.clip(opt_d.step(d_curr, gd), 0.01, 0.99))
        f_curr = float(np.clip(opt_f.step(f_curr, gf), 0.05, 1.5))

    return dict(stiff=s_curr, damp=d_curr, fric=f_curr,
                final_loss=history[-1][3], init_loss=history[0][3])


def main():
    wp.init()
    print("=" * 78)
    print("  3D BULK MULTI-SUBSTEP SYS-ID — Joint Adam, time-horizon observation")
    print("=" * 78)

    env = setup_env()
    env.solver.num_iterations = 6
    env.solver.substeps = 1
    env.solver.use_split_cluster_solve = True
    env.solver.use_ssa_soft_removal = True
    env.solver.cutting_update_interval = 999999
    env.solver.knife_active = 1

    GT_S, GT_D, GT_F = 0.85, 0.10, 0.95
    GT_KP = np.array([0.85, 0.5, 0.12])
    N_SUBSTEPS = 10
    CHECKPOINTS = (2, 5, 10)

    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)
    dt_sub = env.solver.dt / float(env.solver.substeps)

    print(f"  GT: stiff={GT_S}, damping={GT_D}, friction={GT_F}")
    print(f"  Horizon: N={N_SUBSTEPS} substeps, checkpoints={CHECKPOINTS}")
    print()

    print("  Generating target trajectory at GT...")
    target_pos_dict, state_snap = gen_target_trajectory(
        env, GT_S, GT_D, GT_F, GT_KP, dt_sub,
        N=N_SUBSTEPS, checkpoints=CHECKPOINTS)
    print(f"  Captured {len(target_pos_dict)} checkpoint snapshots.")
    print()

    INITS = [
        (0.50, 0.50, 0.50),
        (0.30, 0.30, 0.70),
        (0.70, 0.05, 0.80),
        (0.95, 0.20, 1.20),
        (0.40, 0.50, 0.50),
    ]
    print(f"  {'INIT':>30} {'final_s':>9} {'final_d':>9} {'final_f':>9} "
          f"{'s_err':>7} {'d_err':>9} {'f_err':>7} {'time':>7} {'verdict':>8}")
    results = []
    for init in INITS:
        t0 = time.time()
        try:
            res = calibrate_3d_multi(env, init, (GT_S, GT_D, GT_F),
                                       target_pos_dict, state_snap, dt_sub,
                                       interior_mask_wp, interior_count,
                                       num_iters=80,
                                       N_substeps=N_SUBSTEPS,
                                       checkpoints=CHECKPOINTS)
            t = time.time() - t0
            s_err = (res["stiff"] - GT_S) / GT_S * 100
            d_err = (res["damp"] - GT_D) / GT_D * 100
            f_err = (res["fric"] - GT_F) / GT_F * 100
            max_err = max(abs(s_err), abs(d_err), abs(f_err))
            verdict = "✅ <2%" if max_err < 2 else (
                "✅ <5%" if max_err < 5 else (
                "⚠️ <15%" if max_err < 15 else "❌"))
            print(f"  {str(init):>30} {res['stiff']:>9.4f} {res['damp']:>9.4f} {res['fric']:>9.4f} "
                  f"{s_err:>+6.1f}% {d_err:>+8.1f}% {f_err:>+6.1f}% {t:>6.1f}s {verdict:>8}")
            results.append((init, res, s_err, d_err, f_err, t))
        except Exception as e:
            print(f"  {str(init):>30} ERROR: {type(e).__name__}: {e}")
            results.append(None)

    print()
    print("=" * 78)
    print("  VERDICT (3D bulk multi-substep)")
    print("=" * 78)
    valid = [r for r in results if r is not None]
    all_pass = all(max(abs(s),abs(d),abs(f)) < 5.0 for _,_,s,d,f,_ in valid) if valid else False
    damping_pass = all(abs(d) < 5.0 for _,_,_,d,_,_ in valid) if valid else False
    friction_pass = all(abs(f) < 5.0 for _,_,_,_,f,_ in valid) if valid else False
    print(f"  All params <5%:    {'✅ YES' if all_pass else '⚠️ NO'}")
    print(f"  Damping <5%:       {'✅ YES' if damping_pass else '⚠️ NO'}  (was 0/5 in single-substep)")
    print(f"  Friction <5%:      {'✅ YES' if friction_pass else '⚠️ NO'}  (was 3/5 in single-substep)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
