#!/usr/bin/env python3
"""Path 4 Falsification Test I — Jacobi loop iteration scan.

QUESTION: Within Layer A path (verified broken in Test H1), is the catastrophic
gradient from:
  (a) Initial collision + apply_collision_delta (run ONCE pre-Jacobi)
  (b) Jacobi loop's compounded cluster_solve + apply_kernel + apply_shape_delta
      + collision + apply_collision_delta
  (c) Both, with multiplicative amplification through Jacobi iters

Method: scan num_iterations ∈ {0, 1, 2, 3, 6} keeping H1 setup
(collision ON, cutting OFF, loss=mean pos.z, knife on tumor).

Predictions:
  - num_iter=0 huge → bug is pre-Jacobi (collision_kernel or apply_collision_delta)
  - num_iter=0 small, scaling with iter → Jacobi loop is amplifier; specific
    kernels in the loop (cluster_solve, apply_kernel, apply_shape_delta) are
    the issue
  - num_iter=0 small, num_iter≥1 huge → first Jacobi iter introduces bug;
    cluster_solve or apply_kernel is the source
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

wp.config.verify_autograd_array_access = False

from path4_falsification_test_H_isolate import (
    differentiable_substep_isolated,
    loss_sum_pos_z_kernel,
    setup_env,
)
from path4_poc_v1_dual_purpose import (
    make_param_dict, snap_solver_state, reset_solver_state,
)


def run_at_num_iter(num_iter: int, knife_pos_np):
    print(f"\n  ─── num_iterations = {num_iter} ────")

    env = setup_env()
    env.solver.num_iterations = num_iter
    state_snap = snap_solver_state(env)
    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)

    # Autograd (H1 setup: collision ON, cutting OFF, position loss)
    params = make_param_dict(20.0, 3.0, 0.05, 0.15, requires_grad=True,
                             device=env.device)
    knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                         device=env.device, requires_grad=True)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device,
                        requires_grad=True)
    dt_sub = env.solver.dt / float(env.solver.substeps)

    with wp.Tape() as tape:
        differentiable_substep_isolated(env.solver, params, knife_pos, dt_sub,
                                        enable_layer_a_collision=True,
                                        enable_cutting=False)
        wp.launch(loss_sum_pos_z_kernel, dim=env.solver.N,
                  inputs=[env.solver.p["pos"], interior_mask_wp,
                          interior_count, loss_buf],
                  device=env.device)
    tape.backward(loss=loss_buf)
    wp.synchronize()

    ag_loss = float(loss_buf.numpy()[0])
    ag_kp = knife_pos.grad.numpy()[0].copy()

    # FD per-axis
    fd_kp = np.zeros(3)
    for axis in range(3):
        eps = 5e-3
        for sign in (+1, -1):
            reset_solver_state(env, state_snap)
            env.solver.step_count = 0
            kp_np = list(knife_pos_np); kp_np[axis] += sign * eps
            kp = wp.array([wp.vec3(*kp_np)], dtype=wp.vec3, device=env.device)
            params_fd = make_param_dict(20.0, 3.0, 0.05, 0.15,
                                        requires_grad=False, device=env.device)
            l_buf = wp.zeros(1, dtype=wp.float32, device=env.device)
            differentiable_substep_isolated(env.solver, params_fd, kp, dt_sub,
                                            enable_layer_a_collision=True,
                                            enable_cutting=False)
            wp.launch(loss_sum_pos_z_kernel, dim=env.solver.N,
                      inputs=[env.solver.p["pos"], interior_mask_wp,
                              interior_count, l_buf],
                      device=env.device)
            wp.synchronize()
            if sign == +1:
                lp = float(l_buf.numpy()[0])
            else:
                lm = float(l_buf.numpy()[0])
        fd_kp[axis] = (lp - lm) / (2.0 * eps)

    print(f"    forward loss: {ag_loss:.4e}")
    print(f"    autograd:  ({ag_kp[0]:+.3e}, {ag_kp[1]:+.3e}, {ag_kp[2]:+.3e})")
    print(f"    FD:        ({fd_kp[0]:+.3e}, {fd_kp[1]:+.3e}, {fd_kp[2]:+.3e})")
    max_rel = max(abs(a - f) / max(abs(f), 1e-12)
                  for a, f in zip(ag_kp, fd_kp))
    print(f"    max rel_err: {max_rel*100:.2f}%")

    # Magnitude characterization
    ag_mag = float(np.linalg.norm(ag_kp))
    fd_mag = float(np.linalg.norm(fd_kp))
    ratio = ag_mag / max(fd_mag, 1e-12)
    print(f"    ‖autograd‖ / ‖FD‖ = {ag_mag:.3e} / {fd_mag:.3e} = {ratio:.2e}×")

    return dict(num_iter=num_iter, ag_kp=ag_kp, fd_kp=fd_kp,
                ag_mag=ag_mag, fd_mag=fd_mag, ratio=ratio,
                max_rel=max_rel)


def main():
    wp.init()
    print("=" * 72)
    print("  Test I — Jacobi iteration scan (Layer A only)")
    print("  Setup: H1 (collision ON, cutting OFF, loss=mean pos.z)")
    print("=" * 72)

    knife_pos_np = (0.85, 0.5, 0.12)

    results = []
    for n in (0, 1, 2, 3, 6):
        r = run_at_num_iter(n, knife_pos_np)
        results.append(r)

    print()
    print("=" * 72)
    print("  SUMMARY: gradient magnitude scaling with num_iter")
    print("=" * 72)
    print(f"  {'num_iter':>10} {'‖autograd‖':>15} {'‖FD‖':>15} {'ratio (ag/fd)':>18}  verdict")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*18}  ---")
    for r in results:
        verdict = ("✓ OK" if r["max_rel"] < 0.20 else
                   f"✗ off {r['max_rel']*100:.0f}%" if r["max_rel"] < 100 else
                   "❌ catastrophic")
        print(f"  {r['num_iter']:>10} {r['ag_mag']:>15.3e} {r['fd_mag']:>15.3e} "
              f"{r['ratio']:>18.2e}  {verdict}")

    print()
    print("INTERPRETATION:")
    r0, r1, r6 = results[0], results[1], results[-1]
    if r0["max_rel"] > 100:
        print("  → num_iter=0 ALREADY catastrophic")
        print("  → Bug is in pre-Jacobi part: collision_kernel + apply_collision_delta")
        print("  → Specific suspects:")
        print("    1. collision_kernel_ssa wp.where + 1/safe_dist backward")
        print("    2. apply_collision_delta_kernel atomic_add backward to pred_pos")
    elif r1["max_rel"] > 100 and r0["max_rel"] < 1:
        print("  → num_iter=0 OK, num_iter=1 catastrophic")
        print("  → First Jacobi iter introduces bug")
        print("  → Specific suspects: cluster_solve_kernel_ssa, apply_kernel_ssa,")
        print("    apply_shape_delta_kernel — kernels NEW in Jacobi vs initial collision")
    elif r6["ratio"] > 100 * r1["ratio"]:
        print("  → Multiplicative amplification through Jacobi loop")
        print("  → Some Jacobi kernel has biased backward that compounds")
    else:
        print("  → Mixed pattern — see ratios above for diagnosis")


if __name__ == "__main__":
    main()
