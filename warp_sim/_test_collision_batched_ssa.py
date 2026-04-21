"""Phase 2 gate: validate collision_kernel_batched_ssa.

Same 3-test pattern as Phase 1, but calls the batched kernel variant
with n_per_env = N (effectively a single env — enough to validate the
SSA + per-env indexing pattern).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pf_shape_matching_warp"))

import warp as wp
wp.init()
import numpy as np
import yaml

from warp_sim.envs.debulk_env import DebulkEnv
from pf_shape_matching_warp.sim.kernels import (
    collision_kernel_batched,
    collision_kernel_batched_ssa,
    apply_collision_delta_kernel,
)
from warp_sim.kernels.obs_reward import apply_action_kernel


RESULTS = []


def _record(name, passed, msg=""):
    tag = "✓" if passed else "✗"
    print(f"  [{tag}] {name}" + (f"  — {msg}" if msg else ""), flush=True)
    RESULTS.append((name, passed, msg))


def _setup_env():
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config",
                                            "debulk_cap_armA.yaml")))
    cfg['obs_mode'] = 'baseline'
    env = DebulkEnv(cfg)
    env.reset()
    for _ in range(3):
        env.step(np.array([0.5, -0.5, 0.5], dtype=np.float32))
    env.solver.knife_pos = wp.array([wp.vec3(0.5, 0.25, 0.5)], dtype=wp.vec3,
                                     device=env.device, requires_grad=True)
    wp.copy(env.solver.p["pred_pos"], env.solver.p["pos"])
    return env


@wp.kernel
def _sum_disp_all_kernel(
    pred_pos: wp.array(dtype=wp.vec3),
    rest_pos: wp.array(dtype=wp.vec3),
    fixed: wp.array(dtype=wp.int32),
    out_loss: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if fixed[i] == 1:
        return
    diff = pred_pos[i] - rest_pos[i]
    dsq = wp.dot(diff, diff)
    wp.atomic_add(out_loss, 0, dsq)


def test_forward_equivalence():
    print("\n=== Test 1: batched forward numerical equivalence ===", flush=True)
    env = _setup_env()
    N = env.solver.N
    n_per_env = N  # single "env" for this test

    pred_pos_snap = wp.clone(env.solver.p["pred_pos"])
    wp.launch(collision_kernel_batched, dim=N,
              inputs=[env.solver.p["pred_pos"],
                      env.solver.p["alpha"], env.solver.p["fixed"],
                      env.solver.floor_y,
                      env.solver.knife_pos, env.solver.knife_radius, 1, n_per_env])
    wp.synchronize()
    pred_old = env.solver.p["pred_pos"].numpy().copy()

    wp.copy(env.solver.p["pred_pos"], pred_pos_snap)
    env.solver.p["collision_delta"].zero_()
    wp.launch(collision_kernel_batched_ssa, dim=N,
              inputs=[env.solver.p["pred_pos"],
                      env.solver.p["alpha"], env.solver.p["fixed"],
                      env.solver.floor_y,
                      env.solver.knife_pos, env.solver.knife_radius, 1, n_per_env,
                      env.solver.p["collision_delta"]])
    wp.launch(apply_collision_delta_kernel, dim=N,
              inputs=[env.solver.p["pred_pos"], env.solver.p["collision_delta"]])
    wp.synchronize()
    pred_new = env.solver.p["pred_pos"].numpy().copy()

    diff = np.abs(pred_old - pred_new)
    max_diff = float(diff.max())
    _record("max |pred_old - pred_new| < 1e-5",
            max_diff < 1e-5,
            f"max={max_diff:.3e}")


def test_gradient_nonzero():
    print("\n=== Test 2: batched knife_pos.grad non-zero ===", flush=True)
    env = _setup_env()
    N = env.solver.N
    n_per_env = N

    wp.copy(env.solver.p["pred_pos"], env.solver.p["pos"])
    env.solver.p["collision_delta"].zero_()

    action_leaf = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3,
                           device=env.device, requires_grad=True)

    with wp.Tape() as tape:
        new_kp = wp.zeros(1, dtype=wp.vec3, device=env.device, requires_grad=True)
        wp.launch(apply_action_kernel, dim=1,
                  inputs=[env.solver.knife_pos, action_leaf, env.action_scale],
                  outputs=[new_kp])
        wp.launch(collision_kernel_batched_ssa, dim=N,
                  inputs=[env.solver.p["pred_pos"],
                          env.solver.p["alpha"], env.solver.p["fixed"],
                          env.solver.floor_y,
                          new_kp, env.solver.knife_radius, 1, n_per_env,
                          env.solver.p["collision_delta"]])
        wp.launch(apply_collision_delta_kernel, dim=N,
                  inputs=[env.solver.p["pred_pos"], env.solver.p["collision_delta"]])
        loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)
        wp.launch(_sum_disp_all_kernel, dim=N,
                  inputs=[env.solver.p["pred_pos"], env.rest_pos,
                          env.solver.p["fixed"], loss_buf])
    tape.backward(loss=loss_buf)

    g_new_kp = new_kp.grad.numpy() if new_kp.grad is not None else np.zeros(3)
    norm_new_kp = float(np.linalg.norm(g_new_kp))
    norm_action = float(np.linalg.norm(action_leaf.grad.numpy())) if action_leaf.grad is not None else 0.0

    print(f"  loss value: {float(loss_buf.numpy()[0]):.6e}", flush=True)
    print(f"  new_kp.grad: {g_new_kp}  |g|={norm_new_kp:.4e}", flush=True)
    _record("knife_pos gradient > 1e-6", norm_new_kp > 1e-6, f"|g|={norm_new_kp:.4e}")
    _record("action gradient > 1e-6", norm_action > 1e-6, f"|g|={norm_action:.4e}")


def test_analytic_vs_fd():
    print("\n=== Test 3: batched analytic vs FD cosine ===", flush=True)
    env = _setup_env()
    N = env.solver.N
    n_per_env = N

    # Analytic
    wp.copy(env.solver.p["pred_pos"], env.solver.p["pos"])
    env.solver.p["collision_delta"].zero_()
    action_leaf = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3,
                           device=env.device, requires_grad=True)
    with wp.Tape() as tape:
        new_kp = wp.zeros(1, dtype=wp.vec3, device=env.device, requires_grad=True)
        wp.launch(apply_action_kernel, dim=1,
                  inputs=[env.solver.knife_pos, action_leaf, env.action_scale],
                  outputs=[new_kp])
        wp.launch(collision_kernel_batched_ssa, dim=N,
                  inputs=[env.solver.p["pred_pos"], env.solver.p["alpha"],
                          env.solver.p["fixed"], env.solver.floor_y,
                          new_kp, env.solver.knife_radius, 1, n_per_env,
                          env.solver.p["collision_delta"]])
        wp.launch(apply_collision_delta_kernel, dim=N,
                  inputs=[env.solver.p["pred_pos"], env.solver.p["collision_delta"]])
        loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)
        wp.launch(_sum_disp_all_kernel, dim=N,
                  inputs=[env.solver.p["pred_pos"], env.rest_pos,
                          env.solver.p["fixed"], loss_buf])
    tape.backward(loss=loss_buf)
    g_a = action_leaf.grad.numpy().reshape(3).copy()

    # FD
    eps = 0.01
    pos_snap = wp.clone(env.solver.p["pos"])
    knife_snap = wp.clone(env.solver.knife_pos)
    g_f = np.zeros(3, dtype=np.float32)
    for i in range(3):
        vals = []
        for sign in (+1.0, -1.0):
            wp.copy(env.solver.p["pred_pos"], pos_snap)
            env.solver.p["collision_delta"].zero_()
            act = np.zeros(3, dtype=np.float32)
            act[i] = sign * eps
            action_wp = wp.array([wp.vec3(act[0], act[1], act[2])], dtype=wp.vec3,
                                  device=env.device, requires_grad=False)
            new_kp_fd = wp.zeros(1, dtype=wp.vec3, device=env.device, requires_grad=False)
            wp.launch(apply_action_kernel, dim=1,
                      inputs=[knife_snap, action_wp, env.action_scale],
                      outputs=[new_kp_fd])
            wp.launch(collision_kernel_batched_ssa, dim=N,
                      inputs=[env.solver.p["pred_pos"], env.solver.p["alpha"],
                              env.solver.p["fixed"], env.solver.floor_y,
                              new_kp_fd, env.solver.knife_radius, 1, n_per_env,
                              env.solver.p["collision_delta"]])
            wp.launch(apply_collision_delta_kernel, dim=N,
                      inputs=[env.solver.p["pred_pos"], env.solver.p["collision_delta"]])
            loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device)
            wp.launch(_sum_disp_all_kernel, dim=N,
                      inputs=[env.solver.p["pred_pos"], env.rest_pos,
                              env.solver.p["fixed"], loss_buf])
            wp.synchronize()
            vals.append(float(loss_buf.numpy()[0]))
        g_f[i] = (vals[0] - vals[1]) / (2.0 * eps)

    nA = float(np.linalg.norm(g_a))
    nF = float(np.linalg.norm(g_f))
    print(f"  g_analytic: {g_a}  |g|={nA:.4e}", flush=True)
    print(f"  g_fd:       {g_f}  |g|={nF:.4e}", flush=True)
    if nA > 1e-10 and nF > 1e-10:
        cos = float(np.dot(g_a, g_f) / (nA * nF))
        _record(f"cosine > 0.7  (got {cos:+.3f})", cos > 0.7)
        _record(f"magnitude ratio ∈ [0.1, 10]  (got {nA/nF:.2f})",
                0.1 < nA/nF < 10)
    else:
        _record("non-zero grads for cosine", False,
                f"|g_a|={nA:.2e}, |g_f|={nF:.2e}")


def main():
    wp.config.verify_autograd_array_access = True
    test_forward_equivalence()
    test_gradient_nonzero()
    test_analytic_vs_fd()
    print("\n═══════ SUMMARY ═══════")
    total = len(RESULTS); passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"  {passed}/{total} passed")
    if failed:
        for n in failed: print(f"    ✗ {n}")
        sys.exit(1)
    print(f"  ✓ Phase 2 gate passed — batched SSA is gradient-safe.")


if __name__ == "__main__":
    main()
