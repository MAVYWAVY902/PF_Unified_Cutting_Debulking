"""Phase 3 gate: validate update_edge_damage_kernel_ssa.

Three tests:
  Test 1 — Forward equivalence: new SSA + apply_delta matches OLD edge_damage update
           exactly (edges within knife range should get same damage increment).
  Test 2 — Gradient flows back from edge_damage to knife_pos (via the SSA + delta pattern).
  Test 3 — Analytic vs FD cosine > 0.7.
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
from pf_shape_matching_warp.sim.pf_damage import (
    update_edge_damage_kernel,
    update_edge_damage_kernel_ssa,
    apply_edge_damage_delta_kernel,
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
    # Knife near capsule so many edges fall in knife range
    env.solver.knife_pos = wp.array([wp.vec3(0.5, 0.3, 0.5)], dtype=wp.vec3,
                                     device=env.device, requires_grad=True)
    # Fresh pred_pos
    wp.copy(env.solver.p["pred_pos"], env.solver.p["pos"])
    return env


@wp.kernel
def _sum_edge_damage_kernel(
    edge_damage: wp.array(dtype=wp.float32),
    out_loss: wp.array(dtype=wp.float32),
):
    eid = wp.tid()
    wp.atomic_add(out_loss, 0, edge_damage[eid])


def test_forward_equivalence():
    print("\n=== Test 1: edge_damage forward equivalence ===", flush=True)
    env = _setup_env()
    num_edges = env.solver.e["num_edges"]

    # --- Old kernel ---
    ed_snap = wp.clone(env.solver.e["damage"])
    wp.launch(update_edge_damage_kernel, dim=num_edges,
              inputs=[env.solver.p["pred_pos"], env.solver.p["alpha"],
                      env.solver.e["a"], env.solver.e["b"],
                      env.solver.e["pf_rest"], env.solver.e["damage"],
                      env.solver.knife_pos, env.solver.knife_radius, env.solver.alpha_min])
    wp.synchronize()
    ed_old = env.solver.e["damage"].numpy().copy()

    # --- New SSA + delta ---
    wp.copy(env.solver.e["damage"], ed_snap)
    env.solver.e["damage_delta"].zero_()
    wp.launch(update_edge_damage_kernel_ssa, dim=num_edges,
              inputs=[env.solver.p["pred_pos"], env.solver.p["alpha"],
                      env.solver.e["a"], env.solver.e["b"],
                      env.solver.e["pf_rest"], env.solver.e["damage"],
                      env.solver.knife_pos, env.solver.knife_radius, env.solver.alpha_min,
                      env.solver.e["damage_delta"]])
    wp.launch(apply_edge_damage_delta_kernel, dim=num_edges,
              inputs=[env.solver.e["damage"], env.solver.e["damage_delta"]])
    wp.synchronize()
    ed_new = env.solver.e["damage"].numpy().copy()

    diff = np.abs(ed_old - ed_new)
    max_diff = float(diff.max())
    n_different = int((diff > 1e-6).sum())
    touched = int((ed_new > 1e-6).sum())
    _record(
        "max |ed_old - ed_new| < 1e-5",
        max_diff < 1e-5,
        f"max={max_diff:.3e}, {n_different} edges differ, {touched}/{num_edges} edges damaged",
    )


def test_gradient_nonzero():
    print("\n=== Test 2: knife_pos.grad from edge_damage SSA ===", flush=True)
    env = _setup_env()
    num_edges = env.solver.e["num_edges"]

    env.solver.e["damage"].zero_()
    env.solver.e["damage_delta"].zero_()

    action_leaf = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3,
                           device=env.device, requires_grad=True)

    with wp.Tape() as tape:
        new_kp = wp.zeros(1, dtype=wp.vec3, device=env.device, requires_grad=True)
        wp.launch(apply_action_kernel, dim=1,
                  inputs=[env.solver.knife_pos, action_leaf, env.action_scale],
                  outputs=[new_kp])
        wp.launch(update_edge_damage_kernel_ssa, dim=num_edges,
                  inputs=[env.solver.p["pred_pos"], env.solver.p["alpha"],
                          env.solver.e["a"], env.solver.e["b"],
                          env.solver.e["pf_rest"], env.solver.e["damage"],
                          new_kp, env.solver.knife_radius, env.solver.alpha_min,
                          env.solver.e["damage_delta"]])
        wp.launch(apply_edge_damage_delta_kernel, dim=num_edges,
                  inputs=[env.solver.e["damage"], env.solver.e["damage_delta"]])
        # Loss: sum of all edge damage
        loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)
        wp.launch(_sum_edge_damage_kernel, dim=num_edges,
                  inputs=[env.solver.e["damage"], loss_buf])

    tape.backward(loss=loss_buf)

    g_kp = new_kp.grad.numpy() if new_kp.grad is not None else np.zeros(3)
    norm_kp = float(np.linalg.norm(g_kp))
    g_act = action_leaf.grad.numpy() if action_leaf.grad is not None else np.zeros(3)
    norm_act = float(np.linalg.norm(g_act))

    print(f"  loss (total edge damage): {float(loss_buf.numpy()[0]):.4e}", flush=True)
    print(f"  new_kp.grad: {g_kp}  |g|={norm_kp:.4e}", flush=True)
    _record("new_kp.grad > 1e-6", norm_kp > 1e-6, f"|g|={norm_kp:.4e}")
    _record("action.grad > 1e-6", norm_act > 1e-6, f"|g|={norm_act:.4e}")


def _analytic_grad_edge(env):
    num_edges = env.solver.e["num_edges"]
    env.solver.e["damage"].zero_()
    env.solver.e["damage_delta"].zero_()
    action_leaf = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3,
                           device=env.device, requires_grad=True)
    with wp.Tape() as tape:
        new_kp = wp.zeros(1, dtype=wp.vec3, device=env.device, requires_grad=True)
        wp.launch(apply_action_kernel, dim=1,
                  inputs=[env.solver.knife_pos, action_leaf, env.action_scale],
                  outputs=[new_kp])
        wp.launch(update_edge_damage_kernel_ssa, dim=num_edges,
                  inputs=[env.solver.p["pred_pos"], env.solver.p["alpha"],
                          env.solver.e["a"], env.solver.e["b"],
                          env.solver.e["pf_rest"], env.solver.e["damage"],
                          new_kp, env.solver.knife_radius, env.solver.alpha_min,
                          env.solver.e["damage_delta"]])
        wp.launch(apply_edge_damage_delta_kernel, dim=num_edges,
                  inputs=[env.solver.e["damage"], env.solver.e["damage_delta"]])
        loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)
        wp.launch(_sum_edge_damage_kernel, dim=num_edges,
                  inputs=[env.solver.e["damage"], loss_buf])
    tape.backward(loss=loss_buf)
    return action_leaf.grad.numpy().reshape(3).copy()


def _fd_grad_edge(env, eps=0.01):
    num_edges = env.solver.e["num_edges"]
    ed_snap = wp.clone(env.solver.e["damage"])
    knife_snap = wp.clone(env.solver.knife_pos)
    g = np.zeros(3, dtype=np.float32)
    for i in range(3):
        vals = []
        for sign in (+1.0, -1.0):
            wp.copy(env.solver.e["damage"], ed_snap)
            env.solver.e["damage"].zero_()
            env.solver.e["damage_delta"].zero_()
            act = np.zeros(3, dtype=np.float32)
            act[i] = sign * eps
            action_wp = wp.array([wp.vec3(act[0], act[1], act[2])], dtype=wp.vec3,
                                  device=env.device, requires_grad=False)
            new_kp = wp.zeros(1, dtype=wp.vec3, device=env.device, requires_grad=False)
            wp.launch(apply_action_kernel, dim=1,
                      inputs=[knife_snap, action_wp, env.action_scale],
                      outputs=[new_kp])
            wp.launch(update_edge_damage_kernel_ssa, dim=num_edges,
                      inputs=[env.solver.p["pred_pos"], env.solver.p["alpha"],
                              env.solver.e["a"], env.solver.e["b"],
                              env.solver.e["pf_rest"], env.solver.e["damage"],
                              new_kp, env.solver.knife_radius, env.solver.alpha_min,
                              env.solver.e["damage_delta"]])
            wp.launch(apply_edge_damage_delta_kernel, dim=num_edges,
                      inputs=[env.solver.e["damage"], env.solver.e["damage_delta"]])
            loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device)
            wp.launch(_sum_edge_damage_kernel, dim=num_edges,
                      inputs=[env.solver.e["damage"], loss_buf])
            wp.synchronize()
            vals.append(float(loss_buf.numpy()[0]))
        g[i] = (vals[0] - vals[1]) / (2.0 * eps)
    wp.copy(env.solver.e["damage"], ed_snap)
    return g


def test_analytic_vs_fd():
    print("\n=== Test 3: edge_damage analytic vs FD cosine ===", flush=True)
    env = _setup_env()
    g_a = _analytic_grad_edge(env)
    g_f = _fd_grad_edge(env, eps=0.01)
    nA = float(np.linalg.norm(g_a)); nF = float(np.linalg.norm(g_f))
    print(f"  g_analytic: {g_a}  |g|={nA:.4e}", flush=True)
    print(f"  g_fd:       {g_f}  |g|={nF:.4e}", flush=True)
    if nA > 1e-10 and nF > 1e-10:
        cos = float(np.dot(g_a, g_f) / (nA * nF))
        _record(f"cosine > 0.7  (got {cos:+.3f})", cos > 0.7)
        _record(f"mag ratio ∈ [0.1, 10]  (got {nA/nF:.2f})", 0.1 < nA/nF < 10)
    else:
        _record("non-zero grads", False, f"|g_a|={nA:.2e}, |g_f|={nF:.2e}")


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
    print(f"  ✓ Phase 3 gate passed — edge_damage SSA is gradient-safe.")


if __name__ == "__main__":
    main()
