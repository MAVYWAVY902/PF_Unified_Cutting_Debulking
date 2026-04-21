"""Phase 1 gate: validate collision_kernel_ssa as a drop-in gradient-safe replacement.

Three tests, all must pass before Phase 2:
  Test 1 — Forward numerical equivalence: new kernel (via delta + apply) produces
           the same pred_pos as old kernel within 1e-5.
  Test 2 — Non-zero knife_pos.grad after tape.backward (the fix actually fixes).
  Test 3 — L2 analytic gradient matches finite-difference in direction (cosine > 0.7).
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
    collision_kernel,
    collision_kernel_ssa,
    apply_collision_delta_kernel,
)
from warp_sim.kernels.obs_reward import apply_action_kernel


RESULTS = []


def _record(name, passed, msg=""):
    tag = "✓" if passed else "✗"
    print(f"  [{tag}] {name}" + (f"  — {msg}" if msg else ""), flush=True)
    RESULTS.append((name, passed, msg))


def _setup_env():
    """Build env, push knife slightly into mesh so collision fires on multiple particles."""
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config",
                                            "debulk_cap_armA.yaml")))
    cfg['obs_mode'] = 'baseline'
    env = DebulkEnv(cfg)
    env.reset()
    for _ in range(3):
        env.step(np.array([0.5, -0.5, 0.5], dtype=np.float32))
    # Move knife to interior where many particles will be inside sphere
    env.solver.knife_pos = wp.array([wp.vec3(0.5, 0.25, 0.5)], dtype=wp.vec3,
                                     device=env.device, requires_grad=True)
    # Reset pred_pos to pos (fresh state for collision tests)
    wp.copy(env.solver.p["pred_pos"], env.solver.p["pos"])
    return env


# ============================================================
# Test 1: Forward numerical equivalence
# ============================================================

def test_forward_equivalence():
    print("\n=== Test 1: Forward numerical equivalence ===", flush=True)
    env = _setup_env()
    N = env.solver.N

    # --- Baseline: run OLD collision_kernel ---
    pred_pos_snap = wp.clone(env.solver.p["pred_pos"])
    wp.launch(collision_kernel, dim=N,
              inputs=[env.solver.p["pred_pos"],
                      env.solver.p["alpha"], env.solver.p["fixed"],
                      env.solver.floor_y,
                      env.solver.knife_pos, env.solver.knife_radius, 1])
    wp.synchronize()
    pred_old = env.solver.p["pred_pos"].numpy().copy()

    # --- Restore state and run NEW SSA + apply_delta ---
    wp.copy(env.solver.p["pred_pos"], pred_pos_snap)
    env.solver.p["collision_delta"].zero_()
    wp.launch(collision_kernel_ssa, dim=N,
              inputs=[env.solver.p["pred_pos"],
                      env.solver.p["alpha"], env.solver.p["fixed"],
                      env.solver.floor_y,
                      env.solver.knife_pos, env.solver.knife_radius, 1,
                      env.solver.p["collision_delta"]])
    wp.launch(apply_collision_delta_kernel, dim=N,
              inputs=[env.solver.p["pred_pos"], env.solver.p["collision_delta"]])
    wp.synchronize()
    pred_new = env.solver.p["pred_pos"].numpy().copy()

    # Compare
    diff = np.abs(pred_old - pred_new)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    n_different = int((diff.max(axis=1) > 1e-5).sum())
    _record(
        f"max |pred_old - pred_new| < 1e-5",
        max_diff < 1e-5,
        f"max={max_diff:.3e}, mean={mean_diff:.3e}, {n_different}/{N} particles differ",
    )


# ============================================================
# Test 2: Non-zero gradient
# ============================================================

def test_gradient_nonzero():
    print("\n=== Test 2: knife_pos.grad non-zero after tape.backward ===", flush=True)
    env = _setup_env()
    N = env.solver.N

    # Fresh pred_pos = pos
    wp.copy(env.solver.p["pred_pos"], env.solver.p["pos"])
    env.solver.p["collision_delta"].zero_()

    # Count particles inside knife sphere
    kp_np = env.solver.knife_pos.numpy().reshape(3)
    pos_np = env.solver.p["pos"].numpy()
    inside = int((np.linalg.norm(pos_np - kp_np, axis=1) < env.solver.knife_radius).sum())
    print(f"  setup: knife at {kp_np}, {inside} particles inside sphere (radius={env.solver.knife_radius})",
          flush=True)

    action_leaf = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3,
                           device=env.device, requires_grad=True)

    with wp.Tape() as tape:
        # apply_action → new_kp
        new_kp = wp.zeros(1, dtype=wp.vec3, device=env.device, requires_grad=True)
        wp.launch(apply_action_kernel, dim=1,
                  inputs=[env.solver.knife_pos, action_leaf, env.action_scale],
                  outputs=[new_kp])
        # Collision into delta
        wp.launch(collision_kernel_ssa, dim=N,
                  inputs=[env.solver.p["pred_pos"],
                          env.solver.p["alpha"], env.solver.p["fixed"],
                          env.solver.floor_y,
                          new_kp, env.solver.knife_radius, 1,
                          env.solver.p["collision_delta"]])
        # Apply delta → pred_pos updated
        wp.launch(apply_collision_delta_kernel, dim=N,
                  inputs=[env.solver.p["pred_pos"], env.solver.p["collision_delta"]])
        # Loss: sum of squared displacement of boundary particles
        loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)
        wp.launch(_sum_disp_all_kernel, dim=N,
                  inputs=[env.solver.p["pred_pos"], env.rest_pos,
                          env.solver.p["fixed"], loss_buf])
    tape.backward(loss=loss_buf)

    g_new_kp = new_kp.grad.numpy() if new_kp.grad is not None else np.zeros(3)
    g_action = action_leaf.grad.numpy() if action_leaf.grad is not None else np.zeros(3)
    norm_new_kp = float(np.linalg.norm(g_new_kp))
    norm_action = float(np.linalg.norm(g_action))

    print(f"  loss value: {float(loss_buf.numpy()[0]):.6e}", flush=True)
    print(f"  new_kp.grad: {g_new_kp}  |g|={norm_new_kp:.4e}", flush=True)
    print(f"  action.grad: {g_action}  |g|={norm_action:.4e}", flush=True)
    _record("knife_pos gradient norm > 1e-6", norm_new_kp > 1e-6,
            f"|g|={norm_new_kp:.4e}")
    _record("action gradient norm > 1e-6", norm_action > 1e-6,
            f"|g|={norm_action:.4e}")


@wp.kernel
def _sum_disp_all_kernel(
    pred_pos: wp.array(dtype=wp.vec3),
    rest_pos: wp.array(dtype=wp.vec3),
    fixed: wp.array(dtype=wp.int32),
    out_loss: wp.array(dtype=wp.float32),
):
    """Sum |pred_pos - rest_pos|^2 over ALL non-fixed particles.
    Knife collision affects particles inside its sphere (which are tumor/interior,
    not boundary) — so loss must cover ALL particles to measure knife's effect.
    """
    i = wp.tid()
    if fixed[i] == 1:
        return
    diff = pred_pos[i] - rest_pos[i]
    dsq = wp.dot(diff, diff)
    wp.atomic_add(out_loss, 0, dsq)


# ============================================================
# Test 3: Analytic vs FD cosine
# ============================================================

def _compute_analytic_grad(env):
    """Single-shot analytic gradient of boundary-displacement loss w.r.t. action."""
    N = env.solver.N
    wp.copy(env.solver.p["pred_pos"], env.solver.p["pos"])
    env.solver.p["collision_delta"].zero_()
    action_leaf = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3,
                           device=env.device, requires_grad=True)
    with wp.Tape() as tape:
        new_kp = wp.zeros(1, dtype=wp.vec3, device=env.device, requires_grad=True)
        wp.launch(apply_action_kernel, dim=1,
                  inputs=[env.solver.knife_pos, action_leaf, env.action_scale],
                  outputs=[new_kp])
        wp.launch(collision_kernel_ssa, dim=N,
                  inputs=[env.solver.p["pred_pos"], env.solver.p["alpha"],
                          env.solver.p["fixed"], env.solver.floor_y,
                          new_kp, env.solver.knife_radius, 1,
                          env.solver.p["collision_delta"]])
        wp.launch(apply_collision_delta_kernel, dim=N,
                  inputs=[env.solver.p["pred_pos"], env.solver.p["collision_delta"]])
        loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)
        wp.launch(_sum_disp_all_kernel, dim=N,
                  inputs=[env.solver.p["pred_pos"], env.rest_pos,
                          env.solver.p["fixed"], loss_buf])
    tape.backward(loss=loss_buf)
    g = action_leaf.grad.numpy().reshape(3).copy()
    return g


def _compute_fd_grad(env, eps=0.01):
    """Central finite-difference gradient of same loss."""
    N = env.solver.N
    pos_snap = wp.clone(env.solver.p["pos"])
    pred_snap = wp.clone(env.solver.p["pred_pos"])
    knife_snap = wp.clone(env.solver.knife_pos)

    g = np.zeros(3, dtype=np.float32)
    for i in range(3):
        vals = []
        for sign in (+1.0, -1.0):
            # Restore state
            wp.copy(env.solver.p["pred_pos"], pos_snap)
            env.solver.p["collision_delta"].zero_()
            # Apply perturbed action (no tape needed)
            act = np.zeros(3, dtype=np.float32)
            act[i] = sign * eps
            action_wp = wp.array([wp.vec3(act[0], act[1], act[2])], dtype=wp.vec3,
                                  device=env.device, requires_grad=False)
            new_kp = wp.zeros(1, dtype=wp.vec3, device=env.device, requires_grad=False)
            wp.launch(apply_action_kernel, dim=1,
                      inputs=[knife_snap, action_wp, env.action_scale],
                      outputs=[new_kp])
            wp.launch(collision_kernel_ssa, dim=N,
                      inputs=[env.solver.p["pred_pos"], env.solver.p["alpha"],
                              env.solver.p["fixed"], env.solver.floor_y,
                              new_kp, env.solver.knife_radius, 1,
                              env.solver.p["collision_delta"]])
            wp.launch(apply_collision_delta_kernel, dim=N,
                      inputs=[env.solver.p["pred_pos"], env.solver.p["collision_delta"]])
            loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device)
            wp.launch(_sum_disp_all_kernel, dim=N,
                  inputs=[env.solver.p["pred_pos"], env.rest_pos,
                          env.solver.p["fixed"], loss_buf])
            wp.synchronize()
            vals.append(float(loss_buf.numpy()[0]))
        g[i] = (vals[0] - vals[1]) / (2.0 * eps)

    wp.copy(env.solver.p["pos"], pos_snap)
    wp.copy(env.solver.p["pred_pos"], pred_snap)
    return g


def test_analytic_vs_fd():
    print("\n=== Test 3: Analytic vs finite-difference cosine similarity ===", flush=True)
    env = _setup_env()
    g_a = _compute_analytic_grad(env)
    g_f = _compute_fd_grad(env, eps=0.01)
    nA = float(np.linalg.norm(g_a))
    nF = float(np.linalg.norm(g_f))
    print(f"  g_analytic: {g_a}  |g|={nA:.4e}", flush=True)
    print(f"  g_fd:       {g_f}  |g|={nF:.4e}", flush=True)
    if nA > 1e-10 and nF > 1e-10:
        cos = float(np.dot(g_a, g_f) / (nA * nF))
        _record(f"cosine(analytic, FD) > 0.7  (got {cos:+.3f})", cos > 0.7)
        # Also check magnitude ratio (sanity)
        ratio = nA / nF if nF > 0 else float('inf')
        _record(f"|g_analytic| / |g_fd| ∈ [0.1, 10]  (got {ratio:.2f})",
                0.1 < ratio < 10)
    else:
        _record("cosine computable (both grads non-zero)", False,
                f"|g_a|={nA:.2e}, |g_f|={nF:.2e}")


# ============================================================
# Main
# ============================================================

def main():
    # Enable Warp's autograd safety check
    wp.config.verify_autograd_array_access = True

    test_forward_equivalence()
    test_gradient_nonzero()
    test_analytic_vs_fd()

    print(f"\n═══════ SUMMARY ═══════", flush=True)
    total = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"  {passed}/{total} passed", flush=True)
    if failed:
        print(f"  FAILED:")
        for n in failed:
            print(f"    ✗ {n}")
        sys.exit(1)
    print(f"  ✓ Phase 1 gate passed — collision_kernel_ssa is gradient-safe.", flush=True)


if __name__ == "__main__":
    main()
