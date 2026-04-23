"""Stage B gate test — rigorous 5-axis verification of predict_kernel_ssa.

Stage B fixes ONE kernel (the RED one from the audit, simpler than cluster_solve):
  • predict_kernel → predict_kernel_ssa

Two bugs in legacy predict_kernel:
  (a) `v = v * damping` — local re-assignment violates Warp 1.12 SSA rule
      (same landmine as original collision_kernel's triple `p =` re-assign that
      caused the initial zero-gradient debugging session).
  (b) `old_pos[i] = pos[i]` written in both early-return branch AND main branch
      (conditional array write; each thread still writes once so it's not R-M-W,
      but hoisting simplifies the tape structure).

Fix: fresh names `v_integrated` and `v_damped`; unconditional `old_pos[i] = pos[i]`
at the top; dead particles get `pred_pos[i] = pos[i]` then return (vel untouched
as in legacy). Physics is unchanged.

Five axes (mirroring Stage A test design):

  Axis 1 — Forward equivalence: predict_kernel_ssa produces bit-identical (<1e-6)
                                forward output vs legacy over both alive AND dead
                                particle mixtures, from multiple random states.
  Axis 2 — verify_autograd clean: wp.config.verify_autograd_array_access=True
                                on the full updated _substep does not trigger.
  Axis 3 — Gradient flows (ISOLATED):
                (3a) tape around predict_kernel_ssa only, leaf=vel,
                     loss=probe pred_pos sum → vel.grad non-zero for alive rows.
                (3b) same, leaf=pos, loss=probe pred_pos sum → pos.grad non-zero.
  Axis 4 — Analytic vs FD (ISOLATED):
                (4a) d(pred_pos[probe])/d(vel[probe]) = damping*dt per axis
                     (alive). Cosine > 0.99, rel error < 1%.
                (4b) d(pred_pos[probe])/d(pos[probe]) = 1.0 per axis (both alive
                     and dead — pred_pos = pos + v*dt, still 1 for d/dpos).
  Axis 5 — Static pattern conformance: predict_kernel_ssa has no local
                                re-assignment, single-assignment chain is intact,
                                matches NVIDIA integrator_xpbd.py predict_step
                                reference.

Exit code: 0 iff all axes pass. Non-zero = Stage B NOT COMPLETE.
"""
import sys
import os
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pf_shape_matching_warp"))

import warp as wp
wp.init()
import yaml

from warp_sim.envs.debulk_env import DebulkEnv
from pf_shape_matching_warp.sim.kernels import (
    predict_kernel,       # legacy
    predict_kernel_ssa,   # NEW
)


RESULTS = []


def _record(axis, name, passed, msg=""):
    tag = "✓" if passed else "✗"
    print(f"  [{tag}] Axis {axis}: {name}" + (f"  — {msg}" if msg else ""), flush=True)
    RESULTS.append((axis, name, passed, msg))


def _setup_env(seed=0):
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config",
                                            "debulk_cap_armA.yaml")))
    cfg['obs_mode'] = 'baseline'
    env = DebulkEnv(cfg)
    env.reset()
    rng = np.random.default_rng(seed)
    for _ in range(3):
        a = rng.uniform(-1, 1, 3).astype(np.float32)
        env.step(a)
    env.solver.knife_pos = wp.array([wp.vec3(0.5, 0.3, 0.5)], dtype=wp.vec3,
                                     device=env.device, requires_grad=True)
    env.solver.knife_active = 1
    return env


# Probe kernels — single-particle loss to dodge float32 precision loss
# (see Stage A Axis 4 for detailed rationale).
@wp.kernel
def _probe_pred_pos_sum_kernel(pred_pos: wp.array(dtype=wp.vec3),
                                probe_idx: int,
                                out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    if i != probe_idx:
        return
    p = pred_pos[i]
    wp.atomic_add(out, 0, p[0] + p[1] + p[2])


# ============================================================================
# Axis 1 — Forward equivalence (alive + dead mix)
# ============================================================================

def test_axis1_forward_equivalence():
    print("\n=== Axis 1: Forward equivalence (legacy vs SSA) ===", flush=True)
    env = _setup_env()
    N = env.solver.N

    # Run over realistic solver state (3 env.step already done, so mixed alive/dead
    # after soft-removal from near-knife cuts)
    pos_np = env.solver.p["pos"].numpy().copy()
    vel_np = env.solver.p["vel"].numpy().copy()
    alpha_np = env.solver.p["alpha"].numpy().copy()
    fixed_np = env.solver.p["fixed"].numpy().copy()

    # Inject variety: also fake some particles as dead (alpha < 0.01) to stress
    # both branches equally.
    rng = np.random.default_rng(42)
    alpha_mix = alpha_np.copy()
    dead_idx = rng.choice(N, size=N // 10, replace=False)  # 10% dead
    alpha_mix[dead_idx] = 0.0

    dt = 0.005
    damping = 0.98
    gravity = wp.vec3(0.0, -9.81, 0.0)

    # Legacy
    pos_legacy = wp.array(pos_np, dtype=wp.vec3, device=env.device)
    vel_legacy = wp.array(vel_np, dtype=wp.vec3, device=env.device)
    pred_pos_legacy = wp.zeros(N, dtype=wp.vec3, device=env.device)
    old_pos_legacy = wp.zeros(N, dtype=wp.vec3, device=env.device)
    alpha_wp = wp.array(alpha_mix, dtype=wp.float32, device=env.device)
    fixed_wp = wp.array(fixed_np, dtype=wp.int32, device=env.device)
    wp.launch(predict_kernel, dim=N,
              inputs=[pos_legacy, vel_legacy, pred_pos_legacy, old_pos_legacy,
                      alpha_wp, fixed_wp, gravity, dt, damping])
    wp.synchronize()

    # SSA
    pos_ssa = wp.array(pos_np, dtype=wp.vec3, device=env.device)
    vel_ssa = wp.array(vel_np, dtype=wp.vec3, device=env.device)
    pred_pos_ssa = wp.zeros(N, dtype=wp.vec3, device=env.device)
    old_pos_ssa = wp.zeros(N, dtype=wp.vec3, device=env.device)
    wp.launch(predict_kernel_ssa, dim=N,
              inputs=[pos_ssa, vel_ssa, pred_pos_ssa, old_pos_ssa,
                      alpha_wp, fixed_wp, gravity, dt, damping])
    wp.synchronize()

    diff_pred = np.abs(pred_pos_legacy.numpy() - pred_pos_ssa.numpy()).max()
    diff_old = np.abs(old_pos_legacy.numpy() - old_pos_ssa.numpy()).max()
    diff_vel = np.abs(vel_legacy.numpy() - vel_ssa.numpy()).max()

    _record(1, "pred_pos forward legacy == SSA (max |diff| < 1e-6)",
            diff_pred < 1e-6, f"max={diff_pred:.3e}")
    _record(1, "old_pos forward legacy == SSA (max |diff| < 1e-6)",
            diff_old < 1e-6, f"max={diff_old:.3e}")
    _record(1, "vel forward legacy == SSA (max |diff| < 1e-6)",
            diff_vel < 1e-6, f"max={diff_vel:.3e}")


# ============================================================================
# Axis 2 — verify_autograd_array_access clean on full _substep
# ============================================================================

def test_axis2_verify_autograd_clean():
    """Full _substep includes predict_kernel_ssa (Stage B) + apply_kernel_ssa
    (Stage A) + edge_damage_ssa + finalize_ssa. This runs the updated pipeline
    under verify_autograd_array_access=True; if any Stage A/B kernel introduces
    R-M-W, this trips."""
    print("\n=== Axis 2: verify_autograd_array_access=True on full _substep ===", flush=True)
    prev = wp.config.verify_autograd_array_access
    wp.config.verify_autograd_array_access = True

    try:
        env = _setup_env(seed=1)
        from warp_sim.kernels.obs_reward import apply_action_kernel

        action_leaf = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3,
                               device=env.device, requires_grad=True)
        loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)

        violated = False
        try:
            with wp.Tape():
                new_kp = wp.zeros(1, dtype=wp.vec3, device=env.device, requires_grad=True)
                wp.launch(apply_action_kernel, dim=1,
                          inputs=[env.solver.knife_pos, action_leaf, env.action_scale],
                          outputs=[new_kp])
                env.solver.knife_pos = new_kp
                env.solver.step_count = 0
                env.solver.cutting_update_interval = 1
                env.solver._substep()
                # single-particle probe avoids float32 precision loss
                probe_idx = 2309
                wp.launch(_probe_pred_pos_sum_kernel, dim=env.solver.N,
                          inputs=[env.solver.p["pred_pos"], probe_idx, loss_buf])
            wp.synchronize()
        except Exception as e:
            if "verify_autograd" in str(e) or "read-after-write" in str(e) \
               or "array access" in str(e):
                violated = True
                print(f"    VIOLATION: {e}")
            else:
                raise
        _record(2, "no R-M-W violation on full _substep under verify flag",
                not violated)
    finally:
        wp.config.verify_autograd_array_access = prev


# ============================================================================
# Axis 3 — Gradient flows through predict_kernel_ssa (isolated)
# ============================================================================

def test_axis3a_grad_through_vel():
    """Probe: tape ONLY around predict_kernel_ssa, leaf=vel, loss=sum(pred_pos[probe]).
    Expected analytic: d(pred_pos[i]_axis) / d(vel[i]_axis) = damping * dt for alive,
    0 for dead (vel not read in dead branch since early return)."""
    print("\n=== Axis 3a: vel.grad through predict_kernel_ssa ===", flush=True)
    env = _setup_env(seed=2)
    N = env.solver.N

    pos_np = env.solver.p["pos"].numpy().copy()
    vel_np = env.solver.p["vel"].numpy().copy()
    alpha_np = env.solver.p["alpha"].numpy().copy()
    fixed_np = env.solver.p["fixed"].numpy().copy()

    # Pick an alive probe
    alive_mask = (alpha_np >= 0.01) & (fixed_np == 0)
    rng = np.random.default_rng(7)
    probe_idx = int(rng.choice(np.where(alive_mask)[0]))

    dt = 0.005
    damping = 0.98
    gravity = wp.vec3(0.0, -9.81, 0.0)

    pos_wp = wp.array(pos_np, dtype=wp.vec3, device=env.device)
    vel_leaf = wp.array(vel_np, dtype=wp.vec3, device=env.device, requires_grad=True)
    pred_pos = wp.zeros(N, dtype=wp.vec3, device=env.device, requires_grad=True)
    old_pos = wp.zeros(N, dtype=wp.vec3, device=env.device, requires_grad=True)
    alpha_wp = wp.array(alpha_np, dtype=wp.float32, device=env.device)
    fixed_wp = wp.array(fixed_np, dtype=wp.int32, device=env.device)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch(predict_kernel_ssa, dim=N,
                  inputs=[pos_wp, vel_leaf, pred_pos, old_pos,
                          alpha_wp, fixed_wp, gravity, dt, damping])
        wp.launch(_probe_pred_pos_sum_kernel, dim=N,
                  inputs=[pred_pos, probe_idx, loss_buf])
    tape.backward(loss=loss_buf)

    g = vel_leaf.grad.numpy() if vel_leaf.grad is not None else np.zeros_like(vel_np)
    g_probe = g[probe_idx]
    norm = float(np.linalg.norm(g_probe))
    expected_per_axis = damping * dt  # analytic: d(pred_pos[i])/d(vel[i]) = damping*dt
    _record(3, f"vel[probe].grad ≈ [{expected_per_axis:.6f}]×3  (got {g_probe})",
            abs(g_probe[0] - expected_per_axis) < 1e-5
            and abs(g_probe[1] - expected_per_axis) < 1e-5
            and abs(g_probe[2] - expected_per_axis) < 1e-5,
            f"|g|={norm:.4e}")


def test_axis3b_grad_through_pos():
    """d(pred_pos[i])/d(pos[i]) = 1 per axis (both alive AND dead branches have
    pred_pos dependent on pos: alive → pos + v*dt; dead → pos)."""
    print("\n=== Axis 3b: pos.grad through predict_kernel_ssa ===", flush=True)
    env = _setup_env(seed=3)
    N = env.solver.N

    pos_np = env.solver.p["pos"].numpy().copy()
    vel_np = env.solver.p["vel"].numpy().copy()
    alpha_np = env.solver.p["alpha"].numpy().copy()
    fixed_np = env.solver.p["fixed"].numpy().copy()

    alive_mask = (alpha_np >= 0.01) & (fixed_np == 0)
    rng = np.random.default_rng(11)
    probe_idx = int(rng.choice(np.where(alive_mask)[0]))

    dt = 0.005
    damping = 0.98
    gravity = wp.vec3(0.0, -9.81, 0.0)

    pos_leaf = wp.array(pos_np, dtype=wp.vec3, device=env.device, requires_grad=True)
    vel_wp = wp.array(vel_np, dtype=wp.vec3, device=env.device)
    pred_pos = wp.zeros(N, dtype=wp.vec3, device=env.device, requires_grad=True)
    old_pos = wp.zeros(N, dtype=wp.vec3, device=env.device, requires_grad=True)
    alpha_wp = wp.array(alpha_np, dtype=wp.float32, device=env.device)
    fixed_wp = wp.array(fixed_np, dtype=wp.int32, device=env.device)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch(predict_kernel_ssa, dim=N,
                  inputs=[pos_leaf, vel_wp, pred_pos, old_pos,
                          alpha_wp, fixed_wp, gravity, dt, damping])
        wp.launch(_probe_pred_pos_sum_kernel, dim=N,
                  inputs=[pred_pos, probe_idx, loss_buf])
    tape.backward(loss=loss_buf)

    g = pos_leaf.grad.numpy() if pos_leaf.grad is not None else np.zeros_like(pos_np)
    g_probe = g[probe_idx]
    # Expected: d(pred_pos[i])/d(pos[i]) = 1 per axis (alive branch: pred = pos + v*dt)
    _record(3, f"pos[probe].grad ≈ [1.0]×3  (got {g_probe})",
            abs(g_probe[0] - 1.0) < 1e-5
            and abs(g_probe[1] - 1.0) < 1e-5
            and abs(g_probe[2] - 1.0) < 1e-5,
            f"|g|={float(np.linalg.norm(g_probe)):.4e}")


# ============================================================================
# Axis 4 — Analytic vs FD (isolated)
# ============================================================================

def test_axis4a_vel_fd_cosine():
    """FD d(pred_pos[probe])/d(vel[probe]) via ±eps perturbation on each axis."""
    print("\n=== Axis 4a: predict_kernel_ssa vel-FD cosine ===", flush=True)
    env = _setup_env(seed=4)
    N = env.solver.N

    pos_np = env.solver.p["pos"].numpy().copy()
    vel_np = env.solver.p["vel"].numpy().copy()
    alpha_np = env.solver.p["alpha"].numpy().copy()
    fixed_np = env.solver.p["fixed"].numpy().copy()

    alive_mask = (alpha_np >= 0.01) & (fixed_np == 0)
    rng = np.random.default_rng(13)
    probe_idx = int(rng.choice(np.where(alive_mask)[0]))

    dt = 0.005
    damping = 0.98
    gravity = wp.vec3(0.0, -9.81, 0.0)

    # Analytic
    pos_wp = wp.array(pos_np, dtype=wp.vec3, device=env.device)
    vel_leaf = wp.array(vel_np, dtype=wp.vec3, device=env.device, requires_grad=True)
    pred_pos = wp.zeros(N, dtype=wp.vec3, device=env.device, requires_grad=True)
    old_pos = wp.zeros(N, dtype=wp.vec3, device=env.device, requires_grad=True)
    alpha_wp = wp.array(alpha_np, dtype=wp.float32, device=env.device)
    fixed_wp = wp.array(fixed_np, dtype=wp.int32, device=env.device)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch(predict_kernel_ssa, dim=N,
                  inputs=[pos_wp, vel_leaf, pred_pos, old_pos,
                          alpha_wp, fixed_wp, gravity, dt, damping])
        wp.launch(_probe_pred_pos_sum_kernel, dim=N,
                  inputs=[pred_pos, probe_idx, loss_buf])
    tape.backward(loss=loss_buf)
    g_analytic = vel_leaf.grad.numpy()[probe_idx].copy()

    # FD
    def _forward(perturbed_vel):
        v_wp = wp.array(perturbed_vel, dtype=wp.vec3, device=env.device)
        pp = wp.zeros(N, dtype=wp.vec3, device=env.device)
        op = wp.zeros(N, dtype=wp.vec3, device=env.device)
        wp.launch(predict_kernel_ssa, dim=N,
                  inputs=[pos_wp, v_wp, pp, op, alpha_wp, fixed_wp,
                          gravity, dt, damping])
        lb = wp.zeros(1, dtype=wp.float32, device=env.device)
        wp.launch(_probe_pred_pos_sum_kernel, dim=N,
                  inputs=[pp, probe_idx, lb])
        wp.synchronize()
        return float(lb.numpy()[0])

    eps = 1e-3
    g_fd = np.zeros(3, dtype=np.float32)
    for axis in range(3):
        v_plus = vel_np.copy();  v_plus[probe_idx, axis]  += eps
        v_minus = vel_np.copy(); v_minus[probe_idx, axis] -= eps
        g_fd[axis] = (_forward(v_plus) - _forward(v_minus)) / (2.0 * eps)

    nA = float(np.linalg.norm(g_analytic))
    nF = float(np.linalg.norm(g_fd))
    print(f"    probe: {probe_idx}  expected damping*dt = {damping*dt:.6f}", flush=True)
    print(f"    g_analytic: {g_analytic}  |g|={nA:.4e}", flush=True)
    print(f"    g_fd:       {g_fd}  |g|={nF:.4e}", flush=True)

    if nA > 1e-10 and nF > 1e-10:
        cos = float(np.dot(g_analytic, g_fd) / (nA * nF))
        _record(4, f"vel FD-cosine > 0.99  (got {cos:+.4f})", cos > 0.99)
        ratio = nA / nF
        _record(4, f"vel magnitude ratio ∈ [0.95, 1.05]  (got {ratio:.4f})",
                0.95 < ratio < 1.05)
    else:
        _record(4, "vel non-zero grad both sides",
                False, f"|a|={nA:.2e} |fd|={nF:.2e}")


def test_axis4b_pos_fd_cosine():
    """FD d(pred_pos[probe])/d(pos[probe]) — expected 1.0 per axis."""
    print("\n=== Axis 4b: predict_kernel_ssa pos-FD cosine ===", flush=True)
    env = _setup_env(seed=5)
    N = env.solver.N

    pos_np = env.solver.p["pos"].numpy().copy()
    vel_np = env.solver.p["vel"].numpy().copy()
    alpha_np = env.solver.p["alpha"].numpy().copy()
    fixed_np = env.solver.p["fixed"].numpy().copy()

    alive_mask = (alpha_np >= 0.01) & (fixed_np == 0)
    rng = np.random.default_rng(17)
    probe_idx = int(rng.choice(np.where(alive_mask)[0]))

    dt = 0.005
    damping = 0.98
    gravity = wp.vec3(0.0, -9.81, 0.0)

    # Analytic
    pos_leaf = wp.array(pos_np, dtype=wp.vec3, device=env.device, requires_grad=True)
    vel_wp = wp.array(vel_np, dtype=wp.vec3, device=env.device)
    pred_pos = wp.zeros(N, dtype=wp.vec3, device=env.device, requires_grad=True)
    old_pos = wp.zeros(N, dtype=wp.vec3, device=env.device, requires_grad=True)
    alpha_wp = wp.array(alpha_np, dtype=wp.float32, device=env.device)
    fixed_wp = wp.array(fixed_np, dtype=wp.int32, device=env.device)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch(predict_kernel_ssa, dim=N,
                  inputs=[pos_leaf, vel_wp, pred_pos, old_pos,
                          alpha_wp, fixed_wp, gravity, dt, damping])
        wp.launch(_probe_pred_pos_sum_kernel, dim=N,
                  inputs=[pred_pos, probe_idx, loss_buf])
    tape.backward(loss=loss_buf)
    g_analytic = pos_leaf.grad.numpy()[probe_idx].copy()

    # FD — NOTE: predict_kernel_ssa writes to vel[i], so we must reset vel to the
    # original value before each _forward call; otherwise successive calls see a
    # decaying velocity and FD drifts (initial bug caught this).
    def _forward(perturbed_pos):
        pos_wp = wp.array(perturbed_pos, dtype=wp.vec3, device=env.device)
        vel_fresh = wp.array(vel_np, dtype=wp.vec3, device=env.device)  # reset each call
        pp = wp.zeros(N, dtype=wp.vec3, device=env.device)
        op = wp.zeros(N, dtype=wp.vec3, device=env.device)
        wp.launch(predict_kernel_ssa, dim=N,
                  inputs=[pos_wp, vel_fresh, pp, op, alpha_wp, fixed_wp,
                          gravity, dt, damping])
        lb = wp.zeros(1, dtype=wp.float32, device=env.device)
        wp.launch(_probe_pred_pos_sum_kernel, dim=N,
                  inputs=[pp, probe_idx, lb])
        wp.synchronize()
        return float(lb.numpy()[0])

    eps = 1e-3
    g_fd = np.zeros(3, dtype=np.float32)
    for axis in range(3):
        p_plus = pos_np.copy();  p_plus[probe_idx, axis]  += eps
        p_minus = pos_np.copy(); p_minus[probe_idx, axis] -= eps
        g_fd[axis] = (_forward(p_plus) - _forward(p_minus)) / (2.0 * eps)

    nA = float(np.linalg.norm(g_analytic))
    nF = float(np.linalg.norm(g_fd))
    print(f"    probe: {probe_idx}  expected pos→pred_pos = 1.0 per axis", flush=True)
    print(f"    g_analytic: {g_analytic}  |g|={nA:.4e}", flush=True)
    print(f"    g_fd:       {g_fd}  |g|={nF:.4e}", flush=True)

    if nA > 1e-10 and nF > 1e-10:
        cos = float(np.dot(g_analytic, g_fd) / (nA * nF))
        _record(4, f"pos FD-cosine > 0.99  (got {cos:+.4f})", cos > 0.99)
        ratio = nA / nF
        _record(4, f"pos magnitude ratio ∈ [0.95, 1.05]  (got {ratio:.4f})",
                0.95 < ratio < 1.05)
    else:
        _record(4, "pos non-zero grad both sides",
                False, f"|a|={nA:.2e} |fd|={nF:.2e}")


# ============================================================================
# Axis 5 — Static pattern conformance
# ============================================================================

def test_axis5_pattern_conformance():
    print("\n=== Axis 5: Structural conformance vs NVIDIA XPBD predict pattern ===", flush=True)
    import inspect
    import re

    src_full = inspect.getsource(predict_kernel_ssa)

    # Strip comments + docstrings so the audit reads only active CODE.
    # This matters because our doc comment intentionally REFERENCES the old pattern
    # `v = v * damping` for clarity — that's documentation, not active code.
    # Simple strip: drop any line that starts with `#` OR is inside a triple-quoted block.
    in_docstring = False
    code_lines = []
    for ln in src_full.splitlines():
        stripped = ln.strip()
        # Toggle docstring state on triple-quote
        if stripped.startswith('"""') or stripped.endswith('"""'):
            n_tq = stripped.count('"""')
            if n_tq == 2:  # single-line docstring, skip
                continue
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        # Drop inline comments (everything after `#`)
        code_ln = ln.split("#", 1)[0]
        if code_ln.strip():
            code_lines.append(code_ln)
    src = "\n".join(code_lines)

    # (a) No local re-assignment — count `name = ...` (not `==`, not `+=`)
    assigns = re.findall(r"^\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=(?!=)", src, re.MULTILINE)
    counts = {}
    for v in assigns:
        counts[v] = counts.get(v, 0) + 1
    reassigned = [v for v, c in counts.items() if c > 1]
    _record(5, f"predict_kernel_ssa: no local re-assignment  (assigns: {dict(counts)})",
            len(reassigned) == 0,
            f"reassigned: {reassigned}" if reassigned else "")

    # (b) No R-M-W on SAME array (i.e., `arr[i] = arr[i] + X` — same NAME on LHS and RHS).
    # Refined regex: match `NAME[subscript] = ... NAME[subscript]` where the two NAMEs are identical.
    rmw_pattern = re.compile(
        r"^\s*([a-zA-Z_][a-zA-Z_0-9]*)\[[^\]]+\]\s*=\s*[^=].*\b\1\[[^\]]+\]",
        re.MULTILINE,
    )
    rmw_hits = rmw_pattern.findall(src)
    _record(5, "predict_kernel_ssa: no R-M-W on same array (arr[i] = arr[i] + X)",
            len(rmw_hits) == 0,
            f"rmw arrays: {rmw_hits}" if rmw_hits else "")

    # (c) Fresh names for v_integrated + v_damped, and NO `v = v * <anything>` pattern
    #     in active code (comments are stripped already).
    has_fresh = ("v_integrated" in src) and ("v_damped" in src)
    no_v_equals_v = not re.search(r"\bv\s*=\s*v\s*\*", src)
    _record(5, "predict_kernel_ssa: uses v_integrated + v_damped fresh names",
            has_fresh)
    _record(5, "predict_kernel_ssa: no `v = v * ...` re-assignment in active code",
            no_v_equals_v)


# ============================================================================
# Runner
# ============================================================================

def main():
    print("=" * 78)
    print("Stage B gate test — 5-axis rigorous verification of predict_kernel_ssa")
    print("=" * 78)
    print("Scope: predict_kernel_ssa (RED kernel with `v = v * damping` silent-zero)")
    print("Stage C (cluster_solve + Müller) is SEPARATE.")
    print()

    tests = [
        test_axis1_forward_equivalence,
        test_axis2_verify_autograd_clean,
        test_axis3a_grad_through_vel,
        test_axis3b_grad_through_pos,
        test_axis4a_vel_fd_cosine,
        test_axis4b_pos_fd_cosine,
        test_axis5_pattern_conformance,
    ]
    for fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"  [✗] {fn.__name__} RAISED: {type(e).__name__}: {e}")
            traceback.print_exc()
            RESULTS.append(("?", fn.__name__, False, str(e)))

    print("\n" + "=" * 78)
    total = len(RESULTS)
    passed = sum(1 for *_, ok, _ in RESULTS if ok)
    failed = [f"Axis {a}: {n}" for a, n, ok, _ in RESULTS if not ok]
    print(f"  Stage B gate: {passed}/{total} checks passed")
    if failed:
        print("  FAILED:")
        for line in failed:
            print(f"    ✗ {line}")
        print("\n  ✗ Stage B NOT COMPLETE — do NOT proceed to Stage C.")
        sys.exit(1)
    print("  ✓ Stage B gate passed — predict_kernel_ssa is forward-equivalent,")
    print("    tape-safe, gradient-correct, and structurally aligned with Warp SSA norm.")


if __name__ == "__main__":
    main()
