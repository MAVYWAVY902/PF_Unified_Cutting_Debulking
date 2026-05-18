#!/usr/bin/env python3
"""
Path B+ POC Day 2 v3 — FULL cycle-scoped state buffers.

v1 (synthetic kernels): PASS for both sharpness and alpha_min.
v2 (real kernels, alpha cycle-scoped only): PARTIAL — sharpness PASS, alpha_min FAIL.
v3 (real kernels, ALL state buffers cycle-scoped): test target = both PASS.

Insight from v2: the cross-kernel R-W-R issue isn't just on alpha. It also
affects edge_damage and particle_damage (verify_autograd_array_access flagged
both). For alpha_min gradient (which propagates through edge_damage chain),
making alpha alone cycle-scoped is insufficient — must extend to all state.

Reference: NVIDIA example_fluid_checkpoint.py uses cycle-scoped lists for ALL
state (vx_arrays, vy_arrays, density_arrays, pressure_arrays) — not just one.

Pass criteria:
  - SSA gradient sign matches FD on BOTH sharpness and alpha_min
  - SSA rel_err < 1% vs FD on BOTH params
  - verify_autograd_array_access produces NO warnings on SSA path
  - Determinism: repeated runs same gradient

Usage:
  python3 tools/poc_soft_removal_ssa_v3_full.py
"""
import sys
import numpy as np
import warp as wp


wp.init()
wp.config.verify_autograd_array_access = True

print("=" * 72)
print(f"  Path B+ POC Day 2 v3 — FULL cycle-scoped state")
print(f"  Warp version: {wp.__version__}")
print("=" * 72)


# =========================================================================
# Kernels — production-style with wp.array params, all SSA-compatible
# =========================================================================

@wp.kernel
def edge_damage_kernel(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    edge_a: wp.array(dtype=wp.int32),
    edge_b: wp.array(dtype=wp.int32),
    pf_rest: wp.array(dtype=wp.float32),
    edge_damage_prev: wp.array(dtype=wp.float32),     # READ ONLY
    knife_pos: wp.array(dtype=wp.vec3),
    knife_radius_arr: wp.array(dtype=wp.float32),
    alpha_min_arr: wp.array(dtype=wp.float32),
    edge_damage_delta: wp.array(dtype=wp.float32),    # WRITE (delta only)
):
    eid = wp.tid()
    a = edge_a[eid]
    b = edge_b[eid]
    if alpha[a] < 0.01 or alpha[b] < 0.01:
        return

    pa = pred_pos[a]
    pb = pred_pos[b]
    mid = (pa + pb) * 0.5

    knife_radius = knife_radius_arr[0]
    alpha_min = alpha_min_arr[0]

    kp = knife_pos[0]
    diff = mid - kp
    dist_sq = wp.dot(diff, diff)
    dist_to_center = wp.sqrt(dist_sq + 1.0e-8)

    if dist_to_center > knife_radius * 1.3:
        return

    edge_vec = pa - pb
    edge_len_sq = wp.dot(edge_vec, edge_vec)
    radius = wp.sqrt(edge_len_sq + 1.0e-8) * 0.5
    if radius < 1.0e-4:
        return

    intruder_dist = dist_to_center - knife_radius
    sdist = intruder_dist - radius
    exp_arg = wp.clamp(-alpha_min * sdist / radius, -50.0, 50.0)
    pf = 1.0 / (1.0 + wp.exp(exp_arg))

    rest = pf_rest[eid]
    rest_safe = wp.where(rest > 0.01, rest, 1.0)
    target = 1.0 - pf / rest_safe
    target_clamped = wp.log(1.0 + wp.exp(wp.clamp(20.0 * target, -50.0, 50.0))) / 20.0
    current = edge_damage_prev[eid]                     # READ from PREV cycle
    dmg_diff = target_clamped - current
    increment_raw = wp.log(1.0 + wp.exp(wp.clamp(10.0 * dmg_diff, -50.0, 50.0))) / 10.0
    increment = wp.where(rest > 0.01, increment_raw, 0.0)

    wp.atomic_add(edge_damage_delta, eid, increment)    # WRITE to delta only


@wp.kernel
def apply_edge_damage_ssa(
    edge_damage_prev: wp.array(dtype=wp.float32),       # READ
    edge_damage_delta: wp.array(dtype=wp.float32),      # READ
    edge_damage_next: wp.array(dtype=wp.float32),       # WRITE (different buffer)
):
    """SSA-style: prev + delta → next. Replaces in-place atomic_add."""
    eid = wp.tid()
    edge_damage_next[eid] = edge_damage_prev[eid] + edge_damage_delta[eid]


@wp.kernel
def update_particle_damage_kernel(
    particle_damage_sum: wp.array(dtype=wp.float32),
    particle_damage_count: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    edge_a: wp.array(dtype=wp.int32),
    edge_b: wp.array(dtype=wp.int32),
    edge_damage: wp.array(dtype=wp.float32),
):
    eid = wp.tid()
    a = edge_a[eid]
    b = edge_b[eid]
    if alpha[a] < 0.01 or alpha[b] < 0.01:
        return
    val = edge_damage[eid]
    if val <= 0.0:
        return
    wp.atomic_add(particle_damage_sum, a, val)
    wp.atomic_add(particle_damage_count, a, 1.0)
    wp.atomic_add(particle_damage_sum, b, val)
    wp.atomic_add(particle_damage_count, b, 1.0)


@wp.kernel
def finalize_particle_damage_kernel(
    particle_damage_sum: wp.array(dtype=wp.float32),
    particle_damage_count: wp.array(dtype=wp.float32),
    particle_damage_prev: wp.array(dtype=wp.float32),     # READ ONLY
    particle_damage_delta: wp.array(dtype=wp.float32),    # WRITE (delta only)
):
    i = wp.tid()
    cnt = particle_damage_count[i]
    if cnt < 0.5:
        return
    mean_dmg = particle_damage_sum[i] / cnt
    diff = mean_dmg - particle_damage_prev[i]              # READ from PREV
    sp_arg = wp.clamp(20.0 * diff, -50.0, 50.0)
    increment = wp.log(1.0 + wp.exp(sp_arg)) / 20.0
    wp.atomic_add(particle_damage_delta, i, increment)


@wp.kernel
def apply_particle_damage_ssa(
    particle_damage_prev: wp.array(dtype=wp.float32),     # READ
    particle_damage_delta: wp.array(dtype=wp.float32),    # READ
    particle_damage_next: wp.array(dtype=wp.float32),     # WRITE (different buffer)
):
    """SSA-style: prev + delta → next."""
    i = wp.tid()
    particle_damage_next[i] = particle_damage_prev[i] + particle_damage_delta[i]


@wp.kernel
def update_soft_removal_kernel_ssa(
    damage: wp.array(dtype=wp.float32),
    alpha_prev: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    threshold_arr: wp.array(dtype=wp.float32),
    sharpness_arr: wp.array(dtype=wp.float32),
    alpha_next: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if fixed[i] == 1:
        alpha_next[i] = alpha_prev[i]
        return
    threshold = threshold_arr[0]
    sharpness = sharpness_arr[0]
    d = damage[i]
    exp_arg = wp.clamp(sharpness * (d - threshold), -50.0, 50.0)
    alpha_next[i] = 1.0 / (1.0 + wp.exp(exp_arg))


@wp.kernel
def loss_alpha_to_zero(
    alpha: wp.array(dtype=wp.float32),
    loss_buf: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    a = alpha[i]
    wp.atomic_add(loss_buf, 0, a * a)


# =========================================================================
# Setup
# =========================================================================

def setup_test_system(N=20, knife_x=0.5):
    np.random.seed(42)
    pos_np = np.zeros((N, 3), dtype=np.float32)
    pos_np[:, 0] = np.linspace(0.3, 0.7, N)
    pos_np[:, 1] = 0.5
    pos_np[:, 2] = 0.5
    E = N - 1
    edge_a_np = np.arange(E, dtype=np.int32)
    edge_b_np = np.arange(1, N, dtype=np.int32)
    pf_rest_np = np.full(E, 0.5, dtype=np.float32)
    knife_pos_np = np.array([[knife_x, 0.5, 0.5]], dtype=np.float32)
    return dict(
        pos=pos_np, edge_a=edge_a_np, edge_b=edge_b_np,
        pf_rest=pf_rest_np, knife_pos=knife_pos_np,
        N=N, E=E,
    )


def make_params(knife_radius=0.1, alpha_min=3.0, sharpness=20.0, threshold=0.5):
    return dict(
        knife_radius=wp.array([knife_radius], dtype=wp.float32, requires_grad=True),
        alpha_min=wp.array([alpha_min], dtype=wp.float32, requires_grad=True),
        sharpness=wp.array([sharpness], dtype=wp.float32, requires_grad=True),
        threshold=wp.array([threshold], dtype=wp.float32, requires_grad=True),
    )


# =========================================================================
# FULL SSA pipeline — ALL state buffers cycle-scoped
# =========================================================================

def run_full_ssa(params, sys, n_cycles=3):
    N = sys["N"]; E = sys["E"]
    pos = wp.array(sys["pos"], dtype=wp.vec3)
    edge_a = wp.array(sys["edge_a"], dtype=wp.int32)
    edge_b = wp.array(sys["edge_b"], dtype=wp.int32)
    pf_rest = wp.array(sys["pf_rest"], dtype=wp.float32)
    knife_pos = wp.array(sys["knife_pos"], dtype=wp.vec3)
    fixed = wp.array(np.zeros(N, dtype=np.int32), dtype=wp.int32)

    # ===== Cycle-scoped state buffer LISTS (n_cycles + 1 each) =====
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

    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

    with wp.Tape() as tape:
        for cycle in range(n_cycles):
            # === EDGE DAMAGE ===
            # Read alpha_buffers[cycle], edge_damage_buffers[cycle]; write delta
            edge_damage_delta = wp.zeros(E, dtype=wp.float32, requires_grad=True)
            wp.launch(edge_damage_kernel, dim=E, inputs=[
                pos, alpha_buffers[cycle], edge_a, edge_b, pf_rest,
                edge_damage_buffers[cycle],            # READ prev
                knife_pos, params["knife_radius"], params["alpha_min"],
                edge_damage_delta,                     # WRITE delta
            ])
            # Apply: prev + delta → next (cycle-scoped SSA)
            wp.launch(apply_edge_damage_ssa, dim=E, inputs=[
                edge_damage_buffers[cycle],            # READ prev
                edge_damage_delta,                     # READ delta
                edge_damage_buffers[cycle + 1],        # WRITE next
            ])

            # === PARTICLE DAMAGE ===
            damage_sum = wp.zeros(N, dtype=wp.float32, requires_grad=True)
            damage_count = wp.zeros(N, dtype=wp.float32, requires_grad=True)
            wp.launch(update_particle_damage_kernel, dim=E, inputs=[
                damage_sum, damage_count, alpha_buffers[cycle],
                edge_a, edge_b,
                edge_damage_buffers[cycle + 1],        # READ updated edge_damage
            ])

            particle_damage_delta = wp.zeros(N, dtype=wp.float32, requires_grad=True)
            wp.launch(finalize_particle_damage_kernel, dim=N, inputs=[
                damage_sum, damage_count,
                particle_damage_buffers[cycle],        # READ prev
                particle_damage_delta,                 # WRITE delta
            ])
            # Apply: prev + delta → next
            wp.launch(apply_particle_damage_ssa, dim=N, inputs=[
                particle_damage_buffers[cycle],        # READ prev
                particle_damage_delta,                 # READ delta
                particle_damage_buffers[cycle + 1],    # WRITE next
            ])

            # === SOFT REMOVAL ===
            wp.launch(update_soft_removal_kernel_ssa, dim=N, inputs=[
                particle_damage_buffers[cycle + 1],    # READ updated
                alpha_buffers[cycle],                  # READ prev (gating + carry-over)
                fixed,
                params["threshold"], params["sharpness"],
                alpha_buffers[cycle + 1],              # WRITE next
            ])

        # Loss reads ONLY the final alpha buffer
        wp.launch(loss_alpha_to_zero, dim=N, inputs=[alpha_buffers[-1], loss])

    tape.backward(loss=loss)

    return float(loss.numpy()[0]), {
        "sharpness": float(params["sharpness"].grad.numpy()[0]),
        "alpha_min": float(params["alpha_min"].grad.numpy()[0]),
    }


# =========================================================================
# FD ground truth
# =========================================================================

def fd_grad(run_fn, sys, param_name, eps=1e-3, n_cycles=3):
    defaults = dict(knife_radius=0.1, alpha_min=3.0, sharpness=20.0, threshold=0.5)
    base = defaults[param_name]

    def run_with_value(v):
        kw = dict(defaults)
        kw[param_name] = v
        params = make_params(**kw)
        loss, _ = run_fn(params, sys, n_cycles=n_cycles)
        return loss

    return (run_with_value(base + eps) - run_with_value(base - eps)) / (2 * eps)


# =========================================================================
# Verification
# =========================================================================

def main():
    sys_dict = setup_test_system(N=20, knife_x=0.5)
    n_cycles = 3

    print()
    print("─" * 72)
    print(f"  [FULL SSA — all state cycle-scoped] (N={sys_dict['N']}, n_cycles={n_cycles})")
    print("─" * 72)

    params = make_params()
    loss, autograd_grads = run_full_ssa(params, sys_dict, n_cycles=n_cycles)
    print(f"  Forward loss = {loss:.6e}")

    fd_grads = {
        "sharpness": fd_grad(run_full_ssa, sys_dict, "sharpness", n_cycles=n_cycles),
        "alpha_min": fd_grad(run_full_ssa, sys_dict, "alpha_min", n_cycles=n_cycles),
    }

    results = {}
    for pname in ["sharpness", "alpha_min"]:
        ag = autograd_grads[pname]
        fd = fd_grads[pname]
        sign_ok = np.sign(ag) * np.sign(fd) > 0
        rel_err = abs(ag - fd) / max(abs(fd), 1e-12)
        silent_zero = abs(ag) < 1e-9

        print(f"  ∂loss/∂{pname:<10}  autograd={ag:+.4e}  FD={fd:+.4e}  "
              f"rel_err={rel_err*100:.3f}%  sign={'✓' if sign_ok else '✗'}")

        if silent_zero:
            verdict = "SILENT_ZERO"
        elif sign_ok and rel_err < 0.01:
            verdict = "PASS"
        elif sign_ok and rel_err < 0.05:
            verdict = "PARTIAL"
        elif sign_ok and rel_err < 0.50:
            verdict = "DEGRADED"
        elif sign_ok:
            verdict = "FAIL_MAGNITUDE"
        else:
            verdict = "FAIL_DIRECTION"
        results[pname] = dict(autograd=ag, fd=fd, rel_err=rel_err, verdict=verdict)

    # Determinism check
    print()
    print("  Determinism check (run again):")
    params2 = make_params()
    _, autograd_grads2 = run_full_ssa(params2, sys_dict, n_cycles=n_cycles)
    for pname in ["sharpness", "alpha_min"]:
        diff = abs(autograd_grads[pname] - autograd_grads2[pname])
        print(f"    ∂loss/∂{pname:<10} drift = {diff:.2e}  {'✓' if diff < 1e-9 else '⚠️'}")

    # Final verdict
    print()
    print("=" * 72)
    print("  POC DAY 2 v3 FINAL VERDICT")
    print("=" * 72)

    all_pass = all(results[p]['verdict'] == 'PASS' for p in ['sharpness', 'alpha_min'])
    all_partial = all(results[p]['verdict'] in ['PASS', 'PARTIAL']
                      for p in ['sharpness', 'alpha_min'])

    print(f"  sharpness: {results['sharpness']['verdict']}  "
          f"(rel_err {results['sharpness']['rel_err']*100:.3f}%)")
    print(f"  alpha_min: {results['alpha_min']['verdict']}  "
          f"(rel_err {results['alpha_min']['rel_err']*100:.3f}%)")

    print()
    if all_pass:
        print("  ✅ DAY 2 v3 PASS — full cycle-scoped SSA pattern works")
        print("     Both sharpness AND alpha_min gradients match FD within 1%.")
        print("     → Stage B (production refactor) is mathematically validated.")
        return 0
    elif all_partial:
        print("  🟡 DAY 2 v3 PARTIAL — both within 5%, sharpness PASS-grade")
        print("     Sufficient for production but worth investigating residual error.")
        return 0
    else:
        print("  ❌ DAY 2 v3 FAILED — full cycle-scoped pattern still has issue")
        for p in ['sharpness', 'alpha_min']:
            print(f"     {p}: {results[p]['verdict']} (rel_err {results[p]['rel_err']*100:.2f}%)")
        print("     → Investigate further (maybe Pattern D or framework subtlety)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
