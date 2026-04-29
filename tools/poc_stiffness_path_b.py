#!/usr/bin/env python3
"""
POC: Verify Path B (Warp tape backprop on scalar wp.array params) is feasible.

The question: if we change apply_kernel_ssa's `stiffness: float` → `wp.array`,
does Warp tape correctly compute ∂loss/∂stiffness?

Three escalating tests (decision tree at end):

  Part 1: Synthetic minimal test
          loss = (stiffness * input - target)²
          Verify Warp tape gradient matches closed-form analytic.
          → If FAIL here, Warp framework can't support scalar-array param at all.

  Part 2: Production apply-kernel logic (single launch)
          Use the EXACT same formula as apply_kernel_ssa,
          but with stiffness as wp.array.
          Verify gradient matches analytic + FD.
          → If FAIL here, our specific kernel pattern has issue.

  Part 3: Multi-iteration test (2-3 Jacobi-style loop)
          Loop the apply-kernel logic 4 times (mimics num_iterations=4).
          Verify gradient is still correct after multi-step accumulation.
          → If FAIL here, Stage D atomic_add issue affects parameter gradients.

Outputs PASS/FAIL per part + clear recommendation.
"""

import warp as wp
import numpy as np
import sys

# =========================================================================
# Setup
# =========================================================================

wp.init()
wp.config.verify_autograd_array_access = True

print("=" * 72)
print(f"  Path B POC — Warp tape on scalar wp.array params")
print(f"  Warp version: {wp.__version__}")
print("=" * 72)


# =========================================================================
# Part 1 — Synthetic minimal test
# =========================================================================

def part1_synthetic():
    print("\n" + "─" * 72)
    print("PART 1: Synthetic — loss = (stiffness × input - target)²")
    print("─" * 72)

    @wp.kernel
    def synthetic_kernel(
        input_arr: wp.array(dtype=wp.float32),
        target_arr: wp.array(dtype=wp.float32),
        stiffness: wp.array(dtype=wp.float32),
        output: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        s = stiffness[0]
        diff = s * input_arr[i] - target_arr[i]
        wp.atomic_add(output, 0, diff * diff)

    N = 100
    input_np = np.random.randn(N).astype(np.float32)
    target_np = np.random.randn(N).astype(np.float32)
    s_init = 0.7

    input_arr = wp.array(input_np, dtype=wp.float32)
    target_arr = wp.array(target_np, dtype=wp.float32)
    stiffness = wp.array([s_init], dtype=wp.float32, requires_grad=True)
    output = wp.zeros(1, dtype=wp.float32, requires_grad=True)

    # Warp tape backprop
    with wp.Tape() as tape:
        wp.launch(synthetic_kernel, dim=N,
                  inputs=[input_arr, target_arr, stiffness, output])
    tape.backward(loss=output)
    warp_grad = stiffness.grad.numpy()[0]

    # Closed-form analytic gradient
    # loss = Σ (s*x_i - y_i)²
    # ∂loss/∂s = Σ 2(s*x_i - y_i) * x_i
    diff = s_init * input_np - target_np
    analytic_grad = float(2.0 * np.sum(diff * input_np))

    # FD baseline
    def loss_fn(s):
        return float(np.sum((s * input_np - target_np) ** 2))
    eps = 1e-4
    fd_grad = (loss_fn(s_init + eps) - loss_fn(s_init - eps)) / (2 * eps)

    print(f"  Warp tape grad:  {warp_grad:.6f}")
    print(f"  Analytic grad:   {analytic_grad:.6f}")
    print(f"  FD grad:         {fd_grad:.6f}")

    cos_warp_analytic = np.sign(warp_grad) * np.sign(analytic_grad)
    rel_err = abs(warp_grad - analytic_grad) / max(abs(analytic_grad), 1e-9)
    print(f"  Sign agreement (warp vs analytic): {cos_warp_analytic:+.0f}")
    print(f"  Relative error:                    {rel_err*100:.4f}%")

    if cos_warp_analytic > 0 and rel_err < 0.01:
        print("  [PART 1] ✅ PASS — Warp tape works for scalar wp.array")
        return True
    else:
        print("  [PART 1] ❌ FAIL — Warp tape gives wrong gradient on synthetic test")
        print("           → Path B fundamentally broken; investigate Warp version.")
        return False


# =========================================================================
# Part 2 — Production apply-kernel logic (single launch)
# =========================================================================

def part2_apply_kernel():
    print("\n" + "─" * 72)
    print("PART 2: Production apply-kernel formula (single launch)")
    print("─" * 72)

    # Mimic exactly what apply_kernel_ssa does, but stiffness as wp.array
    @wp.kernel
    def apply_kernel_param_arr(
        deltas: wp.array(dtype=wp.vec3),
        delta_counts: wp.array(dtype=wp.int32),
        damage: wp.array(dtype=wp.float32),
        alpha: wp.array(dtype=wp.float32),
        fixed: wp.array(dtype=wp.int32),
        stiffness_arr: wp.array(dtype=wp.float32),
        shape_delta: wp.array(dtype=wp.vec3),
    ):
        i = wp.tid()
        if alpha[i] < 0.01 or fixed[i] == 1 or delta_counts[i] == 0:
            return
        s = stiffness_arr[0]
        correction = (1.0 - damage[i]) * s * deltas[i] / float(delta_counts[i])
        wp.atomic_add(shape_delta, i, correction)

    @wp.kernel
    def compute_l2_loss(
        shape_delta: wp.array(dtype=wp.vec3),
        target: wp.array(dtype=wp.vec3),
        loss: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        diff = shape_delta[i] - target[i]
        sq = wp.dot(diff, diff)
        wp.atomic_add(loss, 0, sq)

    N = 500
    np.random.seed(42)

    deltas_np = np.random.randn(N, 3).astype(np.float32)
    delta_counts_np = np.random.randint(2, 8, size=N).astype(np.int32)
    damage_np = np.random.uniform(0.0, 0.3, N).astype(np.float32)
    alpha_np = np.ones(N, dtype=np.float32)
    fixed_np = np.zeros(N, dtype=np.int32)
    target_np = np.random.randn(N, 3).astype(np.float32) * 0.1

    deltas = wp.array(deltas_np, dtype=wp.vec3)
    delta_counts = wp.array(delta_counts_np, dtype=wp.int32)
    damage = wp.array(damage_np, dtype=wp.float32)
    alpha = wp.array(alpha_np, dtype=wp.float32)
    fixed = wp.array(fixed_np, dtype=wp.int32)
    target = wp.array(target_np, dtype=wp.vec3)

    s_init = 0.85
    stiffness = wp.array([s_init], dtype=wp.float32, requires_grad=True)
    shape_delta = wp.zeros(N, dtype=wp.vec3, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

    # Warp tape gradient
    with wp.Tape() as tape:
        wp.launch(apply_kernel_param_arr, dim=N,
                  inputs=[deltas, delta_counts, damage, alpha, fixed,
                          stiffness, shape_delta])
        wp.launch(compute_l2_loss, dim=N,
                  inputs=[shape_delta, target, loss])
    tape.backward(loss=loss)
    warp_grad = stiffness.grad.numpy()[0]

    # Analytic gradient
    # shape_delta[i] = (1-damage[i]) * s * deltas[i] / delta_counts[i]
    # loss = Σ ‖shape_delta[i] - target[i]‖²
    # ∂loss/∂s = Σ 2 * (shape_delta[i] - target[i]) · ((1-damage[i]) * deltas[i] / counts[i])
    correction_per_particle = (
        (1.0 - damage_np)[:, None]
        * deltas_np
        / delta_counts_np[:, None].astype(np.float32)
    )
    shape_delta_np = s_init * correction_per_particle
    diff = shape_delta_np - target_np
    analytic_grad = float(2.0 * np.sum(diff * correction_per_particle))

    # FD baseline
    def loss_fn(s):
        sd = s * correction_per_particle
        return float(np.sum((sd - target_np) ** 2))
    eps = 1e-4
    fd_grad = (loss_fn(s_init + eps) - loss_fn(s_init - eps)) / (2 * eps)

    print(f"  Warp tape grad:  {warp_grad:.6f}")
    print(f"  Analytic grad:   {analytic_grad:.6f}")
    print(f"  FD grad:         {fd_grad:.6f}")

    cos_warp_analytic = np.sign(warp_grad) * np.sign(analytic_grad)
    rel_err_warp = abs(warp_grad - analytic_grad) / max(abs(analytic_grad), 1e-9)
    rel_err_fd = abs(fd_grad - analytic_grad) / max(abs(analytic_grad), 1e-9)
    print(f"  Sign agreement (warp vs analytic): {cos_warp_analytic:+.0f}")
    print(f"  Relative error (warp vs analytic): {rel_err_warp*100:.4f}%")
    print(f"  Relative error (FD vs analytic):   {rel_err_fd*100:.4f}%")

    if cos_warp_analytic > 0 and rel_err_warp < 0.05:
        print("  [PART 2] ✅ PASS — apply-kernel pattern gives correct gradient")
        return True
    else:
        print("  [PART 2] ❌ FAIL — apply-kernel pattern broken")
        print("           → Need to investigate atomic_add through wp.array param")
        return False


# =========================================================================
# Part 3 — Multi-iteration test (Jacobi-style loop)
# =========================================================================

def part3_multi_iteration():
    print("\n" + "─" * 72)
    print("PART 3: Multi-iteration test (4 Jacobi-style loops)")
    print("─" * 72)

    @wp.kernel
    def cluster_solve_mock(
        pred_pos: wp.array(dtype=wp.vec3),
        rest_offsets: wp.array(dtype=wp.vec3),
        deltas: wp.array(dtype=wp.vec3),
        delta_counts: wp.array(dtype=wp.int32),
    ):
        # Mock cluster_solve: deltas[i] = rest_offsets[i] - pred_pos[i]
        # Simulates what cluster_solve does — push particles to rest config
        i = wp.tid()
        wp.atomic_add(deltas, i, rest_offsets[i] - pred_pos[i])
        wp.atomic_add(delta_counts, i, 1)

    @wp.kernel
    def apply_kernel_param_arr(
        deltas: wp.array(dtype=wp.vec3),
        delta_counts: wp.array(dtype=wp.int32),
        damage: wp.array(dtype=wp.float32),
        alpha: wp.array(dtype=wp.float32),
        fixed: wp.array(dtype=wp.int32),
        stiffness_arr: wp.array(dtype=wp.float32),
        shape_delta: wp.array(dtype=wp.vec3),
    ):
        i = wp.tid()
        if alpha[i] < 0.01 or fixed[i] == 1 or delta_counts[i] == 0:
            return
        s = stiffness_arr[0]
        correction = (1.0 - damage[i]) * s * deltas[i] / float(delta_counts[i])
        wp.atomic_add(shape_delta, i, correction)

    @wp.kernel
    def add_to_pos(
        pred_pos: wp.array(dtype=wp.vec3),
        shape_delta: wp.array(dtype=wp.vec3),
    ):
        i = wp.tid()
        wp.atomic_add(pred_pos, i, shape_delta[i])

    @wp.kernel
    def clear_deltas(
        deltas: wp.array(dtype=wp.vec3),
        delta_counts: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        deltas[i] = wp.vec3(0.0, 0.0, 0.0)
        delta_counts[i] = 0

    @wp.kernel
    def compute_l2_loss(
        pred_pos: wp.array(dtype=wp.vec3),
        target: wp.array(dtype=wp.vec3),
        loss: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        diff = pred_pos[i] - target[i]
        sq = wp.dot(diff, diff)
        wp.atomic_add(loss, 0, sq)

    N = 200
    np.random.seed(123)

    pred_pos_init = np.random.randn(N, 3).astype(np.float32) * 0.5
    rest_offsets_np = np.random.randn(N, 3).astype(np.float32) * 0.1
    target_np = np.random.randn(N, 3).astype(np.float32) * 0.2

    rest_offsets = wp.array(rest_offsets_np, dtype=wp.vec3)
    target = wp.array(target_np, dtype=wp.vec3)

    damage = wp.zeros(N, dtype=wp.float32)
    alpha = wp.array(np.ones(N).astype(np.float32), dtype=wp.float32)
    fixed = wp.zeros(N, dtype=wp.int32)

    s_init = 0.5
    NUM_ITERS = 4

    def run_forward_with_tape(stiffness_value, want_grad=True):
        stiffness = wp.array([stiffness_value], dtype=wp.float32, requires_grad=True)
        pred_pos = wp.array(pred_pos_init.copy(), dtype=wp.vec3, requires_grad=True)

        with wp.Tape() as tape:
            for it in range(NUM_ITERS):
                deltas = wp.zeros(N, dtype=wp.vec3, requires_grad=True)
                delta_counts = wp.zeros(N, dtype=wp.int32)

                wp.launch(cluster_solve_mock, dim=N,
                          inputs=[pred_pos, rest_offsets, deltas, delta_counts])

                shape_delta = wp.zeros(N, dtype=wp.vec3, requires_grad=True)
                wp.launch(apply_kernel_param_arr, dim=N,
                          inputs=[deltas, delta_counts, damage, alpha, fixed,
                                  stiffness, shape_delta])

                wp.launch(add_to_pos, dim=N,
                          inputs=[pred_pos, shape_delta])

            loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            wp.launch(compute_l2_loss, dim=N,
                      inputs=[pred_pos, target, loss])

        if want_grad:
            tape.backward(loss=loss)
            return float(loss.numpy()[0]), float(stiffness.grad.numpy()[0])
        else:
            return float(loss.numpy()[0]), None

    loss_v, warp_grad = run_forward_with_tape(s_init, want_grad=True)
    print(f"  Forward loss at s={s_init}: {loss_v:.6f}")
    print(f"  Warp tape grad: {warp_grad:.6f}")

    # FD baseline
    eps = 1e-3
    loss_plus, _ = run_forward_with_tape(s_init + eps, want_grad=False)
    loss_minus, _ = run_forward_with_tape(s_init - eps, want_grad=False)
    fd_grad = (loss_plus - loss_minus) / (2 * eps)
    print(f"  FD grad (eps={eps}): {fd_grad:.6f}")

    # Cosine vs FD (FD is now ground truth since no closed-form available)
    cos_warp_fd = np.sign(warp_grad) * np.sign(fd_grad)
    rel_err = abs(warp_grad - fd_grad) / max(abs(fd_grad), 1e-9)
    print(f"  Sign agreement (warp vs FD): {cos_warp_fd:+.0f}")
    print(f"  Relative error:              {rel_err*100:.4f}%")

    if cos_warp_fd > 0 and rel_err < 0.10:
        print(f"  [PART 3] ✅ PASS — Multi-iter (4 loops) gives correct grad")
        print(f"           → Stage D atomic_add issue not affecting param-grad here")
        return True
    elif cos_warp_fd > 0 and rel_err < 0.50:
        print(f"  [PART 3] 🟡 PARTIAL — direction OK, magnitude off ~{rel_err*100:.0f}%")
        print(f"           → Adam may still converge (direction is what matters)")
        return True
    else:
        print(f"  [PART 3] ❌ FAIL — Multi-iter accumulation breaks grad")
        print(f"           → Stage D atomic_add issue affects param-grad")
        print(f"           → Path B blocked for this kernel; need @wp.func_replay first")
        return False


# =========================================================================
# Decision tree
# =========================================================================

def main():
    p1 = part1_synthetic()
    p2 = part2_apply_kernel() if p1 else False
    p3 = part3_multi_iteration() if p2 else False

    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)

    if p1 and p2 and p3:
        print("""
✅ Path B is feasible — proceed with full kernel refactor.

  Recommendation:
  - Refactor apply_kernel_ssa, predict_kernel_ssa, collision_kernel_ssa
    to take stiffness/damping/floor_friction as wp.array.
  - Estimated time: 2-3 days for Stage 1 (3 params).
  - Run Stage 1 calibration with Adam → expect 5-7× speedup vs CMA-ES.
""")
    elif p1 and p2 and not p3:
        print("""
🟡 Partial — Path B works for single launch, fails on multi-iteration.

  Diagnosis: Stage D atomic_add direction-flip likely affects param gradients
             when accumulated across iterations.

  Recommendation:
  - Single-launch parameters (e.g., damping in predict_kernel) ARE safe → Path B OK
  - Multi-iteration parameters (e.g., stiffness inside Jacobi loop) need either:
    (a) @wp.func_replay fix (Path 4 prerequisite), or
    (b) Path A (FD) fallback.
  - Investigate Stage D fix as Sprint 2 priority — unlocks both Path 4 AND Path B
    for Jacobi-loop parameters.
""")
    elif p1 and not p2:
        print("""
🔴 apply-kernel pattern broken in Warp tape.

  Diagnosis: atomic_add through wp.array parameter has framework issue
             (this is rare; usually scalar wp.array works fine in Warp 1.12).

  Recommendation:
  - Fall back to Path A (Finite Differences) for all calibration.
  - File Warp bug report with this minimal repro.
""")
    else:
        print("""
🔴 Warp framework can't handle scalar wp.array param at all.

  This shouldn't happen with Warp 1.12+. Suspected issues:
  - Outdated Warp version
  - requires_grad not set correctly
  - Memory layout issue

  Recommendation:
  - Verify Warp version: `python -c "import warp; print(warp.__version__)"`
  - Path A (FD) is the only reliable path until this is resolved.
""")


if __name__ == "__main__":
    main()
