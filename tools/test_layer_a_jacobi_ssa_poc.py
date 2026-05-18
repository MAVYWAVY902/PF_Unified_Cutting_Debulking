#!/usr/bin/env python3
"""Day 2 POC test for Layer A SSA refactor.

Verifies G1 and G2 from project_layer_a_fix_implementation_plan_2026_05_05.md:

  G1 — verify_autograd warnings disappear:
       Setup wp.config.verify_autograd_array_access=True. Run 1 substep
       through production solver._substep() in SSA mode with knife active,
       record tape, run backward. Capture stderr.

       PASS: 0 warnings mentioning `apply_shape_delta_kernel` or
             `apply_collision_delta_kernel` (these are the legacy in-place
             atomic_add kernels — should not even be reached in SSA mode).

       Other warnings on unrelated kernels are SURFACED but not failure.

  G2 — kernel-level forward + FD parity for apply_shape_delta_ssa_kernel:
       Synthetic small test (N=100, vec3). Run kernel forward; verify
       output matches `pred_pos_in + shape_delta` numerically. FD parity
       on ∂loss/∂pred_pos_in.

       PASS: forward exact match (<1e-6), FD parity rel_err < 0.5%.
"""
import io
import os
import sys
import contextlib
from pathlib import Path

import numpy as np
import warp as wp
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))


# Define loss kernel before turning on verify
@wp.kernel
def loss_sum_pos_z_kernel(
    pos: wp.array(dtype=wp.vec3),
    interior_mask: wp.array(dtype=wp.int32),
    interior_count: float,
    loss_buf: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if interior_mask[i] == 1:
        wp.atomic_add(loss_buf, 0, pos[i][2] / interior_count)


from pf_shape_matching_warp.sim.kernels import (
    apply_shape_delta_ssa_kernel,
    apply_collision_delta_ssa_kernel,
)


# =====================================================================
# G1 — production substep with verify_autograd_array_access
# =====================================================================

def gate_g1():
    print("=" * 70)
    print("G1 — verify_autograd_array_access on production solver._substep()")
    print("=" * 70)

    # Turn ON verify
    wp.config.verify_autograd_array_access = True

    # Import + setup env (after enabling verify so kernels recompile w/ checks)
    from warp_sim.envs.debulk_env import DebulkEnv

    cfg_path = REPO_ROOT / "warp_sim" / "config" / "debulk_cap_armA.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["obs_mode"] = "baseline"
    cfg["task_type"] = "capsule_only"
    cfg["use_ssa_soft_removal"] = True

    env = DebulkEnv(cfg)
    env.reset()
    env.solver.knife_active = 1
    env.solver.knife_radius = 0.15
    env.solver.set_knife_pos(0.85, 0.5, 0.12)  # on torus tube
    env.solver.cutting_update_interval = 1
    env.solver._alloc_ssa_buffers()

    # Build interior mask (capsule_count is in env)
    N = env.solver.N
    interior_mask = wp.zeros(N, dtype=wp.int32, device=env.solver.device)
    # set interior = NOT capsule via existing mask
    # quick path: use env.cap_mask if exists, else mark all as interior
    if hasattr(env, "cap_mask"):
        cap_np = env.cap_mask.numpy()
        interior_np = np.where(cap_np == 0, 1, 0).astype(np.int32)
        interior_mask = wp.array(interior_np, dtype=wp.int32, device=env.solver.device)
    else:
        # fallback: all particles are interior
        interior_mask = wp.array(np.ones(N, dtype=np.int32), dtype=wp.int32,
                                  device=env.solver.device)
    interior_count = float(interior_mask.numpy().sum())
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.solver.device,
                         requires_grad=True)

    # Capture stderr while running tape forward+backward
    stderr_capture = io.StringIO()

    print(f"  N={N}, interior_count={int(interior_count)}, knife at (0.85,0.5,0.12)")
    print("  Recording tape forward + backward with verify_autograd_array_access=True...")

    tape = wp.Tape()
    with contextlib.redirect_stderr(stderr_capture):
        try:
            with tape:
                env.solver._substep()
                wp.launch(loss_sum_pos_z_kernel, dim=N,
                          inputs=[env.solver.p["pos"], interior_mask,
                                  interior_count, loss_buf],
                          device=env.solver.device)

            tape.backward(loss=loss_buf)
            run_ok = True
            run_err = None
        except Exception as e:
            run_ok = False
            run_err = e

    stderr_text = stderr_capture.getvalue()

    print(f"  Tape forward+backward: {'OK' if run_ok else 'FAIL: ' + repr(run_err)}")
    print(f"  Stderr captured: {len(stderr_text)} chars")

    # Parse stderr for warnings on TARGET kernels
    target_kernels = [
        "apply_shape_delta_kernel",
        "apply_collision_delta_kernel",
    ]
    target_warning_lines = []
    other_warning_lines = []

    for line in stderr_text.splitlines():
        line_lower = line.lower()
        if "warning" not in line_lower and "warn" not in line_lower \
            and "verify_autograd" not in line_lower and "rwr" not in line_lower \
            and "read" not in line_lower:
            continue
        is_target = any(tk in line for tk in target_kernels)
        if is_target:
            target_warning_lines.append(line)
        else:
            other_warning_lines.append(line)

    # Heuristic: also flag any line mentioning the target kernel
    for line in stderr_text.splitlines():
        if any(tk in line for tk in target_kernels) and line not in target_warning_lines:
            target_warning_lines.append(line)

    print()
    print(f"  Warnings on target kernels (apply_shape_delta_kernel, apply_collision_delta_kernel):")
    if target_warning_lines:
        print(f"    ❌ FAIL — {len(target_warning_lines)} warning lines found:")
        for line in target_warning_lines[:20]:
            print(f"      {line}")
        return False, stderr_text
    else:
        print(f"    ✅ PASS — 0 warnings on target kernels")

    print()
    print(f"  Other warning lines (informational, not G1 fail):")
    if other_warning_lines:
        print(f"    {len(other_warning_lines)} other warning lines:")
        for line in other_warning_lines[:20]:
            print(f"      {line}")
        if len(other_warning_lines) > 20:
            print(f"      ... ({len(other_warning_lines) - 20} more)")
    else:
        print(f"    (none)")

    return True, stderr_text


# =====================================================================
# G2 — kernel-level forward + FD parity
# =====================================================================

def gate_g2():
    print()
    print("=" * 70)
    print("G2 — apply_shape_delta_ssa_kernel forward + FD parity")
    print("=" * 70)

    device = "cuda:0"
    N = 100
    rng = np.random.default_rng(seed=42)
    pred_pos_in_np = rng.standard_normal((N, 3)).astype(np.float32)
    shape_delta_np = rng.standard_normal((N, 3)).astype(np.float32) * 0.01

    # ---------- Forward correctness ----------
    pred_pos_in = wp.array(pred_pos_in_np, dtype=wp.vec3, device=device,
                            requires_grad=True)
    shape_delta = wp.array(shape_delta_np, dtype=wp.vec3, device=device,
                            requires_grad=True)
    pred_pos_out = wp.zeros(N, dtype=wp.vec3, device=device, requires_grad=True)

    wp.launch(apply_shape_delta_ssa_kernel, dim=N,
              inputs=[pred_pos_in, shape_delta],
              outputs=[pred_pos_out],
              device=device)

    expected = pred_pos_in_np + shape_delta_np
    actual = pred_pos_out.numpy()
    forward_err = float(np.max(np.abs(expected - actual)))
    print(f"  Forward: max abs err = {forward_err:.2e}  "
          f"({'PASS' if forward_err < 1e-6 else 'FAIL'})")

    if forward_err >= 1e-6:
        return False

    # ---------- FD parity for ∂(sum(pred_pos_out))/∂(pred_pos_in) ----------
    # Loss = sum over all axes of pred_pos_out
    @wp.kernel
    def loss_sum_kernel(
        x: wp.array(dtype=wp.vec3),
        loss_buf: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        wp.atomic_add(loss_buf, 0, x[i][0] + x[i][1] + x[i][2])

    # Autograd
    pred_pos_in_a = wp.array(pred_pos_in_np, dtype=wp.vec3, device=device,
                              requires_grad=True)
    shape_delta_a = wp.array(shape_delta_np, dtype=wp.vec3, device=device,
                              requires_grad=True)
    pred_pos_out_a = wp.zeros(N, dtype=wp.vec3, device=device,
                               requires_grad=True)
    loss_buf_a = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(apply_shape_delta_ssa_kernel, dim=N,
                  inputs=[pred_pos_in_a, shape_delta_a],
                  outputs=[pred_pos_out_a], device=device)
        wp.launch(loss_sum_kernel, dim=N,
                  inputs=[pred_pos_out_a, loss_buf_a], device=device)

    tape.backward(loss=loss_buf_a)
    grad_autograd = pred_pos_in_a.grad.numpy().copy()
    # Expected analytical gradient: each pred_pos_in[i][a] contributes 1.0
    expected_grad = np.ones_like(pred_pos_in_np)
    autograd_err = float(np.max(np.abs(grad_autograd - expected_grad)))
    print(f"  Autograd vs analytical (1.0): max abs err = {autograd_err:.2e}  "
          f"({'PASS' if autograd_err < 1e-5 else 'FAIL'})")

    # FD parity
    eps = 1e-3
    fd_grad = np.zeros_like(pred_pos_in_np)
    base_loss_buf = wp.zeros(1, dtype=wp.float32, device=device)

    def compute_loss(x_np):
        x = wp.array(x_np, dtype=wp.vec3, device=device)
        out = wp.zeros(N, dtype=wp.vec3, device=device)
        lb = wp.zeros(1, dtype=wp.float32, device=device)
        wp.launch(apply_shape_delta_ssa_kernel, dim=N,
                  inputs=[x, shape_delta], outputs=[out], device=device)
        wp.launch(loss_sum_kernel, dim=N, inputs=[out, lb], device=device)
        return float(lb.numpy()[0])

    # Sample 5 random indices to FD-check (full 3N=300 too slow but unnecessary)
    sample_idxs = rng.choice(N, size=5, replace=False)
    for ii in sample_idxs:
        for ax in range(3):
            x_p = pred_pos_in_np.copy(); x_p[ii, ax] += eps
            x_m = pred_pos_in_np.copy(); x_m[ii, ax] -= eps
            fd_grad[ii, ax] = (compute_loss(x_p) - compute_loss(x_m)) / (2 * eps)

    fd_sample = fd_grad[sample_idxs]
    autograd_sample = grad_autograd[sample_idxs]
    rel_err = float(np.max(np.abs(autograd_sample - fd_sample) / (np.abs(fd_sample) + 1e-9)))
    print(f"  FD parity (5×3 samples): max rel_err = {rel_err:.2%}  "
          f"({'PASS' if rel_err < 0.005 else 'FAIL'})")

    return forward_err < 1e-6 and autograd_err < 1e-5 and rel_err < 0.005


# =====================================================================
# Main
# =====================================================================

def main():
    g2_ok = gate_g2()
    g1_ok, _ = gate_g1()

    print()
    print("=" * 70)
    print("Day 2 Summary")
    print("=" * 70)
    print(f"  G1 (verify_autograd warnings disappear):   {'PASS' if g1_ok else 'FAIL'}")
    print(f"  G2 (kernel-level FD parity < 0.5%):        {'PASS' if g2_ok else 'FAIL'}")
    if g1_ok and g2_ok:
        print("\n  ✅ Day 2 PASS — proceed to Day 3 (Test E/I/J full FD parity)")
        return 0
    else:
        print("\n  ❌ Day 2 FAIL — STOP, debug per plan §3 Day 2 root-cause guide")
        return 1


if __name__ == "__main__":
    sys.exit(main())
