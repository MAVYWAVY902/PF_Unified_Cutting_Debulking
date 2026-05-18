#!/usr/bin/env python3
"""Path 4 Falsification Test E — verify n=1 gradient via FD parity.

CHALLENGE: Test D showed at n_substeps=1, autograd grad on knife_pos ≈ 5e7,
on threshold ≈ 3.5. Are these CORRECT (autograd matches FD), or is autograd
already broken at n=1?

If autograd ≈ FD at n=1 → gradient really IS that large, multi-substep is
genuine explosion.

If autograd ≠ FD at n=1 → autograd is biased FROM THE START (not explosion).
Then root cause is per-substep autograd bug, not horizon accumulation.

This is the discriminator I missed in Test D — I assumed n=1 was correct
because it was finite, but never verified.
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

wp.config.verify_autograd_array_access = False  # silence warning spam

from path4_poc_v1_dual_purpose import (
    differentiable_substep, make_param_dict, snap_solver_state,
    reset_solver_state, loss_mean_alpha_at_interior_kernel,
)
from warp_sim.envs.debulk_env import DebulkEnv


def setup_env():
    cfg_path = REPO_ROOT / "warp_sim" / "config" / "debulk_cap_armA.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["obs_mode"] = "baseline"
    cfg["task_type"] = "capsule_only"
    cfg["use_ssa_soft_removal"] = True

    env = DebulkEnv(cfg)
    env.reset()
    env.solver.cutting_update_interval = 1
    env.solver.knife_active = 1
    env.solver.knife_radius = 0.15
    env.solver.aspiration_threshold = 0.05
    env.solver.alpha_min = 3.0
    env.solver.soft_removal_sharpness = 20.0
    env.solver._alloc_ssa_buffers()
    return env


def forward_only_loss(env, params, knife_pos, n_substeps,
                      interior_mask_wp, interior_count):
    dt_sub = env.solver.dt / float(env.solver.substeps)
    for _ in range(n_substeps):
        differentiable_substep(env.solver, params, knife_pos, dt_sub)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device)
    wp.launch(loss_mean_alpha_at_interior_kernel, dim=env.solver.N,
              inputs=[env.solver._current_alpha(), interior_mask_wp,
                      interior_count, loss_buf],
              device=env.device)
    wp.synchronize()
    return float(loss_buf.numpy()[0])


def main():
    wp.init()
    print("=" * 72)
    print("  Test E — n=1 gradient FD parity (verify autograd correctness)")
    print("=" * 72)

    env = setup_env()
    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)

    knife_pos_np = (0.85, 0.5, 0.12)
    n_substeps = 1

    # ---- Snapshot pristine state (BEFORE any forward) ----
    state_snap = snap_solver_state(env)

    # ---- Compute autograd gradient at n=1 ----
    print("\n[1] Autograd gradient at n=1...")
    params = make_param_dict(20.0, 3.0, 0.05, 0.15, requires_grad=True,
                             device=env.device)
    knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                         device=env.device, requires_grad=True)

    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device,
                        requires_grad=True)
    dt_sub = env.solver.dt / float(env.solver.substeps)
    with wp.Tape() as tape:
        differentiable_substep(env.solver, params, knife_pos, dt_sub)
        wp.launch(loss_mean_alpha_at_interior_kernel, dim=env.solver.N,
                  inputs=[env.solver._current_alpha(), interior_mask_wp,
                          interior_count, loss_buf],
                  device=env.device)
    tape.backward(loss=loss_buf)
    wp.synchronize()

    ag_loss = float(loss_buf.numpy()[0])
    ag_grads = {
        "knife_pos_x": float(knife_pos.grad.numpy()[0][0]),
        "knife_pos_y": float(knife_pos.grad.numpy()[0][1]),
        "knife_pos_z": float(knife_pos.grad.numpy()[0][2]),
        "knife_radius": float(params["knife_radius"].grad.numpy()[0]),
        "alpha_min": float(params["alpha_min"].grad.numpy()[0]),
        "sharpness": float(params["sharpness"].grad.numpy()[0]),
        "threshold": float(params["threshold"].grad.numpy()[0]),
    }
    print(f"  Loss: {ag_loss:.6e}")
    for k, v in ag_grads.items():
        print(f"    autograd ∂L/∂{k:<13} = {v:+.6e}")

    # ---- Compute FD gradient at n=1 (per scalar, central diff) ----
    print("\n[2] FD gradient at n=1 (central diff, ε per-param)...")

    fd_grads = {}

    # Scalars
    for name, eps in (("alpha_min", 1e-2), ("sharpness", 5e-2),
                      ("threshold", 1e-3), ("knife_radius", 1e-3)):
        base = float(params[name].numpy()[0])

        # plus
        reset_solver_state(env, state_snap)
        env.solver.step_count = 0
        p_plus = make_param_dict(
            float(params["sharpness"].numpy()[0]) if name != "sharpness" else base + eps,
            float(params["alpha_min"].numpy()[0]) if name != "alpha_min" else base + eps,
            float(params["threshold"].numpy()[0]) if name != "threshold" else base + eps,
            float(params["knife_radius"].numpy()[0]) if name != "knife_radius" else base + eps,
            requires_grad=False, device=env.device,
        )
        kp_plus = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                           device=env.device)
        loss_plus = forward_only_loss(env, p_plus, kp_plus, n_substeps,
                                      interior_mask_wp, interior_count)

        # minus
        reset_solver_state(env, state_snap)
        env.solver.step_count = 0
        p_minus = make_param_dict(
            float(params["sharpness"].numpy()[0]) if name != "sharpness" else base - eps,
            float(params["alpha_min"].numpy()[0]) if name != "alpha_min" else base - eps,
            float(params["threshold"].numpy()[0]) if name != "threshold" else base - eps,
            float(params["knife_radius"].numpy()[0]) if name != "knife_radius" else base - eps,
            requires_grad=False, device=env.device,
        )
        kp_minus = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                            device=env.device)
        loss_minus = forward_only_loss(env, p_minus, kp_minus, n_substeps,
                                       interior_mask_wp, interior_count)

        fd = (loss_plus - loss_minus) / (2.0 * eps)
        fd_grads[name] = fd
        print(f"  FD (ε={eps:.0e}) ∂L/∂{name:<13}  L+={loss_plus:.4e}  "
              f"L-={loss_minus:.4e}  → {fd:+.6e}")

    # knife_pos per axis
    for axis, axis_name in enumerate(("x", "y", "z")):
        eps = 5e-3
        base_kp = list(knife_pos_np)

        reset_solver_state(env, state_snap)
        env.solver.step_count = 0
        kp_np = base_kp.copy(); kp_np[axis] += eps
        kp = wp.array([wp.vec3(*kp_np)], dtype=wp.vec3, device=env.device)
        params_fd = make_param_dict(20.0, 3.0, 0.05, 0.15,
                                    requires_grad=False, device=env.device)
        loss_plus = forward_only_loss(env, params_fd, kp, n_substeps,
                                      interior_mask_wp, interior_count)

        reset_solver_state(env, state_snap)
        env.solver.step_count = 0
        kp_np = base_kp.copy(); kp_np[axis] -= eps
        kp = wp.array([wp.vec3(*kp_np)], dtype=wp.vec3, device=env.device)
        loss_minus = forward_only_loss(env, params_fd, kp, n_substeps,
                                       interior_mask_wp, interior_count)

        fd = (loss_plus - loss_minus) / (2.0 * eps)
        fd_grads[f"knife_pos_{axis_name}"] = fd
        print(f"  FD (ε={eps:.0e}) ∂L/∂knife_pos_{axis_name}     L+={loss_plus:.4e}  "
              f"L-={loss_minus:.4e}  → {fd:+.6e}")

    # ---- Compare ----
    print()
    print("=" * 72)
    print("PARITY COMPARISON @ n_substeps=1")
    print("=" * 72)
    print(f"  {'param':<20} {'autograd':>15} {'FD':>15} {'rel_err':>10}")
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*10}")

    pass_count = 0
    fail_count = 0
    for name in ag_grads:
        ag = ag_grads[name]
        fd = fd_grads.get(name, None)
        if fd is None:
            continue
        rel = abs(ag - fd) / max(abs(fd), 1e-12)
        sign_ok = (np.sign(ag) == np.sign(fd)) if (abs(fd) > 1e-12 and abs(ag) > 1e-12) else None
        status = ("PASS" if rel < 0.1 else
                  "MAGNITUDE_WAY_OFF" if rel > 100 else
                  "FAIL")
        if status == "PASS":
            pass_count += 1
        else:
            fail_count += 1
        print(f"  {name:<20} {ag:>+15.4e} {fd:>+15.4e} {rel*100:>9.1f}% [{status}]")

    print()
    print("─" * 72)
    if fail_count == 0:
        print("  ✅ All n=1 gradients match FD → autograd correct at n=1")
        print("  → 'Gradient explosion across substeps' framing is plausible")
        print("    (multi-substep amplifies CORRECT 1-step gradient)")
    else:
        print(f"  ❌ {fail_count}/{pass_count + fail_count} gradients FAIL FD parity at n=1")
        print("  → autograd is BIASED FROM THE START (not just explosion)")
        print("  → 'Gradient explosion' framing was WRONG.")
        print("  → Real root cause: autograd backward in solver._substep mirror")
        print("    has a per-substep bug. Multi-substep just compounds bias.")
    print("─" * 72)


if __name__ == "__main__":
    main()
