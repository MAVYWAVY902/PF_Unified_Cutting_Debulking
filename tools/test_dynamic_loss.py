#!/usr/bin/env python3
"""
Decisive test for the "alpha-only loss saturates sharpness info" hypothesis.

Single variable: loss function.
  - STATIC loss:  MSE(final alpha distribution)
  - DYNAMIC loss: Σ over cycles MSE(alpha_at_each_cycle)

Same Adam settings, same N=20, same 1 knife position. If dynamic loss recovers
sharpness with err < 5%, hypothesis verified. If not, diagnosis is wrong.

This is the test that closes the diagnostic loop. Previous tests only showed
the sharpness *floor* under static loss; they didn't show that switching loss
fixes it. This is the missing positive evidence.
"""
import sys, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import warp as wp
wp.init()

from tools.calibrate_stage2_FD import setup_synthetic_system
from pf_shape_matching_warp.sim.pf_damage import (
    update_edge_damage_kernel_ssa, apply_edge_damage_delta_kernel,
    update_particle_damage_kernel, finalize_particle_damage_kernel_ssa,
    apply_particle_damage_delta_kernel, update_soft_removal_kernel,
)

TRUE_AM, TRUE_SP = 3.0, 20.0
INIT_AM, INIT_SP = 5.0, 10.0


def run_cutting_trajectory(sys_dict, alpha_min, sharpness,
                           knife_radius=0.1, threshold=0.5, n_cycles=3):
    """Cutting pipeline returning alpha at every cycle (not just final)."""
    N, E = sys_dict["N"], sys_dict["E"]
    pos = wp.array(sys_dict["pos"], dtype=wp.vec3)
    edge_a = wp.array(sys_dict["edge_a"], dtype=wp.int32)
    edge_b = wp.array(sys_dict["edge_b"], dtype=wp.int32)
    pf_rest = wp.array(sys_dict["pf_rest"], dtype=wp.float32)
    knife_pos = wp.array(sys_dict["knife_pos"], dtype=wp.vec3)
    fixed = wp.array(sys_dict["fixed"], dtype=wp.int32)
    edge_damage = wp.zeros(E, dtype=wp.float32)
    particle_damage = wp.zeros(N, dtype=wp.float32)
    alpha = wp.array(np.ones(N, dtype=np.float32), dtype=wp.float32)

    trajectory = []
    for _ in range(n_cycles):
        edge_damage_delta = wp.zeros(E, dtype=wp.float32)
        wp.launch(update_edge_damage_kernel_ssa, dim=E, inputs=[
            pos, alpha, edge_a, edge_b, pf_rest, edge_damage,
            knife_pos, knife_radius, alpha_min, edge_damage_delta,
        ])
        wp.launch(apply_edge_damage_delta_kernel, dim=E,
                  inputs=[edge_damage, edge_damage_delta])
        damage_sum = wp.zeros(N, dtype=wp.float32)
        damage_count = wp.zeros(N, dtype=wp.float32)
        wp.launch(update_particle_damage_kernel, dim=E, inputs=[
            damage_sum, damage_count, alpha, edge_a, edge_b, edge_damage,
        ])
        particle_damage_delta = wp.zeros(N, dtype=wp.float32)
        wp.launch(finalize_particle_damage_kernel_ssa, dim=N, inputs=[
            damage_sum, damage_count, particle_damage, particle_damage_delta,
        ])
        wp.launch(apply_particle_damage_delta_kernel, dim=N,
                  inputs=[particle_damage, particle_damage_delta])
        wp.launch(update_soft_removal_kernel, dim=N, inputs=[
            particle_damage, alpha, fixed, threshold, sharpness,
        ])
        wp.synchronize()
        trajectory.append(alpha.numpy().copy())
    return trajectory


def loss_static(traj_current, traj_target):
    return float(np.mean((traj_current[-1] - traj_target[-1]) ** 2))


def loss_dynamic(traj_current, traj_target):
    return float(sum(np.mean((c - t) ** 2) for c, t in zip(traj_current, traj_target)))


def adam_calibrate(sys_dict, loss_kind, n_cycles=3, n_steps=200,
                   lr_a=0.05, lr_s=0.3, eps_a=0.05, eps_s=0.25, verbose=False):
    """Adam with central-diff FD on chosen loss. Returns (best_am, best_sp, err_a, err_s)."""
    target_traj = run_cutting_trajectory(sys_dict, TRUE_AM, TRUE_SP, n_cycles=n_cycles)
    loss_fn = {"static": loss_static, "dynamic": loss_dynamic}[loss_kind]

    def L(a, s):
        return loss_fn(run_cutting_trajectory(sys_dict, a, s, n_cycles=n_cycles),
                       target_traj)

    am, sp = float(INIT_AM), float(INIT_SP)
    m = np.zeros(2); v = np.zeros(2); t = 0
    b1, b2 = 0.9, 0.999
    lr = np.array([lr_a, lr_s])

    best_loss = float("inf")
    best = (am, sp, 0)

    for step in range(n_steps):
        L0 = L(am, sp)
        g_a = (L(am + eps_a, sp) - L(am - eps_a, sp)) / (2 * eps_a)
        g_s = (L(am, sp + eps_s) - L(am, sp - eps_s)) / (2 * eps_s)
        grad = np.array([g_a, g_s])

        if L0 < best_loss:
            best_loss = L0; best = (am, sp, step)

        t += 1
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        m_hat = m / (1 - b1 ** t); v_hat = v / (1 - b2 ** t)
        new = np.array([am, sp]) - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        am = max(0.5, min(50.0, new[0]))
        sp = max(1.0, min(200.0, new[1]))

        if verbose and step % 50 == 0:
            print(f"    [{loss_kind} step {step:3d}]  ({am:.3f}, {sp:.3f})  L={L0:.3e}")

    err_a = abs(best[0] - TRUE_AM) / TRUE_AM
    err_s = abs(best[1] - TRUE_SP) / TRUE_SP
    return best[0], best[1], err_a, err_s, best_loss


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 72)
    print("  DECISIVE TEST: Static vs Dynamic loss")
    print("=" * 72)
    print("  Same setup: N=20, 1 knife, 200 Adam steps, lr=(0.05, 0.3), n_cycles=3")
    print("  ONLY variable: loss function")
    print()

    sys_dict = setup_synthetic_system(N=20)

    # --- Control: STATIC loss ---
    print("  [STATIC]  loss = MSE(final alpha)")
    am_s, sp_s, ea_s, es_s, L_s = adam_calibrate(sys_dict, "static", verbose=True)
    print(f"    best params: ({am_s:.4f}, {sp_s:.4f})")
    print(f"    err: α_min={ea_s*100:.1f}%, sharp={es_s*100:.1f}%, best_loss={L_s:.3e}")
    print()

    # --- Test: DYNAMIC loss ---
    print("  [DYNAMIC] loss = Σ over cycles MSE(alpha_at_each_cycle)")
    am_d, sp_d, ea_d, es_d, L_d = adam_calibrate(sys_dict, "dynamic", verbose=True)
    print(f"    best params: ({am_d:.4f}, {sp_d:.4f})")
    print(f"    err: α_min={ea_d*100:.1f}%, sharp={es_d*100:.1f}%, best_loss={L_d:.3e}")
    print()

    # --- Verdict ---
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    print(f"  α_min err: {ea_s*100:5.1f}% (static)  →  {ea_d*100:5.1f}% (dynamic)")
    print(f"  sharp err: {es_s*100:5.1f}% (static)  →  {es_d*100:5.1f}% (dynamic)")
    drop = (es_s - es_d) / es_s if es_s > 0 else 0
    print(f"  sharp err drop: {drop*100:.1f}%")
    print()
    if es_d < 0.05:
        print("  ✅ DYNAMIC LOSS BREAKS THE FLOOR (sharp err < 5%)")
        print("     → 'alpha-only loss saturates sharpness info' VERIFIED.")
        print("     → Loss design IS the root cause.")
    elif es_d < es_s * 0.5:
        print("  🟡 DYNAMIC LOSS HELPS substantially (>50% drop) but doesn't reach <5%")
        print("     → Hypothesis SUPPORTED. Further: need richer loss (force trace)")
        print("       for full sharpness recovery.")
    elif es_d < es_s:
        print("  🟡 DYNAMIC LOSS HELPS slightly")
        print("     → Hypothesis weakly supported. Direction right, magnitude unclear.")
    else:
        print("  ❌ DYNAMIC LOSS DOES NOT HELP")
        print("     → Hypothesis WRONG. Bottleneck is elsewhere (kernel? FD? sim?).")

    # --- Bonus check: dynamic loss + N=500 ---
    print()
    print("  --- BONUS: dynamic loss + N=500 (cumulative best case) ---")
    sys_500 = setup_synthetic_system(N=500)
    am_b, sp_b, ea_b, es_b, L_b = adam_calibrate(sys_500, "dynamic")
    print(f"    err: α_min={ea_b*100:.1f}%, sharp={es_b*100:.1f}%, best_loss={L_b:.3e}")

    print(f"\n[total time] {time.time() - t0:.1f}s")
