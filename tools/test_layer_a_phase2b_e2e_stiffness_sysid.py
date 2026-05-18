#!/usr/bin/env python3
"""Phase 2B end-to-end: stiffness sys-ID via gradient descent through Layer A.

Validates gradient-based sys-ID through the full Layer A path using POC v3
5-kernel split (centroid + cov + Müller polar + R + scatter).

Workflow:
  1. Run forward with ground-truth stiffness (e.g. 0.85) → record target final pos
  2. Initialize calibrator with wrong stiffness guess (e.g. 0.5)
  3. Adam-style gradient descent: loss = MSE(sim_pos, target_pos)
  4. Tape backward through POC v3 substep → grad on stiffness
  5. Verify stiffness recovers to within 5% of ground-truth in N steps

PASS: stiffness converges to gt within 50 iterations, loss drops by ≥10×
FAIL: gradient is wrong sign, loss increases, or stiffness diverges

Test target: prove POC v3 split kernels are USABLE for production sys-ID.
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

from pf_shape_matching_warp.sim.kernels import (
    predict_kernel_ssa, collision_kernel_ssa,
    update_velocity_kernel,
    apply_shape_delta_ssa_kernel,
    apply_collision_delta_ssa_kernel,
    extract_rotation_muller,
)
from path4_falsification_test_H_isolate import (
    setup_env, loss_sum_pos_z_kernel,
)
from path4_poc_v1_dual_purpose import (
    make_param_dict, snap_solver_state, reset_solver_state,
)


# ============================================================================
#  apply_kernel_ssa variant taking stiffness as wp.array (for autograd tracking)
# ============================================================================

@wp.kernel
def apply_kernel_ssa_stiff_arr(
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    stiffness_arr: wp.array(dtype=wp.float32),    # ★ ARRAY (1-elem) for autograd
    shape_delta: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if alpha[i] < 0.01 or fixed[i] == 1 or delta_counts[i] == 0:
        return
    dmg = damage[i]
    stiffness = stiffness_arr[0]
    local_stiff = stiffness * (1.0 - dmg * 0.95)
    inv_count = 1.0 / wp.float32(delta_counts[i])
    correction = local_stiff * inv_count * deltas[i]
    wp.atomic_add(shape_delta, i, correction)


# ============================================================================
#  POC v3 split kernels (mirror of test_layer_a_phase2b_poc_v3_full_physics)
# ============================================================================

@wp.kernel
def cluster_compute_sums(
    pred_pos: wp.array(dtype=wp.vec3), alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32), cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32), rest_offsets: wp.array(dtype=wp.vec3),
    cx_sum_buf: wp.array(dtype=wp.vec3), qc_sum_buf: wp.array(dtype=wp.vec3),
    tw_buf: wp.array(dtype=wp.float32),
):
    cid = wp.tid()
    start = cluster_starts[cid]; size = cluster_sizes[cid]
    if size < 2: return
    cx_sum = wp.vec3(0.0, 0.0, 0.0); qc_sum = wp.vec3(0.0, 0.0, 0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]; q = rest_offsets[start + j]
        cx_sum = cx_sum + w * p; qc_sum = qc_sum + w * q; tw = tw + w
    cx_sum_buf[cid] = cx_sum; qc_sum_buf[cid] = qc_sum; tw_buf[cid] = tw


@wp.kernel
def cluster_normalize_centroids(
    cluster_sizes: wp.array(dtype=wp.int32),
    cx_sum_buf: wp.array(dtype=wp.vec3), qc_sum_buf: wp.array(dtype=wp.vec3),
    tw_buf: wp.array(dtype=wp.float32),
    cx_mean_buf: wp.array(dtype=wp.vec3), qc_mean_buf: wp.array(dtype=wp.vec3),
):
    cid = wp.tid()
    size = cluster_sizes[cid]
    if size < 2:
        cx_mean_buf[cid] = wp.vec3(0.0, 0.0, 0.0)
        qc_mean_buf[cid] = wp.vec3(0.0, 0.0, 0.0); return
    tw = tw_buf[cid]
    tw_safe = wp.where(tw < 1.5, 1.0, tw)
    cx_mean_buf[cid] = cx_sum_buf[cid] / tw_safe
    qc_mean_buf[cid] = qc_sum_buf[cid] / tw_safe


@wp.kernel
def cluster_compute_covariance(
    pred_pos: wp.array(dtype=wp.vec3), alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32), cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32), rest_offsets: wp.array(dtype=wp.vec3),
    cx_mean_buf: wp.array(dtype=wp.vec3), qc_mean_buf: wp.array(dtype=wp.vec3),
    A_buf: wp.array(dtype=wp.mat33),
):
    cid = wp.tid()
    start = cluster_starts[cid]; size = cluster_sizes[cid]
    if size < 2:
        A_buf[cid] = wp.mat33(0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0); return
    cx_mean = cx_mean_buf[cid]; qc_mean = qc_mean_buf[cid]
    a00=float(0.0); a01=float(0.0); a02=float(0.0)
    a10=float(0.0); a11=float(0.0); a12=float(0.0)
    a20=float(0.0); a21=float(0.0); a22=float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        alive = wp.where(w_raw < 0.01, 0.0, 1.0)
        p = pred_pos[pid]
        px = p[0]-cx_mean[0]; py = p[1]-cx_mean[1]; pz = p[2]-cx_mean[2]
        q = rest_offsets[start + j]
        qx = q[0]-qc_mean[0]; qy = q[1]-qc_mean[1]; qz = q[2]-qc_mean[2]
        a00 = a00 + alive*px*qx; a01 = a01 + alive*px*qy; a02 = a02 + alive*px*qz
        a10 = a10 + alive*py*qx; a11 = a11 + alive*py*qy; a12 = a12 + alive*py*qz
        a20 = a20 + alive*pz*qx; a21 = a21 + alive*pz*qy; a22 = a22 + alive*pz*qz
    A_buf[cid] = wp.mat33(a00,a01,a02, a10,a11,a12, a20,a21,a22)


@wp.kernel
def cluster_polar_rotation(
    cluster_sizes: wp.array(dtype=wp.int32),
    A_buf: wp.array(dtype=wp.mat33),
    cluster_quat: wp.array(dtype=wp.quat),
    num_iter: int,
    R_buf: wp.array(dtype=wp.mat33),
):
    cid = wp.tid()
    size = cluster_sizes[cid]
    if size < 2:
        R_buf[cid] = wp.mat33(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0); return
    A = A_buf[cid]; q_warm = cluster_quat[cid]
    q_new = extract_rotation_muller(A, q_warm, num_iter)
    R_buf[cid] = wp.quat_to_matrix(q_new)


@wp.kernel
def cluster_goal_scatter_with_R(
    pred_pos: wp.array(dtype=wp.vec3), alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32), cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32), rest_offsets: wp.array(dtype=wp.vec3),
    cx_mean_buf: wp.array(dtype=wp.vec3), qc_mean_buf: wp.array(dtype=wp.vec3),
    R_buf: wp.array(dtype=wp.mat33),
    deltas: wp.array(dtype=wp.vec3), delta_counts: wp.array(dtype=wp.int32),
):
    cid = wp.tid()
    start = cluster_starts[cid]; size = cluster_sizes[cid]
    if size < 2: return
    cx_mean = cx_mean_buf[cid]; qc_mean = qc_mean_buf[cid]; R = R_buf[cid]
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        particle_alive = wp.where(w_raw < 0.01, 0.0, 1.0)
        q = rest_offsets[start + j]
        q_centered = wp.vec3(q[0]-qc_mean[0], q[1]-qc_mean[1], q[2]-qc_mean[2])
        goal = R * q_centered + cx_mean
        correction = (goal - pred_pos[pid]) * particle_alive
        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, int(particle_alive))


# ============== Loss kernel ==============

@wp.kernel
def loss_pos_mse_kernel(
    pos: wp.array(dtype=wp.vec3),
    target_pos: wp.array(dtype=wp.vec3),
    interior_mask: wp.array(dtype=wp.int32),
    inv_count: float,
    loss_out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    m = float(interior_mask[i])
    p = pos[i]; t = target_pos[i]
    diff = p - t
    sq = (diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]) * m * inv_count
    wp.atomic_add(loss_out, 0, sq)


# ============================================================================
#                     Substep driver with stiff_arr
# ============================================================================

def differentiable_substep_e2e(
    solver, knife_pos, stiffness_arr, dt_sub,
    enable_layer_a_collision: bool = True,
):
    N = solver.N; NC = solver.num_clusters
    solver._alloc_ssa_buffers()
    knife_active_collision = solver.knife_active if enable_layer_a_collision else 0
    num_iter = solver.num_iterations

    pred_pos_pool = [wp.empty(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
                      for _ in range(2*num_iter+2)]
    deltas_pool = [wp.zeros(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
                    for _ in range(num_iter)]
    delta_counts_pool = [wp.zeros(N, dtype=wp.int32, device=solver.device)
                          for _ in range(num_iter)]
    shape_delta_pool = [wp.zeros(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
                         for _ in range(num_iter)]
    coll_delta_pre = wp.zeros(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
    coll_delta_pool = [wp.zeros(N, dtype=wp.vec3, device=solver.device, requires_grad=True)
                        for _ in range(num_iter)]
    cx_sum_pool = [wp.zeros(NC, dtype=wp.vec3, device=solver.device, requires_grad=True)
                    for _ in range(num_iter)]
    qc_sum_pool = [wp.zeros(NC, dtype=wp.vec3, device=solver.device, requires_grad=True)
                    for _ in range(num_iter)]
    tw_pool = [wp.zeros(NC, dtype=wp.float32, device=solver.device, requires_grad=True)
                for _ in range(num_iter)]
    cx_mean_pool = [wp.zeros(NC, dtype=wp.vec3, device=solver.device, requires_grad=True)
                     for _ in range(num_iter)]
    qc_mean_pool = [wp.zeros(NC, dtype=wp.vec3, device=solver.device, requires_grad=True)
                     for _ in range(num_iter)]
    A_pool = [wp.zeros(NC, dtype=wp.mat33, device=solver.device, requires_grad=True)
                for _ in range(num_iter)]
    R_pool = [wp.zeros(NC, dtype=wp.mat33, device=solver.device, requires_grad=True)
                for _ in range(num_iter)]

    wp.launch(predict_kernel_ssa, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"], pred_pos_pool[0], solver.p["old_pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.gravity, dt_sub, solver.damping,
    ], device=solver.device)
    wp.launch(collision_kernel_ssa, dim=N, inputs=[
        pred_pos_pool[0], solver.p["pos"], solver._current_alpha(), solver.p["fixed"],
        solver.floor_y, solver.floor_friction,
        knife_pos, solver.knife_radius, knife_active_collision, coll_delta_pre,
    ], device=solver.device)
    wp.launch(apply_collision_delta_ssa_kernel, dim=N,
              inputs=[pred_pos_pool[0], coll_delta_pre],
              outputs=[pred_pos_pool[1]], device=solver.device)

    for k in range(num_iter):
        in_idx = 2*k+1; mid_idx = 2*k+2; out_idx = 2*k+3
        wp.launch(cluster_compute_sums, dim=NC, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            cx_sum_pool[k], qc_sum_pool[k], tw_pool[k],
        ], device=solver.device)
        wp.launch(cluster_normalize_centroids, dim=NC, inputs=[
            solver.c["sizes"], cx_sum_pool[k], qc_sum_pool[k], tw_pool[k],
            cx_mean_pool[k], qc_mean_pool[k],
        ], device=solver.device)
        wp.launch(cluster_compute_covariance, dim=NC, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            cx_mean_pool[k], qc_mean_pool[k], A_pool[k],
        ], device=solver.device)
        wp.launch(cluster_polar_rotation, dim=NC, inputs=[
            solver.c["sizes"], A_pool[k], solver.c["cluster_quat"],
            solver.muller_iter, R_pool[k],
        ], device=solver.device)
        wp.launch(cluster_goal_scatter_with_R, dim=NC, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            cx_mean_pool[k], qc_mean_pool[k], R_pool[k],
            deltas_pool[k], delta_counts_pool[k],
        ], device=solver.device)
        # ★ Use array-stiffness variant for autograd tracking
        wp.launch(apply_kernel_ssa_stiff_arr, dim=N, inputs=[
            deltas_pool[k], delta_counts_pool[k],
            solver.p["damage"], solver._current_alpha(), solver.p["fixed"],
            stiffness_arr,    # ← tracked
            shape_delta_pool[k],
        ], device=solver.device)
        wp.launch(apply_shape_delta_ssa_kernel, dim=N,
                  inputs=[pred_pos_pool[in_idx], shape_delta_pool[k]],
                  outputs=[pred_pos_pool[mid_idx]], device=solver.device)
        wp.launch(collision_kernel_ssa, dim=N, inputs=[
            pred_pos_pool[mid_idx], solver.p["pos"],
            solver._current_alpha(), solver.p["fixed"],
            solver.floor_y, solver.floor_friction,
            knife_pos, solver.knife_radius, knife_active_collision,
            coll_delta_pool[k],
        ], device=solver.device)
        wp.launch(apply_collision_delta_ssa_kernel, dim=N,
                  inputs=[pred_pos_pool[mid_idx], coll_delta_pool[k]],
                  outputs=[pred_pos_pool[out_idx]], device=solver.device)

    final_pred_pos = pred_pos_pool[2*num_iter+1]
    inv_dt = 1.0 / dt_sub
    wp.launch(update_velocity_kernel, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"], final_pred_pos, solver.p["old_pos"],
        solver._current_alpha(), solver.p["fixed"], inv_dt,
    ], device=solver.device)
    solver.step_count += 1


# ============================================================================
#                     Sys-ID driver
# ============================================================================

def generate_target_pos(env, gt_stiffness, knife_pos_np, dt_sub):
    """Run sim with ground-truth stiffness and capture final pos."""
    snap = snap_solver_state(env)
    env.solver.step_count = 0
    stiff_gt = wp.array([gt_stiffness], dtype=wp.float32, device=env.device,
                         requires_grad=False)
    knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                          device=env.device, requires_grad=False)
    differentiable_substep_e2e(env.solver, knife_pos, stiff_gt, dt_sub,
                                enable_layer_a_collision=True)
    target_pos = env.solver.p["pos"].numpy().copy()
    reset_solver_state(env, snap)
    env.solver.step_count = 0
    return target_pos, snap


def main():
    wp.init()
    print("=" * 78)
    print("  Phase 2B end-to-end: stiffness sys-ID via Adam through Layer A")
    print("=" * 78)

    env = setup_env()
    env.solver.num_iterations = 6
    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)
    inv_count = 1.0 / interior_count
    knife_pos_np = (0.85, 0.5, 0.12)
    dt_sub = env.solver.dt / float(env.solver.substeps)

    GT_STIFFNESS = 0.85
    INIT_STIFFNESS = 0.50
    LR = 5.0
    NUM_STEPS = 30

    print(f"  GT stiffness = {GT_STIFFNESS}")
    print(f"  INIT stiffness = {INIT_STIFFNESS}")
    print(f"  Learning rate = {LR}, Num steps = {NUM_STEPS}")

    # Step 1: generate target (ground-truth) final pos
    print("\n  Generating target pos with GT stiffness...")
    target_pos_np, snap = generate_target_pos(env, GT_STIFFNESS, knife_pos_np, dt_sub)
    target_pos = wp.array(target_pos_np, dtype=wp.vec3, device=env.device)
    print(f"  Target pos[0]: {target_pos_np[0]}")

    # Step 2: Adam-style optimization on stiffness
    current_stiff = float(INIT_STIFFNESS)
    history = []

    # Adam state
    m_stiff = 0.0
    v_stiff = 0.0
    beta1 = 0.9; beta2 = 0.999; eps_adam = 1e-8

    print("\n  Optimization loop:")
    print(f"  {'step':>5} {'stiff':>8} {'loss':>12} {'grad':>12}")

    for step in range(NUM_STEPS):
        reset_solver_state(env, snap)
        env.solver.step_count = 0

        stiff_arr = wp.array([current_stiff], dtype=wp.float32, device=env.device,
                              requires_grad=True)
        knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                              device=env.device, requires_grad=False)
        loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)

        with wp.Tape() as tape:
            differentiable_substep_e2e(env.solver, knife_pos, stiff_arr, dt_sub,
                                        enable_layer_a_collision=True)
            wp.launch(loss_pos_mse_kernel, dim=env.solver.N, inputs=[
                env.solver.p["pos"], target_pos, interior_mask_wp,
                inv_count, loss_buf,
            ], device=env.device)
        tape.backward(loss=loss_buf)
        wp.synchronize()

        loss_val = float(loss_buf.numpy()[0])
        grad_val = float(stiff_arr.grad.numpy()[0])

        # Adam update
        m_stiff = beta1 * m_stiff + (1 - beta1) * grad_val
        v_stiff = beta2 * v_stiff + (1 - beta2) * grad_val * grad_val
        m_hat = m_stiff / (1 - beta1 ** (step + 1))
        v_hat = v_stiff / (1 - beta2 ** (step + 1))
        new_stiff = current_stiff - LR * m_hat / (np.sqrt(v_hat) + eps_adam)

        # Clip to (0.05, 1.0)
        new_stiff = float(np.clip(new_stiff, 0.05, 1.0))

        history.append({
            'step': step, 'stiff': current_stiff, 'loss': loss_val, 'grad': grad_val,
        })
        if step < 10 or step % 5 == 0 or step == NUM_STEPS - 1:
            print(f"  {step:>5} {current_stiff:>8.4f} {loss_val:>12.4e} {grad_val:>+12.4e}")

        current_stiff = new_stiff

    # Final eval
    reset_solver_state(env, snap)
    env.solver.step_count = 0
    stiff_arr = wp.array([current_stiff], dtype=wp.float32, device=env.device,
                          requires_grad=False)
    knife_pos = wp.array([wp.vec3(*knife_pos_np)], dtype=wp.vec3,
                          device=env.device, requires_grad=False)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device)
    differentiable_substep_e2e(env.solver, knife_pos, stiff_arr, dt_sub,
                                enable_layer_a_collision=True)
    wp.launch(loss_pos_mse_kernel, dim=env.solver.N, inputs=[
        env.solver.p["pos"], target_pos, interior_mask_wp, inv_count, loss_buf,
    ], device=env.device)
    wp.synchronize()
    final_loss = float(loss_buf.numpy()[0])

    print()
    print("=" * 78)
    print("  RESULT")
    print("=" * 78)
    initial_loss = history[0]['loss']
    print(f"  GT stiffness:         {GT_STIFFNESS:.4f}")
    print(f"  INIT stiffness:       {INIT_STIFFNESS:.4f}")
    print(f"  FINAL stiffness:      {current_stiff:.4f}")
    print(f"  Recovery error:       {abs(current_stiff - GT_STIFFNESS):.4f}  ({100*abs(current_stiff - GT_STIFFNESS)/GT_STIFFNESS:.2f}%)")
    print(f"  Initial loss:         {initial_loss:.4e}")
    print(f"  Final loss:           {final_loss:.4e}")
    print(f"  Loss reduction:       {initial_loss / max(final_loss, 1e-20):.2f}×")
    print()

    err_pct = 100 * abs(current_stiff - GT_STIFFNESS) / GT_STIFFNESS
    if err_pct < 5.0 and final_loss < initial_loss * 0.1:
        print(f"  🎯🎯🎯 SYS-ID PASS — recovered to {err_pct:.1f}% with {initial_loss/final_loss:.0f}× loss drop")
        print(f"     End-to-end gradient-based sys-ID through Layer A is USABLE.")
    elif err_pct < 15.0 and final_loss < initial_loss * 0.5:
        print(f"  ✅ partial recovery — {err_pct:.1f}% (works but slow)")
    else:
        print(f"  ❌ FAIL — {err_pct:.1f}% error, loss reduction {initial_loss/final_loss:.2f}×")

    return 0


if __name__ == "__main__":
    sys.exit(main())
