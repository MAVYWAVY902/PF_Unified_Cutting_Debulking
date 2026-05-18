#!/usr/bin/env python3
"""3D bulk material sys-ID — joint Adam on {stiffness, damping, friction}.

Mirrors REAL Stage 2.1 phantom workflow:
  - Knife trajectory: GIVEN (robot encoder)
  - Initial mesh: GIVEN
  - UNKNOWN: stiffness, damping, friction (3 dim)
  - Cutting: NOT triggered (Stage 2.1 is pre-cut bulk contact phase)

Test:
  - GT: stiffness=0.85, damping=0.10, friction=0.95
  - INITs: vary all 3 params widely
  - Pure single Adam, joint 3D
  - Pass: all params recover within 5%
"""
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

wp.config.verify_autograd_array_access = False

from pf_shape_matching_warp.sim.kernels import (
    apply_shape_delta_ssa_kernel, apply_collision_delta_ssa_kernel,
    update_velocity_kernel,
    cluster_compute_sums_kernel_ssa, cluster_normalize_centroids_kernel_ssa,
    cluster_compute_covariance_kernel_ssa, cluster_polar_rotation_kernel_ssa,
    cluster_goal_scatter_kernel_ssa,
)
from path4_falsification_test_H_isolate import setup_env, loss_sum_pos_z_kernel
from path4_poc_v1_dual_purpose import snap_solver_state, reset_solver_state
from test_layer_a_phase2b_e2e_stiffness_sysid import apply_kernel_ssa_stiff_arr


# ============================================================================
#                  Array-version kernels (inline, test-only)
# ============================================================================

@wp.kernel
def predict_kernel_ssa_damping_arr(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    pred_pos: wp.array(dtype=wp.vec3),
    old_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    gravity: wp.vec3,
    dt: float,
    damping_arr: wp.array(dtype=wp.float32),
):
    """SSA predict with damping as wp.array (1-element) for autograd tracking."""
    i = wp.tid()
    a = alpha[i]
    f = fixed[i]
    is_active = wp.where(a < 0.01, 0.0, 1.0) * wp.where(f == 1, 0.0, 1.0)

    damping = damping_arr[0]
    p = pos[i]
    v = vel[i] + gravity * dt
    v_damped = v * damping
    new_pred = p + v_damped * dt
    new_old = p

    pred_pos[i] = is_active * new_pred + (1.0 - is_active) * p
    old_pos[i] = new_old


@wp.kernel
def collision_kernel_ssa_friction_arr(
    pred_pos_in: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    floor_y: float,
    friction_arr: wp.array(dtype=wp.float32),
    knife_pos: wp.array(dtype=wp.vec3),
    knife_radius: float,
    knife_active: int,
    coll_delta: wp.array(dtype=wp.vec3),
):
    """SSA collision with floor_friction as wp.array (1-elem)."""
    i = wp.tid()
    a = alpha[i]
    f = fixed[i]
    is_active = wp.where(a < 0.01, 0.0, 1.0) * wp.where(f == 1, 0.0, 1.0)

    friction = friction_arr[0]
    p = pred_pos_in[i]
    delta = wp.vec3(0.0, 0.0, 0.0)

    # Floor (Z-up)
    if p[2] < floor_y:
        # tangential friction
        anchor = pos_anchor[i]
        tang = wp.vec3(p[0] - anchor[0], p[1] - anchor[1], 0.0)
        delta = delta + wp.vec3(0.0, 0.0, floor_y - p[2]) - tang * friction

    # Knife collision
    if knife_active == 1:
        kp = knife_pos[0]
        d = p - kp
        dist_sq = wp.dot(d, d)
        dist = wp.sqrt(dist_sq + 1.0e-8)
        if dist < knife_radius:
            push_dir = d / (dist + 1.0e-8)
            delta = delta + push_dir * (knife_radius - dist)

    coll_delta[i] = is_active * delta


# ============================================================================
#                  Custom 3D substep wrapper
# ============================================================================

def diff_substep_3d_bulk(solver, knife_pos, stiffness_arr, damping_arr, friction_arr,
                          dt_sub):
    """Substep with stiffness, damping, friction as wp.arrays (3D autograd)."""
    N = solver.N
    NC = solver.num_clusters
    num_iter = solver.num_iterations
    solver.p["pred_pos"] = wp.empty_like(solver.p["pos"])

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

    # Predict (with damping_arr)
    wp.launch(predict_kernel_ssa_damping_arr, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"], pred_pos_pool[0], solver.p["old_pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.gravity, dt_sub, damping_arr,
    ], device=solver.device)

    # Pre-loop collision (with friction_arr)
    wp.launch(collision_kernel_ssa_friction_arr, dim=N, inputs=[
        pred_pos_pool[0], solver.p["pos"], solver._current_alpha(), solver.p["fixed"],
        solver.floor_y, friction_arr,
        knife_pos, solver.knife_radius, solver.knife_active,
        coll_delta_pre,
    ], device=solver.device)
    wp.launch(apply_collision_delta_ssa_kernel, dim=N,
              inputs=[pred_pos_pool[0], coll_delta_pre],
              outputs=[pred_pos_pool[1]], device=solver.device)

    # Jacobi loop with stiffness_arr
    for k in range(num_iter):
        in_idx = 2*k+1; mid_idx = 2*k+2; out_idx = 2*k+3
        wp.launch(cluster_compute_sums_kernel_ssa, dim=NC, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            cx_sum_pool[k], qc_sum_pool[k], tw_pool[k],
        ], device=solver.device)
        wp.launch(cluster_normalize_centroids_kernel_ssa, dim=NC, inputs=[
            solver.c["sizes"], cx_sum_pool[k], qc_sum_pool[k], tw_pool[k],
            cx_mean_pool[k], qc_mean_pool[k],
        ], device=solver.device)
        wp.launch(cluster_compute_covariance_kernel_ssa, dim=NC, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            cx_mean_pool[k], qc_mean_pool[k], A_pool[k],
        ], device=solver.device)
        wp.launch(cluster_polar_rotation_kernel_ssa, dim=NC, inputs=[
            solver.c["sizes"], A_pool[k], solver.c["cluster_quat"],
            solver.muller_iter, R_pool[k],
        ], device=solver.device)
        wp.launch(cluster_goal_scatter_kernel_ssa, dim=NC, inputs=[
            pred_pos_pool[in_idx], solver._current_alpha(),
            solver.c["starts"], solver.c["sizes"],
            solver.c["particle_ids"], solver.c["rest_offsets"],
            cx_mean_pool[k], qc_mean_pool[k], R_pool[k],
            deltas_pool[k], delta_counts_pool[k],
        ], device=solver.device)
        wp.launch(apply_kernel_ssa_stiff_arr, dim=N, inputs=[
            deltas_pool[k], delta_counts_pool[k],
            solver.p["damage"], solver._current_alpha(), solver.p["fixed"],
            stiffness_arr, shape_delta_pool[k],
        ], device=solver.device)
        wp.launch(apply_shape_delta_ssa_kernel, dim=N,
                  inputs=[pred_pos_pool[in_idx], shape_delta_pool[k]],
                  outputs=[pred_pos_pool[mid_idx]], device=solver.device)
        wp.launch(collision_kernel_ssa_friction_arr, dim=N, inputs=[
            pred_pos_pool[mid_idx], solver.p["pos"],
            solver._current_alpha(), solver.p["fixed"],
            solver.floor_y, friction_arr,
            knife_pos, solver.knife_radius, solver.knife_active,
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
#                  Adam + sys-ID driver
# ============================================================================

class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = 0.0; self.v = 0.0; self.t = 0
    def step(self, p, grad):
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*grad
        self.v = self.b2*self.v + (1-self.b2)*grad*grad
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        return p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def calibrate_3d(env, init, GT, target_pos, state_snap, dt_sub,
                  interior_mask_wp, interior_count, num_iters=100):
    """3D Adam optimization on (stiffness, damping, friction)."""
    s_curr, d_curr, f_curr = init
    opt_s = Adam(lr=0.05); opt_d = Adam(lr=0.01); opt_f = Adam(lr=0.05)
    inv_count = 1.0 / interior_count
    GT_KP = np.array([0.85, 0.5, 0.12])

    history = []
    for it in range(num_iters):
        reset_solver_state(env, state_snap)
        env.solver.step_count = 0
        knife_pos_arr = wp.array([wp.vec3(*GT_KP)], dtype=wp.vec3,
                                   device=env.device, requires_grad=False)
        s_arr = wp.array([s_curr], dtype=wp.float32, device=env.device, requires_grad=True)
        d_arr = wp.array([d_curr], dtype=wp.float32, device=env.device, requires_grad=True)
        f_arr = wp.array([f_curr], dtype=wp.float32, device=env.device, requires_grad=True)
        loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)

        with wp.Tape() as tape:
            diff_substep_3d_bulk(env.solver, knife_pos_arr, s_arr, d_arr, f_arr, dt_sub)
            # Loss = MSE on interior pos (compared to target)
            from test_layer_a_phase2b_e2e_stiffness_sysid import loss_pos_mse_kernel
            wp.launch(loss_pos_mse_kernel, dim=env.solver.N,
                      inputs=[env.solver.p["pos"], target_pos, interior_mask_wp,
                              inv_count, loss_buf], device=env.device)
        tape.backward(loss=loss_buf)
        wp.synchronize()

        loss_val = float(loss_buf.numpy()[0])
        gs = float(s_arr.grad.numpy()[0])
        gd = float(d_arr.grad.numpy()[0])
        gf = float(f_arr.grad.numpy()[0])
        history.append((s_curr, d_curr, f_curr, loss_val))

        s_curr = float(np.clip(opt_s.step(s_curr, gs), 0.05, 0.99))
        d_curr = float(np.clip(opt_d.step(d_curr, gd), 0.01, 0.99))
        f_curr = float(np.clip(opt_f.step(f_curr, gf), 0.05, 1.5))

    return dict(stiff=s_curr, damp=d_curr, fric=f_curr,
                final_loss=history[-1][3], init_loss=history[0][3])


def main():
    wp.init()
    print("=" * 78)
    print("  3D BULK MATERIAL SYS-ID — Joint Adam on {stiffness, damping, friction}")
    print("=" * 78)

    env = setup_env()
    env.solver.num_iterations = 6
    env.solver.substeps = 1
    env.solver.use_split_cluster_solve = True
    env.solver.use_ssa_soft_removal = True
    env.solver.cutting_update_interval = 999999
    env.solver.knife_active = 1

    GT_S, GT_D, GT_F = 0.85, 0.10, 0.95
    GT_KP = np.array([0.85, 0.5, 0.12])

    state_snap = snap_solver_state(env)
    interior_mask_wp = env._interior_mask
    interior_count = float(env._total_interior)
    dt_sub = env.solver.dt / float(env.solver.substeps)

    # Generate target at GT
    env.solver.step_count = 0
    knife_gt = wp.array([wp.vec3(*GT_KP)], dtype=wp.vec3, device=env.device)
    s_gt = wp.array([GT_S], dtype=wp.float32, device=env.device)
    d_gt = wp.array([GT_D], dtype=wp.float32, device=env.device)
    f_gt = wp.array([GT_F], dtype=wp.float32, device=env.device)
    diff_substep_3d_bulk(env.solver, knife_gt, s_gt, d_gt, f_gt, dt_sub)
    target_pos = wp.array(env.solver.p["pos"].numpy().copy(),
                            dtype=wp.vec3, device=env.device)

    print(f"  GT: stiff={GT_S}, damping={GT_D}, friction={GT_F}")
    print()

    # Sweep INITs
    INITS = [
        (0.50, 0.50, 0.50),    # all 30-50% off
        (0.30, 0.30, 0.70),    # other side
        (0.70, 0.05, 0.80),    # damping near zero
        (0.95, 0.20, 1.20),    # all higher than GT
        (0.40, 0.50, 0.50),    # mix
    ]
    print(f"  {'INIT':>30} {'final_s':>9} {'final_d':>9} {'final_f':>9} "
          f"{'s_err':>7} {'d_err':>9} {'f_err':>7} {'verdict':>8}")
    results = []
    for init in INITS:
        t0 = time.time()
        res = calibrate_3d(env, init, (GT_S, GT_D, GT_F),
                            target_pos, state_snap, dt_sub,
                            interior_mask_wp, interior_count, num_iters=100)
        t = time.time() - t0
        s_err = (res["stiff"] - GT_S) / GT_S * 100
        d_err = (res["damp"] - GT_D) / GT_D * 100
        f_err = (res["fric"] - GT_F) / GT_F * 100
        max_err = max(abs(s_err), abs(d_err), abs(f_err))
        verdict = "✅ <2%" if max_err < 2 else ("✅ <5%" if max_err < 5
                                                   else ("⚠️ <15%" if max_err < 15 else "❌"))
        print(f"  {str(init):>30} {res['stiff']:>9.4f} {res['damp']:>9.4f} {res['fric']:>9.4f} "
              f"{s_err:>+6.1f}% {d_err:>+8.1f}% {f_err:>+6.1f}% {verdict:>8}")
        results.append((init, res, s_err, d_err, f_err, t))

    print()
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)
    all_pass = all(max(abs(s),abs(d),abs(f)) < 5.0 for _,_,s,d,f,_ in results)
    print(f"  All 5 INITs converge to <5% per param: {'✅ YES' if all_pass else '⚠️ NO'}")
    if all_pass:
        print(f"  → 3D bulk joint Adam works (single Adam, no multi-restart)")
    else:
        print(f"  → Some params fail to recover at <5%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
