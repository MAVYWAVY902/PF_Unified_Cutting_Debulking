#!/usr/bin/env python3
"""2D cutting sys-ID — joint Adam on {alpha_min, sharpness}.

Mirrors REAL Stage 2.2 phantom workflow:
  - Bulk material params (stiffness, damping, friction): FIXED from Stage 2.1
  - Knife trajectory: GIVEN (robot encoder)
  - UNKNOWN: alpha_min (cutting threshold), sharpness (cutting steepness)

Test:
  - GT: alpha_min=50, sharpness=30, threshold=0.4 (fixed)
  - 5 substeps with cutting fire each → damage accumulates → alpha varies
  - Loss = MSE on alpha buffer (direct observation, bypasses update_velocity mask)
  - 2D Adam, no multi-restart
  - Pass: both params recover within 5%
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
    predict_kernel_ssa, collision_kernel_ssa,
    apply_shape_delta_ssa_kernel, apply_collision_delta_ssa_kernel,
    update_velocity_kernel_ssa,
    cluster_compute_sums_kernel_ssa, cluster_normalize_centroids_kernel_ssa,
    cluster_compute_covariance_kernel_ssa, cluster_polar_rotation_kernel_ssa,
    cluster_goal_scatter_kernel_ssa, apply_kernel_ssa,
)
from pf_shape_matching_warp.sim.pf_damage import (
    update_edge_damage_kernel_ssa_param_arr,
    apply_edge_damage_ssa_kernel,
    update_particle_damage_kernel,
    finalize_particle_damage_kernel_ssa,
    apply_particle_damage_ssa_kernel,
    update_soft_removal_kernel_ssa_param_arr,
)
from path4_falsification_test_H_isolate import setup_env
from path4_poc_v1_dual_purpose import snap_solver_state, reset_solver_state


@wp.kernel
def loss_alpha_mse_kernel(
    alpha: wp.array(dtype=wp.float32),
    target_alpha: wp.array(dtype=wp.float32),
    interior_mask: wp.array(dtype=wp.int32),
    inv_count: float,
    loss_out: wp.array(dtype=wp.float32),
):
    """Direct MSE on alpha buffer, bypasses update_velocity sigmoid saturation."""
    i = wp.tid()
    m = float(interior_mask[i])
    diff = alpha[i] - target_alpha[i]
    sq = diff * diff * m * inv_count
    wp.atomic_add(loss_out, 0, sq)


def diff_substep_with_cutting(solver, knife_pos, alpha_min_arr, sharpness_arr,
                                threshold_arr, knife_radius_arr, dt_sub):
    """Run 1 substep including cutting pipeline with all cutting params as wp.arrays."""
    N = solver.N
    NC = solver.num_clusters
    num_iter = solver.num_iterations
    solver._alloc_ssa_buffers()
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

    # 1. Predict
    wp.launch(predict_kernel_ssa, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"], pred_pos_pool[0], solver.p["old_pos"],
        solver._current_alpha(), solver.p["fixed"],
        solver.gravity, dt_sub, solver.damping,
    ], device=solver.device)
    # 2. Pre-loop collision
    wp.launch(collision_kernel_ssa, dim=N, inputs=[
        pred_pos_pool[0], solver.p["pos"], solver._current_alpha(), solver.p["fixed"],
        solver.floor_y, solver.floor_friction,
        knife_pos, solver.knife_radius, solver.knife_active, coll_delta_pre,
    ], device=solver.device)
    wp.launch(apply_collision_delta_ssa_kernel, dim=N,
              inputs=[pred_pos_pool[0], coll_delta_pre],
              outputs=[pred_pos_pool[1]], device=solver.device)
    # 3. Jacobi loop
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
        wp.launch(apply_kernel_ssa, dim=N, inputs=[
            deltas_pool[k], delta_counts_pool[k],
            solver.p["damage"], solver._current_alpha(), solver.p["fixed"],
            solver.stiffness, shape_delta_pool[k],
        ], device=solver.device)
        wp.launch(apply_shape_delta_ssa_kernel, dim=N,
                  inputs=[pred_pos_pool[in_idx], shape_delta_pool[k]],
                  outputs=[pred_pos_pool[mid_idx]], device=solver.device)
        wp.launch(collision_kernel_ssa, dim=N, inputs=[
            pred_pos_pool[mid_idx], solver.p["pos"],
            solver._current_alpha(), solver.p["fixed"],
            solver.floor_y, solver.floor_friction,
            knife_pos, solver.knife_radius, solver.knife_active,
            coll_delta_pool[k],
        ], device=solver.device)
        wp.launch(apply_collision_delta_ssa_kernel, dim=N,
                  inputs=[pred_pos_pool[mid_idx], coll_delta_pool[k]],
                  outputs=[pred_pos_pool[out_idx]], device=solver.device)

    final_pred_pos = pred_pos_pool[2*num_iter+1]

    # 4. Cutting (with param_arr variants)
    c = solver._ssa_cycle_index
    if c + 1 >= solver.ssa_buffer_capacity:
        raise RuntimeError(f"SSA buffer cycle {c+1} >= capacity")
    alpha_prev = solver._ssa_alpha_buffers[c]
    alpha_next = solver._ssa_alpha_buffers[c+1]
    ed_prev = solver._ssa_edge_damage_buffers[c]
    ed_next = solver._ssa_edge_damage_buffers[c+1]
    pd_prev = solver._ssa_particle_damage_buffers[c]
    pd_next = solver._ssa_particle_damage_buffers[c+1]
    num_edges = solver.e["num_edges"]

    edge_damage_delta = wp.zeros(num_edges, dtype=wp.float32,
                                  device=solver.device, requires_grad=True)
    wp.launch(update_edge_damage_kernel_ssa_param_arr, dim=num_edges, inputs=[
        final_pred_pos, alpha_prev,
        solver.e["a"], solver.e["b"],
        solver.e["pf_rest"], ed_prev,
        knife_pos, knife_radius_arr, alpha_min_arr,
        edge_damage_delta,
    ], device=solver.device)
    wp.launch(apply_edge_damage_ssa_kernel, dim=num_edges, inputs=[
        ed_prev, edge_damage_delta, ed_next,
    ], device=solver.device)

    damage_sum = wp.zeros(N, dtype=wp.float32, device=solver.device, requires_grad=True)
    damage_count = wp.zeros(N, dtype=wp.float32, device=solver.device, requires_grad=True)
    wp.launch(update_particle_damage_kernel, dim=num_edges, inputs=[
        damage_sum, damage_count, alpha_prev,
        solver.e["a"], solver.e["b"], ed_next,
    ], device=solver.device)

    particle_damage_delta = wp.zeros(N, dtype=wp.float32,
                                      device=solver.device, requires_grad=True)
    wp.launch(finalize_particle_damage_kernel_ssa, dim=N, inputs=[
        damage_sum, damage_count, pd_prev, particle_damage_delta,
    ], device=solver.device)
    wp.launch(apply_particle_damage_ssa_kernel, dim=N, inputs=[
        pd_prev, particle_damage_delta, pd_next,
    ], device=solver.device)
    wp.launch(update_soft_removal_kernel_ssa_param_arr, dim=N, inputs=[
        pd_next, alpha_prev, solver.p["fixed"],
        threshold_arr, sharpness_arr, alpha_next,
    ], device=solver.device)
    solver._ssa_cycle_index = c + 1

    # 5. Update velocity
    new_pos = wp.empty_like(solver.p["pos"])
    new_vel = wp.empty_like(solver.p["vel"])
    inv_dt = 1.0 / dt_sub
    wp.launch(update_velocity_kernel_ssa, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"], final_pred_pos, solver.p["old_pos"],
        alpha_next, solver.p["fixed"], inv_dt,
    ], outputs=[new_pos, new_vel], device=solver.device)
    solver.p["pos"] = new_pos
    solver.p["vel"] = new_vel
    solver.step_count += 1
    return alpha_next   # final alpha after this substep+cutting


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


def reset_for_cutting(env, state_snap):
    """Reset all state including alpha buffers to fresh."""
    reset_solver_state(env, state_snap)
    env.solver.step_count = 0
    env.solver._ssa_cycle_index = 0
    # alpha/edge_damage/particle_damage all reset to defaults via reset_solver_state
    # (it fills alpha=1.0, ed/pd=0.0)


def gen_target_cutting_alpha(env, GT_AM, GT_SH, GT_TH, GT_KR, GT_KP,
                                state_snap, dt_sub, N_cut=5):
    """Run sim with GT cutting params, return final alpha state."""
    reset_for_cutting(env, state_snap)
    knife_pos = wp.array([wp.vec3(*GT_KP)], dtype=wp.vec3, device=env.device)
    am_arr = wp.array([GT_AM], dtype=wp.float32, device=env.device)
    sh_arr = wp.array([GT_SH], dtype=wp.float32, device=env.device)
    th_arr = wp.array([GT_TH], dtype=wp.float32, device=env.device)
    kr_arr = wp.array([GT_KR], dtype=wp.float32, device=env.device)
    for _ in range(N_cut):
        diff_substep_with_cutting(env.solver, knife_pos, am_arr, sh_arr,
                                    th_arr, kr_arr, dt_sub)
    final_alpha = env.solver._ssa_alpha_buffers[env.solver._ssa_cycle_index].numpy().copy()
    print(f"    target alpha range: [{final_alpha.min():.4f}, {final_alpha.max():.4f}], "
          f"frac<0.5: {(final_alpha<0.5).mean():.2%}")
    target_alpha = wp.array(final_alpha, dtype=wp.float32, device=env.device)
    return target_alpha


def calibrate_cutting(env, init_am, init_sh, GT_TH, GT_KR, GT_KP,
                       target_alpha, state_snap, dt_sub,
                       interior_mask_wp_int, interior_count, num_iters=60, N_cut=5):
    am_curr = float(init_am); sh_curr = float(init_sh)
    opt_am = Adam(lr=1.0); opt_sh = Adam(lr=0.5)
    inv_count = 1.0 / interior_count

    history = []
    for it in range(num_iters):
        reset_for_cutting(env, state_snap)
        knife_pos = wp.array([wp.vec3(*GT_KP)], dtype=wp.vec3, device=env.device)
        am_arr = wp.array([am_curr], dtype=wp.float32, device=env.device, requires_grad=True)
        sh_arr = wp.array([sh_curr], dtype=wp.float32, device=env.device, requires_grad=True)
        th_arr = wp.array([GT_TH], dtype=wp.float32, device=env.device, requires_grad=False)
        kr_arr = wp.array([GT_KR], dtype=wp.float32, device=env.device, requires_grad=False)
        loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)

        with wp.Tape() as tape:
            for _ in range(N_cut):
                final_alpha = diff_substep_with_cutting(
                    env.solver, knife_pos, am_arr, sh_arr, th_arr, kr_arr, dt_sub)
            wp.launch(loss_alpha_mse_kernel, dim=env.solver.N,
                      inputs=[final_alpha, target_alpha, interior_mask_wp_int,
                              inv_count, loss_buf], device=env.device)
        tape.backward(loss=loss_buf)
        wp.synchronize()

        loss_val = float(loss_buf.numpy()[0])
        gam = float(am_arr.grad.numpy()[0])
        gsh = float(sh_arr.grad.numpy()[0])
        history.append((am_curr, sh_curr, loss_val))

        am_curr = float(np.clip(opt_am.step(am_curr, gam), 1.0, 200.0))
        sh_curr = float(np.clip(opt_sh.step(sh_curr, gsh), 5.0, 100.0))

    return dict(alpha_min=am_curr, sharpness=sh_curr,
                final_loss=history[-1][2], init_loss=history[0][2])


def main():
    wp.init()
    print("=" * 78)
    print("  2D CUTTING SYS-ID — Joint Adam on {alpha_min, sharpness}")
    print("=" * 78)

    env = setup_env()
    env.solver.num_iterations = 6
    env.solver.substeps = 1
    env.solver.use_split_cluster_solve = True
    env.solver.use_ssa_soft_removal = True
    env.solver.knife_active = 1

    GT_AM, GT_SH = 50.0, 30.0
    GT_TH, GT_KR = 0.4, 0.10           # ★ larger radius (was 0.05)
    GT_KP = np.array([0.5, 0.5, 0.05])  # ★ deeper into mesh (was 0.85, 0.5, 0.12)
    N_CUT = 15                          # ★ more cycles (was 5)

    state_snap = snap_solver_state(env)
    interior_mask_wp_int = env._interior_mask
    interior_count = float(env._total_interior)
    dt_sub = env.solver.dt / float(env.solver.substeps)

    print(f"  GT: alpha_min={GT_AM}, sharpness={GT_SH}")
    print(f"  Cutting horizon: {N_CUT} substeps with cutting fired each")
    print()
    print("  Generating target alpha state at GT...")
    target_alpha = gen_target_cutting_alpha(env, GT_AM, GT_SH, GT_TH, GT_KR, GT_KP,
                                              state_snap, dt_sub, N_cut=N_CUT)
    print()

    INITS = [
        (20.0, 15.0),
        (30.0, 20.0),
        (50.0, 30.0),  # at GT
        (80.0, 50.0),
        (100.0, 60.0),
    ]
    print(f"  {'INIT (am,sh)':>20} {'final_am':>10} {'final_sh':>10} "
          f"{'am_err':>9} {'sh_err':>9} {'time':>7} {'verdict':>8}")
    results = []
    for init in INITS:
        t0 = time.time()
        try:
            res = calibrate_cutting(env, init[0], init[1], GT_TH, GT_KR, GT_KP,
                                      target_alpha, state_snap, dt_sub,
                                      interior_mask_wp_int, interior_count,
                                      num_iters=60, N_cut=N_CUT)
            t = time.time() - t0
            am_err = (res["alpha_min"] - GT_AM) / GT_AM * 100
            sh_err = (res["sharpness"] - GT_SH) / GT_SH * 100
            max_err = max(abs(am_err), abs(sh_err))
            verdict = "✅ <2%" if max_err < 2 else (
                "✅ <5%" if max_err < 5 else (
                "⚠️ <15%" if max_err < 15 else "❌"))
            print(f"  {str(init):>20} {res['alpha_min']:>10.3f} {res['sharpness']:>10.3f} "
                  f"{am_err:>+8.1f}% {sh_err:>+8.1f}% {t:>6.1f}s {verdict:>8}")
            results.append((init, res, am_err, sh_err))
        except Exception as e:
            print(f"  {str(init):>20} ERROR: {type(e).__name__}: {e}")

    print()
    print("=" * 78)
    print("  VERDICT (2D cutting)")
    print("=" * 78)
    valid = [r for r in results if r is not None]
    all_pass = all(max(abs(am),abs(sh)) < 5.0 for _,_,am,sh in valid) if valid else False
    print(f"  All INITs <5%: {'✅ YES' if all_pass else '⚠️ NO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
