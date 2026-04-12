"""Diagnose exactly which kernel produces NaN gradients."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pf_shape_matching_warp"))

import warp as wp
import numpy as np
wp.init()

from warp_sim.envs.debulk_env import DebulkEnv

cfg = {
    'mesh': 'resources/cube/cube_dense.msh',
    'settle_steps': 100, 'horizon_len': 32, 'sim_substeps': 1,
    'action_scale': 0.005, 'knife_start': [0.5, 0.5, 0.5],
    'knife_radius': 0.15, 'cutting_interval': 1,
    'lambda_nerve': 5.0, 'dt': 0.005, 'num_iterations': 4,
    'stiffness': 0.6, 'damping': 0.98,
}

env = DebulkEnv(cfg)
env.reset()
device = "cuda:0"
solver = env.solver
N = solver.N

# ============================================================
# Test A: Just apply_action + reward (NO physics at all)
# ============================================================
print("=" * 60)
print("Test A: apply_action → reward (skip physics)")

from warp_sim.kernels.obs_reward import (
    apply_action_kernel, compute_tumor_removal_kernel,
    compute_nerve_alpha_kernel, compute_reward_rewarped_kernel,
)

env.reset()
tape = wp.Tape()
with tape:
    action = wp.array([wp.vec3(0.0, -0.5, 0.0)], dtype=wp.vec3, device=device, requires_grad=True)
    new_kp = wp.zeros(1, dtype=wp.vec3, device=device, requires_grad=True)
    wp.launch(apply_action_kernel, dim=1, inputs=[solver.knife_pos, action, 0.005], outputs=[new_kp], device=device)

    # Compute reward directly from current alpha (no physics step)
    ts = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    tc = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    ns = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    nc = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(compute_tumor_removal_kernel, dim=N, inputs=[solver.p["alpha"], solver.p["fixed"], env.nerve_mask, ts, tc], device=device)
    wp.launch(compute_nerve_alpha_kernel, dim=N, inputs=[solver.p["alpha"], env.nerve_mask, ns, nc], device=device)
    rew = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(compute_reward_rewarped_kernel, dim=1, inputs=[ts, tc, ns, nc, 5.0, rew], device=device)

tape.backward(loss=rew)
wp.synchronize()
g = action.grad.numpy() if action.grad is not None else None
print(f"  action.grad: {g}")
print(f"  Result: {'OK (no physics, gradient should be 0 or None)' if g is None or not np.any(np.isnan(g)) else 'NaN!'}")

# ============================================================
# Test B: 1 substep of ONLY predict + collision (no SVD)
# ============================================================
print("\n" + "=" * 60)
print("Test B: predict + collision only (no cluster_solve, no damage)")

from sim.kernels import predict_kernel, collision_kernel, update_velocity_kernel

env.reset()
solver.knife_active = 1
tape = wp.Tape()
with tape:
    action = wp.array([wp.vec3(0.0, -0.5, 0.0)], dtype=wp.vec3, device=device, requires_grad=True)
    new_kp = wp.zeros(1, dtype=wp.vec3, device=device, requires_grad=True)
    wp.launch(apply_action_kernel, dim=1, inputs=[solver.knife_pos, action, 0.005], outputs=[new_kp], device=device)
    solver.knife_pos = new_kp

    dt_sub = solver.dt
    wp.launch(predict_kernel, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"], solver.p["pred_pos"], solver.p["old_pos"],
        solver.p["alpha"], solver.p["fixed"], solver.gravity, dt_sub, solver.damping,
    ], device=device)

    wp.launch(collision_kernel, dim=N, inputs=[
        solver.p["pred_pos"], solver.p["alpha"], solver.p["fixed"],
        solver.floor_y, solver.knife_pos, solver.knife_radius, 1,
    ], device=device)

    # Simple loss: sum of pred_pos y-coordinates (definitely differentiable)
    @wp.kernel
    def sum_y_kernel(pos: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        wp.atomic_add(out, 0, pos[i][1])

    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(sum_y_kernel, dim=N, inputs=[solver.p["pred_pos"], loss], device=device)

tape.backward(loss=loss)
wp.synchronize()
g = action.grad.numpy() if action.grad is not None else None
print(f"  action.grad: {g}")
nan = g is not None and np.any(np.isnan(g))
print(f"  Result: {'NaN!' if nan else 'OK' if g is not None else 'None'}")

# ============================================================
# Test C: predict + collision + 1 cluster_solve (with SVD)
# ============================================================
print("\n" + "=" * 60)
print("Test C: predict + collision + cluster_solve (SVD)")

from sim.kernels import clear_deltas_kernel, cluster_solve_kernel, apply_kernel

env.reset()
solver.knife_active = 1
tape = wp.Tape()
with tape:
    action = wp.array([wp.vec3(0.0, -0.5, 0.0)], dtype=wp.vec3, device=device, requires_grad=True)
    new_kp = wp.zeros(1, dtype=wp.vec3, device=device, requires_grad=True)
    wp.launch(apply_action_kernel, dim=1, inputs=[solver.knife_pos, action, 0.005], outputs=[new_kp], device=device)
    solver.knife_pos = new_kp

    dt_sub = solver.dt
    wp.launch(predict_kernel, dim=N, inputs=[
        solver.p["pos"], solver.p["vel"], solver.p["pred_pos"], solver.p["old_pos"],
        solver.p["alpha"], solver.p["fixed"], solver.gravity, dt_sub, solver.damping,
    ], device=device)
    wp.launch(collision_kernel, dim=N, inputs=[
        solver.p["pred_pos"], solver.p["alpha"], solver.p["fixed"],
        solver.floor_y, solver.knife_pos, solver.knife_radius, 1,
    ], device=device)

    # One iteration of cluster solve
    wp.launch(clear_deltas_kernel, dim=N, inputs=[solver.p["deltas"], solver.p["delta_counts"]], device=device)
    wp.launch(cluster_solve_kernel, dim=solver.num_clusters, inputs=[
        solver.p["pred_pos"], solver.p["alpha"],
        solver.c["starts"], solver.c["sizes"],
        solver.c["particle_ids"], solver.c["rest_offsets"],
        solver.p["deltas"], solver.p["delta_counts"],
    ], device=device)
    wp.launch(apply_kernel, dim=N, inputs=[
        solver.p["pred_pos"], solver.p["deltas"], solver.p["delta_counts"],
        solver.p["damage"], solver.p["alpha"], solver.p["fixed"], solver.stiffness,
    ], device=device)

    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(sum_y_kernel, dim=N, inputs=[solver.p["pred_pos"], loss], device=device)

tape.backward(loss=loss)
wp.synchronize()
g = action.grad.numpy() if action.grad is not None else None
print(f"  action.grad: {g}")
nan = g is not None and np.any(np.isnan(g))
print(f"  Result: {'NaN! → SVD is the problem' if nan else 'OK → SVD is fine' if g is not None else 'None'}")

# ============================================================
# Test D: full substep but NO cutting (knife_active=1 for collision only)
# ============================================================
print("\n" + "=" * 60)
print("Test D: full _substep (collision+SVD+apply) but cutting disabled")

env.reset()
solver.knife_active = 1
solver.cutting_update_interval = 999999  # disable cutting
tape = wp.Tape()
with tape:
    action = wp.array([wp.vec3(0.0, -0.5, 0.0)], dtype=wp.vec3, device=device, requires_grad=True)
    new_kp = wp.zeros(1, dtype=wp.vec3, device=device, requires_grad=True)
    wp.launch(apply_action_kernel, dim=1, inputs=[solver.knife_pos, action, 0.005], outputs=[new_kp], device=device)
    solver.knife_pos = new_kp
    solver._substep()

    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(sum_y_kernel, dim=N, inputs=[solver.p["pred_pos"], loss], device=device)

tape.backward(loss=loss)
wp.synchronize()
g = action.grad.numpy() if action.grad is not None else None
print(f"  action.grad: {g}")
nan = g is not None and np.any(np.isnan(g))
print(f"  Result: {'NaN!' if nan else 'OK' if g is not None else 'None'}")

# ============================================================
# Test E: full substep WITH cutting
# ============================================================
print("\n" + "=" * 60)
print("Test E: full _substep WITH cutting pipeline")

env.reset()
solver.knife_active = 1
solver.cutting_update_interval = 1
tape = wp.Tape()
with tape:
    action = wp.array([wp.vec3(0.0, -0.5, 0.0)], dtype=wp.vec3, device=device, requires_grad=True)
    new_kp = wp.zeros(1, dtype=wp.vec3, device=device, requires_grad=True)
    wp.launch(apply_action_kernel, dim=1, inputs=[solver.knife_pos, action, 0.005], outputs=[new_kp], device=device)
    solver.knife_pos = new_kp
    solver._substep()

    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(sum_y_kernel, dim=N, inputs=[solver.p["pred_pos"], loss], device=device)

tape.backward(loss=loss)
wp.synchronize()
g = action.grad.numpy() if action.grad is not None else None
print(f"  action.grad: {g}")
nan = g is not None and np.any(np.isnan(g))
print(f"  Result: {'NaN! → cutting pipeline is the problem' if nan else 'OK → cutting is fine' if g is not None else 'None'}")

# ============================================================
print("\n" + "=" * 60)
print("SUMMARY: First NaN test identifies the culprit kernel")
