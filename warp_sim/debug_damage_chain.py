"""Diagnose exactly where gradient breaks in the damage → alpha → reward chain."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pf_shape_matching_warp"))

import warp as wp
import numpy as np
wp.init()

from warp_sim.envs.debulk_env import DebulkEnv
from sim.pf_damage import (
    update_edge_damage_kernel, update_particle_damage_kernel,
    finalize_particle_damage_kernel, update_soft_removal_kernel,
)
from warp_sim.kernels.obs_reward import (
    compute_tumor_removal_kernel, compute_nerve_alpha_kernel,
    compute_reward_rewarped_kernel,
)


@wp.kernel
def sum_float_kernel(d: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(out, 0, d[i])


cfg = {
    "mesh": "resources/cube/cube_dense.msh",
    "settle_steps": 100, "horizon_len": 32, "sim_substeps": 1,
    "action_scale": 0.005, "knife_start": [0.5, 0.5, 0.5],
    "knife_radius": 0.15, "cutting_interval": 1, "lambda_nerve": 5.0,
    "dt": 0.005, "num_iterations": 4, "stiffness": 0.6, "damping": 0.98,
}
env = DebulkEnv(cfg)
env.reset()
device = "cuda:0"
solver = env.solver
N = solver.N
num_edges = solver.e["num_edges"]


def check_grad(name, arr):
    if arr is None:
        return "None"
    g = arr.numpy()
    norm = np.linalg.norm(g)
    nan = np.any(np.isnan(g))
    if nan:
        return f"NaN"
    elif norm < 1e-12:
        return f"ZERO"
    else:
        return f"OK norm={norm:.6f} val={g.flatten()[:3]}"


# ============================================================
print("=" * 60)
print("Test G: edge_damage_kernel only → sum(edge_damage)")
print("=" * 60)
env.reset()
tape = wp.Tape()
with tape:
    kp = wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3, device=device, requires_grad=True)
    wp.launch(update_edge_damage_kernel, dim=num_edges, inputs=[
        solver.p["pred_pos"], solver.p["alpha"],
        solver.e["a"], solver.e["b"],
        solver.e["pf_rest"], solver.e["damage"],
        kp, 0.15, 10.0,
    ], device=device)
    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(sum_float_kernel, dim=num_edges, inputs=[solver.e["damage"], loss], device=device)

wp.synchronize()
print(f"  loss (sum edge_damage): {loss.numpy()[0]:.4f}")
tape.backward(loss=loss)
wp.synchronize()
print(f"  kp.grad: {check_grad('kp', kp.grad)}")


# ============================================================
print("\n" + "=" * 60)
print("Test H: + particle_damage (atomic_add mean) → sum(particle_damage)")
print("=" * 60)
env.reset()
tape2 = wp.Tape()
with tape2:
    kp2 = wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3, device=device, requires_grad=True)
    wp.launch(update_edge_damage_kernel, dim=num_edges, inputs=[
        solver.p["pred_pos"], solver.p["alpha"],
        solver.e["a"], solver.e["b"],
        solver.e["pf_rest"], solver.e["damage"],
        kp2, 0.15, 10.0,
    ], device=device)

    ds = wp.zeros(N, dtype=wp.float32, device=device, requires_grad=True)
    dc = wp.zeros(N, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(update_particle_damage_kernel, dim=num_edges, inputs=[
        ds, dc, solver.p["alpha"], solver.e["a"], solver.e["b"], solver.e["damage"],
    ], device=device)
    wp.launch(finalize_particle_damage_kernel, dim=N, inputs=[ds, dc, solver.p["damage"]], device=device)

    loss2 = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(sum_float_kernel, dim=N, inputs=[solver.p["damage"], loss2], device=device)

wp.synchronize()
print(f"  loss (sum particle_damage): {loss2.numpy()[0]:.4f}")
tape2.backward(loss=loss2)
wp.synchronize()
print(f"  kp.grad: {check_grad('kp2', kp2.grad)}")


# ============================================================
print("\n" + "=" * 60)
print("Test I: + soft_removal → sum(1 - alpha)")
print("=" * 60)
env.reset()
tape3 = wp.Tape()
with tape3:
    kp3 = wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3, device=device, requires_grad=True)
    wp.launch(update_edge_damage_kernel, dim=num_edges, inputs=[
        solver.p["pred_pos"], solver.p["alpha"],
        solver.e["a"], solver.e["b"],
        solver.e["pf_rest"], solver.e["damage"],
        kp3, 0.15, 10.0,
    ], device=device)

    ds3 = wp.zeros(N, dtype=wp.float32, device=device, requires_grad=True)
    dc3 = wp.zeros(N, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(update_particle_damage_kernel, dim=num_edges, inputs=[
        ds3, dc3, solver.p["alpha"], solver.e["a"], solver.e["b"], solver.e["damage"],
    ], device=device)
    wp.launch(finalize_particle_damage_kernel, dim=N, inputs=[ds3, dc3, solver.p["damage"]], device=device)
    wp.launch(update_soft_removal_kernel, dim=N, inputs=[
        solver.p["damage"], solver.p["alpha"], solver.p["fixed"], 0.3, 20.0,
    ], device=device)

    # Loss = sum(1 - alpha) = volume removed
    @wp.kernel
    def sum_one_minus_alpha(a: wp.array(dtype=wp.float32), f: wp.array(dtype=wp.int32),
                            out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        if f[i] == 0:
            wp.atomic_add(out, 0, 1.0 - a[i])

    loss3 = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(sum_one_minus_alpha, dim=N, inputs=[
        solver.p["alpha"], solver.p["fixed"], loss3], device=device)

wp.synchronize()
print(f"  loss (sum 1-alpha): {loss3.numpy()[0]:.4f}")
tape3.backward(loss=loss3)
wp.synchronize()
print(f"  kp.grad: {check_grad('kp3', kp3.grad)}")


# ============================================================
print("\n" + "=" * 60)
print("Test J: + reward_kernel")
print("=" * 60)
env.reset()
tape4 = wp.Tape()
with tape4:
    kp4 = wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3, device=device, requires_grad=True)
    wp.launch(update_edge_damage_kernel, dim=num_edges, inputs=[
        solver.p["pred_pos"], solver.p["alpha"],
        solver.e["a"], solver.e["b"],
        solver.e["pf_rest"], solver.e["damage"],
        kp4, 0.15, 10.0,
    ], device=device)

    ds4 = wp.zeros(N, dtype=wp.float32, device=device, requires_grad=True)
    dc4 = wp.zeros(N, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(update_particle_damage_kernel, dim=num_edges, inputs=[
        ds4, dc4, solver.p["alpha"], solver.e["a"], solver.e["b"], solver.e["damage"],
    ], device=device)
    wp.launch(finalize_particle_damage_kernel, dim=N, inputs=[ds4, dc4, solver.p["damage"]], device=device)
    wp.launch(update_soft_removal_kernel, dim=N, inputs=[
        solver.p["damage"], solver.p["alpha"], solver.p["fixed"], 0.3, 20.0,
    ], device=device)

    ts = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    tc = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    ns = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    nc = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(compute_tumor_removal_kernel, dim=N, inputs=[
        solver.p["alpha"], solver.p["fixed"], env.nerve_mask, ts, tc], device=device)
    wp.launch(compute_nerve_alpha_kernel, dim=N, inputs=[
        solver.p["alpha"], env.nerve_mask, ns, nc], device=device)
    rew = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(compute_reward_rewarped_kernel, dim=1, inputs=[
        ts, tc, ns, nc, 5.0, rew], device=device)

wp.synchronize()
print(f"  reward: {rew.numpy()[0]:.6f}")
tape4.backward(loss=rew)
wp.synchronize()
print(f"  kp.grad: {check_grad('kp4', kp4.grad)}")


# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("First FAIL/ZERO = gradient break point")
