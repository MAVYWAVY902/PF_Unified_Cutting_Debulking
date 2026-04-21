"""Isolate collision kernel backward — does gradient flow through just
apply_action + collision, ignoring edge_damage/cutting?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pf_shape_matching_warp"))

import warp as wp
wp.init()
import yaml, numpy as np

from warp_sim.envs.debulk_env import DebulkEnv
from warp_sim.kernels.obs_reward import apply_action_kernel
from pf_shape_matching_warp.sim.kernels import collision_kernel


@wp.kernel
def sum_disp_on_boundary_kernel(
    pred_pos: wp.array(dtype=wp.vec3),
    rest_pos: wp.array(dtype=wp.vec3),
    boundary_mask: wp.array(dtype=wp.int32),
    out_loss: wp.array(dtype=wp.float32),
):
    """Sum |pred_pos - rest_pos|^2 over CAPSULE particles. Differentiable scalar loss."""
    i = wp.tid()
    if boundary_mask[i] == 0:
        return
    diff = pred_pos[i] - rest_pos[i]
    dsq = wp.dot(diff, diff)
    wp.atomic_add(out_loss, 0, dsq)


def main():
    cfg = yaml.safe_load(open(
        os.path.join(os.path.dirname(__file__), "config", "debulk_cap_armA.yaml")))
    cfg['obs_mode'] = 'baseline'
    env = DebulkEnv(cfg)
    env.reset()
    for _ in range(3):
        env.step(np.array([0.5, -0.5, 0.5], dtype=np.float32))

    N = env.solver.N
    # Reset pred_pos to pos (as if predict just ran)
    wp.copy(env.solver.p["pred_pos"], env.solver.p["pos"])

    # --- KEY: move knife deep into mesh interior so collision fires on many particles
    env.solver.knife_pos = wp.array(
        [wp.vec3(0.5, 0.25, 0.5)],  # dead center of mesh
        dtype=wp.vec3, device=env.device, requires_grad=True,
    )
    pos_np = env.solver.p["pos"].numpy()
    kp_np = env.solver.knife_pos.numpy().reshape(3)
    dists = np.linalg.norm(pos_np - kp_np, axis=1)
    n_inside = int((dists < env.solver.knife_radius).sum())
    print(f"[setup] knife at {kp_np}, {n_inside} particles inside sphere (radius={env.solver.knife_radius})")

    action_leaf = wp.array([wp.vec3(0.0, 0.0, 0.0)],
                           dtype=wp.vec3, device=env.device, requires_grad=True)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)

    with wp.Tape() as tape:
        new_kp = wp.zeros(1, dtype=wp.vec3, device=env.device, requires_grad=True)
        wp.launch(apply_action_kernel, dim=1,
                  inputs=[env.solver.knife_pos, action_leaf, env.action_scale],
                  outputs=[new_kp])
        # ONE collision kernel call (no Jacobi, no cutting)
        wp.launch(collision_kernel, dim=N,
                  inputs=[env.solver.p["pred_pos"],
                          env.solver.p["alpha"], env.solver.p["fixed"],
                          env.solver.floor_y,
                          new_kp, env.solver.knife_radius, 1])
        # Loss = sum of boundary-particle displacement^2
        wp.launch(sum_disp_on_boundary_kernel, dim=N,
                  inputs=[env.solver.p["pred_pos"], env.rest_pos,
                          env._boundary_mask, loss_buf])

    wp.synchronize()
    print(f"\n=== COLLISION-ONLY GRADIENT TEST ===")
    print(f"  loss value: {loss_buf.numpy()[0]:.6f}")
    tape.backward(loss=loss_buf)

    def gnorm(name, arr):
        if arr.grad is None:
            return f"NONE"
        g = arr.grad.numpy()
        return f"norm={np.linalg.norm(g):.4e}, nonzero={int((np.abs(g)>1e-15).sum())}/{g.size}"

    print(f"  pred_pos.grad: {gnorm('pred_pos', env.solver.p['pred_pos'])}")
    print(f"  new_kp.grad:   {new_kp.grad.numpy() if new_kp.grad is not None else 'NONE'}")
    print(f"  action.grad:   {action_leaf.grad.numpy() if action_leaf.grad is not None else 'NONE'}")

    print(f"\n=== INTERPRETATION ===")
    if action_leaf.grad is not None and np.linalg.norm(action_leaf.grad.numpy()) > 1e-10:
        print(f"  ✓ collision_kernel backward WORKS.")
        print(f"  → bug is specifically in update_edge_damage_kernel's autograd.")
    else:
        print(f"  ✗ collision_kernel backward ALSO broken.")
        print(f"  → deeper Warp tape issue, needs different strategy.")


if __name__ == "__main__":
    main()
