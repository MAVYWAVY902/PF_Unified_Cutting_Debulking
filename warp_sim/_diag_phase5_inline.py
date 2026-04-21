"""Diagnostic: replicate Phase 3's exact flow but inside a LIVE env.

Goal: prove whether Phase 3 pattern still gives |g|>0 after env.reset() +
3 steps. If YES → env state is fine; bug is in _compute_tactile_analytic.
If NO → env physics leaves some state that poisons the tape.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pf_shape_matching_warp"))
import warp as wp
wp.init()
import numpy as np
import yaml
from warp_sim.envs.debulk_env import DebulkEnv
from pf_shape_matching_warp.sim.pf_damage import (
    update_edge_damage_kernel_ssa, apply_edge_damage_delta_kernel
)
from warp_sim.kernels.obs_reward import apply_action_kernel


@wp.kernel
def _sum_ed(ed: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(out, 0, ed[i])


def main():
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config",
                                            "debulk_cap_armA.yaml")))
    cfg['obs_mode'] = 'baseline'  # no tactile during settling
    env = DebulkEnv(cfg)
    env.reset()
    for _ in range(3):
        env.step(np.array([0.5, -0.5, 0.5], dtype=np.float32))

    print("\n=== state BEFORE tape ===")
    print(f"  knife_pos = {env.solver.knife_pos.numpy()}")
    print(f"  knife_active = {env.solver.knife_active}")
    print(f"  edge_damage stats: max={env.solver.e['damage'].numpy().max():.4e}, "
          f">0.9: {int((env.solver.e['damage'].numpy()>0.9).sum())}")

    # Phase 3 exact replica
    env.solver.knife_pos = wp.array([wp.vec3(0.5, 0.3, 0.5)], dtype=wp.vec3,
                                     device=env.device, requires_grad=True)
    wp.copy(env.solver.p["pred_pos"], env.solver.p["pos"])
    env.solver.e["damage"].zero_()
    env.solver.e["damage_delta"].zero_()
    num_edges = env.solver.e["num_edges"]

    action_leaf = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3,
                           device=env.device, requires_grad=True)
    loss_buf = wp.zeros(1, dtype=wp.float32, device=env.device, requires_grad=True)

    with wp.Tape() as tape:
        new_kp = wp.zeros(1, dtype=wp.vec3, device=env.device, requires_grad=True)
        wp.launch(apply_action_kernel, dim=1,
                  inputs=[env.solver.knife_pos, action_leaf, env.action_scale],
                  outputs=[new_kp])
        wp.launch(update_edge_damage_kernel_ssa, dim=num_edges,
                  inputs=[env.solver.p["pred_pos"], env.solver.p["alpha"],
                          env.solver.e["a"], env.solver.e["b"],
                          env.solver.e["pf_rest"], env.solver.e["damage"],
                          new_kp, env.solver.knife_radius, env.solver.alpha_min,
                          env.solver.e["damage_delta"]])
        wp.launch(apply_edge_damage_delta_kernel, dim=num_edges,
                  inputs=[env.solver.e["damage"], env.solver.e["damage_delta"]])
        wp.launch(_sum_ed, dim=num_edges,
                  inputs=[env.solver.e["damage"], loss_buf])

    tape.backward(loss=loss_buf)
    g_a = action_leaf.grad.numpy().reshape(3) if action_leaf.grad is not None else np.zeros(3)
    g_kp = new_kp.grad.numpy().reshape(3) if new_kp.grad is not None else np.zeros(3)
    g_ed = env.solver.e["damage"].grad.numpy() if env.solver.e["damage"].grad is not None else np.zeros(num_edges)
    g_edd = env.solver.e["damage_delta"].grad.numpy() if env.solver.e["damage_delta"].grad is not None else np.zeros(num_edges)

    print(f"\n=== tape result (Phase 3 replica) ===")
    print(f"  loss = {float(loss_buf.numpy()[0]):.4e}")
    print(f"  |g_edge_damage|       = {np.linalg.norm(g_ed):.4e} (nonzero={int((g_ed!=0).sum())}/{num_edges})")
    print(f"  |g_edge_damage_delta| = {np.linalg.norm(g_edd):.4e} (nonzero={int((g_edd!=0).sum())}/{num_edges})")
    print(f"  |g_new_kp|            = {np.linalg.norm(g_kp):.4e}  g={g_kp}")
    print(f"  |g_action_leaf|       = {np.linalg.norm(g_a):.4e}  g={g_a}")

    if np.linalg.norm(g_a) > 1e-10:
        print("\n✓ Phase 3 pattern WORKS inside live env — bug is in _compute_tactile_analytic")
    else:
        print("\n✗ Phase 3 pattern FAILS inside live env — env physics poisons tape state")


if __name__ == "__main__":
    main()
