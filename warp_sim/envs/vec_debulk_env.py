"""Vectorized Gymnasium environment for P(F) surgical debulking.

N parallel environments in one GPU solver (FF-SRL pattern).
Compatible with stable-baselines3 VecEnv interface.
"""
import sys
import os
import numpy as np
import warp as wp

import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pf_shape_matching_warp"))

from mesh.loader import (
    load_tet_mesh, build_vertex_neighbors,
    build_clusters_from_tets, pack_clusters_csr, build_edges,
)
from sim.batched_solver import BatchedSolver
from warp_sim.kernels.obs_reward_batched import (
    apply_action_batched_kernel,
    downsample_particles_batched_kernel,
    compute_tumor_displacement_batched_kernel,
    compute_nerve_displacement_batched_kernel,
    compute_position_reward_batched_kernel,
)


class VectorizedDebulkEnv(gym.Env):
    """N parallel debulking environments in one batched GPU solver.

    SB3 sees this as a single env returning (num_envs, obs_dim) arrays.
    Use with stable_baselines3.common.vec_env.VecEnv wrapper or directly.
    """

    N_SAMPLE = 250

    def __init__(self, cfg=None, num_envs=4, device="cuda:0"):
        super().__init__()

        if cfg is None:
            cfg = {}

        self.device = device
        self.num_envs = num_envs
        self.sim_substeps = cfg.get("sim_substeps", 2)
        self.action_scale = cfg.get("action_scale", 0.005)
        self.settle_steps = cfg.get("settle_steps", 200)
        self.episode_length = cfg.get("episode_length", 128)
        self.lambda_nerve = cfg.get("lambda_nerve", 2.0)
        self.knife_start = np.array(cfg.get("knife_start", [0.5, 0.5, 0.5]), dtype=np.float32)
        self.knife_start_randomize = cfg.get("knife_start_randomize", True)
        self.knife_start_range = cfg.get("knife_start_range", 0.1)

        # Load single-env mesh data
        single_env_data = self._load_single_env(cfg)

        # Build batched solver
        self.solver = BatchedSolver(single_env_data, num_envs, device)
        self.solver.dt = cfg.get("dt", 0.005)
        self.solver.num_iterations = cfg.get("num_iterations", 4)
        self.solver.stiffness = cfg.get("stiffness", 0.6)
        self.solver.damping = cfg.get("damping", 0.98)
        self.solver.knife_radius = cfg.get("knife_radius", 0.15)
        self.solver.cutting_update_interval = self.sim_substeps

        # Build nerve mask (single-env, used for all)
        self._build_nerve_mask(cfg, single_env_data)
        self._build_sample_indices(single_env_data)

        # Rest positions for reward
        self.rest_pos = wp.clone(self.solver.p["pos"])

        # Gymnasium spaces
        self.obs_dim = self.N_SAMPLE * 4 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(3,), dtype=np.float32,
        )

        # Per-env step counters
        self.step_counts = np.zeros(num_envs, dtype=np.int32)

        # Settle
        self._settle()

        print(f"VectorizedDebulkEnv: {num_envs} envs × {self.solver.N_per_env} particles "
              f"= {self.solver.N_total} total, obs_dim={self.obs_dim}")

    def _load_single_env(self, cfg):
        mesh_path = cfg.get("mesh", os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "cube", "cube_dense.msh"))

        vertices, tets = load_tet_mesh(mesh_path)
        N = len(vertices)
        min_y = vertices[:, 1].min()
        vertices[:, 1] += -min_y + 0.001

        vertex_neighbors = build_vertex_neighbors(tets, N)
        cluster_ids, cluster_cms, cluster_offsets = build_clusters_from_tets(
            vertices, vertex_neighbors, rings=2)
        starts, sizes, pids, offsets, cms = pack_clusters_csr(
            cluster_ids, cluster_offsets, cluster_cms)
        edge_a, edge_b, edge_pf_rest = build_edges(vertices, cluster_ids, vertex_neighbors)

        mesh_height = vertices[:, 1].max() - vertices[:, 1].min()
        fixed_threshold = vertices[:, 1].min() + mesh_height * 0.10
        fixed_mask = np.zeros(N, dtype=np.int32)
        for i in range(N):
            if vertices[i, 1] < fixed_threshold:
                fixed_mask[i] = 1

        return {
            "vertices": vertices.astype(np.float32),
            "fixed_mask": fixed_mask,
            "cluster_starts": starts,
            "cluster_sizes": sizes,
            "cluster_pids": pids,
            "cluster_offsets": offsets,
            "cluster_cms": cms,
            "edge_a": edge_a,
            "edge_b": edge_b,
            "edge_pf_rest": edge_pf_rest,
            "N": N,
        }

    def _build_nerve_mask(self, cfg, data):
        pos = data["vertices"]
        fixed = data["fixed_mask"]
        N = len(pos)
        nerve_cfg = cfg.get("nerve", {})
        nerve_mask = np.zeros(N, dtype=np.int32)

        y_min = nerve_cfg.get("y_min", 0.0)
        y_max = nerve_cfg.get("y_max", None)
        if y_max is None:
            y_max = pos[:, 1].min() + (pos[:, 1].max() - pos[:, 1].min()) * 0.25

        for i in range(N):
            if fixed[i] == 0 and y_min <= pos[i, 1] <= y_max:
                nerve_mask[i] = 1

        print(f"Nerve mask: {nerve_mask.sum()} particles ({nerve_mask.sum()*100//N}%)")
        self.nerve_mask = wp.array(nerve_mask, dtype=wp.int32, device=self.device)
        self._nerve_mask_np = nerve_mask

    def _build_sample_indices(self, data):
        N = data["N"]
        stride = max(1, N // self.N_SAMPLE)
        indices = np.arange(0, N, stride)[:self.N_SAMPLE]
        if len(indices) < self.N_SAMPLE:
            pad = np.full(self.N_SAMPLE - len(indices), indices[-1], dtype=np.int32)
            indices = np.concatenate([indices, pad])
        self.sample_indices = wp.array(indices.astype(np.int32),
                                       dtype=wp.int32, device=self.device)

    def _settle(self):
        print(f"Settling for {self.settle_steps} steps...")
        self.solver.knife_active = 0
        for _ in range(self.settle_steps):
            self.solver._substep()
        wp.synchronize()

        # Update snapshots after settling
        self.solver._snap_pos = wp.clone(self.solver.p["pos"])
        self.solver._snap_vel = wp.clone(self.solver.p["vel"])
        self.solver._snap_pred = wp.clone(self.solver.p["pred_pos"])
        self.solver._snap_old = wp.clone(self.solver.p["old_pos"])
        self.solver._snap_damage = wp.clone(self.solver.p["damage"])
        self.solver._snap_alpha = wp.clone(self.solver.p["alpha"])
        self.solver._snap_edge_damage = wp.clone(self.solver.e["damage"])
        self.rest_pos = wp.clone(self.solver.p["pos"])

        pos_np = self.solver.p["pos"].numpy()
        N = self.solver.N_per_env
        cm = pos_np[:N].reshape(-1, 3).mean(axis=0)
        print(f"  Settled centroid (env 0): ({cm[0]:.3f}, {cm[1]:.3f}, {cm[2]:.3f})")

    # ================================================================
    # Gymnasium Interface
    # ================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.solver.reset_envs(range(self.num_envs))
        self.step_counts[:] = 0
        self.solver.knife_active = 0

        # Randomize knife start per env
        self._randomize_knife_positions(range(self.num_envs))
        self.solver.knife_active = 1

        obs = self._get_all_obs()
        return obs, {}

    def step(self, actions):
        """
        Args:
            actions: np.ndarray (num_envs, 3) or (3,) for single env

        Returns:
            obs: np.ndarray (num_envs, obs_dim)
            rewards: np.ndarray (num_envs,)
            terminated: np.ndarray (num_envs,) bool
            truncated: np.ndarray (num_envs,) bool
            infos: dict
        """
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        actions = np.clip(actions, -1.0, 1.0).astype(np.float32)

        # 1. Apply per-env actions to knife positions
        actions_wp = wp.array(actions, dtype=wp.vec3, device=self.device)
        new_kp = wp.zeros(self.num_envs, dtype=wp.vec3, device=self.device)
        wp.launch(apply_action_batched_kernel, dim=self.num_envs,
                  inputs=[self.solver.knife_pos, actions_wp, self.action_scale],
                  outputs=[new_kp], device=self.device)
        self.solver.knife_pos = new_kp

        # 2. Physics step (all envs simultaneously)
        self.solver.knife_active = 1
        for _ in range(self.sim_substeps):
            self.solver._substep()
        wp.synchronize()

        # 3. Compute per-env rewards
        rewards = self._compute_rewards()

        # 4. Step counts and termination
        self.step_counts += 1
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = self.step_counts >= self.episode_length

        # 5. Auto-reset done envs
        done = terminated | truncated
        if done.any():
            done_ids = np.where(done)[0].tolist()
            self.solver.reset_envs(done_ids)
            self._randomize_knife_positions(done_ids)
            self.step_counts[done] = 0

        # 6. Get obs
        obs = self._get_all_obs()

        infos = {"step_counts": self.step_counts.copy()}
        return obs, rewards, terminated, truncated, infos

    # ================================================================
    # Internal
    # ================================================================

    def _randomize_knife_positions(self, env_ids):
        kp_np = self.solver.knife_pos.numpy()
        for e in env_ids:
            if self.knife_start_randomize:
                offset = np.random.uniform(-self.knife_start_range,
                                           self.knife_start_range, size=3)
            else:
                offset = np.zeros(3)
            kp_np[e] = self.knife_start + offset.astype(np.float32)
        self.solver.knife_pos = wp.array(kp_np, dtype=wp.vec3, device=self.device)

    def _get_all_obs(self):
        """Get batched obs: (num_envs, obs_dim)."""
        total_samples = self.num_envs * self.N_SAMPLE
        obs_pos = wp.zeros(total_samples, dtype=wp.vec3, device=self.device)
        obs_alpha = wp.zeros(total_samples, dtype=wp.float32, device=self.device)

        wp.launch(downsample_particles_batched_kernel, dim=total_samples,
                  inputs=[self.solver.p["pos"], self.solver.p["alpha"],
                          self.sample_indices, obs_pos, obs_alpha,
                          self.solver.N_per_env, self.N_SAMPLE],
                  device=self.device)
        wp.synchronize()

        pos_np = obs_pos.numpy().reshape(self.num_envs, self.N_SAMPLE, 3)
        alpha_np = obs_alpha.numpy().reshape(self.num_envs, self.N_SAMPLE)
        knife_np = self.solver.knife_pos.numpy().reshape(self.num_envs, 3)

        # Flatten per env: (num_envs, 750+250+3)
        obs = np.concatenate([
            pos_np.reshape(self.num_envs, -1),      # (num_envs, 750)
            alpha_np,                                 # (num_envs, 250)
            knife_np,                                 # (num_envs, 3)
        ], axis=1).astype(np.float32)

        return obs

    def _compute_rewards(self):
        """Compute per-env position-based rewards."""
        N_total = self.solver.N_total

        tumor_disp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        tumor_count = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        nerve_disp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        nerve_count = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)

        wp.launch(compute_tumor_displacement_batched_kernel, dim=N_total,
                  inputs=[self.solver.p["pos"], self.rest_pos,
                          self.solver.p["fixed"], self.nerve_mask,
                          tumor_disp, tumor_count, self.solver.N_per_env],
                  device=self.device)

        wp.launch(compute_nerve_displacement_batched_kernel, dim=N_total,
                  inputs=[self.solver.p["pos"], self.rest_pos,
                          self.nerve_mask,
                          nerve_disp, nerve_count, self.solver.N_per_env],
                  device=self.device)

        rewards_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        wp.launch(compute_position_reward_batched_kernel, dim=self.num_envs,
                  inputs=[tumor_disp, tumor_count,
                          nerve_disp, nerve_count,
                          self.lambda_nerve, rewards_wp],
                  device=self.device)
        wp.synchronize()

        return rewards_wp.numpy()
