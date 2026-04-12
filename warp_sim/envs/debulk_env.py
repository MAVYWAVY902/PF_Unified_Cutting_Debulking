"""SAPO-compatible Gym environment for P(F) surgical debulking.

Follows Rewarped design patterns:
- Obs: downsampled particle positions (N_sample, 3) + alpha (N_sample,) + knife_pos (3,)
- Action: position delta (3D), clipped [-1,1], scaled by action_scale
- Reward: simple geometric objective on particle alpha (like RollingFlat)
- No early termination, fixed episode length
"""
import sys
import os
import numpy as np
import warp as wp

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pf_shape_matching_warp"))

from mesh.loader import (
    load_tet_mesh,
    build_vertex_neighbors,
    build_clusters_from_tets,
    pack_clusters_csr,
    build_edges,
)
from sim.particles import init_warp_arrays, init_cluster_arrays, init_edge_arrays
from sim.solver import ShapeMatchingSolver

from warp_sim.kernels.obs_reward import (
    apply_action_kernel,
    downsample_particles_kernel,
    compute_com_kernel,
    normalize_com_kernel,
    compute_tumor_removal_kernel,
    compute_nerve_alpha_kernel,
    compute_reward_rewarped_kernel,
)


class DebulkEnv:
    """Differentiable surgical debulking environment (Rewarped-style).

    Observation (dict, flattened to N_sample*4 + 3):
        particle_q:  (N_sample, 3)  — downsampled particle positions
        particle_alpha: (N_sample,) — alive-ness (1=alive, 0=removed)
        knife_pos:   (3,)           — tool position

    Action (3D): position delta (dx, dy, dz), clipped to [-1,1], scaled.
    Reward: tumor_removal_reward - lambda * nerve_damage (simple geometric).
    Episode: fixed length, no early termination.
    """

    # Rewarped convention: number of downsampled particles for obs
    N_SAMPLE = 250

    def __init__(self, cfg, device="cuda:0"):
        self.device = device
        self.num_envs = 1
        self.num_act = 3

        # Config
        self.horizon_len = cfg.get("horizon_len", 32)
        self.sim_substeps = cfg.get("sim_substeps", 4)
        self.action_scale = cfg.get("action_scale", 0.01)
        self.settle_steps = cfg.get("settle_steps", 200)
        self.episode_length = cfg.get("episode_length", 300)
        self.lambda_nerve = cfg.get("lambda_nerve", 5.0)

        # Knife start
        self.knife_start = cfg.get("knife_start", [0.5, 0.8, 0.5])

        # Physics overrides
        self.dt = cfg.get("dt", 0.005)
        self.num_iterations = cfg.get("num_iterations", 4)
        self.stiffness = cfg.get("stiffness", 0.6)
        self.damping = cfg.get("damping", 0.98)
        self.knife_radius = cfg.get("knife_radius", 0.05)
        self.cutting_interval = cfg.get("cutting_interval", 1)

        # Build solver
        self._build_solver(cfg)

        # Build nerve mask
        self._build_nerve_mask(cfg)

        # Build sample indices for obs downsampling (like Rewarped: stride sampling)
        self._build_sample_indices()

        # Obs dimension: N_SAMPLE*3 (pos) + N_SAMPLE (alpha) + 3 (knife_pos)
        self.num_obs = self.N_SAMPLE * 4 + 3

        # Settle and snapshot
        self._settle_and_snapshot()

        # Env buffers
        self.obs_pos = wp.zeros(self.N_SAMPLE, dtype=wp.vec3,
                                device=device, requires_grad=True)
        self.obs_alpha = wp.zeros(self.N_SAMPLE, dtype=wp.float32,
                                  device=device, requires_grad=True)
        self.rew_buf = wp.zeros(1, dtype=wp.float32,
                                device=device, requires_grad=True)
        self.done_buf = wp.zeros(1, dtype=wp.int32, device=device)

        self.step_count = 0

        print(f"DebulkEnv initialized: {self.solver.N} particles, "
              f"N_sample={self.N_SAMPLE}, obs_dim={self.num_obs}, "
              f"horizon={self.horizon_len}, substeps={self.sim_substeps}")

    def _build_solver(self, cfg):
        """Load mesh, build clusters/edges, create solver."""
        mesh_path = cfg.get("mesh", os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "cube", "cube_dense.msh"))

        vertices, tets = load_tet_mesh(mesh_path)
        N = len(vertices)

        # Offset above floor
        min_y = vertices[:, 1].min()
        vertices[:, 1] += -min_y + 0.001
        self._vertices_raw = vertices.copy()
        self._tets = tets

        # Build topology
        vertex_neighbors = build_vertex_neighbors(tets, N)
        cluster_ids, cluster_cms, cluster_offsets = build_clusters_from_tets(
            vertices, vertex_neighbors, rings=2)
        starts, sizes, pids, offsets, cms = pack_clusters_csr(
            cluster_ids, cluster_offsets, cluster_cms)
        edge_a, edge_b, edge_pf_rest = build_edges(
            vertices, cluster_ids, vertex_neighbors)

        # Fixed mask (bottom 10%)
        mesh_height = vertices[:, 1].max() - vertices[:, 1].min()
        fixed_threshold = vertices[:, 1].min() + mesh_height * 0.10
        fixed_mask = np.zeros(N, dtype=np.int32)
        for i in range(N):
            if vertices[i, 1] < fixed_threshold:
                fixed_mask[i] = 1

        # Init Warp arrays
        p_arrays = init_warp_arrays(vertices, fixed_mask, self.device)
        c_arrays = init_cluster_arrays(starts, sizes, pids, offsets, cms, self.device)
        e_arrays = init_edge_arrays(edge_a, edge_b, edge_pf_rest, self.device)

        # Create solver
        self.solver = ShapeMatchingSolver(p_arrays, c_arrays, e_arrays, self.device)
        self.solver.dt = self.dt
        self.solver.num_iterations = self.num_iterations
        self.solver.stiffness = self.stiffness
        self.solver.damping = self.damping
        self.solver.substeps = 1
        self.solver.knife_radius = self.knife_radius
        self.solver.cutting_update_interval = self.cutting_interval

    def _build_nerve_mask(self, cfg):
        """Create nerve_mask array based on spatial heuristic."""
        N = self.solver.N
        nerve_cfg = cfg.get("nerve", {})
        nerve_type = nerve_cfg.get("type", "y_band")

        nerve_mask_np = np.zeros(N, dtype=np.int32)

        if nerve_type == "y_band":
            pos_np = self.solver.p["pos"].numpy()
            fixed_np = self.solver.p["fixed"].numpy()
            y_min = nerve_cfg.get("y_min", 0.0)
            y_max = nerve_cfg.get("y_max", None)

            if y_max is None:
                mesh_min_y = pos_np[:, 1].min()
                mesh_max_y = pos_np[:, 1].max()
                mesh_height = mesh_max_y - mesh_min_y
                y_max = mesh_min_y + mesh_height * 0.25

            for i in range(N):
                if fixed_np[i] == 0 and y_min <= pos_np[i, 1] <= y_max:
                    nerve_mask_np[i] = 1

        nerve_count = int(nerve_mask_np.sum())
        print(f"Nerve mask: {nerve_count} particles labeled as nerve ({nerve_type})")
        self.nerve_mask = wp.array(nerve_mask_np, dtype=wp.int32, device=self.device)
        self._nerve_mask_np = nerve_mask_np

    def _build_sample_indices(self):
        """Build stride-based sample indices for obs downsampling.
        Follows Rewarped: stride = N // N_SAMPLE, take every stride-th particle.
        """
        N = self.solver.N
        stride = max(1, N // self.N_SAMPLE)
        indices = np.arange(0, N, stride)[:self.N_SAMPLE]

        # If we have fewer than N_SAMPLE, pad with last index
        if len(indices) < self.N_SAMPLE:
            pad = np.full(self.N_SAMPLE - len(indices), indices[-1], dtype=np.int32)
            indices = np.concatenate([indices, pad])

        self.sample_indices = wp.array(indices.astype(np.int32),
                                       dtype=wp.int32, device=self.device)

        # Also build a nerve_mask for the sampled particles (for info)
        self._sample_nerve = self._nerve_mask_np[indices]
        print(f"Obs downsampling: {N} -> {self.N_SAMPLE} (stride={stride}), "
              f"sampled nerve={self._sample_nerve.sum()}")

    def _settle_and_snapshot(self):
        """Settle the mesh under gravity (no knife), then snapshot state."""
        print(f"Settling for {self.settle_steps} steps...")
        self.solver.knife_active = 0
        for _ in range(self.settle_steps):
            self.solver._substep()
        wp.synchronize()

        self._snap_pos = wp.clone(self.solver.p["pos"])
        self._snap_vel = wp.clone(self.solver.p["vel"])
        self._snap_pred = wp.clone(self.solver.p["pred_pos"])
        self._snap_old = wp.clone(self.solver.p["old_pos"])
        self._snap_damage = wp.clone(self.solver.p["damage"])
        self._snap_alpha = wp.clone(self.solver.p["alpha"])
        self._snap_edge_damage = wp.clone(self.solver.e["damage"])

        cm = self.solver.get_centroid()
        print(f"  Settled centroid: ({cm[0]:.3f}, {cm[1]:.3f}, {cm[2]:.3f})")

    # ================================================================
    # Gym Interface (Rewarped-compatible)
    # ================================================================

    def initialize_trajectory(self):
        """Reset env and return initial obs. Cuts gradient between episodes."""
        self.reset()
        return self._compute_obs()

    def step(self, action):
        """One env step (differentiable). Call inside wp.Tape().

        Args:
            action: wp.array(dtype=wp.vec3, shape=(1,), requires_grad=True)
                    Position delta, raw from policy (will be scaled by action_scale).

        Returns:
            obs: dict with particle_q, particle_alpha, knife_pos
            rew_buf: wp.array (1,)
            done_buf: wp.array (1,) int32
            extras: dict
        """
        # 1. Apply action as position delta (Rewarped pattern: kinematic target)
        new_knife_pos = wp.zeros(1, dtype=wp.vec3, device=self.device, requires_grad=True)
        wp.launch(apply_action_kernel, dim=1,
                  inputs=[self.solver.knife_pos, action, self.action_scale],
                  outputs=[new_knife_pos], device=self.device)
        self.solver.knife_pos = new_knife_pos

        # 2. Run physics substeps
        self.solver.knife_active = 1
        for _ in range(self.sim_substeps):
            self.solver._substep()

        # 3. Compute obs
        obs = self._compute_obs()

        # 4. Compute reward
        self._compute_reward()

        # 5. Step count (no early termination, like Rewarped)
        self.step_count += 1

        return obs, self.rew_buf, self.done_buf, {}

    def reset(self):
        """Restore all state to settled snapshot. Cuts gradients."""
        wp.copy(self.solver.p["pos"], self._snap_pos)
        wp.copy(self.solver.p["vel"], self._snap_vel)
        wp.copy(self.solver.p["pred_pos"], self._snap_pred)
        wp.copy(self.solver.p["old_pos"], self._snap_old)
        wp.copy(self.solver.p["damage"], self._snap_damage)
        wp.copy(self.solver.p["alpha"], self._snap_alpha)
        wp.copy(self.solver.e["damage"], self._snap_edge_damage)
        self.solver.p["damage_int"].zero_()
        self.solver.p["deltas"].zero_()
        self.solver.p["delta_counts"].zero_()
        self.solver.step_count = 0
        self.step_count = 0

        self.solver.set_knife_pos(*self.knife_start)
        self.solver.knife_active = 0

    def clear_grad(self):
        """Zero all gradient buffers. Called between SAPO epochs."""
        grad_arrays = [
            self.solver.p["pos"], self.solver.p["vel"],
            self.solver.p["pred_pos"], self.solver.p["old_pos"],
            self.solver.p["damage"], self.solver.p["alpha"],
            self.solver.p["deltas"],
            self.solver.e["damage"], self.solver.e["pf_rest"],
            self.solver.knife_pos,
            self.obs_pos, self.obs_alpha, self.rew_buf,
        ]
        for arr in grad_arrays:
            if hasattr(arr, 'grad') and arr.grad is not None:
                arr.grad.zero_()

    # ================================================================
    # Obs / Reward (Rewarped-style)
    # ================================================================

    def _compute_obs(self):
        """Downsample particle positions + alpha. Returns dict.

        Following Rewarped: obs = {
            particle_q: (N_SAMPLE, 3),
            particle_alpha: (N_SAMPLE,),
            knife_pos: (3,) via knife_pos array
        }
        """
        # Downsample particles
        self.obs_pos = wp.zeros(self.N_SAMPLE, dtype=wp.vec3,
                                device=self.device, requires_grad=True)
        self.obs_alpha = wp.zeros(self.N_SAMPLE, dtype=wp.float32,
                                  device=self.device, requires_grad=True)

        wp.launch(downsample_particles_kernel, dim=self.N_SAMPLE,
                  inputs=[self.solver.p["pos"], self.solver.p["alpha"],
                          self.sample_indices, self.obs_pos, self.obs_alpha],
                  device=self.device)

        return {
            "particle_q": self.obs_pos,       # (250, 3) wp.vec3
            "particle_alpha": self.obs_alpha,  # (250,) wp.float32
            "knife_pos": self.solver.knife_pos,  # (1,) wp.vec3
        }

    def _compute_reward(self):
        """Reward = tumor_removal_score - lambda * nerve_damage.

        Following RollingFlat pattern: simple geometric objective on alpha.
        tumor_score = 1/(1 + mean_alpha_tumor)^2  (lower alpha = more removed = higher score)
        nerve_damage = 1 - mean_alpha_nerve  (any nerve alpha drop is bad)
        """
        N = self.solver.N

        tumor_alpha_sum = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        tumor_count = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        nerve_alpha_sum = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        nerve_count = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)

        wp.launch(compute_tumor_removal_kernel, dim=N,
                  inputs=[self.solver.p["alpha"], self.solver.p["fixed"],
                          self.nerve_mask, tumor_alpha_sum, tumor_count],
                  device=self.device)

        wp.launch(compute_nerve_alpha_kernel, dim=N,
                  inputs=[self.solver.p["alpha"], self.nerve_mask,
                          nerve_alpha_sum, nerve_count],
                  device=self.device)

        self.rew_buf = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)

        wp.launch(compute_reward_rewarped_kernel, dim=1,
                  inputs=[tumor_alpha_sum, tumor_count,
                          nerve_alpha_sum, nerve_count,
                          self.lambda_nerve, self.rew_buf],
                  device=self.device)

    def get_obs_flat_np(self, obs_dict):
        """Convert obs dict to flat numpy array for policy network input.
        Returns: np.array shape (N_SAMPLE*4 + 3,)
        """
        wp.synchronize()
        pos_np = self.obs_pos.numpy()  # (250,) of vec3 -> need to reshape
        # Warp vec3 array .numpy() gives (250, 3)
        alpha_np = self.obs_alpha.numpy()  # (250,)
        knife_np = self.solver.knife_pos.numpy()  # (1,) of vec3 -> (1, 3)

        # Flatten: [pos_x0,y0,z0, pos_x1,y1,z1, ..., alpha_0, alpha_1, ..., kx, ky, kz]
        flat = np.concatenate([
            pos_np.reshape(-1),      # (750,)
            alpha_np.reshape(-1),    # (250,)
            knife_np.reshape(-1),    # (3,)
        ])
        return flat  # (1003,)
