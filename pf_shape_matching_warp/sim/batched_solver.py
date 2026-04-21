"""Batched Shape Matching PBD Solver — N environments in flat arrays.

Follows FF-SRL pattern: all envs concatenated in flat GPU arrays,
single kernel launch processes all envs simultaneously.

Particle i in env e → global index: e * N_per_env + i
"""
import numpy as np
import warp as wp

from .kernels import (
    predict_kernel,
    collision_kernel_batched,
    clear_deltas_kernel,
    cluster_solve_kernel,
    apply_kernel,
    update_velocity_kernel,
)
from .pf_damage import (
    update_edge_damage_kernel_batched,
    update_particle_damage_kernel,
    finalize_particle_damage_kernel,
    update_soft_removal_kernel,
)


class BatchedSolver:
    """GPU Shape Matching PBD solver for N parallel environments.

    All particle/cluster/edge arrays are flat-concatenated across envs.
    Kernel launches use dim=N_total to process all envs in one call.
    """

    def __init__(self, single_env_data, num_envs, device="cuda:0"):
        """
        Args:
            single_env_data: dict with keys:
                vertices: (N, 3) float32
                fixed_mask: (N,) int32
                cluster_starts, cluster_sizes, cluster_pids, cluster_offsets, cluster_cms
                edge_a, edge_b, edge_pf_rest
            num_envs: int
            device: str
        """
        self.device = device
        self.num_envs = num_envs
        self.N_per_env = len(single_env_data["vertices"])
        self.N_total = self.N_per_env * num_envs

        # Build batched arrays
        self.p = self._batch_particles(single_env_data, num_envs)
        self.c = self._batch_clusters(single_env_data, num_envs)
        self.e = self._batch_edges(single_env_data, num_envs)

        self.num_clusters = self.c["num_clusters"]
        self.num_edges = self.e["num_edges"]

        # Per-env knife positions: (num_envs,) of vec3
        self.knife_pos = wp.zeros(num_envs, dtype=wp.vec3, device=device)
        self.knife_radius = 0.15
        self.knife_active = 1

        # Config
        self.dt = 0.005
        self.gravity = wp.vec3(0.0, -9.81, 0.0)
        self.num_iterations = 4
        self.stiffness = 0.6
        self.damping = 0.98
        self.floor_y = 0.0
        self.substeps = 1
        self.alpha_min = 3.0
        self.aspiration_threshold = 0.3
        self.soft_removal_sharpness = 20.0
        self.cutting_update_interval = 2  # once per env.step if substeps=2

        self.step_count = 0

        # Snapshot for reset
        self._snap_pos = wp.clone(self.p["pos"])
        self._snap_vel = wp.clone(self.p["vel"])
        self._snap_pred = wp.clone(self.p["pred_pos"])
        self._snap_old = wp.clone(self.p["old_pos"])
        self._snap_damage = wp.clone(self.p["damage"])
        self._snap_alpha = wp.clone(self.p["alpha"])
        self._snap_edge_damage = wp.clone(self.e["damage"])

        print(f"BatchedSolver: {num_envs} envs × {self.N_per_env} particles = "
              f"{self.N_total} total, {self.num_clusters} clusters, {self.num_edges} edges")

    # ================================================================
    # Batching helpers
    # ================================================================

    def _batch_particles(self, data, num_envs):
        verts = data["vertices"]  # (N, 3)
        fixed = data["fixed_mask"]  # (N,)
        N = len(verts)

        pos_np = np.tile(verts, (num_envs, 1))  # (N*num_envs, 3)
        fixed_np = np.tile(fixed, num_envs)

        arrays = {
            "pos": wp.array(pos_np, dtype=wp.vec3, device=self.device),
            "vel": wp.zeros(N * num_envs, dtype=wp.vec3, device=self.device),
            "pred_pos": wp.array(pos_np, dtype=wp.vec3, device=self.device),
            "old_pos": wp.array(pos_np, dtype=wp.vec3, device=self.device),
            "damage": wp.zeros(N * num_envs, dtype=wp.float32, device=self.device),
            "alpha": wp.ones(N * num_envs, dtype=wp.float32, device=self.device),
            "fixed": wp.array(fixed_np, dtype=wp.int32, device=self.device),
            "deltas": wp.zeros(N * num_envs, dtype=wp.vec3, device=self.device),
            "delta_counts": wp.zeros(N * num_envs, dtype=wp.int32, device=self.device),
            "damage_int": wp.zeros(N * num_envs, dtype=wp.int32, device=self.device),
        }
        return arrays

    def _batch_clusters(self, data, num_envs):
        starts_s = data["cluster_starts"]    # (C,)
        sizes_s = data["cluster_sizes"]      # (C,)
        pids_s = data["cluster_pids"]        # (total_entries,)
        offsets_s = data["cluster_offsets"]   # (total_entries, 3)
        cms_s = data["cluster_cms"]          # (C, 3)

        C = len(starts_s)
        T = len(pids_s)  # total entries per env

        all_starts = []
        all_sizes = []
        all_pids = []
        all_offsets = []
        all_cms = []

        for e in range(num_envs):
            all_starts.append(starts_s + e * T)
            all_sizes.append(sizes_s.copy())
            all_pids.append(pids_s + e * self.N_per_env)
            all_offsets.append(offsets_s.copy())
            all_cms.append(cms_s.copy())

        return {
            "starts": wp.array(np.concatenate(all_starts).astype(np.int32),
                               dtype=wp.int32, device=self.device),
            "sizes": wp.array(np.concatenate(all_sizes).astype(np.int32),
                              dtype=wp.int32, device=self.device),
            "particle_ids": wp.array(np.concatenate(all_pids).astype(np.int32),
                                     dtype=wp.int32, device=self.device),
            "rest_offsets": wp.array(np.concatenate(all_offsets).astype(np.float32),
                                     dtype=wp.vec3, device=self.device),
            "rest_cm": wp.array(np.concatenate(all_cms).astype(np.float32),
                                dtype=wp.vec3, device=self.device),
            "num_clusters": C * num_envs,
            "num_clusters_per_env": C,
            "total_entries": T * num_envs,
        }

    def _batch_edges(self, data, num_envs):
        a_s = data["edge_a"]          # (E,)
        b_s = data["edge_b"]          # (E,)
        pf_s = data["edge_pf_rest"]   # (E,)
        E = len(a_s)

        all_a = []
        all_b = []
        all_pf = []

        for e in range(num_envs):
            all_a.append(a_s + e * self.N_per_env)
            all_b.append(b_s + e * self.N_per_env)
            all_pf.append(pf_s.copy())

        return {
            "a": wp.array(np.concatenate(all_a).astype(np.int32),
                          dtype=wp.int32, device=self.device),
            "b": wp.array(np.concatenate(all_b).astype(np.int32),
                          dtype=wp.int32, device=self.device),
            "pf_rest": wp.array(np.concatenate(all_pf).astype(np.float32),
                                dtype=wp.float32, device=self.device),
            "damage": wp.zeros(E * num_envs, dtype=wp.float32, device=self.device),
            "num_edges": E * num_envs,
            "num_edges_per_env": E,
        }

    # ================================================================
    # Physics step (all envs simultaneously)
    # ================================================================

    def _substep(self):
        N = self.N_total
        dt_sub = self.dt / float(self.substeps)

        # 1. Predict
        wp.launch(predict_kernel, dim=N, inputs=[
            self.p["pos"], self.p["vel"],
            self.p["pred_pos"], self.p["old_pos"],
            self.p["alpha"], self.p["fixed"],
            self.gravity, dt_sub, self.damping,
        ], device=self.device)

        # 2. Collision (batched: per-env knife_pos)
        wp.launch(collision_kernel_batched, dim=N, inputs=[
            self.p["pred_pos"],
            self.p["alpha"], self.p["fixed"],
            self.floor_y,
            self.knife_pos, self.knife_radius, self.knife_active,
            self.N_per_env,
        ], device=self.device)

        # 3. Jacobi iterations
        for it in range(self.num_iterations):
            wp.launch(clear_deltas_kernel, dim=N,
                      inputs=[self.p["deltas"], self.p["delta_counts"]],
                      device=self.device)

            wp.launch(cluster_solve_kernel, dim=self.num_clusters, inputs=[
                self.p["pred_pos"], self.p["alpha"],
                self.c["starts"], self.c["sizes"],
                self.c["particle_ids"], self.c["rest_offsets"],
                self.p["deltas"], self.p["delta_counts"],
            ], device=self.device)

            wp.launch(apply_kernel, dim=N, inputs=[
                self.p["pred_pos"], self.p["deltas"], self.p["delta_counts"],
                self.p["damage"], self.p["alpha"], self.p["fixed"],
                self.stiffness,
            ], device=self.device)

            wp.launch(collision_kernel_batched, dim=N, inputs=[
                self.p["pred_pos"],
                self.p["alpha"], self.p["fixed"],
                self.floor_y,
                self.knife_pos, self.knife_radius, self.knife_active,
                self.N_per_env,
            ], device=self.device)

        # 4. Cutting pipeline
        if (self.knife_active == 1 and
                self.step_count % self.cutting_update_interval == 0):
            num_edges = self.num_edges

            wp.launch(update_edge_damage_kernel_batched, dim=num_edges, inputs=[
                self.p["pred_pos"], self.p["alpha"],
                self.e["a"], self.e["b"],
                self.e["pf_rest"], self.e["damage"],
                self.knife_pos, self.knife_radius, self.alpha_min,
                self.N_per_env, self.e["num_edges_per_env"],
            ], device=self.device)

            damage_sum = wp.zeros(N, dtype=wp.float32, device=self.device)
            damage_count = wp.zeros(N, dtype=wp.float32, device=self.device)

            wp.launch(update_particle_damage_kernel, dim=num_edges, inputs=[
                damage_sum, damage_count,
                self.p["alpha"],
                self.e["a"], self.e["b"],
                self.e["damage"],
            ], device=self.device)

            wp.launch(finalize_particle_damage_kernel, dim=N,
                      inputs=[damage_sum, damage_count, self.p["damage"]],
                      device=self.device)

            wp.launch(update_soft_removal_kernel, dim=N, inputs=[
                self.p["damage"], self.p["alpha"], self.p["fixed"],
                self.aspiration_threshold, self.soft_removal_sharpness,
            ], device=self.device)

        # 5. Update velocity
        inv_dt = 1.0 / dt_sub
        wp.launch(update_velocity_kernel, dim=N, inputs=[
            self.p["pos"], self.p["vel"],
            self.p["pred_pos"], self.p["old_pos"],
            self.p["alpha"], self.p["fixed"],
            inv_dt,
        ], device=self.device)

        self.step_count += 1

    # ================================================================
    # Per-env reset (FF-SRL pattern: wp.copy with offsets)
    # ================================================================

    def reset_envs(self, env_ids):
        """Reset a subset of environments in-place."""
        N = self.N_per_env
        E = self.e["num_edges_per_env"]

        for e in env_ids:
            pid_off = e * N
            wp.copy(self.p["pos"], self._snap_pos,
                    dest_offset=pid_off, src_offset=pid_off, count=N)
            wp.copy(self.p["vel"], self._snap_vel,
                    dest_offset=pid_off, src_offset=pid_off, count=N)
            wp.copy(self.p["pred_pos"], self._snap_pred,
                    dest_offset=pid_off, src_offset=pid_off, count=N)
            wp.copy(self.p["old_pos"], self._snap_old,
                    dest_offset=pid_off, src_offset=pid_off, count=N)
            wp.copy(self.p["damage"], self._snap_damage,
                    dest_offset=pid_off, src_offset=pid_off, count=N)
            wp.copy(self.p["alpha"], self._snap_alpha,
                    dest_offset=pid_off, src_offset=pid_off, count=N)

            eid_off = e * E
            wp.copy(self.e["damage"], self._snap_edge_damage,
                    dest_offset=eid_off, src_offset=eid_off, count=E)

    def set_knife_pos_env(self, env_id, x, y, z):
        """Set knife position for a specific environment."""
        kp_np = self.knife_pos.numpy()
        kp_np[env_id] = [x, y, z]
        self.knife_pos = wp.array(kp_np, dtype=wp.vec3, device=self.device)
