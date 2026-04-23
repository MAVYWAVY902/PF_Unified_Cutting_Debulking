"""Gymnasium environment for P(F) surgical debulking with Coverage Map.

Obs: pos(750) + alpha(250) + knife(3) + coverage(512) + knife_local_cov(1) = 1516D
Action: position delta (3D), clipped [-1,1], scaled, workspace-clamped (negative margin → inside mesh)
Reward (7-term): new_tumor + guidance - λ*new_nerve - waste - step_penalty + completion + surface_term
  surface_term: +bonus when knife on initial-boundary or partial-coverage voxel (CUSA-realistic),
                -penalty when diving into intact interior.

References:
- PaintRL: +new_pixels - overlap penalty
- Coverage Planning: coverage map as observation
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
from sim.particles import init_warp_arrays, init_cluster_arrays, init_edge_arrays
from sim.solver import ShapeMatchingSolver
from warp_sim.kernels.obs_reward import apply_action_kernel, downsample_particles_kernel


# Helper kernel for L2 tactile: sum all edge_damage into a scalar buffer.
# Used by _compute_tactile_analytic (module-level so Warp can generate adjoint
# — @wp.kernel inside a method body fails with "exec() not supported").
@wp.kernel
def _sum_edge_damage_for_tactile_kernel(
    edge_damage: wp.array(dtype=wp.float32),
    out_loss: wp.array(dtype=wp.float32),
):
    eid = wp.tid()
    wp.atomic_add(out_loss, 0, edge_damage[eid])


class DebulkEnv(gym.Env):
    """Coverage-aware surgical debulking environment."""

    metadata = {"render_modes": []}
    N_SAMPLE = 250
    GRID_RES = 8  # 8×8×8 = 512 voxels

    def __init__(self, cfg=None, device="cuda:0", render_mode=None):
        super().__init__()
        if cfg is None:
            cfg = {}

        self.device = device
        self.sim_substeps = cfg.get("sim_substeps", 2)
        self.action_scale = cfg.get("action_scale", 0.003)
        self.settle_steps = cfg.get("settle_steps", 200)
        self.episode_length = cfg.get("episode_length", 512)
        self.lambda_nerve = cfg.get("lambda_nerve", 10.0)
        self.waste_penalty_weight = cfg.get("waste_penalty_weight", 0.5)
        self.knife_start = cfg.get("knife_start", [0.95, 0.7, 0.5])
        self.dt = cfg.get("dt", 0.005)
        self.num_iterations = cfg.get("num_iterations", 4)
        self.stiffness = cfg.get("stiffness", 0.6)
        self.damping = cfg.get("damping", 0.98)
        self.knife_radius = cfg.get("knife_radius", 0.1)
        self.cutting_interval = cfg.get("cutting_interval", 1)
        self.debulk_threshold = cfg.get("debulk_threshold", 0.5)
        self.guidance_weight = cfg.get("guidance_weight", 5.0)
        self.guidance_std = cfg.get("guidance_std", 0.15)
        self.completion_bonus = cfg.get("completion_bonus", 50.0)
        self.step_penalty = cfg.get("step_penalty", 0.1)
        self.workspace_margin = cfg.get("workspace_margin", -0.02)
        self.surface_bonus = cfg.get("surface_bonus", 2.0)          # reward when knife on surface
        self.interior_penalty = cfg.get("interior_penalty", 5.0)    # penalty diving into intact interior
        self.surface_low = cfg.get("surface_low", 0.1)              # coverage in (low, high) = dynamic surface
        self.surface_high = cfg.get("surface_high", 0.9)

        # Capsule-task reward params (M3.3) — only used when task_type == "capsule_only"
        self.capsule_budget = cfg.get("capsule_budget", 0.05)                         # allowed shell breach frac (entry point)
        self.lambda_capsule = cfg.get("lambda_capsule", 30.0)                         # asymmetric penalty weight
        self.completion_interior_frac = cfg.get("completion_interior_frac", 0.90)     # NTR-like threshold
        self.catastrophic_breach_mult = cfg.get("catastrophic_breach_multiplier", 3.0)# terminate at 3× budget
        self.catastrophic_penalty = cfg.get("catastrophic_penalty", 100.0)            # terminal negative reward

        # Obs mode for ablation (M3.4) — selects tactile feature source
        #   "baseline"    : no extra dim           (Arm A)
        #   "random_3d"   : 3D random gaussian     (Arm X, dimension control)
        #   "finite_diff" : 3D FD gradient (6×fwd) (Arm Y, method control)
        #   "l2_analytic" : 3D analytic gradient   (Arm C, main contribution)
        self._obs_mode = cfg.get("obs_mode", "baseline")
        assert self._obs_mode in ("baseline", "random_3d", "finite_diff", "l2_analytic"), \
            f"unknown obs_mode={self._obs_mode!r}"
        self._tactile_scale = cfg.get("tactile_scale", 10.0)   # tanh pre-scale
        self._fd_epsilon = cfg.get("fd_epsilon", 0.01)         # finite-diff step

        # Build solver
        self._build_solver(cfg)
        self._build_nerve_mask(cfg)
        self._build_sample_indices()

        # Obs: pos(750) + alpha(250) + knife(3) + coverage(512) + knife_local_cov(1) [+ tactile(3)]
        grid_size = self.GRID_RES ** 3  # 512
        base_obs_dim = self.N_SAMPLE * 4 + 3 + grid_size + 1  # 1516
        tactile_extra = 3 if self._obs_mode != "baseline" else 0
        self.num_obs = base_obs_dim + tactile_extra           # 1516 or 1519
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_obs,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
        )

        # Settle and snapshot
        self._settle_and_snapshot()

        # Build coverage grid infrastructure
        self._build_coverage_grid()

        # State trackers
        self.step_count = 0
        self._prev_alpha = np.ones(self.solver.N, dtype=np.float32)
        self._knife_local_coverage = 0.0
        self._tactile_g_cap = np.zeros(3, dtype=np.float32)   # M3.4 tactile obs
        self._initialized = True

        print(f"DebulkEnv initialized: {self.solver.N} particles, "
              f"obs_dim={self.num_obs} (grid={grid_size}), "
              f"episode_len={self.episode_length}, obs_mode={self._obs_mode}")
        # Legacy-vs-new task banner (M3.3)
        if self._task_type == "capsule_only":
            print(f"[task] capsule_only — reward = 4-term (interior + capsule_penalty + "
                  f"step + completion), budget={self.capsule_budget}, λ_cap={self.lambda_capsule}")
        else:
            print(f"[task] nerve_avoid (legacy) — reward = 7-term coverage+surface. "
                  f"For capsule task set task_type='capsule_only' in config.")

    # ================================================================
    # Build helpers
    # ================================================================

    def _build_solver(self, cfg):
        mesh_path = cfg.get("mesh", os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "cube", "cube_dense.msh"))
        vertices, tets = load_tet_mesh(mesh_path)
        N = len(vertices)
        min_y = vertices[:, 1].min()
        vertices[:, 1] += -min_y + 0.001
        self._tets = tets  # kept for surface-boundary extraction

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

        p_arrays = init_warp_arrays(vertices, fixed_mask, self.device)
        c_arrays = init_cluster_arrays(starts, sizes, pids, offsets, cms, self.device)
        e_arrays = init_edge_arrays(edge_a, edge_b, edge_pf_rest, self.device)

        self.solver = ShapeMatchingSolver(p_arrays, c_arrays, e_arrays, self.device)
        self.solver.dt = self.dt
        self.solver.num_iterations = self.num_iterations
        self.solver.stiffness = self.stiffness
        self.solver.damping = self.damping
        self.solver.substeps = 1
        self.solver.knife_radius = self.knife_radius
        self.solver.cutting_update_interval = self.sim_substeps

    def _build_nerve_mask(self, cfg):
        N = self.solver.N
        nerve_cfg = cfg.get("nerve", {})
        task_type = cfg.get("task_type", "nerve_avoid")  # default = legacy 3-class behavior

        nerve_mask_np = np.zeros(N, dtype=np.int32)
        pos_np = self.solver.p["pos"].numpy()
        fixed_np = self.solver.p["fixed"].numpy()

        if task_type == "capsule_only":
            # Capsule task: NO nerve layer. All non-fixed particles = tumor candidates.
            # Capsule vs interior distinction is handled later in _build_initial_surface_voxels.
            print("[task] capsule_only mode — nerve layer disabled, using tumor+capsule 2-class")
        else:
            # Legacy "nerve_avoid" task: y-band nerve layer
            y_max = nerve_cfg.get("y_max", None)
            if y_max is None:
                y_max = pos_np[:, 1].min() + (pos_np[:, 1].max() - pos_np[:, 1].min()) * 0.25
            for i in range(N):
                if fixed_np[i] == 0 and pos_np[i, 1] <= y_max:
                    nerve_mask_np[i] = 1

        self._task_type = task_type
        self._nerve_mask_np = nerve_mask_np
        self._fixed_np = fixed_np.copy()
        self._tumor_mask = (fixed_np == 0) & (nerve_mask_np == 0)
        self._nerve_mask_bool = nerve_mask_np == 1
        print(f"Nerve: {nerve_mask_np.sum()} particles, Tumor: {self._tumor_mask.sum()}")
        self.nerve_mask = wp.array(nerve_mask_np, dtype=wp.int32, device=self.device)

    def _build_sample_indices(self):
        N = self.solver.N
        stride = max(1, N // self.N_SAMPLE)
        indices = np.arange(0, N, stride)[:self.N_SAMPLE]
        if len(indices) < self.N_SAMPLE:
            pad = np.full(self.N_SAMPLE - len(indices), indices[-1], dtype=np.int32)
            indices = np.concatenate([indices, pad])
        self.sample_indices = wp.array(indices.astype(np.int32),
                                       dtype=wp.int32, device=self.device)

    def _settle_and_snapshot(self):
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
        self.rest_pos = wp.clone(self.solver.p["pos"])  # for rollout visualization

        pos_np = self.solver.p["pos"].numpy().reshape(-1, 3)
        margin = self.workspace_margin
        self.workspace_low = pos_np.min(axis=0) - margin
        self.workspace_high = pos_np.max(axis=0) + margin

        cm = self.solver.get_centroid()
        print(f"  Settled centroid: ({cm[0]:.3f}, {cm[1]:.3f}, {cm[2]:.3f})")
        print(f"  Workspace: [{self.workspace_low}] to [{self.workspace_high}]")

    def _build_coverage_grid(self):
        """Precompute voxel grid for coverage tracking."""
        G = self.GRID_RES
        self.voxel_size = (self.workspace_high - self.workspace_low) / G

        # Precompute which voxel each particle belongs to (using rest positions)
        pos_np = self._snap_pos.numpy().reshape(-1, 3)
        voxel_idx = np.floor((pos_np - self.workspace_low) / self.voxel_size).astype(np.int32)
        self._particle_voxel = np.clip(voxel_idx, 0, G - 1)  # (N, 3)

        # Count tumor particles per voxel (for normalization)
        self._voxel_tumor_count = np.zeros((G, G, G), dtype=np.float32)
        for i in range(self.solver.N):
            if self._tumor_mask[i]:
                vx, vy, vz = self._particle_voxel[i]
                self._voxel_tumor_count[vx, vy, vz] += 1.0

        occupied = (self._voxel_tumor_count > 0).sum()

        # Precompute all voxel centers (for distance guidance)
        # Only tumor-occupied voxels
        self._tumor_voxel_indices = np.argwhere(self._voxel_tumor_count > 0)  # (K, 3)
        self._tumor_voxel_centers = (
            self.workspace_low + (self._tumor_voxel_indices + 0.5) * self.voxel_size
        )  # (K, 3) world coordinates

        print(f"  Coverage grid: {G}³={G**3} voxels, {occupied} occupied by tumor")

        # Build initial surface voxel mask (boundary tet faces → boundary particles → voxels)
        self._build_initial_surface_voxels()

    def _build_initial_surface_voxels(self):
        """Mark voxels containing mesh-boundary particles. Used so outer surface
        of un-cut mesh is treated as 'on-surface' (coverage=0 initially for all voxels,
        so dynamic-surface detection alone misses the original outer skin).
        """
        from collections import defaultdict
        G = self.GRID_RES

        face_count = defaultdict(int)
        for tet in self._tets:
            a, b, c, d = int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])
            for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
                face_count[tuple(sorted(face))] += 1

        boundary_particle = np.zeros(self.solver.N, dtype=bool)
        for face, count in face_count.items():
            if count == 1:  # face belongs to only one tet => on mesh surface
                for v in face:
                    boundary_particle[v] = True

        # Tumor boundary particles only (nerve surface and fixed-base don't count as cut targets)
        boundary_tumor = boundary_particle & self._tumor_mask

        # --- M3.1: Persist per-particle masks for capsule task (L1/L2 ablation) ---
        # capsule  = tumor boundary particles (outer shell, must be preserved)
        # interior = tumor non-boundary particles (CUSA debulk target)
        interior_tumor = self._tumor_mask & ~boundary_tumor
        self._boundary_mask_np = boundary_tumor.astype(np.int32)
        self._interior_mask_np = interior_tumor.astype(np.int32)
        self._boundary_mask = wp.array(self._boundary_mask_np, dtype=wp.int32, device=self.device)
        self._interior_mask = wp.array(self._interior_mask_np, dtype=wp.int32, device=self.device)
        # Total counts for reward normalization
        self._total_capsule = int(self._boundary_mask_np.sum())
        self._total_interior = int(self._interior_mask_np.sum())

        self._initial_surface_voxel = np.zeros((G, G, G), dtype=bool)
        for i in np.where(boundary_tumor)[0]:
            vx, vy, vz = self._particle_voxel[i]
            self._initial_surface_voxel[vx, vy, vz] = True

        n_boundary = int(boundary_tumor.sum())
        n_surface_vox = int(self._initial_surface_voxel.sum())
        print(f"  Initial surface: {n_boundary} boundary particles → {n_surface_vox} voxels")
        print(f"  Capsule vs interior: {self._total_capsule} capsule / "
              f"{self._total_interior} interior particles "
              f"({100*self._total_capsule/(self._total_capsule+self._total_interior):.1f}% shell)")

    # ================================================================
    # Gymnasium Interface
    # ================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

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
        # Stage C (2026-04-22): reset Müller warm-start buffer. Without this,
        # cluster_quat retains stale rotation from previous episode and Müller
        # iterates from a bad starting point.
        self.solver.reset_cluster_quat()
        self.solver.step_count = 0
        self.step_count = 0

        self.solver.set_knife_pos(*self.knife_start)
        self.solver.knife_active = 0

        # Reset coverage tracking
        self._prev_alpha = np.ones(self.solver.N, dtype=np.float32)
        self._knife_local_coverage = 0.0
        # Capsule task tracking (M3.3)
        self._capsule_breached_frac = 0.0
        self._interior_done_frac = 0.0
        # Tactile obs reset (M3.4)
        self._tactile_g_cap = np.zeros(3, dtype=np.float32)

        return self._get_obs_np(), {}

    def step(self, action):
        # 1. Clip and convert action
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        action_wp = wp.array(
            [wp.vec3(action[0], action[1], action[2])],
            dtype=wp.vec3, device=self.device, requires_grad=True,
        )

        # 2. Apply action
        new_knife_pos = wp.zeros(1, dtype=wp.vec3, device=self.device, requires_grad=True)
        wp.launch(apply_action_kernel, dim=1,
                  inputs=[self.solver.knife_pos, action_wp, self.action_scale],
                  outputs=[new_knife_pos], device=self.device)

        # 3. Workspace clamp
        wp.synchronize()
        kp_np = new_knife_pos.numpy().reshape(3)
        kp_np = np.clip(kp_np, self.workspace_low, self.workspace_high)
        self.solver.knife_pos = wp.array(
            [wp.vec3(kp_np[0], kp_np[1], kp_np[2])],
            dtype=wp.vec3, device=self.device, requires_grad=True,
        )

        # 4. Run physics
        self.solver.knife_active = 1
        for _ in range(self.sim_substeps):
            self.solver._substep()
        wp.synchronize()

        # 5a. Compute tactile feature (M3.4) for next obs — BEFORE reward, since
        #     helpers snapshot/restore solver state. After this call, state == post-physics.
        if self._obs_mode != "baseline":
            self._compute_tactile_feature()

        # 5b. Compute reward (dispatched by task_type)
        # [LEGACY] nerve_avoid → 7-term coverage+surface reward (unchanged)
        # [NEW M3.3] capsule_only → 4-term capsule reward
        terminated = False
        if self._task_type == "capsule_only":
            reward, terminated_by_reward = self._compute_reward_capsule(kp_np)
            if terminated_by_reward:
                terminated = True
        else:
            reward = self._compute_reward(kp_np)

        # 6. Step count
        self.step_count += 1
        truncated = self.step_count >= self.episode_length

        # 7. Get obs (includes coverage grid)
        obs_np = self._get_obs_np()

        info = {
            "step": self.step_count,
            "knife_pos": kp_np.copy(),
        }
        return obs_np, reward, terminated, truncated, info

    # ================================================================
    # Observation (with coverage grid)
    # ================================================================

    def _get_obs_np(self):
        """Obs = particle_pos(750) + particle_alpha(250) + knife(3) + coverage_grid(512)."""
        # Downsampled particles
        obs_pos = wp.zeros(self.N_SAMPLE, dtype=wp.vec3, device=self.device)
        obs_alpha = wp.zeros(self.N_SAMPLE, dtype=wp.float32, device=self.device)
        wp.launch(downsample_particles_kernel, dim=self.N_SAMPLE,
                  inputs=[self.solver.p["pos"], self.solver.p["alpha"],
                          self.sample_indices, obs_pos, obs_alpha],
                  device=self.device)
        wp.synchronize()

        pos_np = obs_pos.numpy().reshape(-1)       # (750,)
        alpha_np = obs_alpha.numpy().reshape(-1)    # (250,)
        knife_np = self.solver.knife_pos.numpy().reshape(-1)  # (3,)

        # Coverage grid: fraction of debulked particles per voxel
        coverage = self._compute_coverage_grid()  # (512,)

        # Knife-local coverage scalar: tells policy if it's on surface / intact / cavity
        local_cov = np.array([self._knife_local_coverage], dtype=np.float32)  # (1,)

        if self._obs_mode == "baseline":
            return np.concatenate([pos_np, alpha_np, knife_np, coverage, local_cov]).astype(np.float32)
        # M3.4: append 3D tactile (random / FD / analytic gradient)
        return np.concatenate([pos_np, alpha_np, knife_np, coverage, local_cov,
                               self._tactile_g_cap]).astype(np.float32)

    def _compute_coverage_grid(self):
        """Compute 8×8×8 coverage grid: fraction debulked per voxel."""
        G = self.GRID_RES
        alpha_np = self.solver.p["alpha"].numpy()

        grid = np.zeros((G, G, G), dtype=np.float32)
        for i in range(self.solver.N):
            if self._tumor_mask[i] and alpha_np[i] < self.debulk_threshold:
                vx, vy, vz = self._particle_voxel[i]
                grid[vx, vy, vz] += 1.0

        # Normalize: fraction of debulked particles in each voxel
        coverage = np.where(self._voxel_tumor_count > 0,
                            grid / self._voxel_tumor_count, 0.0)
        return coverage.flatten()  # (512,)

    # ================================================================
    # Coverage-Aware Reward
    # ================================================================

    def _compute_reward(self, knife_pos):
        """Coverage-aware reward with dense guidance + surface constraint.

        7 terms:
        1. +new_debulked_count          ← coverage progress (PaintRL)
        2. +guidance * (1-tanh(d/std))  ← distance to nearest uncut (Isaac Lab)
        3. -λ × nerve_damage            ← nerve protection
        4. -waste_penalty               ← don't revisit debulked areas
        5. +completion_bonus            ← bonus at 80%+ total coverage (ClothPPO)
        6. -step_penalty                ← per-step baseline: idling is strictly costly
        7. surface_term                 ← +bonus on surface, -penalty diving into intact interior
                                          (surface = initial mesh boundary OR partial-coverage voxel)
        """
        alpha_np = self.solver.p["alpha"].numpy()
        G = self.GRID_RES

        # --- 1. Coverage progress: newly debulked tumor particles ---
        prev_alive = self._prev_alpha >= self.debulk_threshold
        now_debulked = alpha_np < self.debulk_threshold
        newly_debulked = prev_alive & now_debulked
        new_tumor = float(np.sum(newly_debulked & self._tumor_mask))

        # --- 3. Nerve damage ---
        new_nerve = float(np.sum(newly_debulked & self._nerve_mask_bool))

        # --- 2. Distance guidance: tanh kernel to nearest uncut voxel (Isaac Lab pattern) ---
        coverage_3d = self._compute_coverage_grid().reshape(G, G, G)
        # Find uncut voxels (coverage < 0.5 AND has tumor particles)
        uncut_mask = (coverage_3d < 0.5) & (self._voxel_tumor_count > 0)
        uncut_indices = np.argwhere(uncut_mask)

        if len(uncut_indices) > 0:
            uncut_centers = self.workspace_low + (uncut_indices + 0.5) * self.voxel_size
            dists = np.linalg.norm(uncut_centers - knife_pos, axis=1)
            min_dist = dists.min()
            guidance = self.guidance_weight * (1.0 - np.tanh(min_dist / self.guidance_std))
        else:
            guidance = self.guidance_weight  # all cut → max guidance reward

        # --- 4. Waste penalty: knife in already heavily-debulked voxel ---
        knife_voxel = np.floor((knife_pos - self.workspace_low) / self.voxel_size).astype(int)
        knife_voxel = np.clip(knife_voxel, 0, G - 1)
        vx, vy, vz = knife_voxel
        local_coverage = coverage_3d[vx, vy, vz]
        waste_penalty = max(0.0, local_coverage - 0.8)

        # --- 5. Completion bonus: total coverage > 80% ---
        total_occupied = (self._voxel_tumor_count > 0).sum()
        total_cut = (coverage_3d[self._voxel_tumor_count > 0] > 0.5).sum()
        total_coverage_frac = total_cut / max(total_occupied, 1)
        completion = self.completion_bonus if total_coverage_frac > 0.8 else 0.0

        # --- 7. Surface term: CUSA-realistic surface-only constraint ---
        # Surface = initial mesh boundary voxel (geometric) OR partial-coverage voxel (dynamic)
        lc = float(local_coverage)
        is_dynamic_surface = (self.surface_low < lc < self.surface_high)
        is_initial_surface = (bool(self._initial_surface_voxel[vx, vy, vz])
                              and lc < self.surface_high)
        if is_dynamic_surface or is_initial_surface:
            surface_term = self.surface_bonus            # on surface (correct place)
        elif lc < self.surface_low:
            surface_term = -self.interior_penalty        # diving into intact interior
        else:
            surface_term = 0.0                           # in fully-cut cavity (waste_penalty handles)

        # Cache for obs: knife-local coverage scalar
        self._knife_local_coverage = lc

        # Update prev alpha
        self._prev_alpha = alpha_np.copy()

        reward = (
            + 1.0 * new_tumor
            + guidance
            - self.lambda_nerve * new_nerve
            - self.waste_penalty_weight * waste_penalty
            + completion
            - self.step_penalty
            + surface_term
        )

        self.rew_buf = wp.array([reward], dtype=wp.float32, device=self.device)
        return reward

    # ================================================================
    # M3.3 — Capsule task reward (4-term + catastrophic termination)
    # Only called when cfg["task_type"] == "capsule_only".
    # The legacy _compute_reward() above stays untouched for backward compat.
    # ================================================================

    def _compute_reward_capsule(self, knife_pos):
        """4-term reward for intracapsular debulking + capsule preservation.

        Dispatched from step() when self._task_type == 'capsule_only'.

        R1 = +1.0 * new_interior_debulked
            Reward per interior tumor particle newly crossed below alpha threshold.
            Only counts INTERIOR particles (not capsule/boundary) — incentivizes
            "carving into the cavity" rather than "peeling the outside".

        R2 = -lambda_capsule * max(0, capsule_breached_frac - capsule_budget)
            Piecewise linear: zero penalty while breach <= budget (lets policy
            open a small entry hole), steep negative slope beyond budget.

        R3 = -step_penalty (default 0.1)
            Per-step baseline so idling is strictly worse than any progress action.

        R4 = +completion_bonus * 1[interior_done_frac > completion_interior_frac]
            One-off terminal bonus for near-total resection (NTR).

        Catastrophic termination:
            If capsule_breached_frac > catastrophic_breach_mult * capsule_budget
            → terminated=True, reward -= catastrophic_penalty (default 100).

        Returns: (reward: float, terminated: bool)
        """
        alpha_np = self.solver.p["alpha"].numpy()

        # Dead / alive masks at previous and current step
        prev_alive = self._prev_alpha >= self.debulk_threshold
        now_dead = alpha_np < self.debulk_threshold

        # R1: newly debulked INTERIOR (non-boundary tumor) particles this step
        newly_dead = prev_alive & now_dead
        interior_mask_bool = self._interior_mask_np == 1
        new_interior = float(np.sum(newly_dead & interior_mask_bool))

        # Cumulative damage fractions
        boundary_mask_bool = self._boundary_mask_np == 1
        capsule_dead_now = int(np.sum(now_dead & boundary_mask_bool))
        capsule_breached_frac = capsule_dead_now / max(1, self._total_capsule)

        interior_dead_now = int(np.sum(now_dead & interior_mask_bool))
        interior_done_frac = interior_dead_now / max(1, self._total_interior)

        # Cache for logging / obs (used by M3.4 gradient tape + rollout metrics)
        self._capsule_breached_frac = capsule_breached_frac
        self._interior_done_frac = interior_done_frac

        # R2: capsule breach penalty (piecewise linear past budget)
        capsule_overshoot = max(0.0, capsule_breached_frac - self.capsule_budget)
        R2 = -self.lambda_capsule * capsule_overshoot

        # R4: completion bonus (one-off signal, SB3 handles exact timing via episode return)
        completion = self.completion_bonus if interior_done_frac > self.completion_interior_frac else 0.0

        # Also cache knife_local_coverage (obs uses it in nerve_avoid task; keep for compat)
        G = self.GRID_RES
        knife_voxel = np.floor((knife_pos - self.workspace_low) / self.voxel_size).astype(int)
        knife_voxel = np.clip(knife_voxel, 0, G - 1)
        coverage_3d_for_obs = self._compute_coverage_grid().reshape(G, G, G)
        self._knife_local_coverage = float(coverage_3d_for_obs[knife_voxel[0], knife_voxel[1], knife_voxel[2]])

        # Update prev alpha before termination branch (so next step's R1 is consistent)
        self._prev_alpha = alpha_np.copy()

        # ─── 4-term aggregation ───
        reward = (
            + 1.0 * new_interior       # R1
            + R2                        # (capsule penalty, already ≤ 0)
            - self.step_penalty         # R3
            + completion                # R4
        )

        # ─── Catastrophic termination check ───
        terminated = False
        if capsule_breached_frac > self.catastrophic_breach_mult * self.capsule_budget:
            terminated = True
            reward -= self.catastrophic_penalty

        self.rew_buf = wp.array([reward], dtype=wp.float32, device=self.device)
        return reward, terminated

    # ================================================================
    # M3.4 — L2 tactile gradient computation
    # Feeds 3D "tactile" feature into obs for Arm C (l2_analytic),
    # Arm Y (finite_diff), Arm X (random_3d). Arm A (baseline) skips this.
    # ================================================================

    def _snapshot_solver_state(self):
        """Deep-copy all mutable solver arrays for rewind after gradient compute."""
        return {
            "pos":          wp.clone(self.solver.p["pos"]),
            "vel":          wp.clone(self.solver.p["vel"]),
            "pred_pos":     wp.clone(self.solver.p["pred_pos"]),
            "old_pos":      wp.clone(self.solver.p["old_pos"]),
            "damage":       wp.clone(self.solver.p["damage"]),
            "damage_int":   wp.clone(self.solver.p["damage_int"]),
            "alpha":        wp.clone(self.solver.p["alpha"]),
            "deltas":       wp.clone(self.solver.p["deltas"]),
            "delta_counts": wp.clone(self.solver.p["delta_counts"]),
            "edge_damage":  wp.clone(self.solver.e["damage"]),
            "knife_pos":    wp.clone(self.solver.knife_pos),
            "knife_active": int(self.solver.knife_active),
            "step_count":   int(self.solver.step_count),
        }

    def _restore_solver_state(self, snap):
        """Restore solver arrays from snapshot. Inverse of _snapshot_solver_state."""
        wp.copy(self.solver.p["pos"], snap["pos"])
        wp.copy(self.solver.p["vel"], snap["vel"])
        wp.copy(self.solver.p["pred_pos"], snap["pred_pos"])
        wp.copy(self.solver.p["old_pos"], snap["old_pos"])
        wp.copy(self.solver.p["damage"], snap["damage"])
        wp.copy(self.solver.p["damage_int"], snap["damage_int"])
        wp.copy(self.solver.p["alpha"], snap["alpha"])
        wp.copy(self.solver.p["deltas"], snap["deltas"])
        wp.copy(self.solver.p["delta_counts"], snap["delta_counts"])
        wp.copy(self.solver.e["damage"], snap["edge_damage"])
        self.solver.knife_pos = wp.clone(snap["knife_pos"])
        self.solver.knife_active = snap["knife_active"]
        self.solver.step_count = snap["step_count"]

    def _capsule_damage_scalar(self, requires_grad: bool = False):
        """Launch compute_capsule_damage_kernel and return the wp.array scalar buffer."""
        from warp_sim.kernels.obs_reward import compute_capsule_damage_kernel
        buf = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=requires_grad)
        wp.launch(compute_capsule_damage_kernel, dim=self.solver.N,
                  inputs=[self.solver.p["alpha"], self._boundary_mask, buf],
                  device=self.device)
        return buf

    def _compute_tactile_analytic(self):
        """L2: analytic 3D gradient ∂(capsule_damage)/∂(action) at zero action.

        Gradient chain:
          action → new_knife_pos → [ONE physics substep] → alpha → capsule_damage_scalar

        Inside the tape we set:
          - num_iterations=1   (Jacobi kills grad at 4; see memory Test D)
          - cutting_update_interval=1 + step_count=0  (force edge_damage kernel to fire,
            otherwise knife_pos→edge_damage path skipped when step_count %
            cutting_update_interval != 0, giving zero gradient as seen in
            first diagnostic)

        Gradient interpretation: direction in which capsule damage would
        INCREASE if the next action moved the knife. Policy should move AGAINST.
        """
        snap = self._snapshot_solver_state()

        # Only the SSA-fixed kernels (collision, edge_damage) are gradient-safe
        # under Warp 1.12. Jacobi (cluster_solve+apply), particle_damage aggregation,
        # and soft_removal all still have read-then-write patterns that silently
        # zero the gradient. Strategy: BYPASS solver._substep entirely — build a
        # minimal tape that threads only through SSA-validated kernels:
        #
        #   action → apply_action → new_kp
        #          ↓
        #   collision_ssa(pred_pos, new_kp) → collision_delta
        #          ↓
        #   apply_collision_delta → pred_pos
        #          ↓
        #   edge_damage_ssa(pred_pos, new_kp) → edge_damage_delta
        #          ↓
        #   apply_edge_damage_delta → edge_damage
        #          ↓
        #   loss = sum(edge_damage)        ← tactile signal (Phase 3 validated cosine=1.0)
        #
        # Loss is on edge_damage (not capsule alpha) — this is the Phase 3 pattern
        # which gave cosine=1.000 vs FD. Mathematically: "direction in which total
        # tissue fiber damage INCREASES if action moves knife." For the capsule task
        # policy (early learning: knife near capsule), this IS the capsule-damage signal.

        from pf_shape_matching_warp.sim.kernels import (
            collision_kernel_ssa, apply_collision_delta_kernel,
        )
        from pf_shape_matching_warp.sim.pf_damage import (
            update_edge_damage_kernel_ssa, apply_edge_damage_delta_kernel,
        )

        N = self.solver.N
        num_edges = self.solver.e["num_edges"]

        action_leaf = wp.array(
            [wp.vec3(0.0, 0.0, 0.0)],
            dtype=wp.vec3, device=self.device, requires_grad=True,
        )
        loss_buf = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)

        # Phase 3-validated pattern: reset edge_damage + pred_pos to clean "fresh" state
        # before the tape. Warp 1.12 autograd appears to mis-propagate gradient through
        # the SSA kernel when edge_damage holds non-zero carry-over from main physics.
        # State is restored via _restore_solver_state(snap) after the tape.
        self.solver.e["damage"].zero_()
        self.solver.e["damage_delta"].zero_()
        wp.copy(self.solver.p["pred_pos"], self.solver.p["pos"])

        with wp.Tape() as tape:
            # 1. Action → new knife pos
            new_kp = wp.zeros(1, dtype=wp.vec3, device=self.device, requires_grad=True)
            wp.launch(apply_action_kernel, dim=1,
                      inputs=[self.solver.knife_pos, action_leaf, self.action_scale],
                      outputs=[new_kp], device=self.device)

            # 2. Edge damage path — mirrors Phase 3 gate test exactly.
            # SSA kernel writes damage_delta, apply_delta folds into damage,
            # loss sums damage. This pattern achieved FD-cosine=1.000 in isolation.
            wp.launch(update_edge_damage_kernel_ssa, dim=num_edges,
                      inputs=[self.solver.p["pred_pos"], self.solver.p["alpha"],
                              self.solver.e["a"], self.solver.e["b"],
                              self.solver.e["pf_rest"], self.solver.e["damage"],
                              new_kp, self.solver.knife_radius, self.solver.alpha_min,
                              self.solver.e["damage_delta"]],
                      device=self.device)
            wp.launch(apply_edge_damage_delta_kernel, dim=num_edges,
                      inputs=[self.solver.e["damage"], self.solver.e["damage_delta"]],
                      device=self.device)

            # 3. Loss = sum(edge_damage) after delta applied.
            wp.launch(_sum_edge_damage_for_tactile_kernel, dim=num_edges,
                      inputs=[self.solver.e["damage"], loss_buf],
                      device=self.device)

        try:
            tape.backward(loss=loss_buf)
            if action_leaf.grad is None:
                g = np.zeros(3, dtype=np.float32)
            else:
                g = action_leaf.grad.numpy().reshape(3).copy()
                if not np.all(np.isfinite(g)):
                    g = np.zeros(3, dtype=np.float32)
        except Exception:
            g = np.zeros(3, dtype=np.float32)

        # Restore state (solver params untouched — we bypassed _substep, no changes to restore)
        self._restore_solver_state(snap)
        # Bound the feature via tanh; raw magnitude depends on knife-capsule proximity
        return np.tanh(g * self._tactile_scale).astype(np.float32)

    def _compute_tactile_finite_diff(self):
        """Arm Y (method control): 6-point central finite-difference gradient.

        Uses SAME solver-param forcing as L2 analytic (num_iterations=1, cutting every
        step), so FD and analytic measure the SAME operator — essential for cosine
        cross-validation.
        """
        snap = self._snapshot_solver_state()
        # Match L2 analytic's tape-scope: skip Jacobi, force cutting to fire.
        # Keeps FD and analytic measuring the SAME operator for cosine comparability.
        saved_num_iters = self.solver.num_iterations
        saved_cut_interval = self.solver.cutting_update_interval
        self.solver.num_iterations = 0
        self.solver.cutting_update_interval = 1

        eps = self._fd_epsilon
        g = np.zeros(3, dtype=np.float32)
        for i in range(3):
            vals = []
            for sign in (+1.0, -1.0):
                self._restore_solver_state(snap)
                self.solver.num_iterations = 0
                self.solver.cutting_update_interval = 1
                self.solver.step_count = 0
                act = np.zeros(3, dtype=np.float32)
                act[i] = sign * eps
                action_wp = wp.array(
                    [wp.vec3(act[0], act[1], act[2])],
                    dtype=wp.vec3, device=self.device, requires_grad=False,
                )
                new_kp = wp.zeros(1, dtype=wp.vec3, device=self.device, requires_grad=False)
                wp.launch(apply_action_kernel, dim=1,
                          inputs=[self.solver.knife_pos, action_wp, self.action_scale],
                          outputs=[new_kp], device=self.device)
                self.solver.knife_pos = new_kp
                self.solver.knife_active = 1
                self.solver._substep()
                buf = self._capsule_damage_scalar(requires_grad=False)
                wp.synchronize()
                vals.append(float(buf.numpy()[0]))
            g[i] = (vals[0] - vals[1]) / (2.0 * eps)

        self.solver.num_iterations = saved_num_iters
        self.solver.cutting_update_interval = saved_cut_interval
        self._restore_solver_state(snap)
        return np.tanh(g * self._tactile_scale).astype(np.float32)

    def _compute_tactile_feature(self):
        """Dispatch tactile feature source based on obs_mode. Writes self._tactile_g_cap."""
        if self._obs_mode == "l2_analytic":
            self._tactile_g_cap = self._compute_tactile_analytic()
        elif self._obs_mode == "finite_diff":
            self._tactile_g_cap = self._compute_tactile_finite_diff()
        elif self._obs_mode == "random_3d":
            # 3D random gaussian, sampled per step — dimension control (Arm X)
            rng = getattr(self, "np_random", np.random.default_rng())
            self._tactile_g_cap = rng.standard_normal(3).astype(np.float32)
        # "baseline" → no tactile, _tactile_g_cap unused
