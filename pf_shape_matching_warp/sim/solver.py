"""Shape Matching PBD Solver using Warp kernels."""
import warp as wp
from .kernels import (
    predict_kernel,              # legacy — kept for forward-only paths
    predict_kernel_ssa,          # SSA — Stage B (2026-04-22), fixes `v=v*damping`
    collision_kernel,            # legacy (non-gradient-safe) — kept for backcompat
    collision_kernel_ssa,        # SSA version — writes to collision_delta
    apply_collision_delta_kernel,
    clear_deltas_kernel,
    cluster_solve_kernel,            # legacy SVD — kept for forward-only paths
    cluster_solve_kernel_ssa,        # SSA + Müller 2016 polar — Stage C (2026-04-22)
    cluster_quat_update_kernel,      # Müller warm-start state update (isolated)
    reset_cluster_quat_kernel,       # reset warm-start on env.reset
    apply_kernel,                # legacy — kept for forward-only paths
    apply_kernel_ssa,            # SSA — Stage A (2026-04-22), writes to shape_delta
    apply_shape_delta_kernel,    # SSA apply pred_pos += shape_delta
    update_velocity_kernel,
)
from .pf_damage import (
    update_edge_damage_kernel,         # legacy
    update_edge_damage_kernel_ssa,     # SSA version — writes to edge_damage_delta
    apply_edge_damage_delta_kernel,
    update_particle_damage_kernel,
    finalize_particle_damage_kernel,   # legacy — kept for forward-only paths
    finalize_particle_damage_kernel_ssa,   # SSA — Stage A (2026-04-22)
    apply_particle_damage_delta_kernel,    # SSA apply for particle damage
    update_soft_removal_kernel,
)


class ShapeMatchingSolver:
    """GPU Shape Matching PBD solver in Warp.
    Mirrors ShapeMatchingGPUSolver from shape_matching_gpu.cu.
    """

    def __init__(self, particle_arrays, cluster_arrays, edge_arrays=None, device="cuda:0"):
        self.p = particle_arrays
        self.c = cluster_arrays
        self.e = edge_arrays
        self.device = device
        self.N = self.p["pos"].shape[0]
        self.num_clusters = self.c["num_clusters"]

        # Config (same defaults as C++)
        self.dt = 0.005
        self.gravity = wp.vec3(0.0, -9.81, 0.0)
        self.num_iterations = 4
        self.stiffness = 0.6
        self.damping = 0.98
        self.floor_y = 0.0
        self.substeps = 1  # number of substeps per step() call

        # Stage C (2026-04-22): Müller iterations for cluster_solve_kernel_ssa.
        # 8 is the safe default — handles any rotation delta up to ~1 rad with
        # ~7e-5 error. Per-substep rotation delta is typically << 0.1 rad with
        # warm-start, so 8 is generous. Tune via solver.muller_iter if needed.
        self.muller_iter = 8

        # Knife — wp.array for differentiability (gradient flows through wp.Tape)
        self.knife_pos = wp.array(
            [wp.vec3(100.0, 100.0, 100.0)],
            dtype=wp.vec3, device=device, requires_grad=True,
        )
        self.knife_rot = wp.mat33(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        )  # tool rotation (local→world), identity = no rotation
        self.knife_radius = 0.1
        self.knife_active = 0

        # P(F) cutting parameters
        self.alpha_min = 3.0  # lower = smoother P(F) gradient (DiSECt-style)
        self.aspiration_threshold = 0.3
        self.soft_removal_sharpness = 20.0  # sigmoid sharpness for differentiable removal
        self.cutting_update_interval = 50

        self.step_count = 0

    def step(self):
        """One simulation timestep (may contain multiple substeps)."""
        for _sub in range(self.substeps):
            self._substep()

    def _substep(self):
        """One physics substep. dt_sub = self.dt / self.substeps."""
        N = self.N
        dt_sub = self.dt / float(self.substeps)

        # 1. Predict — Stage B (2026-04-22) SSA variant
        # fixes `v = v * damping` silent-zero + hoists old_pos write out of branch.
        wp.launch(
            predict_kernel_ssa,
            dim=N,
            inputs=[
                self.p["pos"], self.p["vel"],
                self.p["pred_pos"], self.p["old_pos"],
                self.p["alpha"], self.p["fixed"],
                self.gravity, dt_sub, self.damping,
            ],
            device=self.device,
        )

        # 2. Collision (after predict) — SSA + delta pattern for Warp 1.12 autograd safety
        self.p["collision_delta"].zero_()
        wp.launch(
            collision_kernel_ssa,
            dim=N,
            inputs=[
                self.p["pred_pos"],
                self.p["alpha"], self.p["fixed"],
                self.floor_y,
                self.knife_pos, self.knife_radius, self.knife_active,
                self.p["collision_delta"],
            ],
            device=self.device,
        )
        wp.launch(
            apply_collision_delta_kernel,
            dim=N,
            inputs=[self.p["pred_pos"], self.p["collision_delta"]],
            device=self.device,
        )

        # 3. Jacobi iterations
        for it in range(self.num_iterations):
            # Clear deltas
            wp.launch(
                clear_deltas_kernel,
                dim=N,
                inputs=[self.p["deltas"], self.p["delta_counts"]],
                device=self.device,
            )

            # Cluster solve — Stage C (2026-04-22) SSA + Müller 2016 polar.
            # Replaces wp.svd3 + det<0 branch with tape-safe iterative quaternion
            # polar decomposition. Warm-started from cluster_quat (persistent).
            wp.launch(
                cluster_solve_kernel_ssa,
                dim=self.num_clusters,
                inputs=[
                    self.p["pred_pos"], self.p["alpha"],
                    self.c["starts"], self.c["sizes"],
                    self.c["particle_ids"], self.c["rest_offsets"],
                    self.c["cluster_quat"], self.muller_iter,
                    self.p["deltas"], self.p["delta_counts"],
                ],
                device=self.device,
            )

            # Apply — SSA pattern (Stage A 2026-04-22):
            # compute correction into shape_delta, then apply pred_pos += shape_delta
            # via atomic_add. Replaces the old tape-unsafe pred_pos R-M-W.
            self.p["shape_delta"].zero_()
            wp.launch(
                apply_kernel_ssa,
                dim=N,
                inputs=[
                    self.p["deltas"], self.p["delta_counts"],
                    self.p["damage"], self.p["alpha"], self.p["fixed"],
                    self.stiffness,
                    self.p["shape_delta"],
                ],
                device=self.device,
            )
            wp.launch(
                apply_shape_delta_kernel,
                dim=N,
                inputs=[self.p["pred_pos"], self.p["shape_delta"]],
                device=self.device,
            )

            # Collision (each iteration) — SSA + delta
            self.p["collision_delta"].zero_()
            wp.launch(
                collision_kernel_ssa,
                dim=N,
                inputs=[
                    self.p["pred_pos"],
                    self.p["alpha"], self.p["fixed"],
                    self.floor_y,
                    self.knife_pos, self.knife_radius, self.knife_active,
                    self.p["collision_delta"],
                ],
                device=self.device,
            )
            wp.launch(
                apply_collision_delta_kernel,
                dim=N,
                inputs=[self.p["pred_pos"], self.p["collision_delta"]],
                device=self.device,
            )

        # 3b. Update cluster_quat warm-start AFTER Jacobi loop — isolated from
        # the differentiable path (kernel reads pred_pos, writes cluster_quat).
        # If we wrote inside cluster_solve_kernel_ssa, the write on a tracked
        # array silently zeroed the backward chain through q_new → R → deltas
        # (Stage C.3 debug 2026-04-22).
        wp.launch(
            cluster_quat_update_kernel,
            dim=self.num_clusters,
            inputs=[
                self.p["pred_pos"], self.p["alpha"],
                self.c["starts"], self.c["sizes"],
                self.c["particle_ids"], self.c["rest_offsets"],
                self.c["cluster_quat"], self.muller_iter,
            ],
            device=self.device,
        )

        # 4. Cutting pipeline (every N steps when knife active)
        if (self.knife_active == 1 and self.e is not None and
                self.step_count % self.cutting_update_interval == 0):
            num_edges = self.e["num_edges"]

            # Edge damage (P(F) computation) — SSA + delta for Warp 1.12 autograd safety
            self.e["damage_delta"].zero_()
            wp.launch(
                update_edge_damage_kernel_ssa,
                dim=num_edges,
                inputs=[
                    self.p["pred_pos"], self.p["alpha"],
                    self.e["a"], self.e["b"],
                    self.e["pf_rest"], self.e["damage"],
                    self.knife_pos, self.knife_radius, self.alpha_min,
                    self.e["damage_delta"],
                ],
                device=self.device,
            )
            wp.launch(
                apply_edge_damage_delta_kernel,
                dim=num_edges,
                inputs=[self.e["damage"], self.e["damage_delta"]],
                device=self.device,
            )

            # Particle damage aggregation (differentiable: atomic_add mean + softplus)
            # Pass 1: accumulate edge damage to particles via atomic_add
            damage_sum = wp.zeros(N, dtype=wp.float32, device=self.device, requires_grad=True)
            damage_count = wp.zeros(N, dtype=wp.float32, device=self.device, requires_grad=True)

            wp.launch(
                update_particle_damage_kernel,
                dim=num_edges,
                inputs=[
                    damage_sum, damage_count,
                    self.p["alpha"],
                    self.e["a"], self.e["b"],
                    self.e["damage"],
                ],
                device=self.device,
            )

            # Pass 2: finalize mean + monotonic softplus update — SSA (Stage A 2026-04-22):
            # compute increment into particle_damage_delta, then apply via atomic_add.
            # Replaces tape-unsafe particle_damage[i] = particle_damage[i] + ... R-M-W.
            self.p["particle_damage_delta"].zero_()
            wp.launch(
                finalize_particle_damage_kernel_ssa,
                dim=N,
                inputs=[damage_sum, damage_count, self.p["damage"],
                        self.p["particle_damage_delta"]],
                device=self.device,
            )
            wp.launch(
                apply_particle_damage_delta_kernel,
                dim=N,
                inputs=[self.p["damage"], self.p["particle_damage_delta"]],
                device=self.device,
            )

            # Soft removal (differentiable sigmoid mask)
            wp.launch(
                update_soft_removal_kernel,
                dim=N,
                inputs=[
                    self.p["damage"], self.p["alpha"], self.p["fixed"],
                    self.aspiration_threshold, self.soft_removal_sharpness,
                ],
                device=self.device,
            )

        # 5. Update velocity
        inv_dt = 1.0 / dt_sub
        wp.launch(
            update_velocity_kernel,
            dim=N,
            inputs=[
                self.p["pos"], self.p["vel"],
                self.p["pred_pos"], self.p["old_pos"],
                self.p["alpha"], self.p["fixed"],
                inv_dt,
            ],
            device=self.device,
        )

        self.step_count += 1

    def set_knife_pos(self, x, y, z):
        """Set knife position. Creates a new wp.array to preserve grad tracking."""
        self.knife_pos = wp.array(
            [wp.vec3(x, y, z)],
            dtype=wp.vec3, device=self.device, requires_grad=True,
        )

    def reset_cluster_quat(self):
        """Reset per-cluster warm-start quaternion to identity. Call on env.reset()
        so Müller iteration starts from a clean state (cluster_quat_init pattern
        in particles.py::init_cluster_arrays, but GPU-side reset without realloc)."""
        wp.launch(
            reset_cluster_quat_kernel, dim=self.num_clusters,
            inputs=[self.c["cluster_quat"]], device=self.device,
        )

    def get_positions_numpy(self):
        """Download positions to CPU numpy array."""
        return self.p["pos"].numpy()

    def get_centroid(self):
        """Compute centroid on CPU."""
        pos = self.get_positions_numpy()
        return pos.mean(axis=0)
