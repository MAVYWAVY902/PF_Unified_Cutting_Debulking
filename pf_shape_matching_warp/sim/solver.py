"""Shape Matching PBD Solver using Warp kernels."""
import warp as wp
from .kernels import (
    predict_kernel,
    collision_kernel,
    clear_deltas_kernel,
    cluster_solve_kernel,
    apply_kernel,
    update_velocity_kernel,
)
from .pf_damage import (
    update_edge_damage_kernel,
    update_particle_damage_kernel,
    finalize_particle_damage_kernel,
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
        self.alpha_min = 10.0
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

        # 1. Predict
        wp.launch(
            predict_kernel,
            dim=N,
            inputs=[
                self.p["pos"], self.p["vel"],
                self.p["pred_pos"], self.p["old_pos"],
                self.p["alpha"], self.p["fixed"],
                self.gravity, dt_sub, self.damping,
            ],
            device=self.device,
        )

        # 2. Collision (after predict)
        wp.launch(
            collision_kernel,
            dim=N,
            inputs=[
                self.p["pred_pos"],
                self.p["alpha"], self.p["fixed"],
                self.floor_y,
                self.knife_pos, self.knife_radius, self.knife_active,
            ],
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

            # Cluster solve (shape matching SVD)
            wp.launch(
                cluster_solve_kernel,
                dim=self.num_clusters,
                inputs=[
                    self.p["pred_pos"], self.p["alpha"],
                    self.c["starts"], self.c["sizes"],
                    self.c["particle_ids"], self.c["rest_offsets"],
                    self.p["deltas"], self.p["delta_counts"],
                ],
                device=self.device,
            )

            # Apply
            wp.launch(
                apply_kernel,
                dim=N,
                inputs=[
                    self.p["pred_pos"], self.p["deltas"], self.p["delta_counts"],
                    self.p["damage"], self.p["alpha"], self.p["fixed"],
                    self.stiffness,
                ],
                device=self.device,
            )

            # Collision (each iteration)
            wp.launch(
                collision_kernel,
                dim=N,
                inputs=[
                    self.p["pred_pos"],
                    self.p["alpha"], self.p["fixed"],
                    self.floor_y,
                    self.knife_pos, self.knife_radius, self.knife_active,
                ],
                device=self.device,
            )

        # 4. Cutting pipeline (every N steps when knife active)
        if (self.knife_active == 1 and self.e is not None and
                self.step_count % self.cutting_update_interval == 0):
            num_edges = self.e["num_edges"]

            # Edge damage (P(F) computation)
            wp.launch(
                update_edge_damage_kernel,
                dim=num_edges,
                inputs=[
                    self.p["pred_pos"], self.p["alpha"],
                    self.e["a"], self.e["b"],
                    self.e["pf_rest"], self.e["damage"],
                    self.knife_pos, self.knife_radius, self.alpha_min,
                ],
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

            # Pass 2: finalize mean + monotonic softplus update
            wp.launch(
                finalize_particle_damage_kernel,
                dim=N,
                inputs=[damage_sum, damage_count, self.p["damage"]],
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

    def get_positions_numpy(self):
        """Download positions to CPU numpy array."""
        return self.p["pos"].numpy()

    def get_centroid(self):
        """Compute centroid on CPU."""
        pos = self.get_positions_numpy()
        return pos.mean(axis=0)
