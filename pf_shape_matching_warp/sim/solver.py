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
    # Phase 2B (2026-05-07): split cluster_solve into 5 SSA kernels to fix Warp 1.13
    # silent-fail in monolithic kernel. POC v3 verified: 0.94 ratio at num_iter=6
    # vs broken cluster_solve_kernel_ssa's 4.5×10¹⁰. See
    # project_layer_a_phase2b_poc_complete_2026_05_07.md.
    cluster_compute_sums_kernel_ssa,
    cluster_normalize_centroids_kernel_ssa,
    cluster_compute_covariance_kernel_ssa,
    cluster_polar_rotation_kernel_ssa,
    cluster_goal_scatter_kernel_ssa,
    cluster_quat_update_kernel,      # Müller warm-start state update (isolated)
    reset_cluster_quat_kernel,       # reset warm-start on env.reset
    apply_kernel,                # legacy — kept for forward-only paths
    apply_kernel_ssa,            # SSA — Stage A (2026-04-22), writes to shape_delta
    apply_kernel_ssa_param_arr,    # Day 3b (2026-05-23): stiffness as wp.array(1,) global
    apply_kernel_ssa_param_arr_pp, # Stage A (2026-05-24): per-particle stiffness wp.array(N,)
    apply_shape_delta_kernel,    # legacy in-place atomic_add (forward-only safe)
    # Layer A SSA (2026-05-05): cycle-scoped buffer pattern for autograd
    # correctness through 6-iter Jacobi loop. See
    # project_layer_a_fix_implementation_plan_2026_05_05.md.
    apply_shape_delta_ssa_kernel,
    apply_collision_delta_ssa_kernel,
    # Self-collision (soft penalty, differentiable) — 2026-06-09, optional (default off)
    self_collision_kernel_ssa,
    apply_self_collision_ssa_kernel,
    update_velocity_kernel,
    # Phase 2C (2026-05-07): SSA variant of update_velocity replaces alpha-based
    # early return with wp.where mask + cycle-scoped output buffers. Unblocks
    # with-cutting gradient correctness (0.094 → 0.94 ratio).
    update_velocity_kernel_ssa,
)
from .pf_damage import (
    update_edge_damage_kernel,         # legacy
    update_edge_damage_kernel_ssa,     # SSA version — writes to edge_damage_delta
    apply_edge_damage_delta_kernel,    # legacy in-place atomic_add apply
    update_particle_damage_kernel,
    finalize_particle_damage_kernel,   # legacy — kept for forward-only paths
    finalize_particle_damage_kernel_ssa,   # SSA — Stage A (2026-04-22)
    apply_particle_damage_delta_kernel,    # legacy in-place atomic_add apply
    update_soft_removal_kernel,        # legacy in-place write
    # Path B+ Stage B (2026-05-01): cycle-scoped buffer SSA path
    update_soft_removal_kernel_ssa,
    apply_edge_damage_ssa_kernel,
    apply_particle_damage_ssa_kernel,
    # Day 6b (2026-05-22): wp.array theta path for ReSECT autograd through θ
    update_edge_damage_kernel_ssa_param_arr,
    update_soft_removal_kernel_ssa_param_arr,
    update_soft_removal_kernel_ssa_param_arr_cusa,
    update_soft_removal_kernel_ssa_param_arr_pp_tau_cusa,
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
        # 2026-04-24: switched from Y-up (gravity=-Y) to Z-up (gravity=-Z) so the
        # torus phantom lies flat in the XY ground plane under both physics and
        # default matplotlib rendering. Floor kernels check pos[2] against floor_y
        # (variable name kept; refers to "floor value" not Y-axis specifically).
        self.dt = 0.005
        self.gravity = wp.vec3(0.0, 0.0, -9.81)
        self.num_iterations = 6      # 2026-04-25: 4 → 6 (firmer cluster convergence per substep)
        self.stiffness = 0.85        # 2026-04-25: 0.6 → 0.85 on corrected Z-up geometry to reduce push-induced wobble
        self.damping = 0.98
        self.floor_y = 0.0  # historical name; now refers to floor_z after Z-up switch
        self.floor_friction = 0.85   # 2026-04-25: tangent slip damping [0..1] in collision_kernel_ssa
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
        # 2026-06-13: master switch for the REMOVAL/cutting block (decoupled from the
        # knife collider). True = debulk (cut). False = non-destructive PRESS (indent only,
        # α stays 1, topology intact) → stiffness-calibration mode. See _substep step 4.
        self.enable_cutting = True

        # CUSA parity mode (2026-05-25): cpp_sim-style inside-knife gate +
        # persistence on alpha. Eliminates zombie particles by restricting alpha
        # drop to particles geometrically inside the knife sphere AND damaged.
        # See pf_damage.update_soft_removal_kernel_ssa_param_arr_cusa.
        # When True (default), the param-arr path uses the CUSA kernel.
        self.use_cusa_aspiration = True

        self.step_count = 0

        # Path B+ Stage B (2026-05-01): cycle-scoped buffer SSA path for autograd-safe
        # cutting parameter calibration. When False (default), uses legacy in-place
        # update_soft_removal_kernel + atomic_add applies (forward-only safe, backward
        # produces silent zero on cutting params). When True, dispatches to SSA variants
        # with cycle-scoped alpha/edge_damage/particle_damage buffer LISTS.
        #
        # POC verification: tools/poc_soft_removal_ssa_v3_full.py
        #   sharpness rel_err 0.95% vs FD; alpha_min rel_err 0.11%; zero drift.
        # Reference: NVIDIA Warp example_fluid_checkpoint.py lines 197-303.
        self.use_ssa_soft_removal = False
        # Phase 2B (2026-05-07): split cluster_solve mode for Layer A backward
        # correctness. When True, dispatches to 5-kernel SSA chain
        # (compute_sums → normalize → covariance → polar → scatter) instead of
        # monolithic cluster_solve_kernel_ssa. Eliminates Warp 1.13 silent-fail
        # at num_iter=6 (4.5×10¹⁰ → 0.94 ratio per POC v3 2026-05-07).
        # Default True after Day 5-6 dual-mode regression (37/38, same as Stage B
        # baseline) + Day 9-10 perf audit (1.12× legacy runtime, 12% overhead).
        # Set False to fall back to monolithic kernel for RL training where
        # autograd is not needed (saves 12% runtime).
        self.use_split_cluster_solve = True
        # Buffer-list capacity = max number of cutting events expected in one tape
        # backward horizon. Sized for Path 4 trajectory-opt episodes: 1024 steps with
        # cutting_update_interval=1 → ≤256 cycles per tape. Memory cost at 14.8k
        # particle mesh × 3 state buffers × 4 bytes × 256 ≈ 45 MB (negligible).
        # 2026-05-01 Stage B Day 4.5: bumped 32 → 256 after SSA regression hit cap
        # at cycle 32 in m3_4_finite (50-step horizon × multi-substep). Two-stage
        # strategy per memory `project_path_b_plus_poc_verified_2026_04_30.md`:
        # Route 1 now (this bump); Route 2 sqrt-segmentation à la
        # example_fluid_checkpoint.py L325-407 only if/when Path 4 outgrows this.
        self.ssa_buffer_capacity = 256
        self._ssa_alpha_buffers = None              # list[wp.array(N)] — alloc on first use
        self._ssa_edge_damage_buffers = None        # list[wp.array(num_edges)]
        self._ssa_particle_damage_buffers = None    # list[wp.array(N)]
        self._ssa_cycle_index = 0                   # which buffer slot to write next

        # Self-collision (soft penalty, differentiable) — 2026-06-09. OPTIONAL, default
        # OFF so all calibration/RL/gradient-verified paths are untouched. When True, a
        # particle-particle repulsion runs after the Jacobi loop so a SEVERED fragment
        # (made free by warp_cluster_split) lands on the base instead of passing through
        # (warp's collision_kernel only does floor+knife). self_collision_dist MUST be
        # < rest nearest-neighbor spacing or bonded neighbors fight shape-matching.
        # Caller sets self_collision_dist (e.g. 0.5×mean_edge) + stiffness before step().
        # α-weighted velocity carry in update_velocity_kernel_ssa (2026-06-12).
        # 0.0 = legacy (bit-identical default); 1.0 = scale carried velocity by α to
        # tame half-removed "zombie" debris flyers (forward-stability fix, regime D in
        # diag_blowup_rootcause.py). Gradient-neutral for per-frame backward — see kernel
        # docstring. Default OFF so production behavior is unchanged until verified.
        self.alpha_vel_scale = 0.0
        # Müller 2007 iteration-count-independent stiffness remap (2026-06-12):
        # k_eff = 1-(1-k)^(1/num_iterations) in apply_kernel_ssa_param_arr_pp. False = legacy
        # (stiff_exp=1.0). True removes the eff-stiffness-grows-with-num_iter inflation that is
        # the literature-named cause of the cutting blow-up (XPBD "iteration-count-dependent
        # stiffness"). Default OFF until verified.
        self.muller_stiff_remap = False
        # (b) 2026-06-15: τ-coupled cohesive softening in apply_kernel_ssa_param_arr_pp.
        # False = legacy raw-damage softening (τ-blind position channel). True = soften by
        # d/τ ratio → high-τ stays stiff longer → pre-fracture deformation carries τ → wakes
        # the dense position channel (so a position/chamfer loss can read τ). Default OFF.
        # Only active when both stiffness AND threshold are per-particle (N,). See memory
        # project_tau_observability_physics (∂α/∂τ band) — this moves τ into position channel.
        self.tau_soften = False
        self.use_self_collision = False
        self.self_collision_dist = 0.005
        self.self_collision_stiffness = 1.0
        self.self_collision_iters = 4          # Jacobi-style passes/substep to resolve
                                               # penetration (one weak pass can't stop a
                                               # fast rigid chunk; cpp iterates in-loop).
        self._self_collision_grid = None       # lazy wp.HashGrid

        # Day 6b (2026-05-22): optional wp.array path for θ — when set, the
        # cutting block uses *_ssa_param_arr kernels so autograd flows back to
        # theta via wp.Tape (enabling Warp ↔ PyTorch autograd bridge).
        # Keys: 'alpha_min', 'sharpness', 'threshold', 'knife_radius' as wp.array(shape=(1,))
        # Set via set_theta_wp({...}); pass None to revert to legacy float path.
        self.theta_wp = None

    def _alloc_ssa_buffers(self):
        """Lazy allocation of cycle-scoped buffer lists for SSA cutting path.

        Called on first cutting event when use_ssa_soft_removal=True. Allocates
        ssa_buffer_capacity buffers for each of: alpha, edge_damage, particle_damage.
        All buffers requires_grad=True per NVIDIA Warp differentiability guide
        (officially documented "allocate separate arrays" prescription).
        """
        if self._ssa_alpha_buffers is not None:
            return  # already allocated

        N = self.N
        num_edges = self.e["num_edges"] if self.e is not None else 0

        import numpy as _np
        # alpha buffers — initialize all to ones (healthy state) like solver init
        self._ssa_alpha_buffers = [
            wp.array(_np.ones(N, dtype=_np.float32), dtype=wp.float32,
                     device=self.device, requires_grad=True)
            for _ in range(self.ssa_buffer_capacity)
        ]
        # edge_damage and particle_damage buffers — initialize to zero (no damage)
        self._ssa_edge_damage_buffers = [
            wp.zeros(num_edges, dtype=wp.float32, device=self.device, requires_grad=True)
            for _ in range(self.ssa_buffer_capacity)
        ]
        self._ssa_particle_damage_buffers = [
            wp.zeros(N, dtype=wp.float32, device=self.device, requires_grad=True)
            for _ in range(self.ssa_buffer_capacity)
        ]
        self._ssa_cycle_index = 0

    def reset_ssa_buffers(self):
        """Reset cycle index + zero all SSA buffer pools. Call BEFORE starting
        a new forward pass that will be backproped through (e.g. before each
        calibration iteration's tape recording). Forward-only RL eval can skip.

        IMPORTANT: this resets ONLY the SSA buffer pools (alpha/edge_damage/
        particle_damage). It does NOT reset pos/vel/cluster_quat/main alpha
        buffer. For a complete "rest state" reset suitable for sys-ID's
        per-trajectory restart, use reset_to_rest(rest_vertices) instead.
        """
        if not self.use_ssa_soft_removal or self._ssa_alpha_buffers is None:
            return
        for buf in self._ssa_alpha_buffers:
            # alpha buffers re-initialized to 1 (healthy)
            buf.fill_(1.0)
        for buf in self._ssa_edge_damage_buffers:
            buf.zero_()
        for buf in self._ssa_particle_damage_buffers:
            buf.zero_()
        self._ssa_cycle_index = 0

    def reset_to_rest(self, rest_vertices):
        """Full reset of ALL evolved state to a clean rest pose.

        Sys-ID per-trajectory restart: when starting a calibration trajectory
        from scratch, the solver state must be reset to the mesh rest pose with
        zero velocity, zero damage, and identity cluster rotations. Without
        all of these, residual state from a prior call contaminates the new
        trajectory (silent bug — physics looks plausible but Test 3 ≠ Test 2).

        Resets:
          - pos / old_pos / pred_pos      ← rest_vertices
          - vel                            ← 0
          - alpha (main + SSA pools)      ← 1
          - damage (main + SSA pools)     ← 0
          - edge_damage (main + SSA pools) ← 0
          - cluster_quat                   ← identity (Müller warm-start)
          - step_count                     ← 0
          - SSA cycle_index                ← 0

        Args:
            rest_vertices: (N, 3) np.array of rest positions

        Note: knife pose / knife_active are NOT reset (they're per-frame
        controls, not evolved state).
        """
        import numpy as _np
        rest_wp = wp.array(_np.asarray(rest_vertices, dtype=_np.float32),
                            dtype=wp.vec3, device=self.device)
        wp.copy(self.p["pos"], rest_wp)
        wp.copy(self.p["old_pos"], rest_wp)
        wp.copy(self.p["pred_pos"], rest_wp)
        self.p["vel"].zero_()
        # Main (non-SSA) buffers
        self.p["alpha"].fill_(1.0)
        self.p["damage"].zero_()
        if self.e is not None:
            self.e["damage"].zero_()
        # SSA buffer pools (alpha → 1, damages → 0)
        self.reset_ssa_buffers()
        # Cluster warm-start quaternions back to identity
        wp.launch(reset_cluster_quat_kernel, dim=self.num_clusters,
                  inputs=[self.c["cluster_quat"]], device=self.device)
        self.step_count = 0

    def set_theta_wp(self, theta_dict):
        """Set θ as wp.arrays for autograd-aware forward (Day 6b + Day 3b).

        theta_dict: {'alpha_min', 'sharpness', 'threshold', 'knife_radius',
                     'stiffness' (optional, Day 3b 2026-05-23)}
                    each a wp.array(shape=(1,), dtype=float32, requires_grad=True)
        Pass None to revert to the legacy Python-float path.

        When set:
          - cutting block uses *_ssa_param_arr kernels (Day 6b)
          - shape-matching apply uses apply_kernel_ssa_param_arr (Day 3b) IFF
            'stiffness' key is present; otherwise stiffness stays as float.

        Caller responsible for keeping the wp.arrays alive while step() runs.
        """
        if theta_dict is None:
            self.theta_wp = None
            return
        required = {"alpha_min", "sharpness", "threshold", "knife_radius"}
        if not required.issubset(theta_dict.keys()):
            raise ValueError(f"theta_dict missing keys: {required - set(theta_dict.keys())}")
        self.theta_wp = dict(theta_dict)
        # Also update Python float mirror (for consistency with read sites
        # that don't go through the cutting kernel — e.g., logging)
        self.alpha_min = float(theta_dict["alpha_min"].numpy()[0])
        self.soft_removal_sharpness = float(theta_dict["sharpness"].numpy()[0])
        self.aspiration_threshold = float(theta_dict["threshold"].numpy()[0])
        self.knife_radius = float(theta_dict["knife_radius"].numpy()[0])
        if "stiffness" in theta_dict:
            stiff_arr = theta_dict["stiffness"]
            if stiff_arr.shape[0] == 1:
                # Global scalar (Day 3b backward-compat path)
                self.stiffness = float(stiff_arr.numpy()[0])
            # else: per-particle (N,) — leave self.stiffness as-is (mirror unused),
            # solver._substep will detect shape and dispatch to *_pp kernel.

    def _current_alpha(self):
        """Returns the wp.array buffer that should be read for alpha at the
        CURRENT moment in the substep. In SSA mode, this is alpha_buffers[
        ssa_cycle_index] — the most recently written cycle's output. In legacy
        mode, this is self.p["alpha"] (the in-place buffer).

        After a cutting event in SSA mode, ssa_cycle_index advances, so
        subsequent calls return the new cycle's buffer. This is the design
        that keeps each buffer single-write multi-read (SSA-clean).

        IMPORTANT: callers reading alpha for visualization / RL observation
        OUTSIDE the tape should use this method (or self._ssa_alpha_buffers[-1]).
        Reading raw self.p["alpha"] in SSA mode would return stale init value.
        """
        if self.use_ssa_soft_removal and self._ssa_alpha_buffers is not None:
            return self._ssa_alpha_buffers[self._ssa_cycle_index]
        return self.p["alpha"]

    def snapshot_to_p_arrays(self):
        """Mirror current SSA state into self.p["alpha"], self.p["damage"],
        and self.e["damage"] for visualization / RL observation code that
        reads these directly.

        Call OUTSIDE the tape recording (after tape.backward() if any). Calling
        this inside a tape would create R-W-R on these arrays, breaking
        gradients of subsequent _substep calls. This method is a no-op in
        legacy mode (use_ssa_soft_removal=False)."""
        if not self.use_ssa_soft_removal or self._ssa_alpha_buffers is None:
            return
        c = self._ssa_cycle_index
        wp.copy(self.p["alpha"], self._ssa_alpha_buffers[c])
        wp.copy(self.p["damage"], self._ssa_particle_damage_buffers[c])
        if self.e is not None:
            wp.copy(self.e["damage"], self._ssa_edge_damage_buffers[c])

    def step(self):
        """One simulation timestep (may contain multiple substeps)."""
        for _sub in range(self.substeps):
            self._substep()

    def _substep(self):
        """One physics substep. dt_sub = self.dt / self.substeps."""
        N = self.N
        dt_sub = self.dt / float(self.substeps)

        # Path B+ Stage B (2026-05-01): in SSA mode, ensure cycle-scoped buffers
        # are allocated up-front. All physics kernels then read alpha via
        # self._current_alpha() which returns the appropriate buffer.
        # Phase 2B (2026-05-07): split mode also needs pred_pos cycle-scoping.
        if self.use_ssa_soft_removal:
            self._alloc_ssa_buffers()
        if self.use_ssa_soft_removal or self.use_split_cluster_solve:
            # Layer A SSA (2026-05-05): allocate fresh pred_pos at substep entry
            # to avoid cross-substep R-W-R. Without this, substep N's
            # predict_kernel_ssa would overwrite the buffer that substep N-1's
            # final Jacobi iter wrote, contaminating backward through that iter.
            # Reference: Newton solver_xpbd.py:156-202 (per-substep state alloc).
            self.p["pred_pos"] = wp.empty_like(self.p["pos"])

        # Phase 2B (2026-05-07): when use_split_cluster_solve, pre-allocate ALL
        # cycle-scoped buffer pools UPFRONT (before predict launch) so they're
        # available for both pre-loop and in-loop collision. Mirrors POC v3
        # verified pattern. Each Jacobi iter writes to its own pool slot —
        # eliminates R-W-R inside the Tape and the F3 trigger (variable divisor
        # + atomic_add scatter co-located).
        if self.use_split_cluster_solve:
            NC = self.num_clusters
            _split_coll_delta_pre = wp.zeros(N, dtype=wp.vec3, device=self.device, requires_grad=True)
            _split_coll_delta_pool = [
                wp.zeros(N, dtype=wp.vec3, device=self.device, requires_grad=True)
                for _ in range(self.num_iterations)
            ]
            _split_deltas_pool = [
                wp.zeros(N, dtype=wp.vec3, device=self.device, requires_grad=True)
                for _ in range(self.num_iterations)
            ]
            _split_delta_counts_pool = [
                wp.zeros(N, dtype=wp.int32, device=self.device)
                for _ in range(self.num_iterations)
            ]
            _split_shape_delta_pool = [
                wp.zeros(N, dtype=wp.vec3, device=self.device, requires_grad=True)
                for _ in range(self.num_iterations)
            ]
            _split_cx_sum_pool = [
                wp.zeros(NC, dtype=wp.vec3, device=self.device, requires_grad=True)
                for _ in range(self.num_iterations)
            ]
            _split_qc_sum_pool = [
                wp.zeros(NC, dtype=wp.vec3, device=self.device, requires_grad=True)
                for _ in range(self.num_iterations)
            ]
            _split_tw_pool = [
                wp.zeros(NC, dtype=wp.float32, device=self.device, requires_grad=True)
                for _ in range(self.num_iterations)
            ]
            _split_cx_mean_pool = [
                wp.zeros(NC, dtype=wp.vec3, device=self.device, requires_grad=True)
                for _ in range(self.num_iterations)
            ]
            _split_qc_mean_pool = [
                wp.zeros(NC, dtype=wp.vec3, device=self.device, requires_grad=True)
                for _ in range(self.num_iterations)
            ]
            _split_A_pool = [
                wp.zeros(NC, dtype=wp.mat33, device=self.device, requires_grad=True)
                for _ in range(self.num_iterations)
            ]
            _split_R_pool = [
                wp.zeros(NC, dtype=wp.mat33, device=self.device, requires_grad=True)
                for _ in range(self.num_iterations)
            ]

        # 1. Predict — Stage B (2026-04-22) SSA variant
        # fixes `v = v * damping` silent-zero + hoists old_pos write out of branch.
        wp.launch(
            predict_kernel_ssa,
            dim=N,
            inputs=[
                self.p["pos"], self.p["vel"],
                self.p["pred_pos"], self.p["old_pos"],
                self._current_alpha(), self.p["fixed"],
                self.gravity, dt_sub, self.damping,
            ],
            device=self.device,
        )

        # 2. Collision (after predict) — SSA + delta pattern for Warp 1.12 autograd safety
        # Phase 2B (2026-05-07): in split mode use cycle-scoped _split_coll_delta_pre
        # to avoid R-W-R on the shared self.p["collision_delta"] buffer.
        if self.use_split_cluster_solve:
            _coll_delta_pre = _split_coll_delta_pre
        else:
            self.p["collision_delta"].zero_()
            _coll_delta_pre = self.p["collision_delta"]
        wp.launch(
            collision_kernel_ssa,
            dim=N,
            inputs=[
                self.p["pred_pos"],
                self.p["pos"],                 # friction anchor (start-of-substep pos)
                self._current_alpha(), self.p["fixed"],
                self.floor_y, self.floor_friction,
                self.knife_pos, self.knife_radius, self.knife_active,
                _coll_delta_pre,
            ],
            device=self.device,
        )
        # Layer A SSA dispatch (2026-05-05): cycle-scoped buffer when SSA mode
        # OR split mode; legacy in-place atomic_add otherwise.
        if self.use_ssa_soft_removal or self.use_split_cluster_solve:
            new_pred_pos = wp.empty_like(self.p["pred_pos"])
            wp.launch(
                apply_collision_delta_ssa_kernel,
                dim=N,
                inputs=[self.p["pred_pos"], _coll_delta_pre],
                outputs=[new_pred_pos],
                device=self.device,
            )
            self.p["pred_pos"] = new_pred_pos
        else:
            wp.launch(
                apply_collision_delta_kernel,
                dim=N,
                inputs=[self.p["pred_pos"], _coll_delta_pre],
                device=self.device,
            )

        # 3. Jacobi iterations
        # NOTE: split-mode buffer pools (incl. _split_coll_delta_pool) are
        # pre-allocated above (line ~253), before pre-loop collision, so they're
        # available for both pre-loop AND in-loop collision.

        for it in range(self.num_iterations):
            if self.use_split_cluster_solve:
                # Phase 2B (2026-05-07): 5-kernel split chain.
                # Each iter writes to fresh pool slot — F3 trigger eliminated
                # by isolating variable divisor (K2) from atomic_add scatter (K5).
                _deltas_it = _split_deltas_pool[it]
                _delta_counts_it = _split_delta_counts_pool[it]
                _shape_delta_it = _split_shape_delta_pool[it]

                wp.launch(
                    cluster_compute_sums_kernel_ssa,
                    dim=NC,
                    inputs=[
                        self.p["pred_pos"], self._current_alpha(),
                        self.c["starts"], self.c["sizes"],
                        self.c["particle_ids"], self.c["rest_offsets"],
                        _split_cx_sum_pool[it], _split_qc_sum_pool[it],
                        _split_tw_pool[it],
                    ],
                    device=self.device,
                )
                wp.launch(
                    cluster_normalize_centroids_kernel_ssa,
                    dim=NC,
                    inputs=[
                        self.c["sizes"],
                        _split_cx_sum_pool[it], _split_qc_sum_pool[it],
                        _split_tw_pool[it],
                        _split_cx_mean_pool[it], _split_qc_mean_pool[it],
                    ],
                    device=self.device,
                )
                wp.launch(
                    cluster_compute_covariance_kernel_ssa,
                    dim=NC,
                    inputs=[
                        self.p["pred_pos"], self._current_alpha(),
                        self.c["starts"], self.c["sizes"],
                        self.c["particle_ids"], self.c["rest_offsets"],
                        _split_cx_mean_pool[it], _split_qc_mean_pool[it],
                        _split_A_pool[it],
                    ],
                    device=self.device,
                )
                wp.launch(
                    cluster_polar_rotation_kernel_ssa,
                    dim=NC,
                    inputs=[
                        self.c["sizes"],
                        _split_A_pool[it], self.c["cluster_quat"],
                        self.muller_iter,
                        _split_R_pool[it],
                    ],
                    device=self.device,
                )
                wp.launch(
                    cluster_goal_scatter_kernel_ssa,
                    dim=NC,
                    inputs=[
                        self.p["pred_pos"], self._current_alpha(),
                        self.c["starts"], self.c["sizes"],
                        self.c["particle_ids"], self.c["rest_offsets"],
                        _split_cx_mean_pool[it], _split_qc_mean_pool[it],
                        _split_R_pool[it],
                        _deltas_it, _delta_counts_it,
                    ],
                    device=self.device,
                )
            else:
                # Legacy: monolithic cluster_solve_kernel_ssa (Stage C 2026-04-22).
                # Reused self.p["deltas"] / delta_counts buffers, cleared each iter.
                # SILENT-FAIL at num_iter≥2 due to F3 trigger (see Phase 2B notes).
                _deltas_it = self.p["deltas"]
                _delta_counts_it = self.p["delta_counts"]
                _shape_delta_it = self.p["shape_delta"]

                wp.launch(
                    clear_deltas_kernel,
                    dim=N,
                    inputs=[_deltas_it, _delta_counts_it],
                    device=self.device,
                )

                # Cluster solve — Stage C (2026-04-22) SSA + Müller 2016 polar.
                # Replaces wp.svd3 + det<0 branch with tape-safe iterative quaternion
                # polar decomposition. Warm-started from cluster_quat (persistent).
                wp.launch(
                    cluster_solve_kernel_ssa,
                    dim=self.num_clusters,
                    inputs=[
                        self.p["pred_pos"], self._current_alpha(),
                        self.c["starts"], self.c["sizes"],
                        self.c["particle_ids"], self.c["rest_offsets"],
                        self.c["cluster_quat"], self.muller_iter,
                        _deltas_it, _delta_counts_it,
                    ],
                    device=self.device,
                )

            # Apply — SSA pattern (Stage A 2026-04-22):
            # compute correction into shape_delta, then apply pred_pos += shape_delta
            # via atomic_add. In legacy mode shape_delta is reused (zero each iter);
            # in split mode shape_delta is cycle-scoped pool[it] (already zero from alloc).
            if not self.use_split_cluster_solve:
                _shape_delta_it.zero_()
            # Day 3b / Stage A: dispatch based on stiffness shape in theta_wp
            #   - no theta_wp        → legacy float path apply_kernel_ssa
            #   - stiffness shape (1,) → apply_kernel_ssa_param_arr (Day 3b global)
            #   - stiffness shape (N,) → apply_kernel_ssa_param_arr_pp (Stage A per-particle)
            if self.theta_wp is not None and "stiffness" in self.theta_wp:
                stiff_arr = self.theta_wp["stiffness"]
                if stiff_arr.shape[0] == 1:
                    wp.launch(
                        apply_kernel_ssa_param_arr,
                        dim=N,
                        inputs=[
                            _deltas_it, _delta_counts_it,
                            self.p["damage"], self._current_alpha(), self.p["fixed"],
                            stiff_arr,
                            _shape_delta_it,
                        ],
                        device=self.device,
                    )
                else:
                    # Stage A: per-particle stiffness
                    assert stiff_arr.shape[0] == N, (
                        f"per-particle stiffness must have shape (N,) = ({N},), "
                        f"got shape ({stiff_arr.shape[0]},)"
                    )
                    _stiff_exp = (1.0 / float(self.num_iterations)) if self.muller_stiff_remap else 1.0
                    # (b) 2026-06-15: τ-coupled softening. Active only when threshold is also
                    # per-particle (N,) AND self.tau_soften. Pass the SAME threshold array used
                    # by the α kernel (so τ gradient is consistent across both channels). When
                    # off, pass damage as a harmless (N,) placeholder (wp.where evaluates both
                    # branches; tau_soften=0 selects the raw-damage branch → exact legacy).
                    _thr = self.theta_wp.get("threshold") if self.theta_wp is not None else None
                    _tau_pp_ok = (_thr is not None and _thr.shape[0] == N)
                    _tau_soft = 1.0 if (self.tau_soften and _tau_pp_ok) else 0.0
                    _tau_arr = _thr if _tau_pp_ok else self.p["damage"]
                    wp.launch(
                        apply_kernel_ssa_param_arr_pp,
                        dim=N,
                        inputs=[
                            _deltas_it, _delta_counts_it,
                            self.p["damage"], self._current_alpha(), self.p["fixed"],
                            stiff_arr,
                            _stiff_exp,
                            _tau_arr,
                            _tau_soft,
                            _shape_delta_it,
                        ],
                        device=self.device,
                    )
            else:
                wp.launch(
                    apply_kernel_ssa,
                    dim=N,
                    inputs=[
                        _deltas_it, _delta_counts_it,
                        self.p["damage"], self._current_alpha(), self.p["fixed"],
                        self.stiffness,
                        _shape_delta_it,
                    ],
                    device=self.device,
                )
            # Layer A SSA dispatch (2026-05-05): apply shape_delta via
            # cycle-scoped fresh buffer when SSA OR split mode, legacy atomic_add otherwise.
            if self.use_ssa_soft_removal or self.use_split_cluster_solve:
                new_pred_pos = wp.empty_like(self.p["pred_pos"])
                wp.launch(
                    apply_shape_delta_ssa_kernel,
                    dim=N,
                    inputs=[self.p["pred_pos"], _shape_delta_it],
                    outputs=[new_pred_pos],
                    device=self.device,
                )
                self.p["pred_pos"] = new_pred_pos
            else:
                wp.launch(
                    apply_shape_delta_kernel,
                    dim=N,
                    inputs=[self.p["pred_pos"], _shape_delta_it],
                    device=self.device,
                )

            # Collision (each iteration) — SSA + delta
            # Phase 2B (2026-05-07): in split mode use cycle-scoped pool[it] to
            # avoid R-W-R on shared self.p["collision_delta"] across Jacobi iters.
            if self.use_split_cluster_solve:
                _coll_delta_it = _split_coll_delta_pool[it]
            else:
                self.p["collision_delta"].zero_()
                _coll_delta_it = self.p["collision_delta"]
            wp.launch(
                collision_kernel_ssa,
                dim=N,
                inputs=[
                    self.p["pred_pos"],
                    self.p["pos"],                 # friction anchor (substep-stable)
                    self._current_alpha(), self.p["fixed"],
                    self.floor_y, self.floor_friction,
                    self.knife_pos, self.knife_radius, self.knife_active,
                    _coll_delta_it,
                ],
                device=self.device,
            )
            # Layer A SSA dispatch (2026-05-05): apply collision_delta via
            # cycle-scoped fresh buffer when SSA OR split mode.
            if self.use_ssa_soft_removal or self.use_split_cluster_solve:
                new_pred_pos = wp.empty_like(self.p["pred_pos"])
                wp.launch(
                    apply_collision_delta_ssa_kernel,
                    dim=N,
                    inputs=[self.p["pred_pos"], _coll_delta_it],
                    outputs=[new_pred_pos],
                    device=self.device,
                )
                self.p["pred_pos"] = new_pred_pos
            else:
                wp.launch(
                    apply_collision_delta_kernel,
                    dim=N,
                    inputs=[self.p["pred_pos"], _coll_delta_it],
                    device=self.device,
                )

        # 3b. Update cluster_quat warm-start AFTER Jacobi loop — isolated from
        # the differentiable path (kernel reads pred_pos, writes cluster_quat).
        # If we wrote inside cluster_solve_kernel_ssa, the write on a tracked
        # array silently zeroed the backward chain through q_new → R → deltas
        # (Stage C.3 debug 2026-04-22).
        # Phase 2B (2026-05-07): verified by Day 4-D test that this kernel does
        # NOT contribute to backward bias when collision_delta is properly
        # cycle-scoped (the real root cause of substeps=1 ratio 2.02).
        wp.launch(
            cluster_quat_update_kernel,
            dim=self.num_clusters,
            inputs=[
                self.p["pred_pos"], self._current_alpha(),
                self.c["starts"], self.c["sizes"],
                self.c["particle_ids"], self.c["rest_offsets"],
                self.c["cluster_quat"], self.muller_iter,
            ],
            device=self.device,
        )

        # 3c. Self-collision (soft penalty, differentiable) — OPTIONAL, after the Jacobi
        # loop so a severed free fragment lands on the base instead of passing through.
        # Detached hash-grid neighbor search + smooth in-tape repulsion (SSA: write to a
        # fresh sc_delta, apply into a fresh pred_pos buffer). Default off → no-op.
        if self.use_self_collision:
            if self._self_collision_grid is None:
                self._self_collision_grid = wp.HashGrid(64, 64, 64, device=self.device)
            # Jacobi-style: rebuild grid + push each pass so deep penetration is resolved
            # progressively (a single weak pass can't stop a fast rigid fragment).
            for _sc_it in range(self.self_collision_iters):
                # build is non-differentiable spatial indexing (detached); cell size = dist.
                self._self_collision_grid.build(self.p["pred_pos"], self.self_collision_dist)
                sc_delta = wp.zeros(N, dtype=wp.vec3, device=self.device, requires_grad=True)
                sc_count = wp.zeros(N, dtype=wp.int32, device=self.device)
                wp.launch(
                    self_collision_kernel_ssa,
                    dim=N,
                    inputs=[
                        self._self_collision_grid.id,
                        self.p["pred_pos"], self._current_alpha(), self.p["fixed"],
                        self.self_collision_dist, self.self_collision_stiffness,
                        sc_delta, sc_count,
                    ],
                    device=self.device,
                )
                new_pred_pos = wp.empty_like(self.p["pred_pos"])
                wp.launch(
                    apply_self_collision_ssa_kernel,
                    dim=N,
                    inputs=[self.p["pred_pos"], sc_delta, sc_count],
                    outputs=[new_pred_pos],
                    device=self.device,
                )
                self.p["pred_pos"] = new_pred_pos

        # 4. Cutting pipeline (every N steps when knife active)
        # enable_cutting (2026-06-13) decouples REMOVAL from the knife COLLIDER: with
        # enable_cutting=False the knife still indents (collision_kernel_ssa, knife_active=1)
        # but no edge/particle damage or α-removal runs → topology intact. This is the
        # non-destructive PRESS mode for stiffness (k) calibration — same solver, just skips
        # the cutting block (no new simulator needed).
        if (self.knife_active == 1 and self.enable_cutting and self.e is not None and
                self.step_count % self.cutting_update_interval == 0):
            num_edges = self.e["num_edges"]

            if self.use_ssa_soft_removal:
                # ===== Path B+ Stage B: cycle-scoped buffer SSA path =====
                # Each cutting event advances ssa_cycle_index by 1.
                # All state buffers (alpha, edge_damage, particle_damage) are
                # indexed lists; this cycle reads [c], writes [c+1].
                self._alloc_ssa_buffers()
                c = self._ssa_cycle_index
                if c + 1 >= self.ssa_buffer_capacity:
                    raise RuntimeError(
                        f"SSA buffer capacity {self.ssa_buffer_capacity} exhausted "
                        f"at cycle {c+1}. Increase solver.ssa_buffer_capacity or "
                        f"call solver.reset_ssa_buffers() to start fresh."
                    )
                alpha_prev = self._ssa_alpha_buffers[c]
                alpha_next = self._ssa_alpha_buffers[c + 1]
                ed_prev = self._ssa_edge_damage_buffers[c]
                ed_next = self._ssa_edge_damage_buffers[c + 1]
                pd_prev = self._ssa_particle_damage_buffers[c]
                pd_next = self._ssa_particle_damage_buffers[c + 1]

                # Edge damage: read from ed_prev, write delta, apply prev+delta→next
                # Day 6b (2026-05-22): dispatch to *_param_arr variant if theta_wp set
                self.e["damage_delta"].zero_()
                if self.theta_wp is not None:
                    wp.launch(
                        update_edge_damage_kernel_ssa_param_arr,
                        dim=num_edges,
                        inputs=[
                            self.p["pred_pos"], alpha_prev,
                            self.e["a"], self.e["b"],
                            self.e["pf_rest"], ed_prev,
                            self.knife_pos,
                            self.theta_wp["knife_radius"],
                            self.theta_wp["alpha_min"],
                            self.e["damage_delta"],
                        ],
                        device=self.device,
                    )
                else:
                    wp.launch(
                        update_edge_damage_kernel_ssa,
                        dim=num_edges,
                        inputs=[
                            self.p["pred_pos"], alpha_prev,
                            self.e["a"], self.e["b"],
                            self.e["pf_rest"], ed_prev,
                            self.knife_pos, self.knife_radius, self.alpha_min,
                            self.e["damage_delta"],
                        ],
                        device=self.device,
                    )
                wp.launch(
                    apply_edge_damage_ssa_kernel,
                    dim=num_edges,
                    inputs=[ed_prev, self.e["damage_delta"], ed_next],
                    device=self.device,
                )

                # Particle damage: aggregate from ed_next, finalize on pd_prev → pd_next
                damage_sum = wp.zeros(N, dtype=wp.float32, device=self.device, requires_grad=True)
                damage_count = wp.zeros(N, dtype=wp.float32, device=self.device, requires_grad=True)
                wp.launch(
                    update_particle_damage_kernel,
                    dim=num_edges,
                    inputs=[
                        damage_sum, damage_count,
                        alpha_prev,
                        self.e["a"], self.e["b"],
                        ed_next,
                    ],
                    device=self.device,
                )
                self.p["particle_damage_delta"].zero_()
                wp.launch(
                    finalize_particle_damage_kernel_ssa,
                    dim=N,
                    inputs=[damage_sum, damage_count, pd_prev,
                            self.p["particle_damage_delta"]],
                    device=self.device,
                )
                wp.launch(
                    apply_particle_damage_ssa_kernel,
                    dim=N,
                    inputs=[pd_prev, self.p["particle_damage_delta"], pd_next],
                    device=self.device,
                )

                # Soft removal: read alpha_prev (gating + carry-over), write alpha_next
                # Day 6b: param_arr dispatch when theta_wp set
                # 2026-05-25: optional CUSA aspiration (cpp_sim-parity inside-knife gate
                # + persistence) when self.use_cusa_aspiration=True.
                if self.theta_wp is not None:
                    if self.use_cusa_aspiration and "knife_radius" in self.theta_wp:
                        # Stage E (2026-05-25): dispatch based on threshold shape.
                        # shape (1,)  → global τ (original CUSA kernel)
                        # shape (N,)  → per-particle τ (new pp_tau kernel)
                        thr_shape = self.theta_wp["threshold"].shape[0]
                        if thr_shape == N:
                            # Per-particle τ path
                            wp.launch(
                                update_soft_removal_kernel_ssa_param_arr_pp_tau_cusa,
                                dim=N,
                                inputs=[
                                    pd_next, alpha_prev, self.p["fixed"],
                                    # START-OF-SUBSTEP pos (pre-collision) for the
                                    # inside-knife gate. pred_pos is post-collision
                                    # (particles pushed onto knife surface → gate
                                    # fails). pos is untouched until update_velocity.
                                    self.p["pos"],
                                    self.knife_pos,
                                    self.theta_wp["knife_radius"],
                                    self.theta_wp["threshold"],  # (N,)
                                    self.theta_wp["sharpness"],
                                    alpha_next,
                                ],
                                device=self.device,
                            )
                        else:
                            # Global τ path (original)
                            wp.launch(
                                update_soft_removal_kernel_ssa_param_arr_cusa,
                                dim=N,
                                inputs=[
                                    pd_next, alpha_prev, self.p["fixed"],
                                    # START-OF-SUBSTEP pos (pre-collision); see
                                    # per-particle τ launch above for rationale.
                                    self.p["pos"],
                                    self.knife_pos,
                                    self.theta_wp["knife_radius"],
                                    self.theta_wp["threshold"],  # (1,)
                                    self.theta_wp["sharpness"],
                                    alpha_next,
                                ],
                                device=self.device,
                            )
                    else:
                        wp.launch(
                            update_soft_removal_kernel_ssa_param_arr,
                            dim=N,
                            inputs=[
                                pd_next, alpha_prev, self.p["fixed"],
                                self.theta_wp["threshold"],
                                self.theta_wp["sharpness"],
                                alpha_next,
                            ],
                            device=self.device,
                        )
                else:
                    wp.launch(
                        update_soft_removal_kernel_ssa,
                        dim=N,
                        inputs=[
                            pd_next, alpha_prev, self.p["fixed"],
                            self.aspiration_threshold, self.soft_removal_sharpness,
                            alpha_next,
                        ],
                        device=self.device,
                    )

                # Advance cycle index. Now self._current_alpha() returns
                # alpha_buffers[c+1] (the just-written cycle) for downstream
                # kernel calls (update_velocity in this substep, predict in
                # next substep, etc.).
                # NOTE: self.p["alpha"], self.p["damage"], self.e["damage"]
                # are NOT updated. Visualization / RL observation code in
                # SSA mode should call solver.snapshot_to_p_arrays() OUTSIDE
                # the tape, or read solver._ssa_alpha_buffers[c_idx] directly.
                # (Avoiding wp.copy here — it would create R-W-R on self.p["alpha"]
                # since predict_kernel_ssa in next substep already reads it.)
                self._ssa_cycle_index = c + 1

            else:
                # ===== Legacy path: in-place writes, atomic_add applies =====
                # Forward-only safe; backward produces silent zero on cutting params
                # (Stage 2 detach-reattach failure, see PlasticineLab §3.4).
                # Use this path for RL training/eval (no gradient needed).

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

                # Pass 2: finalize mean + monotonic softplus update — SSA (Stage A 2026-04-22)
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
        # Note: in SSA mode, _current_alpha() returns the POST-cutting buffer
        # (since cutting block advanced ssa_cycle_index), so newly removed
        # particles have their velocity damped/skipped correctly.
        # Phase 2C (2026-05-07): SSA mode OR split mode → use update_velocity_kernel_ssa
        # which replaces alpha-based early return with wp.where mask + cycle-scoped
        # pos_out/vel_out buffers. Eliminates Stage A/B/C-class silent zero when
        # cutting fires within tape (0.094 → 0.94 ratio at num_iter=6).
        inv_dt = 1.0 / dt_sub
        if self.use_ssa_soft_removal or self.use_split_cluster_solve:
            new_pos = wp.empty_like(self.p["pos"])
            new_vel = wp.empty_like(self.p["vel"])
            wp.launch(
                update_velocity_kernel_ssa,
                dim=N,
                inputs=[
                    self.p["pos"], self.p["vel"],
                    self.p["pred_pos"], self.p["old_pos"],
                    self._current_alpha(), self.p["fixed"],
                    inv_dt, self.alpha_vel_scale,
                ],
                outputs=[new_pos, new_vel],
                device=self.device,
            )
            self.p["pos"] = new_pos
            self.p["vel"] = new_vel
        else:
            wp.launch(
                update_velocity_kernel,
                dim=N,
                inputs=[
                    self.p["pos"], self.p["vel"],
                    self.p["pred_pos"], self.p["old_pos"],
                    self._current_alpha(), self.p["fixed"],
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
