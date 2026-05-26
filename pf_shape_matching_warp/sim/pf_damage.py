"""P(F) edge damage + soft removal kernels in Warp.
Mirrors cutting kernels from shape_matching_gpu.cu.
"""
import warp as wp


# Scale factor for int-encoded float atomic max (damage range [0, 1])
DAMAGE_INT_SCALE = float(1000000)


# ============== Edge Damage (P(F) computation) ==============

@wp.kernel
def update_edge_damage_kernel(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    edge_a: wp.array(dtype=wp.int32),
    edge_b: wp.array(dtype=wp.int32),
    pf_rest: wp.array(dtype=wp.float32),
    edge_damage: wp.array(dtype=wp.float32),
    knife_pos: wp.array(dtype=wp.vec3),
    knife_radius: float,
    alpha_min: float,
):
    """Per-edge: compute P(F) using knife surface distance, update damage (monotonic)."""
    eid = wp.tid()

    a = edge_a[eid]
    b = edge_b[eid]

    # Skip if either endpoint is removed
    if alpha[a] < 0.01 or alpha[b] < 0.01:
        return

    # Edge midpoint
    pa = pred_pos[a]
    pb = pred_pos[b]
    mid = (pa + pb) * 0.5

    # Safe distance: sqrt(|x|^2 + eps^2) — always > 0, bounded gradient
    kp = knife_pos[0]
    diff = mid - kp
    dist_sq = wp.dot(diff, diff)
    dist_to_center = wp.sqrt(dist_sq + 1.0e-8)

    # Only update edges near knife (1.3x radius — localized cutting for CUSA realism)
    if dist_to_center > knife_radius * 1.3:
        return

    # Edge half-length (circumsphere radius), also safe
    edge_vec = pa - pb
    edge_len_sq = wp.dot(edge_vec, edge_vec)
    radius = wp.sqrt(edge_len_sq + 1.0e-8) * 0.5
    if radius < 1.0e-4:
        return

    # P(F) using knife SURFACE distance (not center)
    intruder_dist = dist_to_center - knife_radius
    sdist = intruder_dist - radius
    # Clamp sigmoid argument to prevent float32 overflow
    exp_arg = wp.clamp(-alpha_min * sdist / radius, -50.0, 50.0)
    pf = 1.0 / (1.0 + wp.exp(exp_arg))

    # Damage update (monotonic: only increase)
    # Use smooth max instead of conditional write — Warp tape cannot differentiate
    # through conditional array writes (if target > current: array[i] = target).
    # smooth_max(a, b) = a + softplus(b - a) where softplus(x) = log(1+exp(x))
    rest = pf_rest[eid]
    if rest > 0.01:
        target = 1.0 - pf / rest
        # Smooth clamp to [0, inf): softplus(target) ≈ max(0, target)
        target_clamped = wp.log(1.0 + wp.exp(wp.clamp(20.0 * target, -50.0, 50.0))) / 20.0
        # Smooth monotonic update: softplus(new - old) ≈ max(0, new - old)
        current = edge_damage[eid]
        dmg_diff = target_clamped - current
        # Softplus smooth max: k=10 balances differentiability vs gradient stability
        # Max gradient per step = k/4 = 2.5. Chain of 32: 2.5^32 ≈ 5e12 (within float32)
        # Drift per step: softplus(0)/10 = 0.069. Over 32 steps: 2.2 (acceptable)
        increment = wp.log(1.0 + wp.exp(wp.clamp(10.0 * dmg_diff, -50.0, 50.0))) / 10.0
        edge_damage[eid] = current + increment


# ============== Edge Damage (SSA + delta variant for gradient) ==============
# Same physics as update_edge_damage_kernel, but Warp 1.12 gradient-safe:
#   - No conditional array write (`if rest > 0.01: edge_damage[eid] = ...` → wp.where gating)
#   - No intra-kernel read-then-write on edge_damage (write to edge_damage_delta instead)
# Apply separately via apply_edge_damage_delta_kernel.

@wp.kernel
def update_edge_damage_kernel_ssa(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    edge_a: wp.array(dtype=wp.int32),
    edge_b: wp.array(dtype=wp.int32),
    pf_rest: wp.array(dtype=wp.float32),
    edge_damage: wp.array(dtype=wp.float32),         # INPUT: current edge_damage (read for monotonic update)
    knife_pos: wp.array(dtype=wp.vec3),
    knife_radius: float,
    alpha_min: float,
    edge_damage_delta: wp.array(dtype=wp.float32),   # OUTPUT: increment to apply
):
    """Gradient-safe P(F) edge damage: computes monotonic increment into delta buffer.
    Caller must zero edge_damage_delta before launch, then apply via
    apply_edge_damage_delta_kernel."""
    eid = wp.tid()

    a = edge_a[eid]
    b = edge_b[eid]
    # Early returns OK (not a re-assignment of previously-assigned var)
    if alpha[a] < 0.01 or alpha[b] < 0.01:
        return

    pa = pred_pos[a]
    pb = pred_pos[b]
    mid = (pa + pb) * 0.5

    kp = knife_pos[0]
    diff = mid - kp
    dist_sq = wp.dot(diff, diff)
    dist_to_center = wp.sqrt(dist_sq + 1.0e-8)

    if dist_to_center > knife_radius * 1.3:
        return

    edge_vec = pa - pb
    edge_len_sq = wp.dot(edge_vec, edge_vec)
    radius = wp.sqrt(edge_len_sq + 1.0e-8) * 0.5
    if radius < 1.0e-4:
        return

    intruder_dist = dist_to_center - knife_radius
    sdist = intruder_dist - radius
    exp_arg = wp.clamp(-alpha_min * sdist / radius, -50.0, 50.0)
    pf = 1.0 / (1.0 + wp.exp(exp_arg))

    # SSA: compute delta in a single forward chain; no array re-writes.
    rest = pf_rest[eid]
    # rest_safe avoids div-by-zero in the gated-off branch (value doesn't matter there)
    rest_safe = wp.where(rest > 0.01, rest, 1.0)
    target = 1.0 - pf / rest_safe
    target_clamped = wp.log(1.0 + wp.exp(wp.clamp(20.0 * target, -50.0, 50.0))) / 20.0
    current = edge_damage[eid]
    dmg_diff = target_clamped - current
    increment_raw = wp.log(1.0 + wp.exp(wp.clamp(10.0 * dmg_diff, -50.0, 50.0))) / 10.0
    # Gate by rest using wp.where (not `if rest: edge_damage[eid] = ...` which invalidates grad)
    increment = wp.where(rest > 0.01, increment_raw, 0.0)

    # Write to separate delta buffer; never re-assign edge_damage here.
    wp.atomic_add(edge_damage_delta, eid, increment)


@wp.kernel
def apply_edge_damage_delta_kernel(
    edge_damage: wp.array(dtype=wp.float32),
    edge_damage_delta: wp.array(dtype=wp.float32),
):
    """edge_damage += edge_damage_delta via wp.atomic_add (tape-safe)."""
    eid = wp.tid()
    wp.atomic_add(edge_damage, eid, edge_damage_delta[eid])


# ============== Particle Damage Aggregation ==============
# Differentiable version: atomic_add (mean) instead of atomic_max.
# Mean is differentiable via wp.Tape(); max (int-scaled) is not.
# Slightly more conservative than max (mean < max), but enables
# end-to-end gradient flow: reward → alpha → damage → edge_damage → knife_pos → action.

@wp.kernel
def update_particle_damage_kernel(
    particle_damage_sum: wp.array(dtype=wp.float32),
    particle_damage_count: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    edge_a: wp.array(dtype=wp.int32),
    edge_b: wp.array(dtype=wp.int32),
    edge_damage: wp.array(dtype=wp.float32),
):
    """Per-edge: accumulate edge damage to endpoint particles via atomic_add.
    Differentiable — gradients flow through wp.atomic_add.
    """
    eid = wp.tid()

    a = edge_a[eid]
    b = edge_b[eid]
    if alpha[a] < 0.01 or alpha[b] < 0.01:
        return

    val = edge_damage[eid]
    if val <= 0.0:
        return

    wp.atomic_add(particle_damage_sum, a, val)
    wp.atomic_add(particle_damage_count, a, 1.0)
    wp.atomic_add(particle_damage_sum, b, val)
    wp.atomic_add(particle_damage_count, b, 1.0)


@wp.kernel
def finalize_particle_damage_kernel(
    particle_damage_sum: wp.array(dtype=wp.float32),
    particle_damage_count: wp.array(dtype=wp.float32),
    particle_damage: wp.array(dtype=wp.float32),
):
    """LEGACY — direct R-M-W on particle_damage. Tape-unsafe under Warp 1.12
    verify_autograd_array_access. Kept only for forward-only paths. New solver
    uses finalize_particle_damage_kernel_ssa + apply_particle_damage_delta_kernel.
    """
    i = wp.tid()
    cnt = particle_damage_count[i]
    if cnt < 0.5:
        return
    mean_dmg = particle_damage_sum[i] / cnt
    diff = mean_dmg - particle_damage[i]
    sp_arg = wp.clamp(20.0 * diff, -50.0, 50.0)
    particle_damage[i] = particle_damage[i] + wp.log(1.0 + wp.exp(sp_arg)) / 20.0


# ============== Particle Damage Finalize (SSA + delta variant) ==============
# Same monotonic softplus update, but Warp 1.12 tape-safe:
#   - Writes increment to particle_damage_delta buffer (NOT particle_damage R-M-W)
#   - Caller zeroes particle_damage_delta before launch, applies via
#     apply_particle_damage_delta_kernel
# softplus(diff) ≈ max(0, diff) — ensures damage monotonically increases.

@wp.kernel
def finalize_particle_damage_kernel_ssa(
    particle_damage_sum: wp.array(dtype=wp.float32),
    particle_damage_count: wp.array(dtype=wp.float32),
    particle_damage: wp.array(dtype=wp.float32),          # INPUT: current damage (read for diff)
    particle_damage_delta: wp.array(dtype=wp.float32),    # OUTPUT: increment to apply
):
    i = wp.tid()
    cnt = particle_damage_count[i]
    if cnt < 0.5:
        return

    # Single-assignment chain — no local re-assign, no array R-M-W
    mean_dmg = particle_damage_sum[i] / cnt
    diff = mean_dmg - particle_damage[i]
    sp_arg = wp.clamp(20.0 * diff, -50.0, 50.0)
    increment = wp.log(1.0 + wp.exp(sp_arg)) / 20.0

    # Write to separate delta buffer via atomic_add (tape-safe).
    wp.atomic_add(particle_damage_delta, i, increment)


@wp.kernel
def apply_particle_damage_delta_kernel(
    particle_damage: wp.array(dtype=wp.float32),
    particle_damage_delta: wp.array(dtype=wp.float32),
):
    """Apply damage increment via wp.atomic_add (tape-safe). Mirrors
    apply_edge_damage_delta_kernel — same pattern, different target array."""
    i = wp.tid()
    wp.atomic_add(particle_damage, i, particle_damage_delta[i])


# ============== Soft Removal (Differentiable) ==============

@wp.kernel
def update_soft_removal_kernel(
    damage: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    threshold: float,
    sharpness: float,
):
    """Update soft removal mask alpha from damage.
    alpha = sigmoid(-sharpness * (damage - threshold))
    alpha ~ 1.0 when healthy, alpha ~ 0.0 when damaged beyond threshold.
    This is the KEY differentiability enabler.
    """
    i = wp.tid()
    if fixed[i] == 1:
        return

    d = damage[i]
    # Sigmoid soft mask (clamp exp arg to prevent overflow)
    exp_arg = wp.clamp(sharpness * (d - threshold), -50.0, 50.0)
    alpha[i] = 1.0 / (1.0 + wp.exp(exp_arg))


# ============== Soft Removal (SSA cycle-scoped variant) ==============
# Path B+ Stage B (2026-05-01): cycle-scoped buffer pattern fixes the
# cross-kernel read-write-read on alpha that produced silent-zero gradients
# in autograd backward (PlasticineLab "detach-reattach" failure, §3.4).
#
# Verified by tools/poc_soft_removal_ssa_v3_full.py:
#   ∂loss/∂sharpness  rel_err 0.95%  vs FD
#   ∂loss/∂alpha_min  rel_err 0.11%  vs FD
#   no verify_autograd_array_access warnings
#
# Pattern: read alpha_prev (gating + carry-over), write alpha_next (different
# buffer). Caller responsible for buffer-list management (alpha_buffers[c],
# alpha_buffers[c+1]). Mirrors NVIDIA Warp example_fluid_checkpoint.py
# (line 197-303 Jacobi pressure solver).

@wp.kernel
def update_soft_removal_kernel_ssa(
    damage: wp.array(dtype=wp.float32),
    alpha_prev: wp.array(dtype=wp.float32),       # READ ONLY (gating + carry-over)
    fixed: wp.array(dtype=wp.int32),
    threshold: float,
    sharpness: float,
    alpha_next: wp.array(dtype=wp.float32),       # WRITE ONLY (separate buffer)
):
    """SSA cycle-scoped variant of update_soft_removal_kernel.

    Reads alpha_prev (used for fixed-particle carry-over and as upstream input
    that other kernels in this cycle may have read). Writes alpha_next (the
    cycle's output, which becomes the next cycle's alpha_prev).
    """
    i = wp.tid()
    if fixed[i] == 1:
        alpha_next[i] = alpha_prev[i]   # carry over fixed particles unchanged
        return

    d = damage[i]
    exp_arg = wp.clamp(sharpness * (d - threshold), -50.0, 50.0)
    alpha_next[i] = 1.0 / (1.0 + wp.exp(exp_arg))


# ============== SSA "apply" kernels for cycle-scoped state ==============
# These replace the in-place atomic_add pattern of apply_edge_damage_delta_kernel
# and apply_particle_damage_delta_kernel with explicit prev + delta -> next
# semantics. Used only when caller is in cycle-scoped buffer mode.
#
# The legacy apply_*_delta_kernel are kept for forward-only paths (RL eval,
# non-gradient pipelines) where in-place atomic_add is fine.

@wp.kernel
def apply_edge_damage_ssa_kernel(
    edge_damage_prev: wp.array(dtype=wp.float32),       # READ
    edge_damage_delta: wp.array(dtype=wp.float32),      # READ
    edge_damage_next: wp.array(dtype=wp.float32),       # WRITE (different buffer)
):
    """SSA: edge_damage_next[i] = edge_damage_prev[i] + edge_damage_delta[i].
    Replaces apply_edge_damage_delta_kernel's in-place atomic_add for the
    cycle-scoped buffer path."""
    eid = wp.tid()
    edge_damage_next[eid] = edge_damage_prev[eid] + edge_damage_delta[eid]


@wp.kernel
def apply_particle_damage_ssa_kernel(
    particle_damage_prev: wp.array(dtype=wp.float32),   # READ
    particle_damage_delta: wp.array(dtype=wp.float32),  # READ
    particle_damage_next: wp.array(dtype=wp.float32),   # WRITE (different buffer)
):
    """SSA: particle_damage_next[i] = particle_damage_prev[i] + particle_damage_delta[i].
    Replaces apply_particle_damage_delta_kernel's in-place atomic_add for the
    cycle-scoped buffer path."""
    i = wp.tid()
    particle_damage_next[i] = particle_damage_prev[i] + particle_damage_delta[i]


# ============== Calibration variants (wp.array params for autograd) ==============
# These kernel variants are bit-identical to their float-param counterparts but
# take material/cutting parameters as wp.array(requires_grad=True) so that
# Warp tape autograd can track ∂loss/∂(alpha_min, sharpness, ...).
#
# Use case: Stage 2.2 calibration (calibrate_stage2_autograd.py). RL eval and
# normal forward sim should use the float-param variants for slightly better
# performance (no array indexing per kernel launch).

@wp.kernel
def update_edge_damage_kernel_ssa_param_arr(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    edge_a: wp.array(dtype=wp.int32),
    edge_b: wp.array(dtype=wp.int32),
    pf_rest: wp.array(dtype=wp.float32),
    edge_damage: wp.array(dtype=wp.float32),
    knife_pos: wp.array(dtype=wp.vec3),
    knife_radius_arr: wp.array(dtype=wp.float32),
    alpha_min_arr: wp.array(dtype=wp.float32),
    edge_damage_delta: wp.array(dtype=wp.float32),
):
    """Same SSA logic as update_edge_damage_kernel_ssa, but knife_radius
    and alpha_min are wp.array(requires_grad=True) for autograd tracking.
    """
    eid = wp.tid()
    a = edge_a[eid]
    b = edge_b[eid]
    if alpha[a] < 0.01 or alpha[b] < 0.01:
        return

    pa = pred_pos[a]
    pb = pred_pos[b]
    mid = (pa + pb) * 0.5

    knife_radius = knife_radius_arr[0]
    alpha_min = alpha_min_arr[0]

    kp = knife_pos[0]
    diff = mid - kp
    dist_sq = wp.dot(diff, diff)
    dist_to_center = wp.sqrt(dist_sq + 1.0e-8)

    if dist_to_center > knife_radius * 1.3:
        return

    edge_vec = pa - pb
    edge_len_sq = wp.dot(edge_vec, edge_vec)
    radius = wp.sqrt(edge_len_sq + 1.0e-8) * 0.5
    if radius < 1.0e-4:
        return

    intruder_dist = dist_to_center - knife_radius
    sdist = intruder_dist - radius
    exp_arg = wp.clamp(-alpha_min * sdist / radius, -50.0, 50.0)
    pf = 1.0 / (1.0 + wp.exp(exp_arg))

    rest = pf_rest[eid]
    rest_safe = wp.where(rest > 0.01, rest, 1.0)
    target = 1.0 - pf / rest_safe
    target_clamped = wp.log(1.0 + wp.exp(wp.clamp(20.0 * target, -50.0, 50.0))) / 20.0
    current = edge_damage[eid]
    dmg_diff = target_clamped - current
    increment_raw = wp.log(1.0 + wp.exp(wp.clamp(10.0 * dmg_diff, -50.0, 50.0))) / 10.0
    increment = wp.where(rest > 0.01, increment_raw, 0.0)

    wp.atomic_add(edge_damage_delta, eid, increment)


@wp.kernel
def update_soft_removal_kernel_ssa_param_arr(
    damage: wp.array(dtype=wp.float32),
    alpha_prev: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    threshold_arr: wp.array(dtype=wp.float32),
    sharpness_arr: wp.array(dtype=wp.float32),
    alpha_next: wp.array(dtype=wp.float32),
):
    """Same SSA logic as update_soft_removal_kernel_ssa, but threshold
    and sharpness are wp.array(requires_grad=True) for autograd tracking.
    """
    i = wp.tid()
    if fixed[i] == 1:
        alpha_next[i] = alpha_prev[i]
        return
    threshold = threshold_arr[0]
    sharpness = sharpness_arr[0]
    d = damage[i]
    exp_arg = wp.clamp(sharpness * (d - threshold), -50.0, 50.0)
    alpha_next[i] = 1.0 / (1.0 + wp.exp(exp_arg))


# CUSA (cpp_sim parity, 2026-05-25):
# Replicates cpp_sim/core/cutting_manager.h:process_fragments() —
# binary-style removal with persistence. The inside-knife geometric gate
# approach was REJECTED (2026-05-25 smoke test): collision_kernel pushes
# particles to knife surface BEFORE alpha update runs, so geometric gate
# fails post-collision. cpp_sim avoids this by running process_fragments
# on CPU separate from GPU step; harder to replicate exactly in differentiable
# Warp pipeline.
#
# Practical CUSA-style mitigation (what this kernel does):
#   1. Same damage-driven sigmoid as original kernel
#   2. Persistence: alpha can only decrease (wp.min with alpha_prev)
#      — once removed, stay removed (matches cpp_sim semantic)
#   3. Coupled with alpha-weighted vel damping in predict_kernel (separate
#      kernel) which kills zombie momentum proportional to (1 - alpha)
@wp.kernel
def update_soft_removal_kernel_ssa_param_arr_cusa(
    damage: wp.array(dtype=wp.float32),
    alpha_prev: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    knife_pos: wp.array(dtype=wp.vec3),
    knife_radius_arr: wp.array(dtype=wp.float32),
    threshold_arr: wp.array(dtype=wp.float32),
    sharpness_arr: wp.array(dtype=wp.float32),
    alpha_next: wp.array(dtype=wp.float32),
):
    """cpp_sim-parity CUSA: damage-driven sigmoid + persistence.

    Geometric inside-knife gate REMOVED (caused alpha to never reach 0 because
    collision pushes particles to knife surface BEFORE alpha update). Persistence
    via wp.min(proposed, alpha_prev) preserves cpp_sim's "once removed, stays
    removed" semantic while keeping monotonicity through forward sim.

    Args pos/knife_pos/knife_radius are kept in signature for API stability
    (caller passes them) but unused by current logic. Future inside-knife
    extension could use pre-collision pos.
    """
    i = wp.tid()
    if fixed[i] == 1:
        alpha_next[i] = alpha_prev[i]
        return
    threshold = threshold_arr[0]
    sharpness = sharpness_arr[0]

    # Damage-driven alpha (same as non-CUSA path)
    d = damage[i]
    exp_arg = wp.clamp(sharpness * (d - threshold), -50.0, 50.0)
    alpha_proposed = 1.0 / (1.0 + wp.exp(exp_arg))

    # Persistence: alpha never goes UP (matches cpp_sim "particles[pid].removed"
    # boolean flag which only flips false→true, never reverses).
    # Note: damage is already monotonic (softplus), so wp.min is usually no-op
    # but guarantees the property explicitly + handles edge cases.
    alpha_next[i] = wp.min(alpha_proposed, alpha_prev[i])


# Stage E (2026-05-25): per-particle τ variant of CUSA kernel.
# threshold_arr is now wp.array(N,) — each particle has its own fracture toughness.
# Mirrors Stage A's per-particle stiffness pattern (apply_kernel_ssa_param_arr_pp).
# sharpness still global (kept as wp.array(1,)).
@wp.kernel
def update_soft_removal_kernel_ssa_param_arr_pp_tau_cusa(
    damage: wp.array(dtype=wp.float32),
    alpha_prev: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    knife_pos: wp.array(dtype=wp.vec3),
    knife_radius_arr: wp.array(dtype=wp.float32),
    threshold_per_particle: wp.array(dtype=wp.float32),  # shape (N,) — per-particle τ
    sharpness_arr: wp.array(dtype=wp.float32),           # shape (1,) — global
    alpha_next: wp.array(dtype=wp.float32),
):
    """Per-particle τ CUSA kernel.

    α_i = σ(-s · (damage_i - τ_i))
    Each particle i has its own threshold τ_i = threshold_per_particle[i].
    Same persistence (wp.min) as global-τ CUSA kernel.

    Use cases:
       - sys-ID per-particle toughness field (heterogeneous tissue)
       - tumor (low τ, brittle) vs background (high τ, tough)
    """
    i = wp.tid()
    if fixed[i] == 1:
        alpha_next[i] = alpha_prev[i]
        return

    threshold = threshold_per_particle[i]   # per-particle τ
    sharpness = sharpness_arr[0]             # global

    d = damage[i]
    exp_arg = wp.clamp(sharpness * (d - threshold), -50.0, 50.0)
    alpha_proposed = 1.0 / (1.0 + wp.exp(exp_arg))

    # Persistence (same as global-τ kernel).
    alpha_next[i] = wp.min(alpha_proposed, alpha_prev[i])


# ============== Batched Edge Damage (per-env knife_pos) ==============

@wp.kernel
def update_edge_damage_kernel_batched(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    edge_a: wp.array(dtype=wp.int32),
    edge_b: wp.array(dtype=wp.int32),
    pf_rest: wp.array(dtype=wp.float32),
    edge_damage: wp.array(dtype=wp.float32),
    knife_pos: wp.array(dtype=wp.vec3),  # (num_envs,) per-env knife
    knife_radius: float,
    alpha_min: float,
    n_per_env: int,
    e_per_env: int,
):
    """Edge damage kernel for batched envs. Per-env knife_pos lookup."""
    eid = wp.tid()

    a = edge_a[eid]
    b = edge_b[eid]
    if alpha[a] < 0.01 or alpha[b] < 0.01:
        return

    pa = pred_pos[a]
    pb = pred_pos[b]
    mid = (pa + pb) * 0.5

    # Per-env knife position
    env_id = eid / e_per_env
    kp = knife_pos[env_id]
    diff = mid - kp
    dist_sq = wp.dot(diff, diff)
    dist_to_center = wp.sqrt(dist_sq + 1.0e-8)

    if dist_to_center > knife_radius * 1.3:
        return

    edge_vec = pa - pb
    edge_len_sq = wp.dot(edge_vec, edge_vec)
    radius = wp.sqrt(edge_len_sq + 1.0e-8) * 0.5
    if radius < 1.0e-4:
        return

    intruder_dist = dist_to_center - knife_radius
    sdist = intruder_dist - radius
    exp_arg = wp.clamp(-alpha_min * sdist / radius, -50.0, 50.0)
    pf = 1.0 / (1.0 + wp.exp(exp_arg))

    rest = pf_rest[eid]
    if rest > 0.01:
        target = 1.0 - pf / rest
        target_clamped = wp.log(1.0 + wp.exp(wp.clamp(10.0 * target, -50.0, 50.0))) / 10.0
        current = edge_damage[eid]
        dmg_diff = target_clamped - current
        increment = wp.log(1.0 + wp.exp(wp.clamp(10.0 * dmg_diff, -50.0, 50.0))) / 10.0
        edge_damage[eid] = current + increment


# ============== Batched Edge Damage (SSA + delta variant for gradient) ==============

@wp.kernel
def update_edge_damage_kernel_batched_ssa(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    edge_a: wp.array(dtype=wp.int32),
    edge_b: wp.array(dtype=wp.int32),
    pf_rest: wp.array(dtype=wp.float32),
    edge_damage: wp.array(dtype=wp.float32),
    knife_pos: wp.array(dtype=wp.vec3),           # (num_envs,) per-env knife
    knife_radius: float,
    alpha_min: float,
    n_per_env: int,
    e_per_env: int,
    edge_damage_delta: wp.array(dtype=wp.float32),  # OUTPUT
):
    """Batched gradient-safe edge damage. Same SSA pattern as non-batched variant."""
    eid = wp.tid()

    a = edge_a[eid]
    b = edge_b[eid]
    if alpha[a] < 0.01 or alpha[b] < 0.01:
        return

    pa = pred_pos[a]
    pb = pred_pos[b]
    mid = (pa + pb) * 0.5

    env_id = eid / e_per_env
    kp = knife_pos[env_id]
    diff = mid - kp
    dist_sq = wp.dot(diff, diff)
    dist_to_center = wp.sqrt(dist_sq + 1.0e-8)

    if dist_to_center > knife_radius * 1.3:
        return

    edge_vec = pa - pb
    edge_len_sq = wp.dot(edge_vec, edge_vec)
    radius = wp.sqrt(edge_len_sq + 1.0e-8) * 0.5
    if radius < 1.0e-4:
        return

    intruder_dist = dist_to_center - knife_radius
    sdist = intruder_dist - radius
    exp_arg = wp.clamp(-alpha_min * sdist / radius, -50.0, 50.0)
    pf = 1.0 / (1.0 + wp.exp(exp_arg))

    rest = pf_rest[eid]
    rest_safe = wp.where(rest > 0.01, rest, 1.0)
    target = 1.0 - pf / rest_safe
    target_clamped = wp.log(1.0 + wp.exp(wp.clamp(10.0 * target, -50.0, 50.0))) / 10.0
    current = edge_damage[eid]
    dmg_diff = target_clamped - current
    increment_raw = wp.log(1.0 + wp.exp(wp.clamp(10.0 * dmg_diff, -50.0, 50.0))) / 10.0
    increment = wp.where(rest > 0.01, increment_raw, 0.0)

    wp.atomic_add(edge_damage_delta, eid, increment)
