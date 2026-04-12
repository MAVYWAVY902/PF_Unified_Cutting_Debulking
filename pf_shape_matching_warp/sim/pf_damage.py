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

    # Only update edges near knife (3x radius)
    if dist_to_center > knife_radius * 3.0:
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
        increment = wp.log(1.0 + wp.exp(wp.clamp(20.0 * dmg_diff, -50.0, 50.0))) / 20.0
        edge_damage[eid] = current + increment


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
    """Compute mean damage per particle. Monotonic: only increase.
    damage[i] = max(damage[i], sum[i] / count[i])
    The max here is a per-particle scalar op (not atomic), fine for differentiability.
    """
    i = wp.tid()
    cnt = particle_damage_count[i]
    if cnt < 0.5:
        return
    mean_dmg = particle_damage_sum[i] / cnt
    # Differentiable monotonic update: softplus approximation of max(0, new - old)
    # softplus(x) = log(1 + exp(k*x)) / k, but clamp arg to prevent overflow
    diff = mean_dmg - particle_damage[i]
    sp_arg = wp.clamp(20.0 * diff, -50.0, 50.0)
    particle_damage[i] = particle_damage[i] + wp.log(1.0 + wp.exp(sp_arg)) / 20.0


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
