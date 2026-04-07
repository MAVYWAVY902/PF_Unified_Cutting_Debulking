"""P(F) edge damage + soft removal kernels in Warp.
Mirrors cutting kernels from shape_matching_gpu.cu.
"""
import warp as wp


# ============== Edge Damage (P(F) computation) ==============

@wp.kernel
def update_edge_damage_kernel(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    edge_a: wp.array(dtype=wp.int32),
    edge_b: wp.array(dtype=wp.int32),
    pf_rest: wp.array(dtype=wp.float32),
    edge_damage: wp.array(dtype=wp.float32),
    knife_pos: wp.vec3,
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

    # Distance from midpoint to knife center
    diff = mid - knife_pos
    dist_to_center = wp.length(diff)

    # Only update edges near knife (3x radius)
    if dist_to_center > knife_radius * 3.0:
        return

    # Edge half-length (circumsphere radius)
    edge_vec = pa - pb
    radius = wp.length(edge_vec) * 0.5
    if radius < 1.0e-10:
        return

    # P(F) using knife SURFACE distance (not center)
    intruder_dist = dist_to_center - knife_radius
    sdist = intruder_dist - radius
    pf = 1.0 / (1.0 + wp.exp(-alpha_min * sdist / radius))

    # Damage update (monotonic: only increase)
    rest = pf_rest[eid]
    if rest > 0.01:
        target = wp.max(0.0, 1.0 - pf / rest)
        current = edge_damage[eid]
        if target > current:
            edge_damage[eid] = target


# ============== Particle Damage Aggregation ==============

@wp.kernel
def update_particle_damage_kernel(
    particle_damage: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    edge_a: wp.array(dtype=wp.int32),
    edge_b: wp.array(dtype=wp.int32),
    edge_damage: wp.array(dtype=wp.float32),
):
    """Per-edge: propagate edge damage to both endpoint particles (monotonic max)."""
    eid = wp.tid()

    a = edge_a[eid]
    b = edge_b[eid]
    if alpha[a] < 0.01 or alpha[b] < 0.01:
        return

    val = edge_damage[eid]
    if val <= 0.0:
        return

    # Atomic max via atomic_add trick: only write if larger
    # Warp doesn't have atomic_max for float, use atomic_add approach
    # Since damage is monotonic, we just check and write (small race OK)
    if val > particle_damage[a]:
        particle_damage[a] = val
    if val > particle_damage[b]:
        particle_damage[b] = val


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
    # Sigmoid soft mask
    alpha[i] = 1.0 / (1.0 + wp.exp(sharpness * (d - threshold)))
