"""Observation and reward Warp kernels for DebulkEnv.
Follows Rewarped design: obs = downsampled particle positions + alpha + tool state.
All kernels are differentiable via wp.Tape() unless noted.
"""
import warp as wp


# ============== Action Application ==============

@wp.kernel
def apply_action_kernel(
    knife_pos: wp.array(dtype=wp.vec3),
    action: wp.array(dtype=wp.vec3),
    scale: float,
    knife_pos_out: wp.array(dtype=wp.vec3),
):
    """knife_pos_out = knife_pos + action * scale. Position delta (not velocity)."""
    tid = wp.tid()
    if tid > 0:
        return
    knife_pos_out[0] = knife_pos[0] + action[0] * scale


# ============== Observation Kernels ==============

@wp.kernel
def downsample_particles_kernel(
    pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    sample_indices: wp.array(dtype=wp.int32),
    out_pos: wp.array(dtype=wp.vec3),
    out_alpha: wp.array(dtype=wp.float32),
):
    """Downsample particle positions and alpha by index.
    sample_indices maps [0..N_sample) -> particle index in full array.
    """
    si = wp.tid()
    pid = sample_indices[si]
    out_pos[si] = pos[pid]
    out_alpha[si] = alpha[pid]


@wp.kernel
def compute_com_kernel(
    pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    out_com: wp.array(dtype=wp.vec3),
    out_count: wp.array(dtype=wp.float32),
):
    """Weighted center of mass (alpha-weighted, non-fixed particles)."""
    i = wp.tid()
    if fixed[i] == 1 or alpha[i] < 0.01:
        return
    wp.atomic_add(out_com, 0, pos[i] * alpha[i])
    wp.atomic_add(out_count, 0, alpha[i])


@wp.kernel
def normalize_com_kernel(
    out_com: wp.array(dtype=wp.vec3),
    out_count: wp.array(dtype=wp.float32),
):
    """Divide accumulated com by count."""
    tid = wp.tid()
    if tid > 0:
        return
    cnt = out_count[0]
    if cnt > 0.5:
        out_com[0] = out_com[0] * (1.0 / cnt)


# ============== Reward Kernels (Position-Based) ==============
# Gradient flows through: reward → pos → collision → knife_pos → action
# Does NOT go through damage/alpha chain (which explodes).
# P(F) still affects physics (weakened particles move more), but gradient is indirect.

@wp.kernel
def compute_tumor_displacement_kernel(
    pos: wp.array(dtype=wp.vec3),
    rest_pos: wp.array(dtype=wp.vec3),
    fixed: wp.array(dtype=wp.int32),
    nerve_mask: wp.array(dtype=wp.int32),
    out_disp_sum: wp.array(dtype=wp.float32),
    out_count: wp.array(dtype=wp.float32),
):
    """Sum displacement of tumor particles from rest position.
    Larger displacement = more material pushed away = more debulking.
    Differentiable: gradient flows through pos (collision), not damage chain.
    """
    i = wp.tid()
    if fixed[i] == 1 or nerve_mask[i] == 1:
        return
    diff = pos[i] - rest_pos[i]
    dist_sq = wp.dot(diff, diff)
    dist = wp.sqrt(dist_sq + 1.0e-8)  # safe norm
    wp.atomic_add(out_disp_sum, 0, dist)
    wp.atomic_add(out_count, 0, 1.0)


@wp.kernel
def compute_nerve_displacement_kernel(
    pos: wp.array(dtype=wp.vec3),
    rest_pos: wp.array(dtype=wp.vec3),
    nerve_mask: wp.array(dtype=wp.int32),
    out_disp_sum: wp.array(dtype=wp.float32),
    out_count: wp.array(dtype=wp.float32),
):
    """Sum displacement of nerve particles. Should stay small (nerve undisturbed)."""
    i = wp.tid()
    if nerve_mask[i] == 0:
        return
    diff = pos[i] - rest_pos[i]
    dist_sq = wp.dot(diff, diff)
    dist = wp.sqrt(dist_sq + 1.0e-8)
    wp.atomic_add(out_disp_sum, 0, dist)
    wp.atomic_add(out_count, 0, 1.0)


# ============== Capsule-task damage kernels (M3.2) ==============
# Used by:
#   - _compute_reward_capsule (forward, measuring damage fractions)
#   - L2 tactile gradient (Arm C): differentiate capsule_damage_sum w.r.t. knife_pos

@wp.kernel
def compute_capsule_damage_kernel(
    alpha: wp.array(dtype=wp.float32),
    boundary_mask: wp.array(dtype=wp.int32),
    out_cap_sum: wp.array(dtype=wp.float32),
):
    """Sum (1 - alpha) over CAPSULE particles (boundary of tumor mesh).
    Capsule damage = sum of fractional particle removal on the outer shell.
    Lower is better (capsule should be preserved).

    For L2 tactile gradient: differentiate this sum w.r.t. knife_pos via wp.Tape.
    The gradient vector points in the direction that INCREASES capsule damage,
    so policy should move AGAINST this gradient to protect the shell.
    """
    i = wp.tid()
    if boundary_mask[i] == 0:
        return
    wp.atomic_add(out_cap_sum, 0, 1.0 - alpha[i])


@wp.kernel
def compute_interior_damage_kernel(
    alpha: wp.array(dtype=wp.float32),
    interior_mask: wp.array(dtype=wp.int32),
    out_int_sum: wp.array(dtype=wp.float32),
):
    """Sum (1 - alpha) over INTERIOR particles (non-boundary tumor).
    Interior damage = main task reward signal (we WANT to remove these).
    Higher is better.
    """
    i = wp.tid()
    if interior_mask[i] == 0:
        return
    wp.atomic_add(out_int_sum, 0, 1.0 - alpha[i])


@wp.kernel
def compute_position_reward_kernel(
    tumor_disp_sum: wp.array(dtype=wp.float32),
    tumor_count: wp.array(dtype=wp.float32),
    nerve_disp_sum: wp.array(dtype=wp.float32),
    nerve_count: wp.array(dtype=wp.float32),
    lambda_nerve: float,
    reward_out: wp.array(dtype=wp.float32),
):
    """Position-based reward (PlasticineLab-style).
    reward = mean_tumor_displacement - lambda * mean_nerve_displacement
    Fully differentiable through collision → positions.
    """
    tid = wp.tid()
    if tid > 0:
        return

    tumor_disp = 0.0
    tc = tumor_count[0]
    if tc > 0.5:
        tumor_disp = tumor_disp_sum[0] / tc

    nerve_disp = 0.0
    nc = nerve_count[0]
    if nc > 0.5:
        nerve_disp = nerve_disp_sum[0] / nc

    reward_out[0] = tumor_disp - lambda_nerve * nerve_disp


# ============== Reward Accumulation (for monolithic tape) ==============

@wp.kernel
def accumulate_reward_kernel(
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
    discount: float,
):
    """dst[0] += src[0] * discount. Used to sum discounted rewards inside tape."""
    tid = wp.tid()
    if tid > 0:
        return
    wp.atomic_add(dst, 0, src[0] * discount)
