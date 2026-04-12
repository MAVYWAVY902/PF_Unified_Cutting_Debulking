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


# ============== Reward Kernels ==============

@wp.kernel
def compute_tumor_removal_kernel(
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    nerve_mask: wp.array(dtype=wp.int32),
    out_tumor_alpha_sum: wp.array(dtype=wp.float32),
    out_tumor_count: wp.array(dtype=wp.float32),
):
    """Sum alpha for tumor particles (non-fixed, non-nerve). Lower alpha = more removed."""
    i = wp.tid()
    if fixed[i] == 1 or nerve_mask[i] == 1:
        return
    wp.atomic_add(out_tumor_alpha_sum, 0, alpha[i])
    wp.atomic_add(out_tumor_count, 0, 1.0)


@wp.kernel
def compute_nerve_alpha_kernel(
    alpha: wp.array(dtype=wp.float32),
    nerve_mask: wp.array(dtype=wp.int32),
    out_nerve_alpha_sum: wp.array(dtype=wp.float32),
    out_nerve_count: wp.array(dtype=wp.float32),
):
    """Sum alpha for nerve particles. Lower alpha = more nerve damage (bad)."""
    i = wp.tid()
    if nerve_mask[i] == 0:
        return
    wp.atomic_add(out_nerve_alpha_sum, 0, alpha[i])
    wp.atomic_add(out_nerve_count, 0, 1.0)


@wp.kernel
def compute_reward_rewarped_kernel(
    tumor_alpha_sum: wp.array(dtype=wp.float32),
    tumor_count: wp.array(dtype=wp.float32),
    nerve_alpha_sum: wp.array(dtype=wp.float32),
    nerve_count: wp.array(dtype=wp.float32),
    lambda_nerve: float,
    reward_out: wp.array(dtype=wp.float32),
):
    """Reward = tumor_removed_frac - lambda * nerve_damage_frac.

    tumor_removed_frac = 1 - mean(alpha[tumor])   (higher = more removed = good)
    nerve_damage_frac  = 1 - mean(alpha[nerve])   (higher = more damage = bad)

    Following RollingFlat pattern: simple geometric objective on particle state.
    """
    tid = wp.tid()
    if tid > 0:
        return

    tc = tumor_count[0]
    nc = nerve_count[0]

    tumor_removed = 0.0
    if tc > 0.5:
        tumor_removed = 1.0 - tumor_alpha_sum[0] / tc

    nerve_damage = 0.0
    if nc > 0.5:
        nerve_damage = 1.0 - nerve_alpha_sum[0] / nc

    # Squash tumor reward like RollingFlat: 1/(1+d)^2 where d = mean_alpha
    # Higher removal (lower mean_alpha) -> higher reward
    tumor_rew = 0.0
    if tc > 0.5:
        d = tumor_alpha_sum[0] / tc  # mean alpha [0,1], lower = more removed
        tumor_rew = 1.0 / (1.0 + d)
        tumor_rew = tumor_rew * tumor_rew  # squared

    reward_out[0] = tumor_rew - lambda_nerve * nerve_damage
