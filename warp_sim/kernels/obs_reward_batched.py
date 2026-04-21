"""Batched observation and reward kernels — per-env accumulation."""
import warp as wp


@wp.kernel
def apply_action_batched_kernel(
    knife_pos: wp.array(dtype=wp.vec3),
    actions: wp.array(dtype=wp.vec3),
    scale: float,
    knife_pos_out: wp.array(dtype=wp.vec3),
):
    """Per-env: knife_pos_out[e] = knife_pos[e] + actions[e] * scale."""
    e = wp.tid()
    knife_pos_out[e] = knife_pos[e] + actions[e] * scale


@wp.kernel
def downsample_particles_batched_kernel(
    pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    sample_indices: wp.array(dtype=wp.int32),
    out_pos: wp.array(dtype=wp.vec3),
    out_alpha: wp.array(dtype=wp.float32),
    n_per_env: int,
    n_sample: int,
):
    """Batched downsample: thread = env_id * n_sample + sample_idx."""
    tid = wp.tid()
    env_id = tid / n_sample
    si = tid - env_id * n_sample
    pid = env_id * n_per_env + sample_indices[si]
    out_pos[tid] = pos[pid]
    out_alpha[tid] = alpha[pid]


@wp.kernel
def compute_tumor_displacement_batched_kernel(
    pos: wp.array(dtype=wp.vec3),
    rest_pos: wp.array(dtype=wp.vec3),
    fixed: wp.array(dtype=wp.int32),
    nerve_mask: wp.array(dtype=wp.int32),
    out_disp_sum: wp.array(dtype=wp.float32),
    out_count: wp.array(dtype=wp.float32),
    n_per_env: int,
):
    """Per-env tumor displacement accumulation."""
    i = wp.tid()
    env_id = i / n_per_env
    local_i = i - env_id * n_per_env
    if fixed[i] == 1 or nerve_mask[local_i] == 1:
        return
    diff = pos[i] - rest_pos[i]
    dist_sq = wp.dot(diff, diff)
    dist = wp.sqrt(dist_sq + 1.0e-8)
    wp.atomic_add(out_disp_sum, env_id, dist)
    wp.atomic_add(out_count, env_id, 1.0)


@wp.kernel
def compute_nerve_displacement_batched_kernel(
    pos: wp.array(dtype=wp.vec3),
    rest_pos: wp.array(dtype=wp.vec3),
    nerve_mask: wp.array(dtype=wp.int32),
    out_disp_sum: wp.array(dtype=wp.float32),
    out_count: wp.array(dtype=wp.float32),
    n_per_env: int,
):
    """Per-env nerve displacement accumulation."""
    i = wp.tid()
    env_id = i / n_per_env
    local_i = i - env_id * n_per_env
    if nerve_mask[local_i] == 0:
        return
    diff = pos[i] - rest_pos[i]
    dist_sq = wp.dot(diff, diff)
    dist = wp.sqrt(dist_sq + 1.0e-8)
    wp.atomic_add(out_disp_sum, env_id, dist)
    wp.atomic_add(out_count, env_id, 1.0)


@wp.kernel
def compute_position_reward_batched_kernel(
    tumor_disp_sum: wp.array(dtype=wp.float32),
    tumor_count: wp.array(dtype=wp.float32),
    nerve_disp_sum: wp.array(dtype=wp.float32),
    nerve_count: wp.array(dtype=wp.float32),
    lambda_nerve: float,
    reward_out: wp.array(dtype=wp.float32),
):
    """Per-env position-based reward."""
    e = wp.tid()

    tumor_disp = 0.0
    tc = tumor_count[e]
    if tc > 0.5:
        tumor_disp = tumor_disp_sum[e] / tc

    nerve_disp = 0.0
    nc = nerve_count[e]
    if nc > 0.5:
        nerve_disp = nerve_disp_sum[e] / nc

    reward_out[e] = tumor_disp - lambda_nerve * nerve_disp
