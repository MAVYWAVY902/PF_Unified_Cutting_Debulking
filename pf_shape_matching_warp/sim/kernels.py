"""Core Shape Matching PBD kernels in Warp.
Mirrors CUDA kernels from shape_matching_gpu.cu.
"""
import warp as wp


# ============== Predict ==============

@wp.kernel
def predict_kernel(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    pred_pos: wp.array(dtype=wp.vec3),
    old_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),  # soft removal mask
    fixed: wp.array(dtype=wp.int32),
    gravity: wp.vec3,
    dt: float,
    damping: float,
):
    i = wp.tid()
    if alpha[i] < 0.01 or fixed[i] == 1:
        old_pos[i] = pos[i]
        pred_pos[i] = pos[i]
        return

    old_pos[i] = pos[i]
    v = vel[i] + gravity * dt
    v = v * damping
    vel[i] = v
    pred_pos[i] = pos[i] + v * dt


# ============== Collision ==============

@wp.kernel
def collision_kernel(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    floor_y: float,
    knife_pos: wp.vec3,
    knife_radius: float,
    knife_active: int,
):
    i = wp.tid()
    if alpha[i] < 0.01 or fixed[i] == 1:
        return

    p = pred_pos[i]

    # Floor
    if p[1] < floor_y:
        p = wp.vec3(p[0], floor_y, p[2])

    # Knife sphere SDF
    if knife_active == 1:
        diff = p - knife_pos
        dist = wp.length(diff)
        if dist < knife_radius and dist > 1.0e-10:
            p = knife_pos + diff * (knife_radius / dist)

    pred_pos[i] = p


# ============== Clear Deltas ==============

@wp.kernel
def clear_deltas_kernel(
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    deltas[i] = wp.vec3(0.0, 0.0, 0.0)
    delta_counts[i] = 0


# ============== Cluster Solve (Core: SVD shape matching) ==============

@wp.kernel
def cluster_solve_kernel(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
):
    """Per-cluster kernel: sequential loop over cluster particles.
    Warp has no warp shuffle, so we loop sequentially (OK for <30k clusters).
    """
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]

    if size < 2:
        return

    # 1. Weighted centroid (skip removed particles via alpha)
    cx = float(0.0)
    cy = float(0.0)
    cz = float(0.0)
    tw = float(0.0)

    for j in range(size):
        pid = particle_ids[start + j]
        w = alpha[pid]
        if w < 0.01:
            continue
        p = pred_pos[pid]
        cx = cx + w * p[0]
        cy = cy + w * p[1]
        cz = cz + w * p[2]
        tw = tw + w

    if tw < 1.5:
        return

    cx = cx / tw
    cy = cy / tw
    cz = cz / tw

    # 2a. Rest offset centroid of alive particles (recentering)
    qcx = float(0.0)
    qcy = float(0.0)
    qcz = float(0.0)

    for j in range(size):
        pid = particle_ids[start + j]
        w = alpha[pid]
        if w < 0.01:
            continue
        q = rest_offsets[start + j]
        qcx = qcx + w * q[0]
        qcy = qcy + w * q[1]
        qcz = qcz + w * q[2]

    qcx = qcx / tw
    qcy = qcy / tw
    qcz = qcz / tw

    # 2b. Covariance matrix A_pq (3x3)
    a00 = float(0.0)
    a01 = float(0.0)
    a02 = float(0.0)
    a10 = float(0.0)
    a11 = float(0.0)
    a12 = float(0.0)
    a20 = float(0.0)
    a21 = float(0.0)
    a22 = float(0.0)

    for j in range(size):
        pid = particle_ids[start + j]
        w = alpha[pid]
        if w < 0.01:
            continue
        p = pred_pos[pid]
        px = p[0] - cx
        py = p[1] - cy
        pz = p[2] - cz
        q = rest_offsets[start + j]
        qx = q[0] - qcx
        qy = q[1] - qcy
        qz = q[2] - qcz

        a00 = a00 + px * qx
        a01 = a01 + px * qy
        a02 = a02 + px * qz
        a10 = a10 + py * qx
        a11 = a11 + py * qy
        a12 = a12 + py * qz
        a20 = a20 + pz * qx
        a21 = a21 + pz * qy
        a22 = a22 + pz * qz

    # 3. SVD -> Rotation R = U * V^T
    A = wp.mat33(a00, a01, a02,
                 a10, a11, a12,
                 a20, a21, a22)

    U, sigma, V = wp.svd3(A)
    R = U * wp.transpose(V)

    # Ensure det(R) > 0 (proper rotation)
    det = wp.determinant(R)
    if det < 0.0:
        # Flip last column of U
        U_fixed = wp.mat33(
            U[0, 0], U[0, 1], -U[0, 2],
            U[1, 0], U[1, 1], -U[1, 2],
            U[2, 0], U[2, 1], -U[2, 2],
        )
        R = U_fixed * wp.transpose(V)

    # 4. Goal positions -> atomicAdd deltas
    for j in range(size):
        pid = particle_ids[start + j]
        w = alpha[pid]
        if w < 0.01:
            continue

        q = rest_offsets[start + j]
        q_centered = wp.vec3(q[0] - qcx, q[1] - qcy, q[2] - qcz)
        goal = R * q_centered + wp.vec3(cx, cy, cz)
        correction = goal - pred_pos[pid]

        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, 1)


# ============== Apply ==============

@wp.kernel
def apply_kernel(
    pred_pos: wp.array(dtype=wp.vec3),
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    stiffness: float,
):
    i = wp.tid()
    if alpha[i] < 0.01 or fixed[i] == 1 or delta_counts[i] == 0:
        return

    dmg = damage[i]
    local_stiff = stiffness * (1.0 - dmg * 0.95)
    inv_count = 1.0 / wp.float32(delta_counts[i])
    pred_pos[i] = pred_pos[i] + local_stiff * inv_count * deltas[i]


# ============== Update Velocity ==============

@wp.kernel
def update_velocity_kernel(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    pred_pos: wp.array(dtype=wp.vec3),
    old_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    inv_dt: float,
):
    i = wp.tid()
    if alpha[i] < 0.01 or fixed[i] == 1:
        return

    vel[i] = (pred_pos[i] - old_pos[i]) * inv_dt
    pos[i] = pred_pos[i]
