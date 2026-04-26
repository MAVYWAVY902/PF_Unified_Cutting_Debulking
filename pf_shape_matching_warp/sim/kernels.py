"""Core Shape Matching PBD kernels in Warp.
Mirrors CUDA kernels from shape_matching_gpu.cu.
"""
import warp as wp


# ============== Müller 2016 iterative polar decomposition (Stage C 2026-04-22) ==============
# Replaces `wp.svd3(A); R = U * V.T; if det<0: fix` pattern which (a) has unstable
# backward at clustered singular values (Block 2 of our L3 analysis), and (b) uses
# a conditional matrix re-assign (silent-zero under Warp 1.12 SSA rule).
#
# Paper: Müller et al. 2016, "A Robust Method to Extract the Rotational Part of
# Deformations" — https://matthias-research.github.io/pages/publications/stablePolarDecomp.pdf
# Reference C++: Jan Bender's PositionBasedDynamics library (same co-author),
#   github.com/InteractiveComputerGraphics/PositionBasedDynamics — MathFunctions.cpp
#
# Algorithm: given 3x3 matrix A, find rotation R (quaternion q) via fixed-point
# iteration. Each iteration computes an angular-velocity ω that rotates R toward A,
# applies the rotation, renormalizes. Converges in 4-6 iters with warm start,
# 8-10 cold. Every op (cross, dot, abs, axis-angle→quat, quat-mul, normalize)
# has a hand-written adjoint in warp/native/quat.h, so gradient flows end-to-end.
#
# NO det<0 branch needed: quaternion rotation has det=+1 by construction.
# NO SVD backward singularities: the ∂U/∂A pole at clustered σ is bypassed entirely.

# Fixed iteration count — unrolled for tape safety. Convergence empirically
# verified (C.1 debug 2026-04-22):
#   • 4 iter  — sufficient for warm-start (cluster_quat from previous frame,
#               per-step rotation delta << 1 rad). Gives ~1-5e-3 error cold.
#   • 16 iter — sufficient for cold-start from identity on any rotation up to
#               ~2 rad. Gives float32-exact (0 error) for moderate rotations,
#               sub-1e-6 for large. Newton-like quadratic convergence.
MULLER_NUM_ITER_WARM = 4
MULLER_NUM_ITER_COLD = 16


@wp.func
def extract_rotation_muller(
    A: wp.mat33,
    q_init: wp.quat,
    num_iter: int,
) -> wp.quat:
    """Iterative quaternion polar decomposition of 3x3 matrix A.

    Args:
        A: input 3x3 matrix (typically Apq covariance from shape matching)
        q_init: initial quaternion (warm-start from previous frame, else identity)
        num_iter: fixed iteration count (4 warm / 8 cold). Must be compile-time
                  constant for Warp to unroll the loop for SSA cleanliness.

    Returns:
        q: quaternion representing rotation R such that A ≈ R * S where S is
           symmetric positive semi-definite (polar decomposition).

    Physics notes:
      - Column of R = q.to_matrix() is rotated basis vector; cross product with
        A's column measures angular misalignment → accumulated over 3 basis axes
        gives net torque ω.
      - abs() in denominator is load-bearing (paper §3.2): without it, when
        cos(angle) < 0, the torque direction flips and iteration diverges.
      - Left multiplication `dq * q` is the standard convention for applying
        the correction in the world frame.

    Tape safety:
      - Loop is unrolled (Python `for _ in range(num_iter)` with fixed int arg
        compiles to straight-line code in Warp).
      - Each Python rebind `q = wp.normalize(...)` maps to a fresh SSA variable
        per iteration in Warp's codegen.
      - No conditional early break — that would re-bind across branches.
        Extra unused iterations are cheaper than gradient ambiguity.
    """
    q = q_init
    for _ in range(num_iter):
        R = wp.quat_to_matrix(q)
        # Extract columns of R (current rotated basis) and A (target)
        r0 = wp.vec3(R[0, 0], R[1, 0], R[2, 0])
        r1 = wp.vec3(R[0, 1], R[1, 1], R[2, 1])
        r2 = wp.vec3(R[0, 2], R[1, 2], R[2, 2])
        a0 = wp.vec3(A[0, 0], A[1, 0], A[2, 0])
        a1 = wp.vec3(A[0, 1], A[1, 1], A[2, 1])
        a2 = wp.vec3(A[0, 2], A[1, 2], A[2, 2])

        # Angular velocity: numerator = Σ r_i × a_i ; denom = |Σ r_i · a_i| + eps
        omega_num = wp.cross(r0, a0) + wp.cross(r1, a1) + wp.cross(r2, a2)
        omega_denom = wp.abs(wp.dot(r0, a0) + wp.dot(r1, a1) + wp.dot(r2, a2)) + 1.0e-9
        omega = omega_num / omega_denom

        # Axis-angle update, then left-compose + renormalize
        w = wp.length(omega)
        # Guard against zero-length axis (omega ≈ 0 → already converged);
        # adding tiny eps keeps axis finite; w itself serves as angle.
        axis = omega / (w + 1.0e-20)
        dq = wp.quat_from_axis_angle(axis, w)
        q = wp.normalize(wp.mul(dq, q))  # SSA rebind — Warp codegen emits fresh var
    return q


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
    """LEGACY — contains `v = v * damping` local re-assignment that violates
    Warp 1.12 SSA rule, silently zeroing backward. Kept for forward-only paths.
    New solver uses predict_kernel_ssa."""
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


# ============== Predict (SSA — Stage B 2026-04-22) ==============
# Fixes two silent-zero violations in predict_kernel:
#   (a) `v = v * damping` — local re-assignment breaks Warp 1.12 backward.
#       Fix: fresh name `v_damped` for the damped velocity.
#   (b) `old_pos[i] = pos[i]` written in both early-return and main branches.
#       Both branches wrote the same value, so hoisting it to a single
#       unconditional write is semantically identical but tape-cleaner.
# Physics unchanged (forward-equivalent to predict_kernel; verified by Axis 1).

@wp.kernel
def predict_kernel_ssa(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    pred_pos: wp.array(dtype=wp.vec3),
    old_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    gravity: wp.vec3,
    dt: float,
    damping: float,
):
    i = wp.tid()

    # Unconditional (both legacy branches wrote the same value)
    old_pos[i] = pos[i]

    # Dead / fixed particles: pred stays at pos, vel untouched (same as legacy)
    if alpha[i] < 0.01 or fixed[i] == 1:
        pred_pos[i] = pos[i]
        return

    # Single-assignment chain — each local assigned exactly once
    v_integrated = vel[i] + gravity * dt      # was `v = vel[i] + gravity * dt`
    v_damped = v_integrated * damping         # was `v = v * damping` ← fresh name
    vel[i] = v_damped
    pred_pos[i] = pos[i] + v_damped * dt


# ============== Collision ==============

@wp.kernel
def collision_kernel(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    floor_y: float,
    knife_pos: wp.array(dtype=wp.vec3),
    knife_radius: float,
    knife_active: int,
):
    i = wp.tid()
    if alpha[i] < 0.01 or fixed[i] == 1:
        return

    p = pred_pos[i]

    # Floor (Z-up: 2026-04-24 switched from Y-up, floor_y now refers to floor Z value)
    if p[2] < floor_y:
        p = wp.vec3(p[0], p[1], floor_y)

    # Knife sphere SDF with safe distance (gradient-stable at origin)
    # safe_dist = sqrt(|diff|^2 + eps^2) — always > 0, bounded gradient
    if knife_active == 1:
        kp = knife_pos[0]
        diff = p - kp
        dist_sq = wp.dot(diff, diff)
        safe_dist = wp.sqrt(dist_sq + 1.0e-8)
        if dist_sq < knife_radius * knife_radius:
            p = kp + diff * (knife_radius / safe_dist)

    pred_pos[i] = p


# ============== Collision (SSA + delta variant for gradient) ==============
# Same physics as collision_kernel above, but gradient-safe:
#   - Each local variable assigned exactly once (no re-assignment)
#   - wp.where replaces conditional re-assignment (Warp 1.12 single-assignment rule)
#   - Writes DELTA to separate output buffer via atomic_add, not directly to pred_pos
# Apply separately via apply_collision_delta_kernel so pred_pos is only touched via +=
# (tape-safe pattern per NVIDIA XPBD reference).

@wp.kernel
def collision_kernel_ssa(
    pred_pos: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),                 # 2026-04-25: start-of-substep pos for friction anchor
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    floor_y: float,
    floor_friction: float,                        # 2026-04-25: tangent slip damping coef [0..1]; 0=frictionless, 1=sticky
    knife_pos: wp.array(dtype=wp.vec3),
    knife_radius: float,
    knife_active: int,
    collision_delta: wp.array(dtype=wp.vec3),  # OUTPUT: per-particle position correction
):
    i = wp.tid()
    if alpha[i] < 0.01 or fixed[i] == 1:
        return

    # Single-assignment chain — never re-assign any local
    p0 = pred_pos[i]
    p_anchor = pos[i]                            # start-of-substep position (friction anchor)

    # Floor + tangent friction (Z-up: 2026-04-25 added Coulomb-style tangent damping).
    # When p[2] < floor_y, particle is in floor contact — clamp Z to floor and reduce
    # the tangent (XY) displacement relative to its start-of-substep position by
    # (1 - floor_friction). floor_friction = 0 reproduces frictionless behavior;
    # floor_friction = 1 = full stick (no XY slip during this substep).
    tx = p0[0] - p_anchor[0]                     # tangent disp X relative to substep start
    ty = p0[1] - p_anchor[1]                     # tangent disp Y
    fx = p_anchor[0] + tx * (1.0 - floor_friction)
    fy = p_anchor[1] + ty * (1.0 - floor_friction)
    p_floor = wp.where(
        p0[2] < floor_y,
        wp.vec3(fx, fy, floor_y),
        p0,
    )

    # Knife collision (knife_active is a compile-time arg, kernel specializes per value)
    if knife_active == 1:
        kp = knife_pos[0]
        diff = p_floor - kp
        dist_sq = wp.dot(diff, diff)
        safe_dist = wp.sqrt(dist_sq + 1.0e-8)
        p_knife = kp + diff * (knife_radius / safe_dist)
        # wp.where: select p_knife inside sphere, p_floor outside. All gradients flow.
        p_new = wp.where(
            dist_sq < knife_radius * knife_radius,
            p_knife,
            p_floor,
        )
    else:
        p_new = p_floor

    # Write delta to separate buffer — pred_pos itself is never re-assigned here.
    # atomic_add is explicitly allowlisted for tape adjoint accumulation.
    wp.atomic_add(collision_delta, i, p_new - p0)


@wp.kernel
def apply_collision_delta_kernel(
    pred_pos: wp.array(dtype=wp.vec3),
    collision_delta: wp.array(dtype=wp.vec3),
):
    """Apply collision_delta to pred_pos via wp.atomic_add — the explicitly
    tape-safe in-place operator per Warp docs. Do NOT use `pred_pos[i] = pred_pos[i] + ...`
    (Warp's verify_autograd_array_access flags it as read-after-write conflict)."""
    i = wp.tid()
    wp.atomic_add(pred_pos, i, collision_delta[i])


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
    # Add small regularization to prevent singular/near-singular matrices
    # which cause NaN in SVD backward pass
    eps = 1.0e-6
    A = wp.mat33(a00 + eps, a01,       a02,
                 a10,       a11 + eps,  a12,
                 a20,       a21,        a22 + eps)

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


# ============== Cluster Solve (SSA + Müller — Stage C 2026-04-22) ==============
# Replaces legacy cluster_solve_kernel with:
#   (a) Fresh-name accumulators (cx_sum → cx_mean) — no local re-assignment.
#   (b) extract_rotation_muller instead of wp.svd3 + det<0 fix — bypasses SVD
#       backward singularity (Block 2) AND the conditional matrix re-assign.
#   (c) Persistent cluster_quat warm-start buffer — per-cluster previous-frame
#       quaternion, accelerates convergence (4-8 iter suffice).

@wp.kernel
def cluster_solve_kernel_ssa(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    cluster_quat: wp.array(dtype=wp.quat),    # IN: warm-start q (read only)
    num_iter: int,                             # Müller iterations
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
):
    """Stage C.3 debug (2026-04-22): all `if ...: return/continue` AFTER touching
    requires_grad data were silently zeroing the backward chain. Empirical Warp
    1.12 finding: mid-kernel early exits (control flow) are tape-hostile. Fix:
    replace all with `wp.where` masking (always compute, mask 0 if inactive).

    Early-return BEFORE any tape-tracked read (e.g. `if size < 2` at the very top)
    is still OK — the thread exits before recording anything.
    """
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    # --- 1. Weighted centroid ---
    # wp.where replaces `if w < 0.01: continue` — accumulate with masked weight 0
    # for dead particles (additive identity, no effect on sum).
    cx_sum = float(0.0)
    cy_sum = float(0.0)
    cz_sum = float(0.0)
    tw = float(0.0)

    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)  # mask dead → weight 0
        p = pred_pos[pid]
        cx_sum = cx_sum + w * p[0]
        cy_sum = cy_sum + w * p[1]
        cz_sum = cz_sum + w * p[2]
        tw = tw + w

    # tw-guard via wp.where (was `if tw < 1.5: return` — silent-zero in Warp 1.12).
    # tw_safe prevents div-by-zero. Cluster-level `active_mask` was also tried
    # but empirically zeros the backward chain in Warp 1.12 when used inside the
    # goal loop (diag 2026-04-22) — per-particle alive mask is sufficient since
    # degenerate clusters where tw<1.5 typically have all particles dead already,
    # so particle_alive=0 nullifies corrections naturally.
    tw_safe = wp.where(tw < 1.5, 1.0, tw)

    cx_mean = cx_sum / tw_safe
    cy_mean = cy_sum / tw_safe
    cz_mean = cz_sum / tw_safe

    # --- 2a. Rest-offset centroid ---
    qcx_sum = float(0.0)
    qcy_sum = float(0.0)
    qcz_sum = float(0.0)

    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        q = rest_offsets[start + j]
        qcx_sum = qcx_sum + w * q[0]
        qcy_sum = qcy_sum + w * q[1]
        qcz_sum = qcz_sum + w * q[2]

    qcx_mean = qcx_sum / tw_safe
    qcy_mean = qcy_sum / tw_safe
    qcz_mean = qcz_sum / tw_safe

    # --- 2b. Covariance A_pq ---
    # IMPORTANT: legacy cluster_solve_kernel does NOT weight by alpha in the
    # covariance loop — `continue` skips dead particles, but surviving particles
    # contribute `px * qx` unweighted. To match legacy forward exactly, use a
    # BINARY alive mask (0 or 1), NOT the alpha-weighted mask used in centroid
    # accumulation (which IS alpha-weighted in legacy).
    a00 = float(0.0); a01 = float(0.0); a02 = float(0.0)
    a10 = float(0.0); a11 = float(0.0); a12 = float(0.0)
    a20 = float(0.0); a21 = float(0.0); a22 = float(0.0)

    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        alive = wp.where(w_raw < 0.01, 0.0, 1.0)  # BINARY mask, not alpha-weighted
        p = pred_pos[pid]
        px = p[0] - cx_mean
        py = p[1] - cy_mean
        pz = p[2] - cz_mean
        q = rest_offsets[start + j]
        qx = q[0] - qcx_mean
        qy = q[1] - qcy_mean
        qz = q[2] - qcz_mean

        a00 = a00 + alive * px * qx; a01 = a01 + alive * px * qy; a02 = a02 + alive * px * qz
        a10 = a10 + alive * py * qx; a11 = a11 + alive * py * qy; a12 = a12 + alive * py * qz
        a20 = a20 + alive * pz * qx; a21 = a21 + alive * pz * qy; a22 = a22 + alive * pz * qz

    A = wp.mat33(a00, a01, a02,
                 a10, a11, a12,
                 a20, a21, a22)

    # --- 3. Rotation via Müller 2016 iterative polar ---
    q_warm = cluster_quat[cid]
    q_new = extract_rotation_muller(A, q_warm, num_iter)
    R = wp.quat_to_matrix(q_new)

    # --- 4. Goal positions → atomic_add deltas ---
    # Per-particle alive mask via wp.where (no `if continue` inside loop).
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        particle_alive = wp.where(w_raw < 0.01, 0.0, 1.0)
        q = rest_offsets[start + j]
        q_centered = wp.vec3(q[0] - qcx_mean, q[1] - qcy_mean, q[2] - qcz_mean)
        goal = R * q_centered + wp.vec3(cx_mean, cy_mean, cz_mean)
        correction = (goal - pred_pos[pid]) * particle_alive

        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, int(particle_alive))


# Helper kernel to reset cluster_quat to identity (called on env.reset())
@wp.kernel
def reset_cluster_quat_kernel(cluster_quat: wp.array(dtype=wp.quat)):
    cid = wp.tid()
    cluster_quat[cid] = wp.quat(0.0, 0.0, 0.0, 1.0)


# Stage C.3 (2026-04-22): warm-start update isolated from the cluster_solve tape.
# Re-runs extract_rotation_muller on the CURRENT pred_pos/alpha to produce a fresh
# quaternion for the next substep's warm start. Launched OUTSIDE any differentiable
# tape context — its output is state for the next frame, not part of this frame's
# gradient path. If we wrote to cluster_quat inside cluster_solve_kernel_ssa (read
# + write on the same requires_grad array), Warp's backward chain was silently
# zeroed through q_new → R → deltas.
@wp.kernel
def cluster_quat_update_kernel(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    cluster_quat: wp.array(dtype=wp.quat),   # IN/OUT: the warm-start buffer
    num_iter: int,
):
    """Pure state-update: re-derive Apq covariance + Müller polar, then overwrite
    cluster_quat[cid]. Same math as the main kernel (without the goal-position /
    atomic_add legs). Called AFTER cluster_solve_kernel_ssa in solver._substep,
    outside any tape scope."""
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    # NB: this kernel runs OUTSIDE any tape (warm-start state update), so
    # early-returns/continues would be safe here for gradient purposes.
    # Kept SSA-style for consistency with cluster_solve_kernel_ssa.
    cx_sum = float(0.0); cy_sum = float(0.0); cz_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        cx_sum = cx_sum + w * p[0]
        cy_sum = cy_sum + w * p[1]
        cz_sum = cz_sum + w * p[2]
        tw = tw + w
    tw_safe = wp.where(tw < 1.5, 1.0, tw)
    cx_mean = cx_sum / tw_safe; cy_mean = cy_sum / tw_safe; cz_mean = cz_sum / tw_safe

    qcx_sum = float(0.0); qcy_sum = float(0.0); qcz_sum = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        q = rest_offsets[start + j]
        qcx_sum = qcx_sum + w * q[0]
        qcy_sum = qcy_sum + w * q[1]
        qcz_sum = qcz_sum + w * q[2]
    qcx_mean = qcx_sum / tw_safe; qcy_mean = qcy_sum / tw_safe; qcz_mean = qcz_sum / tw_safe

    # Covariance A_pq — legacy unweighted (skip-dead only). Use BINARY mask
    # to match legacy forward (NOT alpha-weighted).
    a00 = float(0.0); a01 = float(0.0); a02 = float(0.0)
    a10 = float(0.0); a11 = float(0.0); a12 = float(0.0)
    a20 = float(0.0); a21 = float(0.0); a22 = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        alive = wp.where(w_raw < 0.01, 0.0, 1.0)
        p = pred_pos[pid]
        px = p[0] - cx_mean; py = p[1] - cy_mean; pz = p[2] - cz_mean
        q = rest_offsets[start + j]
        qx = q[0] - qcx_mean; qy = q[1] - qcy_mean; qz = q[2] - qcz_mean
        a00 = a00 + alive * px * qx; a01 = a01 + alive * px * qy; a02 = a02 + alive * px * qz
        a10 = a10 + alive * py * qx; a11 = a11 + alive * py * qy; a12 = a12 + alive * py * qz
        a20 = a20 + alive * pz * qx; a21 = a21 + alive * pz * qy; a22 = a22 + alive * pz * qz

    A = wp.mat33(a00, a01, a02, a10, a11, a12, a20, a21, a22)
    q_warm = cluster_quat[cid]
    q_new = extract_rotation_muller(A, q_warm, num_iter)
    cluster_quat[cid] = q_new


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
    """LEGACY — direct read-modify-write on pred_pos. Tape-unsafe under
    Warp 1.12 verify_autograd_array_access. Kept only for forward-only paths.
    New solver code uses apply_kernel_ssa + apply_shape_delta_kernel."""
    i = wp.tid()
    if alpha[i] < 0.01 or fixed[i] == 1 or delta_counts[i] == 0:
        return

    dmg = damage[i]
    local_stiff = stiffness * (1.0 - dmg * 0.95)
    inv_count = 1.0 / wp.float32(delta_counts[i])
    pred_pos[i] = pred_pos[i] + local_stiff * inv_count * deltas[i]


# ============== Apply (SSA + delta variant for gradient) ==============
# Same physics as apply_kernel, but Warp 1.12 tape-safe:
#   - Writes shape-matching correction to separate shape_delta buffer
#   - No R-M-W on pred_pos (verify_autograd_array_access clean)
#   - Caller zeroes shape_delta before launch, applies via apply_shape_delta_kernel

@wp.kernel
def apply_kernel_ssa(
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    stiffness: float,
    shape_delta: wp.array(dtype=wp.vec3),  # OUTPUT: per-particle shape correction
):
    i = wp.tid()
    if alpha[i] < 0.01 or fixed[i] == 1 or delta_counts[i] == 0:
        return

    # Single-assignment chain — each local assigned exactly once
    dmg = damage[i]
    local_stiff = stiffness * (1.0 - dmg * 0.95)
    inv_count = 1.0 / wp.float32(delta_counts[i])
    correction = local_stiff * inv_count * deltas[i]

    # Write to separate delta buffer via atomic_add (tape-safe accumulator).
    wp.atomic_add(shape_delta, i, correction)


@wp.kernel
def apply_shape_delta_kernel(
    pred_pos: wp.array(dtype=wp.vec3),
    shape_delta: wp.array(dtype=wp.vec3),
):
    """Apply shape-matching correction to pred_pos via wp.atomic_add (tape-safe).
    Mirrors apply_collision_delta_kernel — same pattern, different input buffer."""
    i = wp.tid()
    wp.atomic_add(pred_pos, i, shape_delta[i])


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


# ============== Batched Collision (per-env knife_pos) ==============

@wp.kernel
def collision_kernel_batched(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    floor_y: float,
    knife_pos: wp.array(dtype=wp.vec3),  # (num_envs,) — per-env knife
    knife_radius: float,
    knife_active: int,
    n_per_env: int,
):
    """Collision kernel for batched envs. Looks up per-env knife_pos."""
    i = wp.tid()
    if alpha[i] < 0.01 or fixed[i] == 1:
        return

    p = pred_pos[i]

    # Floor (Z-up, 2026-04-24)
    if p[2] < floor_y:
        p = wp.vec3(p[0], p[1], floor_y)

    # Knife sphere SDF — per-env knife position
    if knife_active == 1:
        env_id = i / n_per_env
        kp = knife_pos[env_id]
        diff = p - kp
        dist_sq = wp.dot(diff, diff)
        safe_dist = wp.sqrt(dist_sq + 1.0e-8)
        if dist_sq < knife_radius * knife_radius:
            p = kp + diff * (knife_radius / safe_dist)

    pred_pos[i] = p


# ============== Batched Collision (SSA + delta variant for gradient) ==============
# Multi-env gradient-safe version. Same SSA pattern as collision_kernel_ssa.
# Apply delta via the single-env apply_collision_delta_kernel (works for batched too —
# it's per-particle element-wise).

@wp.kernel
def collision_kernel_batched_ssa(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.int32),
    floor_y: float,
    knife_pos: wp.array(dtype=wp.vec3),       # (num_envs,) per-env knife
    knife_radius: float,
    knife_active: int,
    n_per_env: int,
    collision_delta: wp.array(dtype=wp.vec3), # OUTPUT: per-particle delta
):
    i = wp.tid()
    if alpha[i] < 0.01 or fixed[i] == 1:
        return

    # Single-assignment chain (per NVIDIA XPBD + collision_kernel_ssa pattern)
    p0 = pred_pos[i]

    # Z-up (2026-04-24): floor_y param now refers to floor Z value
    p_floor = wp.where(
        p0[2] < floor_y,
        wp.vec3(p0[0], p0[1], floor_y),
        p0,
    )

    if knife_active == 1:
        env_id = i / n_per_env
        kp = knife_pos[env_id]
        diff = p_floor - kp
        dist_sq = wp.dot(diff, diff)
        safe_dist = wp.sqrt(dist_sq + 1.0e-8)
        p_knife = kp + diff * (knife_radius / safe_dist)
        p_new = wp.where(
            dist_sq < knife_radius * knife_radius,
            p_knife,
            p_floor,
        )
    else:
        p_new = p_floor

    wp.atomic_add(collision_delta, i, p_new - p0)
