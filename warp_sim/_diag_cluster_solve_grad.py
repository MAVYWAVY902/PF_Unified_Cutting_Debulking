"""Isolate which stage of cluster_solve_kernel_ssa kills the backward chain.

Build 4 progressively-larger kernels and check if pred_pos.grad flows at each:
  Stage A: just the loop accumulation pattern (pred_pos → cx_sum → cx_mean → sum_output)
  Stage B: + covariance matrix build (A = sum of (p-cm) ⊗ (q-qcm))
  Stage C: + Müller polar (R = quat_to_matrix(extract_rotation_muller(A)))
  Stage D: + goal + correction + atomic_add to deltas (full kernel)

Whichever stage shows grad=0 is the silent-zero source.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pf_shape_matching_warp"))
import numpy as np
import warp as wp
wp.init()
from pf_shape_matching_warp.sim.kernels import extract_rotation_muller


# ============================================================================
# Test kernel — Stage A: loop accumulator → sum output
# ============================================================================

@wp.kernel
def _stage_a(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    out_scalar: wp.array(dtype=wp.float32),  # per-cluster output
):
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    cx_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w = alpha[pid]
        if w < 0.01:
            continue
        p = pred_pos[pid]
        cx_sum = cx_sum + w * p[0]
        tw = tw + w
    if tw < 1.5:
        return
    cx_mean = cx_sum / tw
    out_scalar[cid] = cx_mean


# Stage A_plus: same as A but using `+=` explicitly
@wp.kernel
def _stage_a_plus(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    out_scalar: wp.array(dtype=wp.float32),
):
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    cx_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w = alpha[pid]
        if w < 0.01:
            continue
        p = pred_pos[pid]
        cx_sum += w * p[0]       # explicit +=
        tw += w
    if tw < 1.5:
        return
    cx_mean = cx_sum / tw
    out_scalar[cid] = cx_mean


@wp.kernel
def _probe_sum(out: wp.array(dtype=wp.float32), loss: wp.array(dtype=wp.float32)):
    cid = wp.tid()
    wp.atomic_add(loss, 0, out[cid])


def run_test(kernel, label):
    # Small synthetic cluster: 3 clusters of 3 particles each
    N = 9
    nc = 3
    pred_pos_np = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],  # cluster 0
        [2.0, 2.0, 0.0], [3.0, 2.0, 0.0], [2.0, 3.0, 0.0],  # cluster 1
        [5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [5.0, 6.0, 0.0],  # cluster 2
    ], dtype=np.float32)
    alpha_np = np.ones(N, dtype=np.float32)
    starts = np.array([0, 3, 6], dtype=np.int32)
    sizes = np.array([3, 3, 3], dtype=np.int32)
    pids = np.arange(N, dtype=np.int32)

    pred_pos_leaf = wp.array(pred_pos_np, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    alpha_wp = wp.array(alpha_np, dtype=wp.float32, device="cuda:0")
    starts_wp = wp.array(starts, dtype=wp.int32, device="cuda:0")
    sizes_wp = wp.array(sizes, dtype=wp.int32, device="cuda:0")
    pids_wp = wp.array(pids, dtype=wp.int32, device="cuda:0")
    out_scalar = wp.zeros(nc, dtype=wp.float32, device="cuda:0", requires_grad=True)
    loss_buf = wp.zeros(1, dtype=wp.float32, device="cuda:0", requires_grad=True)

    with wp.Tape() as tape:
        wp.launch(kernel, dim=nc,
                  inputs=[pred_pos_leaf, alpha_wp, starts_wp, sizes_wp, pids_wp, out_scalar])
        wp.launch(_probe_sum, dim=nc, inputs=[out_scalar, loss_buf])

    tape.backward(loss=loss_buf)
    g = pred_pos_leaf.grad.numpy() if pred_pos_leaf.grad is not None else np.zeros_like(pred_pos_np)
    print(f"{label}: loss={float(loss_buf.numpy()[0]):.4f}, |grad|={np.linalg.norm(g):.4f}, grad[0]={g[0]}")


print("Diagnostic: loop accumulator backward")
run_test(_stage_a, "Stage A (`x = x + y` with continue)")
run_test(_stage_a_plus, "Stage A (`x += y` with continue)")


# Stage A_no_continue: remove `if continue`, always accumulate
@wp.kernel
def _stage_a_no_continue(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    out_scalar: wp.array(dtype=wp.float32),
):
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    cx_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)  # mask dead particles
        p = pred_pos[pid]
        cx_sum = cx_sum + w * p[0]
        tw = tw + w
    if tw < 1.5:
        return
    cx_mean = cx_sum / tw
    out_scalar[cid] = cx_mean


run_test(_stage_a_no_continue, "Stage A (no continue, wp.where mask)")


# Stage A_no_sizecheck: remove early `if size < 2: return`
@wp.kernel
def _stage_a_no_sizecheck(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    out_scalar: wp.array(dtype=wp.float32),
):
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]

    cx_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w = alpha[pid]
        p = pred_pos[pid]
        cx_sum = cx_sum + w * p[0]
        tw = tw + w
    cx_mean = cx_sum / (tw + 1.0e-9)
    out_scalar[cid] = cx_mean


run_test(_stage_a_no_sizecheck, "Stage A (no early returns at all)")


# Stage A with ONLY size<2 early return (no tw early return)
@wp.kernel
def _stage_a_only_size_er(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    out_scalar: wp.array(dtype=wp.float32),
):
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    cx_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w = alpha[pid]
        p = pred_pos[pid]
        cx_sum = cx_sum + w * p[0]
        tw = tw + w
    cx_mean = cx_sum / tw
    out_scalar[cid] = cx_mean


run_test(_stage_a_only_size_er, "Stage A (ONLY size<2 early return)")


# Stage A with ONLY tw<1.5 early return (no size early return)
@wp.kernel
def _stage_a_only_tw_er(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    out_scalar: wp.array(dtype=wp.float32),
):
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]

    cx_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w = alpha[pid]
        p = pred_pos[pid]
        cx_sum = cx_sum + w * p[0]
        tw = tw + w
    if tw < 1.5:
        return
    cx_mean = cx_sum / tw
    out_scalar[cid] = cx_mean


run_test(_stage_a_only_tw_er, "Stage A (ONLY tw<1.5 early return, tw computed then checked)")


# Stage B: centroid + rest centroid (2 loops), output mean
@wp.kernel
def _stage_b_two_loops(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    out_scalar: wp.array(dtype=wp.float32),
):
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return
    cx_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        cx_sum = cx_sum + w * p[0]
        tw = tw + w
    tw_safe = wp.where(tw < 1.5, 1.0, tw)
    cx_mean = cx_sum / tw_safe
    # SECOND LOOP (read rest_offsets, compute qcx_sum)
    qcx_sum = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        q = rest_offsets[start + j]
        qcx_sum = qcx_sum + w * q[0]
    qcx_mean = qcx_sum / tw_safe
    out_scalar[cid] = cx_mean - qcx_mean  # cross pred_pos and rest


def run_test_b(kernel, label):
    N = 9; nc = 3
    pred_pos_np = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [2.0, 2.0, 0.0], [3.0, 2.0, 0.0], [2.0, 3.0, 0.0],
        [5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [5.0, 6.0, 0.0],
    ], dtype=np.float32)
    rest_np = pred_pos_np.copy()
    alpha_np = np.ones(N, dtype=np.float32)
    starts = np.array([0, 3, 6], dtype=np.int32)
    sizes = np.array([3, 3, 3], dtype=np.int32)
    pids = np.arange(N, dtype=np.int32)
    pred_pos_leaf = wp.array(pred_pos_np, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    rest_wp = wp.array(rest_np, dtype=wp.vec3, device="cuda:0")
    alpha_wp = wp.array(alpha_np, dtype=wp.float32, device="cuda:0")
    starts_wp = wp.array(starts, dtype=wp.int32, device="cuda:0")
    sizes_wp = wp.array(sizes, dtype=wp.int32, device="cuda:0")
    pids_wp = wp.array(pids, dtype=wp.int32, device="cuda:0")
    out_scalar = wp.zeros(nc, dtype=wp.float32, device="cuda:0", requires_grad=True)
    loss_buf = wp.zeros(1, dtype=wp.float32, device="cuda:0", requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(kernel, dim=nc,
                  inputs=[pred_pos_leaf, alpha_wp, starts_wp, sizes_wp, pids_wp, rest_wp, out_scalar])
        wp.launch(_probe_sum, dim=nc, inputs=[out_scalar, loss_buf])
    tape.backward(loss=loss_buf)
    g = pred_pos_leaf.grad.numpy() if pred_pos_leaf.grad is not None else np.zeros_like(pred_pos_np)
    print(f"{label}: loss={float(loss_buf.numpy()[0]):.4f}, |grad|={np.linalg.norm(g):.4f}")


run_test_b(_stage_b_two_loops, "Stage B (two loops: centroid + rest-centroid)")


# Stage C: add covariance matrix (3 loops total), output trace(A)
@wp.kernel
def _stage_c_covariance(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    out_scalar: wp.array(dtype=wp.float32),
):
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return
    # Centroid loop
    cx_sum = float(0.0); cy_sum = float(0.0); cz_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        cx_sum = cx_sum + w * p[0]; cy_sum = cy_sum + w * p[1]; cz_sum = cz_sum + w * p[2]
        tw = tw + w
    tw_safe = wp.where(tw < 1.5, 1.0, tw)
    cx_mean = cx_sum / tw_safe; cy_mean = cy_sum / tw_safe; cz_mean = cz_sum / tw_safe
    # Rest centroid loop
    qcx_sum = float(0.0); qcy_sum = float(0.0); qcz_sum = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        q = rest_offsets[start + j]
        qcx_sum = qcx_sum + w * q[0]; qcy_sum = qcy_sum + w * q[1]; qcz_sum = qcz_sum + w * q[2]
    qcx_mean = qcx_sum / tw_safe; qcy_mean = qcy_sum / tw_safe; qcz_mean = qcz_sum / tw_safe
    # Covariance loop
    a00 = float(0.0); a11 = float(0.0); a22 = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        q = rest_offsets[start + j]
        px = p[0] - cx_mean; py = p[1] - cy_mean; pz = p[2] - cz_mean
        qx = q[0] - qcx_mean; qy = q[1] - qcy_mean; qz = q[2] - qcz_mean
        a00 = a00 + w * px * qx; a11 = a11 + w * py * qy; a22 = a22 + w * pz * qz
    out_scalar[cid] = a00 + a11 + a22  # trace(A) — depends on all pred_pos


run_test_b(_stage_c_covariance, "Stage C (3 loops: covariance trace)")


# Stage D: add atomic_add to vec3 output (mimics cluster_solve emitting deltas)
@wp.kernel
def _stage_d_atomic_add(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    deltas: wp.array(dtype=wp.vec3),     # requires_grad=True, atomic_add target
):
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return
    cx_sum = float(0.0); cy_sum = float(0.0); cz_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        cx_sum = cx_sum + w * p[0]; cy_sum = cy_sum + w * p[1]; cz_sum = cz_sum + w * p[2]
        tw = tw + w
    tw_safe = wp.where(tw < 1.5, 1.0, tw)
    cx_mean = cx_sum / tw_safe; cy_mean = cy_sum / tw_safe; cz_mean = cz_sum / tw_safe
    # 4th loop: atomic_add correction = mean - pred_pos[pid]
    for j in range(size):
        pid = particle_ids[start + j]
        p = pred_pos[pid]
        correction = wp.vec3(cx_mean - p[0], cy_mean - p[1], cz_mean - p[2])
        wp.atomic_add(deltas, pid, correction)


def run_test_d(kernel, label):
    N = 9; nc = 3
    pred_pos_np = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [2.0, 2.0, 0.0], [3.0, 2.0, 0.0], [2.0, 3.0, 0.0],
        [5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [5.0, 6.0, 0.0],
    ], dtype=np.float32)
    rest_np = pred_pos_np.copy()
    alpha_np = np.ones(N, dtype=np.float32)
    starts = np.array([0, 3, 6], dtype=np.int32)
    sizes = np.array([3, 3, 3], dtype=np.int32)
    pids = np.arange(N, dtype=np.int32)
    pred_pos_leaf = wp.array(pred_pos_np, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    rest_wp = wp.array(rest_np, dtype=wp.vec3, device="cuda:0")
    alpha_wp = wp.array(alpha_np, dtype=wp.float32, device="cuda:0")
    starts_wp = wp.array(starts, dtype=wp.int32, device="cuda:0")
    sizes_wp = wp.array(sizes, dtype=wp.int32, device="cuda:0")
    pids_wp = wp.array(pids, dtype=wp.int32, device="cuda:0")
    deltas_out = wp.zeros(N, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    loss_buf = wp.zeros(1, dtype=wp.float32, device="cuda:0", requires_grad=True)

    @wp.kernel
    def _sum_deltas(d: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        v = d[i]
        wp.atomic_add(out, 0, v[0] + v[1] + v[2])

    with wp.Tape() as tape:
        wp.launch(kernel, dim=nc,
                  inputs=[pred_pos_leaf, alpha_wp, starts_wp, sizes_wp, pids_wp, rest_wp, deltas_out])
        wp.launch(_sum_deltas, dim=N, inputs=[deltas_out, loss_buf])
    tape.backward(loss=loss_buf)
    g = pred_pos_leaf.grad.numpy() if pred_pos_leaf.grad is not None else np.zeros_like(pred_pos_np)
    print(f"{label}: loss={float(loss_buf.numpy()[0]):.4f}, |grad|={np.linalg.norm(g):.4f}")


run_test_d(_stage_d_atomic_add, "Stage D (centroid + atomic_add to deltas)")


# Stage E: + Müller polar → R → R * q_centered → atomic_add (mimics cluster_solve FULL)
@wp.kernel
def _stage_e_muller(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    cluster_quat: wp.array(dtype=wp.quat),
    deltas: wp.array(dtype=wp.vec3),
):
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return
    # centroid
    cx_sum = float(0.0); cy_sum = float(0.0); cz_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        cx_sum = cx_sum + w * p[0]; cy_sum = cy_sum + w * p[1]; cz_sum = cz_sum + w * p[2]
        tw = tw + w
    tw_safe = wp.where(tw < 1.5, 1.0, tw)
    cx_mean = cx_sum / tw_safe; cy_mean = cy_sum / tw_safe; cz_mean = cz_sum / tw_safe
    # rest centroid
    qcx_sum = float(0.0); qcy_sum = float(0.0); qcz_sum = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        q = rest_offsets[start + j]
        qcx_sum = qcx_sum + w * q[0]; qcy_sum = qcy_sum + w * q[1]; qcz_sum = qcz_sum + w * q[2]
    qcx_mean = qcx_sum / tw_safe; qcy_mean = qcy_sum / tw_safe; qcz_mean = qcz_sum / tw_safe
    # covariance
    a00 = float(0.0); a01 = float(0.0); a02 = float(0.0)
    a10 = float(0.0); a11 = float(0.0); a12 = float(0.0)
    a20 = float(0.0); a21 = float(0.0); a22 = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        q = rest_offsets[start + j]
        px = p[0] - cx_mean; py = p[1] - cy_mean; pz = p[2] - cz_mean
        qx = q[0] - qcx_mean; qy = q[1] - qcy_mean; qz = q[2] - qcz_mean
        a00 = a00 + w*px*qx; a01 = a01 + w*px*qy; a02 = a02 + w*px*qz
        a10 = a10 + w*py*qx; a11 = a11 + w*py*qy; a12 = a12 + w*py*qz
        a20 = a20 + w*pz*qx; a21 = a21 + w*pz*qy; a22 = a22 + w*pz*qz
    A = wp.mat33(a00, a01, a02, a10, a11, a12, a20, a21, a22)
    # Müller polar
    q_warm = cluster_quat[cid]
    q_new = extract_rotation_muller(A, q_warm, 8)
    R = wp.quat_to_matrix(q_new)
    # goal loop + atomic_add
    for j in range(size):
        pid = particle_ids[start + j]
        q = rest_offsets[start + j]
        q_centered = wp.vec3(q[0] - qcx_mean, q[1] - qcy_mean, q[2] - qcz_mean)
        goal = R * q_centered + wp.vec3(cx_mean, cy_mean, cz_mean)
        correction = goal - pred_pos[pid]
        wp.atomic_add(deltas, pid, correction)


def run_test_e(kernel, label):
    N = 9; nc = 3
    pred_pos_np = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [2.0, 2.0, 0.0], [3.0, 2.0, 0.0], [2.0, 3.0, 0.0],
        [5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [5.0, 6.0, 0.0],
    ], dtype=np.float32)
    # Perturb slightly so rotation isn't identity
    pred_pos_np += np.random.default_rng(0).normal(0, 0.01, (N, 3)).astype(np.float32)
    rest_np = pred_pos_np.copy()  # same state → identity rotation
    alpha_np = np.ones(N, dtype=np.float32)
    starts = np.array([0, 3, 6], dtype=np.int32)
    sizes = np.array([3, 3, 3], dtype=np.int32)
    pids = np.arange(N, dtype=np.int32)
    q_identity = np.tile([0, 0, 0, 1], (nc, 1)).astype(np.float32)

    pred_pos_leaf = wp.array(pred_pos_np, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    rest_wp = wp.array(rest_np, dtype=wp.vec3, device="cuda:0")
    alpha_wp = wp.array(alpha_np, dtype=wp.float32, device="cuda:0")
    starts_wp = wp.array(starts, dtype=wp.int32, device="cuda:0")
    sizes_wp = wp.array(sizes, dtype=wp.int32, device="cuda:0")
    pids_wp = wp.array(pids, dtype=wp.int32, device="cuda:0")
    cq_wp = wp.array(q_identity, dtype=wp.quat, device="cuda:0", requires_grad=False)
    deltas_out = wp.zeros(N, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    loss_buf = wp.zeros(1, dtype=wp.float32, device="cuda:0", requires_grad=True)

    @wp.kernel
    def _sum_deltas(d: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        v = d[i]
        wp.atomic_add(out, 0, v[0] + v[1] + v[2])

    with wp.Tape() as tape:
        wp.launch(kernel, dim=nc,
                  inputs=[pred_pos_leaf, alpha_wp, starts_wp, sizes_wp, pids_wp, rest_wp, cq_wp, deltas_out])
        wp.launch(_sum_deltas, dim=N, inputs=[deltas_out, loss_buf])
    tape.backward(loss=loss_buf)
    g = pred_pos_leaf.grad.numpy() if pred_pos_leaf.grad is not None else np.zeros_like(pred_pos_np)
    print(f"{label}: loss={float(loss_buf.numpy()[0]):.4f}, |grad|={np.linalg.norm(g):.4f}")


run_test_e(_stage_e_muller, "Stage E (FULL: centroid + covariance + Müller + atomic_add)")


# Stage F: as Stage E but with active_mask + emit_mask + int cast (matches cluster_solve_kernel_ssa exactly)
@wp.kernel
def _stage_f_full(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    cluster_quat: wp.array(dtype=wp.quat),
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
    num_iter: int,
):
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return
    cx_sum = float(0.0); cy_sum = float(0.0); cz_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        cx_sum = cx_sum + w*p[0]; cy_sum = cy_sum + w*p[1]; cz_sum = cz_sum + w*p[2]
        tw = tw + w
    tw_safe = wp.where(tw < 1.5, 1.0, tw)
    active_mask = wp.where(tw < 1.5, 0.0, 1.0)
    cx_mean = cx_sum / tw_safe; cy_mean = cy_sum / tw_safe; cz_mean = cz_sum / tw_safe
    qcx_sum = float(0.0); qcy_sum = float(0.0); qcz_sum = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        q = rest_offsets[start + j]
        qcx_sum = qcx_sum + w*q[0]; qcy_sum = qcy_sum + w*q[1]; qcz_sum = qcz_sum + w*q[2]
    qcx_mean = qcx_sum / tw_safe; qcy_mean = qcy_sum / tw_safe; qcz_mean = qcz_sum / tw_safe
    a00 = float(0.0); a01 = float(0.0); a02 = float(0.0)
    a10 = float(0.0); a11 = float(0.0); a12 = float(0.0)
    a20 = float(0.0); a21 = float(0.0); a22 = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        q = rest_offsets[start + j]
        px = p[0]-cx_mean; py = p[1]-cy_mean; pz = p[2]-cz_mean
        qx = q[0]-qcx_mean; qy = q[1]-qcy_mean; qz = q[2]-qcz_mean
        a00 = a00 + w*px*qx; a01 = a01 + w*px*qy; a02 = a02 + w*px*qz
        a10 = a10 + w*py*qx; a11 = a11 + w*py*qy; a12 = a12 + w*py*qz
        a20 = a20 + w*pz*qx; a21 = a21 + w*pz*qy; a22 = a22 + w*pz*qz
    A = wp.mat33(a00, a01, a02, a10, a11, a12, a20, a21, a22)
    q_warm = cluster_quat[cid]
    q_new = extract_rotation_muller(A, q_warm, num_iter)
    R = wp.quat_to_matrix(q_new)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        particle_alive = wp.where(w_raw < 0.01, 0.0, 1.0)
        emit_mask = particle_alive * active_mask
        q = rest_offsets[start + j]
        q_centered = wp.vec3(q[0]-qcx_mean, q[1]-qcy_mean, q[2]-qcz_mean)
        goal = R * q_centered + wp.vec3(cx_mean, cy_mean, cz_mean)
        correction = (goal - pred_pos[pid]) * emit_mask
        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, int(emit_mask))


def run_test_f(kernel, label):
    N = 9; nc = 3
    pred_pos_np = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [2.0, 2.0, 0.0], [3.0, 2.0, 0.0], [2.0, 3.0, 0.0],
        [5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [5.0, 6.0, 0.0],
    ], dtype=np.float32)
    pred_pos_np += np.random.default_rng(0).normal(0, 0.01, (N, 3)).astype(np.float32)
    rest_np = pred_pos_np.copy()
    alpha_np = np.ones(N, dtype=np.float32)
    starts = np.array([0, 3, 6], dtype=np.int32)
    sizes = np.array([3, 3, 3], dtype=np.int32)
    pids = np.arange(N, dtype=np.int32)
    q_identity = np.tile([0, 0, 0, 1], (nc, 1)).astype(np.float32)

    pred_pos_leaf = wp.array(pred_pos_np, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    rest_wp = wp.array(rest_np, dtype=wp.vec3, device="cuda:0")
    alpha_wp = wp.array(alpha_np, dtype=wp.float32, device="cuda:0")
    starts_wp = wp.array(starts, dtype=wp.int32, device="cuda:0")
    sizes_wp = wp.array(sizes, dtype=wp.int32, device="cuda:0")
    pids_wp = wp.array(pids, dtype=wp.int32, device="cuda:0")
    cq_wp = wp.array(q_identity, dtype=wp.quat, device="cuda:0", requires_grad=False)
    deltas_out = wp.zeros(N, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    dc_out = wp.zeros(N, dtype=wp.int32, device="cuda:0")
    loss_buf = wp.zeros(1, dtype=wp.float32, device="cuda:0", requires_grad=True)

    @wp.kernel
    def _sum_deltas(d: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        v = d[i]
        wp.atomic_add(out, 0, v[0] + v[1] + v[2])

    with wp.Tape() as tape:
        wp.launch(kernel, dim=nc,
                  inputs=[pred_pos_leaf, alpha_wp, starts_wp, sizes_wp, pids_wp, rest_wp, cq_wp,
                          deltas_out, dc_out, 8])
        wp.launch(_sum_deltas, dim=N, inputs=[deltas_out, loss_buf])
    tape.backward(loss=loss_buf)
    g = pred_pos_leaf.grad.numpy() if pred_pos_leaf.grad is not None else np.zeros_like(pred_pos_np)
    print(f"{label}: loss={float(loss_buf.numpy()[0]):.4f}, |grad|={np.linalg.norm(g):.4f}")


run_test_f(_stage_f_full, "Stage F (FULL cluster_solve clone with active_mask + emit_mask + int cast)")


# Stage G: like F but WITHOUT int(emit_mask) atomic_add on int32 delta_counts
@wp.kernel
def _stage_g_no_int_cast(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    cluster_quat: wp.array(dtype=wp.quat),
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
    num_iter: int,
):
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return
    cx_sum = float(0.0); cy_sum = float(0.0); cz_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        cx_sum = cx_sum + w*p[0]; cy_sum = cy_sum + w*p[1]; cz_sum = cz_sum + w*p[2]
        tw = tw + w
    tw_safe = wp.where(tw < 1.5, 1.0, tw)
    active_mask = wp.where(tw < 1.5, 0.0, 1.0)
    cx_mean = cx_sum / tw_safe; cy_mean = cy_sum / tw_safe; cz_mean = cz_sum / tw_safe
    qcx_sum = float(0.0); qcy_sum = float(0.0); qcz_sum = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        q = rest_offsets[start + j]
        qcx_sum = qcx_sum + w*q[0]; qcy_sum = qcy_sum + w*q[1]; qcz_sum = qcz_sum + w*q[2]
    qcx_mean = qcx_sum / tw_safe; qcy_mean = qcy_sum / tw_safe; qcz_mean = qcz_sum / tw_safe
    a00 = float(0.0); a01 = float(0.0); a02 = float(0.0)
    a10 = float(0.0); a11 = float(0.0); a12 = float(0.0)
    a20 = float(0.0); a21 = float(0.0); a22 = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        q = rest_offsets[start + j]
        px = p[0]-cx_mean; py = p[1]-cy_mean; pz = p[2]-cz_mean
        qx = q[0]-qcx_mean; qy = q[1]-qcy_mean; qz = q[2]-qcz_mean
        a00 = a00 + w*px*qx; a01 = a01 + w*px*qy; a02 = a02 + w*px*qz
        a10 = a10 + w*py*qx; a11 = a11 + w*py*qy; a12 = a12 + w*py*qz
        a20 = a20 + w*pz*qx; a21 = a21 + w*pz*qy; a22 = a22 + w*pz*qz
    A = wp.mat33(a00, a01, a02, a10, a11, a12, a20, a21, a22)
    q_warm = cluster_quat[cid]
    q_new = extract_rotation_muller(A, q_warm, num_iter)
    R = wp.quat_to_matrix(q_new)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        particle_alive = wp.where(w_raw < 0.01, 0.0, 1.0)
        emit_mask = particle_alive * active_mask
        q = rest_offsets[start + j]
        q_centered = wp.vec3(q[0]-qcx_mean, q[1]-qcy_mean, q[2]-qcz_mean)
        goal = R * q_centered + wp.vec3(cx_mean, cy_mean, cz_mean)
        correction = (goal - pred_pos[pid]) * emit_mask
        wp.atomic_add(deltas, pid, correction)
        # NO int(emit_mask) atomic_add on delta_counts — just plain +1
        wp.atomic_add(delta_counts, pid, 1)


run_test_f(_stage_g_no_int_cast, "Stage G (F minus int(emit_mask))")


# Stage H: remove active_mask entirely, only use per-particle mask
@wp.kernel
def _stage_h_no_active_mask(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    cluster_quat: wp.array(dtype=wp.quat),
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
    num_iter: int,
):
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return
    cx_sum = float(0.0); cy_sum = float(0.0); cz_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        cx_sum = cx_sum + w*p[0]; cy_sum = cy_sum + w*p[1]; cz_sum = cz_sum + w*p[2]
        tw = tw + w
    tw_safe = wp.where(tw < 1.5, 1.0, tw)
    cx_mean = cx_sum / tw_safe; cy_mean = cy_sum / tw_safe; cz_mean = cz_sum / tw_safe
    qcx_sum = float(0.0); qcy_sum = float(0.0); qcz_sum = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        q = rest_offsets[start + j]
        qcx_sum = qcx_sum + w*q[0]; qcy_sum = qcy_sum + w*q[1]; qcz_sum = qcz_sum + w*q[2]
    qcx_mean = qcx_sum / tw_safe; qcy_mean = qcy_sum / tw_safe; qcz_mean = qcz_sum / tw_safe
    a00 = float(0.0); a01 = float(0.0); a02 = float(0.0)
    a10 = float(0.0); a11 = float(0.0); a12 = float(0.0)
    a20 = float(0.0); a21 = float(0.0); a22 = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        q = rest_offsets[start + j]
        px = p[0]-cx_mean; py = p[1]-cy_mean; pz = p[2]-cz_mean
        qx = q[0]-qcx_mean; qy = q[1]-qcy_mean; qz = q[2]-qcz_mean
        a00 = a00 + w*px*qx; a01 = a01 + w*px*qy; a02 = a02 + w*px*qz
        a10 = a10 + w*py*qx; a11 = a11 + w*py*qy; a12 = a12 + w*py*qz
        a20 = a20 + w*pz*qx; a21 = a21 + w*pz*qy; a22 = a22 + w*pz*qz
    A = wp.mat33(a00, a01, a02, a10, a11, a12, a20, a21, a22)
    q_warm = cluster_quat[cid]
    q_new = extract_rotation_muller(A, q_warm, num_iter)
    R = wp.quat_to_matrix(q_new)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        particle_alive = wp.where(w_raw < 0.01, 0.0, 1.0)
        # NO active_mask — just use particle_alive
        q = rest_offsets[start + j]
        q_centered = wp.vec3(q[0]-qcx_mean, q[1]-qcy_mean, q[2]-qcz_mean)
        goal = R * q_centered + wp.vec3(cx_mean, cy_mean, cz_mean)
        correction = (goal - pred_pos[pid]) * particle_alive
        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, 1)


run_test_f(_stage_h_no_active_mask, "Stage H (F minus active_mask, only per-particle mask)")


# Stage I: active_mask INLINED (computed inside goal loop instead of pre-stored)
@wp.kernel
def _stage_i_inline_active(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    cluster_quat: wp.array(dtype=wp.quat),
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
    num_iter: int,
):
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return
    cx_sum = float(0.0); cy_sum = float(0.0); cz_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        cx_sum = cx_sum + w*p[0]; cy_sum = cy_sum + w*p[1]; cz_sum = cz_sum + w*p[2]
        tw = tw + w
    tw_safe = wp.where(tw < 1.5, 1.0, tw)
    # NO stored active_mask — inline below
    cx_mean = cx_sum / tw_safe; cy_mean = cy_sum / tw_safe; cz_mean = cz_sum / tw_safe
    qcx_sum = float(0.0); qcy_sum = float(0.0); qcz_sum = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        q = rest_offsets[start + j]
        qcx_sum = qcx_sum + w*q[0]; qcy_sum = qcy_sum + w*q[1]; qcz_sum = qcz_sum + w*q[2]
    qcx_mean = qcx_sum / tw_safe; qcy_mean = qcy_sum / tw_safe; qcz_mean = qcz_sum / tw_safe
    a00 = float(0.0); a01 = float(0.0); a02 = float(0.0)
    a10 = float(0.0); a11 = float(0.0); a12 = float(0.0)
    a20 = float(0.0); a21 = float(0.0); a22 = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        q = rest_offsets[start + j]
        px = p[0]-cx_mean; py = p[1]-cy_mean; pz = p[2]-cz_mean
        qx = q[0]-qcx_mean; qy = q[1]-qcy_mean; qz = q[2]-qcz_mean
        a00 = a00 + w*px*qx; a01 = a01 + w*px*qy; a02 = a02 + w*px*qz
        a10 = a10 + w*py*qx; a11 = a11 + w*py*qy; a12 = a12 + w*py*qz
        a20 = a20 + w*pz*qx; a21 = a21 + w*pz*qy; a22 = a22 + w*pz*qz
    A = wp.mat33(a00, a01, a02, a10, a11, a12, a20, a21, a22)
    q_warm = cluster_quat[cid]
    q_new = extract_rotation_muller(A, q_warm, num_iter)
    R = wp.quat_to_matrix(q_new)
    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        particle_alive = wp.where(w_raw < 0.01, 0.0, 1.0)
        # INLINE the active-cluster mask here, not stored:
        emit_mask = particle_alive * wp.where(tw < 1.5, 0.0, 1.0)
        q = rest_offsets[start + j]
        q_centered = wp.vec3(q[0]-qcx_mean, q[1]-qcy_mean, q[2]-qcz_mean)
        goal = R * q_centered + wp.vec3(cx_mean, cy_mean, cz_mean)
        correction = (goal - pred_pos[pid]) * emit_mask
        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, 1)


run_test_f(_stage_i_inline_active, "Stage I (active_mask inlined inside goal loop)")
