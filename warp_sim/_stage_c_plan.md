# Stage C — Implementation Plan
**cluster_solve_kernel SSA rewrite + Müller 2016 iterative polar decomposition**
2026-04-22, scoped after Stage A + B completion

## Why this is the big one

cluster_solve_kernel is the only remaining non-SSA kernel. It carries **both** major blockers we've been avoiding:

| Blocker | Stage A/B addressed? | Stage C addresses? |
|---|---|---|
| SSA silent-zero in local re-assigns (`cx = cx / tw`, branch `R = U_fixed * V^T`) | NO | YES |
| SVD backward numerical instability (`∂U/∂A` poles at clustered singular values) | NO | YES (via Müller 2016) |

After Stage C, the last "RED" kernel is clean AND we've replaced the `wp.svd3` call that was going to cause NaN/inf on ~some-fraction of 30k clusters per frame. This is what actually enables **full single-step L2**.

## Algorithm: Müller 2016 iterative quaternion polar

**Source (verbatim from paper Appendix D, via research agent):**

```cpp
// Input: A (covariance 3x3), output: q (rotation quaternion)
// Initialize q from previous frame's value (warm start)
for (iter = 0; iter < MAX_ITER; ++iter) {
    R = q.to_matrix();
    omega_num = R.col(0).cross(A.col(0))
              + R.col(1).cross(A.col(1))
              + R.col(2).cross(A.col(2));
    omega_denom = |R.col(0).dot(A.col(0))
                  + R.col(1).dot(A.col(1))
                  + R.col(2).dot(A.col(2))| + 1e-9;
    omega = omega_num / omega_denom;
    w = |omega|;
    if (w < 1e-9) break;
    q = AxisAngle(w, omega/w) * q;    // LEFT multiply (important)
    q = normalize(q);
}
```

**Convergence**: 4-6 iterations warm-start, 8-10 cold. Paper claims 60-80% faster than SVD-based Irving polar.

**No det<0 fix needed**: quaternion rotation has det=+1 by construction. The ugly branch `if det < 0: R = U_fixed * V^T` in current cluster_solve GONE.

**Near-coplanar robustness**: when a cluster is rank-deficient, the corresponding `r_i × a_i` term vanishes naturally. No explicit special-case.

**Reference implementation (canonical ground truth)**: Jan Bender's (paper co-author) PositionBasedDynamics library, file `MathFunctions.cpp::extractRotation`. URL: github.com/InteractiveComputerGraphics/PositionBasedDynamics.

## Warp 1.12 quaternion API (research-verified)

All of these exist AND have hand-written adjoints in `warp/native/quat.h` — i.e. gradient flows through them correctly:

| Warp op | Use in Müller |
|---|---|
| `wp.quat(x,y,z,w)` | constructor |
| `wp.quat_identity()` | cold-start init |
| `wp.quat_from_axis_angle(axis, angle)` | `AxisAngle(w, omega/w)` |
| `wp.quat_to_matrix(q) -> wp.mat33` | `q.to_matrix()` (implemented as 3× quat_rotate, slower but fine) |
| `wp.mul(q_a, q_b)` | Hamilton product (LEFT multiply: `wp.mul(dq, q)` — order matters) |
| `wp.normalize(q)` | quaternion renorm |
| `wp.cross`, `wp.dot`, `wp.abs`, `wp.length` | scalar ops for omega |

**Gotcha to watch**: `wp.normalize` returns identity `(0,0,0,1)` if `|q| < kEps`, skipping the gradient branch. In practice warm-start keeps `|q| ≈ 1` always, so this is fine — but if we ever hit numerical collapse, the backward will silently zero.

## Current `cluster_solve_kernel` problems (from audit)

```python
# Multiple local re-assignments — all silent-zero landmines
cx = cx / tw                  # ❌ non-additive re-assign
cy = cy / tw                  # ❌
cz = cz / tw                  # ❌
qcx = qcx / tw                # ❌
qcy = qcy / tw                # ❌
qcz = qcz / tw                # ❌

# Conditional matrix re-assign
R = U * wp.transpose(V)                       # first assign
if det < 0.0:
    U_fixed = wp.mat33(...)                   # (this is fine, fresh name)
    R = U_fixed * wp.transpose(V)             # ❌ R re-assigned inside conditional

# SVD call that produces NaN/inf in rare configurations
U, sigma, V = wp.svd3(A)                       # ❌ unstable backward
```

## New `cluster_solve_kernel_ssa` design

```python
@wp.func
def extract_rotation_muller(A: wp.mat33, q_init: wp.quat, num_iter: int) -> wp.quat:
    """Müller 2016 iterative polar decomposition. num_iter fixed (no break) for
    tape-safety — 4 iterations with warm-start, 8 for cold-start."""
    q = q_init
    for _ in range(num_iter):
        R = wp.quat_to_matrix(q)
        r0 = wp.vec3(R[0,0], R[1,0], R[2,0])
        r1 = wp.vec3(R[0,1], R[1,1], R[2,1])
        r2 = wp.vec3(R[0,2], R[1,2], R[2,2])
        a0 = wp.vec3(A[0,0], A[1,0], A[2,0])
        a1 = wp.vec3(A[0,1], A[1,1], A[2,1])
        a2 = wp.vec3(A[0,2], A[1,2], A[2,2])
        num = wp.cross(r0, a0) + wp.cross(r1, a1) + wp.cross(r2, a2)
        denom = wp.abs(wp.dot(r0, a0) + wp.dot(r1, a1) + wp.dot(r2, a2)) + 1.0e-9
        omega = num / denom
        w = wp.length(omega)
        axis = omega / (w + 1.0e-20)
        dq = wp.quat_from_axis_angle(axis, w)
        q = wp.normalize(wp.mul(dq, q))      # Python re-bind → Warp compiles as SSA
    return q


@wp.kernel
def cluster_solve_kernel_ssa(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    cluster_quat: wp.array(dtype=wp.quat),     # NEW: per-cluster warm-start q
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32),
):
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        return

    # --- 1. Weighted centroid (SSA: accumulators → fresh-named means) ---
    cx_sum = float(0.0); cy_sum = float(0.0); cz_sum = float(0.0); tw = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w = alpha[pid]
        if w < 0.01:
            continue
        p = pred_pos[pid]
        cx_sum = cx_sum + w * p[0]        # accumulator += (Warp allows)
        cy_sum = cy_sum + w * p[1]
        cz_sum = cz_sum + w * p[2]
        tw = tw + w
    if tw < 1.5:
        return
    # Fresh names for means (was `cx = cx / tw` — silent-zero)
    cx_mean = cx_sum / tw
    cy_mean = cy_sum / tw
    cz_mean = cz_sum / tw

    # --- 2a. Rest-offset centroid (same pattern) ---
    qcx_sum = float(0.0); qcy_sum = float(0.0); qcz_sum = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w = alpha[pid]
        if w < 0.01:
            continue
        q = rest_offsets[start + j]
        qcx_sum = qcx_sum + w * q[0]
        qcy_sum = qcy_sum + w * q[1]
        qcz_sum = qcz_sum + w * q[2]
    qcx_mean = qcx_sum / tw
    qcy_mean = qcy_sum / tw
    qcz_mean = qcz_sum / tw

    # --- 2b. Covariance A_pq (3x3, single-assigned) ---
    a00 = float(0.0); a01 = float(0.0); a02 = float(0.0)
    a10 = float(0.0); a11 = float(0.0); a12 = float(0.0)
    a20 = float(0.0); a21 = float(0.0); a22 = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w = alpha[pid]
        if w < 0.01:
            continue
        p = pred_pos[pid]
        q = rest_offsets[start + j]
        px = p[0] - cx_mean
        py = p[1] - cy_mean
        pz = p[2] - cz_mean
        qx = q[0] - qcx_mean
        qy = q[1] - qcy_mean
        qz = q[2] - qcz_mean
        a00 = a00 + px * qx;  a01 = a01 + px * qy;  a02 = a02 + px * qz
        a10 = a10 + py * qx;  a11 = a11 + py * qy;  a12 = a12 + py * qz
        a20 = a20 + pz * qx;  a21 = a21 + pz * qy;  a22 = a22 + pz * qz
    A = wp.mat33(a00, a01, a02,
                 a10, a11, a12,
                 a20, a21, a22)

    # --- 3. Rotation via Müller polar (replaces wp.svd3) ---
    q_warm = cluster_quat[cid]        # previous-frame rotation as warm-start
    q_new = extract_rotation_muller(A, q_warm, 4)
    # Persist for next frame — write BACK to cluster_quat (single write, no R-M-W)
    cluster_quat[cid] = q_new
    R = wp.quat_to_matrix(q_new)

    # --- 4. Goal positions → atomic_add deltas (unchanged; already SSA-safe) ---
    for j in range(size):
        pid = particle_ids[start + j]
        w = alpha[pid]
        if w < 0.01:
            continue
        q = rest_offsets[start + j]
        q_centered = wp.vec3(q[0] - qcx_mean, q[1] - qcy_mean, q[2] - qcz_mean)
        goal = R * q_centered + wp.vec3(cx_mean, cy_mean, cz_mean)
        correction = goal - pred_pos[pid]
        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, 1)
```

**Key structural improvements:**
1. Six `_sum`/`_mean` pairs replace six re-assigned scalars → SSA-clean
2. `wp.svd3` + `if det < 0: R = ...` gone — replaced by 4-iter Müller
3. `cluster_quat[cid]` written exactly once (at end of kernel) → no R-M-W
4. `@wp.func extract_rotation_muller` is a device function — Warp inlines + differentiates

## Data-structure addition

`pf_shape_matching_warp/sim/particles.py::init_cluster_arrays` — add:

```python
cluster_arrays["quat"] = wp.array(
    [wp.quat(0.0, 0.0, 0.0, 1.0)] * len(starts),  # identity
    dtype=wp.quat, device=device, requires_grad=True,
)
```

This is the warm-start buffer — one quaternion per cluster, persisted across frames.

## Sub-stage breakdown (within Stage C, ~1-2 weeks)

| Sub-stage | Work | Estimated time |
|---|---|---|
| **C.1** | `@wp.func extract_rotation_muller` + unit test vs Eigen/scipy polar on 100 random matrices (forward correctness) | 2 days |
| **C.2** | Warm-start quat array, wire into `cluster_solve_kernel_ssa` + solver, forward equivalence vs legacy `wp.svd3` path | 2 days |
| **C.3** | Gate test: 5 axes — forward equiv, `verify_autograd` clean, gradient through A to deltas non-zero, Jacobian FD-cosine > 0.9, static pattern audit | 2 days |
| **C.4** | Phase 6 regression 38/38, PPO_10-style L1 training smoke test (confirm physics quality unchanged at 20k steps) | 1 day |
| **C.5** | Fig. 3 data — sweep Müller num_iter ∈ {2,4,6,8}, warm vs cold start, document convergence + gradient stability | 2 days |

## Risk register

| Risk | Mitigation |
|---|---|
| Müller 4-iter insufficient for some pathological clusters → forward regression vs SVD | Increase to 6 iters if needed; paper says 8 is safe upper bound for cold start |
| `cluster_quat` state diverges after many frames | Renormalize every step (already in iteration); reset to identity on env.reset() |
| Per-cluster warm-start helps but atomic race if batched envs share quat buffer | Use batched buffer (num_envs × num_clusters) for batched solver — matches existing pattern for per-env knife_pos |
| Gradient through `wp.normalize(q)` dies when `|q| < kEps` | Warm-start keeps `|q| ≈ 1`; monitor worst-case in Fig. 3 sweep |
| Slower than `wp.svd3` (3 quat_rotate per iter × 4 iter = 12 ops vs 1 SVD call) | Paper reports 60-80% speedup vs SVD; validate with benchmark in C.4 |

## What Stage C unlocks (reiterate for clarity)

After Stage C passes all 5 axes + regression:
- ✅ **Full single-step L2 L2 tape** covering complete `_substep` becomes viable (Stage D)
- ✅ **C3b sys-ID** has a working `∂loss/∂stiffness` path through cluster_solve
- ✅ **Fig. 3 gradient sweep** (Stage E) has meaningful data on the new solver's gradient stability

What it does NOT unlock:
- ❌ L3 (Block 3 Jacobi paradox + Block 4 damage chain amplification remain — Stage C only fixes Block 1 SSA + Block 2 SVD for this kernel)

## Reference code sources (no license issues)

- **Canonical C++ reference**: Jan Bender PositionBasedDynamics `MathFunctions.cpp::extractRotation`
- **Warp quat API**: `warp/native/quat.h` lines 181, 305, 395, 456, 1004, 1148, 1449
- **Paper text**: `/tmp/muller2016.txt` (pdftotext output, research agent artifact)
