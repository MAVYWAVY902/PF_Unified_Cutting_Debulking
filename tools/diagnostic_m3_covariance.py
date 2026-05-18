#!/usr/bin/env python3
"""
M3 diagnostic — does the BINARY mask at α=0.01 actually hurt?

Three sub-tests (run separately or together):

  (a)  α regime scan: in a realistic damage-accumulation scenario, do α
       values actually cross the 0.01 threshold? (If never → M3 is dormant.)

  (b)  R deviation: pure numerical comparison. For a 4-vertex cluster, vary
       one vertex's α from 1→0. Compute rotation R two ways:
         - BINARY mask covariance (production: kernels.py:677)
         - ALPHA-WEIGHTED covariance (paper Eq~\ref{eq:covariance})
       Report angular difference of R as a function of α.

  (c)  Gradient discontinuity: same 4-vertex cluster. Sweep one α through
       0.01 with fine resolution. At each point compute ∂(downstream)/∂α via
       (i) Warp autograd through binary kernel, (ii) FD through binary kernel,
       (iii) Warp autograd through alpha-weighted kernel.
       Look for the spike at α=0.01 in (i) vs (ii) — that's the bug.

NOTE: kernels.py:749 (cluster_goal_scatter_kernel_ssa) ALSO uses binary mask
on the goal correction itself, not just on alive-check. Same discontinuity
applies there. This diagnostic focuses on covariance per M3 scope, but the
same critique extends to goal scatter.

Run:
  python tools/diagnostic_m3_covariance.py --test a
  python tools/diagnostic_m3_covariance.py --test b
  python tools/diagnostic_m3_covariance.py --test c
  python tools/diagnostic_m3_covariance.py --test all
Optional:
  --save-plots  → write PNG side-by-side comparison plots to /tmp/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import warp as wp

wp.init()
wp.config.verify_autograd_array_access = True

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pf_shape_matching_warp.sim.kernels import (
    cluster_compute_sums_kernel_ssa,
    cluster_normalize_centroids_kernel_ssa,
    cluster_compute_covariance_kernel_ssa,   # BINARY mask (production)
    cluster_polar_rotation_kernel_ssa,
)
from pf_shape_matching_warp.sim.pf_damage import (
    update_edge_damage_kernel_ssa_param_arr,
    apply_edge_damage_ssa_kernel,
    update_particle_damage_kernel,
    finalize_particle_damage_kernel_ssa,
    apply_particle_damage_ssa_kernel,
    update_soft_removal_kernel_ssa_param_arr,
)


# ============================================================================
# Alpha-weighted covariance kernel — the PAPER Eq~\ref{eq:covariance}
# ============================================================================
@wp.kernel
def cluster_compute_covariance_kernel_alpha_weighted(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    cx_mean_buf: wp.array(dtype=wp.vec3),
    qc_mean_buf: wp.array(dtype=wp.vec3),
    A_buf: wp.array(dtype=wp.mat33),
):
    """A_C = Σ w_i (p_i - μ)(p̄_i - μ̄)ᵀ with w_i = α_v_i (paper formula).
    Differs from cluster_compute_covariance_kernel_ssa only in one line:
      production uses `alive = where(α<0.01, 0, 1)`  (binary, jumps at α=0.01)
      this uses    `w     = where(α<0.01, 0, α)`     (continuous)
    """
    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]
    if size < 2:
        A_buf[cid] = wp.mat33(0., 0., 0., 0., 0., 0., 0., 0., 0.)
        return

    cx_mean = cx_mean_buf[cid]
    qc_mean = qc_mean_buf[cid]

    a00 = float(0.); a01 = float(0.); a02 = float(0.)
    a10 = float(0.); a11 = float(0.); a12 = float(0.)
    a20 = float(0.); a21 = float(0.); a22 = float(0.)

    for j in range(size):
        pid = particle_ids[start + j]
        w_raw = alpha[pid]
        w = wp.where(w_raw < 0.01, 0.0, w_raw)
        p = pred_pos[pid]
        px = p[0] - cx_mean[0]
        py = p[1] - cx_mean[1]
        pz = p[2] - cx_mean[2]
        q = rest_offsets[start + j]
        qx = q[0] - qc_mean[0]
        qy = q[1] - qc_mean[1]
        qz = q[2] - qc_mean[2]
        a00 = a00 + w * px * qx; a01 = a01 + w * px * qy; a02 = a02 + w * px * qz
        a10 = a10 + w * py * qx; a11 = a11 + w * py * qy; a12 = a12 + w * py * qz
        a20 = a20 + w * pz * qx; a21 = a21 + w * pz * qy; a22 = a22 + w * pz * qz

    A_buf[cid] = wp.mat33(a00, a01, a02, a10, a11, a12, a20, a21, a22)


# Loss kernel: sum of squared rotation matrix entries (proxy for "R magnitude").
# This gives a smooth scalar that depends on R, which depends on α through
# both centroid (continuous) and covariance (binary or alpha-weighted).
@wp.kernel
def loss_R_squared(R_buf: wp.array(dtype=wp.mat33),
                   loss_buf: wp.array(dtype=wp.float32)):
    cid = wp.tid()
    R = R_buf[cid]
    s = float(0.0)
    for i in range(3):
        for j in range(3):
            s = s + R[i, j] * R[i, j]
    wp.atomic_add(loss_buf, 0, s)


# ============================================================================
# Small synthetic cluster — 4 vertices, known geometry
# ============================================================================
def build_4vertex_cluster(perturb=0.15, rotate_deg=12.0, seed=0):
    """Square cluster in rest config (corners of unit square in xy plane).
    Current config = rotated + slightly perturbed (so A_C is non-trivial)."""
    rest = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    theta = np.deg2rad(rotate_deg)
    R_gt = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1],
    ], dtype=np.float32)
    current = (R_gt @ rest.T).T

    rng = np.random.default_rng(seed)
    current += perturb * rng.standard_normal(current.shape).astype(np.float32)

    return rest, current


def setup_cluster_warp(rest_np, current_np, alpha_np):
    """Pack a single-cluster setup into warp arrays."""
    N = rest_np.shape[0]
    pred_pos = wp.array(current_np, dtype=wp.vec3)
    alpha    = wp.array(alpha_np,    dtype=wp.float32, requires_grad=True)
    cluster_starts = wp.array([0],            dtype=wp.int32)
    cluster_sizes  = wp.array([N],            dtype=wp.int32)
    particle_ids   = wp.array(np.arange(N),   dtype=wp.int32)
    rest_offsets   = wp.array(rest_np,        dtype=wp.vec3)
    cluster_quat   = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat)

    cx_sum  = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
    qc_sum  = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
    tw      = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    cx_mean = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
    qc_mean = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
    A_buf   = wp.zeros(1, dtype=wp.mat33, requires_grad=True)
    R_buf   = wp.zeros(1, dtype=wp.mat33, requires_grad=True)
    loss_buf = wp.zeros(1, dtype=wp.float32, requires_grad=True)

    return dict(
        pred_pos=pred_pos, alpha=alpha,
        cluster_starts=cluster_starts, cluster_sizes=cluster_sizes,
        particle_ids=particle_ids, rest_offsets=rest_offsets,
        cluster_quat=cluster_quat,
        cx_sum=cx_sum, qc_sum=qc_sum, tw=tw,
        cx_mean=cx_mean, qc_mean=qc_mean,
        A_buf=A_buf, R_buf=R_buf, loss_buf=loss_buf,
        N=N,
    )


def run_cluster_solve(ctx, covariance_kernel, polar_iter=12):
    """Run the 4-kernel cluster solve pipeline. Returns R as 3x3 numpy."""
    # K1: centroid sums (already alpha-weighted)
    wp.launch(cluster_compute_sums_kernel_ssa, dim=1, inputs=[
        ctx["pred_pos"], ctx["alpha"],
        ctx["cluster_starts"], ctx["cluster_sizes"],
        ctx["particle_ids"], ctx["rest_offsets"],
        ctx["cx_sum"], ctx["qc_sum"], ctx["tw"],
    ])
    # K2: normalize
    wp.launch(cluster_normalize_centroids_kernel_ssa, dim=1, inputs=[
        ctx["cluster_sizes"], ctx["cx_sum"], ctx["qc_sum"], ctx["tw"],
        ctx["cx_mean"], ctx["qc_mean"],
    ])
    # K3: covariance (swap-in here — binary or alpha-weighted)
    wp.launch(covariance_kernel, dim=1, inputs=[
        ctx["pred_pos"], ctx["alpha"],
        ctx["cluster_starts"], ctx["cluster_sizes"],
        ctx["particle_ids"], ctx["rest_offsets"],
        ctx["cx_mean"], ctx["qc_mean"],
        ctx["A_buf"],
    ])
    # K4: polar decomp
    wp.launch(cluster_polar_rotation_kernel_ssa, dim=1, inputs=[
        ctx["cluster_sizes"], ctx["A_buf"], ctx["cluster_quat"],
        polar_iter, ctx["R_buf"],
    ])
    return ctx["R_buf"].numpy()[0]


def rotation_angle_deg(R1, R2):
    """Geodesic angle in degrees between two rotation matrices."""
    Rdiff = R1.T @ R2
    # safer arccos clamp
    cos_th = max(-1.0, min(1.0, (np.trace(Rdiff) - 1.0) / 2.0))
    return float(np.degrees(np.arccos(cos_th)))


# ============================================================================
# TEST (a): α regime scan in realistic damage accumulation
# ============================================================================
def test_a_alpha_regime():
    """Run actual damage pipeline N cycles, log min α each cycle.

    Mirrors calibrate_stage2_autograd.py setup: 1D chain N=20, knife at
    center, accumulate damage over many cycles. Report:
      - α distribution over time
      - how many α values cross 0.01
      - which cycles
    """
    print("\n" + "=" * 72)
    print("  TEST (a): α regime scan in realistic damage accumulation")
    print("=" * 72)

    N = 20
    E = N - 1
    pos_np = np.zeros((N, 3), dtype=np.float32)
    pos_np[:, 0] = np.linspace(0.3, 0.7, N)
    pos_np[:, 1] = 0.5
    pos_np[:, 2] = 0.5
    edge_a_np = np.arange(E, dtype=np.int32)
    edge_b_np = np.arange(1, N, dtype=np.int32)

    pos       = wp.array(pos_np,         dtype=wp.vec3)
    edge_a    = wp.array(edge_a_np,      dtype=wp.int32)
    edge_b    = wp.array(edge_b_np,      dtype=wp.int32)
    pf_rest   = wp.array(np.full(E, 0.5, dtype=np.float32), dtype=wp.float32)
    knife_pos = wp.array([[0.5, 0.5, 0.5]], dtype=wp.vec3)
    fixed     = wp.array(np.zeros(N, dtype=np.int32), dtype=wp.int32)

    knife_radius = wp.array([0.1], dtype=wp.float32, requires_grad=True)
    alpha_min    = wp.array([3.0], dtype=wp.float32, requires_grad=True)
    sharpness    = wp.array([20.0], dtype=wp.float32, requires_grad=True)
    threshold    = wp.array([0.3], dtype=wp.float32, requires_grad=True)

    n_cycles = 80
    alpha_buf  = [wp.array(np.ones(N, dtype=np.float32),  dtype=wp.float32, requires_grad=True) for _ in range(n_cycles+1)]
    edge_dmg   = [wp.zeros(E, dtype=wp.float32, requires_grad=True) for _ in range(n_cycles+1)]
    part_dmg   = [wp.zeros(N, dtype=wp.float32, requires_grad=True) for _ in range(n_cycles+1)]

    crossed = []  # (cycle, vertex_id, alpha_before, alpha_after)
    prev_alpha = np.ones(N)
    min_alpha_per_cycle = [1.0]

    for c in range(n_cycles):
        # edge_damage
        ed_delta = wp.zeros(E, dtype=wp.float32, requires_grad=True)
        wp.launch(update_edge_damage_kernel_ssa_param_arr, dim=E, inputs=[
            pos, alpha_buf[c], edge_a, edge_b, pf_rest, edge_dmg[c],
            knife_pos, knife_radius, alpha_min, ed_delta,
        ])
        wp.launch(apply_edge_damage_ssa_kernel, dim=E, inputs=[
            edge_dmg[c], ed_delta, edge_dmg[c+1],
        ])
        # particle_damage
        d_sum   = wp.zeros(N, dtype=wp.float32, requires_grad=True)
        d_count = wp.zeros(N, dtype=wp.float32, requires_grad=True)
        wp.launch(update_particle_damage_kernel, dim=E, inputs=[
            d_sum, d_count, alpha_buf[c], edge_a, edge_b, edge_dmg[c+1],
        ])
        pd_delta = wp.zeros(N, dtype=wp.float32, requires_grad=True)
        wp.launch(finalize_particle_damage_kernel_ssa, dim=N, inputs=[
            d_sum, d_count, part_dmg[c], pd_delta,
        ])
        wp.launch(apply_particle_damage_ssa_kernel, dim=N, inputs=[
            part_dmg[c], pd_delta, part_dmg[c+1],
        ])
        # soft removal
        wp.launch(update_soft_removal_kernel_ssa_param_arr, dim=N, inputs=[
            part_dmg[c+1], alpha_buf[c], fixed, threshold, sharpness,
            alpha_buf[c+1],
        ])

        cur_alpha = alpha_buf[c+1].numpy()
        min_alpha_per_cycle.append(float(cur_alpha.min()))

        # detect crossings
        for i in range(N):
            if prev_alpha[i] > 0.01 and cur_alpha[i] <= 0.01:
                crossed.append((c+1, i, float(prev_alpha[i]), float(cur_alpha[i])))
        prev_alpha = cur_alpha.copy()

    # report
    final_alpha = alpha_buf[-1].numpy()
    print(f"  N={N} vertices, knife at x=0.5, {n_cycles} damage cycles.")
    print(f"  Final α distribution:")
    print(f"    min        = {final_alpha.min():.6f}")
    print(f"    max        = {final_alpha.max():.6f}")
    print(f"    median     = {np.median(final_alpha):.6f}")
    print(f"    < 0.01     = {(final_alpha < 0.01).sum()} / {N} vertices")
    print(f"    in (0.01, 0.5) = {((final_alpha > 0.01) & (final_alpha < 0.5)).sum()}")
    print()
    print(f"  α=0.01 crossings: {len(crossed)} total")
    if crossed[:10]:
        print(f"  First 10 crossings (cycle, vid, before, after):")
        for x in crossed[:10]:
            print(f"    cycle {x[0]:3d}  vid {x[1]:2d}  {x[2]:.4f} → {x[3]:.4f}")
    print()
    print(f"  Min α per cycle (every 10):")
    for i in range(0, n_cycles+1, 10):
        print(f"    cycle {i:3d}: min α = {min_alpha_per_cycle[i]:.6f}")

    verdict = "REAL ISSUE" if crossed else "DORMANT (α never crosses 0.01)"
    print(f"\n  VERDICT (a): {verdict}")
    return crossed, min_alpha_per_cycle


# ============================================================================
# TEST (b): R deviation between binary vs alpha-weighted covariance
# ============================================================================
def test_b_R_deviation(save_plot=False):
    """Sweep one vertex's α from 1→0. Compute R both ways. Plot angular diff."""
    print("\n" + "=" * 72)
    print("  TEST (b): R deviation between binary vs alpha-weighted covariance")
    print("=" * 72)

    rest_np, current_np = build_4vertex_cluster(perturb=0.15, rotate_deg=12.0)

    alpha_values = np.concatenate([
        np.linspace(1.0, 0.05, 40),
        np.linspace(0.05, 0.0,  40),  # finer near 0.01
    ])

    angles = []
    for a in alpha_values:
        # vertex 3 gets varying α; others stay at 1
        alpha_np = np.array([1.0, 1.0, 1.0, float(a)], dtype=np.float32)

        ctx_bin = setup_cluster_warp(rest_np, current_np, alpha_np)
        R_bin = run_cluster_solve(ctx_bin, cluster_compute_covariance_kernel_ssa)

        ctx_alpha = setup_cluster_warp(rest_np, current_np, alpha_np)
        R_alpha = run_cluster_solve(ctx_alpha, cluster_compute_covariance_kernel_alpha_weighted)

        ang = rotation_angle_deg(R_bin, R_alpha)
        angles.append(ang)

    angles = np.array(angles)
    print(f"  Cluster: 4 vertices (unit square), 3 healthy (α=1), 1 swept.")
    print(f"  R angular deviation (binary vs alpha-weighted), degrees:")
    print(f"    at α=1.0   : {angles[0]:.4f} °  (should be 0)")
    print(f"    at α=0.5   : {angles[np.argmin(np.abs(alpha_values-0.5))]:.4f} °")
    print(f"    at α=0.1   : {angles[np.argmin(np.abs(alpha_values-0.1))]:.4f} °")
    print(f"    at α=0.02  : {angles[np.argmin(np.abs(alpha_values-0.02))]:.4f} °")
    print(f"    at α=0.011 : {angles[np.argmin(np.abs(alpha_values-0.011))]:.4f} °  ← just above threshold")
    print(f"    at α=0.005 : {angles[np.argmin(np.abs(alpha_values-0.005))]:.4f} °  ← just below threshold")
    print(f"    at α=0.0   : {angles[-1]:.4f} °  (should be 0)")
    print(f"  Max deviation across sweep: {angles.max():.4f} °")
    print(f"  Argmax α: {alpha_values[np.argmax(angles)]:.4f}")

    if save_plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(alpha_values, angles, 'b-', linewidth=1.5)
            ax.axvline(0.01, color='r', linestyle='--', alpha=0.7, label='α=0.01 threshold')
            ax.set_xlabel('α of vertex 3')
            ax.set_ylabel('R angular deviation (degrees)')
            ax.set_title('M3 test (b): R deviation, binary vs alpha-weighted covariance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out = '/tmp/m3_test_b_R_deviation.png'
            plt.savefig(out, dpi=120)
            print(f"  Plot saved: {out}")
        except ImportError:
            print("  (matplotlib not installed — skipping plot)")

    verdict = "MATERIAL" if angles.max() > 0.5 else "SMALL"
    print(f"\n  VERDICT (b): R deviation is {verdict} ({angles.max():.2f}° max)")
    return alpha_values, angles


# ============================================================================
# TEST (c): gradient discontinuity around α=0.01
# ============================================================================
def test_c_gradient_discontinuity(save_plot=False):
    """For one swept α, compute ∂(loss)/∂α three ways:
       1. autograd through BINARY covariance kernel (production)
       2. FD through BINARY covariance kernel
       3. autograd through ALPHA-WEIGHTED kernel (reference)

    If binary path is consistent: (1) and (2) should agree.
    If binary path has hidden discontinuity: (2) shows spike near α=0.01
       where (1) shows nothing.
    (3) is the smooth reference both should converge toward.
    """
    print("\n" + "=" * 72)
    print("  TEST (c): gradient ∂loss/∂α — autograd vs FD vs reference")
    print("=" * 72)

    rest_np, current_np = build_4vertex_cluster(perturb=0.15, rotate_deg=12.0)

    # finer sampling near the threshold
    alpha_values = np.concatenate([
        np.linspace(0.5,   0.05,  20),
        np.linspace(0.05,  0.015, 20),
        np.linspace(0.015, 0.005, 60),  # finest near threshold
        np.linspace(0.005, 0.0,   20),
    ])

    fd_eps = 1e-4
    ag_bin_grads, fd_bin_grads, ag_alpha_grads = [], [], []

    for a in alpha_values:
        # === (1) autograd through BINARY kernel ===
        alpha_np = np.array([1.0, 1.0, 1.0, float(a)], dtype=np.float32)
        ctx = setup_cluster_warp(rest_np, current_np, alpha_np)
        tape = wp.Tape()
        with tape:
            run_cluster_solve(ctx, cluster_compute_covariance_kernel_ssa)
            wp.launch(loss_R_squared, dim=1, inputs=[ctx["R_buf"], ctx["loss_buf"]])
        tape.backward(loss=ctx["loss_buf"])
        ag_bin = float(ctx["alpha"].grad.numpy()[3])
        ag_bin_grads.append(ag_bin)

        # === (2) FD through BINARY kernel ===
        alpha_plus = alpha_np.copy(); alpha_plus[3]  += fd_eps
        alpha_minus = alpha_np.copy(); alpha_minus[3] -= fd_eps
        ctx_p = setup_cluster_warp(rest_np, current_np, alpha_plus)
        run_cluster_solve(ctx_p, cluster_compute_covariance_kernel_ssa)
        wp.launch(loss_R_squared, dim=1, inputs=[ctx_p["R_buf"], ctx_p["loss_buf"]])
        L_plus = float(ctx_p["loss_buf"].numpy()[0])

        ctx_m = setup_cluster_warp(rest_np, current_np, alpha_minus)
        run_cluster_solve(ctx_m, cluster_compute_covariance_kernel_ssa)
        wp.launch(loss_R_squared, dim=1, inputs=[ctx_m["R_buf"], ctx_m["loss_buf"]])
        L_minus = float(ctx_m["loss_buf"].numpy()[0])
        fd_bin = (L_plus - L_minus) / (2 * fd_eps)
        fd_bin_grads.append(fd_bin)

        # === (3) autograd through ALPHA-WEIGHTED kernel (reference) ===
        ctx_ref = setup_cluster_warp(rest_np, current_np, alpha_np)
        tape_ref = wp.Tape()
        with tape_ref:
            run_cluster_solve(ctx_ref, cluster_compute_covariance_kernel_alpha_weighted)
            wp.launch(loss_R_squared, dim=1, inputs=[ctx_ref["R_buf"], ctx_ref["loss_buf"]])
        tape_ref.backward(loss=ctx_ref["loss_buf"])
        ag_alpha = float(ctx_ref["alpha"].grad.numpy()[3])
        ag_alpha_grads.append(ag_alpha)

    ag_bin_grads   = np.array(ag_bin_grads)
    fd_bin_grads   = np.array(fd_bin_grads)
    ag_alpha_grads = np.array(ag_alpha_grads)

    print(f"  Sweep: {len(alpha_values)} α points, dense near 0.01.")
    print(f"  FD epsilon: {fd_eps}")
    print()
    print(f"  Sample points (α, autograd-binary, FD-binary, autograd-alpha-weighted):")
    for tgt in [0.5, 0.1, 0.02, 0.011, 0.010, 0.009, 0.005, 0.001]:
        i = np.argmin(np.abs(alpha_values - tgt))
        print(f"    α={alpha_values[i]:.4f}  ag_bin={ag_bin_grads[i]:+.4e}  "
              f"fd_bin={fd_bin_grads[i]:+.4e}  ag_alpha={ag_alpha_grads[i]:+.4e}")

    # Quantify discontinuity: at threshold region, look at FD gradient magnitudes
    near_thresh = (alpha_values > 0.005) & (alpha_values < 0.015)
    fd_max_near = np.abs(fd_bin_grads[near_thresh]).max()
    ag_max_near = np.abs(ag_bin_grads[near_thresh]).max()
    ratio = fd_max_near / max(ag_max_near, 1e-12)

    print()
    print(f"  In α ∈ (0.005, 0.015) — the threshold neighborhood:")
    print(f"    Max |autograd_binary|  = {ag_max_near:.4e}")
    print(f"    Max |FD_binary|        = {fd_max_near:.4e}")
    print(f"    Ratio FD/autograd      = {ratio:.2f}")
    print(f"    (If binary mask hides the jump, FD/autograd >> 1 near threshold)")

    if save_plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(alpha_values, ag_bin_grads,   'b-',  label='autograd through BINARY (production)')
            ax.plot(alpha_values, fd_bin_grads,   'r--', label='FD through BINARY',          alpha=0.8)
            ax.plot(alpha_values, ag_alpha_grads, 'g-',  label='autograd through ALPHA-WEIGHTED (reference)', alpha=0.7)
            ax.axvline(0.01, color='k', linestyle=':', alpha=0.5, label='α=0.01 threshold')
            ax.set_xlabel('α of vertex 3')
            ax.set_ylabel('∂(loss)/∂α[3]')
            ax.set_title('M3 test (c): gradient continuity across α=0.01')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            plt.tight_layout()
            out = '/tmp/m3_test_c_gradient.png'
            plt.savefig(out, dpi=120)
            print(f"  Plot saved: {out}")
        except ImportError:
            print("  (matplotlib not installed — skipping plot)")

    verdict = "DISCONTINUITY CONFIRMED" if ratio > 10 else ("MILD" if ratio > 2 else "INVISIBLE in this loss")
    print(f"\n  VERDICT (c): {verdict} (FD/autograd ratio = {ratio:.2f})")
    return alpha_values, ag_bin_grads, fd_bin_grads, ag_alpha_grads


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["a", "b", "c", "all"], default="all")
    ap.add_argument("--save-plots", action="store_true",
                    help="Save matplotlib PNG plots to /tmp/")
    args = ap.parse_args()

    if args.test in ("a", "all"):
        test_a_alpha_regime()
    if args.test in ("b", "all"):
        test_b_R_deviation(save_plot=args.save_plots)
    if args.test in ("c", "all"):
        test_c_gradient_discontinuity(save_plot=args.save_plots)

    print("\n" + "=" * 72)
    print("  All tests done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
