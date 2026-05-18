#!/usr/bin/env python3
"""
DRY-RUN: what happens to Stage B FD parity if we switch covariance kernel?

DOES NOT modify any production code. Uses inline alternative covariance
kernels and compares against the production binary version, on a Stage
B-style setup (full damage pipeline + cluster solve at the end).

Three kernel variants compared:
  V1  BINARY          alive = where(α<0.01, 0, 1)        ← production
  V2  ALPHA + clamp   w = where(α<0.01, 0, α)            ← partial paper-aligned
  V3  ALPHA no clamp  w = α                              ← full paper-aligned

For each variant, compute autograd vs FD gradient of a downstream rotation
loss wrt (sharpness, alpha_min) — the same two calibration params Stage B
validates against.

Report rel_err per variant. If V2/V3 give comparable rel_err to V1, the
switch is safe.

Run:
  python3 tools/dry_run_m3_kernel_switch.py
"""
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
    cluster_compute_covariance_kernel_ssa,   # V1 binary (production)
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
# Three covariance kernel variants
# ============================================================================

# V1 imported above: cluster_compute_covariance_kernel_ssa  (BINARY)


@wp.kernel
def cov_kernel_V2_alpha_with_clamp(
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
    """V2: w = where(α<0.01, 0, α). Keeps the 0.01 clamp for safety."""
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
        px = p[0] - cx_mean[0]; py = p[1] - cx_mean[1]; pz = p[2] - cx_mean[2]
        q = rest_offsets[start + j]
        qx = q[0] - qc_mean[0]; qy = q[1] - qc_mean[1]; qz = q[2] - qc_mean[2]
        a00 = a00 + w * px * qx; a01 = a01 + w * px * qy; a02 = a02 + w * px * qz
        a10 = a10 + w * py * qx; a11 = a11 + w * py * qy; a12 = a12 + w * py * qz
        a20 = a20 + w * pz * qx; a21 = a21 + w * pz * qy; a22 = a22 + w * pz * qz

    # Tikhonov for safety (matches what we'd add if shipping V2)
    A_buf[cid] = wp.mat33(a00 + 1.0e-6, a01,           a02,
                          a10,          a11 + 1.0e-6,  a12,
                          a20,          a21,           a22 + 1.0e-6)


@wp.kernel
def cov_kernel_V3_alpha_no_clamp(
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
    """V3: w = α directly. No clamp — pure paper formula."""
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
        w = alpha[pid]   # no clamp
        p = pred_pos[pid]
        px = p[0] - cx_mean[0]; py = p[1] - cx_mean[1]; pz = p[2] - cx_mean[2]
        q = rest_offsets[start + j]
        qx = q[0] - qc_mean[0]; qy = q[1] - qc_mean[1]; qz = q[2] - qc_mean[2]
        a00 = a00 + w * px * qx; a01 = a01 + w * px * qy; a02 = a02 + w * px * qz
        a10 = a10 + w * py * qx; a11 = a11 + w * py * qy; a12 = a12 + w * py * qz
        a20 = a20 + w * pz * qx; a21 = a21 + w * pz * qy; a22 = a22 + w * pz * qz

    A_buf[cid] = wp.mat33(a00 + 1.0e-6, a01,           a02,
                          a10,          a11 + 1.0e-6,  a12,
                          a20,          a21,           a22 + 1.0e-6)


# Loss: trace((R - I)^T (R - I)) = ||R-I||_F^2 — depends on rotation magnitude.
@wp.kernel
def loss_R_minus_I_squared(
    R_buf: wp.array(dtype=wp.mat33),
    loss_buf: wp.array(dtype=wp.float32),
):
    cid = wp.tid()
    R = R_buf[cid]
    s = float(0.0)
    s = s + (R[0,0] - 1.0)*(R[0,0] - 1.0) + R[0,1]*R[0,1] + R[0,2]*R[0,2]
    s = s + R[1,0]*R[1,0] + (R[1,1] - 1.0)*(R[1,1] - 1.0) + R[1,2]*R[1,2]
    s = s + R[2,0]*R[2,0] + R[2,1]*R[2,1] + (R[2,2] - 1.0)*(R[2,2] - 1.0)
    wp.atomic_add(loss_buf, 0, s)


# ============================================================================
# Full forward pipeline: damage cycles → cluster solve → loss
# ============================================================================
def build_test_system(N=8, seed=42):
    """1D chain with slight perturbation, knife at center."""
    np.random.seed(seed)
    rest_np = np.zeros((N, 3), dtype=np.float32)
    rest_np[:, 0] = np.linspace(0.3, 0.7, N)
    rest_np[:, 1] = 0.5
    rest_np[:, 2] = 0.5

    # Perturb current positions so R ≠ I
    current_np = rest_np.copy()
    theta = np.deg2rad(8.0)
    R_gt = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1],
    ], dtype=np.float32)
    # Rotate around centroid
    cm = current_np.mean(axis=0)
    current_np = (R_gt @ (current_np - cm).T).T + cm
    current_np += 0.005 * np.random.randn(*current_np.shape).astype(np.float32)

    E = N - 1
    return dict(
        rest=rest_np, current=current_np, N=N, E=E,
        edge_a=np.arange(E, dtype=np.int32),
        edge_b=np.arange(1, N, dtype=np.int32),
        pf_rest=np.full(E, 0.5, dtype=np.float32),
        knife_pos=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
    )


def make_params(sharpness, alpha_min, threshold=0.5, knife_radius=0.1,
                requires_grad=True):
    return dict(
        sharpness    = wp.array([sharpness],    dtype=wp.float32, requires_grad=requires_grad),
        alpha_min    = wp.array([alpha_min],    dtype=wp.float32, requires_grad=requires_grad),
        threshold    = wp.array([threshold],    dtype=wp.float32, requires_grad=requires_grad),
        knife_radius = wp.array([knife_radius], dtype=wp.float32, requires_grad=requires_grad),
    )


def run_forward(sys_dict, params, cov_kernel, n_cycles=2):
    """Full pipeline: damage cycles → cluster solve → loss."""
    N = sys_dict["N"]; E = sys_dict["E"]

    # Cluster: single cluster containing all N particles
    rest_centered = sys_dict["rest"] - sys_dict["rest"].mean(axis=0)
    pred_pos = wp.array(sys_dict["current"], dtype=wp.vec3)
    edge_a   = wp.array(sys_dict["edge_a"], dtype=wp.int32)
    edge_b   = wp.array(sys_dict["edge_b"], dtype=wp.int32)
    pf_rest  = wp.array(sys_dict["pf_rest"], dtype=wp.float32)
    knife_pos = wp.array(sys_dict["knife_pos"], dtype=wp.vec3)
    fixed     = wp.array(np.zeros(N, dtype=np.int32), dtype=wp.int32)

    cluster_starts = wp.array([0],         dtype=wp.int32)
    cluster_sizes  = wp.array([N],         dtype=wp.int32)
    particle_ids   = wp.array(np.arange(N, dtype=np.int32), dtype=wp.int32)
    rest_offsets   = wp.array(rest_centered, dtype=wp.vec3)
    cluster_quat   = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat)

    # Cycle-scoped buffers
    alpha_buf = [wp.array(np.ones(N, dtype=np.float32), dtype=wp.float32, requires_grad=True)
                 for _ in range(n_cycles + 1)]
    edge_dmg  = [wp.zeros(E, dtype=wp.float32, requires_grad=True) for _ in range(n_cycles + 1)]
    part_dmg  = [wp.zeros(N, dtype=wp.float32, requires_grad=True) for _ in range(n_cycles + 1)]

    for c in range(n_cycles):
        ed_delta = wp.zeros(E, dtype=wp.float32, requires_grad=True)
        wp.launch(update_edge_damage_kernel_ssa_param_arr, dim=E, inputs=[
            pred_pos, alpha_buf[c], edge_a, edge_b, pf_rest, edge_dmg[c],
            knife_pos, params["knife_radius"], params["alpha_min"], ed_delta,
        ])
        wp.launch(apply_edge_damage_ssa_kernel, dim=E, inputs=[
            edge_dmg[c], ed_delta, edge_dmg[c+1],
        ])
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
        wp.launch(update_soft_removal_kernel_ssa_param_arr, dim=N, inputs=[
            part_dmg[c+1], alpha_buf[c], fixed, params["threshold"],
            params["sharpness"], alpha_buf[c+1],
        ])

    # Cluster solve with the chosen covariance kernel
    cx_sum = wp.zeros(1, dtype=wp.vec3,    requires_grad=True)
    qc_sum = wp.zeros(1, dtype=wp.vec3,    requires_grad=True)
    tw     = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    cx_mean = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
    qc_mean = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
    A_buf   = wp.zeros(1, dtype=wp.mat33, requires_grad=True)
    R_buf   = wp.zeros(1, dtype=wp.mat33, requires_grad=True)
    loss_buf = wp.zeros(1, dtype=wp.float32, requires_grad=True)

    final_alpha = alpha_buf[-1]
    wp.launch(cluster_compute_sums_kernel_ssa, dim=1, inputs=[
        pred_pos, final_alpha, cluster_starts, cluster_sizes,
        particle_ids, rest_offsets, cx_sum, qc_sum, tw,
    ])
    wp.launch(cluster_normalize_centroids_kernel_ssa, dim=1, inputs=[
        cluster_sizes, cx_sum, qc_sum, tw, cx_mean, qc_mean,
    ])
    wp.launch(cov_kernel, dim=1, inputs=[
        pred_pos, final_alpha, cluster_starts, cluster_sizes,
        particle_ids, rest_offsets, cx_mean, qc_mean, A_buf,
    ])
    wp.launch(cluster_polar_rotation_kernel_ssa, dim=1, inputs=[
        cluster_sizes, A_buf, cluster_quat, 12, R_buf,
    ])
    wp.launch(loss_R_minus_I_squared, dim=1, inputs=[R_buf, loss_buf])

    return loss_buf, final_alpha


# ============================================================================
# FD parity test for one (sys_dict, params, cov_kernel) configuration
# ============================================================================
def parity_test(sys_dict, base_params, cov_kernel, label, fd_eps=1e-3):
    """Compute autograd and FD gradients of loss wrt (sharpness, alpha_min)."""
    p = base_params

    # Autograd
    tape = wp.Tape()
    with tape:
        loss_buf, final_alpha = run_forward(sys_dict, p, cov_kernel)
    tape.backward(loss=loss_buf)
    ag_sharp = float(p["sharpness"].grad.numpy()[0])
    ag_amin  = float(p["alpha_min"].grad.numpy()[0])
    loss_val = float(loss_buf.numpy()[0])
    alpha_final = final_alpha.numpy()

    # FD: sharpness
    p_plus  = make_params(sharpness=p["sharpness"].numpy()[0] + fd_eps,
                          alpha_min=p["alpha_min"].numpy()[0],
                          requires_grad=False)
    p_minus = make_params(sharpness=p["sharpness"].numpy()[0] - fd_eps,
                          alpha_min=p["alpha_min"].numpy()[0],
                          requires_grad=False)
    L_plus,  _ = run_forward(sys_dict, p_plus,  cov_kernel)
    L_minus, _ = run_forward(sys_dict, p_minus, cov_kernel)
    fd_sharp = (float(L_plus.numpy()[0]) - float(L_minus.numpy()[0])) / (2 * fd_eps)

    # FD: alpha_min
    p_plus  = make_params(sharpness=p["sharpness"].numpy()[0],
                          alpha_min=p["alpha_min"].numpy()[0] + fd_eps,
                          requires_grad=False)
    p_minus = make_params(sharpness=p["sharpness"].numpy()[0],
                          alpha_min=p["alpha_min"].numpy()[0] - fd_eps,
                          requires_grad=False)
    L_plus,  _ = run_forward(sys_dict, p_plus,  cov_kernel)
    L_minus, _ = run_forward(sys_dict, p_minus, cov_kernel)
    fd_amin = (float(L_plus.numpy()[0]) - float(L_minus.numpy()[0])) / (2 * fd_eps)

    # rel_err
    def rel(a, b):
        denom = max(abs(a), abs(b), 1e-12)
        return 100.0 * abs(a - b) / denom

    print(f"\n  {label}")
    print(f"  ──────────────────────────────────────────────────────────")
    print(f"  loss value              = {loss_val:.6f}")
    print(f"  α final distribution    = [{alpha_final.min():.4f}, {alpha_final.max():.4f}], "
          f"<0.01: {(alpha_final<0.01).sum()}/{len(alpha_final)}")
    print(f"  ∂loss/∂sharpness    autograd = {ag_sharp:+.6e}")
    print(f"                      FD       = {fd_sharp:+.6e}")
    print(f"                      rel_err  = {rel(ag_sharp, fd_sharp):.3f} %  "
          f"{'✅' if rel(ag_sharp, fd_sharp) < 5 else '⚠️ ' if rel(ag_sharp, fd_sharp) < 20 else '❌'}")
    print(f"  ∂loss/∂alpha_min    autograd = {ag_amin:+.6e}")
    print(f"                      FD       = {fd_amin:+.6e}")
    print(f"                      rel_err  = {rel(ag_amin, fd_amin):.3f} %  "
          f"{'✅' if rel(ag_amin, fd_amin) < 5 else '⚠️ ' if rel(ag_amin, fd_amin) < 20 else '❌'}")

    return dict(
        ag_sharp=ag_sharp, fd_sharp=fd_sharp,
        ag_amin=ag_amin,  fd_amin=fd_amin,
        rel_sharp=rel(ag_sharp, fd_sharp), rel_amin=rel(ag_amin, fd_amin),
        alpha_final=alpha_final,
    )


# ============================================================================
# Main: run all 3 variants on identical setup
# ============================================================================
def main():
    print("=" * 72)
    print(" DRY-RUN: M3 covariance kernel switch — FD parity comparison")
    print("=" * 72)

    sys_dict = build_test_system(N=8, seed=42)

    # Use gentle params so final α has values spread across [0, 1]
    # (mirrors realistic phantom scenario where many α are intermediate)
    print("\n Setup:")
    print(f"   N=8 particles, 1 cluster (size 8), n_cycles=2")
    print(f"   sharpness=5.0, alpha_min=2.0, threshold=0.5")
    print(f"   Loss = ||R - I||_F²  (rotation magnitude proxy)")
    print(f"   FD epsilon = 1e-3")
    print()
    print(" Variants:")
    print("   V1: BINARY mask (PRODUCTION code)")
    print("   V2: ALPHA + clamp [where(α<0.01, 0, α)]")
    print("   V3: ALPHA no clamp [w = α directly] — paper formula")

    # V1 binary
    params = make_params(sharpness=5.0, alpha_min=2.0)
    r1 = parity_test(sys_dict, params, cluster_compute_covariance_kernel_ssa,
                     "V1  BINARY  (production, kernels.py:677)")

    # V2 alpha + clamp
    params = make_params(sharpness=5.0, alpha_min=2.0)
    r2 = parity_test(sys_dict, params, cov_kernel_V2_alpha_with_clamp,
                     "V2  ALPHA + clamp (where(α<0.01, 0, α))")

    # V3 alpha no clamp (paper formula)
    params = make_params(sharpness=5.0, alpha_min=2.0)
    r3 = parity_test(sys_dict, params, cov_kernel_V3_alpha_no_clamp,
                     "V3  ALPHA no clamp (paper Eq.~\\ref{eq:covariance})")

    # Summary table
    print("\n" + "=" * 72)
    print(" SUMMARY")
    print("=" * 72)
    print(f"\n  Variant                         rel_err(sharpness)  rel_err(alpha_min)")
    print(f"  ─────────────────────────────  ──────────────────  ──────────────────")
    print(f"  V1  BINARY (production)        {r1['rel_sharp']:>8.3f} %        {r1['rel_amin']:>8.3f} %")
    print(f"  V2  ALPHA + clamp              {r2['rel_sharp']:>8.3f} %        {r2['rel_amin']:>8.3f} %")
    print(f"  V3  ALPHA no clamp (paper)     {r3['rel_sharp']:>8.3f} %        {r3['rel_amin']:>8.3f} %")
    print()
    print(f"  Gradient magnitude comparison (autograd):")
    print(f"    V1 ∂loss/∂sharpness = {r1['ag_sharp']:+.4e}    ∂loss/∂alpha_min = {r1['ag_amin']:+.4e}")
    print(f"    V2 ∂loss/∂sharpness = {r2['ag_sharp']:+.4e}    ∂loss/∂alpha_min = {r2['ag_amin']:+.4e}")
    print(f"    V3 ∂loss/∂sharpness = {r3['ag_sharp']:+.4e}    ∂loss/∂alpha_min = {r3['ag_amin']:+.4e}")
    print()

    # Decision support
    print(f"  α distribution after damage pipeline:")
    print(f"    V1: {r1['alpha_final']}")
    print(f"    V2: {r2['alpha_final']}")
    print(f"    V3: {r3['alpha_final']}")
    print()

    print(f"  INTERPRETATION:")
    if max(r2['rel_sharp'], r2['rel_amin'], r3['rel_sharp'], r3['rel_amin']) < 5:
        print(f"    ✅ All variants pass < 5% FD parity.")
        print(f"       Switching production from V1 to V2/V3 is SAFE w.r.t. autograd.")
    elif max(r2['rel_sharp'], r2['rel_amin'], r3['rel_sharp'], r3['rel_amin']) < 20:
        print(f"    ⚠️  V2/V3 rel_err > 5% but < 20%.")
        print(f"       Switch likely OK but verify on larger setup before committing.")
    else:
        print(f"    ❌ V2/V3 rel_err > 20%.")
        print(f"       Switch may need additional regularization or fd_eps tuning.")

    # Gradient direction comparison
    if r1['ag_sharp'] * r3['ag_sharp'] < 0:
        print(f"    ⚠️  V1 vs V3 sharpness gradients have OPPOSITE SIGN — switching")
        print(f"       will push Adam in different direction from current production.")
    else:
        print(f"    ✅ V1 vs V3 sharpness gradients have same sign — switching")
        print(f"       preserves optimization direction (magnitude may differ).")


if __name__ == "__main__":
    main()
