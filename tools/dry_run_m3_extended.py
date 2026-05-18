#!/usr/bin/env python3
"""
DRY-RUN extended: M3 covariance kernel switch — TWO additional probes.

  --mode horizon    Long horizon (n_cycles up to 25). Let α cross 0.01.
                    Track per-cycle: α distribution, R rotation angle,
                    loss, gradient magnitude — V1 (binary) vs V3 (paper).
                    Shows where the two models diverge over time.

  --mode recovery   Full synth Adam recovery.
                    Generate target with V3 (= paper formula = "ground truth").
                    Init from wrong θ. Run Adam 100 iters with V1 covariance
                    and again with V3 covariance. Compare:
                      - convergence speed (iters to loss < threshold)
                      - final θ*
                      - bias of V1 estimate vs true θ

  --mode both       Run both, side by side.

This DOES NOT modify any production code. Uses inline cov kernel variants.

Run:
  python3 tools/dry_run_m3_extended.py --mode horizon
  python3 tools/dry_run_m3_extended.py --mode recovery
  python3 tools/dry_run_m3_extended.py --mode both
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
    cluster_compute_covariance_kernel_ssa,   # V1 BINARY (production)
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
# V3: alpha-weighted no-clamp covariance (paper formula) + Tikhonov
# ============================================================================
@wp.kernel
def cov_kernel_V3(
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
        w = alpha[pid]   # paper formula
        p = pred_pos[pid]
        px = p[0] - cx_mean[0]; py = p[1] - cx_mean[1]; pz = p[2] - cx_mean[2]
        q = rest_offsets[start + j]
        qx = q[0] - qc_mean[0]; qy = q[1] - qc_mean[1]; qz = q[2] - qc_mean[2]
        a00 = a00 + w*px*qx; a01 = a01 + w*px*qy; a02 = a02 + w*px*qz
        a10 = a10 + w*py*qx; a11 = a11 + w*py*qy; a12 = a12 + w*py*qz
        a20 = a20 + w*pz*qx; a21 = a21 + w*pz*qy; a22 = a22 + w*pz*qz
    # Tikhonov 1e-6
    A_buf[cid] = wp.mat33(a00 + 1.0e-6, a01,           a02,
                          a10,          a11 + 1.0e-6,  a12,
                          a20,          a21,           a22 + 1.0e-6)


@wp.kernel
def loss_R_minus_I_squared(R_buf: wp.array(dtype=wp.mat33),
                            loss_buf: wp.array(dtype=wp.float32)):
    cid = wp.tid()
    R = R_buf[cid]
    s = (R[0,0]-1.)*(R[0,0]-1.) + R[0,1]*R[0,1]       + R[0,2]*R[0,2]
    s = s + R[1,0]*R[1,0]       + (R[1,1]-1.)*(R[1,1]-1.) + R[1,2]*R[1,2]
    s = s + R[2,0]*R[2,0]       + R[2,1]*R[2,1]       + (R[2,2]-1.)*(R[2,2]-1.)
    wp.atomic_add(loss_buf, 0, s)


@wp.kernel
def alpha_mse_kernel(alpha_sim: wp.array(dtype=wp.float32),
                     alpha_target: wp.array(dtype=wp.float32),
                     loss_buf: wp.array(dtype=wp.float32)):
    i = wp.tid()
    d = alpha_sim[i] - alpha_target[i]
    wp.atomic_add(loss_buf, 0, d * d)


@wp.kernel
def R_mse_kernel(R_sim: wp.array(dtype=wp.mat33),
                 R_target: wp.array(dtype=wp.mat33),
                 loss_buf: wp.array(dtype=wp.float32),
                 weight: float):
    """Frobenius norm squared of (R_sim - R_target). Scaled by `weight` so the
    R-component is comparable in magnitude to the α MSE component."""
    cid = wp.tid()
    Rs = R_sim[cid]; Rt = R_target[cid]
    s = float(0.0)
    for i in range(3):
        for j in range(3):
            d = Rs[i, j] - Rt[i, j]
            s = s + d * d
    wp.atomic_add(loss_buf, 0, weight * s)


# ============================================================================
# Common forward pipeline
# ============================================================================
def build_system(N=8, seed=42):
    np.random.seed(seed)
    rest = np.zeros((N, 3), dtype=np.float32)
    rest[:, 0] = np.linspace(0.3, 0.7, N)
    rest[:, 1] = 0.5; rest[:, 2] = 0.5
    current = rest.copy()
    theta = np.deg2rad(8.0)
    R_gt = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ], dtype=np.float32)
    cm = current.mean(axis=0)
    current = (R_gt @ (current - cm).T).T + cm
    current += 0.005 * np.random.randn(*current.shape).astype(np.float32)
    E = N - 1
    return dict(
        N=N, E=E, rest=rest, current=current,
        edge_a=np.arange(E, dtype=np.int32),
        edge_b=np.arange(1, N, dtype=np.int32),
        pf_rest=np.full(E, 0.5, dtype=np.float32),
        knife_pos=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
    )


def make_params(sharpness, alpha_min, threshold=0.5, knife_radius=0.1,
                requires_grad=True):
    return dict(
        sharpness    = wp.array([float(sharpness)],    dtype=wp.float32, requires_grad=requires_grad),
        alpha_min    = wp.array([float(alpha_min)],    dtype=wp.float32, requires_grad=requires_grad),
        threshold    = wp.array([float(threshold)],    dtype=wp.float32, requires_grad=requires_grad),
        knife_radius = wp.array([float(knife_radius)], dtype=wp.float32, requires_grad=requires_grad),
    )


def run_pipeline(sys_dict, params, cov_kernel, n_cycles):
    """Full pipeline: damage cycles -> cluster solve. Returns (final_alpha, R)."""
    N, E = sys_dict["N"], sys_dict["E"]
    rest_centered = sys_dict["rest"] - sys_dict["rest"].mean(axis=0)
    pred_pos = wp.array(sys_dict["current"], dtype=wp.vec3)
    edge_a   = wp.array(sys_dict["edge_a"], dtype=wp.int32)
    edge_b   = wp.array(sys_dict["edge_b"], dtype=wp.int32)
    pf_rest  = wp.array(sys_dict["pf_rest"], dtype=wp.float32)
    knife_pos = wp.array(sys_dict["knife_pos"], dtype=wp.vec3)
    fixed     = wp.array(np.zeros(N, dtype=np.int32), dtype=wp.int32)

    cluster_starts = wp.array([0], dtype=wp.int32)
    cluster_sizes  = wp.array([N], dtype=wp.int32)
    particle_ids   = wp.array(np.arange(N, dtype=np.int32), dtype=wp.int32)
    rest_offsets   = wp.array(rest_centered, dtype=wp.vec3)
    cluster_quat   = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat)

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
        d_sum = wp.zeros(N, dtype=wp.float32, requires_grad=True)
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

    # cluster solve
    final_alpha = alpha_buf[-1]
    cx_sum = wp.zeros(1, dtype=wp.vec3,    requires_grad=True)
    qc_sum = wp.zeros(1, dtype=wp.vec3,    requires_grad=True)
    tw     = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    cx_mean = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
    qc_mean = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
    A_buf   = wp.zeros(1, dtype=wp.mat33, requires_grad=True)
    R_buf   = wp.zeros(1, dtype=wp.mat33, requires_grad=True)

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

    return final_alpha, R_buf


def rot_angle_deg(R1, R2):
    Rd = R1.T @ R2
    c = max(-1.0, min(1.0, (np.trace(Rd) - 1.0) / 2.0))
    return float(np.degrees(np.arccos(c)))


# ============================================================================
# MODE: horizon — track V1 vs V3 over many cycles
# ============================================================================
def mode_horizon():
    print("=" * 72)
    print(" MODE: HORIZON — V1 (binary) vs V3 (paper) over many cycles")
    print("=" * 72)
    # gentle enough that α can drop below 0.01 over time
    sys_dict = build_system(N=8, seed=42)
    sharp_true, amin_true = 10.0, 3.0
    threshold = 0.5

    cycles_to_try = [1, 2, 3, 5, 8, 12, 18, 25]

    print(f"\n Setup: N=8, sharpness={sharp_true}, alpha_min={amin_true}, threshold={threshold}")
    print(f" Tracking: α_min distribution, R(V1) vs R(V3) angular divergence")
    print()
    print(f"  cycles  α_min_V1   α_min_V3   #α<0.01 V1   #α<0.01 V3   R angle Δ")
    print(f"  ──────  ─────────  ─────────  ──────────   ──────────   ──────────")

    for nc in cycles_to_try:
        # V1
        params1 = make_params(sharp_true, amin_true, threshold, requires_grad=False)
        alpha1, R1 = run_pipeline(sys_dict, params1, cluster_compute_covariance_kernel_ssa, nc)
        a1 = alpha1.numpy(); R1_np = R1.numpy()[0]
        # V3
        params3 = make_params(sharp_true, amin_true, threshold, requires_grad=False)
        alpha3, R3 = run_pipeline(sys_dict, params3, cov_kernel_V3, nc)
        a3 = alpha3.numpy(); R3_np = R3.numpy()[0]

        ang = rot_angle_deg(R1_np, R3_np)
        print(f"  {nc:>4d}    "
              f"{a1.min():>8.4f}   {a3.min():>8.4f}   "
              f"{(a1 < 0.01).sum():>4d}/{len(a1):<2d}      "
              f"{(a3 < 0.01).sum():>4d}/{len(a3):<2d}      "
              f"{ang:>6.3f}°")

    print()
    print(" Observations:")
    print("  - 'α_min' = lowest α value among 8 particles after nc cycles")
    print("  - '#α<0.01' = count of particles whose α fell below the binary threshold")
    print("  - 'R angle Δ' = geodesic distance between R from V1 vs V3 (in degrees)")
    print("    Large Δ → the two kernels give materially different rotations")


# ============================================================================
# MODE: recovery — full Adam sys-ID with V1 vs V3
# ============================================================================
def mode_recovery():
    print("=" * 72)
    print(" MODE: RECOVERY — Adam sys-ID with V1 vs V3")
    print("=" * 72)

    sys_dict = build_system(N=8, seed=42)

    # Treat V3 (paper formula) as ground truth physics.
    # Gentler params + fewer cycles → α stays in intermediate range, so
    # covariance kernel choice actually matters.
    TRUE_SHARP = 4.0
    TRUE_AMIN  = 3.0
    INIT_SHARP = 2.0     # 50% wrong
    INIT_AMIN  = 5.0     # 67% wrong
    THRESHOLD  = 0.5
    N_CYCLES   = 2
    R_WEIGHT   = 10.0    # so R-loss is comparable magnitude to α-MSE

    print(f"\n Setup: N=8, n_cycles={N_CYCLES}, threshold={THRESHOLD}")
    print(f"  True θ      : sharpness={TRUE_SHARP}, alpha_min={TRUE_AMIN}")
    print(f"  Init θ      : sharpness={INIT_SHARP}, alpha_min={INIT_AMIN}")
    print(f"  Ground truth physics: V3 (paper formula)")
    print()

    # Generate target with V3
    target_params = make_params(TRUE_SHARP, TRUE_AMIN, THRESHOLD, requires_grad=False)
    target_alpha_arr, target_R_arr = run_pipeline(
        sys_dict, target_params, cov_kernel_V3, N_CYCLES)
    target_alpha = target_alpha_arr.numpy().copy()
    target_R = target_R_arr.numpy()[0].copy()
    print(f"  Target α distribution: [{target_alpha.min():.4f}, {target_alpha.max():.4f}], "
          f"#<0.01: {(target_alpha < 0.01).sum()}/{len(target_alpha)}, "
          f"#in (0.01,0.99): {((target_alpha > 0.01) & (target_alpha < 0.99)).sum()}/{len(target_alpha)}")
    print(f"  Target R (Frobenius diff from I): "
          f"{np.linalg.norm(target_R - np.eye(3)):.4f}")
    print(f"  Loss = α_MSE + {R_WEIGHT}·||R - R_target||²")
    print()

    # Adam params
    LR_SHARP = 0.05
    LR_AMIN  = 0.05
    N_ITERS  = 80
    BETA1, BETA2, EPS = 0.9, 0.999, 1e-8

    def run_adam(cov_kernel, label):
        sharp = INIT_SHARP
        amin = INIT_AMIN
        m_sharp = v_sharp = 0.0
        m_amin = v_amin = 0.0
        loss_history = []
        sharp_history = []
        amin_history = []

        target_alpha_wp = wp.array(target_alpha, dtype=wp.float32)
        target_R_wp = wp.array([target_R], dtype=wp.mat33)

        for it in range(1, N_ITERS + 1):
            params = make_params(sharp, amin, THRESHOLD, requires_grad=True)
            tape = wp.Tape()
            with tape:
                alpha_final, R_buf = run_pipeline(sys_dict, params, cov_kernel, N_CYCLES)
                loss_buf = wp.zeros(1, dtype=wp.float32, requires_grad=True)
                wp.launch(alpha_mse_kernel, dim=sys_dict["N"], inputs=[
                    alpha_final, target_alpha_wp, loss_buf,
                ])
                wp.launch(R_mse_kernel, dim=1, inputs=[
                    R_buf, target_R_wp, loss_buf, R_WEIGHT,
                ])
            tape.backward(loss=loss_buf)
            g_sharp = float(params["sharpness"].grad.numpy()[0])
            g_amin  = float(params["alpha_min"].grad.numpy()[0])
            L = float(loss_buf.numpy()[0])

            # Adam step
            m_sharp = BETA1*m_sharp + (1-BETA1)*g_sharp
            v_sharp = BETA2*v_sharp + (1-BETA2)*g_sharp*g_sharp
            mh = m_sharp / (1 - BETA1**it); vh = v_sharp / (1 - BETA2**it)
            sharp -= LR_SHARP * mh / (np.sqrt(vh) + EPS)
            sharp = max(0.1, min(50.0, sharp))

            m_amin = BETA1*m_amin + (1-BETA1)*g_amin
            v_amin = BETA2*v_amin + (1-BETA2)*g_amin*g_amin
            mh = m_amin / (1 - BETA1**it); vh = v_amin / (1 - BETA2**it)
            amin -= LR_AMIN * mh / (np.sqrt(vh) + EPS)
            amin = max(0.1, min(20.0, amin))

            loss_history.append(L)
            sharp_history.append(sharp)
            amin_history.append(amin)

            if it % 10 == 0 or it in (1, 5):
                print(f"  [{label}] iter {it:3d}  loss={L:.6e}  "
                      f"sharp={sharp:.4f}  amin={amin:.4f}  "
                      f"(g_sharp={g_sharp:+.2e}, g_amin={g_amin:+.2e})")

        return dict(loss=loss_history, sharp=sharp_history, amin=amin_history,
                    final_sharp=sharp, final_amin=amin, final_loss=loss_history[-1])

    print(" === V3 sys-ID (Adam on paper formula, target also V3) ===")
    r3 = run_adam(cov_kernel_V3, "V3")
    print()
    print(" === V1 sys-ID (Adam on binary formula, target is V3) ===")
    r1 = run_adam(cluster_compute_covariance_kernel_ssa, "V1")

    print()
    print("=" * 72)
    print(" RECOVERY SUMMARY")
    print("=" * 72)
    print(f"\n  Variant     final_loss     sharp*    sharp_bias   amin*    amin_bias")
    print(f"  ──────      ──────────     ──────    ──────────   ─────    ─────────")
    print(f"  TRUE                          {TRUE_SHARP:.4f}    (0%)         {TRUE_AMIN:.4f}    (0%)")
    print(f"  V3 recover  {r3['final_loss']:.4e}    {r3['final_sharp']:.4f}    "
          f"{100*(r3['final_sharp']-TRUE_SHARP)/TRUE_SHARP:+.2f}%      "
          f"{r3['final_amin']:.4f}    {100*(r3['final_amin']-TRUE_AMIN)/TRUE_AMIN:+.2f}%")
    print(f"  V1 recover  {r1['final_loss']:.4e}    {r1['final_sharp']:.4f}    "
          f"{100*(r1['final_sharp']-TRUE_SHARP)/TRUE_SHARP:+.2f}%      "
          f"{r1['final_amin']:.4f}    {100*(r1['final_amin']-TRUE_AMIN)/TRUE_AMIN:+.2f}%")
    print()

    # Convergence speed
    def iters_to_threshold(loss_hist, thr):
        for i, L in enumerate(loss_hist):
            if L < thr:
                return i + 1
        return None

    final_target_loss = r3['final_loss']
    threshold_loss = final_target_loss * 10  # 10× of best
    it_v3 = iters_to_threshold(r3['loss'], threshold_loss)
    it_v1 = iters_to_threshold(r1['loss'], threshold_loss)
    print(f"  Convergence speed (iters to reach loss < {threshold_loss:.2e}):")
    print(f"    V3: {it_v3 if it_v3 else f'> {N_ITERS}'} iters")
    print(f"    V1: {it_v1 if it_v1 else f'> {N_ITERS}'} iters")
    print()

    # Bias analysis
    sharp_bias_v1 = abs(r1['final_sharp'] - TRUE_SHARP) / TRUE_SHARP
    amin_bias_v1  = abs(r1['final_amin']  - TRUE_AMIN)  / TRUE_AMIN
    sharp_bias_v3 = abs(r3['final_sharp'] - TRUE_SHARP) / TRUE_SHARP
    amin_bias_v3  = abs(r3['final_amin']  - TRUE_AMIN)  / TRUE_AMIN

    print(f"  Final parameter bias (|θ* - θ_true| / θ_true):")
    print(f"    V3: sharp bias {100*sharp_bias_v3:.2f}%, amin bias {100*amin_bias_v3:.2f}%")
    print(f"    V1: sharp bias {100*sharp_bias_v1:.2f}%, amin bias {100*amin_bias_v1:.2f}%")
    print()
    if max(sharp_bias_v1, amin_bias_v1) > 2 * max(sharp_bias_v3, amin_bias_v3, 0.01):
        print(f"  → V1 (binary kernel) gives a SIGNIFICANTLY BIASED estimate of true θ")
        print(f"    when the true physics is V3 (paper formula).")
        print(f"  → If real phantom physics is closer to V3, V1 calibration will be")
        print(f"    systematically wrong by ~{100*max(sharp_bias_v1, amin_bias_v1):.0f}%.")
    else:
        print(f"  → V1 and V3 give comparable bias on this test.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["horizon", "recovery", "both"], default="both")
    args = ap.parse_args()

    if args.mode in ("horizon", "both"):
        mode_horizon()
        print()
    if args.mode in ("recovery", "both"):
        mode_recovery()


if __name__ == "__main__":
    main()
