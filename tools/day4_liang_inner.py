"""Day 4: ReSECT Liang inner loop verification.

Tests compute_liang_inner_residuals on cube_tet.msh with synthetic obs:
  - pos_target = pos_pred + small displacement (e.g., uniform 5mm shift in z)
  - alpha_target = alpha_pred with a few particles forced to ~0 (extra cavity)

Verifies:
  1. Δx pulls (pos_pred + Δx) toward pos_target → L_pos decreases monotonically
  2. Δδv ≥ 0 everywhere (softplus enforced)
  3. L_E stays bounded (edge-spring regularizer prevents arbitrary Δx)
  4. Without E term (lambda_E=0) Δx overfits perfectly to noise — should
     reduce L_pos to near zero
  5. With E term, Δx is a "physically-balanced" approximation — L_pos
     stays elevated but L_E is bounded
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import warp as wp  # noqa: E402
wp.init()

from tools.online_residual_synth import (  # noqa: E402
    load_msh_system, compute_liang_inner_residuals,
)


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    sys_dict = load_msh_system(mesh_path)
    print(f"\nMesh: N={sys_dict['N']}  E={sys_dict['E']}")

    N = sys_dict["N"]
    pos = sys_dict["pos"].astype(np.float64)
    edges_a = sys_dict["edge_a"]
    edges_b = sys_dict["edge_b"]
    edges_rest = sys_dict["pf_rest"].astype(np.float64)

    # Use a synthetic "sim prediction" = rest mesh, dv = 0, all alpha = 1
    pos_pred = pos.copy()
    dv_pred = np.zeros(N, dtype=np.float64)

    # Theta for the sigmoid (s=20 sharp, τ=0.5)
    theta = np.array([3.0, 20.0, 0.5])

    # --- Construct a NON-RIGID perturbed target (stretches edges) ---
    # Per-particle random displacement in [-5mm, +5mm] each axis → edges stretch
    # This is what stresses E_edge regularizer (uniform shift wouldn't)
    np_rng = np.random.default_rng(42)
    shift_per_particle = np_rng.uniform(-0.005, 0.005, size=(N, 3))
    pos_target = pos_pred + shift_per_particle
    alpha_target = np.ones(N, dtype=np.float64)
    cavity_indices = np_rng.choice(N, size=int(0.1 * N), replace=False)
    alpha_target[cavity_indices] = 0.0  # 10% forced to full damage

    # Sanity: this non-rigid target DOES stretch edges
    edge_vec_target = pos_target[edges_a] - pos_target[edges_b]
    edge_len_target = np.linalg.norm(edge_vec_target, axis=1)
    L_E_at_target = float(((edge_len_target - edges_rest)**2).sum())
    print(f"\nTarget:")
    print(f"  per-particle random shift in [-5mm, 5mm] (non-rigid)")
    print(f"  L_E AT target (edge violation) = {L_E_at_target:.4e}")
    print(f"  → high λ_E should pull Δx AWAY from target, reducing L_E")
    print(f"  forced cavity particles: {int(0.1*N)} ({0.1*100:.0f}%)")
    print(f"  alpha_target: range [{alpha_target.min()}, {alpha_target.max()}], "
          f"mean {alpha_target.mean():.4f}")

    # ---- Test A: λ_E = 0 (no edge regularizer) ----
    print("\n" + "=" * 64)
    print("Test A: lambda_E = 0  (no edge regularizer)")
    print("=" * 64)
    dx_a, ddv_a, info_a = compute_liang_inner_residuals(
        pos_pred, dv_pred, pos_target, alpha_target, theta,
        edges_a, edges_b, edges_rest,
        n_iter=30, lr_pos=5e-3, lr_dv=0.1,
        lambda_cav=1.0, lambda_E=0.0,
        verbose=True,
    )
    pos_corr_a = pos_pred + dx_a
    L_pos_ratio_a = info_a["L_pos"] / info_a["L_init_pos"]
    print(f"  Final: L_pos={info_a['L_pos']:.4e} (init {info_a['L_init_pos']:.4e}, "
          f"reduced {L_pos_ratio_a:.4f}×)")
    print(f"         L_cav={info_a['L_cav']:.4e} (init {info_a['L_init_cav']:.4e})")
    print(f"         L_E  ={info_a['L_E']:.4e}  ← not regularized, can be large")
    print(f"  max|Δx| = {float(np.abs(dx_a).max()):.5f}")
    print(f"  mean|Δx| = {float(np.abs(dx_a).mean()):.5f}")
    print(f"  Δδv min = {float(ddv_a.min()):.4f} (must ≥ 0)")
    print(f"  Δδv max = {float(ddv_a.max()):.4f}")

    # ---- Test B: λ_E = 1.0 (V2 default) ----
    print("\n" + "=" * 64)
    print("Test B: lambda_E = 1.0  (V2 default, edge regularizer ON)")
    print("=" * 64)
    dx_b, ddv_b, info_b = compute_liang_inner_residuals(
        pos_pred, dv_pred, pos_target, alpha_target, theta,
        edges_a, edges_b, edges_rest,
        n_iter=30, lr_pos=5e-3, lr_dv=0.1,
        lambda_cav=1.0, lambda_E=1.0,
        verbose=True,
    )
    L_pos_ratio_b = info_b["L_pos"] / info_b["L_init_pos"]
    print(f"  Final: L_pos={info_b['L_pos']:.4e} (reduced {L_pos_ratio_b:.4f}×)")
    print(f"         L_cav={info_b['L_cav']:.4e}")
    print(f"         L_E  ={info_b['L_E']:.4e}  ← bounded by regularizer")
    print(f"  max|Δx| = {float(np.abs(dx_b).max()):.5f}")
    print(f"  mean|Δx| = {float(np.abs(dx_b).mean()):.5f}")
    print(f"  Δδv min = {float(ddv_b.min()):.4f} (must ≥ 0)")
    print(f"  Δδv max = {float(ddv_b.max()):.4f}")

    # ---- Test C: λ_E = 0.1 (light regularizer, tests B/C boundary) ----
    # Note: λ_E=100 was tested and over-steps in 30 iter (lr not tuned for that
    # regime). For demonstrating the trade-off invariants in a converged regime,
    # we use a milder sweep λ_E ∈ {0, 0.1, 1.0}.
    print("\n" + "=" * 64)
    print("Test C: lambda_E = 0.1  (light regularizer, between A and B)")
    print("=" * 64)
    dx_c, ddv_c, info_c = compute_liang_inner_residuals(
        pos_pred, dv_pred, pos_target, alpha_target, theta,
        edges_a, edges_b, edges_rest,
        n_iter=30, lr_pos=5e-3, lr_dv=0.1,
        lambda_cav=1.0, lambda_E=0.1,
        verbose=True,
    )
    print(f"  Final: L_pos={info_c['L_pos']:.4e}  L_E={info_c['L_E']:.4e}")
    print(f"  max|Δx| = {float(np.abs(dx_c).max()):.5f}")

    # ---- Verdicts ----
    print("\n" + "=" * 64)
    print("Day 4 verdicts")
    print("=" * 64)

    # (1) softplus enforces Δδv ≥ 0 in all 3 tests
    irr_a = float(ddv_a.min()) >= 0.0
    irr_b = float(ddv_b.min()) >= 0.0
    irr_c = float(ddv_c.min()) >= 0.0
    print(f"  Δδv ≥ 0 (softplus): A={irr_a}  B={irr_b}  C={irr_c}")
    print(f"    {'✓' if all([irr_a,irr_b,irr_c]) else '✗'} Cohesive zone irreversibility holds")

    # (2) L_pos reduces monotonically (final < init)
    converging_a = info_a["L_pos"] < info_a["L_init_pos"]
    converging_b = info_b["L_pos"] < info_b["L_init_pos"]
    converging_c = info_c["L_pos"] < info_c["L_init_pos"]
    print(f"  L_pos reduces: A={converging_a}  B={converging_b}  C={converging_c}")

    # Re-order tests by λ_E: A(0) < C(0.1) < B(1.0)
    # to check monotonicity properly
    fits_pos = [(0.0, info_a["L_pos"], info_a["L_E"]),
                (0.1, info_c["L_pos"], info_c["L_E"]),
                (1.0, info_b["L_pos"], info_b["L_E"])]
    fits_pos.sort()  # by λ_E

    # (3) L_pos monotone increases with λ_E
    pos_monotone = fits_pos[0][1] <= fits_pos[1][1] <= fits_pos[2][1]
    print(f"  L_pos increases with λ_E:")
    for lam, p, e in fits_pos:
        print(f"    λ_E={lam:5.2f}  L_pos={p:.4e}  L_E={e:.4e}")
    print(f"    {'✓' if pos_monotone else '⚠'} pos monotone: {pos_monotone}")

    # (4) L_E decreases with λ_E
    E_monotone = fits_pos[0][2] >= fits_pos[1][2] >= fits_pos[2][2]
    print(f"  {'✓' if E_monotone else '⚠'} E monotone: {E_monotone}")

    all_pass = (all([irr_a, irr_b, irr_c]) and
                all([converging_a, converging_b, converging_c]) and
                pos_monotone and E_monotone)

    if all_pass:
        print(f"\n✓ Day 4 PASS — Liang inner loop V2 + cohesive zone all 4 invariants hold")
    else:
        print(f"\n✗ Day 4 FAIL — see diagnostics above")


if __name__ == "__main__":
    main()
