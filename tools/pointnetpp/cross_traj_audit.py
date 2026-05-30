"""Rigorous audit of the cross-trajectory test.

Before declaring hierarchical wins, verify these CRITICAL sanity checks:

  AUDIT 1: Truth-params self-test
      For each trajectory T_X, predict with TRUTH params on T_X. The result
      should match truth obs exactly (the truth obs was generated with same
      params). Non-zero error → state contamination between forward calls.

  AUDIT 2: Order independence
      Swap baseline / hierarchical prediction order. If results change,
      there's solver state contamination across predict_on_trajectory calls.

  AUDIT 3: Trajectories actually distinct
      Print last-frame positions of each truth obs. T_A, T_B, T_C, T_D should
      produce visibly DIFFERENT deformation patterns. If they all look the
      same, "cross-trajectory" claim is moot.

  AUDIT 4: Calibration outcome detail
      Print stiff/tau snapshots from baseline vs hierarchical. Are they
      actually different? Hierarchical winning could mean:
        (a) found a different (better) basin
        (b) regularized — smaller L_gap_best
        (c) effectively similar params

  AUDIT 5: L_gap_best comparison
      If hier has lower L_gap on calibration → it just fits T_A better, not
      a generalization win. If similar L_gap → win is from regularization.

  AUDIT 6: Predict-on-T_A self-consistency
      Predict with calibrated params on T_A. Compare to truth obs of T_A
      (which was used for calibration target). Should be small error.
"""

import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))

import warp as wp  # noqa: E402
wp.init()
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402

from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff, compute_metrics,
)
from tools.pointnetpp.cross_trajectory_test import (  # noqa: E402
    make_traj, predict_on_trajectory, chamfer_per_traj,
)


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    print(f"\n=== Cross-trajectory AUDIT ===")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = 2

    device = "cuda:0"
    knife_radius = 0.08
    T = 10

    theta_truth = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    pos_rest = mesh_info["vertices"]
    stiff_truth, inclusion_mask, center, incl_r = build_inclusion_truth(
        pos_rest, stiff_in=0.8, stiff_out=0.3,
        center_override=(0.15, 0.25, 0.15),
        radius_override=0.06)
    tau_truth = np.where(inclusion_mask, 0.20, 0.40).astype(np.float32)

    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_radius=knife_radius,
        theta_3d_np=theta_truth,
        pos_rest=pos_rest,
    )

    # Generate truth obs for all 4 trajectories
    truth = {}
    for name in ["A", "B", "C", "D"]:
        traj = make_traj(name, T=T)
        obs_pos, obs_alpha = gen_obs_per_particle_stiff(
            solver, rest_state, theta_truth, stiff_truth, traj, knife_radius, device,
            tau_truth_np=tau_truth)
        truth[name] = dict(traj=traj, pos=obs_pos, alpha=obs_alpha)
    print("Truth obs generated for T_A, T_B, T_C, T_D")

    # ===== AUDIT 1: truth-params self-test =====
    print(f"\n{'='*72}\nAUDIT 1: predict(truth_params) vs truth_obs (should be exact)\n{'='*72}")
    audit1 = {}
    for name in ["A", "B", "C", "D"]:
        traj = truth[name]["traj"]
        pred_pos, pred_alpha = predict_on_trajectory(stiff_truth, tau_truth, traj, ctx)
        pos_err, alpha_err = chamfer_per_traj(pred_pos, truth[name]["pos"],
                                                  pred_alpha, truth[name]["alpha"])
        audit1[name] = (pos_err, alpha_err)
        marker = "✓" if pos_err < 1e-8 else "⚠"
        print(f"  {marker} T_{name}: pos_err = {pos_err:.3e}   α_err = {alpha_err:.3e}")
    if all(audit1[n][0] < 1e-7 for n in audit1):
        print("  → AUDIT 1 PASS: solver state hygiene OK")
    else:
        print("  → AUDIT 1 FAIL: prediction with truth params is NOT exact.")
        print("                   → state contamination or solver-config drift")
        print("                   → all subsequent results SUSPECT")

    # ===== AUDIT 2: order independence =====
    print(f"\n{'='*72}\nAUDIT 2: order independence (predict order matters?)\n{'='*72}")
    # Use truth params for two consecutive predicts on same trajectory.
    # If second predict differs from first, state contamination exists.
    for name in ["A", "B"]:
        traj = truth[name]["traj"]
        p1, _ = predict_on_trajectory(stiff_truth, tau_truth, traj, ctx)
        p2, _ = predict_on_trajectory(stiff_truth, tau_truth, traj, ctx)
        diff = np.max([np.max(np.abs(p1[t] - p2[t])) for t in range(T)])
        marker = "✓" if diff < 1e-8 else "⚠"
        print(f"  {marker} T_{name}: max |predict_1 - predict_2| = {diff:.3e}")

    # ===== AUDIT 3: trajectories actually distinct =====
    print(f"\n{'='*72}\nAUDIT 3: trajectories distinct in obs space?\n{'='*72}")
    # Compare truth obs across trajectories. T_A vs T_B should be very different.
    obs_A_final = truth["A"]["pos"][-1]
    for name in ["B", "C", "D"]:
        obs_X_final = truth[name]["pos"][-1]
        # Per-particle displacement A vs X
        diff = np.linalg.norm(obs_A_final - obs_X_final, axis=1)
        cavity_A = (truth["A"]["alpha"][-1] < 0.5).mean()
        cavity_X = (truth[name]["alpha"][-1] < 0.5).mean()
        cavity_diff = abs(cavity_A - cavity_X)
        print(f"  T_A vs T_{name}: mean per-particle displacement diff = {diff.mean():.6f} m, "
              f"max = {diff.max():.6f} m  cavity A={cavity_A:.4f} X={cavity_X:.4f}  "
              f"Δcav={cavity_diff:.4f}")
    print("  (If displacement diff << 1e-4 m, trajectories produce ~same deformation)")
    print("  (If cavity fractions equal, trajectories cut SAME region — generalization is moot)")

    # ===== AUDIT 4 + 5: load cross-traj results, compare calibration outcomes =====
    print(f"\n{'='*72}\nAUDIT 4 + 5: calibration outcomes from cross_traj_test\n{'='*72}")
    npz_path = REPO / "tools" / "pointnetpp" / "cross_trajectory_results.npz"
    if not npz_path.exists():
        print(f"  ✗ cross_trajectory_results.npz not found — run cross_trajectory_test.py first")
        return
    data = np.load(npz_path, allow_pickle=True)
    stiff_base = data["stiff_base"]
    stiff_hier = data["stiff_hier"]
    tau_base = data["tau_base"]
    tau_hier = data["tau_hier"]
    print(f"  Calibrated stiff stats:")
    print(f"    Baseline    : mean={stiff_base.mean():.4f}  std={stiff_base.std():.4f}  "
          f"in={stiff_base[inclusion_mask].mean():.4f}  out={stiff_base[~inclusion_mask].mean():.4f}")
    print(f"    Hierarchical: mean={stiff_hier.mean():.4f}  std={stiff_hier.std():.4f}  "
          f"in={stiff_hier[inclusion_mask].mean():.4f}  out={stiff_hier[~inclusion_mask].mean():.4f}")
    print(f"    Diff (hier-base) per-particle: mean={np.mean(stiff_hier-stiff_base):+.4f}  "
          f"|max|={np.max(np.abs(stiff_hier-stiff_base)):.4f}")
    print(f"  Calibrated tau stats:")
    print(f"    Baseline    : in={tau_base[inclusion_mask].mean():.4f}  "
          f"out={tau_base[~inclusion_mask].mean():.4f}")
    print(f"    Hierarchical: in={tau_hier[inclusion_mask].mean():.4f}  "
          f"out={tau_hier[~inclusion_mask].mean():.4f}")
    print(f"\n  IoU vs truth inclusion (top-K):")
    m_b = compute_metrics(stiff_base, stiff_truth, inclusion_mask)
    m_h = compute_metrics(stiff_hier, stiff_truth, inclusion_mask)
    print(f"    Baseline IoU    : {m_b['iou']:.4f}")
    print(f"    Hierarchical IoU: {m_h['iou']:.4f}")

    # ===== AUDIT 6: predict-on-T_A self-consistency =====
    print(f"\n{'='*72}\nAUDIT 6: predict on calibration T_A vs truth_obs_A\n{'='*72}")
    traj_A = truth["A"]["traj"]
    pb_A, pa_b_A = predict_on_trajectory(stiff_base, tau_base, traj_A, ctx)
    ph_A, pa_h_A = predict_on_trajectory(stiff_hier, tau_hier, traj_A, ctx)
    e_b_A, ea_b_A = chamfer_per_traj(pb_A, truth["A"]["pos"], pa_b_A, truth["A"]["alpha"])
    e_h_A, ea_h_A = chamfer_per_traj(ph_A, truth["A"]["pos"], pa_h_A, truth["A"]["alpha"])
    print(f"  Baseline    on T_A: pos_err = {e_b_A:.6f}   α_err = {ea_b_A:.6f}")
    print(f"  Hierarchical on T_A: pos_err = {e_h_A:.6f}  α_err = {ea_h_A:.6f}")
    if e_h_A < e_b_A:
        rel = (e_h_A - e_b_A) / e_b_A * 100
        print(f"  → hier {abs(rel):.2f}% BETTER on calibration trajectory")
        print(f"     ⚠ this means hier just fits the calibration data better,")
        print(f"     ⚠ not purely generalization advantage")

    print(f"\n{'='*72}\nAUDIT SUMMARY\n{'='*72}")
    print(f"  AUDIT 1 (truth self-test):  {'PASS' if all(audit1[n][0] < 1e-7 for n in audit1) else 'FAIL'}")
    print(f"  AUDIT 2 (order indep):       run inspection above")
    print(f"  AUDIT 3 (traj distinct):    inspect cavity & displacement diffs")
    print(f"  AUDIT 4/5 (calib outcomes):  compare stiff/tau diff above")
    print(f"  AUDIT 6 (self-consistency): {'hier overfits T_A' if e_h_A < e_b_A else 'hier underfits T_A'}")


if __name__ == "__main__":
    main()
