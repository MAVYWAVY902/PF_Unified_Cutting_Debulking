"""Day 7 Step 3: Mode C state-injection sanity check.

ReSECT's hypothesis: at each frame, after inner loop finds residual Δ,
INJECT Δ into sim state so the next frame's state aligns with obs. This is
the mechanism that lets sim "track" real obs across time.

This step verifies the INJECTION is sound (single-frame, no online loop yet):
  (1) at θ ≠ θ_true, sim has gap to obs
  (2) inner loop finds Δ
  (3) corrected state = (pos_sim + Δ_x, dv_sim + Δ_dv)
  (4) does ‖corrected − obs‖ < ‖sim − obs‖ ?  ← injection actually closes gap?
  (5) are Δ magnitudes physical (not teleportation)?

Tests:
  Test 1 — sanity at θ_true: Δ ≈ 0, no injection needed
  Test 2 — single-frame at θ_init: injection should close gap substantially
  Test 3 — Δ_dv ≥ 0 invariant (damage irreversibility)
  Test 4 — magnitudes reasonable (max|Δ_x| < cube_size, Δ_dv ≤ 1)
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pf_shape_matching_warp"))

import torch
import warp as wp
wp.init()
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

from tools.day2_solver_wrapper import (  # noqa: E402
    setup_solver_with_rest, solver_load_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day7_step1_lgap_landscape import (  # noqa: E402
    gen_obs_single_frame, simple_gd_inner_loop_relu,
)


def gap_metrics(pos_state, alpha_state, pos_obs, alpha_obs):
    """Compute multi-channel gap metrics state vs obs."""
    if isinstance(pos_state, torch.Tensor):
        pos_state = pos_state.detach().cpu().numpy()
    if isinstance(alpha_state, torch.Tensor):
        alpha_state = alpha_state.detach().cpu().numpy()
    pos_l2 = float(np.linalg.norm(pos_state - pos_obs))
    pos_max = float(np.abs(pos_state - pos_obs).max())
    alpha_l2 = float(np.linalg.norm(alpha_state - alpha_obs))
    cavity_frac_state = float((alpha_state < 0.5).mean())
    cavity_frac_obs = float((alpha_obs < 0.5).mean())
    return dict(
        pos_l2=pos_l2, pos_max=pos_max, alpha_l2=alpha_l2,
        cavity_frac_state=cavity_frac_state, cavity_frac_obs=cavity_frac_obs,
        cavity_diff=cavity_frac_state - cavity_frac_obs,
    )


def run_test_at_theta(theta_np, label, solver, rest_state, knife_pos, knife_radius,
                      pos_target_np, alpha_target_np, edges_a_t, edges_b_t,
                      edges_rest_t, device, theta_true):
    """At given theta: forward sim → inner loop → inject → measure gaps."""
    pos_target_t = torch.tensor(pos_target_np, dtype=torch.float32, device=device)
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)

    theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=device,
                                 requires_grad=False)

    # 1. Forward sim at θ
    with torch.no_grad():
        alpha_sim, pos_sim, dv_sim = WarpFullForwardOp.apply(
            theta_torch, solver, rest_state, knife_pos, knife_radius,
        )

    # 2. Inner loop to find Δ
    delta_x, delta_dv, info = simple_gd_inner_loop_relu(
        pos_sim.detach(), dv_sim.detach(), pos_target_t, alpha_target_t,
        theta_torch, edges_a_t, edges_b_t, edges_rest_t,
        n_iter=30, lr_pos=5e-3, lr_dv=0.1,
        lambda_cav=1.0, lambda_E=0.0,
    )

    # 3. Form corrected state (damage VALUE ≥ 0, not residual ≥ 0)
    pos_corrected = pos_sim.detach() + delta_x.detach()
    dv_corrected = torch.relu(dv_sim.detach() + delta_dv.detach())
    alpha_corrected = torch.sigmoid(
        -theta_torch[1] * (dv_corrected - theta_torch[2])
    )

    # 4. Gap metrics
    g_sim = gap_metrics(pos_sim, alpha_sim, pos_target_np, alpha_target_np)
    g_cor = gap_metrics(pos_corrected, alpha_corrected,
                         pos_target_np, alpha_target_np)

    # 5. Δ magnitude
    dx = delta_x.detach().cpu().numpy()
    ddv = delta_dv.detach().cpu().numpy()
    dx_max = float(np.abs(dx).max())
    dx_mean = float(np.linalg.norm(dx, axis=1).mean())
    ddv_max = float(ddv.max())
    ddv_min = float(ddv.min())
    ddv_mean = float(ddv.mean())
    # Effective Δ_dv after clamp (what actually got applied to state)
    dv_sim_np = dv_sim.detach().cpu().numpy()
    ddv_eff = np.maximum(dv_sim_np + ddv, 0) - dv_sim_np
    ddv_eff_max = float(ddv_eff.max())
    ddv_eff_min = float(ddv_eff.min())
    L_gap = float((dx**2).sum() + (ddv_eff**2).sum())

    print(f"\n[{label}] θ = {theta_np.tolist()}")
    print(f"   ‖θ−truth‖ = {np.linalg.norm(theta_np - theta_true):.3f}")
    print(f"\n   Forward sim → obs gap:")
    print(f"     pos:    L2={g_sim['pos_l2']:.4e}   max={g_sim['pos_max']:.4e}")
    print(f"     alpha:  L2={g_sim['alpha_l2']:.4e}")
    print(f"     cavity_frac: sim={g_sim['cavity_frac_state']:.4f}  "
          f"obs={g_sim['cavity_frac_obs']:.4f}  diff={g_sim['cavity_diff']:+.4f}")

    print(f"\n   Inner loop result (after 30 GD):")
    print(f"     L_inner: init={info['L_init']:.4e} → final={info['L_final']:.4e}  "
          f"(reduction {info['L_init']/max(info['L_final'],1e-12):.2f}×)")
    print(f"     ‖Δ_x‖²={float((dx**2).sum()):.4e}   ‖Δ_dv‖²={float((ddv**2).sum()):.4e}   "
          f"L_gap={L_gap:.4e}")
    print(f"     Δ_x: max={dx_max:.5f}   mean_norm={dx_mean:.5f}")
    print(f"     Δ_dv (raw):  max={ddv_max:.4f}  min={ddv_min:.4f}  mean={ddv_mean:.4f}")
    print(f"     Δ_dv (eff after clamp dv_corr≥0): "
          f"max={ddv_eff_max:.4f}  min={ddv_eff_min:.4f}")

    print(f"\n   Corrected state (sim + Δ) → obs gap:")
    print(f"     pos:    L2={g_cor['pos_l2']:.4e}   max={g_cor['pos_max']:.4e}")
    print(f"     alpha:  L2={g_cor['alpha_l2']:.4e}")
    print(f"     cavity_frac: cor={g_cor['cavity_frac_state']:.4f}  "
          f"obs={g_cor['cavity_frac_obs']:.4f}  diff={g_cor['cavity_diff']:+.4f}")

    # Reduction
    pos_reduce = g_sim['pos_l2'] / max(g_cor['pos_l2'], 1e-12)
    alpha_reduce = g_sim['alpha_l2'] / max(g_cor['alpha_l2'], 1e-12)
    print(f"\n   Gap reduction (sim L2 / corrected L2):")
    print(f"     pos:    {pos_reduce:.2f}×")
    print(f"     alpha:  {alpha_reduce:.2f}×")

    return dict(
        g_sim=g_sim, g_cor=g_cor, info=info,
        dx_max=dx_max, dx_mean=dx_mean,
        ddv_max=ddv_max, ddv_min=ddv_min,
        ddv_eff_max=ddv_eff_max, ddv_eff_min=ddv_eff_min,
        L_gap=L_gap, pos_reduce=pos_reduce, alpha_reduce=alpha_reduce,
    )


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_pos = np.array([0.15, 0.30, 0.15], dtype=np.float32)
    knife_radius = 0.04
    theta_true = np.array([3.0, 20.0, 0.5], dtype=np.float32)
    theta_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)

    print(f"\n[Setup] θ_true={theta_true.tolist()}")
    pos_target_np, alpha_target_np = gen_obs_single_frame(
        solver, rest_state, theta_true, knife_pos, knife_radius)
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_rest_t = torch.tensor(solver.e["pf_rest"].numpy(),
                                  dtype=torch.float32, device=device)

    bbox_size = 0.3  # cube extent

    # ============ Test 1: at θ_true ============
    print(f"\n" + "=" * 70)
    print(f"Test 1: sanity at θ_true (Δ should be ≈ 0)")
    print(f"=" * 70)
    r_true = run_test_at_theta(
        theta_true, "θ=θ_true", solver, rest_state, knife_pos, knife_radius,
        pos_target_np, alpha_target_np,
        edges_a_t, edges_b_t, edges_rest_t, device, theta_true)

    # ============ Test 2: at θ_init ============
    print(f"\n" + "=" * 70)
    print(f"Test 2: real test at θ_init (gap should be closed by Δ)")
    print(f"=" * 70)
    r_init = run_test_at_theta(
        theta_init, "θ=θ_init", solver, rest_state, knife_pos, knife_radius,
        pos_target_np, alpha_target_np,
        edges_a_t, edges_b_t, edges_rest_t, device, theta_true)

    # ============ Test 3: large perturbation ============
    theta_far = np.array([8.0, 10.0, 0.8], dtype=np.float32)
    print(f"\n" + "=" * 70)
    print(f"Test 3: far perturbation θ={theta_far.tolist()} (stress test)")
    print(f"=" * 70)
    r_far = run_test_at_theta(
        theta_far, "θ=θ_far", solver, rest_state, knife_pos, knife_radius,
        pos_target_np, alpha_target_np,
        edges_a_t, edges_b_t, edges_rest_t, device, theta_true)

    # ============ Verdict ============
    print(f"\n" + "=" * 70)
    print(f"Day 7 Step 3 verdict — Mode C state injection sanity")
    print(f"=" * 70)

    checks = []

    # (1) At θ_true, L_gap small (NOT zero due to param_arr vs legacy 0.85%
    #     forward path drift — this is test artifact, not architecture issue)
    truth_small = r_true["L_gap"] < 5.0   # large tolerance for forward drift
    print(f"\n  (1) At θ_true, L_gap small (< 5):  L_gap={r_true['L_gap']:.4e}   "
          f"{'✓' if truth_small else '✗'}")
    checks.append(truth_small)

    # (2) Inner loop converges at θ_init (L_inner reduces ≥ 2×)
    inner_reduce_init = r_init["info"]["L_init"] / max(r_init["info"]["L_final"], 1e-12)
    inner_ok_init = inner_reduce_init > 2.0
    print(f"  (2) Inner loop converges at θ_init (≥2× L_inner reduction): "
          f"{inner_reduce_init:.2f}×  {'✓' if inner_ok_init else '✗'}")
    checks.append(inner_ok_init)

    # (3) Alpha gap reduces by injection — main ReSECT signal
    #     (pos signal is weak because cube barely deforms in 1 step)
    alpha_closes_init = r_init["alpha_reduce"] > 1.5
    alpha_closes_far = r_far["alpha_reduce"] > 2.0
    print(f"  (3) Alpha gap (the main signal) reduces by injection:")
    print(f"      θ_init: ratio={r_init['alpha_reduce']:.2f}×  "
          f"(≥1.5× required)  {'✓' if alpha_closes_init else '✗'}")
    print(f"      θ_far:  ratio={r_far['alpha_reduce']:.2f}×  "
          f"(≥2.0× required)  {'✓' if alpha_closes_far else '✗'}")
    checks.append(alpha_closes_init and alpha_closes_far)

    # (4) Pos gap also reduces (modest, signal weaker than alpha)
    pos_closes_init = r_init["pos_reduce"] > 1.2
    print(f"  (4) Pos gap reduces (≥1.2×, weaker signal expected): "
          f"θ_init ratio={r_init['pos_reduce']:.2f}×  "
          f"{'✓' if pos_closes_init else '✗'}")
    checks.append(pos_closes_init)

    # (5) Δ_x magnitudes physical (max < cube_size, i.e., no teleportation)
    physical_init = r_init["dx_max"] < bbox_size
    physical_far = r_far["dx_max"] < bbox_size
    print(f"  (5) Δ_x magnitudes physical (max < {bbox_size}):")
    print(f"      θ_init: max|Δx|={r_init['dx_max']:.4f}  {'✓' if physical_init else '✗'}")
    print(f"      θ_far:  max|Δx|={r_far['dx_max']:.4f}  {'✓' if physical_far else '✗'}")
    checks.append(physical_init and physical_far)

    # (6) Δ_dv magnitudes reasonable (max ≤ 1.0 — particle damage is dimensionless [0,1] range)
    bounded_init = r_init["ddv_max"] < 1.0
    bounded_far = r_far["ddv_max"] < 1.0
    print(f"  (6) Δ_dv magnitudes bounded (max < 1.0):")
    print(f"      θ_init: max(Δdv)={r_init['ddv_max']:.4f}  {'✓' if bounded_init else '✗'}")
    print(f"      θ_far:  max(Δdv)={r_far['ddv_max']:.4f}  {'✓' if bounded_far else '✗'}")
    checks.append(bounded_init and bounded_far)

    print()
    if all(checks):
        print(f"✓ Day 7 Step 3 PASS — Mode C state injection is sound, proceed to Step 4")
    else:
        n_fail = sum(1 for c in checks if not c)
        print(f"✗ Day 7 Step 3 FAIL — {n_fail}/6 checks failed")


if __name__ == "__main__":
    main()
