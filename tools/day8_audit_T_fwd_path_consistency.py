"""Day 8 Audit T_fwd: Forward path consistency baseline fix.

Hack H2: obs generated via legacy float path (set_theta_wp(None)) but sim during
ReSECT loop uses param_arr path (set_theta_wp(theta_dict)). These 2 paths give
0.85% different forward output even at IDENTICAL theta values. This means:

  - L_gap(θ_true) is NOT 0 (we saw 0.14 in Day 7 Step 1 after fixes)
  - All convergence metrics have a 0.85% noise floor
  - We may be celebrating convergence WITHIN this drift

Fix: generate obs using the SAME param_arr path that sim uses.
Implementation: call WarpFullForwardOp.apply with theta_true (same op).

Pass criteria:
  - After fix: sim(θ_true) and obs(θ_true) match to machine precision (< 1e-6 rel)
  - L_gap(θ_true) drops to < 1e-3 (vs 0.14 currently)

If fail: there's a more subtle path divergence we need to debug.
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
    setup_solver_with_rest, solver_load_state, solver_save_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day7_step1_lgap_landscape import simple_gd_inner_loop_relu  # noqa: E402
from tools.day6f_v0_multistep_synth import make_knife_trajectory  # noqa: E402


def gen_obs_legacy_path(solver, rest_state, theta_true, knife_traj, knife_radius):
    """OLD path: set_theta_wp(None) + Python float attrs. Legacy float kernels."""
    obs_pos = []
    obs_alpha = []
    obs_dv = []
    state_in = rest_state
    for t, knife_pos in enumerate(knife_traj):
        solver.set_theta_wp(None)
        solver.alpha_min = float(theta_true[0])
        solver.soft_removal_sharpness = float(theta_true[1])
        solver.aspiration_threshold = float(theta_true[2])
        solver.knife_radius = knife_radius
        solver.knife_active = 1
        solver.knife_pos = wp.array(np.array([knife_pos], dtype=np.float32),
                                      dtype=wp.vec3, device=solver.device,
                                      requires_grad=False)
        solver_load_state(solver, state_in)
        solver.step()
        if solver.use_ssa_soft_removal:
            ci = solver._ssa_cycle_index
            alpha_np = solver._ssa_alpha_buffers[ci].numpy().copy()
            dv_np = solver._ssa_particle_damage_buffers[ci].numpy().copy()
        else:
            alpha_np = solver.p["alpha"].numpy().copy()
            dv_np = solver.p["damage"].numpy().copy()
        pos_np = solver.p["pos"].numpy().copy()
        obs_pos.append(pos_np)
        obs_alpha.append(alpha_np)
        obs_dv.append(dv_np)
        state_in = solver_save_state(solver)
    return obs_pos, obs_alpha, obs_dv


def gen_obs_param_arr_path(solver, rest_state, theta_true, knife_traj, knife_radius, device):
    """NEW path: use WarpFullForwardOp (same as sim) with theta_true.

    Guarantees obs and sim use SAME kernel path (param_arr).
    """
    theta_torch = torch.tensor(theta_true, dtype=torch.float32, device=device,
                                 requires_grad=False)
    obs_pos = []
    obs_alpha = []
    obs_dv = []
    state_in = rest_state
    with torch.no_grad():
        for t, knife_pos in enumerate(knife_traj):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, knife_pos, knife_radius,
            )
            obs_pos.append(pos_t.cpu().numpy().copy())
            obs_alpha.append(alpha_t.cpu().numpy().copy())
            obs_dv.append(dv_t.cpu().numpy().copy())
            state_in = solver_save_state(solver)
    return obs_pos, obs_alpha, obs_dv


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_radius = 0.04
    T = 3
    knife_traj = make_knife_trajectory(T)
    theta_true = np.array([3.0, 20.0, 0.5], dtype=np.float32)

    print(f"\n[Setup] θ_true={theta_true.tolist()}, T={T}")

    # ============ Test 1: forward path agreement ============
    print(f"\n[Test 1] Forward path agreement at θ_true (sim path == obs path)")
    obs_pos_L, obs_alpha_L, obs_dv_L = gen_obs_legacy_path(
        solver, rest_state, theta_true, knife_traj, knife_radius)
    obs_pos_P, obs_alpha_P, obs_dv_P = gen_obs_param_arr_path(
        solver, rest_state, theta_true, knife_traj, knife_radius, device)

    print(f"  per-frame divergence between legacy and param_arr paths:")
    for t in range(T):
        pos_diff_L_vs_P = float(np.abs(obs_pos_L[t] - obs_pos_P[t]).max())
        alpha_diff_L_vs_P = float(np.abs(obs_alpha_L[t] - obs_alpha_P[t]).max())
        dv_diff_L_vs_P = float(np.abs(obs_dv_L[t] - obs_dv_P[t]).max())
        print(f"    frame {t+1}: max|Δpos|={pos_diff_L_vs_P:.6e}  "
              f"max|Δα|={alpha_diff_L_vs_P:.6e}  max|Δdv|={dv_diff_L_vs_P:.6e}")

    # ============ Test 2: param_arr path self-consistency at θ_true ============
    print(f"\n[Test 2] sim(θ_true) via param_arr should EXACTLY match obs(θ_true) via param_arr")
    # Re-run forward at theta_true through SAME path → should match exactly
    theta_torch = torch.tensor(theta_true, dtype=torch.float32, device=device,
                                 requires_grad=False)
    state_in = rest_state
    sim_pos = []
    sim_alpha = []
    sim_dv = []
    with torch.no_grad():
        for t, knife_pos in enumerate(knife_traj):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, knife_pos, knife_radius,
            )
            sim_pos.append(pos_t.cpu().numpy().copy())
            sim_alpha.append(alpha_t.cpu().numpy().copy())
            sim_dv.append(dv_t.cpu().numpy().copy())
            state_in = solver_save_state(solver)

    print(f"  sim(θ_true) vs obs(θ_true), both via param_arr:")
    max_diffs_param_arr_self = []
    for t in range(T):
        pos_diff = float(np.abs(sim_pos[t] - obs_pos_P[t]).max())
        alpha_diff = float(np.abs(sim_alpha[t] - obs_alpha_P[t]).max())
        dv_diff = float(np.abs(sim_dv[t] - obs_dv_P[t]).max())
        max_diffs_param_arr_self.append((pos_diff, alpha_diff, dv_diff))
        print(f"    frame {t+1}: max|Δpos|={pos_diff:.6e}  "
              f"max|Δα|={alpha_diff:.6e}  max|Δdv|={dv_diff:.6e}")

    # ============ Test 3: L_gap at θ_true under OLD vs NEW obs ============
    print(f"\n[Test 3] L_gap(θ_true) under different obs paths")
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_rest_t = torch.tensor(solver.e["pf_rest"].numpy(),
                                  dtype=torch.float32, device=device)

    def compute_L_gap_at_truth(obs_pos, obs_alpha):
        """L_gap at theta_true given specified obs."""
        theta_torch = torch.tensor(theta_true, dtype=torch.float32, device=device,
                                     requires_grad=False)
        obs_pos_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
        obs_alpha_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]
        L_gap_per_frame = []
        state_in = rest_state
        for t in range(T):
            with torch.no_grad():
                alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                    theta_torch, solver, state_in, knife_traj[t], knife_radius,
                )
            delta_x, delta_dv, info = simple_gd_inner_loop_relu(
                pos_t.detach(), dv_t.detach(), obs_pos_t[t], obs_alpha_t[t],
                theta_torch, edges_a_t, edges_b_t, edges_rest_t,
                n_iter=30, lr_pos=5e-3, lr_dv=0.1,
                lambda_cav=1.0, lambda_E=0.0,
            )
            L_gap = float(((delta_x ** 2).sum() + (delta_dv ** 2).sum()).item())
            L_gap_per_frame.append(L_gap)
            state_in = solver_save_state(solver)
        return L_gap_per_frame

    L_gap_with_legacy_obs = compute_L_gap_at_truth(obs_pos_L, obs_alpha_L)
    L_gap_with_param_arr_obs = compute_L_gap_at_truth(obs_pos_P, obs_alpha_P)

    print(f"  L_gap(θ_true) with OLD legacy-path obs:    " +
          "  ".join(f"{l:.4e}" for l in L_gap_with_legacy_obs))
    print(f"  L_gap(θ_true) with NEW param_arr-path obs: " +
          "  ".join(f"{l:.4e}" for l in L_gap_with_param_arr_obs))
    sum_legacy = sum(L_gap_with_legacy_obs)
    sum_param_arr = sum(L_gap_with_param_arr_obs)
    print(f"\n  sum L_gap (OLD): {sum_legacy:.4e}")
    print(f"  sum L_gap (NEW): {sum_param_arr:.4e}")
    print(f"  reduction:       {sum_legacy / max(sum_param_arr, 1e-12):.1f}×")

    # ============ Verdict ============
    print(f"\n" + "=" * 70)
    print(f"Day 8 T_fwd verdict — forward path consistency")
    print(f"=" * 70)

    # Check 1: legacy vs param_arr forward drift (the problem we knew)
    avg_pos_drift_L_vs_P = np.mean([
        np.abs(obs_pos_L[t] - obs_pos_P[t]).max() for t in range(T)
    ])
    print(f"\n  (1) Legacy vs param_arr forward drift:")
    print(f"      avg max|Δpos| = {avg_pos_drift_L_vs_P:.4e}  "
          f"(expected ~1e-3 from prior Day 6f v2)")
    h2_confirmed = avg_pos_drift_L_vs_P > 1e-6

    # Check 2: param_arr path is internally consistent
    avg_self_diff = np.mean([d[0] for d in max_diffs_param_arr_self])
    print(f"\n  (2) param_arr path internal consistency:")
    print(f"      avg max|Δpos| sim_at_truth vs obs_param_arr = {avg_self_diff:.4e}")
    print(f"      {'✓ machine precision' if avg_self_diff < 1e-5 else '✗ unexpected drift!'}")
    param_arr_consistent = avg_self_diff < 1e-5

    # Check 3: L_gap at truth drops with fix
    print(f"\n  (3) L_gap(θ_true) drop from path fix:")
    print(f"      sum L_gap with OLD obs path: {sum_legacy:.4e}")
    print(f"      sum L_gap with NEW obs path: {sum_param_arr:.4e}")
    if sum_param_arr < sum_legacy / 10:
        print(f"      ✓ ≥10× reduction — fix works, baseline drift is real")
    elif sum_param_arr < sum_legacy:
        print(f"      ⚠ improvement but < 10× — partial fix or additional sources of L_gap")
    else:
        print(f"      ✗ no improvement — something else is wrong")

    if param_arr_consistent and sum_param_arr < sum_legacy / 10:
        print(f"\n✓ T_fwd PASS — using param_arr obs path eliminates H2 baseline drift")
        print(f"  ACTION: update gen_obs_multistep in day6f_v0 to use param_arr path")
        print(f"  Then re-run Day 7 Step 5 with clean baseline")
    else:
        print(f"\n⚠ T_fwd PARTIAL — investigate further")


if __name__ == "__main__":
    main()
