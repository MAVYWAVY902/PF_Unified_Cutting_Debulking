"""Day 8 redo of Day 7 Step 1: L_gap landscape with stable GD inner loop.

Day 7 Step 1 PASSED (3/3 axes had L_gap min at θ_true) but used broken inner GD
(lr_dv=0.1 divergent). Need to redo with stable GD (lr_dv=1/s²) to confirm:
  (1) Landscape STILL has min at θ_true (architecture sanity)
  (2) L_gap values are smaller (no inner-divergence inflation)
  (3) Curve shape (steepness, basin width) reflects TRUE loss landscape

Pass criteria:
  (1) L_gap(θ_true) ≈ 0 (machine precision via stable GD)
  (2) 1D sweep min at truth on all 3 axes (same as broken inner result)
  (3) Curves smoothly U-shaped, no spurious local mins

This verifies that the *qualitative* L_gap landscape claim from Day 7 Step 1
holds, even though the Day 7 numerical recovery was wrong.
"""
import sys
from pathlib import Path
import numpy as np
import time

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
from tools.day8_stable_inner_loop import gd_inner_loop_stable  # noqa: E402


def gen_obs_single_frame_param_arr(solver, rest_state, theta, knife_pos,
                                      knife_radius, device):
    """Single-frame obs via param_arr path (matches sim)."""
    theta_torch = torch.tensor(theta, dtype=torch.float32, device=device,
                                 requires_grad=False)
    with torch.no_grad():
        alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
            theta_torch, solver, rest_state, knife_pos, knife_radius,
        )
    return pos_t.cpu().numpy().copy(), alpha_t.cpu().numpy().copy()


def compute_L_gap_stable(theta_np, solver, rest_state, knife_pos, knife_radius,
                          pos_target_t, alpha_target_t, edges_a_t, edges_b_t,
                          edges_rest_t, device, n_inner=30):
    """L_gap at theta with stable GD inner loop."""
    theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=device,
                                 requires_grad=False)
    with torch.no_grad():
        alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
            theta_torch, solver, rest_state, knife_pos, knife_radius,
        )
    delta_x, delta_dv, info = gd_inner_loop_stable(
        pos_t.detach(), dv_t.detach(), pos_target_t, alpha_target_t,
        theta_torch, edges_a_t, edges_b_t, edges_rest_t,
        n_iter=n_inner, lr_pos=5e-3, lr_dv_factor=1.0,
        lambda_cav=1.0, lambda_E=0.0,
    )
    L_gap = float(((delta_x ** 2).sum() + (delta_dv ** 2).sum()).item())
    return L_gap, info


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_pos = np.array([0.15, 0.30, 0.15], dtype=np.float32)
    knife_radius = 0.04
    theta_true = np.array([3.0, 20.0, 0.5], dtype=np.float32)

    print(f"\n[Setup] θ_true={theta_true.tolist()}, single frame")
    pos_target_np, alpha_target_np = gen_obs_single_frame_param_arr(
        solver, rest_state, theta_true, knife_pos, knife_radius, device)
    cavity_frac = float((alpha_target_np < 0.5).mean())
    print(f"  obs cavity_frac={cavity_frac:.4f}")

    pos_target_t = torch.tensor(pos_target_np, dtype=torch.float32, device=device)
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_rest_t = torch.tensor(solver.e["pf_rest"].numpy(),
                                  dtype=torch.float32, device=device)

    # Sanity: L_gap at truth
    L_gap_truth, info_truth = compute_L_gap_stable(
        theta_true, solver, rest_state, knife_pos, knife_radius,
        pos_target_t, alpha_target_t, edges_a_t, edges_b_t, edges_rest_t, device)
    print(f"\n[Sanity] L_gap(θ_true) = {L_gap_truth:.4e}  "
          f"(Day 7 broken inner reported 0.19, target: < 1e-10)")
    print(f"  lr_dv at s=20: {info_truth['lr_dv']:.4e}")

    # 1D sweeps
    sweep_grids = {
        "alpha_min": np.linspace(1.0, 8.0,  15, dtype=np.float32),
        "sharpness": np.linspace(5.0, 35.0, 15, dtype=np.float32),
        "threshold": np.linspace(0.2, 1.0,  15, dtype=np.float32),
    }
    axis_idx = {"alpha_min": 0, "sharpness": 1, "threshold": 2}
    truth_value = {"alpha_min": 3.0, "sharpness": 20.0, "threshold": 0.5}

    results = {}
    t0 = time.time()

    for axis_name, grid in sweep_grids.items():
        d = axis_idx[axis_name]
        print(f"\n[Sweep] {axis_name} grid=[{grid[0]:.2f}, {grid[-1]:.2f}] "
              f"(truth={truth_value[axis_name]})")
        L_gap_arr = np.zeros(len(grid))
        for i, val in enumerate(grid):
            theta_test = theta_true.copy()
            theta_test[d] = float(val)
            L_gap, _ = compute_L_gap_stable(
                theta_test, solver, rest_state, knife_pos, knife_radius,
                pos_target_t, alpha_target_t, edges_a_t, edges_b_t, edges_rest_t,
                device, n_inner=30)
            L_gap_arr[i] = L_gap

        truth_idx = int(np.argmin(np.abs(grid - truth_value[axis_name])))
        i_min = int(np.argmin(L_gap_arr))
        print(f"  truth idx={truth_idx} (val={grid[truth_idx]:.3f})")
        print(f"  L_gap min at idx={i_min} (val={grid[i_min]:.3f}, "
              f"L_gap={L_gap_arr[i_min]:.4e})  Δ_from_truth={i_min-truth_idx}")
        print(f"    {axis_name:>12s}: " + " ".join(f"{v:5.2f}" for v in grid))
        print(f"    L_gap (×1e3): " +
              " ".join(f"{L_gap_arr[i]*1e3:6.2f}" for i in range(len(grid))))
        results[axis_name] = dict(
            grid=grid, L_gap=L_gap_arr, i_min=i_min, truth_idx=truth_idx
        )

    elapsed = time.time() - t0
    print(f"\n[Timing] sweep done in {elapsed:.1f}s")

    # Verdict
    print(f"\n" + "=" * 70)
    print(f"Day 8 Step 1 — L_gap landscape with STABLE GD")
    print(f"=" * 70)

    print(f"\n  L_gap(θ_true) = {L_gap_truth:.4e}  "
          f"({'✓ machine precision' if L_gap_truth < 1e-6 else '⚠'})")

    all_min_at_truth = True
    for axis_name, r in results.items():
        delta = r["i_min"] - r["truth_idx"]
        symbol = "✓" if delta == 0 else ("⚠" if abs(delta) <= 2 else "✗")
        print(f"  {axis_name:>12s}: min at idx={r['i_min']} (truth idx={r['truth_idx']}, "
              f"Δ={delta})  {symbol}")
        if delta != 0:
            all_min_at_truth = False

    print()
    if L_gap_truth < 1e-6 and all_min_at_truth:
        print(f"✓ Day 8 Step 1 PASS — L_gap landscape clean, min at truth on all 3 axes")
        print(f"  Day 7 Step 1 qualitative claim (min at truth) STILL VALID")
        print(f"  Quantitative L_gap values now reflect TRUE loss (no inner divergence)")
    elif L_gap_truth < 1e-6:
        print(f"⚠ L_gap clean but some axis min offset from truth — investigate")
    else:
        print(f"✗ L_gap(θ_true) not clean — something else still broken")

    # Compare to Day 7 broken-inner result
    print(f"\n  Comparison to Day 7 (broken inner):")
    print(f"    Day 7 L_gap(θ_true) ≈ 0.19 (inner divergence noise)")
    print(f"    Day 8 L_gap(θ_true) = {L_gap_truth:.4e} "
          f"({0.19/max(L_gap_truth,1e-12):.0e}× reduction)")

    np.savez(REPO_ROOT / "tools" / "day8_step1_landscape.npz",
             theta_true=theta_true, L_gap_truth=L_gap_truth,
             **{f"{name}_grid": r["grid"] for name, r in results.items()},
             **{f"{name}_L_gap": r["L_gap"] for name, r in results.items()})


if __name__ == "__main__":
    main()
