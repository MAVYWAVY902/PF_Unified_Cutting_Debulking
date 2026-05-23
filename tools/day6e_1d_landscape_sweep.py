"""Day 6e diagnostic: 1D loss landscape sweep around θ_true.

Question: is L_direct = L_pos + λ_cav·L_cav INFORMATIVE about each
individual parameter? I.e., when we fix 2 params at truth and vary the 3rd,
does the loss have a minimum at the true value?

  - If YES on all 3 axes → identifiability OK per-axis; the wrong-direction
    gradient at θ_init=(5,15,0.6) is a COUPLING / LANDSCAPE issue (saddle,
    non-convex valley), fixable with better init or sequential optimization
  - If NO on some axis → that param is genuinely under-determined by our
    1-step single-cut single-observable obs. Need more data: multi-step
    trajectory, multi-knife-position, force trace, or raw dv (not just α)

Sweep design (all at θ_truth = (3.0, 20.0, 0.5), grids 15 points each):
  - α_min ∈ [1.0, 8.0]    (truth 3.0)
  - sharpness ∈ [5, 35]   (truth 20)
  - τ ∈ [0.2, 1.0]        (truth 0.5)

Outputs:
  - Loss curve per axis, with truth marker
  - Min location per axis
  - Saves npz for plotting
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


def forward_no_grad(solver, rest_state, theta_np, knife_pos, knife_radius, device):
    """One forward without autograd via legacy float path → (pos, alpha)."""
    solver.set_theta_wp(None)
    solver.alpha_min = float(theta_np[0])
    solver.soft_removal_sharpness = float(theta_np[1])
    solver.aspiration_threshold = float(theta_np[2])
    solver.knife_radius = knife_radius
    solver.knife_active = 1
    solver.knife_pos = wp.array(np.array([knife_pos], dtype=np.float32),
                                  dtype=wp.vec3, device=solver.device,
                                  requires_grad=False)
    solver_load_state(solver, rest_state)
    solver.step()
    if solver.use_ssa_soft_removal:
        ci = solver._ssa_cycle_index
        alpha_np = solver._ssa_alpha_buffers[ci].numpy().copy()
    else:
        alpha_np = solver.p["alpha"].numpy().copy()
    pos_np = solver.p["pos"].numpy().copy()
    return pos_np, alpha_np


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_pos = np.array([0.15, 0.30, 0.15], dtype=np.float32)
    knife_radius = 0.04
    theta_true = np.array([3.0, 20.0, 0.5], dtype=np.float32)

    # ---- Generate target ----
    print(f"\n[Setup] generating obs at θ_true = {theta_true.tolist()}")
    pos_target_np, alpha_target_np = forward_no_grad(
        solver, rest_state, theta_true, knife_pos, knife_radius, device)
    cavity_frac = float((alpha_target_np < 0.5).mean())
    print(f"  target cavity_frac = {cavity_frac:.4f}  "
          f"(~{int(cavity_frac*solver.N)} particles 'damaged')")

    # Probe λ_cav at truth (value-balance)
    L_pos_t = float(((pos_target_np - pos_target_np) ** 2).sum())  # = 0 by def
    # Use θ_init perturbation for non-zero scales
    theta_probe_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)
    pos_probe, alpha_probe = forward_no_grad(
        solver, rest_state, theta_probe_init, knife_pos, knife_radius, device)
    L_pos_p = float(((pos_probe - pos_target_np) ** 2).sum())
    L_cav_p = float(((alpha_probe - alpha_target_np) ** 2).sum())
    lambda_cav = max(L_pos_p / max(L_cav_p, 1e-12), 1e-6)
    print(f"  λ_cav (probe at θ_init=(5,15,0.6)) = {lambda_cav:.4e}")

    # ---- 1D sweep per axis ----
    sweep_grids = {
        "alpha_min":   np.linspace(1.0, 8.0,  15, dtype=np.float32),
        "sharpness":   np.linspace(5.0, 35.0, 15, dtype=np.float32),
        "threshold":   np.linspace(0.2, 1.0,  15, dtype=np.float32),
    }
    axis_idx = {"alpha_min": 0, "sharpness": 1, "threshold": 2}
    truth_value = {"alpha_min": 3.0, "sharpness": 20.0, "threshold": 0.5}

    results = {}

    for axis_name, grid in sweep_grids.items():
        d = axis_idx[axis_name]
        print(f"\n[Sweep] {axis_name}  grid = "
              f"[{grid[0]:.3f} … {grid[-1]:.3f}]  (truth={truth_value[axis_name]:.3f})")
        L_pos_arr = np.zeros(len(grid))
        L_cav_arr = np.zeros(len(grid))
        L_outer_arr = np.zeros(len(grid))
        for i, val in enumerate(grid):
            theta_test = theta_true.copy()
            theta_test[d] = float(val)
            pos_t, alpha_t = forward_no_grad(
                solver, rest_state, theta_test, knife_pos, knife_radius, device)
            L_pos = float(((pos_t - pos_target_np) ** 2).sum())
            L_cav = float(((alpha_t - alpha_target_np) ** 2).sum())
            L_outer = L_pos + lambda_cav * L_cav
            L_pos_arr[i] = L_pos
            L_cav_arr[i] = L_cav
            L_outer_arr[i] = L_outer
        # Find min
        i_min_outer = int(np.argmin(L_outer_arr))
        i_min_pos = int(np.argmin(L_pos_arr))
        i_min_cav = int(np.argmin(L_cav_arr))
        truth_idx = int(np.argmin(np.abs(grid - truth_value[axis_name])))
        print(f"  truth idx={truth_idx} (val={grid[truth_idx]:.3f})")
        print(f"  L_pos min   at idx={i_min_pos} (val={grid[i_min_pos]:.3f}, "
              f"L={L_pos_arr[i_min_pos]:.4e})  Δ_from_truth={i_min_pos-truth_idx}")
        print(f"  L_cav min   at idx={i_min_cav} (val={grid[i_min_cav]:.3f}, "
              f"L={L_cav_arr[i_min_cav]:.4e})  Δ_from_truth={i_min_cav-truth_idx}")
        print(f"  L_outer min at idx={i_min_outer} (val={grid[i_min_outer]:.3f}, "
              f"L={L_outer_arr[i_min_outer]:.4e})  Δ_from_truth={i_min_outer-truth_idx}")
        # Print the full curve compactly
        print(f"    {axis_name:>12s}: " + " ".join(f"{v:.2f}" for v in grid))
        print(f"    L_outer scaled (×1e4): " +
              " ".join(f"{L_outer_arr[i]*1e4:6.2f}" for i in range(len(grid))))
        results[axis_name] = dict(
            grid=grid, L_pos=L_pos_arr, L_cav=L_cav_arr, L_outer=L_outer_arr,
            i_min_pos=i_min_pos, i_min_cav=i_min_cav, i_min_outer=i_min_outer,
            truth_idx=truth_idx,
        )

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("1D landscape sweep verdict")
    print("=" * 70)
    for axis_name in sweep_grids:
        r = results[axis_name]
        truth_aligned = (r["i_min_outer"] == r["truth_idx"])
        within_2_grid = abs(r["i_min_outer"] - r["truth_idx"]) <= 2
        if truth_aligned:
            verdict = "✓ truth IS L_outer min"
        elif within_2_grid:
            verdict = f"⚠ L_outer min within 2 grid-steps of truth (Δ={r['i_min_outer']-r['truth_idx']})"
        else:
            verdict = f"✗ L_outer min FAR from truth (Δ={r['i_min_outer']-r['truth_idx']})"
        print(f"  {axis_name:>12s}: {verdict}")
        # Also check if L_outer is monotone-ish around truth (basin shape)
        truth_L = r["L_outer"][r["truth_idx"]]
        left_L = r["L_outer"][:r["truth_idx"]] if r["truth_idx"] > 0 else np.array([])
        right_L = r["L_outer"][r["truth_idx"]+1:] if r["truth_idx"] < len(r["grid"])-1 else np.array([])
        n_left_below = int((left_L < truth_L).sum())
        n_right_below = int((right_L < truth_L).sum())
        if n_left_below + n_right_below == 0:
            shape = "    → clean valley around truth"
        else:
            shape = f"    → {n_left_below} grid pts left + {n_right_below} right are BELOW truth value"
        print(shape)

    # Save
    npz_path = REPO_ROOT / "tools" / "day6e_1d_sweep.npz"
    save_dict = {}
    for axis_name, r in results.items():
        for key, val in r.items():
            save_dict[f"{axis_name}_{key}"] = val
    save_dict["theta_true"] = theta_true
    save_dict["lambda_cav"] = lambda_cav
    np.savez(npz_path, **save_dict)
    print(f"\n  → sweep data saved to {npz_path}")


if __name__ == "__main__":
    main()
