"""Day 7 Step 1: 1D landscape sweep of L_gap (Liang's residual norm loss).

Day 6e showed L_direct's landscape: minimum at θ_true on all 3 axes (clean U).
Day 7 Step 1 asks the SAME question for L_gap:

    L_gap(θ) = ‖Δ_x(θ)‖² + ‖Δ_dv(θ)‖²

where Δ_x, Δ_dv come from `simple_gd_inner_loop` solved at the given θ.

Why this matters:
  ReSECT's outer loop minimizes L_gap. If L_gap has min at θ_true, optimization
  is well-posed. If min is elsewhere, ReSECT is broken regardless of optimizer.

Day 5 failed here (sharpness sign-flip). Day 5's gradient was unverified;
NOW we have Day 6d v2-verified inner loop + Day 6b-verified forward, so this
sweep tells us if L_gap LANDSCAPE itself is sound.

Setup:
  Same single frame as Day 6e (1 step, knife at (0.15, 0.30, 0.15))
  obs generated at θ_true = (3, 20, 0.5)
  For each axis in {α_min, sharpness, τ}, sweep with others at truth, plot L_gap

Pass criterion:
  L_gap min within 2 grid steps of truth on ALL 3 axes
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
    setup_solver_with_rest, solver_load_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day6d_v2_simple_gd import simple_gd_inner_loop  # noqa: E402


def simple_gd_inner_loop_relu(
    pos_pred, dv_pred, pos_target, alpha_target, theta,
    edges_a, edges_b, edges_rest,
    n_iter=30, lr_pos=5e-3, lr_dv=0.1,
    lambda_cav=1.0, lambda_E=0.0,
):
    """Inner loop with proper constraint: dv_corrected = max(0, dv_sim + Δ_dv).

    NOT Δ_dv ≥ 0 (Mod 1 conflated this — that's only true when sim always
    under-damages real). The correct invariant is damage VALUE ≥ 0, i.e.,
    dv_corrected ≥ 0, allowing Δ_dv ∈ [−dv_sim, +∞).

    Implementation: Δ_dv is unconstrained; clamp dv_corr ≥ 0 after addition.
    """
    N = pos_pred.shape[0]
    s = theta[1]; tau = theta[2]

    delta_x = torch.zeros((N, 3), dtype=pos_pred.dtype, device=pos_pred.device,
                            requires_grad=True)
    delta_dv = torch.zeros(N, dtype=pos_pred.dtype, device=pos_pred.device,
                              requires_grad=True)
    L_init = None
    for k in range(n_iter):
        pos_corr = pos_pred + delta_x
        # Damage VALUE ≥ 0 constraint (not residual ≥ 0)
        dv_corr = torch.relu(dv_pred + delta_dv)
        alpha_corr = torch.sigmoid(-s * (dv_corr - tau))

        diff_pos = pos_corr - pos_target
        diff_alpha = alpha_corr - alpha_target
        L_pos = (diff_pos ** 2).sum()
        L_cav = (diff_alpha ** 2).sum()

        if lambda_E > 0:
            edge_vec = pos_corr[edges_a] - pos_corr[edges_b]
            edge_len = edge_vec.norm(dim=1).clamp(min=1e-12)
            L_E = ((edge_len - edges_rest) ** 2).sum()
        else:
            L_E = torch.tensor(0.0, device=pos_pred.device)

        L_inner = L_pos + lambda_cav * L_cav + lambda_E * L_E
        if L_init is None:
            L_init = float(L_inner)
        grad_x, grad_dv = torch.autograd.grad(
            L_inner, (delta_x, delta_dv),
            create_graph=True, retain_graph=True,
        )
        delta_x = delta_x - lr_pos * grad_x
        delta_dv = delta_dv - lr_dv * grad_dv

    return delta_x, delta_dv, dict(L_init=L_init, L_final=float(L_inner))


def gen_obs_single_frame(solver, rest_state, theta, knife_pos, knife_radius):
    """Single-frame obs at theta via legacy float path."""
    solver.set_theta_wp(None)
    solver.alpha_min = float(theta[0])
    solver.soft_removal_sharpness = float(theta[1])
    solver.aspiration_threshold = float(theta[2])
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


def compute_L_gap_at(theta_np, solver, rest_state, knife_pos, knife_radius,
                       pos_target_t, alpha_target_t, edges_a_t, edges_b_t,
                       edges_rest_t, device, n_inner=30, lambda_E=0.0):
    """At given theta, run forward + inner loop, return L_gap = ‖Δ_x‖² + ‖Δ_dv‖².

    lambda_E=0 (default for Step 1): pure data-fitting inner loop, no physics
    constraint. Original lambda_E=1.0 caused the bug where inner loop pulled
    pos toward REST (edges_rest = pf_rest_initial), conflicting with the
    target which is the END state after 1 sim step (already deformed).
    """
    theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=device,
                                 requires_grad=False)
    with torch.no_grad():
        alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
            theta_torch, solver, rest_state, knife_pos, knife_radius,
        )
    delta_x, delta_dv, info = simple_gd_inner_loop_relu(
        pos_t.detach(), dv_t.detach(), pos_target_t, alpha_target_t,
        theta_torch, edges_a_t, edges_b_t, edges_rest_t,
        n_iter=n_inner, lr_pos=5e-3, lr_dv=0.1,
        lambda_cav=1.0, lambda_E=lambda_E,
    )
    L_gap = float(((delta_x ** 2).sum() + (delta_dv ** 2).sum()).item())
    Lx = float((delta_x ** 2).sum().item())
    Ldv = float((delta_dv ** 2).sum().item())
    L_inner_init = info["L_init"]
    L_inner_final = info["L_final"]
    return L_gap, L_inner_init, L_inner_final, Lx, Ldv


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_pos = np.array([0.15, 0.30, 0.15], dtype=np.float32)
    knife_radius = 0.04
    theta_true = np.array([3.0, 20.0, 0.5], dtype=np.float32)

    print(f"\n[Setup] θ_true = {theta_true.tolist()}")
    print(f"[Setup] generating single-frame obs at θ_true …")
    pos_target_np, alpha_target_np = gen_obs_single_frame(
        solver, rest_state, theta_true, knife_pos, knife_radius)
    cavity_frac = float((alpha_target_np < 0.5).mean())
    print(f"  cavity_frac = {cavity_frac:.4f}  (~{int(cavity_frac*solver.N)} damaged)")

    pos_target_t = torch.tensor(pos_target_np, dtype=torch.float32, device=device)
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_rest_t = torch.tensor(solver.e["pf_rest"].numpy(),
                                  dtype=torch.float32, device=device)

    # Sanity: at θ_true, ‖Δ‖² should be very small (~0) — but ONLY if λ_E=0
    print(f"\n[Sanity] L_gap at θ_true (lambda_E=0, pure data-fitting):")
    L_gap_truth, L_inner_init, L_inner_final, Lx, Ldv = compute_L_gap_at(
        theta_true, solver, rest_state, knife_pos, knife_radius,
        pos_target_t, alpha_target_t, edges_a_t, edges_b_t, edges_rest_t, device,
        n_inner=30, lambda_E=0.0)
    print(f"  L_gap(θ_true)   = {L_gap_truth:.4e}   (‖Δx‖²={Lx:.4e}, ‖Δdv‖²={Ldv:.4e})")
    print(f"  L_inner: init={L_inner_init:.4e} → final={L_inner_final:.4e}  "
          f"(reduction {L_inner_init/max(L_inner_final,1e-12):.2f}×)")

    # Side-by-side: what happens with lambda_E=1.0?
    L_gap_truth_E1, _, _, Lx_E1, Ldv_E1 = compute_L_gap_at(
        theta_true, solver, rest_state, knife_pos, knife_radius,
        pos_target_t, alpha_target_t, edges_a_t, edges_b_t, edges_rest_t, device,
        n_inner=30, lambda_E=1.0)
    print(f"  L_gap(θ_true) with lambda_E=1.0 = {L_gap_truth_E1:.4e}   "
          f"(‖Δx‖²={Lx_E1:.4e}, ‖Δdv‖²={Ldv_E1:.4e})  ← BAD (forces Δx≠0)")

    # ---- 1D sweep per axis ----
    sweep_grids = {
        "alpha_min":   np.linspace(1.0, 8.0,  15, dtype=np.float32),
        "sharpness":   np.linspace(5.0, 35.0, 15, dtype=np.float32),
        "threshold":   np.linspace(0.2, 1.0,  15, dtype=np.float32),
    }
    axis_idx = {"alpha_min": 0, "sharpness": 1, "threshold": 2}
    truth_value = {"alpha_min": 3.0, "sharpness": 20.0, "threshold": 0.5}

    results = {}
    t0 = time.time()

    for axis_name, grid in sweep_grids.items():
        d = axis_idx[axis_name]
        print(f"\n[Sweep] {axis_name}  grid=[{grid[0]:.2f}, {grid[-1]:.2f}] "
              f"(truth={truth_value[axis_name]})  n_inner=30")
        L_gap_arr = np.zeros(len(grid))
        L_inner_init_arr = np.zeros(len(grid))
        L_inner_final_arr = np.zeros(len(grid))
        for i, val in enumerate(grid):
            theta_test = theta_true.copy()
            theta_test[d] = float(val)
            L_gap, L_inner_i, L_inner_f, _, _ = compute_L_gap_at(
                theta_test, solver, rest_state, knife_pos, knife_radius,
                pos_target_t, alpha_target_t, edges_a_t, edges_b_t, edges_rest_t,
                device, n_inner=30, lambda_E=0.0)
            L_gap_arr[i] = L_gap
            L_inner_init_arr[i] = L_inner_i
            L_inner_final_arr[i] = L_inner_f

        truth_idx = int(np.argmin(np.abs(grid - truth_value[axis_name])))
        i_min = int(np.argmin(L_gap_arr))
        print(f"  truth idx={truth_idx} (val={grid[truth_idx]:.3f})")
        print(f"  L_gap min at idx={i_min} (val={grid[i_min]:.3f}, "
              f"L_gap={L_gap_arr[i_min]:.4e})  Δ_from_truth={i_min-truth_idx}")
        print(f"    {axis_name:>12s}: " + " ".join(f"{v:.2f}" for v in grid))
        print(f"    L_gap (×1e3):   " + " ".join(f"{L_gap_arr[i]*1e3:6.2f}" for i in range(len(grid))))
        results[axis_name] = dict(
            grid=grid, L_gap=L_gap_arr,
            L_inner_init=L_inner_init_arr, L_inner_final=L_inner_final_arr,
            i_min=i_min, truth_idx=truth_idx,
        )

    elapsed = time.time() - t0
    print(f"\n[Timing] sweep done in {elapsed:.1f}s  "
          f"({elapsed/(3*15):.2f}s per L_gap eval)")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("Day 7 Step 1 verdict — is L_gap landscape sound?")
    print("=" * 70)
    print(f"  L_gap(θ_true) = {L_gap_truth:.4e}  "
          f"(should be small — sanity check)")

    all_pass_tight = True
    all_pass_loose = True
    for axis_name, r in results.items():
        delta = abs(r["i_min"] - r["truth_idx"])
        if delta == 0:
            verdict = "✓ L_gap min EXACTLY at truth"
        elif delta <= 2:
            verdict = f"⚠ L_gap min within 2 grid-steps (Δ={r['i_min']-r['truth_idx']})"
            all_pass_tight = False
        else:
            verdict = f"✗ L_gap min FAR from truth (Δ={r['i_min']-r['truth_idx']})"
            all_pass_tight = False
            all_pass_loose = False
        print(f"  {axis_name:>12s}: {verdict}")

    print()
    if all_pass_tight and L_gap_truth < 1e-2:
        print("✓ Day 7 Step 1 PASS — L_gap landscape clean, proceed to Step 2")
    elif all_pass_loose:
        print("⚠ Day 7 Step 1 PARTIAL — minima close but not exact, may still work")
    else:
        print("✗ Day 7 Step 1 FAIL — L_gap landscape has minima far from truth")
        print("   → ReSECT framework structurally broken; need different loss formulation")

    # Save
    npz_path = REPO_ROOT / "tools" / "day7_step1_lgap_landscape.npz"
    save_dict = {"theta_true": theta_true, "L_gap_truth": L_gap_truth}
    for axis_name, r in results.items():
        for key, val in r.items():
            save_dict[f"{axis_name}_{key}"] = val
    np.savez(npz_path, **save_dict)
    print(f"  → sweep data saved to {npz_path}")


if __name__ == "__main__":
    main()
