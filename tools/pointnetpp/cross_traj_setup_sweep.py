"""Robustness check: does hierarchical's cross-trajectory advantage hold
across multiple truth setups?

5 setup variations:
   S1 default     — center (0.15, 0.25, 0.15), r=0.06, contrast 0.3/0.8
   S2 smaller     — r=0.04 (less obs signal — harder task)
   S3 larger      — r=0.10 (more signal — easier)
   S4 off-center  — center (0.10, 0.25, 0.20) (asymmetric)
   S5 hi-contrast — stiff_in=1.5, stiff_out=0.2 (wider dynamic range)

For each setup:
   - Regenerate stiff/tau truth fields
   - Generate truth obs for T_A, T_B, T_C, T_D
   - Calibrate baseline on T_A
   - Calibrate hierarchical on T_A
   - Predict each method on each trajectory → mean pos err + alpha err
   - Δ = hier - base (negative = hier better)

Aggregate: out of 5 × 4 = 20 (setup, traj) cells:
   - How many cells does hier beat base on pos err? on alpha err?
   - If consistent (e.g., 15+/20), claim is robust.
   - If patchy (e.g., 10/20), claim depends on setup.

Compute budget: ~40 min total (15 epochs × 2 methods × 5 setups).
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))

import warp as wp  # noqa: E402
wp.init()
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

from tools.day2_solver_wrapper import setup_solver_with_rest  # noqa: E402
from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff, compute_metrics,
)
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.cross_trajectory_test import (  # noqa: E402
    make_traj, calibrate, predict_on_trajectory, chamfer_per_traj,
)


SETUPS = [
    dict(name="S1_default",     center=(0.15, 0.25, 0.15), radius=0.06,
         stiff_in=0.8, stiff_out=0.3, tau_in=0.20, tau_out=0.40),
    dict(name="S2_smaller",     center=(0.15, 0.25, 0.15), radius=0.04,
         stiff_in=0.8, stiff_out=0.3, tau_in=0.20, tau_out=0.40),
    dict(name="S3_larger",      center=(0.15, 0.25, 0.15), radius=0.10,
         stiff_in=0.8, stiff_out=0.3, tau_in=0.20, tau_out=0.40),
    dict(name="S4_offcenter",   center=(0.10, 0.25, 0.20), radius=0.06,
         stiff_in=0.8, stiff_out=0.3, tau_in=0.20, tau_out=0.40),
    dict(name="S5_hicontrast",  center=(0.15, 0.25, 0.15), radius=0.06,
         stiff_in=1.5, stiff_out=0.2, tau_in=0.15, tau_out=0.50),
]


def run_one_setup(setup_cfg, solver, mesh_info, H, ctx_base):
    """Run a complete setup: generate truth, calibrate both methods, predict."""
    pos_rest = mesh_info["vertices"]
    N = solver.N
    device = ctx_base["device"]
    knife_radius = ctx_base["knife_radius"]
    T = ctx_base["T"]
    theta_truth = ctx_base["theta_truth"]
    rest_state = ctx_base["rest_state"]
    edges_a_t = ctx_base["edges_a_t"]
    edges_b_t = ctx_base["edges_b_t"]
    edges_rest_t = ctx_base["edges_rest_t"]

    # Build setup-specific truth fields
    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest,
        stiff_in=setup_cfg["stiff_in"],
        stiff_out=setup_cfg["stiff_out"],
        center_override=setup_cfg["center"],
        radius_override=setup_cfg["radius"],
    )
    tau_truth = np.where(inclusion_mask, setup_cfg["tau_in"], setup_cfg["tau_out"]).astype(np.float32)
    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)
    n_incl = int(inclusion_mask.sum())
    print(f"  inclusion: {n_incl}/{N} particles ({n_incl/N*100:.2f}%)")

    # Generate truth obs for 4 trajectories
    truth = {}
    for name in ["A", "B", "C", "D"]:
        traj = make_traj(name, T=T)
        solver.reset_to_rest(pos_rest)
        obs_pos, obs_alpha = gen_obs_per_particle_stiff(
            solver, rest_state, theta_truth, stiff_truth, traj, knife_radius, device,
            tau_truth_np=tau_truth)
        truth[name] = dict(traj=traj, pos=obs_pos, alpha=obs_alpha)

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in truth["A"]["pos"]]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in truth["A"]["alpha"]]

    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=truth["A"]["traj"], knife_radius=knife_radius,
        stiff_init_np=stiff_init, stiff_truth_np=stiff_truth,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        theta_3d_np=theta_truth, inclusion_mask=inclusion_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H, pos_rest=pos_rest,
    )

    # Calibrate (15 epochs to save time)
    solver.reset_to_rest(pos_rest)
    t0 = time.time()
    stiff_b, tau_b, Lgap_b = calibrate("baseline", ctx, N_epochs=15)
    t_b = time.time() - t0
    solver.reset_to_rest(pos_rest)
    t0 = time.time()
    stiff_h, tau_h, Lgap_h = calibrate("hierarchical", ctx, N_epochs=15)
    t_h = time.time() - t0
    iou_b = compute_metrics(stiff_b, stiff_truth, inclusion_mask)["iou"]
    iou_h = compute_metrics(stiff_h, stiff_truth, inclusion_mask)["iou"]
    print(f"  calibrate: base {t_b:.0f}s  hier {t_h:.0f}s")
    print(f"  Lgap_best: base {Lgap_b:.3e}  hier {Lgap_h:.3e}")
    print(f"  IoU: base {iou_b:.3f}  hier {iou_h:.3f}")

    # Predict each method on each trajectory
    results = {}
    for name in ["A", "B", "C", "D"]:
        traj = truth[name]["traj"]
        pb, ab = predict_on_trajectory(stiff_b, tau_b, traj, ctx)
        ph, ah = predict_on_trajectory(stiff_h, tau_h, traj, ctx)
        e_pos_b, e_a_b = chamfer_per_traj(pb, truth[name]["pos"], ab, truth[name]["alpha"])
        e_pos_h, e_a_h = chamfer_per_traj(ph, truth[name]["pos"], ah, truth[name]["alpha"])
        results[name] = dict(
            pos_b=e_pos_b, pos_h=e_pos_h, alpha_b=e_a_b, alpha_h=e_a_h,
        )

    return dict(
        results=results, iou_b=iou_b, iou_h=iou_h,
        Lgap_b=Lgap_b, Lgap_h=Lgap_h,
        stiff_b=stiff_b, stiff_h=stiff_h,
        tau_b=tau_b, tau_h=tau_h,
        inclusion_mask=inclusion_mask,
        n_incl=n_incl,
    )


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    print(f"\n=== Cross-trajectory sweep over {len(SETUPS)} setups ===")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = 2

    device = "cuda:0"
    knife_radius = 0.08
    T = 10
    theta_truth = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    pos_rest = mesh_info["vertices"]

    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)

    # Hierarchy is mesh-dependent, not setup-dependent — build once
    x_rest = torch.from_numpy(pos_rest).to(device, dtype=torch.float32)
    H = build_hierarchy(x_rest, M1=512, M2=64, k_neighbors=3)

    ctx_base = dict(
        device=device, knife_radius=knife_radius, T=T,
        theta_truth=theta_truth, rest_state=rest_state,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
    )

    all_results = {}
    t_total = time.time()
    for setup_cfg in SETUPS:
        print(f"\n{'='*72}")
        print(f"Setup: {setup_cfg['name']}  center={setup_cfg['center']}  r={setup_cfg['radius']}  "
              f"stiff={setup_cfg['stiff_in']}/{setup_cfg['stiff_out']}")
        print(f"{'='*72}")
        t0 = time.time()
        all_results[setup_cfg["name"]] = run_one_setup(setup_cfg, solver, mesh_info, H, ctx_base)
        print(f"  setup elapsed {time.time()-t0:.0f}s")
    print(f"\nTotal: {(time.time()-t_total)/60:.1f} min")

    # ===== Aggregate report =====
    print(f"\n{'='*72}")
    print(f"SWEEP SUMMARY — pos_err per (setup, traj) — hier vs base")
    print(f"{'='*72}")
    print(f"\n  Per-cell pos_err (× 1e-5 m). Δ = hier - base (negative = hier better)")
    print(f"  {'Setup':18s}  {'T_A':>15s}    {'T_B':>15s}    {'T_C':>15s}    {'T_D':>15s}")
    for name, res in all_results.items():
        cells = []
        for tr in ["A", "B", "C", "D"]:
            b = res["results"][tr]["pos_b"] * 1e5
            h = res["results"][tr]["pos_h"] * 1e5
            d = h - b
            mark = "★" if d < -0.1 else (" " if abs(d) < 0.1 else "✗")
            cells.append(f"{b:>5.2f}/{h:>5.2f} {mark}{d:+5.2f}")
        print(f"  {name:18s}  " + "    ".join(cells))

    print(f"\n  ★ = hier beat base, ✗ = base beat hier (Δ > 0.1e-5 m threshold)")
    print(f"  Format: base/hier  Δ×1e5_m")

    print(f"\n  IoU per setup (lower = baseline finds better top-K than hier):")
    for name, res in all_results.items():
        di = res["iou_h"] - res["iou_b"]
        print(f"  {name:18s}  IoU base={res['iou_b']:.3f}  hier={res['iou_h']:.3f}  Δ={di:+.3f}")

    # Count wins on held-out
    print(f"\n  Held-out trajectory wins (T_B + T_C + T_D × {len(SETUPS)} setups = {3*len(SETUPS)} cells):")
    n_pos_hier_wins = 0
    n_alpha_hier_wins = 0
    n_cells = 0
    sum_pos_delta = 0.0
    sum_alpha_delta = 0.0
    for name, res in all_results.items():
        for tr in ["B", "C", "D"]:
            r = res["results"][tr]
            d_pos = r["pos_h"] - r["pos_b"]
            d_alpha = r["alpha_h"] - r["alpha_b"]
            if d_pos < 0:
                n_pos_hier_wins += 1
            if d_alpha < 0:
                n_alpha_hier_wins += 1
            sum_pos_delta += d_pos
            sum_alpha_delta += d_alpha
            n_cells += 1
    mean_pos_delta = sum_pos_delta / n_cells
    mean_alpha_delta = sum_alpha_delta / n_cells
    base_avg_pos = np.mean([res["results"][tr]["pos_b"]
                              for res in all_results.values() for tr in ["B", "C", "D"]])
    base_avg_alpha = np.mean([res["results"][tr]["alpha_b"]
                                for res in all_results.values() for tr in ["B", "C", "D"]])
    rel_pos = mean_pos_delta / base_avg_pos * 100
    rel_alpha = mean_alpha_delta / base_avg_alpha * 100

    print(f"    Position wins for hier:  {n_pos_hier_wins}/{n_cells}  "
          f"(mean Δ = {mean_pos_delta:+.3e} m, {rel_pos:+.2f}% vs base avg)")
    print(f"    Alpha    wins for hier:  {n_alpha_hier_wins}/{n_cells}  "
          f"(mean Δ = {mean_alpha_delta:+.3e}, {rel_alpha:+.2f}% vs base avg)")

    # Final verdict
    print(f"\n{'='*72}")
    print(f"ROBUSTNESS VERDICT")
    print(f"{'='*72}")
    if n_pos_hier_wins >= 0.85 * n_cells:
        print(f"  ✓ ROBUST: hierarchical wins position on ≥85% of held-out cells")
    elif n_pos_hier_wins >= 0.65 * n_cells:
        print(f"  △ PARTIAL: hierarchical wins on majority but not consistently")
    else:
        print(f"  ✗ NOT ROBUST: hierarchical fails on >35% of held-out cells")

    out_path = REPO / "tools" / "pointnetpp" / "cross_traj_sweep_results.npz"
    np.savez(
        out_path,
        setup_names=np.array([s["name"] for s in SETUPS]),
        results_grid=np.array([[[
            all_results[s["name"]]["results"][tr]["pos_b"],
            all_results[s["name"]]["results"][tr]["pos_h"],
            all_results[s["name"]]["results"][tr]["alpha_b"],
            all_results[s["name"]]["results"][tr]["alpha_h"],
        ] for tr in ["A", "B", "C", "D"]] for s in SETUPS]),
    )
    print(f"\n  Saved {out_path}")


if __name__ == "__main__":
    main()
