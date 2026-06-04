"""Stiffness sweep — PARADIGM-CORRECT metrics.

Earlier diag_stiffness_sweep.py reported stiff_recovered vs stiff_truth — that
is the MATERIAL SYS-ID frame (we don't do that).

This script reports the CALIBRATION frame metrics:
   - L_direct on T_A (sim ≈ obs at calibration trajectory)
   - L_direct on T_B (sim ≈ obs at held-out trajectory)
   - Alpha L1 (cut pattern fidelity)
   - All measured on full mesh particles after calibration

Whether stiff_recovered = stiff_truth NOT measured (irrelevant for calibration).

For each truth stiff_in ∈ {0.5, 0.7, 0.9, 1.1, 1.3, 1.5}:
   Run Liang inner alone + Hier+Liang
   Report: sim-obs distance on T_A (training) + T_B (held-out)
"""

import sys
import time
import math
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))

import warp as wp  # noqa: E402
wp.init()
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402

from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff,
)
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import (  # noqa: E402
    MESH_CONFIGS, make_traj_from_spec, predict_on_traj,
)
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402


def setup_for_stiff_truth(stiff_in_truth, cfg, device, T=10):
    """Build ctx and truth obs for a specific stiff_in truth value."""
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size
    incl_radius = cfg["incl_radius_abs"]
    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=float(stiff_in_truth), stiff_out=0.3,
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)
    top_axis = cfg["top_axis"]
    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold

    theta_truth = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    truth = {}
    for tr in ["A", "B"]:
        traj = make_traj_from_spec(cfg["traj_specs"][tr], bbox_min, bbox_max, T)
        solver.reset_to_rest(pos_rest)
        obs_pos, obs_alpha = gen_obs_per_particle_stiff(
            solver, rest_state, theta_truth, stiff_truth, traj, knife_radius, device,
            tau_truth_np=tau_truth)
        truth[tr] = dict(traj=traj, pos=obs_pos, alpha=obs_alpha)

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in truth["A"]["pos"]]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in truth["A"]["alpha"]]
    surface_idx = torch.tensor(np.where(surface_mask)[0], dtype=torch.long, device=device)
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64), dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64), dtype=torch.long, device=device)
    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)
    x_rest = torch.from_numpy(pos_rest).to(device, dtype=torch.float32)
    H = build_hierarchy(x_rest, M1=512, M2=64, k_neighbors=3)

    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=truth["A"]["traj"], knife_radius=knife_radius,
        pos_rest=pos_rest, theta_truth=theta_truth, theta_3d_np=theta_truth,
        stiff_init_np=stiff_init, stiff_truth=stiff_truth,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        inclusion_mask=inclusion_mask,
        surface_idx=surface_idx, surface_mask=surface_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H,
    )
    return ctx, truth


def sim_obs_distance(pred_pos, pred_alpha, obs_pos, obs_alpha, ctx=None,
                     surface_only=False):
    """Paradigm-correct metric: mean per-particle pos L2 distance + alpha L1.

    Args:
        pred_pos, pred_alpha: list of [N,3] / [N] sim outputs per frame
        obs_pos, obs_alpha:   same for truth observation
        surface_only: if True, restrict to visible surface particles
    """
    T = len(pred_pos)
    pos_err_t = []
    alpha_err_t = []
    for t in range(T):
        if surface_only and ctx is not None:
            mask = ctx["surface_mask"]
            d = np.linalg.norm(pred_pos[t][mask] - obs_pos[t][mask], axis=1)
            pos_err_t.append(float(d.mean()))
            alpha_err_t.append(float(np.abs(pred_alpha[t][mask] - obs_alpha[t][mask]).mean()))
        else:
            d = np.linalg.norm(pred_pos[t] - obs_pos[t], axis=1)
            pos_err_t.append(float(d.mean()))
            alpha_err_t.append(float(np.abs(pred_alpha[t] - obs_alpha[t]).mean()))
    return float(np.mean(pos_err_t)), float(np.mean(alpha_err_t))


def main():
    cfg = MESH_CONFIGS[0]
    device = "cuda:0"
    T = 10
    N_epochs = 60
    lambda_E = 1e-5

    stiff_in_grid = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    print(f"\n=== Paradigm-correct stiffness sweep ===")
    print(f"  Metric: sim-obs distance (L_direct) on T_A (training) + T_B (held-out)")
    print(f"  NOT measuring: stiff_recovered vs stiff_truth (material sys-ID frame)")

    results = {}
    for s_truth in stiff_in_grid:
        print(f"\n{'='*72}")
        print(f"Truth stiff_in = {s_truth}")
        print(f"{'='*72}")
        ctx, truth = setup_for_stiff_truth(s_truth, cfg, device, T=T)

        for label, condition in [("Liang", "Liang_inner"), ("Hier+L", "Hier+Liang_inner")]:
            t0 = time.time()
            stiff, tau, hist = calibrate(
                condition=condition, ctx=ctx,
                N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                lambda_smooth=0.0, lambda_E=lambda_E,
            )
            if np.isnan(stiff).any():
                print(f"    {label}: NaN")
                continue
            # Predict on T_A (training) and T_B (held-out)
            pred_A_pos, pred_A_alpha = predict_on_traj(stiff, tau, truth["A"]["traj"], ctx)
            pred_B_pos, pred_B_alpha = predict_on_traj(stiff, tau, truth["B"]["traj"], ctx)

            # Full mesh sim-obs distance
            L_A_full, L_A_alpha = sim_obs_distance(pred_A_pos, pred_A_alpha,
                                                      truth["A"]["pos"], truth["A"]["alpha"])
            L_B_full, L_B_alpha = sim_obs_distance(pred_B_pos, pred_B_alpha,
                                                      truth["B"]["pos"], truth["B"]["alpha"])
            # Surface-only (the "visible" obs)
            L_A_surf, L_A_alpha_surf = sim_obs_distance(pred_A_pos, pred_A_alpha,
                                                            truth["A"]["pos"], truth["A"]["alpha"],
                                                            ctx, surface_only=True)
            L_B_surf, L_B_alpha_surf = sim_obs_distance(pred_B_pos, pred_B_alpha,
                                                            truth["B"]["pos"], truth["B"]["alpha"],
                                                            ctx, surface_only=True)
            elapsed = time.time() - t0
            interior_in = float(stiff[ctx["inclusion_mask"] & ~ctx["surface_mask"]].mean())
            print(f"    {label} {elapsed:.0f}s  "
                  f"L_A_full={L_A_full:.3e}  L_B_full={L_B_full:.3e}  "
                  f"L_A_surf={L_A_surf:.3e}  L_B_surf={L_B_surf:.3e}  "
                  f"(stiff_int={interior_in:.3f})", flush=True)
            results[(s_truth, label)] = dict(
                L_A_full=L_A_full, L_B_full=L_B_full,
                L_A_surf=L_A_surf, L_B_surf=L_B_surf,
                L_A_alpha=L_A_alpha, L_B_alpha=L_B_alpha,
                stiff_in_int=interior_in,
            )

    # Summary
    print(f"\n{'='*100}")
    print(f"PARADIGM-CORRECT SWEEP SUMMARY  (sim-obs distance, NOT stiff_truth match)")
    print(f"{'='*100}")
    print(f"  {'truth':>6s}  {'Liang L_A_full':>14s}  {'Hier L_A_full':>14s}  "
          f"{'Liang L_B_full':>14s}  {'Hier L_B_full':>14s}  "
          f"{'Hier vs Liang T_A':>18s}  {'Hier vs Liang T_B':>18s}")
    for s in stiff_in_grid:
        rL = results.get((s, "Liang"))
        rH = results.get((s, "Hier+L"))
        if rL is None or rH is None:
            continue
        relA = (rH["L_A_full"] - rL["L_A_full"]) / rL["L_A_full"] * 100
        relB = (rH["L_B_full"] - rL["L_B_full"]) / rL["L_B_full"] * 100
        print(f"  {s:>6.2f}  {rL['L_A_full']:>14.3e}  {rH['L_A_full']:>14.3e}  "
              f"{rL['L_B_full']:>14.3e}  {rH['L_B_full']:>14.3e}  "
              f"{relA:>+17.2f}%  {relB:>+17.2f}%")

    # Plot — 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    s_vals = stiff_in_grid
    Liang_LA = [results[(s, "Liang")]["L_A_full"] for s in s_vals]
    Hier_LA = [results[(s, "Hier+L")]["L_A_full"] for s in s_vals]
    Liang_LB = [results[(s, "Liang")]["L_B_full"] for s in s_vals]
    Hier_LB = [results[(s, "Hier+L")]["L_B_full"] for s in s_vals]
    Liang_aA = [results[(s, "Liang")]["L_A_alpha"] for s in s_vals]
    Hier_aA = [results[(s, "Hier+L")]["L_A_alpha"] for s in s_vals]
    Liang_aB = [results[(s, "Liang")]["L_B_alpha"] for s in s_vals]
    Hier_aB = [results[(s, "Hier+L")]["L_B_alpha"] for s in s_vals]

    axes[0, 0].plot(s_vals, Liang_LA, "o-", label="Liang", lw=2, markersize=8)
    axes[0, 0].plot(s_vals, Hier_LA, "s-", label="Hier+Liang", lw=2, markersize=8)
    axes[0, 0].set_xlabel("Truth stiff_in")
    axes[0, 0].set_ylabel("L_pos_full on T_A")
    axes[0, 0].set_title("Calibration fit: sim-vs-obs distance on T_A (training)\n"
                          "Lower = sim matches obs better")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_yscale("log")

    axes[0, 1].plot(s_vals, Liang_LB, "o-", label="Liang", lw=2, markersize=8)
    axes[0, 1].plot(s_vals, Hier_LB, "s-", label="Hier+Liang", lw=2, markersize=8)
    axes[0, 1].set_xlabel("Truth stiff_in")
    axes[0, 1].set_ylabel("L_pos_full on T_B (held-out)")
    axes[0, 1].set_title("Predictive validity: sim-vs-obs on T_B (cross-traj)\n"
                          "Lower = calibration generalizes better")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_yscale("log")

    axes[1, 0].plot(s_vals, Liang_aA, "o-", label="Liang", lw=2)
    axes[1, 0].plot(s_vals, Hier_aA, "s-", label="Hier+Liang", lw=2)
    axes[1, 0].set_xlabel("Truth stiff_in")
    axes[1, 0].set_ylabel("Alpha L1 on T_A")
    axes[1, 0].set_title("Cut pattern fidelity (T_A)")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(s_vals, Liang_aB, "o-", label="Liang", lw=2)
    axes[1, 1].plot(s_vals, Hier_aB, "s-", label="Hier+Liang", lw=2)
    axes[1, 1].set_xlabel("Truth stiff_in")
    axes[1, 1].set_ylabel("Alpha L1 on T_B")
    axes[1, 1].set_title("Cut pattern cross-traj prediction")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    fig.suptitle(f"Paradigm-correct metrics: sim-vs-obs (calibration frame), NOT stiff recovery\n"
                  f"Partial obs (10%), λE=1e-5, 60 epochs",
                  fontsize=12)
    fig.tight_layout()
    out_path = REPO / "tools" / "pointnetpp" / "diag_stiffness_sweep_paradigm.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
