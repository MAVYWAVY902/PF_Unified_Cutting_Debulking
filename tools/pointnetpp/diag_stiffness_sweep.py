"""Stiffness truth sweep — does hier benefit depend on truth stiff_in?

Vary truth stiff_in ∈ {0.5, 0.7, 0.9, 1.1, 1.3, 1.5} (stiff_out=0.3 fixed).
For each truth, run both Liang inner alone vs Hier+Liang in PARTIAL obs regime.

Hypothesis to test:
   - Does hier improvement scale with truth distance from init (0.5)?
   - Does hier help even when truth_in is close to init (0.5)?
   - Does hier help more on extreme contrast (truth_in=1.5)?

Setup:
   M1_cube buried inclusion
   Partial obs (top 10%, y > 0.27)
   λE = 1e-5 (best from earlier sweep)
   60 epochs each

Total: 6 truth values × 2 methods = 12 runs ~15 min.
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
    build_inclusion_truth, gen_obs_per_particle_stiff, compute_metrics, STIFF_LOW, STIFF_HIGH,
)
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import (  # noqa: E402
    MESH_CONFIGS, make_traj_from_spec, predict_on_traj, err_per_traj,
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

    # Vary stiff_in_truth, fix stiff_out=0.3, tau=truth
    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=float(stiff_in_truth), stiff_out=0.3,
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)

    # Surface mask = top 10% of cube
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


def main():
    cfg = MESH_CONFIGS[0]
    device = "cuda:0"
    T = 10
    N_epochs = 60
    lambda_E = 1e-5  # best from prior sweep

    stiff_in_grid = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    print(f"\n=== Stiffness truth sweep ===")
    print(f"  Truth stiff_in: {stiff_in_grid}")
    print(f"  stiff_out fixed at 0.3, init uniform 0.5")
    print(f"  Loss: Liang inner + L_E (λE={lambda_E})")
    print(f"  Partial obs (top 10% surface coverage)")

    results = {}  # (stiff_in_truth, method) -> stats
    for s_truth in stiff_in_grid:
        print(f"\n{'='*72}")
        print(f"Truth stiff_in = {s_truth}")
        print(f"{'='*72}")
        ctx, truth = setup_for_stiff_truth(s_truth, cfg, device, T=T)
        n_int = int((ctx["inclusion_mask"] & ~ctx["surface_mask"]).sum())
        n_inc = int(ctx["inclusion_mask"].sum())
        print(f"  N={ctx['N']}, interior inclusion {n_int}/{n_inc}")

        for label, condition in [("Liang", "Liang_inner"), ("Hier+L", "Hier+Liang_inner")]:
            t0 = time.time()
            stiff, tau, hist = calibrate(
                condition=condition, ctx=ctx,
                N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                lambda_smooth=0.0, lambda_E=lambda_E,
            )
            elapsed = time.time() - t0
            if np.isnan(stiff).any():
                print(f"    {label}: NaN")
                continue
            interior_mask = ~ctx["surface_mask"]
            interior_incl_mask = ctx["inclusion_mask"] & interior_mask
            stiff_in_int = float(stiff[interior_incl_mask].mean())
            stiff_out_recovered = float(stiff[~ctx["inclusion_mask"]].mean())
            L1_int = float(np.abs(stiff[interior_incl_mask] - ctx["stiff_truth"][interior_incl_mask]).mean())
            m = compute_metrics(stiff, ctx["stiff_truth"], ctx["inclusion_mask"])
            # Cross-traj T_B
            pred_pos, pred_alpha = predict_on_traj(stiff, tau, truth["B"]["traj"], ctx)
            err_pos, _ = err_per_traj(pred_pos, truth["B"]["pos"], pred_alpha, truth["B"]["alpha"])
            print(f"    {label} {elapsed:.0f}s  recovered stiff_in_int={stiff_in_int:.3f}  "
                  f"stiff_out={stiff_out_recovered:.3f}  L1={L1_int:.3f}  IoU={m['iou']:.3f}  "
                  f"T_B={err_pos:.3e}",
                  flush=True)
            results[(s_truth, label)] = dict(
                stiff_in_int=stiff_in_int, stiff_out=stiff_out_recovered,
                L1_int=L1_int, iou=m["iou"], T_B=err_pos,
                gap=stiff_in_int - s_truth,
            )

    # Summary table
    print(f"\n{'='*72}")
    print(f"STIFFNESS SWEEP SUMMARY  (recovered vs truth)")
    print(f"{'='*72}")
    print(f"  {'truth_in':>9s}  {'Liang_int':>9s}  {'Hier_int':>9s}  {'Hier-Liang Δ':>14s}  "
          f"{'Liang gap':>10s}  {'Hier gap':>10s}")
    for s in stiff_in_grid:
        rL = results.get((s, "Liang"))
        rH = results.get((s, "Hier+L"))
        if rL is None or rH is None:
            continue
        gap_L = rL["gap"]  # signed: positive means over-recovered, negative means under
        gap_H = rH["gap"]
        print(f"  {s:>9.2f}  {rL['stiff_in_int']:>9.3f}  {rH['stiff_in_int']:>9.3f}  "
              f"{rH['stiff_in_int']-rL['stiff_in_int']:>+14.3f}  "
              f"{gap_L:>+10.3f}  {gap_H:>+10.3f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    s_vals = stiff_in_grid
    Liang_in = [results[(s, "Liang")]["stiff_in_int"] for s in s_vals]
    Hier_in = [results[(s, "Hier+L")]["stiff_in_int"] for s in s_vals]
    Liang_L1 = [results[(s, "Liang")]["L1_int"] for s in s_vals]
    Hier_L1 = [results[(s, "Hier+L")]["L1_int"] for s in s_vals]
    Liang_iou = [results[(s, "Liang")]["iou"] for s in s_vals]
    Hier_iou = [results[(s, "Hier+L")]["iou"] for s in s_vals]
    Liang_TB = [results[(s, "Liang")]["T_B"] for s in s_vals]
    Hier_TB = [results[(s, "Hier+L")]["T_B"] for s in s_vals]

    # 1. Recovered vs truth (diagonal = perfect)
    axes[0, 0].plot(s_vals, s_vals, "k--", label="perfect (y=x)", lw=1.5)
    axes[0, 0].plot(s_vals, Liang_in, "o-", label="Liang inner", lw=2, markersize=8)
    axes[0, 0].plot(s_vals, Hier_in, "s-", label="Hier+Liang", lw=2, markersize=8)
    axes[0, 0].axhline(0.5, color="gray", linestyle=":", label="init 0.5")
    axes[0, 0].set_xlabel("Truth stiff_in")
    axes[0, 0].set_ylabel("Recovered stiff_in interior mean")
    axes[0, 0].set_title("Recovery vs truth")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # 2. Gap (recovered - truth)
    axes[0, 1].plot(s_vals, [r-t for r,t in zip(Liang_in, s_vals)], "o-", label="Liang gap", lw=2)
    axes[0, 1].plot(s_vals, [r-t for r,t in zip(Hier_in, s_vals)], "s-", label="Hier gap", lw=2)
    axes[0, 1].axhline(0, color="k", linestyle="--", label="perfect (0)")
    axes[0, 1].set_xlabel("Truth stiff_in")
    axes[0, 1].set_ylabel("Recovered - truth (negative = under-recovered)")
    axes[0, 1].set_title("Recovery gap vs truth")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # 3. L1 interior error
    axes[1, 0].plot(s_vals, Liang_L1, "o-", label="Liang L1", lw=2)
    axes[1, 0].plot(s_vals, Hier_L1, "s-", label="Hier L1", lw=2)
    axes[1, 0].set_xlabel("Truth stiff_in")
    axes[1, 0].set_ylabel("L1 error interior particles")
    axes[1, 0].set_title("Per-particle L1 error (lower better)")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # 4. T_B err + IoU twin
    ax = axes[1, 1]
    ax.plot(s_vals, Liang_TB, "o-", label="Liang T_B", lw=2, color="tab:blue")
    ax.plot(s_vals, Hier_TB, "s-", label="Hier T_B", lw=2, color="tab:red")
    ax.set_xlabel("Truth stiff_in")
    ax.set_ylabel("T_B pos err", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(s_vals, Liang_iou, "v--", label="Liang IoU", lw=1.5, color="tab:purple", alpha=0.7)
    ax2.plot(s_vals, Hier_iou, "^--", label="Hier IoU", lw=1.5, color="tab:orange", alpha=0.7)
    ax2.set_ylabel("IoU", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")
    ax.set_title("T_B cross-traj err (solid) + IoU (dashed)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Truth stiff_in sweep — partial obs (10% surface), λE={lambda_E}, 60 epochs",
                  fontsize=13)
    fig.tight_layout()
    out_path = REPO / "tools" / "pointnetpp" / "diag_stiffness_sweep.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
