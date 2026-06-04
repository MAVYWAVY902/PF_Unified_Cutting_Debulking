"""Hier value map — systematic grid sweep to characterize WHERE hier helps/hurts.

Goal: replace single-config conclusion with configuration-dependent map.

Sweep axes:
   Inclusion size:  r=0.04 (~0.8%) | r=0.07 (~5%) | r=0.10 (~11%)
   num_iter:        2 (low sensitivity) | 16 (high sensitivity)
   Obs regime:      full mesh obs | partial obs (top 10% surface)

Total: 3 × 2 × 2 = 12 configurations × 2 methods (Liang vs Hier+Liang)
Per config: calibrate both, predict T_B held-out, compare.

Output:
   Multi-panel heatmap showing:
   - stiff_in recovery (Hier vs Liang vs truth 0.8)
   - T_B prediction error (Hier vs Liang)
   - Hier improvement % (positive = hier helps)

Estimated compute: 24 runs × ~2 min = ~50 min total.
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


def setup_cell(incl_radius, num_iter, obs_regime, cfg, device, T=10):
    """Build ctx for one configuration cell.

    obs_regime: 'full' or 'partial'
    """
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    # 2026-05-30 PHYSICS CALIBRATION (sanity audit, see eval_hier_phantom_realistic):
    # damping 0.95 (kill zombie momentum), gravity=0 (axis-mismatch isolation),
    # floor below mesh (tumor was half-buried).
    solver.damping = 0.95
    solver.num_iterations = num_iter
    solver.gravity = wp.vec3(0.0, 0.0, 0.0)
    pos_rest = mesh_info["vertices"]
    solver.floor_y = float(pos_rest[:, 2].min() - 0.1 * (pos_rest[:, 2].max() - pos_rest[:, 2].min()))
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size
    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=0.8, stiff_out=0.3,
        center_override=tuple(incl_center), radius_override=incl_radius)
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)
    top_axis = cfg["top_axis"]

    # surface_mask depends on obs_regime
    if obs_regime == "full":
        surface_mask = np.ones(N, dtype=bool)  # all particles visible
    else:  # partial
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


def evaluate_cell(ctx, truth, hier_mode, N_epochs=30, lambda_E=1e-5):
    """Calibrate + cross-traj predict for one method."""
    condition = "Hier+Liang_inner" if hier_mode else "Liang_inner"
    t0 = time.time()
    stiff, tau, hist = calibrate(
        condition=condition, ctx=ctx,
        N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
        lambda_smooth=0.0, lambda_E=lambda_E,
    )
    if np.isnan(stiff).any():
        return None
    # Cross-traj T_B
    pred_pos, pred_alpha = predict_on_traj(stiff, tau, truth["B"]["traj"], ctx)
    # Compute prediction error vs truth_B  (full mesh L2)
    T = len(truth["B"]["pos"])
    pos_errs = []
    for t in range(T):
        d = np.linalg.norm(pred_pos[t] - truth["B"]["pos"][t], axis=1)
        pos_errs.append(float(d.mean()))
    T_B_err = float(np.mean(pos_errs))

    stiff_in = float(stiff[ctx["inclusion_mask"]].mean())
    elapsed = time.time() - t0
    return dict(
        stiff=stiff, tau=tau, stiff_in=stiff_in, T_B_err=T_B_err,
        elapsed=elapsed,
    )


def main():
    cfg = MESH_CONFIGS[0]
    device = "cuda:0"
    T = 10

    # Sweep axes
    INCL_SIZES = [
        ("small", 0.04),    # ~0.8%
        ("medium", 0.07),   # ~5%
        ("large", 0.10),    # ~11%
    ]
    NUM_ITERS = [2, 16]
    OBS_REGIMES = ["full", "partial"]

    print(f"\n=== Hier value map sweep ===")
    print(f"  Configurations: {len(INCL_SIZES)} × {len(NUM_ITERS)} × {len(OBS_REGIMES)} = "
          f"{len(INCL_SIZES) * len(NUM_ITERS) * len(OBS_REGIMES)} cells")
    print(f"  Methods per cell: 2 (Liang baseline, Hier+Liang)")
    print(f"  Total runs: {len(INCL_SIZES) * len(NUM_ITERS) * len(OBS_REGIMES) * 2}")

    results = {}  # (incl_name, num_iter, obs_regime, method) -> dict
    t_total = time.time()
    for incl_name, incl_r in INCL_SIZES:
        for num_iter in NUM_ITERS:
            for obs_regime in OBS_REGIMES:
                key_base = (incl_name, num_iter, obs_regime)
                print(f"\n{'='*72}")
                print(f"Cell: incl={incl_name}(r={incl_r}) iter={num_iter} obs={obs_regime}")
                print(f"{'='*72}", flush=True)
                ctx, truth = setup_cell(incl_r, num_iter, obs_regime, cfg, device, T=T)
                n_incl = int(ctx["inclusion_mask"].sum())
                print(f"  N={ctx['N']}, incl={n_incl} ({n_incl/ctx['N']*100:.2f}%), "
                      f"surface={int(ctx['surface_mask'].sum())}")
                # Liang baseline
                r_L = evaluate_cell(ctx, truth, hier_mode=False, N_epochs=30, lambda_E=1e-5)
                # Hier+Liang
                r_H = evaluate_cell(ctx, truth, hier_mode=True, N_epochs=30, lambda_E=1e-5)
                if r_L is None or r_H is None:
                    print(f"  ✗ NaN in calibration, skipping")
                    continue
                # Hier improvement (positive = better, negative = worse)
                gap_L = abs(r_L["stiff_in"] - 0.8)
                gap_H = abs(r_H["stiff_in"] - 0.8)
                stiff_improvement = (gap_L - gap_H) / max(gap_L, 1e-9) * 100
                T_B_improvement = (r_L["T_B_err"] - r_H["T_B_err"]) / max(r_L["T_B_err"], 1e-9) * 100
                print(f"  Liang: stiff_in={r_L['stiff_in']:.3f} (gap {gap_L:.3f})  T_B_err={r_L['T_B_err']:.3e}")
                print(f"  Hier:  stiff_in={r_H['stiff_in']:.3f} (gap {gap_H:.3f})  T_B_err={r_H['T_B_err']:.3e}")
                print(f"  Hier improvement: stiff {stiff_improvement:+.1f}%, T_B {T_B_improvement:+.1f}%")
                results[(*key_base, "Liang")] = r_L
                results[(*key_base, "Hier")] = r_H

    print(f"\nTotal: {(time.time()-t_total)/60:.1f} min")

    # Build summary table + heatmaps
    print(f"\n{'='*100}")
    print(f"HIER VALUE MAP SUMMARY (truth stiff_in = 0.8)")
    print(f"{'='*100}")
    print(f"\n  {'incl':>8s}  {'iter':>5s}  {'obs':>9s}  "
          f"{'Liang stiff':>12s}  {'Hier stiff':>11s}  "
          f"{'L err':>9s}  {'H err':>9s}  {'Δstiff%':>9s}  {'ΔT_B%':>9s}  Verdict")
    for incl_name, _ in INCL_SIZES:
        for num_iter in NUM_ITERS:
            for obs_regime in OBS_REGIMES:
                rL = results.get((incl_name, num_iter, obs_regime, "Liang"))
                rH = results.get((incl_name, num_iter, obs_regime, "Hier"))
                if rL is None or rH is None:
                    continue
                gap_L = abs(rL["stiff_in"] - 0.8)
                gap_H = abs(rH["stiff_in"] - 0.8)
                d_stiff = (gap_L - gap_H) / max(gap_L, 1e-9) * 100
                d_TB = (rL["T_B_err"] - rH["T_B_err"]) / max(rL["T_B_err"], 1e-9) * 100
                verdict = "★ hier helps" if d_stiff > 5 else ("✗ hier hurts" if d_stiff < -5 else "≈ marginal")
                print(f"  {incl_name:>8s}  {num_iter:>5d}  {obs_regime:>9s}  "
                      f"{rL['stiff_in']:>12.3f}  {rH['stiff_in']:>11.3f}  "
                      f"{rL['T_B_err']:>9.2e}  {rH['T_B_err']:>9.2e}  "
                      f"{d_stiff:>+8.1f}%  {d_TB:>+8.1f}%  {verdict}")

    # Heatmap visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # 4 panels: (stiff_imp, T_B_imp) × (iter=2, iter=16)
    for col, num_iter in enumerate(NUM_ITERS):
        for row, metric in enumerate(["stiff_recovery", "T_B_prediction"]):
            ax = axes[row, col]
            mat = np.full((len(INCL_SIZES), len(OBS_REGIMES)), np.nan)
            for i, (incl_name, _) in enumerate(INCL_SIZES):
                for j, obs_regime in enumerate(OBS_REGIMES):
                    rL = results.get((incl_name, num_iter, obs_regime, "Liang"))
                    rH = results.get((incl_name, num_iter, obs_regime, "Hier"))
                    if rL is None or rH is None:
                        continue
                    if metric == "stiff_recovery":
                        gap_L = abs(rL["stiff_in"] - 0.8)
                        gap_H = abs(rH["stiff_in"] - 0.8)
                        mat[i, j] = (gap_L - gap_H) / max(gap_L, 1e-9) * 100
                    else:
                        mat[i, j] = (rL["T_B_err"] - rH["T_B_err"]) / max(rL["T_B_err"], 1e-9) * 100
            v_lim = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 5)
            im = ax.imshow(mat, cmap="RdYlGn", vmin=-v_lim, vmax=v_lim, aspect="auto")
            for i in range(len(INCL_SIZES)):
                for j in range(len(OBS_REGIMES)):
                    val = mat[i, j]
                    if np.isnan(val):
                        ax.text(j, i, "NaN", ha="center", va="center", fontsize=9)
                    else:
                        ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                                color="black" if abs(val) < v_lim * 0.6 else "white",
                                fontsize=10, fontweight="bold")
            ax.set_xticks(range(len(OBS_REGIMES)))
            ax.set_xticklabels(OBS_REGIMES)
            ax.set_yticks(range(len(INCL_SIZES)))
            ax.set_yticklabels([f"{n} (r={r})" for n, r in INCL_SIZES])
            ax.set_xlabel("Obs regime")
            ax.set_ylabel("Inclusion size")
            title = (f"{metric.replace('_', ' ').title()}\n"
                       f"num_iter={num_iter}  (green=hier helps, red=hier hurts)")
            ax.set_title(title, fontsize=11)
            plt.colorbar(im, ax=ax, label="Hier improvement %")
    fig.suptitle("HIER VALUE MAP — where does hier help vs hurt?\n"
                  f"3 inclusion sizes × 2 num_iter × 2 obs regimes = 12 cells",
                  fontsize=13)
    fig.tight_layout()
    out_path = REPO / "tools" / "pointnetpp" / "diag_hier_value_map.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
