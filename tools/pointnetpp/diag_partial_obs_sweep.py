"""Comprehensive partial-obs sweep — λE refinement + multi-mesh confirm.

Phase E confirmed Liang inner loop + L_E recovers interior in partial obs.
Now refine:
   1. λE sweep on cube — find sweet spot, characterize sensitivity
   2. Multi-mesh confirm (tumor + torus) — geometry robustness
   3. Cross-traj T_B prediction — calibration generalizes

Tests:
   On M1_cube:
     λE ∈ {1e-5, 5e-5, 1e-4, 5e-4, 1e-3} × {Liang, Hier+Liang}  = 10 runs
   On M2_tumor (best λE from cube):
     {Liang, Hier+Liang}  = 2 runs
   On M3_torus (best λE from cube):
     {Liang, Hier+Liang}  = 2 runs

Total 14 runs × 60 epochs ≈ 25 min compute.

Metrics:
   - stiff_in_interior recovery (truth=0.8)
   - L1_interior_in (lower better)
   - IoU full inclusion
   - Cross-traj T_B prediction error
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
    build_inclusion_truth, gen_obs_per_particle_stiff, compute_metrics,
    STIFF_LOW, STIFF_HIGH,
)
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import (  # noqa: E402
    MESH_CONFIGS, make_traj_from_spec, predict_on_traj, err_per_traj,
)
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402


def setup_mesh(cfg, device, T=10):
    """Initialize a mesh setup for partial-obs test."""
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size
    incl_radius = cfg["incl_radius_abs"]
    top_axis = cfg["top_axis"]
    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=cfg["stiff_in"], stiff_out=cfg["stiff_out"],
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)

    # Surface mask = top 10% of mesh along top_axis
    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    n_surface = int(surface_mask.sum())
    n_incl = int(inclusion_mask.sum())
    n_incl_interior = int((inclusion_mask & ~surface_mask).sum())

    # Generate truth obs T_A + T_B
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
    H = build_hierarchy(x_rest, M1=min(512, N // 20), M2=min(64, N // 200), k_neighbors=3)

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
    print(f"  N={N}  incl interior {n_incl_interior}/{n_incl}  surface {n_surface}/{N} "
          f"({n_surface/N*100:.1f}%)")
    return ctx, truth, n_incl_interior, n_incl


def run_config(condition, lambda_E, ctx, truth, N_epochs=60, label=""):
    """One calibration + cross-traj eval."""
    t0 = time.time()
    stiff, tau, hist = calibrate(
        condition=condition, ctx=ctx,
        N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
        lambda_smooth=0.0, lambda_E=lambda_E,
    )
    elapsed = time.time() - t0
    if np.isnan(stiff).any():
        print(f"    {label} → NaN (failed)")
        return None
    m = compute_metrics(stiff, ctx["stiff_truth"], ctx["inclusion_mask"])
    pred_pos, pred_alpha = predict_on_traj(stiff, tau, truth["B"]["traj"], ctx)
    err_pos, err_a = err_per_traj(pred_pos, truth["B"]["pos"], pred_alpha, truth["B"]["alpha"])
    h = hist[-1]
    print(f"    {label} {elapsed:.0f}s  stiff_in={h['stiff_in']:.3f}  "
          f"interior_in={h['stiff_in_interior']:.3f}  "
          f"L1_int={h['L1_interior_in']:.3f}  IoU={m['iou']:.3f}  T_B={err_pos:.3e}",
          flush=True)
    return dict(stiff=stiff, tau=tau, hist=hist, iou=m["iou"],
                T_B_pos=err_pos, T_B_alpha=err_a, elapsed=elapsed)


def main():
    device = "cuda:0"
    T = 10
    N_epochs = 60

    # ===== 1. λE sweep on cube (10 runs) =====
    print(f"\n{'='*80}\n[Phase E.1] λE sweep on M1_cube (10 runs)\n{'='*80}")
    cfg_cube = MESH_CONFIGS[0]
    ctx_cube, truth_cube, n_int_cube, n_incl_cube = setup_mesh(cfg_cube, device, T=T)

    lambda_E_grid = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    sweep_results = {}
    for lE in lambda_E_grid:
        print(f"\n  λE = {lE}")
        sweep_results[("Liang", lE)] = run_config(
            "Liang_inner", lE, ctx_cube, truth_cube, N_epochs, f"  Liang λE={lE}")
        sweep_results[("Hier+Liang", lE)] = run_config(
            "Hier+Liang_inner", lE, ctx_cube, truth_cube, N_epochs, f"  Hier+L λE={lE}")

    # Find best λE for Liang baseline by interior_in closest to truth 0.8
    best_lE = min(
        [lE for lE in lambda_E_grid
         if sweep_results[("Liang", lE)] is not None],
        key=lambda lE: abs(sweep_results[("Liang", lE)]["hist"][-1]["stiff_in_interior"] - 0.8),
    )
    print(f"\n  Best λE for cube Liang: {best_lE}")

    # ===== 2. Multi-mesh confirm at best λE (4 runs) =====
    print(f"\n{'='*80}\n[Phase E.2] Multi-mesh confirm at λE={best_lE}\n{'='*80}")
    multi_results = {("M1_cube", "Liang"): sweep_results[("Liang", best_lE)],
                     ("M1_cube", "Hier+Liang"): sweep_results[("Hier+Liang", best_lE)]}
    for cfg in MESH_CONFIGS[1:]:  # M2_tumor, M3_torus
        print(f"\n  Mesh: {cfg['name']}")
        ctx_m, truth_m, n_int_m, n_incl_m = setup_mesh(cfg, device, T=T)
        if n_int_m == 0:
            print(f"    skipping {cfg['name']}: no interior inclusion")
            continue
        multi_results[(cfg["name"], "Liang")] = run_config(
            "Liang_inner", best_lE, ctx_m, truth_m, N_epochs, f"  Liang")
        multi_results[(cfg["name"], "Hier+Liang")] = run_config(
            "Hier+Liang_inner", best_lE, ctx_m, truth_m, N_epochs, f"  Hier+L")

    # ===== Summary tables =====
    print(f"\n{'='*80}\nλE SWEEP SUMMARY (M1_cube)\n{'='*80}")
    print(f"  {'λE':>8s}  {'Liang interior_in':>18s}  {'Hier+L interior_in':>20s}  "
          f"{'Liang IoU':>10s}  {'Hier+L IoU':>11s}")
    for lE in lambda_E_grid:
        rL = sweep_results.get(("Liang", lE))
        rH = sweep_results.get(("Hier+Liang", lE))
        rL_in = rL["hist"][-1]["stiff_in_interior"] if rL else float("nan")
        rH_in = rH["hist"][-1]["stiff_in_interior"] if rH else float("nan")
        rL_iou = rL["iou"] if rL else float("nan")
        rH_iou = rH["iou"] if rH else float("nan")
        print(f"  {lE:>8.0e}  {rL_in:>18.3f}  {rH_in:>20.3f}  {rL_iou:>10.3f}  {rH_iou:>11.3f}")

    print(f"\n{'='*80}\nMULTI-MESH at λE={best_lE}\n{'='*80}")
    print(f"  {'Mesh':>10s}  {'Liang interior_in':>18s}  {'Hier+L interior_in':>20s}  "
          f"{'Liang T_B':>11s}  {'Hier+L T_B':>12s}")
    for mesh_name in ["M1_cube", "M2_tumor", "M3_torus"]:
        rL = multi_results.get((mesh_name, "Liang"))
        rH = multi_results.get((mesh_name, "Hier+Liang"))
        rL_in = rL["hist"][-1]["stiff_in_interior"] if rL else float("nan")
        rH_in = rH["hist"][-1]["stiff_in_interior"] if rH else float("nan")
        rL_TB = rL["T_B_pos"] if rL else float("nan")
        rH_TB = rH["T_B_pos"] if rH else float("nan")
        print(f"  {mesh_name:>10s}  {rL_in:>18.3f}  {rH_in:>20.3f}  "
              f"{rL_TB:>11.3e}  {rH_TB:>12.3e}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # λE sweep
    Liang_in = [sweep_results[("Liang", lE)]["hist"][-1]["stiff_in_interior"]
                  if sweep_results[("Liang", lE)] else float("nan") for lE in lambda_E_grid]
    Hier_in = [sweep_results[("Hier+Liang", lE)]["hist"][-1]["stiff_in_interior"]
                 if sweep_results[("Hier+Liang", lE)] else float("nan") for lE in lambda_E_grid]
    axes[0].semilogx(lambda_E_grid, Liang_in, "o-", label="Liang inner")
    axes[0].semilogx(lambda_E_grid, Hier_in, "s-", label="Hier + Liang")
    axes[0].axhline(0.8, color="k", linestyle="--", label="truth 0.8")
    axes[0].axhline(0.5, color="gray", linestyle=":", label="init 0.5")
    axes[0].set_xlabel("λE (L_E weight)")
    axes[0].set_ylabel("stiff_in interior mean")
    axes[0].set_title("λE sweep on M1_cube (60 epochs)\n"
                       f"best λE={best_lE} → Liang interior={sweep_results[('Liang', best_lE)]['hist'][-1]['stiff_in_interior']:.3f}")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Multi-mesh bars
    meshes = ["M1_cube", "M2_tumor", "M3_torus"]
    L_vals = [multi_results.get((m, "Liang"), {}).get("hist", [{}])[-1].get("stiff_in_interior", 0)
               if multi_results.get((m, "Liang")) else 0 for m in meshes]
    H_vals = [multi_results.get((m, "Hier+Liang"), {}).get("hist", [{}])[-1].get("stiff_in_interior", 0)
                if multi_results.get((m, "Hier+Liang")) else 0 for m in meshes]
    x = np.arange(len(meshes))
    w = 0.35
    axes[1].bar(x - w/2, L_vals, w, label="Liang inner")
    axes[1].bar(x + w/2, H_vals, w, label="Hier + Liang")
    axes[1].axhline(0.8, color="k", linestyle="--", label="truth 0.8")
    axes[1].axhline(0.5, color="gray", linestyle=":", label="init 0.5")
    axes[1].set_xticks(x); axes[1].set_xticklabels(meshes)
    axes[1].set_ylabel("stiff_in interior mean")
    axes[1].set_title(f"Multi-mesh confirm at λE={best_lE}")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out_path = REPO / "tools" / "pointnetpp" / "diag_partial_obs_sweep.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
