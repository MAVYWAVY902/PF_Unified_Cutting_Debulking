"""Visualize with SURFACE inclusion — knife directly interacts with inclusion.

Previous setup had inclusion buried below top surface (center y=0.24,
knife at y=0.30 only penetrates 3cm → barely touches inclusion).

New setup: inclusion center on top surface (y=0.30, half-hemisphere within
mesh). Knife sweeping at y=0.30 directly cuts through inclusion.

Expected: stronger signal → both methods recover inclusion better → visual
difference between baseline and hierarchical becomes apparent.
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
from matplotlib.patches import Circle  # noqa: E402

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402

from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff, compute_metrics, STIFF_LOW, STIFF_HIGH,
)
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import (  # noqa: E402
    make_traj_from_spec, predict_on_traj, err_per_traj,
)
from tools.pointnetpp.visualize_behavior import calibrate  # noqa: E402


# Surface inclusion config — inclusion center AT top surface
SURFACE_CFG = dict(
    name="M1_cube_surface",
    path="resources/cube/cube_tet.msh",
    top_axis=1,
    knife_radius_abs=0.030,
    # incl_center on top surface (y=1.00 of bbox → y=0.30 = bbox_max_y)
    incl_center_rel=(0.5, 1.00, 0.5),
    incl_radius_abs=0.060,    # bigger so half-hemisphere has substantial particles
    traj_specs={
        "A": dict(sweep_axis=0, sweep_range=(0.40, 0.60), fixed={1: 1.00, 2: 0.50}),
        "B": dict(sweep_axis=2, sweep_range=(0.40, 0.60), fixed={0: 0.50, 1: 1.00}),
        "C": dict(sweep_axis=1, sweep_range=(1.00, 0.95), fixed={0: 0.50, 2: 0.50}),
        "D": dict(sweep_axis=0, sweep_range=(0.58, 0.74), fixed={1: 1.00, 2: 0.50}),
    },
    stiff_in=0.8, stiff_out=0.3,
    tau_in=0.20, tau_out=0.40,
)


def plot_slice(ax, pos, values, vmin, vmax, slice_axis, slice_val, slice_tol,
                title, cmap="viridis"):
    mask = np.abs(pos[:, slice_axis] - slice_val) < slice_tol
    others = [i for i in [0, 1, 2] if i != slice_axis]
    sc = ax.scatter(pos[mask, others[0]], pos[mask, others[1]],
                       c=values[mask], cmap=cmap, s=15, vmin=vmin, vmax=vmax)
    ax.set_xlabel(f"axis {others[0]} ({['x','y','z'][others[0]]})")
    ax.set_ylabel(f"axis {others[1]} ({['x','y','z'][others[1]]})")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    return sc


def main():
    cfg = SURFACE_CFG
    device = "cuda:0"
    T = 10
    print(f"\n=== Surface-inclusion viz: {cfg['name']} ===")
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    bbox_min = pos_rest.min(0)
    bbox_max = pos_rest.max(0)
    bbox_size = bbox_max - bbox_min

    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size
    incl_radius = cfg["incl_radius_abs"]
    print(f"  N={N}  knife_radius={knife_radius}  incl_center={incl_center.round(3)}  incl_radius={incl_radius}")
    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=cfg["stiff_in"], stiff_out=cfg["stiff_out"],
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    n_incl = int(inclusion_mask.sum())
    print(f"  inclusion: {n_incl}/{N} particles ({n_incl/N*100:.2f}%)")
    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)

    # Truth obs for T_A (calibration) and T_B (held-out)
    theta_truth = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    truth = {}
    for tr in ["A", "B"]:
        traj = make_traj_from_spec(cfg["traj_specs"][tr], bbox_min, bbox_max, T)
        solver.reset_to_rest(pos_rest)
        obs_pos, obs_alpha = gen_obs_per_particle_stiff(
            solver, rest_state, theta_truth, stiff_truth, traj, knife_radius, device,
            tau_truth_np=tau_truth)
        cav = float((obs_alpha[-1] < 0.5).mean())
        truth[tr] = dict(traj=traj, pos=obs_pos, alpha=obs_alpha)
        print(f"  T_{tr}: cavity final = {cav*100:.2f}%")

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in truth["A"]["pos"]]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in truth["A"]["alpha"]]
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)
    x_rest = torch.from_numpy(pos_rest).to(device, dtype=torch.float32)
    H = build_hierarchy(x_rest, M1=512, M2=64, k_neighbors=3)

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

    print("\n[Calibrate] baseline + hierarchical, 30 epochs each", flush=True)
    t0 = time.time()
    stiff_b, tau_b, Lgap_b = calibrate("baseline", ctx, N_epochs=30)
    print(f"  baseline {time.time()-t0:.0f}s  Lgap={Lgap_b:.3e}", flush=True)
    t0 = time.time()
    stiff_h, tau_h, Lgap_h = calibrate("hierarchical", ctx, N_epochs=30)
    print(f"  hierarchical {time.time()-t0:.0f}s  Lgap={Lgap_h:.3e}", flush=True)
    m_b = compute_metrics(stiff_b, stiff_truth, inclusion_mask)
    m_h = compute_metrics(stiff_h, stiff_truth, inclusion_mask)
    print(f"\n  IoU base={m_b['iou']:.3f}  hier={m_h['iou']:.3f}")
    print(f"  stiff_in mean: base {stiff_b[inclusion_mask].mean():.3f}  hier {stiff_h[inclusion_mask].mean():.3f}  (truth 0.8)")
    print(f"  stiff_out mean: base {stiff_b[~inclusion_mask].mean():.3f}  hier {stiff_h[~inclusion_mask].mean():.3f}  (truth 0.3)")

    # Predict cross-traj T_B
    pred_b_pos, pred_b_alpha = predict_on_traj(stiff_b, tau_b, truth["B"]["traj"], ctx)
    pred_h_pos, pred_h_alpha = predict_on_traj(stiff_h, tau_h, truth["B"]["traj"], ctx)
    err_b_pos, _ = err_per_traj(pred_b_pos, truth["B"]["pos"], pred_b_alpha, truth["B"]["alpha"])
    err_h_pos, _ = err_per_traj(pred_h_pos, truth["B"]["pos"], pred_h_alpha, truth["B"]["alpha"])
    print(f"  T_B pos err: base {err_b_pos:.3e}  hier {err_h_pos:.3e}  Δ {(err_h_pos-err_b_pos)/err_b_pos*100:+.2f}%")

    # ===== Figure 1: stiff field xz slice through inclusion =====
    slice_y = float(incl_center[1])
    slice_tol = 0.025
    stiff_vmin = STIFF_LOW
    stiff_vmax = max(stiff_truth.max(), stiff_b.max(), stiff_h.max())
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, vals, ttl in zip(axes,
                                [stiff_truth, stiff_b, stiff_h],
                                ["Truth (incl in=0.8, out=0.3)",
                                 f"Baseline (IoU={m_b['iou']:.3f}, in={stiff_b[inclusion_mask].mean():.2f})",
                                 f"Hierarchical (IoU={m_h['iou']:.3f}, in={stiff_h[inclusion_mask].mean():.2f})"]):
        sc = plot_slice(ax, pos_rest, vals, stiff_vmin, stiff_vmax, 1, slice_y, slice_tol, ttl)
        ax.add_patch(Circle((incl_center[0], incl_center[2]), incl_radius,
                              fill=False, color="red", linewidth=2.5))
        plt.colorbar(sc, ax=ax, shrink=0.7, label="stiff_pp")
    fig1.suptitle(f"Surface inclusion (center at top y={slice_y:.2f}, radius {incl_radius})\n"
                  f"xz slice through inclusion center  ·  T_A cavity {(truth['A']['alpha'][-1] < 0.5).mean()*100:.1f}%",
                  fontsize=13)
    fig1.tight_layout()
    fig1.savefig(str(REPO / "tools" / "pointnetpp" / "viz_surface_stiff.png"), dpi=120, bbox_inches="tight")
    print(f"\nsaved viz_surface_stiff.png")

    # ===== Figure 2: 1D profile through inclusion =====
    # Pick particles in a narrow band along x at center y, center z
    band_y_tol = 0.025
    band_z_tol = 0.025
    band_mask = ((np.abs(pos_rest[:, 1] - slice_y) < band_y_tol)
                  & (np.abs(pos_rest[:, 2] - incl_center[2]) < band_z_tol))
    band_x = pos_rest[band_mask, 0]
    sort_idx = np.argsort(band_x)
    band_x = band_x[sort_idx]
    band_truth = stiff_truth[band_mask][sort_idx]
    band_b = stiff_b[band_mask][sort_idx]
    band_h = stiff_h[band_mask][sort_idx]
    fig2, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(band_x, band_truth, "ko-", label="Truth", lw=2, markersize=5)
    ax.plot(band_x, band_b, "bs--", label=f"Baseline (IoU {m_b['iou']:.3f})", lw=2, alpha=0.8, markersize=4)
    ax.plot(band_x, band_h, "r^--", label=f"Hierarchical (IoU {m_h['iou']:.3f})", lw=2, alpha=0.8, markersize=4)
    # Mark inclusion x extent
    x_lo = incl_center[0] - incl_radius
    x_hi = incl_center[0] + incl_radius
    ax.axvline(x_lo, color="red", linestyle=":", alpha=0.5)
    ax.axvline(x_hi, color="red", linestyle=":", alpha=0.5)
    ax.axvspan(x_lo, x_hi, alpha=0.05, color="red", label="inclusion extent")
    ax.set_xlabel("x position on cube top")
    ax.set_ylabel("stiff_pp recovered")
    ax.set_title(f"Stiff_pp profile along x at top of cube (through inclusion center)\n"
                 f"{int(band_mask.sum())} particles in band, T_B pos err: base {err_b_pos:.2e} → hier {err_h_pos:.2e} "
                 f"({(err_h_pos-err_b_pos)/err_b_pos*100:+.1f}%)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(str(REPO / "tools" / "pointnetpp" / "viz_surface_profile.png"), dpi=130, bbox_inches="tight")
    print("saved viz_surface_profile.png")

    # ===== Figure 3: stiff error map =====
    err_b_field = stiff_b - stiff_truth
    err_h_field = stiff_h - stiff_truth
    err_lim = max(float(np.abs(err_b_field).max()), float(np.abs(err_h_field).max()))
    fig3, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, vals, ttl in zip(axes, [err_b_field, err_h_field],
                                [f"Baseline error (mean |err|={np.abs(err_b_field).mean():.3f})",
                                 f"Hierarchical error (mean |err|={np.abs(err_h_field).mean():.3f})"]):
        sc = plot_slice(ax, pos_rest, vals, -err_lim, err_lim, 1, slice_y, slice_tol, ttl, cmap="seismic")
        ax.add_patch(Circle((incl_center[0], incl_center[2]), incl_radius,
                              fill=False, color="black", linewidth=2.5))
        plt.colorbar(sc, ax=ax, shrink=0.7, label="stiff_pp - truth")
    fig3.suptitle("Per-particle stiff error (red=overestimate, blue=underestimate, black circle=true inclusion)",
                  fontsize=12)
    fig3.tight_layout()
    fig3.savefig(str(REPO / "tools" / "pointnetpp" / "viz_surface_error.png"), dpi=120, bbox_inches="tight")
    print("saved viz_surface_error.png")


if __name__ == "__main__":
    main()
