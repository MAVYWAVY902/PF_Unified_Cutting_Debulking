"""Visualize ACTUAL model behavior: baseline vs hierarchical.

Generates a 3-panel figure showing:
   PANEL 1: stiff_pp field on rest mesh — truth / baseline / hierarchical
   PANEL 2: 2D slice through inclusion plane — clearer view
   PANEL 3: cross-traj prediction error on held-out T_B trajectory
   PANEL 4: deformed mesh at T_B end frame — truth / base / hier predictions

Uses M1_cube setup from multi_mesh_test (proper surface debulk).
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, E402

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day8_stable_inner_loop import gd_inner_loop_stable  # noqa: E402

from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff, compute_metrics,
    inv_sigmoid_to_z, STIFF_LOW, STIFF_HIGH,
)
from tools.pointnetpp.hierarchy import build_hierarchy, reconstruct  # noqa: E402
from tools.pointnetpp.multi_mesh_test import (  # noqa: E402
    MESH_CONFIGS, make_traj_from_spec, predict_on_traj, err_per_traj,
)


def calibrate(mode, ctx, N_epochs=15, lr=0.03, lr_tau=0.01):
    """Calibrate baseline or hierarchical on T_A. Returns final stiff_pp + tau_pp arrays."""
    device = ctx["device"]
    N = ctx["N"]
    stiff_init_np = ctx["stiff_init_np"]
    tau_init_np = ctx["tau_init_np"]
    tau_bound = ctx["tau_bound"]
    theta_3d_np = ctx["theta_3d_np"]
    knife_traj = ctx["knife_traj_A"]
    knife_radius = ctx["knife_radius"]
    rest_state = ctx["rest_state"]
    solver = ctx["solver"]
    obs_pos_t = ctx["obs_pos_A_t"]
    obs_alpha_t = ctx["obs_alpha_A_t"]
    edges_a_t = ctx["edges_a_t"]
    edges_b_t = ctx["edges_b_t"]
    edges_rest_t = ctx["edges_rest_t"]

    theta_torch = torch.tensor(theta_3d_np, dtype=torch.float32, device=device,
                                 requires_grad=False)
    z_L0_stiff = torch.tensor(inv_sigmoid_to_z(stiff_init_np),
                                dtype=torch.float32, device=device, requires_grad=True)
    z_L0_tau = torch.tensor(inv_sigmoid_to_z(tau_init_np, lo=tau_bound[0], hi=tau_bound[1]),
                              dtype=torch.float32, device=device, requires_grad=True)

    if mode == "baseline":
        params = [{"params": [z_L0_stiff], "lr": lr},
                   {"params": [z_L0_tau],   "lr": lr_tau}]
        use_hier = False
        H = None
        z_L1_stiff = z_L2_stiff = z_L1_tau = z_L2_tau = None
    elif mode == "hierarchical":
        H = ctx["H"]
        M1, M2 = H.M1, H.M2
        z_L1_stiff = torch.zeros(M1, dtype=torch.float32, device=device, requires_grad=True)
        z_L2_stiff = torch.zeros(M2, dtype=torch.float32, device=device, requires_grad=True)
        z_L1_tau = torch.zeros(M1, dtype=torch.float32, device=device, requires_grad=True)
        z_L2_tau = torch.zeros(M2, dtype=torch.float32, device=device, requires_grad=True)
        sL1 = 1.0 / math.sqrt(N / M1)
        sL2 = 1.0 / math.sqrt(N / M2)
        params = [
            {"params": [z_L0_stiff], "lr": lr},
            {"params": [z_L1_stiff], "lr": lr * sL1},
            {"params": [z_L2_stiff], "lr": lr * sL2},
            {"params": [z_L0_tau],   "lr": lr_tau},
            {"params": [z_L1_tau],   "lr": lr_tau * sL1},
            {"params": [z_L2_tau],   "lr": lr_tau * sL2},
        ]
        use_hier = True

    opt = torch.optim.Adam(params)
    T = len(knife_traj)
    best_L_gap = float("inf")
    best_stiff = stiff_init_np.copy()
    best_tau = tau_init_np.copy()

    for epoch in range(N_epochs):
        solver.reset_to_rest(ctx["pos_rest"])
        state = rest_state
        L_gaps = []
        for t in range(T):
            if use_hier:
                z_flat_stiff = reconstruct(z_L0_stiff, z_L1_stiff, z_L2_stiff, H)
                z_flat_tau = reconstruct(z_L0_tau, z_L1_tau, z_L2_tau, H)
            else:
                z_flat_stiff = z_L0_stiff
                z_flat_tau = z_L0_tau
            stiff_pp_torch = STIFF_LOW + (STIFF_HIGH - STIFF_LOW) * torch.sigmoid(z_flat_stiff)
            tau_pp_torch = tau_bound[0] + (tau_bound[1] - tau_bound[0]) * torch.sigmoid(z_flat_tau)

            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state, knife_traj[t], knife_radius,
                stiff_pp_torch, tau_pp_torch,
            )
            tau_inner_val = float(tau_pp_torch.mean().item())
            delta_x, delta_dv, _ = gd_inner_loop_stable(
                pos_t, dv_t, obs_pos_t[t], obs_alpha_t[t],
                theta_torch, edges_a_t, edges_b_t, edges_rest_t,
                n_iter=30, lr_pos=0.5, lr_dv_factor=1.0,
                lambda_cav=1.0, lambda_E=0.0,
                s_inner=float(theta_torch[1].item()),
                tau_inner=tau_inner_val,
            )
            L_gap = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
            opt.zero_grad()
            L_gap.backward()
            opt.step()
            L_gaps.append(float(L_gap.item()))

            state = solver_save_state(solver)
            dx_np = delta_x.detach().cpu().numpy()
            ddv_np = delta_dv.detach().cpu().numpy()
            state["pos"] = state["pos"] + dx_np
            state["particle_damage"] = np.maximum(state["particle_damage"] + ddv_np, 0.0)
            new_dv = state["particle_damage"]
            cur_s = float(theta_torch[1].item())
            cur_tau_arr = tau_pp_torch.detach().cpu().numpy()
            state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(
                cur_s * (new_dv - cur_tau_arr), -50, 50)))

        mean_L = float(np.mean(L_gaps))
        with torch.no_grad():
            if use_hier:
                z_flat_stiff_now = reconstruct(z_L0_stiff, z_L1_stiff, z_L2_stiff, H)
                z_flat_tau_now = reconstruct(z_L0_tau, z_L1_tau, z_L2_tau, H)
            else:
                z_flat_stiff_now = z_L0_stiff
                z_flat_tau_now = z_L0_tau
            stiff_now = (STIFF_LOW + (STIFF_HIGH - STIFF_LOW)
                            * torch.sigmoid(z_flat_stiff_now)).cpu().numpy()
            tau_now = (tau_bound[0] + (tau_bound[1] - tau_bound[0])
                          * torch.sigmoid(z_flat_tau_now)).cpu().numpy()
        if mean_L < best_L_gap - 1e-4:
            best_L_gap = mean_L
            best_stiff = stiff_now.copy()
            best_tau = tau_now.copy()
    return best_stiff, best_tau, best_L_gap


def plot_stiff_field_3d(ax, pos, values, vmin, vmax, title, cmap="viridis"):
    """3D scatter colored by `values` (e.g., stiff)."""
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=values, cmap=cmap, s=3, vmin=vmin, vmax=vmax, alpha=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title, fontsize=11)
    return sc


def plot_stiff_slice(ax, pos, values, vmin, vmax, slice_axis, slice_val,
                     slice_tol, title, cmap="viridis"):
    """2D slice in (axis_a, axis_b) plane at slice_axis ≈ slice_val."""
    mask = np.abs(pos[:, slice_axis] - slice_val) < slice_tol
    others = [i for i in [0, 1, 2] if i != slice_axis]
    sc = ax.scatter(pos[mask, others[0]], pos[mask, others[1]],
                       c=values[mask], cmap=cmap, s=12, vmin=vmin, vmax=vmax)
    ax.set_xlabel(f"axis {others[0]} (={['x','y','z'][others[0]]})")
    ax.set_ylabel(f"axis {others[1]} (={['x','y','z'][others[1]]})")
    ax.set_title(title + f"\n(slice {['x','y','z'][slice_axis]}≈{slice_val:.3f}±{slice_tol:.3f}, "
                 f"{int(mask.sum())} pts)", fontsize=10)
    ax.set_aspect("equal")
    return sc


def main():
    cfg = MESH_CONFIGS[0]  # M1_cube — surface debulk setup
    device = "cuda:0"
    T = 10
    print(f"\n=== Visualize behavior: {cfg['name']} ===")
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
    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=cfg["stiff_in"], stiff_out=cfg["stiff_out"],
        center_override=tuple(incl_center),
        radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)
    n_incl = int(inclusion_mask.sum())
    print(f"  N={N}  incl={n_incl} ({n_incl/N*100:.2f}%)")

    # Generate truth obs for T_A and T_B
    theta_truth_np = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    truth = {}
    for tr_name in ["A", "B"]:
        spec = cfg["traj_specs"][tr_name]
        traj = make_traj_from_spec(spec, bbox_min, bbox_max, T)
        solver.reset_to_rest(pos_rest)
        obs_pos, obs_alpha = gen_obs_per_particle_stiff(
            solver, rest_state, theta_truth_np, stiff_truth, traj, knife_radius, device,
            tau_truth_np=tau_truth)
        truth[tr_name] = dict(traj=traj, pos=obs_pos, alpha=obs_alpha)
        cav = float((obs_alpha[-1] < 0.5).mean())
        print(f"  T_{tr_name} cavity={cav:.4f}")

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
        theta_3d_np=theta_truth_np, inclusion_mask=inclusion_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H, pos_rest=pos_rest,
    )

    # Calibrate
    print("\n[Calibrate baseline]")
    t0 = time.time()
    stiff_b, tau_b, Lgap_b = calibrate("baseline", ctx, N_epochs=15)
    print(f"  done in {time.time()-t0:.0f}s  Lgap_best={Lgap_b:.3e}")
    print("\n[Calibrate hierarchical]")
    t0 = time.time()
    stiff_h, tau_h, Lgap_h = calibrate("hierarchical", ctx, N_epochs=15)
    print(f"  done in {time.time()-t0:.0f}s  Lgap_best={Lgap_h:.3e}")
    m_b = compute_metrics(stiff_b, stiff_truth, inclusion_mask)
    m_h = compute_metrics(stiff_h, stiff_truth, inclusion_mask)
    print(f"\n  IoU base={m_b['iou']:.3f}  hier={m_h['iou']:.3f}")

    # Predict on T_B (held-out)
    print("\n[Predict on T_B held-out]")
    pred_b_pos, pred_b_alpha = predict_on_traj(stiff_b, tau_b, truth["B"]["traj"], ctx)
    pred_h_pos, pred_h_alpha = predict_on_traj(stiff_h, tau_h, truth["B"]["traj"], ctx)
    err_b_pos, err_b_a = err_per_traj(pred_b_pos, truth["B"]["pos"], pred_b_alpha, truth["B"]["alpha"])
    err_h_pos, err_h_a = err_per_traj(pred_h_pos, truth["B"]["pos"], pred_h_alpha, truth["B"]["alpha"])
    print(f"  baseline    : pos_err {err_b_pos:.3e}   α_err {err_b_a:.3e}")
    print(f"  hierarchical: pos_err {err_h_pos:.3e}   α_err {err_h_a:.3e}")

    # ===== FIGURE 1: stiff_pp field comparison =====
    print("\n[Render Figure 1: stiff field comparison]")
    stiff_vmin = STIFF_LOW
    stiff_vmax = max(stiff_truth.max(), stiff_b.max(), stiff_h.max())
    fig1 = plt.figure(figsize=(15, 10))

    # Row 1: 3D scatter
    for j, (vals, ttl) in enumerate([(stiff_truth, "Truth stiff (inclusion 0.8, outside 0.3)"),
                                       (stiff_b, f"Baseline calibrated (IoU {m_b['iou']:.3f})"),
                                       (stiff_h, f"Hierarchical calibrated (IoU {m_h['iou']:.3f})")]):
        ax = fig1.add_subplot(2, 3, j + 1, projection="3d")
        sc = plot_stiff_field_3d(ax, pos_rest, vals, stiff_vmin, stiff_vmax, ttl)
        fig1.colorbar(sc, ax=ax, shrink=0.6, label="stiff_pp")

    # Row 2: 2D slice through inclusion plane (y = incl_center_y)
    slice_axis = 1  # y
    slice_val = float(incl_center[1])
    slice_tol = 0.015  # ~5% of bbox dim
    for j, (vals, ttl) in enumerate([(stiff_truth, "Truth"),
                                       (stiff_b, "Baseline"),
                                       (stiff_h, "Hierarchical")]):
        ax = fig1.add_subplot(2, 3, j + 4)
        sc = plot_stiff_slice(ax, pos_rest, vals, stiff_vmin, stiff_vmax,
                                slice_axis, slice_val, slice_tol, ttl + " (xz slice)")
        # Draw inclusion circle
        from matplotlib.patches import Circle
        c = Circle((incl_center[0], incl_center[2]), incl_radius,
                     fill=False, color="red", linewidth=2)
        ax.add_patch(c)
        fig1.colorbar(sc, ax=ax, shrink=0.6, label="stiff_pp")
    fig1.suptitle(f"Recovered stiff_pp field on M1_cube (N={N})\n"
                  f"Inclusion center {incl_center.round(3).tolist()}, radius {incl_radius}",
                  fontsize=13)
    fig1.tight_layout()
    p1 = REPO / "tools" / "pointnetpp" / "viz_stiff_field.png"
    fig1.savefig(str(p1), dpi=130, bbox_inches="tight")
    print(f"  saved {p1}")

    # ===== FIGURE 2: stiff error maps =====
    print("\n[Render Figure 2: stiff error maps]")
    err_b = stiff_b - stiff_truth
    err_h = stiff_h - stiff_truth
    err_vmin = min(err_b.min(), err_h.min())
    err_vmax = max(err_b.max(), err_h.max())
    err_lim = max(abs(err_vmin), abs(err_vmax))
    fig2 = plt.figure(figsize=(15, 5))
    for j, (vals, ttl) in enumerate([(err_b, "Baseline stiff_b - truth"),
                                       (err_h, "Hierarchical stiff_h - truth")]):
        ax = fig2.add_subplot(1, 2, j + 1)
        sc = plot_stiff_slice(ax, pos_rest, vals, -err_lim, err_lim,
                                slice_axis, slice_val, slice_tol,
                                ttl + f" (mean |err|={np.abs(vals).mean():.3f})",
                                cmap="seismic")
        from matplotlib.patches import Circle
        c = Circle((incl_center[0], incl_center[2]), incl_radius,
                     fill=False, color="black", linewidth=2)
        ax.add_patch(c)
        fig2.colorbar(sc, ax=ax, shrink=0.6, label="stiff error")
    fig2.suptitle("Stiff recovery error per particle (xz slice through inclusion)\n"
                  "Red = over-estimated, Blue = under-estimated, Black circle = true inclusion",
                  fontsize=12)
    fig2.tight_layout()
    p2 = REPO / "tools" / "pointnetpp" / "viz_stiff_error.png"
    fig2.savefig(str(p2), dpi=130, bbox_inches="tight")
    print(f"  saved {p2}")

    # ===== FIGURE 3: T_B prediction quality =====
    print("\n[Render Figure 3: T_B prediction error map]")
    # Per-particle position error at final frame
    truth_pos_final = truth["B"]["pos"][-1]
    truth_alpha_final = truth["B"]["alpha"][-1]
    pred_b_pos_final = pred_b_pos[-1]
    pred_h_pos_final = pred_h_pos[-1]
    pred_b_alpha_final = pred_b_alpha[-1]
    pred_h_alpha_final = pred_h_alpha[-1]

    perp_err_b = np.linalg.norm(pred_b_pos_final - truth_pos_final, axis=1)
    perp_err_h = np.linalg.norm(pred_h_pos_final - truth_pos_final, axis=1)
    p_lim = max(perp_err_b.max(), perp_err_h.max())

    fig3 = plt.figure(figsize=(16, 5))
    for j, (vals, ttl) in enumerate([(perp_err_b, f"Baseline T_B pos error (mean {perp_err_b.mean():.3e})"),
                                       (perp_err_h, f"Hier T_B pos error (mean {perp_err_h.mean():.3e})")]):
        ax = fig3.add_subplot(1, 3, j + 1)
        sc = plot_stiff_slice(ax, pos_rest, vals, 0, p_lim,
                                slice_axis, slice_val, slice_tol,
                                ttl, cmap="hot_r")
        from matplotlib.patches import Circle
        c = Circle((incl_center[0], incl_center[2]), incl_radius,
                     fill=False, color="cyan", linewidth=2)
        ax.add_patch(c)
        fig3.colorbar(sc, ax=ax, shrink=0.6, label="|pred - truth| m")
    # Third panel: hier - baseline (negative = hier better)
    diff = perp_err_h - perp_err_b
    ax = fig3.add_subplot(1, 3, 3)
    d_lim = float(np.abs(diff).max())
    sc = plot_stiff_slice(ax, pos_rest, diff, -d_lim, d_lim,
                            slice_axis, slice_val, slice_tol,
                            f"Hier - Baseline (blue = hier better)\nmean={diff.mean():.3e}",
                            cmap="seismic")
    from matplotlib.patches import Circle
    c = Circle((incl_center[0], incl_center[2]), incl_radius,
                 fill=False, color="black", linewidth=2)
    ax.add_patch(c)
    fig3.colorbar(sc, ax=ax, shrink=0.6, label="Δ error m")
    fig3.suptitle(f"Cross-trajectory prediction on T_B (held-out, NOT used in calibration)\n"
                  f"baseline mean pos err {err_b_pos:.3e} m, hier {err_h_pos:.3e} m → "
                  f"hier {(err_h_pos-err_b_pos)/err_b_pos*100:+.1f}% relative",
                  fontsize=12)
    fig3.tight_layout()
    p3 = REPO / "tools" / "pointnetpp" / "viz_traj_prediction.png"
    fig3.savefig(str(p3), dpi=130, bbox_inches="tight")
    print(f"  saved {p3}")

    # ===== FIGURE 4: cavity time series =====
    print("\n[Render Figure 4: cavity evolution T_B]")
    cav_truth = [float((truth["B"]["alpha"][t] < 0.5).mean()) for t in range(T)]
    cav_b = [float((pred_b_alpha[t] < 0.5).mean()) for t in range(T)]
    cav_h = [float((pred_h_alpha[t] < 0.5).mean()) for t in range(T)]
    fig4, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(T), cav_truth, "o-", label="Truth (T_B)", lw=2, color="black")
    ax.plot(range(T), cav_b, "s--", label="Baseline pred", lw=2, color="tab:blue")
    ax.plot(range(T), cav_h, "^--", label="Hierarchical pred", lw=2, color="tab:red")
    ax.set_xlabel("frame")
    ax.set_ylabel("cavity fraction (particles with α < 0.5)")
    ax.set_title(f"Cavity evolution on held-out trajectory T_B\n(calibrated only on T_A)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig4.tight_layout()
    p4 = REPO / "tools" / "pointnetpp" / "viz_cavity_evolution.png"
    fig4.savefig(str(p4), dpi=130, bbox_inches="tight")
    print(f"  saved {p4}")

    print(f"\nAll figures saved under tools/pointnetpp/viz_*.png")


if __name__ == "__main__":
    main()
