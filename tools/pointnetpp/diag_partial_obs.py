"""SYNTH partial-obs test — simulate phantom regime by masking obs.

Hypothesis: L_gap (Liang inner loop) framework was designed for partial obs
(D405 surface camera). In our synth FULL OBS tests it was overhead. But in
partial obs, the L_E + Mode C machinery should propagate surface signal to
unobserved interior particles, which direct L_pos can't.

Test on M1_cube buried inclusion (interior particles invisible to camera).
Mask obs to particles with y > 0.28 (top ~7% layer, D405-like coverage).

4 conditions × 60 epochs:
   A. Baseline + L_pos_partial (surface only, NO inner loop, NO L_E)
   B. Baseline + L_pos_partial + L_smooth (edge regularizer)
   C. Baseline + Liang inner loop (L_pos surface + L_E + Mode C)
   D. Hier + Liang inner loop

Critical metric: recovery on INTERIOR inclusion particles (the ones obs can't see).
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
from tools.day8_stable_inner_loop import gd_inner_loop_stable  # noqa: E402

from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff, compute_metrics,
    inv_sigmoid_to_z, STIFF_LOW, STIFF_HIGH,
)
from tools.pointnetpp.hierarchy import build_hierarchy, reconstruct  # noqa: E402
from tools.pointnetpp.multi_mesh_test import (  # noqa: E402
    MESH_CONFIGS, make_traj_from_spec, predict_on_traj, err_per_traj,
)


def calibrate(condition, ctx, N_epochs=60, lr=0.03, lr_tau=0.01,
              lambda_smooth=0.0, lambda_E=0.1, s_inner_override=None,
              lambda_outer_cav=0.0,
              lr_schedule=None, return_best=False,
              multi_traj_obs=None):
    """Calibrate with chosen condition.

    condition: 'L_pos_partial', 'L_pos_partial+L_smooth', 'Liang_inner', 'Hier+Liang'
    s_inner_override: inner GD sigmoid sharpness override (Day 3c trick).
    lambda_outer_cav: weight for outer cavity loss = sum_surf (α_sim − α_obs)².
    lr_schedule: optional list of (epoch_threshold, lr_multiplier) tuples for
        step decay. E.g., [(30, 0.2), (60, 0.2)] = lr×0.2 at epoch 30 + 60.
    return_best: if True, return tau corresponding to lowest mean_L epoch.
    multi_traj_obs: optional list of (traj, obs_pos_t, obs_alpha_t) tuples.
        If provided, calibration loops over ALL trajectories per epoch
        (over-determined: forces tau to satisfy multiple cut patterns).
        If None, uses ctx["knife_traj_A"] + obs_pos_A_t + obs_alpha_A_t (legacy).
        2026-06-01 v8 fix: multi-trajectory regularization to escape T_A local minimum.
    """
    device = ctx["device"]
    N = ctx["N"]
    stiff_init_np = ctx["stiff_init_np"]
    tau_init_np = ctx["tau_init_np"]
    tau_bound = ctx["tau_bound"]
    theta_3d_np = ctx["theta_truth"]
    knife_traj = ctx["knife_traj_A"]
    knife_radius = ctx["knife_radius"]
    rest_state = ctx["rest_state"]
    solver = ctx["solver"]
    obs_pos_t = ctx["obs_pos_A_t"]
    obs_alpha_t = ctx["obs_alpha_A_t"]
    edges_a_t = ctx["edges_a_t"]
    edges_b_t = ctx["edges_b_t"]
    edges_rest_t = ctx["edges_rest_t"]
    inclusion_mask = ctx["inclusion_mask"]
    stiff_truth = ctx["stiff_truth"]
    surface_idx = ctx["surface_idx"]
    surface_mask = ctx["surface_mask"]

    theta_torch = torch.tensor(theta_3d_np, dtype=torch.float32, device=device,
                                 requires_grad=False)

    # Hier vs baseline parameterization
    use_hier = "Hier" in condition
    z_L0_stiff = torch.tensor(inv_sigmoid_to_z(stiff_init_np),
                                dtype=torch.float32, device=device, requires_grad=True)
    z_L0_tau = torch.tensor(inv_sigmoid_to_z(tau_init_np, lo=tau_bound[0], hi=tau_bound[1]),
                              dtype=torch.float32, device=device, requires_grad=True)
    if use_hier:
        H = ctx["H"]
        M1, M2 = H.M1, H.M2
        sL1 = 1.0 / math.sqrt(N / M1)
        sL2 = 1.0 / math.sqrt(N / M2)
        z_L1_stiff = torch.zeros(M1, dtype=torch.float32, device=device, requires_grad=True)
        z_L2_stiff = torch.zeros(M2, dtype=torch.float32, device=device, requires_grad=True)
        z_L1_tau = torch.zeros(M1, dtype=torch.float32, device=device, requires_grad=True)
        z_L2_tau = torch.zeros(M2, dtype=torch.float32, device=device, requires_grad=True)
        params = [
            {"params": [z_L0_stiff], "lr": lr},
            {"params": [z_L1_stiff], "lr": lr * sL1},
            {"params": [z_L2_stiff], "lr": lr * sL2},
            {"params": [z_L0_tau], "lr": lr_tau},
            {"params": [z_L1_tau], "lr": lr_tau * sL1},
            {"params": [z_L2_tau], "lr": lr_tau * sL2},
        ]
    else:
        z_L1_stiff = z_L2_stiff = z_L1_tau = z_L2_tau = None
        params = [
            {"params": [z_L0_stiff], "lr": lr},
            {"params": [z_L0_tau], "lr": lr_tau},
        ]
    opt = torch.optim.Adam(params)

    use_inner_loop = "Liang" in condition

    # Best-checkpoint tracking (anti-overshoot for high lr_tau regime)
    best_mean_L = float("inf")
    best_stiff = None
    best_tau = None

    # Step LR scheduler (warm-decay)
    if lr_schedule is not None:
        # Capture initial lr per group; on threshold reach, multiply
        initial_lrs = [pg["lr"] for pg in opt.param_groups]

    # Build trajectory list — single (legacy) or multi
    if multi_traj_obs is None:
        traj_obs_list = [(knife_traj, obs_pos_t, obs_alpha_t)]
    else:
        traj_obs_list = multi_traj_obs

    history = []
    for epoch in range(N_epochs):
        # Apply lr schedule (multiply by all thresholds passed)
        if lr_schedule is not None:
            mult = 1.0
            for thresh, m in lr_schedule:
                if epoch >= thresh:
                    mult *= m
            for k, pg in enumerate(opt.param_groups):
                pg["lr"] = initial_lrs[k] * mult
        L_epoch = []
        # Loop over trajectories — each gives independent gradient steps
        for traj_idx, (cur_traj, cur_obs_pos_t, cur_obs_alpha_t) in enumerate(traj_obs_list):
            solver.reset_to_rest(ctx["pos_rest"])
            state = rest_state
            T = len(cur_traj)
            for t in range(T):
                if use_hier:
                    z_flat_stiff = reconstruct(z_L0_stiff, z_L1_stiff, z_L2_stiff, ctx["H"])
                    z_flat_tau = reconstruct(z_L0_tau, z_L1_tau, z_L2_tau, ctx["H"])
                else:
                    z_flat_stiff = z_L0_stiff
                    z_flat_tau = z_L0_tau
                stiff_pp_torch = STIFF_LOW + (STIFF_HIGH - STIFF_LOW) * torch.sigmoid(z_flat_stiff)
                tau_pp_torch = tau_bound[0] + (tau_bound[1] - tau_bound[0]) * torch.sigmoid(z_flat_tau)

                alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                    theta_torch, solver, state, cur_traj[t], knife_radius,
                    stiff_pp_torch, tau_pp_torch,
                )

                if use_inner_loop:
                    # Liang's inner GD with surface_idx + L_E
                    tau_inner_val = float(tau_pp_torch.mean().item())
                    s_inner_val = (float(theta_torch[1].item())
                                      if s_inner_override is None
                                      else float(s_inner_override))
                    delta_x, delta_dv, _ = gd_inner_loop_stable(
                        pos_t, dv_t, cur_obs_pos_t[t], cur_obs_alpha_t[t],
                        theta_torch, edges_a_t, edges_b_t, edges_rest_t,
                        n_iter=30, lr_pos=0.5, lr_dv_factor=1.0,
                        lambda_cav=1.0, lambda_E=lambda_E,    # L_E ON
                        s_inner=s_inner_val,
                        tau_inner=tau_inner_val,
                        surface_idx=surface_idx,              # surface only for L_pos
                    )
                    L = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
                else:
                    # Direct L_pos on surface particles only
                    L = ((pos_t[surface_idx] - cur_obs_pos_t[t][surface_idx]) ** 2).sum()
                    if lambda_smooth > 0:
                        # L_smooth on per-particle stiff
                        neighbor_diff = stiff_pp_torch[edges_a_t] - stiff_pp_torch[edges_b_t]
                        L_smooth = (neighbor_diff ** 2).sum()
                        L = L + lambda_smooth * L_smooth
                    if lambda_outer_cav > 0:
                        # Direct outer cavity loss — per-particle alpha mismatch
                        L_cav_outer = ((alpha_t[surface_idx] - cur_obs_alpha_t[t][surface_idx]) ** 2).sum()
                        L = L + lambda_outer_cav * L_cav_outer

                opt.zero_grad()
                L.backward()
                opt.step()
                L_epoch.append(float(L.item()))

                # Mode C inject only if Liang inner loop used
                if use_inner_loop:
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
                else:
                    state = solver_save_state(solver)

        mean_L = float(np.mean(L_epoch))
        with torch.no_grad():
            if use_hier:
                z_flat_stiff_now = reconstruct(z_L0_stiff, z_L1_stiff, z_L2_stiff, ctx["H"])
                z_flat_tau_now = reconstruct(z_L0_tau, z_L1_tau, z_L2_tau, ctx["H"])
            else:
                z_flat_stiff_now = z_L0_stiff
                z_flat_tau_now = z_L0_tau
            stiff_now = (STIFF_LOW + (STIFF_HIGH - STIFF_LOW)
                            * torch.sigmoid(z_flat_stiff_now)).cpu().numpy()
            tau_now = (tau_bound[0] + (tau_bound[1] - tau_bound[0])
                          * torch.sigmoid(z_flat_tau_now)).cpu().numpy()
        # Compute metrics on interior (NOT on surface — that's already observed)
        interior_mask = ~surface_mask
        interior_inclusion_mask = inclusion_mask & interior_mask
        m = compute_metrics(stiff_now, stiff_truth, inclusion_mask)

        L1_interior_in = float(np.abs(stiff_now[interior_inclusion_mask] -
                                          stiff_truth[interior_inclusion_mask]).mean()
                                  if interior_inclusion_mask.sum() > 0 else 0.0)
        L1_interior_out = float(np.abs(stiff_now[interior_mask & ~inclusion_mask] -
                                           stiff_truth[interior_mask & ~inclusion_mask]).mean())

        history.append(dict(
            epoch=epoch, mean_L=mean_L, iou=m["iou"],
            stiff_in=float(stiff_now[inclusion_mask].mean()),
            stiff_out=float(stiff_now[~inclusion_mask].mean()),
            stiff_in_interior=float(stiff_now[interior_inclusion_mask].mean()) if interior_inclusion_mask.sum() > 0 else 0.0,
            L1_interior_in=L1_interior_in,
            L1_interior_out=L1_interior_out,
        ))

        # Best-checkpoint tracking (v7 fix #3)
        if return_best and mean_L < best_mean_L:
            best_mean_L = mean_L
            best_stiff = stiff_now.copy()
            best_tau = tau_now.copy()

    if return_best and best_tau is not None:
        return best_stiff, best_tau, history
    return stiff_now, tau_now, history


def main():
    cfg = MESH_CONFIGS[0]
    device = "cuda:0"
    T = 10
    N_epochs = 60
    print(f"\n=== Synth partial-obs phantom-simulated test ===")
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
        pos_rest, stiff_in=cfg["stiff_in"], stiff_out=cfg["stiff_out"],
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)

    # ===== Partial obs mask — simulate D405 camera coverage =====
    # Top layer of cube particles only (y > 0.28 = top 7% of bbox)
    surface_y_threshold = 0.28
    surface_mask = pos_rest[:, 1] > surface_y_threshold
    n_surface = int(surface_mask.sum())
    n_interior = N - n_surface
    n_incl = int(inclusion_mask.sum())
    n_incl_surface = int((inclusion_mask & surface_mask).sum())
    n_incl_interior = int((inclusion_mask & ~surface_mask).sum())

    print(f"  N={N}  total inclusion={n_incl} ({n_incl/N*100:.2f}%)")
    print(f"  Surface mask: y > {surface_y_threshold}")
    print(f"    Surface particles: {n_surface} ({n_surface/N*100:.1f}%)  ← simulated D405 view")
    print(f"    Interior particles: {n_interior} ({n_interior/N*100:.1f}%)  ← invisible")
    print(f"    Inclusion in surface: {n_incl_surface}/{n_incl} ({n_incl_surface/n_incl*100:.1f}%)")
    print(f"    Inclusion in interior: {n_incl_interior}/{n_incl} ({n_incl_interior/n_incl*100:.1f}%)  ← TEST TARGET")

    if n_incl_interior == 0:
        print("  ⚠ No inclusion in interior — partial obs sees everything, test moot")
        return

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

    conditions = [
        ("A: L_pos_partial",          dict(condition="L_pos_partial",     lambda_smooth=0.0, lambda_E=0.0)),
        ("B: +L_smooth_1e-4",         dict(condition="L_pos_partial",     lambda_smooth=1e-4, lambda_E=0.0)),
        ("C: +L_smooth_1e-2",         dict(condition="L_pos_partial",     lambda_smooth=1e-2, lambda_E=0.0)),
        ("D: Liang inner loop, λE=0", dict(condition="Liang_inner",       lambda_smooth=0.0, lambda_E=0.0)),
        ("E: Liang inner, λE=1e-4",   dict(condition="Liang_inner",       lambda_smooth=0.0, lambda_E=1e-4)),
        ("F: Liang inner, λE=1e-2",   dict(condition="Liang_inner",       lambda_smooth=0.0, lambda_E=1e-2)),
        ("G: Hier+Liang, λE=1e-4",    dict(condition="Hier+Liang_inner",  lambda_smooth=0.0, lambda_E=1e-4)),
    ]
    results = {}
    for label, kwargs in conditions:
        print(f"\n--- {label} ---", flush=True)
        t0 = time.time()
        stiff, tau, hist = calibrate(ctx=ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01, **kwargs)
        elapsed = time.time() - t0
        m = compute_metrics(stiff, stiff_truth, inclusion_mask)
        # Cross-traj T_B
        pred_pos, pred_alpha = predict_on_traj(stiff, tau, truth["B"]["traj"], ctx)
        err_pos, err_a = err_per_traj(pred_pos, truth["B"]["pos"], pred_alpha, truth["B"]["alpha"])
        print(f"  done {elapsed:.0f}s  stiff_in={hist[-1]['stiff_in']:.3f}  "
              f"interior_in={hist[-1]['stiff_in_interior']:.3f}  IoU={m['iou']:.3f}  "
              f"L1_int_in={hist[-1]['L1_interior_in']:.3f}  T_B={err_pos:.3e}")
        results[label] = dict(stiff=stiff, tau=tau, hist=hist, iou=m["iou"],
                                T_B_pos=err_pos, T_B_alpha=err_a)

    # Summary
    print(f"\n{'='*100}")
    print(f"PARTIAL-OBS PHANTOM-SIM SUMMARY (60 epochs, surface coverage {n_surface/N*100:.1f}%)")
    print(f"{'='*100}")
    print(f"  Truth: stiff_in=0.8 (inclusion mostly INTERIOR: {n_incl_interior}/{n_incl} particles)")
    print()
    print(f"  {'Condition':28s}  {'stiff_in':>10s}  {'interior_in':>13s}  {'L1_int_in':>10s}  "
          f"{'IoU':>6s}  {'T_B err':>9s}")
    for label in results:
        r = results[label]; h = r["hist"][-1]
        print(f"  {label:28s}  {h['stiff_in']:>10.3f}  {h['stiff_in_interior']:>13.3f}  "
              f"{h['L1_interior_in']:>10.3f}  {r['iou']:>6.3f}  {r['T_B_pos']:>9.3e}")

    print(f"\n  Interpretation:")
    print(f"  - interior_in close to 0.8 → calibration successfully propagated to invisible interior")
    print(f"  - L1_int_in low → interior recovery accurate")
    print(f"  - L_pos_partial alone (A) should FAIL — no signal for interior")
    print(f"  - Liang inner loop (C) should SUCCEED — L_E propagates surface to interior")
    print(f"  - Hier + Liang (D) tests if hier adds value in partial obs regime")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = list(range(N_epochs))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, label in enumerate(results):
        hist = results[label]["hist"]
        c = colors[i]
        axes[0, 0].plot(epochs, [h["stiff_in_interior"] for h in hist], label=label, color=c)
        axes[0, 1].plot(epochs, [h["L1_interior_in"] for h in hist], label=label, color=c)
        axes[1, 0].plot(epochs, [h["stiff_in"] for h in hist], label=label, color=c)
        axes[1, 1].plot(epochs, [h["iou"] for h in hist], label=label, color=c)
    axes[0, 0].axhline(0.8, color="k", linestyle="--", label="truth")
    axes[0, 0].axhline(0.5, color="gray", linestyle=":", label="init")
    axes[0, 0].set_xlabel("epoch"); axes[0, 0].set_ylabel("stiff_in mean (interior only)")
    axes[0, 0].set_title("INTERIOR inclusion recovery (critical metric)")
    axes[0, 0].legend(fontsize=9); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].set_xlabel("epoch"); axes[0, 1].set_ylabel("L1 error interior")
    axes[0, 1].set_title("L1 error on interior inclusion particles")
    axes[0, 1].legend(fontsize=9); axes[0, 1].grid(alpha=0.3)
    axes[1, 0].axhline(0.8, color="k", linestyle="--")
    axes[1, 0].set_xlabel("epoch"); axes[1, 0].set_ylabel("stiff_in mean (all inclusion)")
    axes[1, 0].set_title("All inclusion recovery (surface + interior)")
    axes[1, 0].legend(fontsize=9); axes[1, 0].grid(alpha=0.3)
    axes[1, 1].set_xlabel("epoch"); axes[1, 1].set_ylabel("IoU")
    axes[1, 1].set_title("IoU")
    axes[1, 1].legend(fontsize=9); axes[1, 1].grid(alpha=0.3)
    fig.suptitle(f"Synth partial-obs phantom-sim test\n"
                  f"Surface obs {n_surface}/{N} particles ({n_surface/N*100:.1f}%), "
                  f"inclusion: {n_incl_surface} surface + {n_incl_interior} interior",
                  fontsize=13)
    fig.tight_layout()
    out_path = REPO / "tools" / "pointnetpp" / "diag_partial_obs.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
