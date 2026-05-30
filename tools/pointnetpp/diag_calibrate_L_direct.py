"""Decisive test: Adam calibration with L_direct vs L_gap.

Hypothesis (from diag_loss_landscape.py):
   L_direct = ||sim_pos - obs_pos||² + λ·||sim_α - obs_α||²
      → argmin AT truth (stiff_in 0.80) ✓ verified
   L_gap (current, inner GD residual)
      → argmin AT wrong place (stiff_in 0.30) ✗ broken

If L_direct is truly the right loss:
   - Adam should push stiff_in from init 0.5 toward truth 0.8
   - stiff_in recovery should jump from ~0.48 to ~0.7+
   - IoU should jump from ~0.05 to ~0.5+

This script runs both losses side by side on identical setup, reports:
   - Per-epoch trajectory of stiff_in mean, IoU, loss
   - Final stiff field comparison
   - Cross-trajectory prediction on T_B
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
from tools.pointnetpp.multi_mesh_test import (  # noqa: E402
    MESH_CONFIGS, make_traj_from_spec, predict_on_traj, err_per_traj,
)


def calibrate_with_loss(loss_kind, ctx, N_epochs=30, lr=0.03, lr_tau=0.01,
                          lambda_alpha=0.1, verbose_every=5):
    """Calibrate per-particle stiff + tau with chosen loss formulation.

    loss_kind: "L_gap" (current, inner GD residual) or "L_direct" (||sim - obs||²)
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

    theta_torch = torch.tensor(theta_3d_np, dtype=torch.float32, device=device,
                                 requires_grad=False)
    z_stiff = torch.tensor(inv_sigmoid_to_z(stiff_init_np),
                              dtype=torch.float32, device=device, requires_grad=True)
    z_tau = torch.tensor(inv_sigmoid_to_z(tau_init_np, lo=tau_bound[0], hi=tau_bound[1]),
                            dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([
        {"params": [z_stiff], "lr": lr},
        {"params": [z_tau], "lr": lr_tau},
    ])

    T = len(knife_traj)
    history = []

    for epoch in range(N_epochs):
        solver.reset_to_rest(ctx["pos_rest"])
        state = rest_state
        L_epoch = []
        for t in range(T):
            stiff_pp_torch = STIFF_LOW + (STIFF_HIGH - STIFF_LOW) * torch.sigmoid(z_stiff)
            tau_pp_torch = tau_bound[0] + (tau_bound[1] - tau_bound[0]) * torch.sigmoid(z_tau)
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state, knife_traj[t], knife_radius,
                stiff_pp_torch, tau_pp_torch,
            )

            if loss_kind == "L_gap":
                tau_inner_val = float(tau_pp_torch.mean().item())
                delta_x, delta_dv, _ = gd_inner_loop_stable(
                    pos_t, dv_t, obs_pos_t[t], obs_alpha_t[t],
                    theta_torch, edges_a_t, edges_b_t, edges_rest_t,
                    n_iter=30, lr_pos=0.5, lr_dv_factor=1.0,
                    lambda_cav=1.0, lambda_E=0.0,
                    s_inner=float(theta_torch[1].item()),
                    tau_inner=tau_inner_val,
                )
                L = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
            elif loss_kind == "L_direct":
                L_pos = ((pos_t - obs_pos_t[t]) ** 2).sum()
                L_alpha = ((alpha_t - obs_alpha_t[t]) ** 2).sum()
                L = L_pos + lambda_alpha * L_alpha
            else:
                raise ValueError(loss_kind)

            opt.zero_grad()
            L.backward()
            opt.step()
            L_epoch.append(float(L.item()))

            # Mode C inject (only for L_gap; L_direct uses raw sim chain)
            if loss_kind == "L_gap":
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
                # L_direct: chain raw sim state without correction
                state = solver_save_state(solver)

        mean_L = float(np.mean(L_epoch))
        with torch.no_grad():
            stiff_now = (STIFF_LOW + (STIFF_HIGH - STIFF_LOW)
                            * torch.sigmoid(z_stiff)).cpu().numpy()
            tau_now = (tau_bound[0] + (tau_bound[1] - tau_bound[0])
                          * torch.sigmoid(z_tau)).cpu().numpy()
        m = compute_metrics(stiff_now, stiff_truth, inclusion_mask)
        history.append(dict(
            epoch=epoch, mean_L=mean_L, iou=m["iou"],
            stiff_in=float(stiff_now[inclusion_mask].mean()),
            stiff_out=float(stiff_now[~inclusion_mask].mean()),
            tau_in=float(tau_now[inclusion_mask].mean()),
            tau_out=float(tau_now[~inclusion_mask].mean()),
        ))
        if epoch % verbose_every == 0 or epoch == N_epochs - 1:
            print(f"    ep {epoch:3d}: L={mean_L:.3e}  IoU={m['iou']:.3f}  "
                  f"stiff_in={history[-1]['stiff_in']:.3f}  "
                  f"stiff_out={history[-1]['stiff_out']:.3f}  "
                  f"tau_in={history[-1]['tau_in']:.3f}  "
                  f"tau_out={history[-1]['tau_out']:.3f}", flush=True)

    return stiff_now, tau_now, history


def main():
    cfg = MESH_CONFIGS[0]
    device = "cuda:0"
    T = 10
    print(f"\n=== Decisive test: L_gap vs L_direct on {cfg['name']} ===")
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
    n_incl = int(inclusion_mask.sum())
    print(f"  N={N}  incl={n_incl} ({n_incl/N*100:.2f}%)")
    print(f"  Truth: stiff_in=0.8, stiff_out=0.3, tau_in={cfg['tau_in']}, tau_out={cfg['tau_out']}")
    print(f"  Init:  stiff=0.5, tau=0.30 (uniform)")

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
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64), dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64), dtype=torch.long, device=device)
    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)

    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=truth["A"]["traj"], knife_radius=knife_radius,
        pos_rest=pos_rest, theta_truth=theta_truth,
        stiff_init_np=stiff_init, stiff_truth=stiff_truth,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        inclusion_mask=inclusion_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
    )

    print(f"\n{'='*72}")
    print(f"RUN 1: L_gap (current, broken)")
    print(f"{'='*72}")
    t0 = time.time()
    stiff_gap, tau_gap, hist_gap = calibrate_with_loss("L_gap", ctx, N_epochs=30, verbose_every=5)
    print(f"  elapsed {time.time()-t0:.0f}s")

    print(f"\n{'='*72}")
    print(f"RUN 2: L_direct (||sim - obs||² + 0.1·||α - α_obs||²)")
    print(f"{'='*72}")
    t0 = time.time()
    stiff_dir, tau_dir, hist_dir = calibrate_with_loss("L_direct", ctx, N_epochs=30, verbose_every=5)
    print(f"  elapsed {time.time()-t0:.0f}s")

    # Predict cross-traj T_B
    print(f"\n{'='*72}")
    print(f"Cross-trajectory T_B prediction")
    print(f"{'='*72}")
    ctx["theta_3d_np"] = theta_truth
    pred_gap_pos, pred_gap_alpha = predict_on_traj(stiff_gap, tau_gap, truth["B"]["traj"], ctx)
    pred_dir_pos, pred_dir_alpha = predict_on_traj(stiff_dir, tau_dir, truth["B"]["traj"], ctx)
    err_gap_pos, err_gap_a = err_per_traj(pred_gap_pos, truth["B"]["pos"], pred_gap_alpha, truth["B"]["alpha"])
    err_dir_pos, err_dir_a = err_per_traj(pred_dir_pos, truth["B"]["pos"], pred_dir_alpha, truth["B"]["alpha"])
    print(f"  L_gap:    T_B pos_err = {err_gap_pos:.3e}   α_err = {err_gap_a:.3e}")
    print(f"  L_direct: T_B pos_err = {err_dir_pos:.3e}   α_err = {err_dir_a:.3e}")
    pos_rel = (err_dir_pos - err_gap_pos) / err_gap_pos * 100
    print(f"  L_direct vs L_gap pos err: {pos_rel:+.2f}% (negative = L_direct better)")

    # Summary
    m_gap = compute_metrics(stiff_gap, stiff_truth, inclusion_mask)
    m_dir = compute_metrics(stiff_dir, stiff_truth, inclusion_mask)
    print(f"\n{'='*72}")
    print(f"DECISIVE TEST SUMMARY")
    print(f"{'='*72}")
    print(f"  Truth:   stiff_in=0.800, stiff_out=0.300, tau_in=0.20, tau_out=0.40")
    print(f"  Init:    stiff_in=0.500, stiff_out=0.500, tau_in=0.30, tau_out=0.30 (uniform)")
    print()
    print(f"  L_gap (current, broken):")
    print(f"     stiff_in={stiff_gap[inclusion_mask].mean():.3f} (Δfrom truth = {stiff_gap[inclusion_mask].mean()-0.8:+.3f})")
    print(f"     stiff_out={stiff_gap[~inclusion_mask].mean():.3f} (Δfrom truth = {stiff_gap[~inclusion_mask].mean()-0.3:+.3f})")
    print(f"     IoU={m_gap['iou']:.3f}")
    print(f"     T_B pos err = {err_gap_pos:.3e}")
    print()
    print(f"  L_direct (new, hypothesis):")
    print(f"     stiff_in={stiff_dir[inclusion_mask].mean():.3f} (Δfrom truth = {stiff_dir[inclusion_mask].mean()-0.8:+.3f})")
    print(f"     stiff_out={stiff_dir[~inclusion_mask].mean():.3f} (Δfrom truth = {stiff_dir[~inclusion_mask].mean()-0.3:+.3f})")
    print(f"     IoU={m_dir['iou']:.3f}")
    print(f"     T_B pos err = {err_dir_pos:.3e}")

    # Plot trajectories
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    epochs = [h["epoch"] for h in hist_gap]
    # Loss
    axes[0, 0].plot(epochs, [h["mean_L"] for h in hist_gap], "b-", label="L_gap")
    axes[0, 0].plot(epochs, [h["mean_L"] for h in hist_dir], "r-", label="L_direct")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("epoch"); axes[0, 0].set_ylabel("loss (log)")
    axes[0, 0].set_title("Loss trajectory")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    # stiff_in
    axes[0, 1].plot(epochs, [h["stiff_in"] for h in hist_gap], "b-", label="L_gap stiff_in")
    axes[0, 1].plot(epochs, [h["stiff_in"] for h in hist_dir], "r-", label="L_direct stiff_in")
    axes[0, 1].axhline(0.8, color="k", linestyle="--", label="truth 0.8")
    axes[0, 1].axhline(0.5, color="gray", linestyle=":", label="init 0.5")
    axes[0, 1].set_xlabel("epoch"); axes[0, 1].set_ylabel("stiff_in mean")
    axes[0, 1].set_title("stiff_in recovery")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    # stiff_out
    axes[1, 0].plot(epochs, [h["stiff_out"] for h in hist_gap], "b-", label="L_gap stiff_out")
    axes[1, 0].plot(epochs, [h["stiff_out"] for h in hist_dir], "r-", label="L_direct stiff_out")
    axes[1, 0].axhline(0.3, color="k", linestyle="--", label="truth 0.3")
    axes[1, 0].axhline(0.5, color="gray", linestyle=":", label="init 0.5")
    axes[1, 0].set_xlabel("epoch"); axes[1, 0].set_ylabel("stiff_out mean")
    axes[1, 0].set_title("stiff_out recovery")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    # IoU
    axes[1, 1].plot(epochs, [h["iou"] for h in hist_gap], "b-", label="L_gap IoU")
    axes[1, 1].plot(epochs, [h["iou"] for h in hist_dir], "r-", label="L_direct IoU")
    axes[1, 1].axhline(1.0, color="k", linestyle="--", label="perfect")
    axes[1, 1].set_xlabel("epoch"); axes[1, 1].set_ylabel("IoU")
    axes[1, 1].set_title("IoU trajectory")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
    fig.suptitle(f"Decisive test: L_gap vs L_direct (30 epochs Adam)\n"
                 f"Truth: stiff_in=0.8, stiff_out=0.3", fontsize=13)
    fig.tight_layout()
    out_path = REPO / "tools" / "pointnetpp" / "diag_L_direct_decisive.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
