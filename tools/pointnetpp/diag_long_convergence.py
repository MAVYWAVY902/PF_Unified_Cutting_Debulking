"""Longer & more rigorous convergence test — 6 configurations.

Earlier decisive test (30 epochs) showed both L_gap and L_direct converging
slowly toward truth but plateauing partial. This tests:

   1. L_gap   lr=0.03  (baseline current setting)
   2. L_gap   lr=0.01  (slower, maybe higher accuracy)
   3. L_direct_α  lr=0.03   λ_α=0.1  (current decisive test)
   4. L_direct_α  lr=0.001  λ_α=0.1  (smaller lr matches larger loss scale)
   5. L_pos_only  lr=0.03   (no α term — α may bias optimization)
   6. L_pos_only  lr=0.001

All 60 epochs. Track:
   - stiff_in / stiff_out trajectories
   - L1 error to truth (preferred over IoU which is rank-noisy)
   - IoU for completeness
   - Final cross-traj T_B prediction
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


def calibrate(loss_kind, ctx, N_epochs=60, lr=0.03, lr_tau=0.01,
              lambda_alpha=0.1):
    """Calibrate with specified loss + lr."""
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
            elif loss_kind == "L_pos":
                L = ((pos_t - obs_pos_t[t]) ** 2).sum()
            else:
                raise ValueError(loss_kind)

            opt.zero_grad()
            L.backward()
            opt.step()
            L_epoch.append(float(L.item()))

            # Mode C inject only for L_gap
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
                state = solver_save_state(solver)

        mean_L = float(np.mean(L_epoch))
        with torch.no_grad():
            stiff_now = (STIFF_LOW + (STIFF_HIGH - STIFF_LOW)
                            * torch.sigmoid(z_stiff)).cpu().numpy()
            tau_now = (tau_bound[0] + (tau_bound[1] - tau_bound[0])
                          * torch.sigmoid(z_tau)).cpu().numpy()
        m = compute_metrics(stiff_now, stiff_truth, inclusion_mask)
        L1_in = float(np.abs(stiff_now[inclusion_mask] - stiff_truth[inclusion_mask]).mean())
        L1_out = float(np.abs(stiff_now[~inclusion_mask] - stiff_truth[~inclusion_mask]).mean())
        history.append(dict(
            epoch=epoch, mean_L=mean_L, iou=m["iou"],
            stiff_in=float(stiff_now[inclusion_mask].mean()),
            stiff_out=float(stiff_now[~inclusion_mask].mean()),
            L1_in=L1_in, L1_out=L1_out, L1_total=(L1_in + L1_out) / 2,
        ))
    return stiff_now, tau_now, history


def main():
    cfg = MESH_CONFIGS[0]
    device = "cuda:0"
    T = 10
    N_epochs = 60
    print(f"\n=== Long convergence comparison ({N_epochs} epochs each) ===")
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
    print(f"  Truth: stiff_in=0.8 stiff_out=0.3")

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
        pos_rest=pos_rest, theta_truth=theta_truth, theta_3d_np=theta_truth,
        stiff_init_np=stiff_init, stiff_truth=stiff_truth,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        inclusion_mask=inclusion_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
    )

    configs = [
        ("L_gap lr=0.03",       "L_gap", 0.03, None),
        ("L_gap lr=0.01",       "L_gap", 0.01, None),
        ("L_direct lr=0.03",    "L_direct", 0.03, 0.1),
        ("L_direct lr=0.001",   "L_direct", 0.001, 0.1),
        ("L_pos lr=0.03",       "L_pos", 0.03, None),
        ("L_pos lr=0.001",      "L_pos", 0.001, None),
    ]
    results = {}
    for label, loss_kind, lr, la in configs:
        print(f"\n--- {label} ---", flush=True)
        t0 = time.time()
        kwargs = dict(N_epochs=N_epochs, lr=lr, lr_tau=lr/3)
        if la is not None:
            kwargs["lambda_alpha"] = la
        stiff, tau, hist = calibrate(loss_kind, ctx, **kwargs)
        elapsed = time.time() - t0
        m = compute_metrics(stiff, stiff_truth, inclusion_mask)
        # Cross-traj T_B
        pred_pos, pred_alpha = predict_on_traj(stiff, tau, truth["B"]["traj"], ctx)
        err_pos, err_a = err_per_traj(pred_pos, truth["B"]["pos"], pred_alpha, truth["B"]["alpha"])
        print(f"  done {elapsed:.0f}s  final: stiff_in={hist[-1]['stiff_in']:.3f}  "
              f"stiff_out={hist[-1]['stiff_out']:.3f}  IoU={m['iou']:.3f}  "
              f"L1_in={hist[-1]['L1_in']:.3f}  T_B err={err_pos:.3e}")
        results[label] = dict(stiff=stiff, tau=tau, hist=hist, iou=m["iou"],
                                T_B_pos=err_pos, T_B_alpha=err_a)

    # Summary
    print(f"\n{'='*100}")
    print(f"60-EPOCH SUMMARY")
    print(f"{'='*100}")
    print(f"  {'config':22s}  {'stiff_in':>9s}  {'stiff_out':>9s}  {'L1_in':>7s}  {'L1_out':>7s}  "
          f"{'IoU':>5s}  {'T_B err':>9s}")
    print(f"  {'(truth)':22s}  {0.8:>9.3f}  {0.3:>9.3f}  {0.0:>7.3f}  {0.0:>7.3f}  "
          f"{1.0:>5.3f}  {'-':>9s}")
    for label in results:
        r = results[label]; h = r["hist"][-1]
        print(f"  {label:22s}  {h['stiff_in']:>9.3f}  {h['stiff_out']:>9.3f}  "
              f"{h['L1_in']:>7.3f}  {h['L1_out']:>7.3f}  "
              f"{r['iou']:>5.3f}  {r['T_B_pos']:>9.3e}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = list(range(N_epochs))
    colors = ["tab:blue", "tab:cyan", "tab:red", "tab:pink", "tab:green", "tab:olive"]
    for i, label in enumerate(results):
        hist = results[label]["hist"]
        c = colors[i]
        axes[0, 0].plot(epochs, [h["stiff_in"] for h in hist], label=label, color=c)
        axes[0, 1].plot(epochs, [h["stiff_out"] for h in hist], label=label, color=c)
        axes[1, 0].plot(epochs, [h["L1_in"] for h in hist], label=label, color=c)
        axes[1, 1].plot(epochs, [h["iou"] for h in hist], label=label, color=c)
    axes[0, 0].axhline(0.8, color="k", linestyle="--", label="truth")
    axes[0, 0].axhline(0.5, color="gray", linestyle=":", label="init")
    axes[0, 0].set_xlabel("epoch"); axes[0, 0].set_ylabel("stiff_in mean")
    axes[0, 0].set_title("stiff_in mean trajectory (truth=0.8)")
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].axhline(0.3, color="k", linestyle="--", label="truth")
    axes[0, 1].axhline(0.5, color="gray", linestyle=":", label="init")
    axes[0, 1].set_xlabel("epoch"); axes[0, 1].set_ylabel("stiff_out mean")
    axes[0, 1].set_title("stiff_out mean trajectory (truth=0.3)")
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].set_xlabel("epoch"); axes[1, 0].set_ylabel("L1 error inside")
    axes[1, 0].set_title("L1 error |stiff_in - truth_in|  (lower better)")
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axhline(1.0, color="k", linestyle="--", label="perfect")
    axes[1, 1].set_xlabel("epoch"); axes[1, 1].set_ylabel("IoU")
    axes[1, 1].set_title("IoU (note: top-K argmax is noisy near uniform fields)")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)
    fig.suptitle(f"60-epoch convergence — 6 loss/lr configurations on M1_cube",
                  fontsize=14)
    fig.tight_layout()
    out_path = REPO / "tools" / "pointnetpp" / "diag_long_convergence.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
