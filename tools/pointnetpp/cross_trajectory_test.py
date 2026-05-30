"""Cross-trajectory predictive validity test — baseline vs hierarchical.

For each calibrated parameter set (baseline / hierarchical), forward-sim multiple
HELD-OUT trajectories (NOT used during calibration) and measure how close the
predicted deformation matches the ground-truth deformation at those trajectories.

The hypothesis is: hierarchical reparameterization injects spatial coherence
that captures the inclusion's physical structure, which is invariant across
trajectories. Per-particle baseline can fit T_A's deformation pattern but
without spatial structure prior may overfit; trajectories that probe different
regions of the inclusion would then have larger prediction error.

Experiment:
    1. Truth params (stiff + tau, both inclusion fields) FROZEN.
    2. 4 trajectories defined: T_A (calibration), T_B/C/D (validation).
    3. Generate truth obs for each trajectory independently, from rest.
    4. Calibrate baseline (per-particle z) on T_A only → params_base.
    5. Calibrate hierarchical (z_L0, z_L1, z_L2) on T_A only → params_hier.
    6. For each trajectory X in {A, B, C, D}:
         - Forward-sim with frozen params_base on T_X → predict_base[X]
         - Forward-sim with frozen params_hier on T_X → predict_hier[X]
         - Compute L2 position error vs truth obs.
    7. Compare base[X] vs hier[X] for X ≠ A (cross-trajectory).
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

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day8_stable_inner_loop import gd_inner_loop_stable  # noqa: E402

from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff, compute_metrics,
    inv_sigmoid_to_z, STIFF_LOW, STIFF_HIGH,
)
from tools.pointnetpp.hierarchy import build_hierarchy, reconstruct  # noqa: E402


def make_traj(name, T=10):
    """Define 4 distinct knife trajectories over the cube.

    Cube is (0,0,0) to (0.30, 0.30, 0.30). Inclusion center (0.15, 0.25, 0.15),
    radius 0.06 → inclusion bbox roughly ±0.06 from center.

    All 4 trajectories pass near the inclusion to ensure SOME signal — but
    probe it from different directions / depths.
    """
    if name == "A":
        # Calibration: sweep x at top surface, through inclusion center.
        xs = np.linspace(0.12, 0.18, T)
        return [np.array([x, 0.30, 0.15], dtype=np.float32) for x in xs]
    elif name == "B":
        # Validation: rotated 90° — sweep z at top, through inclusion z-extent.
        zs = np.linspace(0.12, 0.18, T)
        return [np.array([0.15, 0.30, z], dtype=np.float32) for z in zs]
    elif name == "C":
        # Validation: depth plunge — knife descends y from top toward inclusion.
        ys = np.linspace(0.30, 0.22, T)
        return [np.array([0.15, y, 0.15], dtype=np.float32) for y in ys]
    elif name == "D":
        # Validation: lateral offset — sweep x to side of inclusion (still grazes).
        xs = np.linspace(0.18, 0.24, T)
        return [np.array([x, 0.30, 0.15], dtype=np.float32) for x in xs]
    else:
        raise ValueError(name)


def calibrate(mode, ctx, N_epochs=25, lr=0.03, lr_tau=0.01):
    """Calibrate on T_A only. Returns final stiff_pp + tau_pp arrays (numpy)
    selected by L_gap-best (no oracle ground truth used).

    mode: "baseline" or "hierarchical".
    """
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
        params = [
            {"params": [z_L0_stiff], "lr": lr},
            {"params": [z_L0_tau],   "lr": lr_tau},
        ]
        use_hierarchy = False
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
        use_hierarchy = True
    else:
        raise ValueError(mode)

    opt = torch.optim.Adam(params)

    T = len(knife_traj)
    best_L_gap = float("inf")
    best_stiff = stiff_init_np.copy()
    best_tau = tau_init_np.copy()

    for epoch in range(N_epochs):
        state = rest_state
        L_gaps = []
        for t in range(T):
            if use_hierarchy:
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

            # Mode C inject
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
            if use_hierarchy:
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


def predict_on_trajectory(stiff_pp_np, tau_pp_np, knife_traj, ctx):
    """Forward-sim with FROZEN params on a trajectory. Returns
    (predicted pos list, predicted alpha list) over T frames.

    CRITICAL: solver.reset_to_rest() is called at start to ensure full state
    reset (cluster_quat identity, SSA cycle index 0, etc.) — solver_load_state
    alone does NOT reset all internal state; per solver.py docstring,
    residual state from prior calls causes silent bug (Test 3 ≠ Test 2).
    """
    device = ctx["device"]
    theta_torch = torch.tensor(ctx["theta_3d_np"], dtype=torch.float32, device=device,
                                 requires_grad=False)
    stiff_torch = torch.tensor(stiff_pp_np, dtype=torch.float32, device=device,
                                  requires_grad=False)
    tau_torch = torch.tensor(tau_pp_np, dtype=torch.float32, device=device,
                                requires_grad=False)
    solver = ctx["solver"]
    rest_state = ctx["rest_state"]
    knife_radius = ctx["knife_radius"]
    pos_rest = ctx["pos_rest"]  # required for full reset

    # Full reset of evolved state (cluster_quat identity, SSA cycle = 0,
    # damages zeroed, etc.) — beyond what solver_load_state does.
    solver.reset_to_rest(pos_rest)

    state_in = rest_state
    pred_pos = []
    pred_alpha = []
    with torch.no_grad():
        for t, knife_pos in enumerate(knife_traj):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, knife_pos, knife_radius,
                stiff_torch, tau_torch,
            )
            pred_pos.append(pos_t.cpu().numpy().copy())
            pred_alpha.append(alpha_t.cpu().numpy().copy())
            state_in = solver_save_state(solver)
    return pred_pos, pred_alpha


def chamfer_per_traj(pred_pos_list, truth_pos_list, pred_alpha_list, truth_alpha_list):
    """Per-trajectory error.

    Since particles are in correspondence (same N from same mesh), use per-
    particle L2, weighted by truth alive mask (alpha > 0.5 in truth).
    Aggregate across T frames as mean.

    Also report alpha L1.
    """
    T = len(pred_pos_list)
    pos_errs = []
    alpha_errs = []
    for t in range(T):
        pp = pred_pos_list[t]
        tp = truth_pos_list[t]
        pa = pred_alpha_list[t]
        ta = truth_alpha_list[t]
        alive = ta > 0.5
        # Per-particle L2 distance
        d = np.linalg.norm(pp - tp, axis=1)  # [N]
        if alive.sum() > 0:
            pos_errs.append(float(d[alive].mean()))
        else:
            pos_errs.append(float(d.mean()))
        alpha_errs.append(float(np.abs(pa - ta).mean()))
    return float(np.mean(pos_errs)), float(np.mean(alpha_errs))


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    print(f"\n=== Cross-trajectory predictive validity test ===")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = 2

    device = "cuda:0"
    knife_radius = 0.08
    T = 10

    theta_truth = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    pos_rest = mesh_info["vertices"]
    stiff_truth, inclusion_mask, center, incl_r = build_inclusion_truth(
        pos_rest, stiff_in=0.8, stiff_out=0.3,
        center_override=(0.15, 0.25, 0.15),
        radius_override=0.06)
    tau_truth = np.where(inclusion_mask, 0.20, 0.40).astype(np.float32)
    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)

    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)

    # Build hierarchy
    x_rest = torch.from_numpy(pos_rest).to(device, dtype=torch.float32)
    H = build_hierarchy(x_rest, M1=512, M2=64, k_neighbors=3)

    # --- Generate truth obs for ALL trajectories independently ----------
    trajectories = ["A", "B", "C", "D"]
    truth = {}
    for name in trajectories:
        traj = make_traj(name, T=T)
        print(f"\n[Truth gen] T_{name}: from {traj[0]} to {traj[-1]}")
        obs_pos, obs_alpha = gen_obs_per_particle_stiff(
            solver, rest_state, theta_truth, stiff_truth, traj, knife_radius, device,
            tau_truth_np=tau_truth)
        cav = float((obs_alpha[-1] < 0.5).mean())
        print(f"             cavity fraction at last frame: {cav:.4f}")
        truth[name] = dict(traj=traj, pos=obs_pos, alpha=obs_alpha)

    # Torch tensors for T_A obs (calibration target)
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

    # --- Calibrate baseline + hierarchical on T_A -----------------------
    print(f"\n{'='*72}")
    print(f"Calibration on T_A only (N_epochs=25)")
    print(f"{'='*72}")

    t0 = time.time()
    stiff_base, tau_base, Lgap_base = calibrate("baseline", ctx, N_epochs=25)
    print(f"  baseline    : Lgap_best={Lgap_base:.3e}  ({time.time()-t0:.0f}s)")
    m_base = compute_metrics(stiff_base, stiff_truth, inclusion_mask)
    print(f"               IoU(stiff)={m_base['iou']:.3f}  L1={m_base['L1_total']:.4f}")

    t0 = time.time()
    stiff_hier, tau_hier, Lgap_hier = calibrate("hierarchical", ctx, N_epochs=25)
    print(f"  hierarchical: Lgap_best={Lgap_hier:.3e}  ({time.time()-t0:.0f}s)")
    m_hier = compute_metrics(stiff_hier, stiff_truth, inclusion_mask)
    print(f"               IoU(stiff)={m_hier['iou']:.3f}  L1={m_hier['L1_total']:.4f}")

    # --- Predict each trajectory + compute error ------------------------
    print(f"\n{'='*72}")
    print(f"Predictive validity per trajectory")
    print(f"{'='*72}")

    results = {}
    for name in trajectories:
        traj = truth[name]["traj"]
        truth_pos = truth[name]["pos"]
        truth_alpha = truth[name]["alpha"]

        pred_pos_b, pred_alpha_b = predict_on_trajectory(stiff_base, tau_base, traj, ctx)
        pos_err_b, alpha_err_b = chamfer_per_traj(pred_pos_b, truth_pos, pred_alpha_b, truth_alpha)

        pred_pos_h, pred_alpha_h = predict_on_trajectory(stiff_hier, tau_hier, traj, ctx)
        pos_err_h, alpha_err_h = chamfer_per_traj(pred_pos_h, truth_pos, pred_alpha_h, truth_alpha)

        results[name] = dict(
            pos_err_base=pos_err_b, pos_err_hier=pos_err_h,
            alpha_err_base=alpha_err_b, alpha_err_hier=alpha_err_h,
        )

    # --- Report ----------------------------------------------------------
    print(f"\n  {'Traj':6s}  {'pos_err base':>14s}  {'pos_err hier':>14s}  {'Δ (hier-base)':>14s}  "
          f"{'α_err base':>12s}  {'α_err hier':>12s}")
    for name in trajectories:
        r = results[name]
        delta = r["pos_err_hier"] - r["pos_err_base"]
        marker = "★" if name != "A" and delta < 0 else (" " if name != "A" else "·")
        print(f"  {marker} T_{name}   "
              f"{r['pos_err_base']:>14.6f}  {r['pos_err_hier']:>14.6f}  "
              f"{delta:>+14.6f}  "
              f"{r['alpha_err_base']:>12.6f}  {r['alpha_err_hier']:>12.6f}")

    print(f"\n  ★ = hierarchical beats baseline on held-out trajectory (lower pos_err)")
    print(f"  · = calibration trajectory (T_A) — expected to fit both methods")

    # Aggregate: mean over B, C, D (held-out)
    held = ["B", "C", "D"]
    pos_b_held = np.mean([results[n]["pos_err_base"] for n in held])
    pos_h_held = np.mean([results[n]["pos_err_hier"] for n in held])
    alpha_b_held = np.mean([results[n]["alpha_err_base"] for n in held])
    alpha_h_held = np.mean([results[n]["alpha_err_hier"] for n in held])

    print(f"\n{'='*72}")
    print(f"CROSS-TRAJECTORY AGGREGATE (mean over T_B, T_C, T_D held-out)")
    print(f"{'='*72}")
    print(f"  Baseline    pos err: {pos_b_held:.6f}   α err: {alpha_b_held:.6f}")
    print(f"  Hierarchical pos err: {pos_h_held:.6f}   α err: {alpha_h_held:.6f}")
    pos_rel = (pos_h_held - pos_b_held) / pos_b_held * 100
    alpha_rel = (alpha_h_held - alpha_b_held) / alpha_b_held * 100
    print(f"  Δ pos (hier rel to base): {pos_rel:+.2f}%  ({'hier better' if pos_rel < 0 else 'base better'})")
    print(f"  Δ α   (hier rel to base): {alpha_rel:+.2f}%  ({'hier better' if alpha_rel < 0 else 'base better'})")

    # Save
    out_path = REPO / "tools" / "pointnetpp" / "cross_trajectory_results.npz"
    np.savez(
        out_path,
        trajectories=np.array(trajectories),
        stiff_base=stiff_base, stiff_hier=stiff_hier,
        tau_base=tau_base, tau_hier=tau_hier,
        results=results,
    )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
