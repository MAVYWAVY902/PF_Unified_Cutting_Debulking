"""Phase 2.1: per-level hierarchy ablation.

Quantify each hierarchy level's contribution to cross-trajectory advantage:

   Mode L0     :  z_L0 ∈ R^N only          (= baseline per-particle scalar)
   Mode L0+L1  :  z_L0 + z_L1              (mid-scale prior only)
   Mode L0+L2  :  z_L0 + z_L2              (coarse-scale prior only)
   Mode FULL   :  z_L0 + z_L1 + z_L2       (full hierarchy)

Run on cube mesh with surface-debulk setup (from multi_mesh_test M1_cube),
4 trajectories (T_A calibration, T_B/C/D held-out), 15 epochs.

If each level helps incrementally → "hierarchical" framing is genuine.
If only one level matters → reframe as "spatial prior at that scale".

Compute: 4 modes × ~8s calibration + 4 traj predict ≈ 1 min per mode × 4 = 4 min.
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
from tools.pointnetpp.multi_mesh_test import (  # noqa: E402
    MESH_CONFIGS, make_traj_from_spec, predict_on_traj, err_per_traj,
)


# 4 ablation modes
MODES = ["L0", "L0+L1", "L0+L2", "FULL"]


def calibrate_ablation(mode, ctx, N_epochs=15, lr=0.03, lr_tau=0.01):
    """Calibrate with selected levels active.

    mode: "L0" / "L0+L1" / "L0+L2" / "FULL"

    Frozen z_l are kept at 0 (so reconstruct returns z_L0 + 0 + 0 = z_L0,
    matching baseline).
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
    H = ctx["H"]
    M1, M2 = H.M1, H.M2
    sL1 = 1.0 / math.sqrt(N / M1)
    sL2 = 1.0 / math.sqrt(N / M2)

    theta_torch = torch.tensor(theta_3d_np, dtype=torch.float32, device=device,
                                 requires_grad=False)
    z_L0_stiff = torch.tensor(inv_sigmoid_to_z(stiff_init_np),
                                dtype=torch.float32, device=device, requires_grad=True)
    z_L0_tau = torch.tensor(inv_sigmoid_to_z(tau_init_np, lo=tau_bound[0], hi=tau_bound[1]),
                              dtype=torch.float32, device=device, requires_grad=True)
    # L1, L2 always created, but only put in optimizer when active.
    # BUG fix 2026-05-29: "L1" in "FULL" was False — FULL silently ran as L0.
    L1_active = mode in ("L0+L1", "FULL")
    L2_active = mode in ("L0+L2", "FULL")
    z_L1_stiff = torch.zeros(M1, dtype=torch.float32, device=device, requires_grad=L1_active)
    z_L2_stiff = torch.zeros(M2, dtype=torch.float32, device=device, requires_grad=L2_active)
    z_L1_tau = torch.zeros(M1, dtype=torch.float32, device=device, requires_grad=L1_active)
    z_L2_tau = torch.zeros(M2, dtype=torch.float32, device=device, requires_grad=L2_active)

    param_groups = [
        {"params": [z_L0_stiff], "lr": lr},
        {"params": [z_L0_tau],   "lr": lr_tau},
    ]
    if L1_active:
        param_groups.append({"params": [z_L1_stiff], "lr": lr * sL1})
        param_groups.append({"params": [z_L1_tau],   "lr": lr_tau * sL1})
    if L2_active:
        param_groups.append({"params": [z_L2_stiff], "lr": lr * sL2})
        param_groups.append({"params": [z_L2_tau],   "lr": lr_tau * sL2})
    opt = torch.optim.Adam(param_groups)

    T = len(knife_traj)
    best_L_gap = float("inf")
    best_stiff = stiff_init_np.copy()
    best_tau = tau_init_np.copy()

    for epoch in range(N_epochs):
        solver.reset_to_rest(ctx["pos_rest"])
        state = rest_state
        L_gaps = []
        for t in range(T):
            z_flat_stiff = reconstruct(z_L0_stiff, z_L1_stiff, z_L2_stiff, H)
            z_flat_tau = reconstruct(z_L0_tau, z_L1_tau, z_L2_tau, H)
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
            z_flat_stiff_now = reconstruct(z_L0_stiff, z_L1_stiff, z_L2_stiff, H)
            z_flat_tau_now = reconstruct(z_L0_tau, z_L1_tau, z_L2_tau, H)
            stiff_now = (STIFF_LOW + (STIFF_HIGH - STIFF_LOW)
                            * torch.sigmoid(z_flat_stiff_now)).cpu().numpy()
            tau_now = (tau_bound[0] + (tau_bound[1] - tau_bound[0])
                          * torch.sigmoid(z_flat_tau_now)).cpu().numpy()

        if mean_L < best_L_gap - 1e-4:
            best_L_gap = mean_L
            best_stiff = stiff_now.copy()
            best_tau = tau_now.copy()
    return best_stiff, best_tau, best_L_gap


def main():
    cfg = MESH_CONFIGS[0]  # M1_cube — same setup that worked for multi-mesh
    device = "cuda:0"
    T = 10
    mesh_path = str(REPO / cfg["path"])
    print(f"\n=== Per-level ablation on {cfg['name']} ===")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
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
    print(f"  N={N}  inclusion={int(inclusion_mask.sum())} particles ({inclusion_mask.sum()/N*100:.2f}%)")

    theta_truth_np = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    truth = {}
    for name, spec in cfg["traj_specs"].items():
        traj = make_traj_from_spec(spec, bbox_min, bbox_max, T)
        solver.reset_to_rest(pos_rest)
        obs_pos, obs_alpha = gen_obs_per_particle_stiff(
            solver, rest_state, theta_truth_np, stiff_truth, traj, knife_radius, device,
            tau_truth_np=tau_truth)
        truth[name] = dict(traj=traj, pos=obs_pos, alpha=obs_alpha)

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

    # Run 4 modes
    all_results = {}
    for mode in MODES:
        print(f"\n--- Mode: {mode} ---", flush=True)
        t0 = time.time()
        stiff, tau, Lgap = calibrate_ablation(mode, ctx, N_epochs=15)
        elapsed = time.time() - t0
        iou = compute_metrics(stiff, stiff_truth, inclusion_mask)["iou"]
        print(f"  calibrate {elapsed:.0f}s   Lgap_best={Lgap:.3e}   IoU={iou:.3f}", flush=True)

        # Predict on all 4 trajectories
        results = {}
        for tr_name in ["A", "B", "C", "D"]:
            traj = truth[tr_name]["traj"]
            pp, pa = predict_on_traj(stiff, tau, traj, ctx)
            e_pos, e_a = err_per_traj(pp, truth[tr_name]["pos"], pa, truth[tr_name]["alpha"])
            results[tr_name] = dict(pos=e_pos, alpha=e_a)
        all_results[mode] = dict(
            results=results, stiff=stiff, tau=tau, Lgap_best=Lgap, iou=iou,
        )

    # Summary table — pos_err per (mode, traj)
    print(f"\n{'='*80}")
    print(f"PER-LEVEL ABLATION SUMMARY")
    print(f"{'='*80}")
    print(f"\n  Position error (×1e-5 m) per (mode, traj). Δ_rel vs L0 baseline.")
    print(f"  {'Mode':10s}  {'IoU':>5s}  " + "  ".join([f"{tr:>18s}" for tr in ["T_A","T_B","T_C","T_D"]]))
    L0_pos = {tr: all_results["L0"]["results"][tr]["pos"] for tr in ["A","B","C","D"]}
    for mode in MODES:
        cells = []
        for tr in ["A","B","C","D"]:
            v = all_results[mode]["results"][tr]["pos"] * 1e5
            rel = (all_results[mode]["results"][tr]["pos"] - L0_pos[tr]) / L0_pos[tr] * 100
            mark = "★" if (mode != "L0" and rel < -1) else ("✗" if (mode != "L0" and rel > 1) else " ")
            cells.append(f"{v:6.3f} {mark}{rel:+6.1f}%")
        print(f"  {mode:10s}  {all_results[mode]['iou']:>5.3f}  " + "  ".join(cells))

    print(f"\n  Alpha error per (mode, traj). Δ_rel vs L0 baseline.")
    print(f"  {'Mode':10s}  " + "  ".join([f"{tr:>18s}" for tr in ["T_A","T_B","T_C","T_D"]]))
    L0_a = {tr: all_results["L0"]["results"][tr]["alpha"] for tr in ["A","B","C","D"]}
    for mode in MODES:
        cells = []
        for tr in ["A","B","C","D"]:
            v = all_results[mode]["results"][tr]["alpha"]
            rel = (all_results[mode]["results"][tr]["alpha"] - L0_a[tr]) / L0_a[tr] * 100
            mark = "★" if (mode != "L0" and rel < -1) else ("✗" if (mode != "L0" and rel > 1) else " ")
            cells.append(f"{v:.4f} {mark}{rel:+6.1f}%")
        print(f"  {mode:10s}  " + "  ".join(cells))

    # Aggregate over held-out (T_B+T_C+T_D)
    print(f"\n  Aggregate held-out (T_B + T_C + T_D) vs L0 baseline:")
    print(f"  {'Mode':10s}  {'pos_rel%':>10s}  {'alpha_rel%':>11s}")
    for mode in MODES:
        if mode == "L0":
            continue
        rel_p = np.mean([(all_results[mode]["results"][tr]["pos"] - L0_pos[tr]) / L0_pos[tr] * 100
                           for tr in ["B","C","D"]])
        rel_a = np.mean([(all_results[mode]["results"][tr]["alpha"] - L0_a[tr]) / L0_a[tr] * 100
                           for tr in ["B","C","D"]])
        print(f"  {mode:10s}  {rel_p:>+9.2f}%  {rel_a:>+10.2f}%")

    # Marginal contribution analysis
    print(f"\n  Marginal contribution per level (FULL - other):")
    full_rel_p = np.mean([(all_results["FULL"]["results"][tr]["pos"] - L0_pos[tr]) / L0_pos[tr] * 100
                            for tr in ["B","C","D"]])
    l1_only_rel = np.mean([(all_results["L0+L1"]["results"][tr]["pos"] - L0_pos[tr]) / L0_pos[tr] * 100
                             for tr in ["B","C","D"]])
    l2_only_rel = np.mean([(all_results["L0+L2"]["results"][tr]["pos"] - L0_pos[tr]) / L0_pos[tr] * 100
                             for tr in ["B","C","D"]])
    print(f"    L1 contribution (L0+L1 - L0):    {l1_only_rel:+.2f}%")
    print(f"    L2 contribution (L0+L2 - L0):    {l2_only_rel:+.2f}%")
    print(f"    Full hierarchy (FULL  - L0):     {full_rel_p:+.2f}%")
    print(f"    Synergy (FULL - L0+L1 - L0+L2):  {full_rel_p - l1_only_rel - l2_only_rel:+.2f}%  "
          f"(if << 0, levels are synergistic)")

    # Verdict
    print(f"\n{'='*80}")
    print(f"VERDICT")
    print(f"{'='*80}")
    if abs(l1_only_rel - full_rel_p) < 1 and abs(l2_only_rel) < 1:
        print(f"  → L1 dominates; reframe as 'mid-scale spatial prior'")
    elif abs(l2_only_rel - full_rel_p) < 1 and abs(l1_only_rel) < 1:
        print(f"  → L2 dominates; reframe as 'coarse-scale spatial prior'")
    elif l1_only_rel < -2 and l2_only_rel < -2 and full_rel_p < l1_only_rel and full_rel_p < l2_only_rel:
        print(f"  ✓ Both levels contribute & synergize → 'hierarchical' framing genuine")
    else:
        print(f"  △ Mixed picture — see per-cell numbers above")

    out_path = REPO / "tools" / "pointnetpp" / "per_level_ablation_results.npz"
    np.savez(out_path, results=all_results)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
