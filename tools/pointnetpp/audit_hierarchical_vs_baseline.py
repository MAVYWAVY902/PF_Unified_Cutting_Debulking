"""Rigorous audit: does hierarchical REALLY help vs baseline, or is the 0.21
vs 0.20 IoU difference just stochastic noise?

3 experiments:
    A. Baseline (per-particle z_stiff ∈ R^N)
    B. Hierarchical with z_L1=z_L2 FROZEN at 0 — should match baseline EXACTLY
       (sanity: confirms my reconstruct() reduces to identity when coarse z=0)
    C. Hierarchical with all 3 levels active + LR-balanced

If B ≈ A → code path is consistent (no hidden bug).
If C > A statistically → hierarchical helps.
If C ≈ A → hierarchical doesn't help on this single-trajectory task.

3 seeds per mode = 9 runs. Each ~12s → ~2 min total.

Honest reporting: mean ± std on peak IoU across seeds. Welch's t-test for
significance.
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

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day8_stable_inner_loop import gd_inner_loop_stable  # noqa: E402
from tools.day6f_v0_multistep_synth import make_knife_trajectory  # noqa: E402

from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff, compute_metrics,
    inv_sigmoid_to_z, z_to_stiff, STIFF_LOW, STIFF_HIGH,
)
from tools.pointnetpp.hierarchy import build_hierarchy, reconstruct  # noqa: E402


def run_one(mode, seed, ctx, N_epochs=25):
    """Run a single recovery experiment.

    mode: "baseline" / "frozen_hier" / "active_hier"
    seed: int (sets torch + numpy seeds for FPS randomness via start_idx)
    ctx: dict of shared inputs (solver, obs, etc.)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = ctx["device"]
    N = ctx["N"]
    stiff_init_np = ctx["stiff_init_np"]
    tau_init_np = ctx["tau_init_np"]
    tau_bound = ctx["tau_bound"]
    theta_3d_np = ctx["theta_3d_np"]
    inclusion_mask = ctx["inclusion_mask"]
    stiff_truth_np = ctx["stiff_truth_np"]

    # Init z_L0 (per-particle) — same for all 3 modes
    z_L0_stiff = torch.tensor(inv_sigmoid_to_z(stiff_init_np),
                                dtype=torch.float32, device=device, requires_grad=True)
    z_L0_tau = torch.tensor(inv_sigmoid_to_z(tau_init_np, lo=tau_bound[0], hi=tau_bound[1]),
                              dtype=torch.float32, device=device, requires_grad=True)

    if mode == "baseline":
        # Pure per-particle — use ONLY z_L0, no hierarchy.
        params = [
            {"params": [z_L0_stiff], "lr": 0.03},
            {"params": [z_L0_tau],   "lr": 0.01},
        ]
        use_hierarchy = False
        H = None
    else:
        # Build hierarchy (mode b and c)
        H = ctx["H"]
        M1, M2 = H.M1, H.M2
        # Init coarse z's at 0
        z_L1_stiff = torch.zeros(M1, dtype=torch.float32, device=device,
                                    requires_grad=(mode == "active_hier"))
        z_L2_stiff = torch.zeros(M2, dtype=torch.float32, device=device,
                                    requires_grad=(mode == "active_hier"))
        z_L1_tau = torch.zeros(M1, dtype=torch.float32, device=device,
                                  requires_grad=(mode == "active_hier"))
        z_L2_tau = torch.zeros(M2, dtype=torch.float32, device=device,
                                  requires_grad=(mode == "active_hier"))
        if mode == "frozen_hier":
            # ONLY z_L0 in optimizer — coarse z's frozen at 0.
            # reconstruct returns z_L0 + 0 + 0 = z_L0 → identical math to baseline
            params = [
                {"params": [z_L0_stiff], "lr": 0.03},
                {"params": [z_L0_tau],   "lr": 0.01},
            ]
        elif mode == "active_hier":
            import math
            sL1 = 1.0 / math.sqrt(N / M1)
            sL2 = 1.0 / math.sqrt(N / M2)
            params = [
                {"params": [z_L0_stiff], "lr": 0.03},
                {"params": [z_L1_stiff], "lr": 0.03 * sL1},
                {"params": [z_L2_stiff], "lr": 0.03 * sL2},
                {"params": [z_L0_tau],   "lr": 0.01},
                {"params": [z_L1_tau],   "lr": 0.01 * sL1},
                {"params": [z_L2_tau],   "lr": 0.01 * sL2},
            ]
        use_hierarchy = True

    opt = torch.optim.Adam(params)
    theta_torch = torch.tensor(theta_3d_np, dtype=torch.float32, device=device,
                                 requires_grad=False)

    knife_traj = ctx["knife_traj"]
    knife_radius = ctx["knife_radius"]
    rest_state = ctx["rest_state"]
    solver = ctx["solver"]
    obs_pos_t = ctx["obs_pos_t"]
    obs_alpha_t = ctx["obs_alpha_t"]
    edges_a_t = ctx["edges_a_t"]
    edges_b_t = ctx["edges_b_t"]
    edges_rest_t = ctx["edges_rest_t"]

    peak_iou = -1.0
    peak_iou_epoch = 0
    T = len(knife_traj)
    L_gap_log = []

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

        L_gap_log.append(float(np.mean(L_gaps)))

        # Snapshot
        with torch.no_grad():
            if use_hierarchy:
                z_flat_stiff_now = reconstruct(z_L0_stiff, z_L1_stiff, z_L2_stiff, H)
            else:
                z_flat_stiff_now = z_L0_stiff
            stiff_now = (STIFF_LOW + (STIFF_HIGH - STIFF_LOW)
                            * torch.sigmoid(z_flat_stiff_now)).cpu().numpy()
        cur_iou = compute_metrics(stiff_now, stiff_truth_np, inclusion_mask)["iou"]
        if cur_iou > peak_iou:
            peak_iou = cur_iou
            peak_iou_epoch = epoch

    return dict(
        mode=mode, seed=seed,
        peak_iou=peak_iou, peak_epoch=peak_iou_epoch,
        final_L_gap=L_gap_log[-1] if L_gap_log else float("inf"),
    )


def welch_t(a, b):
    """Welch's t-test (unequal variance). Returns (t, df, two-sided p approx)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if (va / na + vb / nb) == 0:
        return float("inf"), float("inf"), 0.0
    t = (ma - mb) / np.sqrt(va / na + vb / nb)
    df_num = (va / na + vb / nb) ** 2
    df_den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = df_num / df_den if df_den > 0 else float("inf")
    # Approx two-sided p via normal tail (df ≥ ~5 OK)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return t, df, p


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    print(f"\n=== Audit: baseline vs frozen-hierarchy vs active-hierarchy ===")
    print(f"3 modes × 3 seeds = 9 runs (each ~12s)")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = 2

    device = "cuda:0"
    knife_radius = 0.08
    T = 10
    knife_traj = make_knife_trajectory(T)
    theta_truth = np.array([3.0, 100.0, 0.3], dtype=np.float32)

    pos_rest = mesh_info["vertices"]
    stiff_truth, inclusion_mask, center, incl_r = build_inclusion_truth(
        pos_rest, stiff_in=0.8, stiff_out=0.3,
        center_override=(0.15, 0.25, 0.15),
        radius_override=0.06)
    tau_truth = np.where(inclusion_mask, 0.20, 0.40).astype(np.float32)
    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)

    obs_pos, obs_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth, stiff_truth, knife_traj, knife_radius, device,
        tau_truth_np=tau_truth)
    obs_pos_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)

    # Hierarchy (FPS centers stay same across seeds — they're deterministic on
    # x_rest given start_idx=0; seed only affects torch RNG which doesn't drive FPS)
    x_rest = torch.from_numpy(pos_rest).to(device, dtype=torch.float32)
    H = build_hierarchy(x_rest, M1=512, M2=64, k_neighbors=3)

    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj=knife_traj, knife_radius=knife_radius,
        stiff_init_np=stiff_init, stiff_truth_np=stiff_truth,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        theta_3d_np=theta_truth, inclusion_mask=inclusion_mask,
        obs_pos_t=obs_pos_t, obs_alpha_t=obs_alpha_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H,
    )

    SEEDS = [0, 1, 2]
    MODES = ["baseline", "frozen_hier", "active_hier"]
    results = []
    for mode in MODES:
        for seed in SEEDS:
            print(f"\n>>> mode={mode:14s} seed={seed} …", flush=True)
            t0 = time.time()
            r = run_one(mode, seed, ctx)
            elapsed = time.time() - t0
            r["elapsed"] = elapsed
            results.append(r)
            print(f"    peak IoU = {r['peak_iou']:.4f}  (ep {r['peak_epoch']})  "
                  f"final L_gap = {r['final_L_gap']:.3e}  ({elapsed:.0f}s)")

    # Summary
    print(f"\n{'='*72}")
    print(f"AUDIT SUMMARY (peak IoU mean ± std over {len(SEEDS)} seeds)")
    print(f"{'='*72}")
    summary = {}
    for mode in MODES:
        ious = [r["peak_iou"] for r in results if r["mode"] == mode]
        summary[mode] = ious
        print(f"  {mode:14s}: {np.mean(ious):.4f} ± {np.std(ious, ddof=1):.4f}   "
              f"individual: {[f'{x:.3f}' for x in ious]}")

    # Welch's t-tests
    print(f"\n--- statistical tests (Welch's t, two-sided) ---")
    for (a, b) in [("baseline", "frozen_hier"),
                    ("baseline", "active_hier"),
                    ("frozen_hier", "active_hier")]:
        t, df, p = welch_t(summary[a], summary[b])
        marker = "★" if p < 0.05 else " "
        print(f"  {marker} {a:14s} vs {b:14s}: t={t:+.2f}  df≈{df:.1f}  p≈{p:.3f}")

    # Audit conclusions
    print(f"\n--- audit conclusions ---")
    base_mean = np.mean(summary["baseline"])
    frozen_mean = np.mean(summary["frozen_hier"])
    active_mean = np.mean(summary["active_hier"])

    if abs(frozen_mean - base_mean) < 0.01:
        print(f"  ✓ Sanity: frozen-hierarchy ≈ baseline (diff {frozen_mean-base_mean:+.4f})")
        print(f"            confirms reconstruct() reduces to identity when coarse z=0")
    else:
        print(f"  ✗ FLAG: frozen-hierarchy ≠ baseline (diff {frozen_mean-base_mean:+.4f})")
        print(f"            possible code-path inconsistency — investigate")

    _, _, p_ac = welch_t(summary["active_hier"], summary["baseline"])
    if active_mean > base_mean + 0.02 and p_ac < 0.10:
        print(f"  ✓ Active hierarchy SIGNIFICANTLY helps  (Δ={active_mean-base_mean:+.4f}, p={p_ac:.3f})")
    elif active_mean > base_mean and p_ac < 0.20:
        print(f"  ⚠ Active hierarchy MARGINALLY helps  (Δ={active_mean-base_mean:+.4f}, p={p_ac:.3f})")
    else:
        print(f"  ✗ Active hierarchy NOT statistically distinguishable from baseline")
        print(f"     Δ={active_mean-base_mean:+.4f}, p={p_ac:.3f}")
        print(f"     Implications: hierarchical doesn't help on single-trajectory")
        print(f"                   task — cross-trajectory may be needed")

    out_path = REPO / "tools" / "pointnetpp" / "audit_results.npz"
    np.savez(out_path,
              modes=np.array(MODES), seeds=np.array(SEEDS),
              peak_iou_grid=np.array([[r["peak_iou"] for r in results
                                          if r["mode"] == m and r["seed"] == s]
                                         for m in MODES for s in SEEDS]).reshape(len(MODES), len(SEEDS)))
    print(f"\n  Saved {out_path}")


if __name__ == "__main__":
    main()
