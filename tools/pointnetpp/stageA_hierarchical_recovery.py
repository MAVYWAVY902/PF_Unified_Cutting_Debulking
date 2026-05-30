"""Phase 1.2/1.3 — Stage A inclusion recovery with PointNet++-inspired
hierarchical (z_L0, z_L1, z_L2) reparameterization of stiff_pp and tau_pp.

This is the hierarchical variant of tools/stageA_inclusion_recovery.py. It
imports the obs generation / metrics / inclusion truth helpers verbatim, and
replaces only the parameter representation in the inner Adam loop:

    Before (baseline):
        z_stiff ∈ R^N (per-particle)        Adam tunes N scalars
        z_tau   ∈ R^N (per-particle)        Adam tunes N scalars

    After (hierarchical):
        z_stiff = (z_L0 ∈ R^N, z_L1 ∈ R^M1, z_L2 ∈ R^M2)
        z_tau   = (z_L0 ∈ R^N, z_L1 ∈ R^M1, z_L2 ∈ R^M2)
        stiff_pp = LOW + RANGE * sigmoid(z_L0 + Σ_l w_l(i)·z_L_l[c_l(i)])
        same for tau_pp
        Adam tunes (N + M1 + M2) scalars × 2 = ~2N scalars

Forward physics is unchanged: stiff_pp / tau_pp tensors fed to
WarpFullForwardOp identically.

Phase 1.3 gate (smoke test, re-measured 2026-05-29): IoU ≥ baseline ≈ 0.20.
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
    inv_sigmoid_to_z, STIFF_LOW, STIFF_HIGH,
)
from tools.pointnetpp.hierarchy import build_hierarchy, reconstruct  # noqa: E402


def run_recovery_hierarchical(
    solver, rest_state, knife_traj, knife_radius,
    obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
    theta_3d_np, stiff_init_np, stiff_truth_np, inclusion_mask,
    tau_init_np, tau_truth_np, device,
    H,
    N_epochs=30, lr=0.03, lr_tau=0.01, n_inner=30, log_every=5,
    early_stop_patience=5,
    tau_bound=(0.05, 0.95),
):
    """Hierarchical Adam over (z_L0, z_L1, z_L2) for both stiff and tau."""
    T = len(knife_traj)
    N = stiff_init_np.shape[0]
    assert N == H.N, f"H.N={H.N} != N={N}"
    M1, M2 = H.M1, H.M2

    # Frozen global θ (α_min, sharp, τ_mean). τ_mean used only as inner-loop
    # constant when per-particle τ active; outer optimization is per-particle.
    theta_torch = torch.tensor(theta_3d_np, dtype=torch.float32, device=device,
                                 requires_grad=False)

    # Init z_L0 from per-particle init values, z_L1 / z_L2 zero.
    # When all coarser levels = 0, reconstruct() returns z_L0 → matches init.
    z_L0_stiff = torch.tensor(inv_sigmoid_to_z(stiff_init_np),
                                dtype=torch.float32, device=device, requires_grad=True)
    z_L1_stiff = torch.zeros(M1, dtype=torch.float32, device=device, requires_grad=True)
    z_L2_stiff = torch.zeros(M2, dtype=torch.float32, device=device, requires_grad=True)

    z_L0_tau = torch.tensor(inv_sigmoid_to_z(tau_init_np, lo=tau_bound[0], hi=tau_bound[1]),
                              dtype=torch.float32, device=device, requires_grad=True)
    z_L1_tau = torch.zeros(M1, dtype=torch.float32, device=device, requires_grad=True)
    z_L2_tau = torch.zeros(M2, dtype=torch.float32, device=device, requires_grad=True)

    # LR-balanced per level. Coarse z's accumulate gradient from ~N/M particles
    # → bigger raw grad → smaller lr to keep effective step ~constant per param.
    # Scale: lr_L_l = lr_base / sqrt(N / M_l)  (sqrt of fan-in normalization)
    import math
    scale_L1 = 1.0 / math.sqrt(N / M1)
    scale_L2 = 1.0 / math.sqrt(N / M2)
    print(f"  [LR balance] lr_L0=1.0× lr_L1={scale_L1:.4f}× lr_L2={scale_L2:.4f}×  (× lr_base)")
    opt = torch.optim.Adam([
        {"params": [z_L0_stiff], "lr": lr},
        {"params": [z_L1_stiff], "lr": lr * scale_L1},
        {"params": [z_L2_stiff], "lr": lr * scale_L2},
        {"params": [z_L0_tau],   "lr": lr_tau},
        {"params": [z_L1_tau],   "lr": lr_tau * scale_L1},
        {"params": [z_L2_tau],   "lr": lr_tau * scale_L2},
    ])

    best_L_gap = float("inf")
    best_stiff = stiff_init_np.copy()
    best_tau = tau_init_np.copy()
    best_epoch = 0
    patience_counter = 0
    epoch_logs = []
    # Also track peak IoU separately — L_gap "best" doesn't always coincide
    # with best inclusion recovery (hierarchical can over-smooth past sweet spot).
    peak_iou = -1.0
    peak_iou_stiff = stiff_init_np.copy()
    peak_iou_tau = tau_init_np.copy()
    peak_iou_epoch = 0

    for epoch in range(N_epochs):
        state = rest_state
        L_gaps = []
        for t in range(T):
            # Reconstruct stiff and tau per-particle latents from hierarchy
            z_flat_stiff = reconstruct(z_L0_stiff, z_L1_stiff, z_L2_stiff, H)
            stiff_pp_torch = STIFF_LOW + (STIFF_HIGH - STIFF_LOW) * torch.sigmoid(z_flat_stiff)

            z_flat_tau = reconstruct(z_L0_tau, z_L1_tau, z_L2_tau, H)
            tau_pp_torch = tau_bound[0] + (tau_bound[1] - tau_bound[0]) * torch.sigmoid(z_flat_tau)

            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state, knife_traj[t], knife_radius,
                stiff_pp_torch, tau_pp_torch,
            )

            # Inner loop uses outer θ values via .item() — detached path
            tau_inner_val = float(tau_pp_torch.mean().item())
            delta_x, delta_dv, _ = gd_inner_loop_stable(
                pos_t, dv_t, obs_pos_t[t], obs_alpha_t[t],
                theta_torch, edges_a_t, edges_b_t, edges_rest_t,
                n_iter=n_inner, lr_pos=0.5, lr_dv_factor=1.0,
                lambda_cav=1.0, lambda_E=0.0,
                s_inner=float(theta_torch[1].item()),
                tau_inner=tau_inner_val,
            )
            L_gap = (delta_x ** 2).sum() + (delta_dv ** 2).sum()

            opt.zero_grad()
            L_gap.backward()
            # NaN check on all 6 latents
            for name, t_ in [("z_L0_stiff", z_L0_stiff), ("z_L1_stiff", z_L1_stiff),
                              ("z_L2_stiff", z_L2_stiff), ("z_L0_tau", z_L0_tau),
                              ("z_L1_tau", z_L1_tau), ("z_L2_tau", z_L2_tau)]:
                if t_.grad is not None and torch.isnan(t_.grad).any():
                    print(f"  ✗ NaN in {name} at epoch {epoch}, t={t}")
                    return None

            opt.step()
            L_gaps.append(float(L_gap.item()))

            # Mode C inject — same pattern as baseline
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

        # Snapshot current stiff/tau in physical units
        with torch.no_grad():
            z_flat_stiff_now = reconstruct(z_L0_stiff, z_L1_stiff, z_L2_stiff, H)
            stiff_now = (STIFF_LOW + (STIFF_HIGH - STIFF_LOW)
                            * torch.sigmoid(z_flat_stiff_now)).cpu().numpy()
            z_flat_tau_now = reconstruct(z_L0_tau, z_L1_tau, z_L2_tau, H)
            tau_now = (tau_bound[0] + (tau_bound[1] - tau_bound[0])
                          * torch.sigmoid(z_flat_tau_now)).cpu().numpy()

        if mean_L < best_L_gap - 1e-4:
            best_L_gap = mean_L
            best_stiff = stiff_now.copy()
            best_tau = tau_now.copy()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # Track peak IoU separately
        cur_iou = compute_metrics(stiff_now, stiff_truth_np, inclusion_mask)["iou"]
        if cur_iou > peak_iou:
            peak_iou = cur_iou
            peak_iou_stiff = stiff_now.copy()
            peak_iou_tau = tau_now.copy()
            peak_iou_epoch = epoch

        if epoch % log_every == 0 or epoch == N_epochs - 1:
            m = compute_metrics(stiff_now, stiff_truth_np, inclusion_mask)
            tau_in = tau_now[inclusion_mask].mean()
            tau_out = tau_now[~inclusion_mask].mean()
            epoch_logs.append(dict(epoch=epoch, mean_L_gap=mean_L,
                                     stiff_snapshot=stiff_now.copy(),
                                     tau_snapshot=tau_now.copy()))
            print(f"  ep {epoch:3d}: L_gap={mean_L:.3e}  "
                  f"L1_in={m['L1_inside']:.4f}  IoU={m['iou']:.3f}  "
                  f"stiff_in={stiff_now[inclusion_mask].mean():.3f}  "
                  f"stiff_out={stiff_now[~inclusion_mask].mean():.3f}  "
                  f"τ_in={tau_in:.3f} τ_out={tau_out:.3f}  "
                  f"|z_L1|={z_L1_stiff.abs().mean().item():.3f} "
                  f"|z_L2|={z_L2_stiff.abs().mean().item():.3f}")

        if patience_counter >= early_stop_patience:
            print(f"  early-stop @ ep {epoch}  (best ep {best_epoch}, L_gap {best_L_gap:.3e})")
            break

    return dict(
        stiff_final=stiff_now,
        stiff_best=best_stiff,
        tau_final=tau_now,
        tau_best=best_tau,
        best_epoch=best_epoch,
        best_L_gap=best_L_gap,
        stiff_peak_iou=peak_iou_stiff,
        tau_peak_iou=peak_iou_tau,
        peak_iou=peak_iou,
        peak_iou_epoch=peak_iou_epoch,
        epoch_logs=epoch_logs,
        H=H,
        z_L0_stiff_final=z_L0_stiff.detach().cpu().numpy(),
        z_L1_stiff_final=z_L1_stiff.detach().cpu().numpy(),
        z_L2_stiff_final=z_L2_stiff.detach().cpu().numpy(),
    )


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    N = solver.N

    # Same physics tuning as baseline Stage A
    solver.damping = 0.99
    solver.num_iterations = 2
    print(f"\nSolver: N={N}  E={int(solver.e['num_edges'])}")
    print(f"  damping=0.99  num_iterations=2  (matches baseline Stage A)")

    device = "cuda:0"
    knife_radius = 0.08
    T = 10
    knife_traj = make_knife_trajectory(T)
    theta_truth = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    theta_init  = np.array([3.0, 100.0, 0.3], dtype=np.float32)

    # Truth inclusion: same as baseline
    pos_rest = mesh_info["vertices"]
    stiff_truth, inclusion_mask, center, incl_r = build_inclusion_truth(
        pos_rest, stiff_in=0.8, stiff_out=0.3,
        center_override=(0.15, 0.25, 0.15),
        radius_override=0.06)
    n_incl = int(inclusion_mask.sum())
    TAU_IN, TAU_OUT = 0.20, 0.40
    tau_truth = np.where(inclusion_mask, TAU_IN, TAU_OUT).astype(np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)
    stiff_init = np.full(N, 0.5, dtype=np.float32)

    print(f"\n[Truth] center={center}, radius={incl_r:.4f}, "
          f"incl particles: {n_incl}/{N} ({n_incl/N*100:.2f}%)")
    print(f"  stiff: in={stiff_truth[inclusion_mask].mean():.2f} out={stiff_truth[~inclusion_mask].mean():.2f}")
    print(f"  τ:     in={TAU_IN} (brittle) out={TAU_OUT} (tough)")

    # Generate obs at truth
    print(f"\n[Setup] Generating obs at truth (T={T}) …")
    obs_pos, obs_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth, stiff_truth, knife_traj, knife_radius, device,
        tau_truth_np=tau_truth)
    cavity = [float((a < 0.5).mean()) for a in obs_alpha]
    print(f"  cavity fraction per frame: {[f'{c:.4f}' for c in cavity]}")
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

    # Init metrics
    init_metrics = compute_metrics(stiff_init, stiff_truth, inclusion_mask)
    print(f"\n[Init] stiff_init: uniform {stiff_init[0]:.2f} → "
          f"L1_total={init_metrics['L1_total']:.4f} L1_inside={init_metrics['L1_inside']:.4f}")

    # Build PointNet++-inspired hierarchy on x_rest
    print(f"\n[Hierarchy] Building FPS centers on x_rest (N={N}) …")
    x_rest = torch.from_numpy(pos_rest).to(device, dtype=torch.float32)
    H = build_hierarchy(x_rest, M1=512, M2=64, k_neighbors=3)
    print(f"  L0 (per-particle): {N}")
    print(f"  L1 (mid-scale)  : {H.M1} centers")
    print(f"  L2 (coarse)     : {H.M2} centers")

    # Run hierarchical recovery
    print(f"\n[Run] Hierarchical recovery: Adam over 6 latents")
    print(f"  z_stiff: L0({N}) + L1({H.M1}) + L2({H.M2}) = {N + H.M1 + H.M2} scalars")
    print(f"  z_tau:   L0({N}) + L1({H.M1}) + L2({H.M2}) = {N + H.M1 + H.M2} scalars")
    print(f"  lr_stiff=0.03  lr_tau=0.01  30 epochs × T={T}")

    t0 = time.time()
    result = run_recovery_hierarchical(
        solver, rest_state, knife_traj, knife_radius,
        obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
        theta_init, stiff_init, stiff_truth, inclusion_mask,
        tau_init, tau_truth, device, H,
        N_epochs=30, lr=0.03, lr_tau=0.01, n_inner=30, log_every=1,
        early_stop_patience=15,
    )
    elapsed = time.time() - t0

    if result is None:
        print(f"\n✗ NaN — recovery failed")
        return False

    stiff_recovered = result["stiff_best"]            # by L_gap proxy
    tau_recovered = result["tau_best"]
    stiff_peak = result["stiff_peak_iou"]              # by IoU
    tau_peak = result["tau_peak_iou"]
    final = compute_metrics(stiff_recovered, stiff_truth, inclusion_mask)
    peak_metrics = compute_metrics(stiff_peak, stiff_truth, inclusion_mask)
    print(f"\n{'='*72}")
    print(f"HIERARCHICAL RESULT (elapsed {elapsed:.0f}s)")
    print(f"{'='*72}")
    print(f"  -- by L_gap best (ep {result['best_epoch']}) --")
    print(f"  Per-particle L1 total:     {final['L1_total']:.4f}  (init {init_metrics['L1_total']:.4f})")
    print(f"  Per-particle L1 inside:    {final['L1_inside']:.4f}  (init {init_metrics['L1_inside']:.4f})")
    print(f"  Spatial IoU (top-K):       {final['iou']:.3f}")
    print(f"  -- by peak IoU (ep {result['peak_iou_epoch']}) --")
    print(f"  Peak IoU:                  {peak_metrics['iou']:.3f}")
    print(f"  L1 total at peak:          {peak_metrics['L1_total']:.4f}")
    print(f"  L1 inside at peak:         {peak_metrics['L1_inside']:.4f}")
    print(f"  stiff_in mean: {stiff_peak[inclusion_mask].mean():.3f} (truth {stiff_truth[inclusion_mask].mean():.2f})")
    print(f"  stiff_out mean: {stiff_peak[~inclusion_mask].mean():.3f} (truth {stiff_truth[~inclusion_mask].mean():.2f})")
    print(f"  τ_in mean: {tau_peak[inclusion_mask].mean():.3f} (truth {TAU_IN})")
    print(f"  τ_out mean: {tau_peak[~inclusion_mask].mean():.3f} (truth {TAU_OUT})")

    # Save for viz / comparison
    out_path = REPO / "tools" / "pointnetpp" / "stageA_hierarchical_recovery.npz"
    np.savez(
        out_path,
        stiff_truth=stiff_truth, stiff_recovered=stiff_recovered, stiff_init=stiff_init,
        tau_truth=tau_truth, tau_recovered=tau_recovered, tau_init=tau_init,
        positions=pos_rest, inclusion_mask=inclusion_mask,
        z_L0_stiff=result["z_L0_stiff_final"],
        z_L1_stiff=result["z_L1_stiff_final"],
        z_L2_stiff=result["z_L2_stiff_final"],
        centers_L1=H.centers_L1.cpu().numpy(),
        centers_L2=H.centers_L2.cpu().numpy(),
    )
    print(f"\nSaved {out_path}")

    # Gate — current baseline (re-measured on this branch) peaks at IoU ≈ 0.20
    BASELINE_IOU = 0.20
    iou_for_gate = peak_metrics["iou"]
    print(f"\n{'='*72}")
    print(f"PHASE 1.3 GATE  (baseline IoU peak ≈ {BASELINE_IOU:.2f} on this branch)")
    print(f"{'='*72}")
    if iou_for_gate >= BASELINE_IOU:
        print(f"  ✓ PASS — hierarchical peak IoU {iou_for_gate:.3f} ≥ baseline {BASELINE_IOU:.2f}")
        ok = True
    else:
        print(f"  ⚠ BELOW baseline — peak IoU {iou_for_gate:.3f} < {BASELINE_IOU:.2f}")
        print(f"      Next: (a) larger M1/M2, (b) more epochs, (c) different lr scaling")
        ok = False
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
