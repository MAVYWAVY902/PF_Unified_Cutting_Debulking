"""Stage A-4: end-to-end sys-ID recovery of per-particle stiffness inclusion.

The actual paper-level validation: given OBS generated at heterogeneous stiff
field (center inclusion), can our Stage A pipeline LEARN the spatial map from
uniform init?

Setup:
   θ_global = (α=3.0, sharp=20, τ=0.5) FROZEN
   stiff_truth: center sphere (radius = 0.25 * half-extent) = 1.5, outside = 0.3
   stiff_init: uniform 0.85 (no prior knowledge of inclusion)

   Generate obs at (θ_global, stiff_truth) via WarpFullForwardOp T=3 frames.
   Optimize stiff_pp_torch with Adam to minimize L_gap.

Metrics:
   - Per-particle L1: mean |stiff_recovered - stiff_truth|
   - Inclusion IoU: top-X% recovered stiff vs true inclusion particles
   - Visualization: 3D scatter colored by stiff value
"""
import sys
from pathlib import Path
import numpy as np
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))

import torch
import warp as wp
wp.init()
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

from tools.day2_solver_wrapper import (  # noqa: E402
    setup_solver_with_rest, solver_save_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day8_stable_inner_loop import (  # noqa: E402
    gd_inner_loop_stable, SHARPNESS_INNER, TAU_INNER,
)
from tools.day6f_v0_multistep_synth import make_knife_trajectory  # noqa: E402


def build_inclusion_truth(positions_np, stiff_in, stiff_out, radius_frac=0.25):
    """Center sphere of stiff_in, outside of stiff_out."""
    pos = positions_np
    bbox_min = pos.min(axis=0)
    bbox_max = pos.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    half_extent = 0.5 * (bbox_max - bbox_min).max()
    incl_radius = radius_frac * half_extent
    dist = np.linalg.norm(pos - center, axis=1)
    mask = dist < incl_radius
    stiff_field = np.where(mask, stiff_in, stiff_out).astype(np.float32)
    return stiff_field, mask, center, incl_radius


def gen_obs_per_particle_stiff(solver, rest_state, theta_3d_np, stiff_truth_np,
                                  knife_traj, knife_radius, device):
    """Forward T frames at (θ, stiff_truth) → list of (pos, alpha) obs."""
    theta_torch = torch.tensor(theta_3d_np, dtype=torch.float32, device=device,
                                 requires_grad=False)
    stiff_truth_torch = torch.tensor(stiff_truth_np, dtype=torch.float32, device=device,
                                       requires_grad=False)
    obs_pos = []
    obs_alpha = []
    state_in = rest_state
    with torch.no_grad():
        for t, knife_pos in enumerate(knife_traj):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, knife_pos, knife_radius,
                stiff_truth_torch,
            )
            obs_pos.append(pos_t.cpu().numpy().copy())
            obs_alpha.append(alpha_t.cpu().numpy().copy())
            state_in = solver_save_state(solver)
    return obs_pos, obs_alpha


def run_recovery(solver, rest_state, knife_traj, knife_radius,
                   obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
                   theta_3d_np, stiff_init_np, stiff_truth_np, inclusion_mask,
                   device,
                   N_epochs=100, lr=0.05, n_inner=30, log_every=5,
                   live_print=True):
    """Adam on stiff_pp only. θ_global frozen."""
    T = len(knife_traj)
    theta_torch = torch.tensor(theta_3d_np, dtype=torch.float32, device=device,
                                 requires_grad=False)
    stiff_pp_torch = torch.tensor(stiff_init_np, dtype=torch.float32, device=device,
                                    requires_grad=True)
    opt = torch.optim.Adam([stiff_pp_torch], lr=lr)
    epoch_logs = []

    for epoch in range(N_epochs):
        state = rest_state
        L_gaps = []
        for t in range(T):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state, knife_traj[t], knife_radius,
                stiff_pp_torch,
            )
            delta_x, delta_dv, _ = gd_inner_loop_stable(
                pos_t, dv_t, obs_pos_t[t], obs_alpha_t[t],
                theta_torch, edges_a_t, edges_b_t, edges_rest_t,
                n_iter=n_inner, lr_pos=0.5, lr_dv_factor=1.0,
                lambda_cav=1.0, lambda_E=0.0,
            )
            L_gap = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
            opt.zero_grad()
            L_gap.backward()
            if torch.isnan(stiff_pp_torch.grad).any():
                print(f"  ✗ NaN at epoch {epoch}, t={t}")
                return None
            opt.step()
            # Clamp stiff to reasonable physical range
            with torch.no_grad():
                stiff_pp_torch.clamp_(0.05, 5.0)
            L_gaps.append(float(L_gap.item()))
            # Mode C inject
            state = solver_save_state(solver)
            dx_np = delta_x.detach().cpu().numpy()
            ddv_np = delta_dv.detach().cpu().numpy()
            state["pos"] = state["pos"] + dx_np
            state["particle_damage"] = np.maximum(
                state["particle_damage"] + ddv_np, 0.0)
            new_dv = state["particle_damage"]
            state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(
                SHARPNESS_INNER * (new_dv - TAU_INNER), -50, 50)))
        if epoch % log_every == 0 or epoch == N_epochs - 1:
            stiff_now = stiff_pp_torch.detach().cpu().numpy()
            epoch_logs.append(dict(epoch=epoch, mean_L_gap=np.mean(L_gaps),
                                     stiff_snapshot=stiff_now.copy()))
            if live_print:
                m = compute_metrics(stiff_now, stiff_truth_np, inclusion_mask)
                print(f"  ep {epoch:3d}: L_gap={np.mean(L_gaps):.3e}  "
                      f"L1_total={m['L1_total']:.4f}  L1_in={m['L1_inside']:.4f}  "
                      f"L1_out={m['L1_outside']:.4f}  IoU={m['iou']:.3f}  "
                      f"stiff_in={stiff_now[inclusion_mask].mean():.3f}  "
                      f"stiff_out={stiff_now[~inclusion_mask].mean():.3f}")
    return dict(
        stiff_final=stiff_pp_torch.detach().cpu().numpy(),
        epoch_logs=epoch_logs,
    )


def compute_metrics(stiff_recovered, stiff_truth, inclusion_mask):
    """Per-particle L1 + inclusion IoU."""
    N = len(stiff_truth)
    # L1
    L1_total = float(np.mean(np.abs(stiff_recovered - stiff_truth)))
    L1_inside = float(np.mean(np.abs(stiff_recovered[inclusion_mask]
                                       - stiff_truth[inclusion_mask])))
    L1_outside = float(np.mean(np.abs(stiff_recovered[~inclusion_mask]
                                        - stiff_truth[~inclusion_mask])))

    # Spatial IoU: take top-K recovered (K = n_inclusion), check overlap
    n_incl = int(inclusion_mask.sum())
    if n_incl > 0:
        top_k_idx = np.argsort(stiff_recovered)[::-1][:n_incl]
        true_incl_idx = np.where(inclusion_mask)[0]
        intersection = len(set(top_k_idx) & set(true_incl_idx))
        union = len(set(top_k_idx) | set(true_incl_idx))
        iou = intersection / union if union > 0 else 0
    else:
        iou = 0
    return dict(L1_total=L1_total, L1_inside=L1_inside, L1_outside=L1_outside,
                 iou=iou, n_inclusion=n_incl)


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    N = solver.N
    print(f"\nSolver: N={N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_radius = 0.04
    T = 3
    knife_traj = make_knife_trajectory(T)
    theta_3D = np.array([3.0, 20.0, 0.5], dtype=np.float32)

    # Build inclusion ground truth
    pos_rest = mesh_info["vertices"]
    stiff_truth, inclusion_mask, center, incl_r = build_inclusion_truth(
        pos_rest, stiff_in=1.5, stiff_out=0.3, radius_frac=0.25)
    n_incl = int(inclusion_mask.sum())
    print(f"\n[Ground truth] Inclusion: center={center}, radius={incl_r:.4f}")
    print(f"  Inclusion particles: {n_incl}/{N} ({n_incl/N*100:.2f}%)")
    print(f"  stiff_truth: range [{stiff_truth.min():.2f}, {stiff_truth.max():.2f}]")

    # Generate obs
    print(f"\n[Setup] Generating obs at (θ, stiff_truth) for T={T} frames …")
    obs_pos, obs_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_3D, stiff_truth, knife_traj, knife_radius, device)
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
    edge_rest_len = np.linalg.norm(
        pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)

    # Init uniform
    stiff_init = np.full(N, 0.85, dtype=np.float32)
    print(f"\n[Init] stiff_init: uniform {stiff_init[0]:.2f}")
    init_metrics = compute_metrics(stiff_init, stiff_truth, inclusion_mask)
    print(f"  Init metrics: L1_total={init_metrics['L1_total']:.4f}  "
          f"L1_inside={init_metrics['L1_inside']:.4f}  "
          f"L1_outside={init_metrics['L1_outside']:.4f}  IoU=N/A (uniform)")

    # Run recovery
    print(f"\n[Run] Outer Adam on stiff_pp ({N} dim), 60 epochs × T=3 = 180 Adam steps")
    print(f"      θ_global = {theta_3D.tolist()} FROZEN")
    print(f"      lr=0.01 (reduced from 0.05 due to per-particle 21745-dim sensitivity)")
    t0 = time.time()
    result = run_recovery(
        solver, rest_state, knife_traj, knife_radius,
        obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
        theta_3D, stiff_init, stiff_truth, inclusion_mask, device,
        N_epochs=60, lr=0.01, n_inner=30, log_every=5)
    elapsed = time.time() - t0
    if result is None:
        print(f"\n✗ Recovery FAILED (NaN)")
        return
    print(f"  elapsed {elapsed:.0f}s")

    # Per-epoch metrics
    print(f"\n[Per-epoch trajectory]")
    print(f"  {'ep':>4s}  {'L_gap mean':>12s}  {'L1_total':>10s}  {'L1_inside':>10s}  "
          f"{'L1_outside':>10s}  {'IoU':>6s}")
    for log in result["epoch_logs"]:
        m = compute_metrics(log["stiff_snapshot"], stiff_truth, inclusion_mask)
        print(f"  {log['epoch']:>4d}  {log['mean_L_gap']:12.4e}  "
              f"{m['L1_total']:10.4f}  {m['L1_inside']:10.4f}  "
              f"{m['L1_outside']:10.4f}  {m['iou']:6.3f}")

    # Final metrics
    stiff_recovered = result["stiff_final"]
    final = compute_metrics(stiff_recovered, stiff_truth, inclusion_mask)
    print(f"\n{'='*72}")
    print(f"FINAL METRICS")
    print(f"{'='*72}")
    print(f"  Per-particle L1 (total):         {final['L1_total']:.4f}")
    print(f"  Per-particle L1 (inclusion):     {final['L1_inside']:.4f}")
    print(f"  Per-particle L1 (outside):       {final['L1_outside']:.4f}")
    print(f"  Spatial IoU (top-K vs true):     {final['iou']:.3f}")

    # Improvement over init
    init = init_metrics
    print(f"\n  Vs. init:")
    print(f"    L1_total: {init['L1_total']:.4f} → {final['L1_total']:.4f}  "
          f"(× {final['L1_total']/init['L1_total']:.2f})")
    print(f"    L1_inside: {init['L1_inside']:.4f} → {final['L1_inside']:.4f}  "
          f"(× {final['L1_inside']/init['L1_inside']:.2f})")

    # Recovered stiff distribution
    print(f"\n[Recovered stiff distribution]")
    print(f"  Overall: min={stiff_recovered.min():.3f}, max={stiff_recovered.max():.3f}, "
          f"mean={stiff_recovered.mean():.3f}")
    print(f"  Inside inclusion: min={stiff_recovered[inclusion_mask].min():.3f}, "
          f"max={stiff_recovered[inclusion_mask].max():.3f}, "
          f"mean={stiff_recovered[inclusion_mask].mean():.3f}  (truth=1.500)")
    print(f"  Outside inclusion: min={stiff_recovered[~inclusion_mask].min():.3f}, "
          f"max={stiff_recovered[~inclusion_mask].max():.3f}, "
          f"mean={stiff_recovered[~inclusion_mask].mean():.3f}  (truth=0.300)")

    # Save for visualization
    np.savez(str(REPO / "tools" / "stageA_inclusion_recovery.npz"),
              stiff_truth=stiff_truth,
              stiff_recovered=stiff_recovered,
              stiff_init=stiff_init,
              positions=pos_rest,
              inclusion_mask=inclusion_mask,
              )
    print(f"\n  saved → tools/stageA_inclusion_recovery.npz")

    # Verdict
    print(f"\n{'='*72}")
    print(f"VERDICT")
    print(f"{'='*72}")
    if final["iou"] > 0.6 and final["L1_total"] < init["L1_total"] * 0.5:
        print(f"  ✓ STRONG: spatial inclusion recovered (IoU>0.6, L1<half init)")
    elif final["iou"] > 0.3 and final["L1_total"] < init["L1_total"] * 0.8:
        print(f"  ⚠ PARTIAL: some spatial signal but not clean recovery")
    else:
        print(f"  ✗ WEAK: inclusion not recovered. May need:")
        print(f"     - L_smooth regularization (Liang Eq. 9)")
        print(f"     - More epochs / different lr")
        print(f"     - Different obs (surface-only or force)")


if __name__ == "__main__":
    main()
