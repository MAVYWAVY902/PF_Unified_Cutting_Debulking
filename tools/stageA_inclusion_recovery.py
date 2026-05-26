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


def build_inclusion_truth(positions_np, stiff_in, stiff_out, radius_frac=None,
                            center_override=None, radius_override=None):
    """Sphere inclusion. Default center = bbox center (legacy);
    optional override for knife-overlapping placement (Option 1 sanity test).
    """
    pos = positions_np
    bbox_min = pos.min(axis=0)
    bbox_max = pos.max(axis=0)
    if center_override is not None:
        center = np.array(center_override, dtype=np.float32)
    else:
        center = 0.5 * (bbox_min + bbox_max)
    if radius_override is not None:
        incl_radius = float(radius_override)
    else:
        half_extent = 0.5 * (bbox_max - bbox_min).max()
        incl_radius = (radius_frac or 0.25) * half_extent
    dist = np.linalg.norm(pos - center, axis=1)
    mask = dist < incl_radius
    stiff_field = np.where(mask, stiff_in, stiff_out).astype(np.float32)
    return stiff_field, mask, center, incl_radius


def gen_obs_per_particle_stiff(solver, rest_state, theta_3d_np, stiff_truth_np,
                                  knife_traj, knife_radius, device,
                                  tau_truth_np=None):
    """Forward T frames at (θ, stiff_truth, [tau_truth]) → list of (pos, alpha) obs.

    Stage E (2026-05-25): tau_truth_np (N,) optional for per-particle τ obs.
    """
    theta_torch = torch.tensor(theta_3d_np, dtype=torch.float32, device=device,
                                 requires_grad=False)
    stiff_truth_torch = torch.tensor(stiff_truth_np, dtype=torch.float32, device=device,
                                       requires_grad=False)
    tau_truth_torch = None
    if tau_truth_np is not None:
        tau_truth_torch = torch.tensor(tau_truth_np, dtype=torch.float32, device=device,
                                          requires_grad=False)
    obs_pos = []
    obs_alpha = []
    state_in = rest_state
    with torch.no_grad():
        for t, knife_pos in enumerate(knife_traj):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, knife_pos, knife_radius,
                stiff_truth_torch, tau_truth_torch,
            )
            obs_pos.append(pos_t.cpu().numpy().copy())
            obs_alpha.append(alpha_t.cpu().numpy().copy())
            state_in = solver_save_state(solver)
    return obs_pos, obs_alpha


# P4b (2026-05-24): Liang-style sigmoid parameter bounding.
# Replaces hard clamp(stiff, 0.05, 5.0) — see Liang 2023 Section III-B:
# "We impose bounds on our simulation parameters with a sigmoid operation."
# Optimize unconstrained z ∈ R, stiff = LOW + RANGE × sigmoid(z).
# At sharp=100 the raw-stiff path was provably unstable (P4 75% err, Adam cliff).
STIFF_LOW = 0.05
STIFF_HIGH = 2.0


def inv_sigmoid_to_z(s, lo=STIFF_LOW, hi=STIFF_HIGH):
    """Inverse sigmoid: stiff ∈ (lo, hi) → z ∈ R for Adam optimization."""
    s_clamped = np.clip(s, lo + 1e-4, hi - 1e-4)
    return np.log((s_clamped - lo) / (hi - s_clamped)).astype(np.float32)


def z_to_stiff(z_torch, lo=STIFF_LOW, hi=STIFF_HIGH):
    """Forward sigmoid: z ∈ R → stiff ∈ (lo, hi). Differentiable."""
    return lo + (hi - lo) * torch.sigmoid(z_torch)


def run_recovery(solver, rest_state, knife_traj, knife_radius,
                   obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
                   theta_3d_np, stiff_init_np, stiff_truth_np, inclusion_mask,
                   device,
                   N_epochs=30, lr=0.03, lr_tau=0.005, n_inner=30, log_every=5,
                   live_print=True, lambda_smooth=0.0,
                   early_stop_patience=5,
                   optimize_tau=True,
                   tau_bound=(0.05, 0.95),
                   tau_per_particle=False, tau_init_np=None, tau_truth_np=None):
    """Joint Adam on (z=sigmoid-bounded stiff_pp) AND τ (theta[2]).

    Pipeline (2026-05-25, post ISSUE 1 fix + τ-optimization):
       - sigmoid bounding (Liang Section III-B) — required for sharp=100 stability
       - lr=0.03 for stiff (P4c sweet spot)
       - lr_tau=0.005 for τ (different scale, smaller lr; tunable)
       - α_min, sharpness FROZEN (grad zeroed manually after backward);
         only τ + stiff_pp updated by Adam
       - Inner GD receives outer (s, τ) via .item() — detached, so gradient
         flows to τ ONLY through forward sim (real physical path), not through
         inner formula (would be Day 3c's "fake gradient" path)
       - τ clamped to tau_bound after each Adam step

    L_total = L_gap + lambda_smooth · L_smooth
    """
    T = len(knife_traj)
    N = stiff_init_np.shape[0]
    # theta_torch is now optimizable (requires_grad=True) when optimize_tau=True
    theta_torch = torch.tensor(theta_3d_np, dtype=torch.float32, device=device,
                                 requires_grad=optimize_tau and not tau_per_particle)
    # Optimize z (unconstrained) instead of raw stiff.
    z_init = inv_sigmoid_to_z(stiff_init_np)
    z_torch = torch.tensor(z_init, dtype=torch.float32, device=device, requires_grad=True)

    # Stage E (2026-05-25): per-particle τ optimization
    z_tau_torch = None
    if tau_per_particle:
        # Initialize z_tau from tau_init_np (per-particle), sigmoid-bound
        if tau_init_np is None:
            tau_init_np = np.full(N, float(theta_3d_np[2]), dtype=np.float32)
        z_tau_init = inv_sigmoid_to_z(tau_init_np, lo=tau_bound[0], hi=tau_bound[1])
        z_tau_torch = torch.tensor(z_tau_init, dtype=torch.float32, device=device,
                                     requires_grad=True)

    # Build optimizer
    if tau_per_particle:
        # z_stiff + z_tau (no global theta in optim)
        opt = torch.optim.Adam([
            {"params": [z_torch],     "lr": lr},
            {"params": [z_tau_torch], "lr": lr_tau},
        ])
    elif optimize_tau:
        # z_stiff + global theta (only theta[2] active via grad mask)
        opt = torch.optim.Adam([
            {"params": [z_torch],     "lr": lr},
            {"params": [theta_torch], "lr": lr_tau},
        ])
    else:
        opt = torch.optim.Adam([z_torch], lr=lr)
    epoch_logs = []

    # Track best-epoch stiff (Adam drifts past sweet spot — see P4c).
    best_L_gap = float("inf")
    best_stiff = stiff_init_np.copy()
    best_tau = float(theta_3d_np[2])
    best_epoch = 0
    patience_counter = 0

    for epoch in range(N_epochs):
        state = rest_state
        L_gaps = []
        for t in range(T):
            stiff_pp_torch = z_to_stiff(z_torch)  # derive stiff via sigmoid
            # Stage E: derive per-particle τ via sigmoid if enabled
            if tau_per_particle:
                tau_pp_torch = (tau_bound[0]
                                + (tau_bound[1] - tau_bound[0])
                                * torch.sigmoid(z_tau_torch))
            else:
                tau_pp_torch = None
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state, knife_traj[t], knife_radius,
                stiff_pp_torch, tau_pp_torch,
            )
            # ISSUE 1 + τ-optimization (2026-05-25):
            # Inner sigmoid uses outer (s, τ) VALUES via .item() — detached, so
            # autograd path from inner to outer τ is SEVERED. Outer τ still
            # receives gradient via the forward-sim path (true physical sys-ID),
            # but NOT via Day 3c's "fake gradient" path through L_cav.
            # Stage E: when per-particle τ enabled, use the MEAN τ as inner constant
            # (inner doesn't yet support per-particle τ; mean is reasonable proxy).
            if tau_per_particle:
                tau_inner_val = float(tau_pp_torch.mean().item())
            else:
                tau_inner_val = float(theta_torch[2].item())
            delta_x, delta_dv, _ = gd_inner_loop_stable(
                pos_t, dv_t, obs_pos_t[t], obs_alpha_t[t],
                theta_torch, edges_a_t, edges_b_t, edges_rest_t,
                n_iter=n_inner, lr_pos=0.5, lr_dv_factor=1.0,
                lambda_cav=1.0, lambda_E=0.0,
                s_inner=float(theta_torch[1].item()),   # detached
                tau_inner=tau_inner_val,                # detached (current value)
            )
            L_gap = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
            if lambda_smooth > 0:
                neighbor_diff = stiff_pp_torch[edges_a_t] - stiff_pp_torch[edges_b_t]
                L_smooth = neighbor_diff.abs().sum()
                L_total = L_gap + lambda_smooth * L_smooth
            else:
                L_total = L_gap
            opt.zero_grad()
            L_total.backward()
            if torch.isnan(z_torch.grad).any():
                print(f"  ✗ NaN at epoch {epoch}, t={t}")
                return None
            if tau_per_particle and z_tau_torch.grad is not None and torch.isnan(z_tau_torch.grad).any():
                print(f"  ✗ NaN in z_tau at epoch {epoch}, t={t}")
                return None
            # Freeze α_min and sharpness when optimizing global θ
            if optimize_tau and not tau_per_particle and theta_torch.grad is not None:
                theta_torch.grad[0] = 0.0   # α_min frozen
                theta_torch.grad[1] = 0.0   # sharpness frozen (Day 9 lock)
            opt.step()
            # τ clamp: keep global θ[2] in valid range
            if optimize_tau and not tau_per_particle:
                with torch.no_grad():
                    theta_torch[2].clamp_(tau_bound[0], tau_bound[1])
            # NO CLAMP for stiff / z_tau — sigmoid bounding handles it.
            L_gaps.append(float(L_gap.item()))
            # Mode C inject
            state = solver_save_state(solver)
            dx_np = delta_x.detach().cpu().numpy()
            ddv_np = delta_dv.detach().cpu().numpy()
            state["pos"] = state["pos"] + dx_np
            state["particle_damage"] = np.maximum(
                state["particle_damage"] + ddv_np, 0.0)
            new_dv = state["particle_damage"]
            # Mode C alpha: use CURRENT outer (s, τ) values.
            # Stage E: when per-particle τ enabled, use per-particle τ values.
            cur_s = float(theta_torch[1].item())
            if tau_per_particle:
                cur_tau_arr = tau_pp_torch.detach().cpu().numpy()
                state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(
                    cur_s * (new_dv - cur_tau_arr), -50, 50)))
            else:
                cur_tau = float(theta_torch[2].item())
                state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(
                    cur_s * (new_dv - cur_tau), -50, 50)))

        mean_L = float(np.mean(L_gaps))
        stiff_now = z_to_stiff(z_torch).detach().cpu().numpy()
        # Capture current τ (scalar or per-particle array)
        if tau_per_particle:
            tau_now = (tau_bound[0]
                       + (tau_bound[1] - tau_bound[0])
                       * torch.sigmoid(z_tau_torch)).detach().cpu().numpy()
            tau_display = float(tau_now.mean())  # mean for logging
        else:
            tau_now = float(theta_torch[2].item())
            tau_display = tau_now
        # Track best by L_gap (proxy for sweet spot when no ground truth).
        if mean_L < best_L_gap - 1e-4:
            best_L_gap = mean_L
            best_stiff = stiff_now.copy()
            best_tau = (tau_now.copy() if isinstance(tau_now, np.ndarray) else tau_now)
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % log_every == 0 or epoch == N_epochs - 1:
            epoch_logs.append(dict(epoch=epoch, mean_L_gap=mean_L,
                                     stiff_snapshot=stiff_now.copy(),
                                     tau=(tau_now.copy() if isinstance(tau_now, np.ndarray) else tau_now)))
            if live_print:
                m = compute_metrics(stiff_now, stiff_truth_np, inclusion_mask)
                tau_str = ""
                if optimize_tau or tau_per_particle:
                    if tau_per_particle and tau_truth_np is not None:
                        # report inside/outside tau mean
                        tau_in = tau_now[inclusion_mask].mean()
                        tau_out = tau_now[~inclusion_mask].mean()
                        tau_str = f"  τ_in={tau_in:.3f} τ_out={tau_out:.3f}"
                    else:
                        tau_str = f"  τ={tau_display:.4f}"
                print(f"  ep {epoch:3d}: L_gap={mean_L:.3e}  "
                      f"L1_in={m['L1_inside']:.4f}  IoU={m['iou']:.3f}  "
                      f"stiff_in={stiff_now[inclusion_mask].mean():.3f}  "
                      f"stiff_out={stiff_now[~inclusion_mask].mean():.3f}{tau_str}")

        if patience_counter >= early_stop_patience:
            if live_print:
                print(f"  early-stop @ ep {epoch}  (best ep {best_epoch}, L_gap {best_L_gap:.3e})")
            break

    return dict(
        stiff_final=stiff_now,
        stiff_best=best_stiff,
        tau_final=tau_now,
        tau_best=best_tau,
        best_epoch=best_epoch,
        best_L_gap=best_L_gap,
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

    # Step 1 (2026-05-24): physics tuning for stronger deformation.
    # Original setup has aggressive damping + small knife → most particles
    # barely move → sys-ID signal too weak outside cutting zone.
    # Increase obs information density by relaxing physics:
    solver.damping = 0.99                  # was 0.98, now per-substep retains 99%
    solver.num_iterations = 2              # Step 2 (2026-05-24): 6 → 2 to escape near-rigid regime
    print(f"\nSolver: N={N}  E={int(solver.e['num_edges'])}")
    print(f"  [Step 1 tuning] damping: 0.98 → 0.99 (less velocity decay)")
    print(f"  [Step 2 tuning] num_iterations: 6 → 2 (residual per substep 1e-5 → 0.25, less rigid)")

    device = "cuda:0"
    knife_radius = 0.08                    # was 0.04 → 2× larger, more particles in cut zone
    print(f"  [Step 1 tuning] knife_radius: 0.04 → 0.08 (2× larger sphere)")
    T = 10  # Step 2b (2026-05-24): longer trajectory — compliant PBD's larger deformation only visible at T>3
    knife_traj = make_knife_trajectory(T)
    # Stage E (2026-05-25): per-particle τ. Truth τ also has inclusion structure.
    theta_truth = np.array([3.0, 100.0, 0.3], dtype=np.float32)  # global (mean used in fallback)
    theta_init  = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    print(f"\n[θ] global frame: truth = {theta_truth.tolist()}")
    print(f"     Stage E: per-particle τ field (tumor brittle vs tissue tough)")

    # Build inclusion ground truth — both stiff AND τ fields
    pos_rest = mesh_info["vertices"]
    stiff_truth, inclusion_mask, center, incl_r = build_inclusion_truth(
        pos_rest, stiff_in=0.8, stiff_out=0.3,
        center_override=(0.15, 0.25, 0.15),
        radius_override=0.06)
    n_incl = int(inclusion_mask.sum())
    # τ inclusion: tumor (inside) is BRITTLE → low τ, tissue (outside) TOUGH → high τ
    TAU_IN = 0.20   # brittle tumor, cuts at lower damage
    TAU_OUT = 0.40  # tough tissue, needs more damage
    tau_truth = np.where(inclusion_mask, TAU_IN, TAU_OUT).astype(np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)  # uniform init at midpoint
    print(f"\n[Ground truth] Inclusion: center={center}, radius={incl_r:.4f}")
    print(f"  Inclusion particles: {n_incl}/{N} ({n_incl/N*100:.2f}%)")
    print(f"  stiff_truth: in={stiff_truth[inclusion_mask].mean():.2f} out={stiff_truth[~inclusion_mask].mean():.2f}")
    print(f"  τ_truth    : in={TAU_IN:.2f} (brittle) out={TAU_OUT:.2f} (tough)")
    print(f"  τ_init     : uniform {tau_init[0]:.2f} (midpoint)")

    # Generate obs at TRUTH theta + truth stiff + truth τ (per-particle)
    print(f"\n[Setup] Generating obs at (theta_truth, stiff_truth, tau_truth) T={T} frames …")
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
    edge_rest_len = np.linalg.norm(
        pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)

    # Init uniform (Step 2: 0.85 → 0.5, midway between truth 0.3 outside / 0.8 inside)
    stiff_init = np.full(N, 0.5, dtype=np.float32)
    print(f"\n[Init] stiff_init: uniform {stiff_init[0]:.2f}")
    init_metrics = compute_metrics(stiff_init, stiff_truth, inclusion_mask)
    print(f"  Init metrics: L1_total={init_metrics['L1_total']:.4f}  "
          f"L1_inside={init_metrics['L1_inside']:.4f}  "
          f"L1_outside={init_metrics['L1_outside']:.4f}  IoU=N/A (uniform)")

    # Sweep λ_smooth: test 4 values to find the right regularization strength
    lambda_sweep = [0.0, 1e-6, 1e-4, 1e-2]
    print(f"\n[Run] L_smooth λ sweep: {lambda_sweep}")
    print(f"      Outer Adam: z_stiff ({N} dim) + z_τ ({N} dim) — Stage E per-particle τ")
    print(f"      lr_stiff=0.03, lr_τ=0.01, 30 epochs × T=10")
    print(f"      Sigmoid bounds: stiff ∈ [{STIFF_LOW}, {STIFF_HIGH}], τ ∈ (0.05, 0.95)")

    sweep_results = {}
    for lam in lambda_sweep:
        print(f"\n{'='*72}")
        print(f"λ_smooth = {lam}")
        print(f"{'='*72}")
        t0 = time.time()
        result = run_recovery(
            solver, rest_state, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, edges_a_t, edges_b_t, edges_rest_t,
            theta_init, stiff_init, stiff_truth, inclusion_mask, device,
            N_epochs=30, lr=0.03, lr_tau=0.01, n_inner=30, log_every=5,
            lambda_smooth=lam, early_stop_patience=5, optimize_tau=True,
            tau_per_particle=True, tau_init_np=tau_init, tau_truth_np=tau_truth)
        elapsed = time.time() - t0
        if result is None:
            print(f"\n  ✗ NaN")
            sweep_results[lam] = None
            continue
        print(f"  elapsed {elapsed:.0f}s")
        sweep_results[lam] = result

    # ==============================================================
    # Sweep summary: compare all λ_smooth values
    # ==============================================================
    print(f"\n{'='*72}")
    print(f"L_smooth SWEEP SUMMARY")
    print(f"{'='*72}")
    truth_in = stiff_truth[inclusion_mask].mean()
    truth_out = stiff_truth[~inclusion_mask].mean()
    truth_tau_in = float(tau_truth[inclusion_mask].mean())
    truth_tau_out = float(tau_truth[~inclusion_mask].mean())
    print(f"\n  {'λ_smooth':>10s}  {'L1_t':>8s}  {'L1_in':>8s}  {'L1_out':>8s}  "
          f"{'IoU':>6s}  {'s_in':>8s}  {'s_out':>8s}  {'τ_in':>8s}  {'τ_out':>8s}")
    print(f"  {'(truth)':>10s}  {'0':>8s}  {'0':>8s}  {'0':>8s}  "
          f"{'1.0':>6s}  {truth_in:8.3f}  {truth_out:8.3f}  "
          f"{truth_tau_in:8.4f}  {truth_tau_out:8.4f}")
    for lam in lambda_sweep:
        r = sweep_results.get(lam)
        if r is None:
            print(f"  {lam:>10.0e}  {'(NaN)':>10s}")
            continue
        s = r.get("stiff_best", r["stiff_final"])
        m = compute_metrics(s, stiff_truth, inclusion_mask)
        si = s[inclusion_mask].mean()
        so = s[~inclusion_mask].mean()
        # τ can be scalar OR per-particle array
        tau_val = r.get("tau_best", r.get("tau_final"))
        if isinstance(tau_val, np.ndarray):
            tau_in = tau_val[inclusion_mask].mean()
            tau_out = tau_val[~inclusion_mask].mean()
        else:
            tau_in = tau_out = float(tau_val) if tau_val is not None else 0.0
        print(f"  {lam:>10.0e}  {m['L1_total']:8.4f}  {m['L1_inside']:8.4f}  "
              f"{m['L1_outside']:8.4f}  {m['iou']:6.3f}  {si:8.3f}  {so:8.3f}  "
              f"{tau_in:8.4f}  {tau_out:8.4f}")

    # Pick best λ by IoU
    best_lam = max(sweep_results.keys(),
                     key=lambda l: (sweep_results[l]["stiff_final"].sum() if sweep_results[l] is not None else -float('inf'),
                                     compute_metrics(sweep_results[l]["stiff_final"], stiff_truth, inclusion_mask)['iou'] if sweep_results[l] is not None else -float('inf'))[1] if sweep_results[l] is not None else -float('inf'))
    best_r = sweep_results.get(best_lam)
    if best_r is not None:
        best_m = compute_metrics(best_r["stiff_final"], stiff_truth, inclusion_mask)
        print(f"\n  Best λ (by IoU): {best_lam:.0e}  →  IoU={best_m['iou']:.3f}")

    # Use best λ result as "main" for detailed report
    result = best_r if best_r is not None else sweep_results[0.0]
    if result is None:
        print(f"\n✗ All λ values failed")
        return
    # Prefer best-epoch over final (Adam drifts past sweet spot — P4c finding).
    stiff_recovered = result.get("stiff_best", result["stiff_final"])
    if "best_epoch" in result:
        print(f"\n  Using best-epoch result (ep {result['best_epoch']}, L_gap {result['best_L_gap']:.3e})")
    final = compute_metrics(stiff_recovered, stiff_truth, inclusion_mask)
    print(f"\n{'='*72}")
    print(f"DETAILED REPORT (best λ={best_lam:.0e})")
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
          f"mean={stiff_recovered[inclusion_mask].mean():.3f}  (truth={stiff_truth[inclusion_mask].mean():.3f})")
    print(f"  Outside inclusion: min={stiff_recovered[~inclusion_mask].min():.3f}, "
          f"max={stiff_recovered[~inclusion_mask].max():.3f}, "
          f"mean={stiff_recovered[~inclusion_mask].mean():.3f}  (truth={stiff_truth[~inclusion_mask].mean():.3f})")

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
