"""Day 3b AUDIT: stiffness identifiability — Steps 1 + 2 of real methodology.

Methodology check (no hack): before claiming 4D sys-ID can recover stiffness,
we MUST verify (a) stiffness has measurable signal in α output, and
(b) L_α(stiffness) landscape is unimodal near truth.

Forward-only — no inner loop, no Adam. Just sim(θ) → α, compute differences.

STEP 1 — SENSITIVITY PROBE:
  At θ_true = (3, 20, 0.5, 0.85), perturb each of 4 params by ±10%.
  Measure |Δα|_2 / (relative perturbation magnitude).
  Compare 4 dims apples-to-apples.

  Pass: stiffness sensitivity within 0.1× to 10× of cohesive dims.
  Fail: stiffness sensitivity < 0.01× others → stiffness invisible to L_α.

STEP 2 — STIFFNESS LANDSCAPE:
  Fix (α, sharp, τ) at truth. Sweep stiffness ∈ [0.1, 1.5] × 20 points.
  Compute L_α(stiff) = ||α(stiff) - α_truth||² / N. Plot.

  Pass: L_α has clear unimodal minimum near 0.85.
  Fail: flat / multimodal / noisy → recovery via gradient descent impossible.

Setup mirrors calibrate_resect.py:
  T=3 multi-step forward, Mode-C-style state chaining (state_t → state_{t+1})
  via solver_save_state. Knife trajectory from make_knife_trajectory.
  Loss target: cube_tet.msh, knife_radius=0.04, sharpness frozen=20.
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pf_shape_matching_warp"))

import warp as wp
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
wp.init()
print(f"Warp: {wp.__version__}  Torch: {torch.__version__}")

from tools.day2_solver_wrapper import (  # noqa: E402
    setup_solver_with_rest, solver_load_state, solver_save_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day6f_v0_multistep_synth import make_knife_trajectory  # noqa: E402


PARAM_NAMES = ["α_min", "sharpness", "τ", "stiffness"]


def forward_multistep(solver, rest_state, theta_np, knife_traj, knife_radius, device):
    """T-step forward, state-chained. Returns list of α (T,N) and final state.

    Uses WarpFullForwardOp.apply (param_arr path) for consistency with sys-ID.
    Forward only — no autograd needed but uses same kernel chain.
    """
    theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=device,
                                 requires_grad=False)
    alphas = []
    state_in = rest_state
    with torch.no_grad():
        for t, knife_pos in enumerate(knife_traj):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, knife_pos, knife_radius,
            )
            alphas.append(alpha_t.cpu().numpy().copy())
            state_in = solver_save_state(solver)
    return alphas


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_radius = 0.04
    T = 3
    knife_traj = make_knife_trajectory(T)
    theta_true = np.array([3.0, 20.0, 0.5, 0.85], dtype=np.float32)

    # ---- Ground truth alphas (T frames) ----
    print(f"\n[Setup] T={T}, θ_true={theta_true.tolist()}, knife r={knife_radius}")
    print(f"[Setup] Generating α_truth (T={T} frames) …")
    alphas_truth = forward_multistep(solver, rest_state, theta_true,
                                       knife_traj, knife_radius, device)
    a_truth_flat = np.concatenate([a for a in alphas_truth])
    print(f"  truth cavity per frame: " +
          ", ".join(f"{float((a<0.5).mean()):.4f}" for a in alphas_truth))

    # ============ STEP 1: Sensitivity probe ============
    print(f"\n{'='*72}")
    print(f"STEP 1 — Sensitivity probe (|Δα|_2 / relative perturbation)")
    print(f"{'='*72}")
    print(f"At θ_true, perturb each dim ±10% (relative), measure ‖α - α_truth‖_2 / N")
    print()
    print(f"  {'param':>10s}  {'+10% err':>14s}  {'-10% err':>14s}  "
          f"{'avg':>14s}  {'normalized':>12s}")
    sens = {}
    for d, name in enumerate(PARAM_NAMES):
        delta_rel = 0.10
        eps = theta_true[d] * delta_rel
        # +ε
        theta_plus = theta_true.copy(); theta_plus[d] += eps
        a_plus = forward_multistep(solver, rest_state, theta_plus,
                                     knife_traj, knife_radius, device)
        a_plus_flat = np.concatenate([a for a in a_plus])
        err_plus = float(np.linalg.norm(a_plus_flat - a_truth_flat) / len(a_truth_flat) ** 0.5)
        # -ε
        theta_minus = theta_true.copy(); theta_minus[d] -= eps
        a_minus = forward_multistep(solver, rest_state, theta_minus,
                                      knife_traj, knife_radius, device)
        a_minus_flat = np.concatenate([a for a in a_minus])
        err_minus = float(np.linalg.norm(a_minus_flat - a_truth_flat) / len(a_truth_flat) ** 0.5)
        avg = 0.5 * (err_plus + err_minus)
        sens[name] = avg
        # Normalize per unit-relative perturbation (divide by 0.10)
        sens_per_unit = avg / delta_rel
        print(f"  {name:>10s}  {err_plus:14.6e}  {err_minus:14.6e}  "
              f"{avg:14.6e}  {sens_per_unit:12.4e}")

    # Compute stiffness identifiability vs others
    stiff_sens = sens["stiffness"]
    cohesive_avg = (sens["α_min"] + sens["sharpness"] + sens["τ"]) / 3
    ratio = stiff_sens / max(cohesive_avg, 1e-20)
    print(f"\n  stiffness sensitivity / mean(cohesive sensitivity) = {ratio:.3e}")
    if ratio > 0.1:
        print(f"  ✓ STEP 1 PASS — stiffness signal comparable to cohesive params")
        step1_pass = True
    elif ratio > 0.01:
        print(f"  ⚠ STEP 1 WEAK — stiffness signal 10-100× weaker; recovery will be slow")
        step1_pass = True
    else:
        print(f"  ✗ STEP 1 FAIL — stiffness signal < 1% of cohesive → essentially invisible to L_α")
        print(f"     RECOVERY VIA POSITION-ONLY LOSS NOT POSSIBLE.")
        print(f"     Real phantom needs force / time-resolved data.")
        step1_pass = False

    # ============ STEP 2: Landscape sweep ============
    print(f"\n{'='*72}")
    print(f"STEP 2 — Stiffness landscape: L_α(stiff | α,sharp,τ = truth)")
    print(f"{'='*72}")
    stiff_sweep = np.linspace(0.10, 1.50, 21)  # 21 points covers truth 0.85
    print(f"  Sweeping stiffness ∈ [{stiff_sweep[0]:.2f}, {stiff_sweep[-1]:.2f}] × {len(stiff_sweep)} points")
    print()
    print(f"  {'stiff':>8s}  {'L_α':>14s}  {'cavity':>8s}  {'note':<20s}")
    L_landscape = []
    cavities = []
    for s in stiff_sweep:
        theta_s = theta_true.copy(); theta_s[3] = s
        a_s = forward_multistep(solver, rest_state, theta_s,
                                  knife_traj, knife_radius, device)
        a_s_flat = np.concatenate([a for a in a_s])
        L = float(np.sum((a_s_flat - a_truth_flat) ** 2) / len(a_truth_flat))
        cavity = float((a_s[-1] < 0.5).mean())
        L_landscape.append(L)
        cavities.append(cavity)
        is_truth = "← TRUTH" if abs(s - 0.85) < 0.04 else ""
        print(f"  {s:8.3f}  {L:14.6e}  {cavity:8.4f}  {is_truth}")

    L_landscape = np.array(L_landscape)
    cavities = np.array(cavities)
    L_truth_idx = int(np.argmin(np.abs(stiff_sweep - 0.85)))
    L_at_truth = L_landscape[L_truth_idx]
    L_min_idx = int(np.argmin(L_landscape))
    L_min = L_landscape[L_min_idx]
    stiff_at_min = stiff_sweep[L_min_idx]

    print(f"\n  L_α minimum at stiff={stiff_at_min:.3f}  (truth=0.85), L_min={L_min:.4e}")
    print(f"  L_α at truth={L_at_truth:.4e}")

    # Unimodality check: count local minima (strict)
    n_minima = 0
    for i in range(1, len(L_landscape) - 1):
        if L_landscape[i] < L_landscape[i-1] and L_landscape[i] < L_landscape[i+1]:
            n_minima += 1
    print(f"  number of local minima in L_α(stiff): {n_minima}")
    print(f"  L_α range: [{L_landscape.min():.4e}, {L_landscape.max():.4e}]")
    print(f"  L_α dynamic range (max/min): {L_landscape.max() / max(L_landscape.min(), 1e-30):.1e}")

    step2_pass = (n_minima == 1) and (abs(stiff_at_min - 0.85) < 0.15)
    if step2_pass:
        print(f"  ✓ STEP 2 PASS — unimodal landscape, minimum near truth")
    elif n_minima == 1:
        print(f"  ⚠ STEP 2 PARTIAL — unimodal but minimum drifted from truth (likely L_α not sufficient)")
    else:
        print(f"  ✗ STEP 2 FAIL — {n_minima} local minima → gradient descent will get trapped")

    # Plot landscape
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    ax0 = axes[0]
    ax0.semilogy(stiff_sweep, L_landscape, "o-", color="C0")
    ax0.axvline(0.85, color="red", ls="--", lw=1.5, alpha=0.7, label="truth=0.85")
    ax0.axvline(stiff_at_min, color="green", ls=":", lw=1.5, alpha=0.7, label=f"min={stiff_at_min:.3f}")
    ax0.set_ylabel("L_α(stiff) [log]")
    ax0.set_title("Step 2: Stiffness landscape (α,sharp,τ at truth)")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(stiff_sweep, cavities, "s-", color="C2")
    ax1.axvline(0.85, color="red", ls="--", lw=1.5, alpha=0.7, label="truth=0.85")
    ax1.set_xlabel("stiffness")
    ax1.set_ylabel("cavity frac (final frame)")
    ax1.set_title("Cavity vs stiffness — shows how much α actually changes")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = REPO_ROOT / "tools" / "day3b_stiffness_landscape.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n  plot saved: {out_path}")

    # ============ Verdict ============
    print(f"\n{'='*72}")
    print(f"AUDIT VERDICT")
    print(f"{'='*72}")
    print(f"  Step 1 (sensitivity): {'PASS' if step1_pass else 'FAIL'}")
    print(f"  Step 2 (landscape):   {'PASS' if step2_pass else 'PARTIAL/FAIL'}")
    print()
    if step1_pass and step2_pass:
        print(f"  ✓ stiffness IS identifiable from cube+knife L_α.")
        print(f"    → proceed to multi-point FD parity (Step 3) and 4D sys-ID")
    elif step1_pass and not step2_pass:
        print(f"  ⚠ stiffness has signal but landscape is degenerate.")
        print(f"    → debug landscape before sys-ID; may need TPE warm-start")
    elif not step1_pass:
        print(f"  ✗ stiffness NOT identifiable from this setup + L_α.")
        print(f"    → architecture decision: either")
        print(f"       (a) change loss (add L_pos or L_force), or")
        print(f"       (b) accept stiffness from Stage 2.1 indenter pre-calibration")


if __name__ == "__main__":
    main()
