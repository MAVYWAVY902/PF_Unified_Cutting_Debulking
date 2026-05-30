"""Diagnostic — is IoU 0.2 plateau caused by under-determined obs?

Three tests on M1_cube buried-inclusion setup (the working multi-mesh config):

   TEST 1: Loss landscape probe along stiff_in axis.
      Hold all OTHER particles at truth, vary inclusion-particle stiff
      from 0.3 to 1.5. Measure L_gap. Expect:
        - If clear minimum at truth 0.8 → loss correctly oriented
        - If flat → loss landscape is degenerate (under-determined)
        - If minimum elsewhere → forward sim or loss has bug

   TEST 2: Oracle init.
      Start Adam at truth stiff_pp / tau_pp. Run 30 epochs. Track:
        - Lgap (should stay near initial truth Lgap)
        - stiff_in mean (should stay near 0.8)
        - IoU (should stay near 1.0)
      Diagnoses:
        - Stays near truth → forward + loss + Adam are all correctly oriented
        - Drifts → loss flat / multiple minima / Adam too aggressive

   TEST 3: Cut-zone-only vs full-mesh obs.
      Compute L_gap two ways:
        - Standard (already does all N particles)
        - Cut-zone only (mask out particles that never moved across T frames)
      Compare gradient magnitudes. Confirms which obs are actually informative.
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
from tools.pointnetpp.multi_mesh_test import (  # noqa: E402
    MESH_CONFIGS, make_traj_from_spec,
)


def compute_L_gap(stiff_pp_np, tau_pp_np, ctx):
    """Forward sim + inner GD residual on T_A. Returns Lgap mean across frames."""
    device = ctx["device"]
    theta_torch = torch.tensor(ctx["theta_truth"], dtype=torch.float32,
                                 device=device, requires_grad=False)
    stiff_torch = torch.tensor(stiff_pp_np, dtype=torch.float32,
                                 device=device, requires_grad=False)
    tau_torch = torch.tensor(tau_pp_np, dtype=torch.float32,
                               device=device, requires_grad=False)
    solver = ctx["solver"]
    rest_state = ctx["rest_state"]
    obs_pos_t = ctx["obs_pos_A_t"]
    obs_alpha_t = ctx["obs_alpha_A_t"]
    knife_traj = ctx["knife_traj_A"]
    knife_radius = ctx["knife_radius"]
    edges_a_t = ctx["edges_a_t"]
    edges_b_t = ctx["edges_b_t"]
    edges_rest_t = ctx["edges_rest_t"]

    solver.reset_to_rest(ctx["pos_rest"])
    state = rest_state
    L_gaps = []
    with torch.no_grad():
        tau_inner_val = float(tau_pp_np.mean())
    for t in range(len(knife_traj)):
        alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
            theta_torch, solver, state, knife_traj[t], knife_radius,
            stiff_torch, tau_torch,
        )
        delta_x, delta_dv, _ = gd_inner_loop_stable(
            pos_t, dv_t, obs_pos_t[t], obs_alpha_t[t],
            theta_torch, edges_a_t, edges_b_t, edges_rest_t,
            n_iter=30, lr_pos=0.5, lr_dv_factor=1.0,
            lambda_cav=1.0, lambda_E=0.0,
            s_inner=float(theta_torch[1].item()),
            tau_inner=tau_inner_val,
        )
        L_gap = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
        L_gaps.append(float(L_gap.item()))
        # Mode C inject
        state = solver_save_state(solver)
        dx_np = delta_x.detach().cpu().numpy()
        ddv_np = delta_dv.detach().cpu().numpy()
        state["pos"] = state["pos"] + dx_np
        state["particle_damage"] = np.maximum(state["particle_damage"] + ddv_np, 0.0)
        new_dv = state["particle_damage"]
        cur_s = float(theta_torch[1].item())
        state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(
            cur_s * (new_dv - tau_pp_np), -50, 50)))
    return float(np.mean(L_gaps))


def calibrate_oracle_init(ctx, stiff_truth_np, tau_truth_np, N_epochs=30,
                           lr=0.03, lr_tau=0.01):
    """Adam from truth init — diagnostics. Returns trajectory of Lgap, IoU,
    stiff_in mean, stiff_out mean."""
    device = ctx["device"]
    N = ctx["N"]
    inclusion_mask = ctx["inclusion_mask"]
    tau_bound = (0.05, 0.95)

    # Oracle init: start at TRUTH (with tiny perturbation to keep Adam moving)
    rng = np.random.default_rng(0)
    stiff_oracle = stiff_truth_np + rng.normal(0, 0.001, size=N).astype(np.float32)
    tau_oracle = tau_truth_np + rng.normal(0, 0.001, size=N).astype(np.float32)
    stiff_oracle = np.clip(stiff_oracle, STIFF_LOW + 1e-4, STIFF_HIGH - 1e-4)
    tau_oracle = np.clip(tau_oracle, tau_bound[0] + 1e-4, tau_bound[1] - 1e-4)

    theta_torch = torch.tensor(ctx["theta_truth"], dtype=torch.float32,
                                 device=device, requires_grad=False)
    z_stiff = torch.tensor(inv_sigmoid_to_z(stiff_oracle),
                              dtype=torch.float32, device=device, requires_grad=True)
    z_tau = torch.tensor(inv_sigmoid_to_z(tau_oracle, lo=tau_bound[0], hi=tau_bound[1]),
                            dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([
        {"params": [z_stiff], "lr": lr},
        {"params": [z_tau], "lr": lr_tau},
    ])

    T = len(ctx["knife_traj_A"])
    history = []
    for epoch in range(N_epochs):
        solver = ctx["solver"]
        solver.reset_to_rest(ctx["pos_rest"])
        state = ctx["rest_state"]
        L_gaps = []
        for t in range(T):
            stiff_pp_torch = STIFF_LOW + (STIFF_HIGH - STIFF_LOW) * torch.sigmoid(z_stiff)
            tau_pp_torch = tau_bound[0] + (tau_bound[1] - tau_bound[0]) * torch.sigmoid(z_tau)
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state, ctx["knife_traj_A"][t], ctx["knife_radius"],
                stiff_pp_torch, tau_pp_torch,
            )
            tau_inner_val = float(tau_pp_torch.mean().item())
            delta_x, delta_dv, _ = gd_inner_loop_stable(
                pos_t, dv_t, ctx["obs_pos_A_t"][t], ctx["obs_alpha_A_t"][t],
                theta_torch, ctx["edges_a_t"], ctx["edges_b_t"], ctx["edges_rest_t"],
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
            stiff_now = (STIFF_LOW + (STIFF_HIGH - STIFF_LOW)
                            * torch.sigmoid(z_stiff)).cpu().numpy()
            tau_now = (tau_bound[0] + (tau_bound[1] - tau_bound[0])
                          * torch.sigmoid(z_tau)).cpu().numpy()
        m = compute_metrics(stiff_now, ctx["stiff_truth"], inclusion_mask)
        history.append(dict(
            epoch=epoch, mean_L_gap=mean_L, iou=m["iou"],
            stiff_in=float(stiff_now[inclusion_mask].mean()),
            stiff_out=float(stiff_now[~inclusion_mask].mean()),
            stiff_in_max_dev=float(np.abs(stiff_now[inclusion_mask] - stiff_truth_np[inclusion_mask]).max()),
        ))
    return history


def main():
    cfg = MESH_CONFIGS[0]  # M1_cube buried inclusion
    device = "cuda:0"
    T = 10
    print(f"\n=== Diagnostic on {cfg['name']} (buried inclusion at y={cfg['incl_center_rel'][1]}) ===")
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
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
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    n_incl = int(inclusion_mask.sum())
    print(f"  N={N}  incl={n_incl} ({n_incl/N*100:.2f}%)")

    theta_truth = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    traj_A = make_traj_from_spec(cfg["traj_specs"]["A"], bbox_min, bbox_max, T)
    solver.reset_to_rest(pos_rest)
    obs_pos, obs_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth, stiff_truth, traj_A, knife_radius, device,
        tau_truth_np=tau_truth)
    cav = float((obs_alpha[-1] < 0.5).mean())
    print(f"  Truth cavity: {cav*100:.2f}%")

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)

    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=traj_A, knife_radius=knife_radius,
        pos_rest=pos_rest,
        stiff_truth=stiff_truth, tau_truth=tau_truth,
        inclusion_mask=inclusion_mask, theta_truth=theta_truth,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
    )

    # ===================================================================
    # TEST 1: Loss landscape probe along stiff_in axis
    # ===================================================================
    print("\n" + "=" * 72)
    print("TEST 1: Loss landscape probe — vary stiff_in only, others at truth")
    print("=" * 72)
    stiff_in_grid = np.linspace(0.30, 1.50, 13)
    landscape_L = []
    for s_in in stiff_in_grid:
        stiff_probe = stiff_truth.copy()
        stiff_probe[inclusion_mask] = s_in
        L = compute_L_gap(stiff_probe, tau_truth, ctx)
        landscape_L.append(L)
        print(f"  stiff_in = {s_in:.3f}    L_gap = {L:.3e}")
    landscape_L = np.array(landscape_L)
    truth_idx = int(np.argmin(np.abs(stiff_in_grid - 0.80)))
    L_at_truth = landscape_L[truth_idx]
    L_min = landscape_L.min()
    arg_min = stiff_in_grid[np.argmin(landscape_L)]
    print(f"\n  Truth stiff_in = 0.80, L_gap at truth = {L_at_truth:.3e}")
    print(f"  Minimum L_gap = {L_min:.3e} at stiff_in = {arg_min:.3f}")
    print(f"  Range of L_gap across probe: [{landscape_L.min():.3e}, {landscape_L.max():.3e}]")
    print(f"  Relative spread: {(landscape_L.max()-landscape_L.min())/landscape_L.min()*100:.1f}%")
    if abs(arg_min - 0.80) < 0.1:
        print("  ✓ L_gap minimum near truth → loss IS oriented")
    else:
        print(f"  ⚠ L_gap minimum NOT at truth (at {arg_min:.3f}) → loss flat or biased")

    # Quick plot
    fig1, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stiff_in_grid, landscape_L, "o-", lw=2, markersize=8)
    ax.axvline(0.80, color="red", linestyle="--", label="truth stiff_in = 0.80")
    ax.axvline(0.50, color="gray", linestyle=":", label="init stiff_in = 0.50")
    ax.set_xlabel("stiff_in (others at truth)")
    ax.set_ylabel("L_gap (mean across T frames)")
    ax.set_title(f"TEST 1: Loss landscape along stiff_in axis\n"
                 f"L_min at stiff_in={arg_min:.3f}, L_at_truth={L_at_truth:.3e}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig1.tight_layout()
    p1 = REPO / "tools" / "pointnetpp" / "diag_test1_landscape.png"
    fig1.savefig(str(p1), dpi=120, bbox_inches="tight")
    print(f"  saved {p1}")

    # ===================================================================
    # TEST 2: Oracle init — does Adam stay at truth?
    # ===================================================================
    print("\n" + "=" * 72)
    print("TEST 2: Oracle init — Adam starts at truth, does it stay?")
    print("=" * 72)
    history = calibrate_oracle_init(ctx, stiff_truth, tau_truth, N_epochs=30, lr=0.03, lr_tau=0.01)

    epochs_ = [h["epoch"] for h in history]
    Lgap_ = [h["mean_L_gap"] for h in history]
    iou_ = [h["iou"] for h in history]
    stiff_in_ = [h["stiff_in"] for h in history]

    print(f"  Initial: Lgap={Lgap_[0]:.3e}  IoU={iou_[0]:.3f}  stiff_in_mean={stiff_in_[0]:.3f}")
    print(f"  Mid (ep 15): Lgap={Lgap_[15]:.3e}  IoU={iou_[15]:.3f}  stiff_in_mean={stiff_in_[15]:.3f}")
    print(f"  Final (ep 29): Lgap={Lgap_[-1]:.3e}  IoU={iou_[-1]:.3f}  stiff_in_mean={stiff_in_[-1]:.3f}")
    drift = stiff_in_[-1] - stiff_in_[0]
    print(f"  Drift of stiff_in_mean: {drift:+.3f}")
    if abs(drift) < 0.05 and iou_[-1] > 0.7:
        print("  ✓ Adam STAYS at truth → loss + sim + opt correctly oriented")
    elif iou_[-1] < 0.3:
        print("  ✗ Adam DRIFTS AWAY from truth → loss flat or multiple minima")
    else:
        print("  △ Partial drift — loss has shallow minimum around truth")

    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs_, Lgap_, "o-")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("L_gap")
    axes[0].set_title("L_gap from oracle init")
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs_, iou_, "o-")
    axes[1].axhline(1.0, color="red", linestyle="--", label="perfect")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("IoU")
    axes[1].set_title("IoU from oracle init")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].plot(epochs_, stiff_in_, "o-")
    axes[2].axhline(0.80, color="red", linestyle="--", label="truth")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("stiff_in mean")
    axes[2].set_title("stiff_in mean from oracle init")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig2.suptitle("TEST 2: Adam from oracle init (truth + 0.001 noise)", fontsize=12)
    fig2.tight_layout()
    p2 = REPO / "tools" / "pointnetpp" / "diag_test2_oracle.png"
    fig2.savefig(str(p2), dpi=120, bbox_inches="tight")
    print(f"  saved {p2}")

    # ===================================================================
    # TEST 3: How many particles actually moved during T_A?
    # ===================================================================
    print("\n" + "=" * 72)
    print("TEST 3: Effective informative obs — how many particles actually moved?")
    print("=" * 72)
    # Compare obs positions to rest positions per frame
    max_displ_per_particle = np.zeros(N, dtype=np.float32)
    for t in range(T):
        d = np.linalg.norm(obs_pos[t] - pos_rest, axis=1)
        max_displ_per_particle = np.maximum(max_displ_per_particle, d)

    thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    print(f"  Particle motion thresholds:")
    for thr in thresholds:
        moved = (max_displ_per_particle > thr).sum()
        print(f"    Moved > {thr:.0e} m: {moved}/{N} ({moved/N*100:.2f}%)")

    n_moved_1mm = int((max_displ_per_particle > 1e-3).sum())
    print(f"\n  Effective informative particles (moved >1mm): {n_moved_1mm}/{N} ({n_moved_1mm/N*100:.2f}%)")
    print(f"  Unknowns / informative obs ratio: {2 * N / max(n_moved_1mm, 1):.0f} : 1")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 72)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 72)
    print(f"  TEST 1 (landscape): Lmin at stiff_in={arg_min:.3f} (truth 0.80)")
    print(f"                      Relative spread {(landscape_L.max()-landscape_L.min())/landscape_L.min()*100:.1f}%")
    print(f"  TEST 2 (oracle):    final IoU {iou_[-1]:.3f}, stiff_in drift {drift:+.3f}")
    print(f"  TEST 3 (info):      {n_moved_1mm} informative particles for {2*N} unknowns "
          f"(ratio 1:{2*N/max(n_moved_1mm, 1):.0f})")

    print(f"\nInterpretation:")
    if abs(arg_min - 0.80) < 0.1 and iou_[-1] > 0.7:
        print("  → Loss IS oriented at truth; Adam stays near truth from oracle init.")
        print("  → IoU 0.2 plateau IS due to under-determined obs (info ceiling).")
        print("  → Fix: add more informative obs (multi-cut, force sensor, etc.)")
    elif iou_[-1] < 0.3:
        print("  → Adam drifts even from truth init → loss landscape pathological.")
        print("  → Possible bug in forward sim or Liang inner loop.")
    else:
        print("  → Partial drift — likely combination of weak signal + Adam aggressiveness.")
        print("  → Try lower lr or fewer epochs with oracle init.")


if __name__ == "__main__":
    main()
