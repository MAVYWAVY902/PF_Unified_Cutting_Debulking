"""Step-by-step loss landscape diagnostic.

For each candidate loss formulation, probe along stiff_in axis and look for
minimum location. If a formulation gives clean minimum at truth, we've found
the right loss.

Loss candidates:
   A. L_direct_pos = ||sim_pos - obs_pos||²
      (Simplest possible. No inner GD. Direct sim-obs distance.)

   B. L_direct_combined = L_direct_pos + ||sim_alpha - obs_alpha||²
      (Adds alpha matching.)

   C. L_gap_30iter  (current: gd_inner_loop_stable with n_iter=30, lr_pos=0.5)

   D. L_gap_100iter (same but more iterations to test convergence)

   E. L_gap_closed_form (closed_form_inner — analytical exact correction)
      Should be equivalent to L_direct at convergence per Liang formula.

Mode C inject is INCLUDED in each measurement (same as calibration uses).

Probe stiff_in ∈ {0.3, 0.4, ..., 1.5}.
"""

import sys
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
from tools.day8_stable_inner_loop import gd_inner_loop_stable, closed_form_inner  # noqa: E402

from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff,
)
from tools.pointnetpp.multi_mesh_test import (  # noqa: E402
    MESH_CONFIGS, make_traj_from_spec,
)


def run_forward_T_frames(stiff_pp_np, tau_pp_np, ctx, mode_c_inject=True):
    """Run T frames forward, optionally with Mode C inject between frames.

    Returns list of (sim_pos, sim_alpha, sim_dv) per frame.
    """
    device = ctx["device"]
    theta_torch = torch.tensor(ctx["theta_truth"], dtype=torch.float32, device=device, requires_grad=False)
    stiff_torch = torch.tensor(stiff_pp_np, dtype=torch.float32, device=device, requires_grad=False)
    tau_torch = torch.tensor(tau_pp_np, dtype=torch.float32, device=device, requires_grad=False)
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
    sim_frames = []
    for t in range(len(knife_traj)):
        # Forward sim — no_grad fine here since we don't need to backprop through it
        with torch.no_grad():
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state, knife_traj[t], knife_radius,
                stiff_torch, tau_torch,
            )
        sim_frames.append((pos_t.cpu().numpy().copy(),
                              alpha_t.cpu().numpy().copy(),
                              dv_t.cpu().numpy().copy()))
        # Mode C inject — inner GD needs autograd (NOT in no_grad context)
        if mode_c_inject:
            tau_inner_val = float(tau_pp_np.mean())
            # Detach pos_t and dv_t (no need for grad through sim), but allow inner GD to track delta
            pos_t_d = pos_t.detach()
            dv_t_d = dv_t.detach()
            delta_x, delta_dv, _ = gd_inner_loop_stable(
                pos_t_d, dv_t_d, obs_pos_t[t], obs_alpha_t[t],
                theta_torch, edges_a_t, edges_b_t, edges_rest_t,
                n_iter=30, lr_pos=0.5, lr_dv_factor=1.0,
                lambda_cav=1.0, lambda_E=0.0,
                s_inner=float(theta_torch[1].item()),
                tau_inner=tau_inner_val,
            )
            state = solver_save_state(solver)
            dx_np = delta_x.detach().cpu().numpy()
            ddv_np = delta_dv.detach().cpu().numpy()
            state["pos"] = state["pos"] + dx_np
            state["particle_damage"] = np.maximum(state["particle_damage"] + ddv_np, 0.0)
            new_dv = state["particle_damage"]
            cur_s = float(theta_torch[1].item())
            state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(
                cur_s * (new_dv - tau_pp_np), -50, 50)))
        else:
            state = solver_save_state(solver)
    return sim_frames


def loss_direct_pos(sim_frames, obs_pos_list):
    """L = mean over t of ||sim_pos[t] - obs_pos[t]||²"""
    losses = []
    for t, (sp, _, _) in enumerate(sim_frames):
        losses.append(float(((sp - obs_pos_list[t]) ** 2).sum()))
    return float(np.mean(losses))


def loss_direct_combined(sim_frames, obs_pos_list, obs_alpha_list):
    """L = mean over t of (||sim_pos - obs_pos||² + ||sim_alpha - obs_alpha||²)"""
    losses = []
    for t, (sp, sa, _) in enumerate(sim_frames):
        L = float(((sp - obs_pos_list[t]) ** 2).sum()) + float(((sa - obs_alpha_list[t]) ** 2).sum())
        losses.append(L)
    return float(np.mean(losses))


def loss_inner_gd(sim_frames, ctx, n_iter=30, tau_pp_np=None):
    """L_gap = mean over t of ||delta_x_GD||² + ||delta_dv_GD||²"""
    device = ctx["device"]
    obs_pos_t = ctx["obs_pos_A_t"]
    obs_alpha_t = ctx["obs_alpha_A_t"]
    edges_a_t = ctx["edges_a_t"]
    edges_b_t = ctx["edges_b_t"]
    edges_rest_t = ctx["edges_rest_t"]
    theta_torch = torch.tensor(ctx["theta_truth"], dtype=torch.float32, device=device, requires_grad=False)
    tau_inner_val = float(tau_pp_np.mean())
    losses = []
    for t, (sp, sa, sdv) in enumerate(sim_frames):
        sp_t = torch.tensor(sp, dtype=torch.float32, device=device, requires_grad=False)
        sdv_t = torch.tensor(sdv, dtype=torch.float32, device=device, requires_grad=False)
        delta_x, delta_dv, info = gd_inner_loop_stable(
            sp_t, sdv_t, obs_pos_t[t], obs_alpha_t[t],
            theta_torch, edges_a_t, edges_b_t, edges_rest_t,
            n_iter=n_iter, lr_pos=0.5, lr_dv_factor=1.0,
            lambda_cav=1.0, lambda_E=0.0,
            s_inner=float(theta_torch[1].item()),
            tau_inner=tau_inner_val,
        )
        L_gap = float((delta_x ** 2).sum() + (delta_dv ** 2).sum())
        losses.append(L_gap)
    return float(np.mean(losses))


def loss_closed_form(sim_frames, ctx, tau_pp_np=None):
    """L_gap_cf = mean over t of ||delta_x_cf||² + ||delta_dv_cf||²
    using closed-form inner solver (exact analytical correction)."""
    device = ctx["device"]
    obs_pos_t = ctx["obs_pos_A_t"]
    obs_alpha_t = ctx["obs_alpha_A_t"]
    theta_torch = torch.tensor(ctx["theta_truth"], dtype=torch.float32, device=device, requires_grad=False)
    losses = []
    for t, (sp, sa, sdv) in enumerate(sim_frames):
        sp_t = torch.tensor(sp, dtype=torch.float32, device=device, requires_grad=False)
        sdv_t = torch.tensor(sdv, dtype=torch.float32, device=device, requires_grad=False)
        delta_x, delta_dv = closed_form_inner(sp_t, sdv_t, obs_pos_t[t], obs_alpha_t[t], theta_torch)
        L_gap = float((delta_x ** 2).sum() + (delta_dv ** 2).sum())
        losses.append(L_gap)
    return float(np.mean(losses))


def main():
    cfg = MESH_CONFIGS[0]
    device = "cuda:0"
    T = 10
    print(f"\n=== Loss landscape diagnostic on {cfg['name']} ===")
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
    n_incl = int(inclusion_mask.sum())
    print(f"  N={N}  incl={n_incl} ({n_incl/N*100:.2f}%)")

    theta_truth = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    traj_A = make_traj_from_spec(cfg["traj_specs"]["A"], bbox_min, bbox_max, T)
    solver.reset_to_rest(pos_rest)
    obs_pos, obs_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth, stiff_truth, traj_A, knife_radius, device,
        tau_truth_np=tau_truth)
    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64), dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64), dtype=torch.long, device=device)
    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)

    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=traj_A, knife_radius=knife_radius,
        pos_rest=pos_rest, theta_truth=theta_truth,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
    )

    stiff_in_grid = np.linspace(0.30, 1.50, 13)

    # Compute 5 losses for each stiff_in value
    print(f"\n  Probing 5 loss formulations across {len(stiff_in_grid)} stiff_in values...")
    print(f"  (with Mode C inject — matches calibration setup)\n")

    results = dict(L_direct_pos=[], L_direct_combined=[],
                     L_inner_30=[], L_inner_100=[], L_closed_form=[])

    print(f"  {'stiff_in':>8}  {'L_direct_pos':>14}  {'L_dir_combined':>14}  "
          f"{'L_inner_30':>12}  {'L_inner_100':>12}  {'L_closed_form':>14}")

    for s_in in stiff_in_grid:
        stiff_probe = stiff_truth.copy()
        stiff_probe[inclusion_mask] = s_in
        # Forward T frames (single run, reused for all losses)
        sim_frames = run_forward_T_frames(stiff_probe, tau_truth, ctx, mode_c_inject=True)
        L_a = loss_direct_pos(sim_frames, obs_pos)
        L_b = loss_direct_combined(sim_frames, obs_pos, obs_alpha)
        # Re-run forward without mode_c inject for inner GD (to keep sim_frames consistent)
        L_c = loss_inner_gd(sim_frames, ctx, n_iter=30, tau_pp_np=tau_truth)
        L_d = loss_inner_gd(sim_frames, ctx, n_iter=100, tau_pp_np=tau_truth)
        L_e = loss_closed_form(sim_frames, ctx, tau_pp_np=tau_truth)
        results["L_direct_pos"].append(L_a)
        results["L_direct_combined"].append(L_b)
        results["L_inner_30"].append(L_c)
        results["L_inner_100"].append(L_d)
        results["L_closed_form"].append(L_e)
        print(f"  {s_in:>8.3f}  {L_a:>14.4e}  {L_b:>14.4e}  "
              f"{L_c:>12.4e}  {L_d:>12.4e}  {L_e:>14.4e}", flush=True)

    # Summary per loss
    print(f"\n{'='*72}")
    print(f"SUMMARY — minimum location per loss formulation (truth = 0.80)")
    print(f"{'='*72}")
    for name in results:
        vals = np.array(results[name])
        argmin = stiff_in_grid[int(np.argmin(vals))]
        rel_truth = (vals[int(np.argmin(np.abs(stiff_in_grid - 0.80)))] - vals.min()) / max(vals.min(), 1e-12) * 100
        spread = (vals.max() - vals.min()) / max(vals.min(), 1e-12) * 100
        verdict = "✓ at truth" if abs(argmin - 0.80) < 0.1 else f"✗ at {argmin:.2f} (truth+{argmin-0.80:+.2f})"
        print(f"  {name:18s}  argmin = {argmin:.2f}  {verdict}  "
              f"(L at truth is {rel_truth:.1f}% above min, spread {spread:.1f}%)")

    # Plot 5 landscapes
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for idx, (name, label) in enumerate([
        ("L_direct_pos", "L_direct_pos = ||sim_pos - obs_pos||²"),
        ("L_direct_combined", "L_dir + ||sim_α - obs_α||²"),
        ("L_inner_30", "L_gap inner GD (30 iter)"),
        ("L_inner_100", "L_gap inner GD (100 iter)"),
        ("L_closed_form", "L_gap closed-form (analytical)"),
    ]):
        ax = axes.flatten()[idx]
        vals = results[name]
        ax.plot(stiff_in_grid, vals, "o-", lw=2)
        ax.axvline(0.80, color="red", linestyle="--", label="truth")
        ax.axvline(0.50, color="gray", linestyle=":", label="init")
        argmin = stiff_in_grid[int(np.argmin(vals))]
        ax.axvline(argmin, color="green", linestyle="-.", alpha=0.7, label=f"argmin={argmin:.2f}")
        ax.set_xlabel("stiff_in (others at truth)")
        ax.set_ylabel(name)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes.flatten()[5].axis("off")
    fig.suptitle("Loss landscape probe — which loss has minimum at truth stiff_in=0.80?",
                  fontsize=14)
    fig.tight_layout()
    out_path = REPO / "tools" / "pointnetpp" / "diag_loss_landscape.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
