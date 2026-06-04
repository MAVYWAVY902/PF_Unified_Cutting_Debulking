"""Online per-frame calibration following Liang 2023 Algorithm 1.

KEY DIFFERENCES from `diag_partial_obs.calibrate`:
   - Per-frame Adam step (NOT per-epoch)
   - state passed as numpy dict between frames -> auto-detach (no chained BPTT)
   - Memory O(N) constant, NOT O(T_calib * N)
   - L_hist (4 history equilibrium check) — Phase 2 (lambda_hist=0 in Phase 1)
   - Multi-traj: serial concat (run traj_A end-to-end, reset state, run traj_B)

Backward per frame goes through ONE PBD step only — exactly Liang's design.
"""
import math
import time
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch

from tools.day2_solver_wrapper import solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.day8_stable_inner_loop import gd_inner_loop_stable
from tools.pointnetpp.hierarchy import reconstruct as hier_reconstruct
from tools.pointnetpp.diag_partial_obs import (
    inv_sigmoid_to_z, STIFF_LOW, STIFF_HIGH,
)


def calibrate_online(
    condition: str,                          # "L_pos_partial" or "Hier+L_pos_partial"
    ctx: dict,
    multi_traj_obs: List[Tuple],             # [(knife_traj, obs_pos_t, obs_alpha_t), ...]
    N_passes: int = 1,                       # passes through the concatenated stream
    lr: float = 0.01,
    lr_tau: float = 0.15,
    lr_decay_per_pass: float = 1.0,          # multiply all lr by this each pass (e.g. 0.5)
    lambda_smooth: float = 0.0,
    lambda_hist: float = 0.0,                # Replay L_hist weight (depth=1 truncated BPTT)
    lambda_outer_cav: float = 0.0,           # surface alpha (cavity) mismatch — tau signal
    use_inner_gd: bool = False,              # ★ Full Liang: 30-step inner GD + Mode C inject
    inner_lambda_E: float = 0.1,             # elastic energy weight in inner GD
    inner_n_iter: int = 30,
    window_size: int = 20,
    n_hist_samples: int = 4,
    teacher_force: bool = True,              # overwrite sim pos with obs (auto-OFF when use_inner_gd)
    teacher_force_every: int = 1,            # ★ reset pos to obs every W frames (1=every frame).
                                             #   W>1 lets cavity free-evolve within window so the
                                             #   long-horizon signal accumulates (recovers landscape
                                             #   gradient that per-frame teacher forcing kills).
    log_every: int = 30,
    verbose: bool = True,
):
    """Online per-frame calibration. Returns (stiff_pp_np, tau_pp_np, history_log).

    Arguments mirror diag_partial_obs.calibrate where overlapping, but:
       - N_epochs replaced by N_passes (single-pass through stream by default)
       - lambda_hist activates L_hist re-equilibrium check
       - window_size / n_hist_samples control history buffer
    """
    solver = ctx["solver"]
    rest_state = ctx["rest_state"]
    pos_rest = ctx["pos_rest"]
    N = ctx["N"]
    device = ctx["device"]
    knife_radius = ctx["knife_radius"]
    theta_3d_np = ctx["theta_3d_np"]
    stiff_init_np = ctx["stiff_init_np"]
    tau_init_np = ctx["tau_init_np"]
    tau_bound = ctx["tau_bound"]
    surface_idx = ctx["surface_idx"]
    edges_a_t = ctx["edges_a_t"]
    edges_b_t = ctx["edges_b_t"]
    edges_rest_t = ctx.get("edges_rest_t")

    theta_torch = torch.tensor(theta_3d_np, dtype=torch.float32, device=device,
                                 requires_grad=False)
    far_knife = (pos_rest.max(0) + 10.0).astype(np.float32)

    # Parameterization
    use_hier = "Hier" in condition
    z_L0_stiff = torch.tensor(inv_sigmoid_to_z(stiff_init_np),
                                dtype=torch.float32, device=device, requires_grad=True)
    z_L0_tau = torch.tensor(inv_sigmoid_to_z(tau_init_np, lo=tau_bound[0], hi=tau_bound[1]),
                              dtype=torch.float32, device=device, requires_grad=True)
    if use_hier:
        H = ctx["H"]
        M1, M2 = H.M1, H.M2
        sL1 = 1.0 / math.sqrt(N / M1)
        sL2 = 1.0 / math.sqrt(N / M2)
        z_L1_stiff = torch.zeros(M1, dtype=torch.float32, device=device, requires_grad=True)
        z_L2_stiff = torch.zeros(M2, dtype=torch.float32, device=device, requires_grad=True)
        z_L1_tau = torch.zeros(M1, dtype=torch.float32, device=device, requires_grad=True)
        z_L2_tau = torch.zeros(M2, dtype=torch.float32, device=device, requires_grad=True)
        params = [
            {"params": [z_L0_stiff], "lr": lr},
            {"params": [z_L1_stiff], "lr": lr * sL1},
            {"params": [z_L2_stiff], "lr": lr * sL2},
            {"params": [z_L0_tau], "lr": lr_tau},
            {"params": [z_L1_tau], "lr": lr_tau * sL1},
            {"params": [z_L2_tau], "lr": lr_tau * sL2},
        ]
    else:
        z_L1_stiff = z_L2_stiff = z_L1_tau = z_L2_tau = None
        params = [
            {"params": [z_L0_stiff], "lr": lr},
            {"params": [z_L0_tau], "lr": lr_tau},
        ]
    opt = torch.optim.Adam(params)

    def reconstruct_fields():
        """Reconstruct per-particle stiff/tau from z (Hier or direct)."""
        if use_hier:
            z_flat_stiff = hier_reconstruct(z_L0_stiff, z_L1_stiff, z_L2_stiff, ctx["H"])
            z_flat_tau = hier_reconstruct(z_L0_tau, z_L1_tau, z_L2_tau, ctx["H"])
        else:
            z_flat_stiff = z_L0_stiff
            z_flat_tau = z_L0_tau
        stiff_pp = STIFF_LOW + (STIFF_HIGH - STIFF_LOW) * torch.sigmoid(z_flat_stiff)
        tau_pp = tau_bound[0] + (tau_bound[1] - tau_bound[0]) * torch.sigmoid(z_flat_tau)
        return stiff_pp, tau_pp

    # History buffer for REPLAY L_hist:
    #   deque of (state_BEFORE_step_h, knife_at_h, obs_AFTER_step_h_np)
    # Semantics: PBD(state_h_in, knife_h, current_k) should reproduce obs_h_out.
    # This is depth=1 truncated BPTT applied at random historical (IC, BC, target)
    # tuples — adapts Liang L_hist for dynamic non-equilibrium cutting.
    H_buffer = deque(maxlen=window_size)
    rng = np.random.RandomState(42)

    history_log = []
    total_steps = 0
    t0 = time.time()
    initial_lrs = [pg["lr"] for pg in opt.param_groups]

    for pass_idx in range(N_passes):
        # apply lr decay
        decay = lr_decay_per_pass ** pass_idx
        for kk, pg in enumerate(opt.param_groups):
            pg["lr"] = initial_lrs[kk] * decay
        if verbose and pass_idx > 0:
            print(f"  [pass {pass_idx}] lr decay × {decay:.3f}", flush=True)
        for traj_idx, (cur_traj, cur_obs_pos_t, cur_obs_alpha_t) in enumerate(multi_traj_obs):
            # Reset solver to rest at start of each traj
            solver.reset_to_rest(pos_rest)
            state = rest_state  # numpy dict from setup
            T = len(cur_traj)

            for t in range(T):
                # Snapshot the IC for this frame BEFORE the forward call.
                # We need this to push (state_in, knife, obs_out) into H_buffer.
                state_in_for_hist = {kk: (vv.copy() if hasattr(vv, "copy") else vv)
                                       for kk, vv in state.items()}

                # ===== current-frame forward =====
                stiff_pp_torch, tau_pp_torch = reconstruct_fields()
                alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                    theta_torch, solver, state, cur_traj[t], knife_radius,
                    stiff_pp_torch, tau_pp_torch,
                )
                obs_pos_now = cur_obs_pos_t[t]

                # ===== L_gap =====
                # Two modes:
                #   A. use_inner_gd=False — direct surface L_pos (simplified, our previous online)
                #   B. use_inner_gd=True  — Liang's g layer: 30-step inner GD finds (Δ_x, Δ_dv)
                #                            then L_gap = ||Δ_x||² + ||Δ_dv||²
                #                            This isolates geometric mismatch from physical params.
                obs_alpha_now = cur_obs_alpha_t[t]
                if use_inner_gd:
                    delta_x, delta_dv, _ = gd_inner_loop_stable(
                        pos_t, dv_t, obs_pos_now, obs_alpha_now,
                        None, edges_a_t, edges_b_t, edges_rest_t,
                        n_iter=inner_n_iter, lr_pos=0.5, lr_dv_factor=1.0,
                        lambda_cav=1.0, lambda_E=inner_lambda_E,
                        surface_idx=surface_idx,
                    )
                    L_gap = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
                else:
                    delta_x = None
                    delta_dv = None
                    L_gap = ((pos_t[surface_idx] - obs_pos_now[surface_idx]) ** 2).sum()

                # ===== L_cav (only used in simplified mode; inner GD includes cav already) =====
                L_cav = torch.tensor(0.0, device=device)
                if lambda_outer_cav > 0 and not use_inner_gd:
                    L_cav = ((alpha_t[surface_idx] - obs_alpha_now[surface_idx]) ** 2).sum()

                # ===== Replay L_hist (pos + alpha jointly) =====
                # PBD(state_in_h, knife_h, current_k) ?= obs_out_h
                # Treats history as additional (IC, BC, target) supervised samples.
                L_hist = torch.tensor(0.0, device=device)
                if lambda_hist > 0 and len(H_buffer) >= n_hist_samples:
                    sample_indices = rng.choice(len(H_buffer), n_hist_samples, replace=False)
                    for s_idx in sample_indices:
                        h_state_in, h_knife, h_obs_pos_np, h_obs_alpha_np = H_buffer[s_idx]
                        s_pp_h, t_pp_h = reconstruct_fields()
                        alpha_h_replay, pos_h_replay, _ = WarpFullForwardOp.apply(
                            theta_torch, solver, h_state_in, h_knife, knife_radius,
                            s_pp_h, t_pp_h,
                        )
                        h_pos_t = torch.tensor(h_obs_pos_np, dtype=torch.float32,
                                                 device=device)
                        L_hist = L_hist + ((pos_h_replay[surface_idx]
                                              - h_pos_t[surface_idx]) ** 2).sum()
                        if lambda_outer_cav > 0:
                            h_alpha_t = torch.tensor(h_obs_alpha_np, dtype=torch.float32,
                                                       device=device)
                            L_hist = L_hist + lambda_outer_cav * ((alpha_h_replay[surface_idx]
                                                                      - h_alpha_t[surface_idx]) ** 2).sum()

                # ===== L_smooth (per-particle stiff smoothness over edges) =====
                L_smooth = torch.tensor(0.0, device=device)
                if lambda_smooth > 0:
                    diff = stiff_pp_torch[edges_a_t] - stiff_pp_torch[edges_b_t]
                    L_smooth = (diff ** 2).sum()

                # ===== total + Adam step =====
                L = L_gap + lambda_outer_cav * L_cav + lambda_hist * L_hist \
                    + lambda_smooth * L_smooth
                opt.zero_grad()
                L.backward()
                opt.step()

                # ===== push (state_in, knife, obs_pos, obs_alpha) to history =====
                H_buffer.append((
                    state_in_for_hist,
                    cur_traj[t],
                    obs_pos_now.detach().cpu().numpy(),
                    cur_obs_alpha_t[t].detach().cpu().numpy(),
                ))

                # ===== advance state =====
                # Mode C inject is mathematically equivalent to teacher forcing
                # under direct correspondence (Δ_t ≈ obs - sim, plus elastic regularization).
                # BUT Mode C with our PBD cluster-quat physics causes cluster_quat
                # to go stale → polar decomp NaN. So we decouple:
                #   - Inner GD computes Δ for L_gap (gradient signal preserved)
                #   - State update uses teacher force (stable, same effect)
                state_after = solver_save_state(solver)
                # Teacher-force position to obs only every `teacher_force_every` frames.
                # Within a window, sim free-evolves so cavity errors accumulate (strong signal);
                # at window boundary we reset to obs to prevent runaway divergence / NaN.
                do_tf = (use_inner_gd or teacher_force) and ((t + 1) % teacher_force_every == 0)
                if do_tf:
                    obs_np = obs_pos_now.detach().cpu().numpy().astype(np.float32)
                    state_after["pos"] = obs_np.copy()
                state = state_after

                # logging
                total_steps += 1
                if verbose and (total_steps % log_every == 0):
                    L_gap_v = float(L_gap.detach().item())
                    L_cav_v = float(L_cav.detach().item()) if lambda_outer_cav > 0 else 0.0
                    L_hist_v = float(L_hist.detach().item()) if lambda_hist > 0 else 0.0
                    elapsed = time.time() - t0
                    rate = total_steps / max(elapsed, 1e-6)
                    print(f"  [pass {pass_idx} traj {traj_idx} t={t:3d}] "
                            f"step {total_steps:4d}  L_gap={L_gap_v:.3e}  "
                            f"L_cav={L_cav_v:.3e}  L_hist={L_hist_v:.3e}  "
                            f"({rate:.1f} step/s)", flush=True)
                history_log.append(dict(
                    step=total_steps, pass_idx=pass_idx, traj_idx=traj_idx, t=t,
                    L_gap=float(L_gap.detach().item()),
                    L_cav=(float(L_cav.detach().item()) if lambda_outer_cav > 0 else 0.0),
                    L_hist=(float(L_hist.detach().item()) if lambda_hist > 0 else 0.0),
                ))

    # ===== final reconstruction =====
    with torch.no_grad():
        stiff_final, tau_final = reconstruct_fields()
    return (stiff_final.detach().cpu().numpy(),
              tau_final.detach().cpu().numpy(),
              history_log)
