"""Day 6f v1: T=3 frame multi-step with KEEP-STATE-IN-GRAPH (single big tape).

Contrast to v0:
  v0: T separate tapes; per-frame backward; state DETACHED between frames
      → gradient = ∂L_t/∂θ via THIS frame only, summed over t
  v1: ONE tape covering all T solver.step()s; backward once at end
      → gradient INCLUDES indirect path ∂L_t/∂state_{t-1}·∂state_{t-1}/∂θ
        (chained backward through ALL prior solver.step()s)

Compare:
  (1) Are v1 gradients meaningfully different from v0?
  (2) Does v1 cause numerical instability (NaN, large grad spikes)?
  (3) Does v1 improve θ recovery vs v0?
  (4) Memory & timing penalty of single-tape T forwards?

Implementation challenge: solver.p["pos"] is mutated in place across solver.step()
calls. To capture frame-t snapshot inside ONE tape, use wp.clone(pos) which
allocates new tape-tracked array. Alpha/dv use SSA buffer slots (different slot
per step under capacity budget) — already snapshotted by reference.
"""
import sys
from pathlib import Path
import numpy as np
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pf_shape_matching_warp"))

import torch
import warp as wp
wp.init()
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

from tools.day2_solver_wrapper import (  # noqa: E402
    setup_solver_with_rest, solver_load_state, solver_save_state,
)
from tools.day6b_full_forward_torch_op import (  # noqa: E402
    make_theta_wp_arrays, WarpFullForwardOp,
)


THETA_BOUNDS = {
    "alpha_min":  (0.1,  50.0),
    "sharpness":  (1.0,  50.0),
    "threshold":  (0.05, 2.0),
}


def clamp_theta_inplace(theta_tensor):
    with torch.no_grad():
        theta_tensor.data[0].clamp_(*THETA_BOUNDS["alpha_min"])
        theta_tensor.data[1].clamp_(*THETA_BOUNDS["sharpness"])
        theta_tensor.data[2].clamp_(*THETA_BOUNDS["threshold"])


def make_knife_trajectory(T):
    x_positions = np.linspace(0.12, 0.18, T)
    return [np.array([x, 0.30, 0.15], dtype=np.float32) for x in x_positions]


# ---------------------------------------------------------------------------
# Multi-frame torch.autograd.Function — single tape, T solver.step()s inside
# ---------------------------------------------------------------------------

class MultiFrameSolverOp(torch.autograd.Function):
    """Run T solver.step() calls inside ONE wp.Tape, keeping state chain alive.

    Forward returns (alpha[T,N], pos[T,N,3], dv[T,N]) — per-frame snapshots.
    Backward propagates gradient through ALL T steps via single tape.backward.
    """

    @staticmethod
    def forward(ctx, theta_torch, solver, rest_state, knife_traj, knife_radius, T):
        theta_wp_dict = make_theta_wp_arrays(theta_torch, knife_radius)
        solver.set_theta_wp(theta_wp_dict)
        solver_load_state(solver, rest_state)
        solver.knife_active = 1

        alpha_wp_list = []
        pos_wp_list = []
        dv_wp_list = []

        tape = wp.Tape()
        with tape:
            for t in range(T):
                # Set knife pos for this frame
                solver.knife_pos = wp.array(
                    np.array([knife_traj[t]], dtype=np.float32),
                    dtype=wp.vec3, device=solver.device, requires_grad=False
                )
                solver.step()

                # Snapshot — pos shared across steps so MUST wp.clone (tape-tracked)
                pos_t = wp.clone(solver.p["pos"])
                if solver.use_ssa_soft_removal:
                    ci = solver._ssa_cycle_index
                    # SSA slot ref — different slot per step under capacity budget
                    alpha_t = solver._ssa_alpha_buffers[ci]
                    dv_t = solver._ssa_particle_damage_buffers[ci]
                else:
                    alpha_t = wp.clone(solver.p["alpha"])
                    dv_t = wp.clone(solver.p["damage"])

                alpha_wp_list.append(alpha_t)
                pos_wp_list.append(pos_t)
                dv_wp_list.append(dv_t)

        ctx.tape = tape
        ctx.solver = solver
        ctx.theta_wp_dict = theta_wp_dict
        ctx.alpha_wp_list = alpha_wp_list
        ctx.pos_wp_list = pos_wp_list
        ctx.dv_wp_list = dv_wp_list
        ctx.T = T

        alpha_torch = torch.stack([wp.to_torch(a).clone() for a in alpha_wp_list])
        pos_torch = torch.stack([wp.to_torch(p).clone() for p in pos_wp_list])
        dv_torch = torch.stack([wp.to_torch(d).clone() for d in dv_wp_list])
        return alpha_torch, pos_torch, dv_torch

    @staticmethod
    def backward(ctx, grad_alpha_torch, grad_pos_torch, grad_dv_torch):
        grads_dict = {}
        for t in range(ctx.T):
            if grad_alpha_torch is not None:
                grads_dict[ctx.alpha_wp_list[t]] = wp.from_torch(
                    grad_alpha_torch[t].contiguous()
                )
            if grad_pos_torch is not None:
                grads_dict[ctx.pos_wp_list[t]] = wp.from_torch(
                    grad_pos_torch[t].contiguous()
                )
            if grad_dv_torch is not None:
                grads_dict[ctx.dv_wp_list[t]] = wp.from_torch(
                    grad_dv_torch[t].contiguous()
                )

        ctx.tape.backward(grads=grads_dict)

        ga = float(ctx.theta_wp_dict["alpha_min"].grad.numpy()[0])
        gs = float(ctx.theta_wp_dict["sharpness"].grad.numpy()[0])
        gt = float(ctx.theta_wp_dict["threshold"].grad.numpy()[0])
        ctx.solver.set_theta_wp(None)

        grad_theta = torch.tensor([ga, gs, gt], dtype=torch.float32,
                                    device=grad_alpha_torch.device)
        return grad_theta, None, None, None, None, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gen_obs_multistep(solver, rest_state, theta_true, knife_traj, knife_radius):
    obs_pos = []
    obs_alpha = []
    state_in = rest_state
    for t, knife_pos in enumerate(knife_traj):
        solver.set_theta_wp(None)
        solver.alpha_min = float(theta_true[0])
        solver.soft_removal_sharpness = float(theta_true[1])
        solver.aspiration_threshold = float(theta_true[2])
        solver.knife_radius = knife_radius
        solver.knife_active = 1
        solver.knife_pos = wp.array(np.array([knife_pos], dtype=np.float32),
                                      dtype=wp.vec3, device=solver.device,
                                      requires_grad=False)
        solver_load_state(solver, state_in)
        solver.step()
        if solver.use_ssa_soft_removal:
            ci = solver._ssa_cycle_index
            alpha_np = solver._ssa_alpha_buffers[ci].numpy().copy()
        else:
            alpha_np = solver.p["alpha"].numpy().copy()
        pos_np = solver.p["pos"].numpy().copy()
        obs_pos.append(pos_np)
        obs_alpha.append(alpha_np)
        state_in = solver_save_state(solver)
    return obs_pos, obs_alpha


def compute_v0_gradient(solver, rest_state, theta_torch, knife_traj, knife_radius,
                          obs_pos_t, obs_alpha_t, lambda_cav):
    """Per-frame backward (v0 baseline) — returns accumulated grad."""
    if theta_torch.grad is not None:
        theta_torch.grad.zero_()
    state_in = rest_state
    L_frames = []
    T = len(knife_traj)
    for t in range(T):
        alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
            theta_torch, solver, state_in, knife_traj[t], knife_radius,
        )
        L_t = ((pos_t - obs_pos_t[t]) ** 2).sum() + lambda_cav * ((alpha_t - obs_alpha_t[t]) ** 2).sum()
        L_t.backward()
        L_frames.append(float(L_t.item()))
        state_in = solver_save_state(solver)
    grad = theta_torch.grad.detach().cpu().numpy().copy()
    L_total = sum(L_frames)
    return grad, L_total, L_frames


def compute_v1_gradient(solver, rest_state, theta_torch, knife_traj, knife_radius,
                          obs_pos_t, obs_alpha_t, lambda_cav):
    """Keep-state-in-graph (v1) — single big tape, single backward."""
    if theta_torch.grad is not None:
        theta_torch.grad.zero_()
    T = len(knife_traj)
    alpha_T, pos_T, dv_T = MultiFrameSolverOp.apply(
        theta_torch, solver, rest_state, knife_traj, knife_radius, T,
    )
    # alpha_T: (T, N), pos_T: (T, N, 3), dv_T: (T, N)
    L_frames_torch = []
    for t in range(T):
        L_t = ((pos_T[t] - obs_pos_t[t]) ** 2).sum() + lambda_cav * ((alpha_T[t] - obs_alpha_t[t]) ** 2).sum()
        L_frames_torch.append(L_t)
    L_total_torch = sum(L_frames_torch)
    L_total_torch.backward()
    grad = theta_torch.grad.detach().cpu().numpy().copy()
    L_frames = [float(l.item()) for l in L_frames_torch]
    return grad, float(L_total_torch.item()), L_frames


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_radius = 0.04
    T = 3
    knife_traj = make_knife_trajectory(T)
    theta_true = np.array([3.0, 20.0, 0.5], dtype=np.float32)
    theta_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)
    param_names = ["α_min", "sharpness", "τ"]

    print(f"\n[Setup] T={T}, knife sweep along x")
    print(f"[Setup] θ_true={theta_true.tolist()}  θ_init={theta_init.tolist()}")

    print(f"\n[Setup] generating obs at θ_true …")
    obs_pos, obs_alpha = gen_obs_multistep(
        solver, rest_state, theta_true, knife_traj, knife_radius)
    for t in range(T):
        cf = float((obs_alpha[t] < 0.5).mean())
        print(f"  frame {t+1}: cavity_frac={cf:.4f}")

    obs_pos_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]

    # ---- Probe λ_cav (value-balance at θ_init) ----
    theta_probe = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                 requires_grad=False)
    state_in = rest_state
    L_pos_per_frame = []
    L_cav_per_frame = []
    with torch.no_grad():
        for t in range(T):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_probe, solver, state_in, knife_traj[t], knife_radius,
            )
            L_pos_per_frame.append(float(((pos_t - obs_pos_t[t]) ** 2).sum()))
            L_cav_per_frame.append(float(((alpha_t - obs_alpha_t[t]) ** 2).sum()))
            state_in = solver_save_state(solver)
    lambda_cav = max(sum(L_pos_per_frame) / max(sum(L_cav_per_frame), 1e-12), 1e-6)
    print(f"\n[Probe] λ_cav={lambda_cav:.4e}  (value-balance at θ_init)")

    # =========================================================================
    # Comparison: v0 vs v1 gradient at θ_init
    # =========================================================================
    print(f"\n[Compare] computing gradients at θ_init …")

    theta_v0 = torch.tensor(theta_init, dtype=torch.float32, device=device,
                              requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    grad_v0, L_v0, L_frames_v0 = compute_v0_gradient(
        solver, rest_state, theta_v0, knife_traj, knife_radius,
        obs_pos_t, obs_alpha_t, lambda_cav)
    t_v0 = time.time() - t0
    mem_v0 = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"  v0 (per-frame backward, detached): "
          f"L_total={L_v0:.4e}  mem={mem_v0:.1f} MB  time={t_v0*1000:.1f}ms")
    print(f"     grad = ({grad_v0[0]:+.4e}, {grad_v0[1]:+.4e}, {grad_v0[2]:+.4e})")

    theta_v1 = torch.tensor(theta_init, dtype=torch.float32, device=device,
                              requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        grad_v1, L_v1, L_frames_v1 = compute_v1_gradient(
            solver, rest_state, theta_v1, knife_traj, knife_radius,
            obs_pos_t, obs_alpha_t, lambda_cav)
        t_v1 = time.time() - t0
        mem_v1 = torch.cuda.max_memory_allocated() / 1024 / 1024
        has_nan_v1 = bool(np.isnan(grad_v1).any())
        print(f"  v1 (single tape, keep state in graph): "
              f"L_total={L_v1:.4e}  mem={mem_v1:.1f} MB  time={t_v1*1000:.1f}ms")
        print(f"     grad = ({grad_v1[0]:+.4e}, {grad_v1[1]:+.4e}, {grad_v1[2]:+.4e})")
        v1_ran = True
    except Exception as exc:
        print(f"  v1 FAILED: {type(exc).__name__}: {exc}")
        v1_ran = False
        has_nan_v1 = True
        grad_v1 = np.array([np.nan, np.nan, np.nan])
        mem_v1 = -1; t_v1 = -1; L_v1 = -1; L_frames_v1 = []

    # ---- Comparison ----
    print("\n" + "=" * 70)
    print("v0 vs v1 gradient comparison at θ_init")
    print("=" * 70)
    if not v1_ran:
        print("  v1 crashed — see exception above")
    else:
        # Per-frame loss should be identical (forward physics same)
        print(f"  Per-frame loss agreement:")
        for t in range(T):
            diff = abs(L_frames_v0[t] - L_frames_v1[t]) / max(L_frames_v0[t], 1e-12)
            print(f"    frame {t+1}: v0={L_frames_v0[t]:.4e}  v1={L_frames_v1[t]:.4e}  "
                  f"rel_diff={diff*100:.3f}%")

        print(f"\n  Gradient comparison:")
        for d, name in enumerate(param_names):
            ratio = grad_v1[d] / grad_v0[d] if abs(grad_v0[d]) > 1e-15 else float('nan')
            diff = abs(grad_v1[d] - grad_v0[d]) / max(abs(grad_v0[d]), 1e-15)
            print(f"    {name:>10s}: v0={grad_v0[d]:+.4e}  v1={grad_v1[d]:+.4e}  "
                  f"v1/v0={ratio:+.3f}  rel_diff={diff*100:.2f}%")

        print(f"\n  Resource:")
        print(f"    Memory:  v0={mem_v0:.1f} MB  v1={mem_v1:.1f} MB  ratio={mem_v1/max(mem_v0,1):.2f}×")
        print(f"    Time:    v0={t_v0*1000:.1f}ms  v1={t_v1*1000:.1f}ms  ratio={t_v1/max(t_v0,1e-6):.2f}×")
        print(f"    NaN in v1: {'YES ✗' if has_nan_v1 else 'no ✓'}")

    if not v1_ran or has_nan_v1:
        print("\n✗ Day 6f v1 FAIL — pipeline crashed or NaN")
        return

    # =========================================================================
    # Recovery loop using v1
    # =========================================================================
    print(f"\n[Recover] v1 outer Adam, N_outer=30, lr=0.10 …")
    theta_torch = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                 requires_grad=True)
    opt = torch.optim.Adam([theta_torch], lr=0.10)
    trajectory = []
    losses = []
    per_frame_L = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    n_outer = 30
    nan_step = -1
    for step in range(n_outer):
        opt.zero_grad()
        try:
            grad_arr, L_tot, L_frames = compute_v1_gradient(
                solver, rest_state, theta_torch, knife_traj, knife_radius,
                obs_pos_t, obs_alpha_t, lambda_cav)
        except Exception as exc:
            print(f"  step {step}: v1 crashed {type(exc).__name__}: {exc}")
            nan_step = step
            break
        if np.isnan(grad_arr).any():
            print(f"  step {step}: NaN gradient — aborting")
            nan_step = step
            break
        opt.step()
        clamp_theta_inplace(theta_torch)
        th = theta_torch.detach().cpu().numpy().copy()
        trajectory.append(th)
        losses.append(L_tot)
        per_frame_L.append(L_frames)
        if step < 5 or step % 5 == 4 or step == n_outer - 1:
            print(f"  step {step:>3d}: L={L_tot:.4e}  "
                  f"θ=({th[0]:.4f},{th[1]:.4f},{th[2]:.4f})  "
                  f"grad=({grad_arr[0]:+.3e},{grad_arr[1]:+.3e},{grad_arr[2]:+.3e})")

    elapsed = time.time() - t0
    mem_recover = torch.cuda.max_memory_allocated() / 1024 / 1024

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("Day 6f v1 verdict — keep-state-in-graph recovery")
    print("=" * 70)
    print(f"  Timing: {elapsed:.1f}s for {len(losses)} outer steps  ({elapsed/max(len(losses),1):.2f}s/step)")
    print(f"  Memory peak during recovery: {mem_recover:.1f} MB")
    if nan_step >= 0:
        print(f"  ✗ aborted at step {nan_step} due to NaN/exception")
        return

    theta_final = trajectory[-1]
    rel_err_v1 = np.abs(theta_final - theta_true) / np.maximum(np.abs(theta_true), 1e-12)
    L_init = losses[0]; L_final = losses[-1]
    print(f"  L_total: {L_init:.4e} → {L_final:.4e}  "
          f"(reduction {L_init/max(L_final,1e-12):.2f}×)")
    print(f"\n  per-dim θ recovery (v1):")
    for d, name in enumerate(param_names):
        print(f"    {name:>10s}: true={theta_true[d]:.4f}  init={theta_init[d]:.4f}  "
              f"final={theta_final[d]:.4f}  rel_err={rel_err_v1[d]*100:.2f}%")

    # v0 reference (from prior run)
    print(f"\n  v0 reference (from Day 6f v0 result):")
    print(f"    α_min     err 17.13%   sharpness err 25.91%   τ err 4.82%")

    moved_correctly = []
    for d, name in enumerate(param_names):
        init_err = abs(theta_init[d] - theta_true[d]) / theta_true[d]
        ok = rel_err_v1[d] < init_err
        moved_correctly.append(ok)

    if all(moved_correctly) and L_final < L_init:
        print(f"\n✓ Day 6f v1 PASS — keep-state-in-graph pipeline works, recovery moves toward truth")
    elif L_final < L_init:
        print(f"\n⚠ Day 6f v1 PARTIAL — pipeline ok but some params don't approach truth")
    else:
        print(f"\n✗ Day 6f v1 FAIL — loss did not decrease")

    # Save
    npz_path = REPO_ROOT / "tools" / "day6f_v1_trajectory.npz"
    np.savez(npz_path,
             trajectory=np.array(trajectory),
             losses=np.array(losses),
             per_frame_L=np.array(per_frame_L),
             theta_true=theta_true, theta_init=theta_init,
             grad_v0=grad_v0, grad_v1=grad_v1,
             mem_v0=mem_v0, mem_v1=mem_v1,
             rel_err_v1=rel_err_v1)
    print(f"  → saved to {npz_path}")


if __name__ == "__main__":
    main()
