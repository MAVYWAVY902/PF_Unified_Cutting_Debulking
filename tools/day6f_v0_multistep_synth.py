"""Day 6f v0: T=3 frame multi-step synth-to-synth ReSECT recovery.

Liang Algorithm 1 pattern: per-frame backward, state DETACHED between frames.
This means gradient at frame t only captures ∂L_t/∂θ via THIS frame's forward
— the indirect path ∂L_t/∂state_{t-1}·∂state_{t-1}/∂θ is dropped. This is
the (a) approach from our design discussion.

Pipeline per outer Adam step:
  state_0 = rest_state
  opt.zero_grad()
  for t in 1..T:
    α_t, pos_t, dv_t = WarpFullForwardOp(θ, solver, state_{t-1}, knife_t, r)
    L_t = ‖pos_t − obs_pos_t‖² + λ·‖α_t − obs_α_t‖²
    L_t.backward()                    # accumulates into θ.grad
    state_t = solver_save_state()     # detached snapshot for next frame
  opt.step()

Goals for v0 (minimal viable):
  (1) Pipeline runs T=3 without crash or OOM
  (2) Per-frame backward accumulates θ.grad correctly
  (3) Loss decreases over outer steps
  (4) Recovery progress (θ moves toward θ_true) — NOT necessarily 5% precision
  (5) Memory budget tracked

NOT in v0 (deferred to v1):
  - keep-state-in-graph for true multi-frame backward
  - OOM stress test
  - chaotic gradient analysis
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
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402


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
    """Sweep knife along x at top surface."""
    x_positions = np.linspace(0.12, 0.18, T)
    return [np.array([x, 0.30, 0.15], dtype=np.float32) for x in x_positions]


def gen_obs_multistep(solver, rest_state, theta_true, knife_traj, knife_radius):
    """Generate T frames of (pos, alpha) obs at θ_true via legacy float path."""
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
        cavity_frac = float((alpha_np < 0.5).mean())
        print(f"  obs frame {t+1}/{len(knife_traj)}: knife={knife_pos.tolist()}  "
              f"cavity_frac={cavity_frac:.4f}  ({int(cavity_frac*solver.N)} particles)")
    return obs_pos, obs_alpha


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

    print(f"\n[Setup] T={T} frames")
    print(f"[Setup] θ_true = {theta_true.tolist()}")
    print(f"[Setup] θ_init = {theta_init.tolist()}")
    print(f"[Setup] knife trajectory:")
    for t, k in enumerate(knife_traj):
        print(f"  frame {t+1}: {k.tolist()}")

    # ---- Generate obs (T frames at θ_true) ----
    print(f"\n[Setup] generating T={T} frames of obs at θ_true …")
    obs_pos, obs_alpha = gen_obs_multistep(
        solver, rest_state, theta_true, knife_traj, knife_radius)

    # Move obs to torch
    obs_pos_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos]
    obs_alpha_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha]

    # ---- Probe λ_cav per frame at θ_init (value-balance) ----
    print(f"\n[Probe] computing λ_cav at θ_init across T frames …")
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
            L_pos = float(((pos_t - obs_pos_t[t]) ** 2).sum())
            L_cav = float(((alpha_t - obs_alpha_t[t]) ** 2).sum())
            L_pos_per_frame.append(L_pos)
            L_cav_per_frame.append(L_cav)
            state_in = solver_save_state(solver)
            print(f"  frame {t+1}: L_pos={L_pos:.4e}  L_cav={L_cav:.4e}")
    lambda_cav = max(sum(L_pos_per_frame) / max(sum(L_cav_per_frame), 1e-12), 1e-6)
    print(f"  → λ_cav = {lambda_cav:.4e}  (value-balance over T frames)")

    # ---- Outer Adam loop ----
    print(f"\n[Recover] outer Adam, T={T} frames, per-frame backward (detached state)")
    print(f"  N_outer = 30, lr = 0.10")

    theta_torch = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                 requires_grad=True)
    opt = torch.optim.Adam([theta_torch], lr=0.10)

    trajectory = []
    losses = []
    per_frame_L = []
    mem_peak_mb = 0

    n_outer = 30
    t0 = time.time()
    print(f"\n  {'step':>4s}  {'L_tot':>11s}  {'L_1':>11s}  {'L_2':>11s}  {'L_3':>11s}  "
          f"{'α':>7s}  {'s':>7s}  {'τ':>7s}  {'grad α':>10s}  {'grad s':>10s}  {'grad τ':>10s}")

    for step in range(n_outer):
        opt.zero_grad()
        state_in = rest_state
        L_frames = []
        for t in range(T):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, knife_traj[t], knife_radius,
            )
            L_pos = ((pos_t - obs_pos_t[t]) ** 2).sum()
            L_cav = ((alpha_t - obs_alpha_t[t]) ** 2).sum()
            L_t = L_pos + lambda_cav * L_cav
            L_t.backward()  # accumulates into θ.grad
            L_frames.append(float(L_t.item()))
            state_in = solver_save_state(solver)  # detached snapshot

        # Track memory
        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        mem_peak_mb = max(mem_peak_mb, mem_mb)

        grad_vals = theta_torch.grad.detach().cpu().numpy().copy()
        opt.step()
        clamp_theta_inplace(theta_torch)

        L_tot = sum(L_frames)
        th = theta_torch.detach().cpu().numpy().copy()
        trajectory.append(th)
        losses.append(L_tot)
        per_frame_L.append(L_frames)

        if step < 5 or step % 5 == 4 or step == n_outer - 1:
            print(f"  {step:>4d}  {L_tot:>11.4e}  {L_frames[0]:>11.4e}  "
                  f"{L_frames[1]:>11.4e}  {L_frames[2]:>11.4e}  "
                  f"{th[0]:>7.4f}  {th[1]:>7.4f}  {th[2]:>7.4f}  "
                  f"{grad_vals[0]:>+10.3e}  {grad_vals[1]:>+10.3e}  {grad_vals[2]:>+10.3e}")

    elapsed = time.time() - t0
    print(f"\n[Timing] {n_outer} outer × {T} frames = {n_outer*T} forwards in {elapsed:.1f}s "
          f"({elapsed/(n_outer*T):.2f}s/forward)")
    print(f"[Memory] peak {mem_peak_mb:.1f} MB during recovery")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("Day 6f v0 — multi-step synth-to-synth verdict")
    print("=" * 70)
    theta_final = trajectory[-1]
    rel_err = np.abs(theta_final - theta_true) / np.maximum(np.abs(theta_true), 1e-12)
    L_init = losses[0]; L_final = losses[-1]
    print(f"  L_total: {L_init:.4e} → {L_final:.4e}  "
          f"(reduction {L_init/max(L_final,1e-12):.2f}×)")
    print(f"\n  per-frame loss (init → final):")
    for t in range(T):
        print(f"    frame {t+1}: {per_frame_L[0][t]:.4e} → {per_frame_L[-1][t]:.4e}")
    print(f"\n  per-dim θ recovery:")
    for d, name in enumerate(param_names):
        print(f"    {name:>10s}: true={theta_true[d]:.4f}  init={theta_init[d]:.4f}  "
              f"final={theta_final[d]:.4f}  rel_err={rel_err[d]*100:.2f}%")

    # Pass/fail by goal
    print(f"\n  Goal checks:")
    print(f"    (1) Pipeline ran T={T} frames × {n_outer} outer steps: ✓")
    print(f"    (2) Per-frame backward + θ.grad accumulation: ✓")
    print(f"    (3) Loss decreased: {'✓' if L_final < L_init else '✗'}")
    print(f"    (4) θ moved TOWARD truth (init err > final err per dim):")
    moved_correctly = []
    for d, name in enumerate(param_names):
        init_err = abs(theta_init[d] - theta_true[d]) / theta_true[d]
        final_err = rel_err[d]
        ok = final_err < init_err
        moved_correctly.append(ok)
        print(f"         {name:>10s}: init_err={init_err*100:.1f}% → final_err={final_err*100:.1f}%  "
              f"{'✓' if ok else '✗'}")
    print(f"    (5) Memory budget (peak {mem_peak_mb:.0f} MB / 8192 MB): "
          f"{'✓' if mem_peak_mb < 7000 else '⚠'}")

    if all(moved_correctly) and L_final < L_init:
        print(f"\n✓ Day 6f v0 PASS — multi-frame pipeline works, recovery direction correct")
    elif L_final < L_init:
        print(f"\n⚠ Day 6f v0 PARTIAL — pipeline ok but coupling remains "
              f"(some params don't approach truth)")
    else:
        print(f"\n✗ Day 6f v0 FAIL — loss did not decrease")

    # Save trajectory
    npz_path = REPO_ROOT / "tools" / "day6f_v0_trajectory.npz"
    np.savez(npz_path,
             trajectory=np.array(trajectory),
             losses=np.array(losses),
             per_frame_L=np.array(per_frame_L),
             theta_true=theta_true,
             theta_init=theta_init,
             knife_traj=np.array(knife_traj),
             lambda_cav=lambda_cav,
             rel_err_final=rel_err,
             mem_peak_mb=mem_peak_mb)
    print(f"\n  → trajectory saved to {npz_path}")


if __name__ == "__main__":
    main()
