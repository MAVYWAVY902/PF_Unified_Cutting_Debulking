"""Day 6d v2 follow-up: FD ε-sweep for α_min.

The original Day 6d v2 test showed:
  α_min:     autograd=+0.6394   FD(ε=1e-2)=+0.6104   rel_err=4.77%
  sharpness: autograd=...       FD(ε=1e-1)=...       rel_err=0.008%
  τ:         autograd=...       FD(ε=1e-2)=...       rel_err=0.026%

α_min is the only param > 0.1% rel_err. Hypothesis: 4.77% is FD ε noise
floor (the chain through edge_damage→particle_damage→soft_removal is more
chaotic than the direct paths for sharpness/τ), NOT autograd error.

Test: sweep α_min FD over ε ∈ {2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1} with
sharpness/τ frozen at θ_init. Compute FD grad at each ε.

Verdicts:
  - If FD itself varies significantly across ε (Nesterov-Spokoiny U-curve)
    → 4.77% is FD precision floor, autograd is the trustworthy value
  - If FD is stable across ε but differs from autograd → autograd has a bug
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pf_shape_matching_warp"))

import torch
import warp as wp
wp.init()
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

from tools.day2_solver_wrapper import (  # noqa: E402
    setup_solver_with_rest, solver_load_state,
)
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day6d_v2_simple_gd import simple_gd_inner_loop  # noqa: E402


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    device = "cuda:0"
    knife_pos = np.array([0.15, 0.30, 0.15], dtype=np.float32)
    knife_radius = 0.04
    theta_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)

    # ---- Generate obs ----
    print("\nGenerating obs target …")
    solver.set_theta_wp(None)
    solver.alpha_min = 3.0; solver.soft_removal_sharpness = 20.0
    solver.aspiration_threshold = 0.5; solver.knife_radius = knife_radius
    solver.knife_active = 1
    solver.knife_pos = wp.array(np.array([knife_pos], dtype=np.float32),
                                  dtype=wp.vec3, device=solver.device, requires_grad=False)
    solver_load_state(solver, rest_state)
    solver.step()
    if solver.use_ssa_soft_removal:
        ci = solver._ssa_cycle_index
        alpha_target_np = solver._ssa_alpha_buffers[ci].numpy().copy()
    else:
        alpha_target_np = solver.p["alpha"].numpy().copy()
    pos_target_np = solver.p["pos"].numpy().copy()

    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_rest_t = torch.tensor(solver.e["pf_rest"].numpy(),
                                  dtype=torch.float32, device=device)
    pos_target_t = torch.tensor(pos_target_np, dtype=torch.float32, device=device)
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)

    # ====== Path A: autograd grad α_min (one shot) ======
    print("\n[Path A] autograd grad α_min …")
    theta_torch = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                 requires_grad=True)
    alpha, pos, dv = WarpFullForwardOp.apply(
        theta_torch, solver, rest_state, knife_pos, knife_radius,
    )
    delta_x, delta_dv, _ = simple_gd_inner_loop(
        pos_pred=pos, dv_pred=dv,
        pos_target=pos_target_t, alpha_target=alpha_target_t,
        theta=theta_torch,
        edges_a=edges_a_t, edges_b=edges_b_t, edges_rest=edges_rest_t,
        n_iter=10, lr_pos=5e-3, lr_dv=0.1,
        lambda_cav=1.0, lambda_E=1.0,
    )
    L_outer = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
    L_outer.backward()
    grad_A = float(theta_torch.grad[0].item())
    print(f"  autograd grad α_min = {grad_A:+.6e}")

    # ====== Path B: FD grad α_min, sweep over ε ======
    def L_outer_at_no_tape(theta_np, n_inner=10):
        """Forward via legacy float, inner loop in torch without create_graph."""
        solver.set_theta_wp(None)
        solver.alpha_min = float(theta_np[0])
        solver.soft_removal_sharpness = float(theta_np[1])
        solver.aspiration_threshold = float(theta_np[2])
        solver.knife_radius = knife_radius
        solver.knife_active = 1
        solver.knife_pos = wp.array(np.array([knife_pos], dtype=np.float32),
                                      dtype=wp.vec3, device=solver.device,
                                      requires_grad=False)
        solver_load_state(solver, rest_state)
        solver.step()

        if solver.use_ssa_soft_removal:
            ci = solver._ssa_cycle_index
            alpha_np = solver._ssa_alpha_buffers[ci].numpy()
            dv_np = solver._ssa_particle_damage_buffers[ci].numpy()
        else:
            alpha_np = solver.p["alpha"].numpy()
            dv_np = solver.p["damage"].numpy()
        pos_np = solver.p["pos"].numpy()

        pos_t = torch.tensor(pos_np, dtype=torch.float32, device=device)
        dv_t = torch.tensor(dv_np, dtype=torch.float32, device=device)
        theta_t = torch.tensor(theta_np, dtype=torch.float32, device=device)

        N = pos_t.shape[0]; s = theta_t[1]; tau = theta_t[2]
        delta_x = torch.zeros((N, 3), dtype=torch.float32, device=device,
                                requires_grad=True)
        delta_dv_raw = torch.zeros(N, dtype=torch.float32, device=device,
                                      requires_grad=True)
        for _ in range(n_inner):
            pos_corr = pos_t + delta_x
            delta_dv = torch.nn.functional.softplus(delta_dv_raw)
            dv_corr = dv_t + delta_dv
            alpha_corr = torch.sigmoid(-s * (dv_corr - tau))
            L_pos = ((pos_corr - pos_target_t) ** 2).sum()
            L_cav = ((alpha_corr - alpha_target_t) ** 2).sum()
            edge_vec = pos_corr[edges_a_t] - pos_corr[edges_b_t]
            edge_len = edge_vec.norm(dim=1).clamp(min=1e-12)
            L_E = ((edge_len - edges_rest_t) ** 2).sum()
            L_inner = L_pos + L_cav + L_E
            grad_x, grad_raw = torch.autograd.grad(
                L_inner, (delta_x, delta_dv_raw), retain_graph=False)
            with torch.no_grad():
                delta_x = (delta_x - 5e-3 * grad_x).requires_grad_(True)
                delta_dv_raw = (delta_dv_raw - 0.1 * grad_raw).requires_grad_(True)
        delta_dv_final = torch.nn.functional.softplus(delta_dv_raw)
        return float((delta_x ** 2).sum() + (delta_dv_final ** 2).sum())

    eps_list = [2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
    print(f"\n[Path B] FD α_min sweep over ε ∈ {eps_list} …")
    print(f"  {'ε':>8s}  {'L(+ε)':>14s}  {'L(-ε)':>14s}  {'L(0)':>14s}  "
          f"{'FD grad':>13s}  {'rel vs A':>10s}  {'(L+-L-)/L0':>12s}")

    L_0 = L_outer_at_no_tape(theta_init.copy(), n_inner=10)
    print(f"  L(θ_init) = {L_0:.6e}")

    fd_grads = []
    for eps in eps_list:
        th_plus = theta_init.copy(); th_plus[0] += eps
        th_minus = theta_init.copy(); th_minus[0] -= eps
        L_p = L_outer_at_no_tape(th_plus, n_inner=10)
        L_m = L_outer_at_no_tape(th_minus, n_inner=10)
        g = (L_p - L_m) / (2 * eps)
        rel = abs(g - grad_A) / max(abs(grad_A), 1e-12)
        signal = (L_p - L_m) / max(L_0, 1e-12)
        fd_grads.append(g)
        print(f"  {eps:>8.0e}  {L_p:>14.6e}  {L_m:>14.6e}  {L_0:>14.6e}  "
              f"{g:>+13.6e}  {rel*100:>9.3f}%  {signal:>+12.3e}")

    fd_grads = np.array(fd_grads)
    print(f"\n  FD grad α_min over ε: min={fd_grads.min():+.4e}  "
          f"max={fd_grads.max():+.4e}  spread={fd_grads.max()-fd_grads.min():+.4e}")
    print(f"  spread / |autograd| = {(fd_grads.max()-fd_grads.min())/max(abs(grad_A),1e-12)*100:.2f}%")

    # ====== Verdict ======
    print("\n" + "=" * 70)
    print("Verdict: α_min 4.77% rel_err — is it FD noise floor?")
    print("=" * 70)
    fd_self_spread = (fd_grads.max() - fd_grads.min()) / max(abs(grad_A), 1e-12)
    autograd_vs_fd_mean = abs(grad_A - fd_grads.mean()) / max(abs(grad_A), 1e-12)
    if fd_self_spread > 0.03:
        print(f"  FD-self spread = {fd_self_spread*100:.2f}% → FD itself is noisy across ε")
        print(f"  → 4.77% rel_err vs autograd is consistent with FD noise floor")
        print(f"  → autograd is the trustworthy value")
        print(f"  ✓ Hypothesis CONFIRMED: α_min 4.77% is FD ε noise floor")
    else:
        print(f"  FD-self spread = {fd_self_spread*100:.2f}% → FD is stable across ε")
        print(f"  autograd vs FD mean = {autograd_vs_fd_mean*100:.2f}%")
        if autograd_vs_fd_mean > 0.03:
            print(f"  ✗ Hypothesis REJECTED: autograd genuinely differs from stable FD")
        else:
            print(f"  ⚠ Ambiguous: both stable + close")


if __name__ == "__main__":
    main()
