"""Day 6b: wrap full ShapeMatchingSolver.step() as a torch.autograd.Function.

Goal: prove that gradient ∂L/∂θ via PyTorch backward (using full solver forward
that produces BOTH positions and damage state) matches FD ground truth.

Day 6a proved the bridge works for the damage-only forward.
Day 6b proves it works for the FULL forward (PBD position update + cohesive zone).

The key enabling change (in solver.py): set_theta_wp({α_min, sharpness, τ,
knife_radius}) → cutting block uses *_ssa_param_arr kernels, gradient flows
back to those wp.arrays.

Test plan:
  1. Set up solver with cube_tet.msh, rest_state.
  2. Build WarpFullForwardOp(torch.autograd.Function).
     forward: (θ_torch, state_in_dict_torch, knife_torch) → (pos, dv, α) torch tensors
     backward: tape.backward with seeded gradients → θ.grad torch tensor
  3. Compute simple torch-side loss (e.g., MSE on alpha vs target)
  4. .backward() → θ.grad
  5. Compare to central finite-difference gradient over 3 cohesive params

Pass criterion: rel_err < 2% on each of (α_min, sharpness, τ).
(Looser than Day 6a's 1% because the full forward has more numerical paths;
FD ε must also span a more chaotic regime.)
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pf_shape_matching_warp"))

import warp as wp
import torch

wp.init()
print(f"Warp: {wp.__version__}")
print(f"Torch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")

from tools.day2_solver_wrapper import (  # noqa: E402
    setup_solver_with_rest, solver_load_state, solver_save_state,
)


# ---------------------------------------------------------------------------
# Helper: set solver theta from torch tensor (wp.arrays the solver retains)
# ---------------------------------------------------------------------------

def make_theta_wp_arrays(theta_torch, knife_radius, stiff_pp_torch=None,
                            tau_pp_torch=None):
    """Build the wp.array dict expected by solver.set_theta_wp().

    theta_torch: (3,) — (α_min, sharpness, threshold)
              or (4,) — (α_min, sharpness, threshold, stiffness)  [Day 3b 2026-05-23]
    knife_radius: scalar float
    stiff_pp_torch: (N,) torch tensor, optional [Stage A 2026-05-24]
                  When provided, overrides theta_torch[3] (if any) and dispatches
                  to per-particle stiffness kernel in solver.
    tau_pp_torch:  (N,) torch tensor, optional [Stage E 2026-05-25]
                  When provided, overrides theta_torch[2] and dispatches to
                  per-particle τ kernel in solver.

    Returns dict of wp.array, where stiffness/threshold shape is (1,) or (N,)
    depending on per-particle vs global.
    """
    theta_torch = theta_torch.contiguous()
    n = theta_torch.shape[0]
    assert n in (3, 4), f"theta_torch must be (3,) or (4,), got {theta_torch.shape}"
    d = dict(
        alpha_min=wp.from_torch(theta_torch[0:1].contiguous(),
                                  requires_grad=True),
        sharpness=wp.from_torch(theta_torch[1:2].contiguous(),
                                  requires_grad=True),
        knife_radius=wp.array([float(knife_radius)], dtype=wp.float32,
                               device="cuda:0", requires_grad=False),
    )
    # threshold: scalar (1,) by default; (N,) if per-particle tau provided
    if tau_pp_torch is not None:
        assert tau_pp_torch.dim() == 1, (
            f"tau_pp_torch must be 1D (N,), got shape {tau_pp_torch.shape}")
        d["threshold"] = wp.from_torch(tau_pp_torch.contiguous(),
                                         requires_grad=True)
    else:
        d["threshold"] = wp.from_torch(theta_torch[2:3].contiguous(),
                                         requires_grad=True)
    # stiffness: same dispatch pattern as before
    if stiff_pp_torch is not None:
        assert stiff_pp_torch.dim() == 1, (
            f"stiff_pp_torch must be 1D (N,), got shape {stiff_pp_torch.shape}")
        d["stiffness"] = wp.from_torch(stiff_pp_torch.contiguous(),
                                         requires_grad=True)
    elif n == 4:
        d["stiffness"] = wp.from_torch(theta_torch[3:4].contiguous(),
                                         requires_grad=True)
    return d


# ---------------------------------------------------------------------------
# torch.autograd.Function wrapping the full solver.step()
# ---------------------------------------------------------------------------

class WarpFullForwardOp(torch.autograd.Function):
    """Wraps a single solver.step() as a differentiable torch op.

    Forward inputs:
        theta_torch    — (3,) tensor: (α_min, sharpness, threshold)
        ctx_args       — dict with 'solver', 'state_in', 'knife_pos', 'knife_radius'
                         (state_in and knife_pos may be torch tensors or numpy)

    Forward outputs:
        alpha_out_torch  — (N,) tensor
        pos_out_torch    — (N, 3) tensor
        dv_out_torch     — (N,) tensor

    Backward inputs (from upstream):
        grad_alpha, grad_pos, grad_dv  — gradients on each output

    Backward outputs:
        grad_theta_torch  — (3,) tensor
    """

    @staticmethod
    def forward(ctx, theta_torch, solver, state_in_dict, knife_pos_np, knife_radius,
                stiff_pp_torch=None, tau_pp_torch=None):
        # Build wp.array theta with grad tracking
        # theta_torch shape: (3,) → (α, sharp, τ) or (4,) → (α, sharp, τ, stiffness)
        # stiff_pp_torch: optional (N,) per-particle stiffness override [Stage A]
        # tau_pp_torch:   optional (N,) per-particle τ override [Stage E 2026-05-25]
        ctx.theta_n = theta_torch.shape[0]
        ctx.has_stiff_pp = stiff_pp_torch is not None
        ctx.has_tau_pp = tau_pp_torch is not None
        theta_wp_dict = make_theta_wp_arrays(theta_torch, knife_radius,
                                                stiff_pp_torch=stiff_pp_torch,
                                                tau_pp_torch=tau_pp_torch)
        solver.set_theta_wp(theta_wp_dict)

        # Load state into solver (numpy/torch → wp.arrays)
        solver_load_state(solver, state_in_dict)

        # Knife pos
        solver.knife_pos = wp.array(np.array([knife_pos_np], dtype=np.float32),
                                      dtype=wp.vec3, device=solver.device,
                                      requires_grad=False)
        solver.knife_active = 1

        # Run forward under wp.Tape
        tape = wp.Tape()
        with tape:
            solver.step()

        # Snapshot output state (the wp.arrays we need to seed gradient on)
        if solver.use_ssa_soft_removal:
            ci = solver._ssa_cycle_index
            alpha_out_wp = solver._ssa_alpha_buffers[ci]
            dv_out_wp = solver._ssa_particle_damage_buffers[ci]
        else:
            alpha_out_wp = solver.p["alpha"]
            dv_out_wp = solver.p["damage"]
        pos_out_wp = solver.p["pos"]

        ctx.tape = tape
        ctx.solver = solver
        ctx.theta_wp_dict = theta_wp_dict
        ctx.alpha_out_wp = alpha_out_wp
        ctx.dv_out_wp = dv_out_wp
        ctx.pos_out_wp = pos_out_wp

        # Return torch views (clone so subsequent assignment of upstream grad
        # doesn't clobber wp memory)
        alpha_torch = wp.to_torch(alpha_out_wp).clone()
        pos_torch = wp.to_torch(pos_out_wp).clone()
        dv_torch = wp.to_torch(dv_out_wp).clone()
        return alpha_torch, pos_torch, dv_torch

    @staticmethod
    def backward(ctx, grad_alpha_torch, grad_pos_torch, grad_dv_torch):
        # Build seed gradient dict for Warp tape backward
        grads_dict = {}
        if grad_alpha_torch is not None:
            grads_dict[ctx.alpha_out_wp] = wp.from_torch(
                grad_alpha_torch.contiguous()
            )
        if grad_pos_torch is not None:
            grads_dict[ctx.pos_out_wp] = wp.from_torch(
                grad_pos_torch.contiguous()
            )
        if grad_dv_torch is not None:
            grads_dict[ctx.dv_out_wp] = wp.from_torch(
                grad_dv_torch.contiguous()
            )

        # Run backward through tape — populates theta_wp[*].grad
        ctx.tape.backward(grads=grads_dict)

        # Read back gradients on global theta dims
        ga = float(ctx.theta_wp_dict["alpha_min"].grad.numpy()[0])
        gs = float(ctx.theta_wp_dict["sharpness"].grad.numpy()[0])

        # Threshold gradient — depends on threshold shape
        # Stage E: if per-particle τ, threshold grad is (N,) → return as grad_tau_pp;
        #          pad theta[2] grad with 0
        grad_tau_pp = None
        if ctx.has_tau_pp:
            tau_grad_wp = ctx.theta_wp_dict["threshold"].grad
            grad_tau_pp = torch.from_numpy(tau_grad_wp.numpy().copy()).to(
                grad_alpha_torch.device)
            gt = 0.0  # theta[2] not directly used; gradient went to tau_pp
        else:
            gt = float(ctx.theta_wp_dict["threshold"].grad.numpy()[0])
        grad_list = [ga, gs, gt]

        # Stiffness gradient — depends on stiffness shape
        grad_stiff_pp = None  # (N,) tensor if per-particle
        if "stiffness" in ctx.theta_wp_dict:
            stiff_grad_wp = ctx.theta_wp_dict["stiffness"].grad
            if ctx.has_stiff_pp:
                # Per-particle: keep as (N,) tensor for separate return
                grad_stiff_pp = torch.from_numpy(stiff_grad_wp.numpy().copy()).to(
                    grad_alpha_torch.device)
                # Pad theta grad with 0 for index [3] so theta_torch maintains its shape
                if ctx.theta_n == 4:
                    grad_list.append(0.0)
            else:
                # Global scalar (Day 3b path)
                gst = float(stiff_grad_wp.numpy()[0])
                grad_list.append(gst)

        # Clear theta_wp from solver (avoid stale references)
        ctx.solver.set_theta_wp(None)

        # Day 3b: zero gradient buffers on all wp.arrays touched by this tape.
        ctx.tape.zero()

        grad_theta = torch.tensor(grad_list, dtype=torch.float32,
                                    device=grad_alpha_torch.device)
        # Return: (grad_theta, None×4 for non-grad inputs, grad_stiff_pp, grad_tau_pp)
        return grad_theta, None, None, None, None, grad_stiff_pp, grad_tau_pp


# ---------------------------------------------------------------------------
# Test: PyTorch bridge vs FD ground truth
# ---------------------------------------------------------------------------

def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    print(f"\nSolver: N={solver.N}  E={int(solver.e['num_edges'])}")

    # ---- Setup ----
    knife_pos = np.array([0.15, 0.30, 0.15], dtype=np.float32)
    knife_radius = 0.04
    theta_true = np.array([3.0, 20.0, 0.5], dtype=np.float32)
    theta_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)

    # ---- Generate target via 1-step forward at theta_true ----
    print("\nGenerating α_target at θ_true (1 step) …")
    # Use solver directly (legacy float path) for target gen
    solver.alpha_min = float(theta_true[0])
    solver.soft_removal_sharpness = float(theta_true[1])
    solver.aspiration_threshold = float(theta_true[2])
    solver.knife_radius = knife_radius
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
    print(f"  target cavity: {float((alpha_target_np<0.5).mean()):.4f}")

    # ---- Path A: torch bridge gradient ----
    print("\n[Path A] PyTorch backward via WarpFullForwardOp …")
    device = "cuda:0"
    theta_torch = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                 requires_grad=True)
    alpha_target_torch = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)

    # Forward: full solver step
    alpha_out, pos_out, dv_out = WarpFullForwardOp.apply(
        theta_torch, solver, rest_state, knife_pos, knife_radius,
    )

    # Torch-side loss: MSE on alpha (kernel convention: sum diff², no 0.5)
    N = solver.N
    diff = alpha_out - alpha_target_torch
    loss_torch = (diff ** 2).sum() / float(N)
    print(f"  loss = {loss_torch.item():.6e}")

    # Backward
    loss_torch.backward()
    grad_A = theta_torch.grad.cpu().numpy().copy()
    print(f"  grad = ({grad_A[0]:+.6e}, {grad_A[1]:+.6e}, {grad_A[2]:+.6e})")

    # ---- Path B: central finite-difference ground truth ----
    print("\n[Path B] Central FD ground truth …")

    def loss_at(theta_np):
        """Eval loss = (1/N)·Σ(α - α_target)² at given theta, using legacy float path."""
        solver.set_theta_wp(None)  # ensure legacy path
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
            alpha = solver._ssa_alpha_buffers[ci].numpy()
        else:
            alpha = solver.p["alpha"].numpy()
        return float(np.sum((alpha - alpha_target_np)**2)) / float(N)

    eps_per_dim = [1e-2, 1e-1, 1e-2]
    grad_B = np.zeros(3)
    for d in range(3):
        eps = eps_per_dim[d]
        th_plus = theta_init.copy(); th_plus[d] += eps
        th_minus = theta_init.copy(); th_minus[d] -= eps
        L_p = loss_at(th_plus); L_m = loss_at(th_minus)
        grad_B[d] = (L_p - L_m) / (2 * eps)
    print(f"  grad = ({grad_B[0]:+.6e}, {grad_B[1]:+.6e}, {grad_B[2]:+.6e})")

    # ---- Parity check ----
    print("\n" + "=" * 64)
    print("Day 6b parity check: torch full-forward bridge ↔ FD")
    print("=" * 64)
    rel_err = np.abs(grad_A - grad_B) / np.maximum(np.abs(grad_B), 1e-12)
    sign_match = np.sign(grad_A) == np.sign(grad_B)
    for d, name in enumerate(["α_min", "sharpness", "τ"]):
        ag = grad_A[d]; bg = grad_B[d]
        rel = rel_err[d] * 100
        sgn = "✓" if sign_match[d] else "✗"
        print(f"  {name:>10s}: bridge={ag:+.4e}  FD={bg:+.4e}  rel_err={rel:.3f}%  sign {sgn}")

    nonzero = (np.abs(grad_A) > 1e-12).all()
    pass_strict = bool(sign_match.all() and (rel_err < 0.02).all())   # < 2%
    pass_loose = bool(sign_match.all() and (rel_err < 0.20).all())    # < 20%

    print()
    if pass_strict:
        print(f"✓ Day 6b PASS — torch bridge through full solver matches FD < 2%")
    elif pass_loose:
        print(f"⚠ Day 6b partial — signs match, magnitudes < 20% but not < 2%")
    elif nonzero and sign_match.all():
        print(f"⚠ Day 6b partial — non-zero, signs match, but large numerical diff")
    else:
        print(f"✗ Day 6b FAIL — see diagnostics above")


if __name__ == "__main__":
    main()
