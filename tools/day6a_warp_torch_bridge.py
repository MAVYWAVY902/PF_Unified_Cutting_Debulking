"""Day 6a: minimal Warp ↔ PyTorch autograd bridge verification.

Goal: prove that gradient computed via torch.autograd.Function wrapping a
Warp Tape EQUALS the gradient from a direct Warp Tape backward.

Why this matters:
  - Liang 2023 uses PyTorch autograd to unroll 30 inner-loop steps
  - For Day 6 (faithful ReSECT), we want the same: PyTorch on outside, Warp on inside
  - This requires that gradients pass cleanly across the wp.from_torch / wp.to_torch boundary
  - Day 3 already proved Warp direct backward matches FD at 0.03% rel_err
  - So if Day 6a's PyTorch bridge gives the SAME gradient → bridge is correct

Test plan:
  1. Build the same setup as Day 3: cube_tet.msh + θ_init + θ_true
  2. Path A (ground truth): direct Warp tape backward → grad_warp_direct
  3. Path B (test): wrap forward as torch.autograd.Function, compute torch loss,
     call torch loss.backward() → grad_via_torch
  4. Verify grad_via_torch ≈ grad_warp_direct (rel_err < 1%)
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import warp as wp
import torch

wp.init()
print(f"Warp: {wp.__version__}")
print(f"Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print()

from tools.online_residual_synth import (  # noqa: E402
    load_msh_system, make_params, run_ssa_full_with_state, mse_loss_kernel,
)


# ---------------------------------------------------------------------------
# torch.autograd.Function wrapping the Warp damage chain
# ---------------------------------------------------------------------------

class WarpDamageForward(torch.autograd.Function):
    """Wraps run_ssa_full_with_state in a torch.autograd.Function.

    Forward:
        Inputs (torch tensors):
            theta_torch — (3,) float32 on cuda: (α_min, sharpness, τ)
        Outputs:
            alpha_torch — (N,) float32 on cuda: final α from SSA chain

    Backward:
        Receives grad_alpha_torch — (N,) gradient from upstream torch loss
        Returns grad_theta_torch — (3,) gradient pushed back to θ

    The Warp Tape is held in ctx so backward can run the existing
    differentiable pipeline. All gradient computation happens via
    wp.array.grad → wp.to_torch.
    """

    @staticmethod
    def forward(ctx, theta_torch, sys_dict_static, knife_radius, n_cycles):
        # Detach + ensure contiguous + correct dtype
        theta_np = theta_torch.detach().cpu().numpy().astype(np.float32)

        # Build params with requires_grad=True (Warp side)
        params = make_params(
            alpha_min=float(theta_np[0]),
            sharpness=float(theta_np[1]),
            threshold=float(theta_np[2]),
            knife_radius=knife_radius,
            requires_grad=True,
        )

        # Open Warp Tape, run forward
        tape = wp.Tape()
        with tape:
            alpha_wp, _delta_v_wp, _edge_dmg_wp = run_ssa_full_with_state(
                params, sys_dict_static, n_cycles=n_cycles, state_in=None,
            )

        # Stash for backward
        ctx.tape = tape
        ctx.params = params
        ctx.alpha_wp = alpha_wp

        # Convert alpha to torch (shares memory; we want a clone so grad flows
        # cleanly when torch later writes its own gradient to the output)
        alpha_torch = wp.to_torch(alpha_wp).clone()
        return alpha_torch

    @staticmethod
    def backward(ctx, grad_alpha_torch):
        # Seed Warp's alpha gradient with what torch hands us
        # alpha_wp.grad is a wp.array — assign from torch
        grad_alpha_torch = grad_alpha_torch.contiguous()
        grad_wp = wp.from_torch(grad_alpha_torch)

        # Warp tape backward needs grads dict: {output: dL/d_output}
        ctx.tape.backward(grads={ctx.alpha_wp: grad_wp})

        # Read back gradients from Warp params (each is shape (1,))
        # NOTE: Warp 1.13's tape.backward populates params["x"].grad in place
        grad_alpha_min = float(ctx.params["alpha_min"].grad.numpy()[0])
        grad_sharpness = float(ctx.params["sharpness"].grad.numpy()[0])
        grad_threshold = float(ctx.params["threshold"].grad.numpy()[0])

        # Pack into a torch tensor
        grad_theta = torch.tensor(
            [grad_alpha_min, grad_sharpness, grad_threshold],
            dtype=torch.float32, device=grad_alpha_torch.device,
        )
        # No gradient for sys_dict_static, knife_radius, n_cycles (constants)
        return grad_theta, None, None, None


# ---------------------------------------------------------------------------
# Ground truth: direct Warp Tape gradient (as in Day 3)
# ---------------------------------------------------------------------------

def direct_warp_gradient(theta_np, sys_dict, alpha_target_np,
                          knife_radius=0.04, n_cycles=3):
    """Direct Warp tape backward — exactly Day 3 path. Returns (loss, grad_3d)."""
    N = sys_dict["N"]
    params = make_params(
        alpha_min=float(theta_np[0]),
        sharpness=float(theta_np[1]),
        threshold=float(theta_np[2]),
        knife_radius=knife_radius, requires_grad=True,
    )
    target_wp = wp.array(alpha_target_np.astype(np.float32), dtype=wp.float32)
    loss_buf = wp.zeros(1, dtype=wp.float32, requires_grad=True)

    tape = wp.Tape()
    with tape:
        alpha_sim_wp, _, _ = run_ssa_full_with_state(
            params, sys_dict, n_cycles=n_cycles, state_in=None,
        )
        wp.launch(mse_loss_kernel, dim=N,
                  inputs=[alpha_sim_wp, target_wp, loss_buf])

    tape.backward(loss=loss_buf)

    loss = float(loss_buf.numpy()[0]) / float(N)
    grad = np.array([
        float(params["alpha_min"].grad.numpy()[0]) / float(N),
        float(params["sharpness"].grad.numpy()[0]) / float(N),
        float(params["threshold"].grad.numpy()[0]) / float(N),
    ])
    return loss, grad


# ---------------------------------------------------------------------------
# Test: bridge gradient vs direct gradient
# ---------------------------------------------------------------------------

def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    sys_dict = load_msh_system(mesh_path)
    print(f"Mesh: N={sys_dict['N']}  E={sys_dict['E']}")

    # Set up knife (same as Day 3)
    knife_pos = np.array([[0.15, 0.30, 0.15]], dtype=np.float32)
    sys_dict["knife_pos"] = knife_pos
    knife_radius = 0.04
    n_cycles = 3

    # Generate target from θ_true (same setup as Day 3)
    theta_true = np.array([3.0, 20.0, 0.5])
    params_true = make_params(
        alpha_min=3.0, sharpness=20.0, threshold=0.5,
        knife_radius=knife_radius, requires_grad=False,
    )
    alpha_target_wp, _, _ = run_ssa_full_with_state(
        params_true, sys_dict, n_cycles=n_cycles, state_in=None,
    )
    alpha_target_np = alpha_target_wp.numpy().copy()
    print(f"Target cavity ratio: {float((alpha_target_np<0.5).mean()):.4f}")

    # θ_init for evaluation point (same as Day 3)
    theta_init = np.array([5.0, 15.0, 0.6], dtype=np.float32)
    print(f"\nθ_init = {theta_init}")

    # ====== Path A: direct Warp gradient (Day 3 ground truth) ======
    print("\n[Path A] Direct Warp Tape backward (Day 3 ground truth) …")
    loss_A, grad_A = direct_warp_gradient(
        theta_init, sys_dict, alpha_target_np,
        knife_radius=knife_radius, n_cycles=n_cycles,
    )
    print(f"  loss = {loss_A:.6e}")
    print(f"  grad = ({grad_A[0]:+.6e}, {grad_A[1]:+.6e}, {grad_A[2]:+.6e})")

    # ====== Path B: PyTorch autograd via Warp bridge ======
    print("\n[Path B] PyTorch autograd via WarpDamageForward bridge …")

    device = "cuda:0"
    N = sys_dict["N"]

    theta_torch = torch.tensor(theta_init, dtype=torch.float32, device=device,
                                 requires_grad=True)
    alpha_target_torch = torch.tensor(alpha_target_np, dtype=torch.float32,
                                        device=device)

    # Forward through bridge
    alpha_sim_torch = WarpDamageForward.apply(
        theta_torch, sys_dict, knife_radius, n_cycles,
    )

    # Torch-side loss (MUST match Warp's mse_loss_kernel: Σ diff², no 0.5)
    diff = alpha_sim_torch - alpha_target_torch
    loss_torch = (diff ** 2).sum() / float(N)
    print(f"  loss (torch) = {loss_torch.item():.6e}")

    # Backward through bridge
    loss_torch.backward()
    grad_B = theta_torch.grad.cpu().numpy()
    print(f"  grad = ({grad_B[0]:+.6e}, {grad_B[1]:+.6e}, {grad_B[2]:+.6e})")

    # ====== Parity check ======
    print("\n" + "=" * 60)
    print("Day 6a parity check: PyTorch bridge ↔ direct Warp")
    print("=" * 60)

    abs_diff = np.abs(grad_A - grad_B)
    rel_err = abs_diff / np.maximum(np.abs(grad_A), 1e-12)
    sign_match = np.sign(grad_A) == np.sign(grad_B)
    for d, name in enumerate(["α_min", "sharpness", "τ"]):
        ag = grad_A[d]; bg = grad_B[d]
        rel = rel_err[d] * 100
        sgn = "✓" if sign_match[d] else "✗"
        print(f"  {name:>10s}: direct={ag:+.6e}  bridge={bg:+.6e}  "
              f"rel_err={rel:.4f}%  sign {sgn}")

    nonzero = (np.abs(grad_B) > 1e-12).all()
    all_match = bool(sign_match.all() and (rel_err < 0.01).all())  # < 1%
    print()
    if nonzero and all_match:
        print(f"✓ Day 6a PASS — Warp ↔ PyTorch autograd bridge works (rel_err < 1%)")
        print(f"  PyTorch can drive the Warp differentiable forward")
        print(f"  → Day 6b: wrap full ShapeMatchingSolver.step() the same way")
    elif nonzero and sign_match.all():
        print(f"⚠ Day 6a partial: signs match, magnitudes close but >1% diff")
        print(f"  Possibly a normalization or detach inconsistency — investigate")
    elif not nonzero:
        print(f"✗ Day 6a FAIL — bridge produces zero gradient (autograd chain broken)")
    else:
        print(f"✗ Day 6a FAIL — sign mismatch (bridge incorrect)")


if __name__ == "__main__":
    main()
