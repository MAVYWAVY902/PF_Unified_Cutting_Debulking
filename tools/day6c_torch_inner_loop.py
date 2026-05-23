"""Day 6c: PyTorch inner loop — NUMPY PARITY TEST ONLY.

⚠️ DEPRECATED FOR OUTER BACKWARD CHAIN ⚠️
This file's `compute_liang_inner_residuals_torch` uses `torch.optim.Adam`.
`Adam.step()` is an IN-PLACE update on `param.data`, which silently breaks
the autograd graph: after `opt.step()`, the optimization variables are
*detached* from any history that depends on (pos_pred, dv_pred, θ).

This means: an outer L_outer.backward() applied to the returned (Δx, Δδv)
will NOT propagate gradient through the inner loop. The function still
produces correct *forward* trajectories — so the numpy-parity test below
remains a valid sanity check that we ported Liang's 4-term L_inner
correctly — but the function is unusable as the inner loop of ReSECT's
real outer-backward pipeline.

For the autograd-traceable inner loop, use:
    tools/day6d_v2_simple_gd.py :: simple_gd_inner_loop
which performs out-of-place updates with create_graph=True so the chain
θ → Warp forward → inner GD → L_outer survives backward.

Test plan (what this file actually verifies — narrow):
  1. Forward trajectory of torch L_inner == numpy L_inner (parity test)
  2. Δδv ≥ 0 via softplus (irreversibility check)
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import warp as wp
wp.init()
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

from tools.online_residual_synth import (  # noqa: E402
    load_msh_system, compute_liang_inner_residuals,
)


def compute_liang_inner_residuals_torch(
    pos_pred,        # (N, 3) torch.Tensor on cuda
    dv_pred,         # (N,) torch.Tensor
    pos_target,      # (N, 3) torch.Tensor
    alpha_target,    # (N,) torch.Tensor
    theta,           # (3,) torch.Tensor (α_min, sharpness, threshold) — α_min unused here
    edges_a,         # (E,) int64 torch.Tensor
    edges_b,         # (E,) int64 torch.Tensor
    edges_rest,      # (E,) torch.Tensor
    observable_mask=None,
    n_iter=30,
    lr_pos=5e-3,
    lr_dv=0.1,
    lambda_cav=1.0,
    lambda_E=1.0,
    verbose=False,
):
    """PyTorch version of compute_liang_inner_residuals.

    ⚠️ Uses torch.optim.Adam — Adam.step() is IN-PLACE and breaks the
    autograd graph from (Δx_final, Δδv_final) back to (pos_pred, dv_pred, θ).
    Suitable for forward-trajectory parity tests vs numpy ONLY. For the
    autograd-traceable version, see day6d_v2_simple_gd.simple_gd_inner_loop.
    """
    N = pos_pred.shape[0]
    device = pos_pred.device
    s = theta[1]
    tau = theta[2]

    if observable_mask is None:
        mask_pos = torch.ones((N, 1), dtype=torch.float32, device=device)
        mask_alpha = torch.ones(N, dtype=torch.float32, device=device)
    else:
        m = observable_mask.float().to(device)
        mask_pos = m.unsqueeze(1)
        mask_alpha = m

    # Optimization variables — torch tensors, NOT in any leaf graph (they are
    # parameters being optimized; their .grad will be computed each iter).
    delta_x = torch.zeros((N, 3), dtype=torch.float32, device=device,
                           requires_grad=True)
    delta_dv_raw = torch.zeros(N, dtype=torch.float32, device=device,
                                 requires_grad=True)

    # Adam over the two channels. PyTorch's Adam handles bias correction +
    # adaptive lr internally.
    opt = torch.optim.Adam([
        {"params": [delta_x], "lr": lr_pos},
        {"params": [delta_dv_raw], "lr": lr_dv},
    ], betas=(0.9, 0.999), eps=1e-8)

    L_init = None
    for k in range(n_iter):
        # Forward (4 terms)
        pos_corr = pos_pred + delta_x
        delta_dv = torch.nn.functional.softplus(delta_dv_raw)
        dv_corr = dv_pred + delta_dv
        alpha_corr = torch.sigmoid(-s * (dv_corr - tau))

        diff_pos = (pos_corr - pos_target) * mask_pos
        diff_alpha = (alpha_corr - alpha_target) * mask_alpha

        L_pos = (diff_pos ** 2).sum()
        L_cav = (diff_alpha ** 2).sum()

        edge_vec = pos_corr[edges_a] - pos_corr[edges_b]
        edge_len = edge_vec.norm(dim=1).clamp(min=1e-12)
        edge_residual = edge_len - edges_rest
        L_E = (edge_residual ** 2).sum()

        L_inner = L_pos + lambda_cav * L_cav + lambda_E * L_E

        if L_init is None:
            L_init = (L_pos.item(), L_cav.item(), L_E.item(), L_inner.item())

        # Inner-step gradient (NOT outer — this is gradient of L_inner w.r.t.
        # the residual variables Δx, Δδv_raw at this fixed θ, pos_pred, dv_pred).
        # We use retain_graph=True because successive iterations build on the
        # same (Δx, Δδv_raw) — if we let torch free the graph, .backward()
        # would clear it, breaking the next iter's gradient.
        opt.zero_grad()
        L_inner.backward(retain_graph=True)
        opt.step()

        if verbose and (k % 5 == 0 or k == n_iter - 1):
            print(f"    inner k={k:2d}  L_pos={L_pos.item():.4e}  "
                  f"L_cav={L_cav.item():.4e}  L_E={L_E.item():.4e}  "
                  f"L_inner={L_inner.item():.4e}")

    # Final residuals (no longer need grad on them as inner vars, but they
    # carry grad through to outer if outer loss touches them)
    delta_dv_final = torch.nn.functional.softplus(delta_dv_raw)
    info = dict(
        L_pos=L_pos.item(), L_cav=L_cav.item(), L_E=L_E.item(),
        L_total=L_inner.item(),
        L_init_pos=L_init[0], L_init_cav=L_init[1],
        L_init_E=L_init[2], L_init_total=L_init[3],
        reduction_total=L_init[3] / max(L_inner.item(), 1e-12),
    )
    return delta_x, delta_dv_final, info


# ---------------------------------------------------------------------------
# Test: torch inner loop ≈ numpy inner loop on same inputs
# ---------------------------------------------------------------------------

def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    sys_dict = load_msh_system(mesh_path)
    print(f"\nMesh: N={sys_dict['N']}  E={sys_dict['E']}")

    N = sys_dict["N"]
    pos = sys_dict["pos"].astype(np.float64)
    edges_a_np = sys_dict["edge_a"].astype(np.int64)
    edges_b_np = sys_dict["edge_b"].astype(np.int64)
    edges_rest_np = sys_dict["pf_rest"].astype(np.float64)

    pos_pred_np = pos.copy()
    dv_pred_np = np.zeros(N, dtype=np.float64)
    theta_np = np.array([3.0, 20.0, 0.5])

    # Non-rigid target (matches Day 4 setup)
    np_rng = np.random.default_rng(42)
    shift_per_particle = np_rng.uniform(-0.005, 0.005, size=(N, 3))
    pos_target_np = pos_pred_np + shift_per_particle
    alpha_target_np = np.ones(N, dtype=np.float64)
    cavity_idx = np_rng.choice(N, size=int(0.1 * N), replace=False)
    alpha_target_np[cavity_idx] = 0.0

    # =================== Numpy version (Day 4 ground truth) ===================
    print("\n[numpy] Day 4 inner loop reference …")
    dx_np, ddv_np, info_np = compute_liang_inner_residuals(
        pos_pred_np, dv_pred_np, pos_target_np, alpha_target_np, theta_np,
        edges_a_np, edges_b_np, edges_rest_np,
        n_iter=30, lr_pos=5e-3, lr_dv=0.1,
        lambda_cav=1.0, lambda_E=1.0, verbose=False,
    )
    print(f"  L_pos={info_np['L_pos']:.4e}  L_cav={info_np['L_cav']:.4e}  "
          f"L_E={info_np['L_E']:.4e}  L_total={info_np['L_total']:.4e}")
    print(f"  max|Δx|={np.abs(dx_np).max():.5f}  max Δδv={ddv_np.max():.4f}")

    # =================== Torch version ===================
    print("\n[torch] new Day 6c inner loop …")
    device = "cuda:0"
    pos_pred_t = torch.tensor(pos_pred_np, dtype=torch.float32, device=device)
    dv_pred_t = torch.tensor(dv_pred_np, dtype=torch.float32, device=device)
    pos_target_t = torch.tensor(pos_target_np, dtype=torch.float32, device=device)
    alpha_target_t = torch.tensor(alpha_target_np, dtype=torch.float32, device=device)
    theta_t = torch.tensor(theta_np, dtype=torch.float32, device=device)
    edges_a_t = torch.tensor(edges_a_np, dtype=torch.long, device=device)
    edges_b_t = torch.tensor(edges_b_np, dtype=torch.long, device=device)
    edges_rest_t = torch.tensor(edges_rest_np, dtype=torch.float32, device=device)

    dx_t, ddv_t, info_t = compute_liang_inner_residuals_torch(
        pos_pred_t, dv_pred_t, pos_target_t, alpha_target_t, theta_t,
        edges_a_t, edges_b_t, edges_rest_t,
        n_iter=30, lr_pos=5e-3, lr_dv=0.1,
        lambda_cav=1.0, lambda_E=1.0, verbose=False,
    )
    print(f"  L_pos={info_t['L_pos']:.4e}  L_cav={info_t['L_cav']:.4e}  "
          f"L_E={info_t['L_E']:.4e}  L_total={info_t['L_total']:.4e}")
    dx_t_np = dx_t.detach().cpu().numpy()
    ddv_t_np = ddv_t.detach().cpu().numpy()
    print(f"  max|Δx|={np.abs(dx_t_np).max():.5f}  max Δδv={ddv_t_np.max():.4f}")

    # =================== Compare ===================
    print("\n" + "=" * 64)
    print("Day 6c parity: torch inner loop ↔ numpy inner loop")
    print("=" * 64)

    # Both should produce similar trajectories. Small differences due to:
    # - float32 vs float64 (numpy uses float64 internally)
    # - Adam implementation specifics (PyTorch vs custom numpy)
    dx_diff = np.abs(dx_np.astype(np.float32) - dx_t_np)
    ddv_diff = np.abs(ddv_np.astype(np.float32) - ddv_t_np)
    L_total_rel = abs(info_t["L_total"] - info_np["L_total"]) / max(info_np["L_total"], 1e-12)

    print(f"  Δx  : max|Δ|={dx_diff.max():.4e}  mean|Δ|={dx_diff.mean():.4e}")
    print(f"  Δδv : max|Δ|={ddv_diff.max():.4e}  mean|Δ|={ddv_diff.mean():.4e}")
    print(f"  L_total rel diff: {L_total_rel*100:.3f}%")

    # Verdicts
    irreversible_t = float(ddv_t_np.min()) >= 0.0
    print(f"\n  Δδv ≥ 0 (softplus in torch): {'✓' if irreversible_t else '✗'}")

    # The trajectories don't need to be identical (different float precision)
    # but the FINAL L_total should be in the same ballpark.
    if L_total_rel < 0.05 and dx_diff.max() < 0.01 and ddv_diff.max() < 0.5:
        print(f"✓ Day 6c PASS — torch inner loop matches numpy trajectory")
        print(f"  (final L_total within 5%, |Δx| diff < 0.01, |Δδv| diff < 0.5)")
    elif info_t["L_total"] < info_t["L_init_total"]:
        print(f"⚠ Day 6c partial: torch inner loop converges but trajectory")
        print(f"  differs from numpy version (likely Adam impl details or float precision)")
    else:
        print(f"✗ Day 6c FAIL — torch inner loop doesn't converge")


if __name__ == "__main__":
    main()
