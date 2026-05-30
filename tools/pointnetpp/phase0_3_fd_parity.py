"""Phase 0.3 — FD parity test for hierarchical reconstruction.

For each of the three latent fields (z_L0, z_L1, z_L2), perturb one element by
±eps, recompute the reconstructed [N] output, and compare the central
difference to the analytical autograd gradient. Pass if max rel_err < 0.5%.

The downstream loss is differentiable_PBD(reconstruct(z0, z1, z2)). If
reconstruct() has correct autograd, the full chain rule is automatic — this
test isolates the new layer.
"""

import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.pointnetpp.hierarchy import build_hierarchy, reconstruct  # noqa: E402


def fd_parity_one_level(z_L0, z_L1, z_L2, H, level_name, eps=1e-3, num_probe=10):
    """Probe `num_probe` random elements of one latent level, return max rel_err."""
    # Use a fixed downstream loss for grad extraction:
    #   L(z0, z1, z2) = sum( reconstruct ** 2 )
    # Then dL/dz_l = autograd → compare to FD on individual elements.

    target_z = {"z0": z_L0, "z1": z_L1, "z2": z_L2}[level_name]
    M = target_z.shape[0]
    probes = torch.randperm(M)[:num_probe].tolist()

    # Analytical
    z0 = z_L0.detach().clone().requires_grad_(True)
    z1 = z_L1.detach().clone().requires_grad_(True)
    z2 = z_L2.detach().clone().requires_grad_(True)
    out = reconstruct(z0, z1, z2, H)
    L = (out ** 2).sum()
    L.backward()
    grad_target = {"z0": z0.grad, "z1": z1.grad, "z2": z2.grad}[level_name]

    rel_errs = []
    for probe_idx in probes:
        # FD: perturb single element ±eps
        z_lvl = {"z0": z_L0, "z1": z_L1, "z2": z_L2}[level_name]
        zp = z_lvl.detach().clone()
        zm = z_lvl.detach().clone()
        zp[probe_idx] += eps
        zm[probe_idx] -= eps

        kw_p = {"z0": z_L0, "z1": z_L1, "z2": z_L2}.copy()
        kw_m = {"z0": z_L0, "z1": z_L1, "z2": z_L2}.copy()
        kw_p[level_name] = zp
        kw_m[level_name] = zm

        out_p = reconstruct(kw_p["z0"], kw_p["z1"], kw_p["z2"], H)
        out_m = reconstruct(kw_m["z0"], kw_m["z1"], kw_m["z2"], H)
        L_p = (out_p ** 2).sum().item()
        L_m = (out_m ** 2).sum().item()
        fd = (L_p - L_m) / (2 * eps)

        ana = grad_target[probe_idx].item()
        if abs(ana) < 1e-10 and abs(fd) < 1e-10:
            rel_err = 0.0
        else:
            denom = max(abs(ana), abs(fd), 1e-10)
            rel_err = abs(ana - fd) / denom
        rel_errs.append((probe_idx, ana, fd, rel_err))

    max_rel = max(r[3] for r in rel_errs)
    return rel_errs, max_rel


def main():
    print("=== Phase 0.3 FD parity test ===")

    # Synthetic small problem for fast FD; float64 to eliminate FD round-off floor
    torch.manual_seed(0)
    N = 256
    M1, M2 = 32, 8
    x_rest = torch.rand(N, 3, device="cuda", dtype=torch.float64)
    H = build_hierarchy(x_rest, M1=M1, M2=M2)
    print(f"  N={N}, M1={M1}, M2={M2}")
    print(f"  idx_L1 shape: {tuple(H.idx_L1.shape)},   w_L1 sum check: "
          f"{H.w_L1.sum(dim=1).mean().item():.6f} (should be 1)")
    print(f"  idx_L2 shape: {tuple(H.idx_L2.shape)},   w_L2 sum check: "
          f"{H.w_L2.sum(dim=1).mean().item():.6f}")

    z_L0 = torch.randn(N, device="cuda", dtype=torch.float64)
    z_L1 = torch.randn(M1, device="cuda", dtype=torch.float64)
    z_L2 = torch.randn(M2, device="cuda", dtype=torch.float64)

    all_pass = True
    for lvl in ["z0", "z1", "z2"]:
        rel_errs, max_rel = fd_parity_one_level(z_L0, z_L1, z_L2, H, lvl, eps=1e-3, num_probe=10)
        marker = "✓" if max_rel < 0.005 else "✗"
        print(f"\n  [{marker}] {lvl} FD parity over 10 probes: max rel_err = {max_rel*100:.4f}%")
        for idx, ana, fd, rel in rel_errs[:3]:
            print(f"      probe {idx}: ana={ana:+.6f}  fd={fd:+.6f}  rel={rel*100:.4f}%")
        if max_rel >= 0.005:
            all_pass = False

    print(f"\n=== Phase 0.3 Gate ===")
    if all_pass:
        print("  ✓ PASS — all 3 levels' FD parity < 0.5%")
    else:
        print("  ✗ FAIL")
    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
