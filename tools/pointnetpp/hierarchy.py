"""Hierarchical parameter representation — FPS centers + Feature Propagation interp.

Inspired by PointNet++ (Qi 2017) Multi-Scale Grouping + Feature Propagation,
but used as a pure-geometric reparameterization (no NN, no training). Adam
optimizes coarse-to-fine latents (z_L0, z_L1, z_L2) jointly; per-particle
stiff_pp / tau_pp is reconstructed by distance-weighted interpolation from
the latents at each forward pass.

Pipeline:
    1. build_hierarchy(x_rest)   one-shot init at solver setup
    2. reconstruct(z0, z1, z2)   each Adam forward step (autograd-friendly)

The reconstruction is pure-torch tensor ops, so autograd handles the backward
gradient flow from grad_stiff_pp → (grad_z0, grad_z1, grad_z2) automatically.
"""

from dataclasses import dataclass
import torch

from .fps import fps


@dataclass
class Hierarchy:
    """Precomputed FPS centers + inverse-distance interpolation weights."""
    N: int                       # number of particles
    idx_L1: torch.Tensor         # [N, 3] long — 3 nearest L1 center indices
    w_L1: torch.Tensor           # [N, 3] float — normalized inv-distance weights
    idx_L2: torch.Tensor         # [N, 3] long
    w_L2: torch.Tensor           # [N, 3] float
    centers_L1: torch.Tensor     # [M1, 3] float — for viz/debug
    centers_L2: torch.Tensor     # [M2, 3] float
    M1: int
    M2: int


def _compute_fp_weights(
    x_rest: torch.Tensor,
    centers: torch.Tensor,
    k: int = 3,
    eps: float = 1e-8,
):
    """Distance-weighted barycentric weights (PointNet++ FP-layer style).

    For each particle, find the k nearest centers and compute weights as
    1/d normalized to sum to 1.

    Args:
        x_rest:  [N, 3]
        centers: [M, 3]
        k:       number of nearest centers per particle (default 3)
    Returns:
        idx:     [N, k] long indices into centers
        weights: [N, k] float, rows sum to 1
    """
    dists = torch.cdist(x_rest, centers)               # [N, M]
    knn_dists, idx = dists.topk(k, dim=1, largest=False)   # [N, k] each
    inv = 1.0 / (knn_dists + eps)
    weights = inv / inv.sum(dim=1, keepdim=True)
    return idx, weights


def build_hierarchy(
    x_rest: torch.Tensor,
    M1: int = 512,
    M2: int = 64,
    k_neighbors: int = 3,
) -> Hierarchy:
    """One-shot hierarchical center selection + FP weight precomputation.

    Args:
        x_rest:      [N, 3] rest particle positions (torch tensor on GPU).
        M1:          number of mid-scale centers (default 512).
        M2:          number of coarse-scale centers (default 64).
        k_neighbors: nearest-k centers per particle for FP interp (default 3).
    """
    N = x_rest.shape[0]
    assert M2 < M1 < N, f"need M2 < M1 < N, got M2={M2} M1={M1} N={N}"

    idx_L1_centers = fps(x_rest, num_samples=M1)
    idx_L2_centers = fps(x_rest, num_samples=M2)
    centers_L1 = x_rest[idx_L1_centers].contiguous()
    centers_L2 = x_rest[idx_L2_centers].contiguous()

    idx_L1, w_L1 = _compute_fp_weights(x_rest, centers_L1, k=k_neighbors)
    idx_L2, w_L2 = _compute_fp_weights(x_rest, centers_L2, k=k_neighbors)

    return Hierarchy(
        N=N,
        idx_L1=idx_L1, w_L1=w_L1,
        idx_L2=idx_L2, w_L2=w_L2,
        centers_L1=centers_L1, centers_L2=centers_L2,
        M1=M1, M2=M2,
    )


def reconstruct(
    z_L0: torch.Tensor,    # [N]
    z_L1: torch.Tensor,    # [M1]
    z_L2: torch.Tensor,    # [M2]
    H: Hierarchy,
) -> torch.Tensor:
    """Reconstruct per-particle pre-bound latent from hierarchy.

    Returns z_flat ∈ R^N. Caller applies sigmoid bounding:
        stiff_pp = LOW + RANGE * sigmoid(z_flat)
    so this function is range-agnostic.

    Autograd flows: ∂L/∂z_flat → ∂L/∂z_L0 directly, ∂L/∂z_L1/L2 scattered
    by the interpolation weights.
    """
    contrib_L1 = (z_L1[H.idx_L1] * H.w_L1).sum(dim=1)   # [N]
    contrib_L2 = (z_L2[H.idx_L2] * H.w_L2).sum(dim=1)   # [N]
    return z_L0 + contrib_L1 + contrib_L2
