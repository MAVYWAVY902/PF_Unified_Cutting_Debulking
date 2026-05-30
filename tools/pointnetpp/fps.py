"""Pure-PyTorch Farthest Point Sampling (FPS).

Reason for pure-torch implementation: torch_cluster has no prebuilt wheels for
torch 2.11+cu130 and source build fails. FPS runs once at init (16 + 64 centers
on ~1k particles), so optimized C++ is unnecessary — vectorized PyTorch suffices.

Algorithm (greedy, M iterations on N points):
    1. Pick start point (index 0 or random).
    2. Maintain min-distance-to-selected array d ∈ R^N (init = +inf).
    3. Each iteration k = 1..M-1:
       - d_i ← min(d_i, ||x_i - x_{selected[k-1]}||) for all i  (parallel)
       - selected[k] ← argmax_i d_i                              (parallel reduce)

Complexity: O(M·N), each iter is fully vectorized on GPU.
"""

import torch


def fps(points: torch.Tensor, num_samples: int, start_idx: int = 0) -> torch.Tensor:
    """Farthest Point Sampling.

    Args:
        points:     [N, 3] tensor of point positions.
        num_samples: M, number of centers to sample (M <= N).
        start_idx:  index of the first sampled point (default 0).

    Returns:
        indices: [M] long tensor of sampled indices into `points`.
    """
    N = points.shape[0]
    assert num_samples <= N, f"num_samples {num_samples} > N {N}"
    device = points.device

    indices = torch.empty(num_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float("inf"), device=device)

    indices[0] = start_idx
    farthest = start_idx
    for k in range(1, num_samples):
        centroid = points[farthest].unsqueeze(0)            # [1, 3]
        new_dist = torch.norm(points - centroid, dim=1)     # [N]
        distances = torch.minimum(distances, new_dist)      # [N], in-place would break
        farthest = int(distances.argmax().item())
        indices[k] = farthest
    return indices
