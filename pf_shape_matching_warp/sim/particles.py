"""Initialize Warp arrays for simulation."""
import numpy as np
import warp as wp


def init_warp_arrays(vertices: np.ndarray, fixed_mask: np.ndarray, device="cuda:0"):
    """Initialize particle Warp arrays from numpy.

    Args:
        vertices: (N, 3) float32 positions
        fixed_mask: (N,) int32, 1 = fixed, 0 = free
        device: Warp device string

    Returns:
        dict of wp.arrays
    """
    N = len(vertices)

    arrays = {
        "pos": wp.array(vertices, dtype=wp.vec3, device=device, requires_grad=True),
        "vel": wp.zeros(N, dtype=wp.vec3, device=device, requires_grad=True),
        "pred_pos": wp.array(vertices, dtype=wp.vec3, device=device, requires_grad=True),
        "old_pos": wp.array(vertices, dtype=wp.vec3, device=device, requires_grad=True),
        "damage": wp.zeros(N, dtype=wp.float32, device=device, requires_grad=True),
        "damage_int": wp.zeros(N, dtype=wp.int32, device=device),      # scratch for atomic_max
        "alpha": wp.ones(N, dtype=wp.float32, device=device, requires_grad=True),
        "fixed": wp.array(fixed_mask, dtype=wp.int32, device=device),  # int, no grad
        "deltas": wp.zeros(N, dtype=wp.vec3, device=device, requires_grad=True),
        "delta_counts": wp.zeros(N, dtype=wp.int32, device=device),    # int, no grad
        # Separate collision-delta buffer: SSA gradient-safe pattern (XPBD reference).
        # Zero-filled per collision call, then applied via apply_collision_delta_kernel.
        "collision_delta": wp.zeros(N, dtype=wp.vec3, device=device, requires_grad=True),
    }

    print(f"Warp arrays initialized: {N} particles on {device}")
    return arrays


def init_cluster_arrays(starts, sizes, pids, offsets, cms, device="cuda:0"):
    """Initialize cluster CSR Warp arrays.

    Args:
        starts: (C,) int32
        sizes: (C,) int32
        pids: (total,) int32
        offsets: (total, 3) float32
        cms: (C, 3) float32

    Returns:
        dict of wp.arrays
    """
    cluster_arrays = {
        "starts": wp.array(starts, dtype=wp.int32, device=device),
        "sizes": wp.array(sizes, dtype=wp.int32, device=device),
        "particle_ids": wp.array(pids, dtype=wp.int32, device=device),
        "rest_offsets": wp.array(offsets, dtype=wp.vec3, device=device),
        "rest_cm": wp.array(cms, dtype=wp.vec3, device=device),
        "num_clusters": len(starts),
        "total_entries": len(pids),
    }

    print(f"Cluster arrays: {len(starts)} clusters, {len(pids)} total entries")
    return cluster_arrays


def init_edge_arrays(edge_a, edge_b, edge_pf_rest, device="cuda:0"):
    """Initialize edge Warp arrays for P(F) damage.

    Returns:
        dict of wp.arrays
    """
    E = len(edge_a)
    edge_arrays = {
        "a": wp.array(edge_a, dtype=wp.int32, device=device),           # int, no grad
        "b": wp.array(edge_b, dtype=wp.int32, device=device),           # int, no grad
        "pf_rest": wp.array(edge_pf_rest, dtype=wp.float32, device=device, requires_grad=True),
        "damage": wp.zeros(E, dtype=wp.float32, device=device, requires_grad=True),
        # SSA delta buffer (Warp 1.12 gradient-safe pattern): zero-filled before each
        # update_edge_damage_kernel_ssa launch, applied via apply_edge_damage_delta_kernel.
        "damage_delta": wp.zeros(E, dtype=wp.float32, device=device, requires_grad=True),
        "num_edges": E,
    }

    print(f"Edge arrays: {E} edges")
    return edge_arrays
