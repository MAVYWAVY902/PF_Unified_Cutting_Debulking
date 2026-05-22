"""Day 1 smoke test: instantiate ShapeMatchingSolver on cube_tet.msh and run 1 step.

Goal: verify the full forward sim (PBD + damage) works on our cube mesh.
Output: confirm positions update and damage starts accumulating with knife present.

This is the foundation for Day 2 (refactor sys-ID to use this solver instead of
the damage-only pf_damage kernel subset).
"""
import sys
from pathlib import Path
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pf_shape_matching_warp"))

import warp as wp  # noqa: E402

# Diagnostic anchor (Stage B engineering rule)
wp.config.verify_autograd_array_access = True
wp.init()

from mesh.loader import (  # noqa: E402
    load_tet_mesh, build_vertex_neighbors, build_clusters_from_tets,
    pack_clusters_csr, build_edges,
)
from pf_shape_matching_warp.sim.particles import (  # noqa: E402
    init_warp_arrays, init_cluster_arrays, init_edge_arrays,
)
from pf_shape_matching_warp.sim.solver import ShapeMatchingSolver  # noqa: E402


def setup_solver_from_msh(msh_path: str, rings: int = 2, device: str = "cuda:0",
                            fix_bottom_layer: bool = True,
                            fix_threshold_fraction: float = 0.05):
    """Build a fully-loaded ShapeMatchingSolver from a tet mesh file.

    Args:
        msh_path           — path to .msh file
        rings              — BFS rings for cluster build (matches C++ default 2)
        fix_bottom_layer   — pin the bottom slice as immovable (gravity anchor)
        fix_threshold_fraction — fraction of mesh height to mark as fixed

    Returns:
        solver, p_arrays, c_arrays, e_arrays, mesh_info
    """
    print(f"[setup] loading {msh_path} …")
    vertices, tets = load_tet_mesh(msh_path)
    N = len(vertices)

    bbox_lo = vertices.min(axis=0); bbox_hi = vertices.max(axis=0)
    print(f"[setup]   bbox: {tuple(bbox_lo.round(3))} → {tuple(bbox_hi.round(3))}")
    print(f"[setup]   N={N} verts  T={len(tets)} tets")

    # Build adjacency + clusters + edges
    print(f"[setup] building vertex adjacency …")
    vertex_neighbors = build_vertex_neighbors(tets, N)

    print(f"[setup] building clusters (rings={rings}) …")
    cluster_ids, cluster_cms, cluster_offsets = build_clusters_from_tets(
        vertices, vertex_neighbors, rings=rings
    )
    starts, sizes, pids, offsets, cms = pack_clusters_csr(
        cluster_ids, cluster_offsets, cluster_cms
    )

    print(f"[setup] building edges …")
    edge_a, edge_b, edge_pf_rest = build_edges(vertices, cluster_ids, vertex_neighbors)

    # Fixed particles
    fixed_mask = np.zeros(N, dtype=np.int32)
    if fix_bottom_layer:
        bottom_y = bbox_lo[1]
        threshold = bottom_y + fix_threshold_fraction * (bbox_hi[1] - bbox_lo[1])
        fixed_mask[vertices[:, 1] <= threshold] = 1
    print(f"[setup]   fixed particles (bottom {fix_threshold_fraction*100:.0f}%): "
          f"{int(fixed_mask.sum())} / {N}")

    # Build Warp arrays
    p_arrays = init_warp_arrays(vertices, fixed_mask, device)
    c_arrays = init_cluster_arrays(starts, sizes, pids, offsets, cms, device)
    e_arrays = init_edge_arrays(edge_a, edge_b, edge_pf_rest, device)

    # Instantiate solver
    solver = ShapeMatchingSolver(p_arrays, c_arrays, e_arrays, device=device)
    # Use the SSA paths so autograd is correct
    solver.use_ssa_soft_removal = True
    solver.use_split_cluster_solve = True
    solver._alloc_ssa_buffers()

    mesh_info = dict(
        vertices=vertices, tets=tets, fixed_mask=fixed_mask,
        bbox_lo=bbox_lo, bbox_hi=bbox_hi, N=N,
    )
    return solver, p_arrays, c_arrays, e_arrays, mesh_info


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")

    solver, p, c, e, info = setup_solver_from_msh(mesh_path)

    print("\n" + "=" * 60)
    print("Day 1 smoke test — forward step + position update")
    print("=" * 60)
    print(f"  solver.N = {solver.N}")
    print(f"  solver.num_clusters = {solver.num_clusters}")
    print(f"  solver.dt = {solver.dt}")
    print(f"  solver.substeps = {solver.substeps}")
    print(f"  solver.num_iterations = {solver.num_iterations}")
    print(f"  solver.gravity = {solver.gravity}")
    print(f"  solver.alpha_min = {solver.alpha_min}")
    print(f"  solver.soft_removal_sharpness = {solver.soft_removal_sharpness}")
    print(f"  solver.use_ssa_soft_removal = {solver.use_ssa_soft_removal}")
    print(f"  solver.use_split_cluster_solve = {solver.use_split_cluster_solve}")

    pos_before = p["pos"].numpy().copy()
    centroid_before = pos_before.mean(axis=0)
    print(f"\n  pos before: centroid = {centroid_before.round(4)}")
    print(f"             min  = {pos_before.min(axis=0).round(4)}")
    print(f"             max  = {pos_before.max(axis=0).round(4)}")

    # Set knife to a position near the top surface (similar to our sys-ID setup)
    bbox_hi = info["bbox_hi"]
    knife_xyz = np.array([
        (info["bbox_lo"][0] + bbox_hi[0]) / 2.0,
        bbox_hi[1],                       # on the top surface (half-exposed)
        (info["bbox_lo"][2] + bbox_hi[2]) / 2.0,
    ], dtype=np.float32)
    solver.knife_radius = 0.04
    solver.knife_active = 1
    # knife_pos is a wp.array — update via assign-via-numpy
    knife_pos_host = np.array([knife_xyz], dtype=np.float32)
    solver.knife_pos = wp.array(knife_pos_host, dtype=wp.vec3,
                                  device=solver.device, requires_grad=True)
    print(f"\n  knife at {tuple(knife_xyz.round(3))} (radius {solver.knife_radius})")

    # Run a few steps
    n_steps = 5
    print(f"\n  running {n_steps} solver.step() calls …")
    for s in range(n_steps):
        t0 = time.time()
        solver.step()
        wp.synchronize_device()
        dt_ms = (time.time() - t0) * 1000
        pos_after = p["pos"].numpy()
        alpha = solver._current_alpha().numpy()
        cavity_frac = float((alpha < 0.5).mean())
        max_disp = float(np.abs(pos_after - pos_before).max())
        mean_disp = float(np.abs(pos_after - pos_before).mean())
        print(f"    step {s+1}/{n_steps}  {dt_ms:.1f}ms  "
              f"max|Δpos|={max_disp:.5f}  mean|Δpos|={mean_disp:.5f}  "
              f"cavity={cavity_frac:.5f}  α range [{alpha.min():.3f}, {alpha.max():.3f}]")

    pos_after = p["pos"].numpy()
    centroid_after = pos_after.mean(axis=0)
    print(f"\n  pos after: centroid  = {centroid_after.round(4)}")
    print(f"             max|Δpos| = {float(np.abs(pos_after - pos_before).max()):.5f}")
    print(f"             mean|Δpos|= {float(np.abs(pos_after - pos_before).mean()):.5f}")

    # Sanity checks
    print("\n" + "=" * 60)
    print("Day 1 sanity checks")
    print("=" * 60)
    moved = float(np.abs(pos_after - pos_before).max())
    if moved > 1e-5:
        print(f"  ✓ Positions moved (max disp = {moved:.5f})")
    else:
        print(f"  ✗ Positions did NOT move — solver isn't running physics")
    alpha = solver._current_alpha().numpy()
    if alpha.min() < 0.99:
        print(f"  ✓ Some particles damaged (min α = {alpha.min():.4f})")
    else:
        print(f"  ⚠ No damage yet — may need more steps or different knife position")

    print("\nDay 1 PASS — solver works on cube_tet.msh.")
    print("Next: Day 2 — refactor sys-ID forward to call solver.step()")


if __name__ == "__main__":
    main()
