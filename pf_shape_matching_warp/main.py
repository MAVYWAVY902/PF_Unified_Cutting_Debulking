"""Phase 1: Mesh loading + Warp array initialization.
Validates against C++ output: particle count, cluster count, edge count.
"""
import sys
import numpy as np
import warp as wp

from mesh.loader import (
    load_tet_mesh,
    build_vertex_neighbors,
    build_clusters_from_tets,
    pack_clusters_csr,
    build_edges,
)
from sim.particles import init_warp_arrays, init_cluster_arrays, init_edge_arrays


def main():
    # Default mesh path
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "/home/yunxin/xpbd-tissue-sim/resource/cube/cube32.msh"

    # Initialize Warp
    wp.init()

    # Step 1: Load tet mesh
    print("=" * 50)
    print("Phase 1: Mesh Loading + Warp Array Init")
    print("=" * 50)

    vertices, tets = load_tet_mesh(mesh_path)
    N = len(vertices)

    # Offset to place above floor (same as C++)
    min_y = vertices[:, 1].min()
    y_offset = -min_y + 0.001
    vertices[:, 1] += y_offset

    # Step 2: Build vertex neighbors
    print("\nBuilding vertex neighbors...")
    vertex_neighbors = build_vertex_neighbors(tets, N)
    avg_nbrs = np.mean([len(n) for n in vertex_neighbors])
    print(f"  Average neighbors per vertex: {avg_nbrs:.1f}")

    # Step 3: Build clusters (2-ring BFS, same as C++)
    print("\nBuilding clusters (rings=2)...")
    cluster_ids, cluster_cms, cluster_offsets = build_clusters_from_tets(
        vertices, vertex_neighbors, rings=2
    )

    # Step 4: Pack into CSR
    print("\nPacking CSR...")
    starts, sizes, pids, offsets, cms = pack_clusters_csr(
        cluster_ids, cluster_offsets, cluster_cms
    )

    # Step 5: Build edges for P(F)
    print("\nBuilding edges...")
    edge_a, edge_b, edge_pf_rest = build_edges(vertices, cluster_ids, vertex_neighbors)

    # Step 6: Fixed particles (bottom layer, y < 0.05)
    fixed_mask = np.zeros(N, dtype=np.int32)
    fixed_count = 0
    for i in range(N):
        if vertices[i, 1] < 0.05:
            fixed_mask[i] = 1
            fixed_count += 1
    print(f"\nFixed particles (y < 0.05): {fixed_count}")

    # Step 7: Initialize Warp arrays
    print("\nInitializing Warp arrays...")
    device = "cuda:0"
    p_arrays = init_warp_arrays(vertices, fixed_mask, device)
    c_arrays = init_cluster_arrays(starts, sizes, pids, offsets, cms, device)
    e_arrays = init_edge_arrays(edge_a, edge_b, edge_pf_rest, device)

    # Step 8: Validation against C++ output
    print("\n" + "=" * 50)
    print("Validation (compare with C++ output)")
    print("=" * 50)

    cpp_expected = {
        "particles": 20945,
        "clusters": 20945,  # 1 cluster per vertex for 2-ring
        "edges_approx": 322809,  # approximate
        "fixed": 1766,
    }

    print(f"  Particles:  {N:>8}  (C++ expected: {cpp_expected['particles']})")
    print(f"  Clusters:   {c_arrays['num_clusters']:>8}  (C++ expected: {cpp_expected['clusters']})")
    print(f"  Edges:      {e_arrays['num_edges']:>8}  (C++ expected: ~{cpp_expected['edges_approx']})")
    print(f"  Fixed:      {fixed_count:>8}  (C++ expected: {cpp_expected['fixed']})")
    print(f"  CSR total:  {c_arrays['total_entries']:>8}")

    # Verify Warp arrays are on GPU
    print(f"\n  pos device: {p_arrays['pos'].device}")
    print(f"  pos shape:  {p_arrays['pos'].shape}")

    # Quick sanity: read back centroid
    pos_np = p_arrays["pos"].numpy()
    cm = pos_np.mean(axis=0)
    print(f"  Centroid:   ({cm[0]:.3f}, {cm[1]:.3f}, {cm[2]:.3f})")

    # Check match
    all_ok = True
    if N != cpp_expected["particles"]:
        print(f"\n  WARNING: particle count mismatch!")
        all_ok = False
    if c_arrays["num_clusters"] != cpp_expected["clusters"]:
        print(f"\n  WARNING: cluster count mismatch!")
        all_ok = False
    if fixed_count != cpp_expected["fixed"]:
        print(f"\n  WARNING: fixed count mismatch!")
        all_ok = False

    if all_ok:
        print("\n  ALL CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED - investigate above")

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
