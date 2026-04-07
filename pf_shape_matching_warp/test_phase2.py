"""Phase 2 test: Core Shape Matching simulation.
Cube drops onto floor, deforms, settles. Verify correctness + performance.
"""
import sys
import time
import numpy as np
import warp as wp

from mesh.loader import load_tet_mesh, build_vertex_neighbors, build_clusters_from_tets, pack_clusters_csr
from sim.particles import init_warp_arrays, init_cluster_arrays
from sim.solver import ShapeMatchingSolver


def main():
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "/home/yunxin/xpbd-tissue-sim/resource/cube/cube32.msh"

    wp.init()
    device = "cuda:0"

    # Load mesh
    print("Loading mesh...")
    vertices, tets = load_tet_mesh(mesh_path)
    N = len(vertices)

    # Offset above floor
    min_y = vertices[:, 1].min()
    vertices[:, 1] += -min_y + 0.001

    # Build clusters
    print("Building clusters...")
    vertex_neighbors = build_vertex_neighbors(tets, N)
    cluster_ids, cluster_cms, cluster_offsets = build_clusters_from_tets(vertices, vertex_neighbors, rings=2)
    starts, sizes, pids, offsets, cms = pack_clusters_csr(cluster_ids, cluster_offsets, cluster_cms)

    # Fixed particles
    fixed_mask = np.zeros(N, dtype=np.int32)
    for i in range(N):
        if vertices[i, 1] < 0.05:
            fixed_mask[i] = 1

    # Init Warp arrays
    print("Initializing Warp arrays...")
    p_arrays = init_warp_arrays(vertices, fixed_mask, device)
    c_arrays = init_cluster_arrays(starts, sizes, pids, offsets, cms, device)

    # Create solver
    solver = ShapeMatchingSolver(p_arrays, c_arrays, device)
    solver.dt = 0.005
    solver.stiffness = 0.6
    solver.damping = 0.98
    solver.num_iterations = 4

    # Test 1: Correctness — run 50 steps, check centroid drops
    print("\n" + "=" * 50)
    print("Test 1: Correctness (50 steps)")
    print("=" * 50)

    cm_initial = solver.get_centroid()
    print(f"  Initial centroid: ({cm_initial[0]:.4f}, {cm_initial[1]:.4f}, {cm_initial[2]:.4f})")

    for step in range(50):
        solver.step()
        if step < 5 or step % 10 == 0:
            cm = solver.get_centroid()
            print(f"  Step {step:3d}: cm=({cm[0]:.4f}, {cm[1]:.4f}, {cm[2]:.4f})")

    cm_final = solver.get_centroid()
    print(f"  Final centroid:   ({cm_final[0]:.4f}, {cm_final[1]:.4f}, {cm_final[2]:.4f})")

    # Verify cube dropped (y decreased) and didn't explode
    y_dropped = cm_initial[1] - cm_final[1]
    print(f"\n  Y dropped: {y_dropped:.4f}")
    assert y_dropped > 0.01, f"Cube didn't drop! y_dropped={y_dropped}"
    assert cm_final[1] > -1.0, f"Cube exploded! y={cm_final[1]}"
    assert cm_final[1] < 2.0, f"Cube flew up! y={cm_final[1]}"
    print("  CORRECTNESS PASSED: cube dropped and is stable")

    # Test 2: Performance — 500 steps
    print("\n" + "=" * 50)
    print("Test 2: Performance (500 steps)")
    print("=" * 50)

    wp.synchronize()
    t0 = time.perf_counter()
    for step in range(500):
        solver.step()
    wp.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    fps = 500.0 / elapsed
    ms_per_step = elapsed / 500.0 * 1000.0
    print(f"  Total: {elapsed:.3f}s")
    print(f"  Per step: {ms_per_step:.2f}ms")
    print(f"  FPS: {fps:.1f}")

    cm_after_perf = solver.get_centroid()
    print(f"  Final centroid: ({cm_after_perf[0]:.4f}, {cm_after_perf[1]:.4f}, {cm_after_perf[2]:.4f})")

    # Test 3: Compare with C++ reference
    print("\n" + "=" * 50)
    print("Test 3: C++ Reference Comparison")
    print("=" * 50)
    print(f"  C++ headless (cube32): 522 FPS (~1.9ms/step)")
    print(f"  Warp:                  {fps:.0f} FPS (~{ms_per_step:.1f}ms/step)")
    if fps > 100:
        print("  PERFORMANCE OK (>100 FPS)")
    elif fps > 30:
        print("  PERFORMANCE ACCEPTABLE (>30 FPS)")
    else:
        print("  PERFORMANCE WARNING (<30 FPS)")

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
