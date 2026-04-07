"""Phase 3 test: P(F) Damage + Soft Removal.
Knife contacts cube -> damage accumulates -> particles fade via sigmoid alpha.
"""
import sys
import numpy as np
import warp as wp

from mesh.loader import (
    load_tet_mesh, build_vertex_neighbors,
    build_clusters_from_tets, pack_clusters_csr, build_edges,
)
from sim.particles import init_warp_arrays, init_cluster_arrays, init_edge_arrays
from sim.solver import ShapeMatchingSolver


def main():
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "/home/yunxin/xpbd-tissue-sim/resource/cube/cube32.msh"

    wp.init()
    device = "cuda:0"

    # Load and prepare
    print("Loading mesh + clusters + edges...")
    vertices, tets = load_tet_mesh(mesh_path)
    N = len(vertices)
    min_y = vertices[:, 1].min()
    vertices[:, 1] += -min_y + 0.001

    vertex_neighbors = build_vertex_neighbors(tets, N)
    cluster_ids, cluster_cms, cluster_offsets = build_clusters_from_tets(vertices, vertex_neighbors, rings=2)
    starts, sizes, pids, offsets, cms = pack_clusters_csr(cluster_ids, cluster_offsets, cluster_cms)
    edge_a, edge_b, edge_pf_rest = build_edges(vertices, cluster_ids, vertex_neighbors)

    fixed_mask = np.zeros(N, dtype=np.int32)
    for i in range(N):
        if vertices[i, 1] < 0.05:
            fixed_mask[i] = 1

    # Init Warp arrays
    p_arrays = init_warp_arrays(vertices, fixed_mask, device)
    c_arrays = init_cluster_arrays(starts, sizes, pids, offsets, cms, device)
    e_arrays = init_edge_arrays(edge_a, edge_b, edge_pf_rest, device)

    # Create solver with edges
    solver = ShapeMatchingSolver(p_arrays, c_arrays, e_arrays, device)
    solver.dt = 0.005
    solver.stiffness = 0.6
    solver.damping = 0.98
    solver.num_iterations = 4
    solver.cutting_update_interval = 50

    # Phase 1: Let cube settle (200 steps, no knife)
    print("\n" + "=" * 50)
    print("Phase 3 Test: P(F) Damage + Soft Removal")
    print("=" * 50)

    print("\nSettling (200 steps, no knife)...")
    for step in range(200):
        solver.step()
    cm = solver.get_centroid()
    print(f"  Settled centroid: ({cm[0]:.3f}, {cm[1]:.3f}, {cm[2]:.3f})")

    # Phase 2: Activate knife at center of cube
    print("\nActivating knife at cube center...")
    solver.knife_pos = wp.vec3(0.5, 0.5, 0.5)
    solver.knife_radius = 0.15
    solver.knife_active = 1

    # Run 500 steps with knife active
    print("Running 500 steps with knife active...")
    for step in range(500):
        solver.step()
        if step % 100 == 0:
            wp.synchronize()
            # Check damage and alpha stats
            damage_np = solver.p["damage"].numpy()
            alpha_np = solver.p["alpha"].numpy()
            edge_damage_np = solver.e["damage"].numpy()

            damaged_particles = np.sum(damage_np > 0.01)
            removed_particles = np.sum(alpha_np < 0.5)
            damaged_edges = np.sum(edge_damage_np > 0.01)
            max_edge_damage = edge_damage_np.max()
            max_particle_damage = damage_np.max()

            print(f"  Step {step:3d}: edges_damaged={damaged_edges}, "
                  f"max_edge_dmg={max_edge_damage:.3f}, "
                  f"particles_damaged={damaged_particles}, "
                  f"max_p_dmg={max_particle_damage:.3f}, "
                  f"removed(alpha<0.5)={removed_particles}")

    # Final stats
    wp.synchronize()
    damage_np = solver.p["damage"].numpy()
    alpha_np = solver.p["alpha"].numpy()
    edge_damage_np = solver.e["damage"].numpy()

    print(f"\n--- Final Stats ---")
    print(f"  Particles with damage > 0:     {np.sum(damage_np > 0.01)}")
    print(f"  Particles with alpha < 0.5:    {np.sum(alpha_np < 0.5)} (soft removed)")
    print(f"  Particles with alpha < 0.01:   {np.sum(alpha_np < 0.01)} (fully removed)")
    print(f"  Max particle damage:           {damage_np.max():.4f}")
    print(f"  Edges with damage > 0:         {np.sum(edge_damage_np > 0.01)}")
    print(f"  Max edge damage:               {edge_damage_np.max():.4f}")

    # Verify
    print("\n--- Verification ---")
    has_damage = np.sum(damage_np > 0.01) > 0
    has_removal = np.sum(alpha_np < 0.5) > 0
    print(f"  Damage accumulated:  {'PASS' if has_damage else 'FAIL'}")
    print(f"  Soft removal active: {'PASS' if has_removal else 'FAIL'}")

    if has_damage and has_removal:
        print("\n  PHASE 3 PASSED: P(F) damage + soft removal working.")
    else:
        print("\n  PHASE 3 NEEDS INVESTIGATION")
        if not has_damage:
            print("    - No edge/particle damage detected. Check knife position/radius.")
        if not has_removal:
            print("    - No particles removed. Check aspiration threshold or damage levels.")

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
