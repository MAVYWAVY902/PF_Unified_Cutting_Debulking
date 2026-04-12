"""Tet mesh loader: .msh -> numpy arrays + cluster building"""
import numpy as np
import meshio
from collections import defaultdict


def load_tet_mesh(filepath: str):
    """Load .msh tet mesh, return vertices and tets."""
    mesh = meshio.read(filepath)
    vertices = mesh.points.astype(np.float32)  # (N, 3)

    # Find tet cells
    tets = None
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            tets = cell_block.data
            break
    if tets is None:
        raise ValueError(f"No tetrahedra found in {filepath}")

    print(f"Loaded: {len(vertices)} vertices, {len(tets)} tetrahedra")
    return vertices, tets


def build_vertex_neighbors(tets: np.ndarray, num_vertices: int):
    """Build vertex adjacency from tet connectivity.
    Each tet has 4 vertices, all pairs are neighbors.
    Returns list of sorted neighbor arrays per vertex.
    """
    neighbors = defaultdict(set)
    for tet in tets:
        for i in range(4):
            for j in range(i + 1, 4):
                neighbors[tet[i]].add(tet[j])
                neighbors[tet[j]].add(tet[i])

    result = []
    for v in range(num_vertices):
        result.append(np.array(sorted(neighbors.get(v, [])), dtype=np.int32))
    return result


def build_clusters_from_tets(vertices: np.ndarray, vertex_neighbors: list, rings: int = 2):
    """Build overlapping clusters via BFS ring expansion from each vertex.
    Mirrors C++ tet_mesh.h::build_clusters_from_tets().

    Returns:
        cluster_particle_ids: list of np.array, each cluster's particle IDs
        cluster_rest_cm: np.array (C, 3), rest center of mass per cluster
        cluster_rest_offsets: list of np.array (n_i, 3), rest offsets per cluster
    """
    N = len(vertices)
    cluster_particle_ids = []
    cluster_rest_cm = []
    cluster_rest_offsets = []

    for i in range(N):
        # BFS: collect `rings`-hop neighbors
        members = {i}
        frontier = [i]

        for r in range(rings):
            next_frontier = []
            for v in frontier:
                for nb in vertex_neighbors[v]:
                    if nb not in members:
                        members.add(nb)
                        next_frontier.append(nb)
            frontier = next_frontier

        ids = np.array(sorted(members), dtype=np.int32)
        if len(ids) < 2:
            continue

        # Compute rest center of mass (uniform mass = 1.0)
        positions = vertices[ids]  # (n, 3)
        rest_cm = positions.mean(axis=0)  # (3,)
        rest_off = positions - rest_cm    # (n, 3)

        cluster_particle_ids.append(ids)
        cluster_rest_cm.append(rest_cm)
        cluster_rest_offsets.append(rest_off)

    print(f"Built {len(cluster_particle_ids)} clusters (rings={rings})")

    # Stats
    sizes = [len(c) for c in cluster_particle_ids]
    print(f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")

    return cluster_particle_ids, cluster_rest_cm, cluster_rest_offsets


def pack_clusters_csr(cluster_particle_ids, cluster_rest_offsets, cluster_rest_cm):
    """Pack cluster data into CSR format for GPU.

    Returns:
        starts: np.array (C,) int32 — CSR row pointers
        sizes:  np.array (C,) int32 — cluster sizes
        pids:   np.array (total,) int32 — all particle IDs concatenated
        offsets: np.array (total, 3) float32 — all rest offsets concatenated
        cms:    np.array (C, 3) float32 — rest center of mass per cluster
    """
    C = len(cluster_particle_ids)
    starts = np.zeros(C, dtype=np.int32)
    sizes = np.zeros(C, dtype=np.int32)

    all_pids = []
    all_offsets = []
    offset = 0

    for ci in range(C):
        starts[ci] = offset
        sizes[ci] = len(cluster_particle_ids[ci])
        all_pids.append(cluster_particle_ids[ci])
        all_offsets.append(cluster_rest_offsets[ci])
        offset += sizes[ci]

    pids = np.concatenate(all_pids).astype(np.int32)
    offsets = np.concatenate(all_offsets).astype(np.float32)
    cms = np.array(cluster_rest_cm, dtype=np.float32)

    print(f"CSR packed: {C} clusters, {len(pids)} total entries")
    return starts, sizes, pids, offsets, cms


def build_edges(vertices: np.ndarray, cluster_particle_ids: list, vertex_neighbors: list):
    """Build near-neighbor edges for P(F) damage tracking.
    Mirrors C++ cutting_manager.h::init().

    Returns:
        edge_a, edge_b: np.array (E,) int32
        edge_pf_rest: np.array (E,) float32 — initial P(F) values
    """
    N = len(vertices)

    # Compute average nearest neighbor distance
    avg_spacing = 0.0
    count = 0
    for v in range(min(N, 500)):
        if len(vertex_neighbors[v]) > 0:
            dists = np.linalg.norm(vertices[vertex_neighbors[v]] - vertices[v], axis=1)
            avg_spacing += dists.min()
            count += 1
    avg_spacing /= max(count, 1)
    max_edge_len = avg_spacing * 2.5  # match C++ cutting_manager.h

    # Collect unique edges within distance threshold
    edge_set = set()
    edges_a = []
    edges_b = []

    for ci, ids in enumerate(cluster_particle_ids):
        id_set = set(ids.tolist())
        for i_idx in range(len(ids)):
            for j_idx in range(i_idx + 1, len(ids)):
                a, b = int(ids[i_idx]), int(ids[j_idx])
                if a > b:
                    a, b = b, a
                if (a, b) in edge_set:
                    continue
                dist = np.linalg.norm(vertices[a] - vertices[b])
                if dist > max_edge_len:
                    continue
                edge_set.add((a, b))
                edges_a.append(a)
                edges_b.append(b)

    edge_a = np.array(edges_a, dtype=np.int32)
    edge_b = np.array(edges_b, dtype=np.int32)

    # Compute initial P(F) for each edge (simplified: based on edge length ratio)
    # Full P(F) requires nearest-intruder query; for now store edge half-length as proxy
    edge_pf_rest = np.ones(len(edge_a), dtype=np.float32)
    alpha_min = 10.0

    for ei in range(len(edge_a)):
        a, b = edge_a[ei], edge_b[ei]
        center = (vertices[a] + vertices[b]) / 2.0
        radius = np.linalg.norm(vertices[a] - vertices[b]) / 2.0
        if radius < 1e-10:
            continue

        # Find nearest non-member point
        min_dist = 1e10
        for nb in vertex_neighbors[a]:
            if nb == b:
                continue
            d = np.linalg.norm(vertices[nb] - center)
            min_dist = min(min_dist, d)
        for nb in vertex_neighbors[b]:
            if nb == a:
                continue
            d = np.linalg.norm(vertices[nb] - center)
            min_dist = min(min_dist, d)

        if min_dist < 1e9:
            sdist = min_dist - radius
            pf = 1.0 / (1.0 + np.exp(-alpha_min * sdist / radius))
            edge_pf_rest[ei] = pf

    print(f"Built {len(edge_a)} edges, P(F) range: [{edge_pf_rest.min():.3f}, {edge_pf_rest.max():.3f}]")
    return edge_a, edge_b, edge_pf_rest
