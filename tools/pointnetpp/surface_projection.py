"""Surface-projection grouping for observability-aligned tau parameterization.

Motivation (verified 2026-06-03):
   - Surface-only obs (D405) makes per-particle tau under-determined: 87% interior
     particles get ZERO gradient (L_cav only on surface) -> recovery capped ~14%.
   - Depth (along surface-normal / unobservable) tau info is STRICTLY zero in the
     measurement (buried hard inclusion -> surface RMS = 0.0000, proven).
   - Fix: group particles by SURFACE PROJECTION. Each interior particle shares tau
     with its nearest surface patch (along the unobservable direction). Every group
     touches the surface -> gradient flows + interior inherits surface-learned value.

Two assignment schemes:
   1. column_grid_assignment   — (x,z) grid for axis-aligned shapes (cube). Fast,
       validated (GRID=8 -> ~49-54% vs per-particle 5.5%).
   2. surface_bfs_assignment   — general mesh-graph BFS: each interior particle ->
       nearest surface particle along edges. Handles concave shapes (torus/tumor)
       where Euclidean nearest would cross voids.

Both return col_id[N] (int per-particle group id, 0..K-1) where EVERY group is
guaranteed to contain >=1 surface particle.
"""
import numpy as np
from collections import deque


def column_grid_assignment(pos, surf_mask, top_axis, grid_n):
    """Bin particles into grid_n x grid_n columns on the two non-top axes.

    Each column spans the full depth (top_axis), so every occupied column that
    has surface particles connects surface->interior. Groups with zero surface
    particles are merged into their nearest surface-bearing neighbor.

    Args:
        pos:        (N,3) rest positions
        surf_mask:  (N,) bool, True for observable surface particles
        top_axis:   int 0/1/2 — the depth axis (column runs along this)
        grid_n:     resolution per lateral axis -> up to grid_n^2 columns
    Returns:
        col_id: (N,) int in 0..K-1, every group has >=1 surface particle
        info:   dict with K, surface_per_group stats
    """
    N = pos.shape[0]
    lateral = [a for a in range(3) if a != top_axis]
    a0, a1 = lateral
    lo = pos.min(0); hi = pos.max(0); span = hi - lo + 1e-9
    i0 = np.clip(((pos[:, a0] - lo[a0]) / span[a0] * grid_n).astype(int), 0, grid_n - 1)
    i1 = np.clip(((pos[:, a1] - lo[a1]) / span[a1] * grid_n).astype(int), 0, grid_n - 1)
    raw = i0 * grid_n + i1                      # 0..grid_n^2-1

    col_id, info = _remap_and_fix_surface(raw, surf_mask, pos)
    info["scheme"] = f"column_grid_{grid_n}x{grid_n}"
    info["top_axis"] = top_axis
    return col_id, info


def surface_bfs_assignment(pos, edges_a, edges_b, surf_mask):
    """Multi-source BFS from all surface particles along mesh edges. Each interior
    particle is assigned to the surface particle whose BFS wavefront reaches it
    first -> nearest surface particle along the MESH (geodesic-ish), not Euclidean.

    This is the general scheme for concave shapes: the projection follows the
    surface-normal direction into the volume (the unobservable / knife-entry
    direction), never crossing voids.

    Args:
        pos:        (N,3) rest positions (only for tie-break / info)
        edges_a,b:  (E,) int edge endpoints (mesh connectivity)
        surf_mask:  (N,) bool surface particles
    Returns:
        col_id: (N,) int — each particle's group = the surface particle it maps to
                (relabeled to 0..K-1 over surface particles actually used as seeds)
        info:   dict
    """
    N = pos.shape[0]
    # adjacency list
    adj = [[] for _ in range(N)]
    for a, b in zip(edges_a.tolist(), edges_b.tolist()):
        adj[a].append(b)
        adj[b].append(a)

    seed = np.full(N, -1, dtype=np.int64)       # which surface particle owns me
    dq = deque()
    surf_idx = np.where(surf_mask)[0]
    for s in surf_idx:
        seed[s] = s
        dq.append(s)
    # multi-source BFS
    while dq:
        u = dq.popleft()
        su = seed[u]
        for v in adj[u]:
            if seed[v] == -1:
                seed[v] = su
                dq.append(v)
    # any isolated particle (no edge path to surface) -> nearest surface by Euclidean
    iso = np.where(seed == -1)[0]
    if len(iso):
        sp = pos[surf_idx]
        for i in iso:
            d = ((sp - pos[i]) ** 2).sum(1)
            seed[i] = surf_idx[d.argmin()]

    # relabel seeds (surface particles used) to 0..K-1
    used = np.unique(seed)
    remap = {int(s): k for k, s in enumerate(used)}
    col_id = np.array([remap[int(s)] for s in seed], dtype=np.int64)
    K = len(used)
    spg = np.array([int(((col_id == k) & surf_mask).sum()) for k in range(K)])
    info = dict(scheme="surface_bfs", K=K,
                surf_per_group_mean=float(spg.mean()),
                surf_per_group_min=int(spg.min()),
                groups_touching_surface=int((spg > 0).sum()),
                isolated_fixed=int(len(iso)))
    return col_id, info


def coarse_surface_bfs_assignment(pos, edges_a, edges_b, surf_mask, n_seeds):
    """Production general scheme: pick n_seeds surface seeds via farthest-point
    sampling (even spatial spread on the surface), then multi-source BFS along
    mesh edges. Each particle -> nearest seed along the mesh. Controllable K=n_seeds.

    Handles concave shapes (torus/tumor): BFS follows mesh connectivity (never
    crosses voids), and grouping runs along the surface-normal / depth direction.

    Args:
        pos, edges_a, edges_b, surf_mask: as above
        n_seeds: target number of surface patches (= K). Choose ~ N_surface/40.
    Returns:
        col_id (N,) in 0..K-1, info dict. Every group touches surface (seed is surface).
    """
    N = pos.shape[0]
    surf_idx = np.where(surf_mask)[0]
    ns = min(n_seeds, len(surf_idx))
    # farthest-point sampling on surface particles (Euclidean FPS)
    sp = pos[surf_idx]
    seeds_local = [0]
    d2 = ((sp - sp[0]) ** 2).sum(1)
    for _ in range(1, ns):
        nxt = int(d2.argmax())
        seeds_local.append(nxt)
        d2 = np.minimum(d2, ((sp - sp[nxt]) ** 2).sum(1))
    seed_particles = surf_idx[np.array(seeds_local)]   # global indices of seeds

    # adjacency
    adj = [[] for _ in range(N)]
    for a, b in zip(edges_a.tolist(), edges_b.tolist()):
        adj[a].append(b); adj[b].append(a)
    owner = np.full(N, -1, dtype=np.int64)
    dq = deque()
    for k, s in enumerate(seed_particles):
        owner[s] = k; dq.append(s)
    while dq:
        u = dq.popleft(); ou = owner[u]
        for v in adj[u]:
            if owner[v] == -1:
                owner[v] = ou; dq.append(v)
    iso = np.where(owner == -1)[0]
    if len(iso):
        sp2 = pos[seed_particles]
        for i in iso:
            owner[i] = int(((sp2 - pos[i]) ** 2).sum(1).argmin())
    K = ns
    spg = np.array([int(((owner == k) & surf_mask).sum()) for k in range(K)])
    info = dict(scheme="coarse_surface_bfs", K=K,
                surf_per_group_mean=float(spg.mean()),
                surf_per_group_min=int(spg.min()),
                groups_touching_surface=int((spg > 0).sum()),
                isolated_fixed=int(len(iso)))
    return owner.astype(np.int64), info


def _remap_and_fix_surface(raw, surf_mask, pos):
    """Relabel raw ids to 0..K-1 over OCCUPIED cells, then merge any group with no
    surface particle into the nearest (by centroid) surface-bearing group."""
    uniq = np.unique(raw)
    remap = {int(c): i for i, c in enumerate(uniq)}
    col = np.array([remap[int(c)] for c in raw], dtype=np.int64)
    K = len(uniq)
    has_surf = np.array([bool(((col == k) & surf_mask).any()) for k in range(K)])
    if not has_surf.all():
        # centroids
        cent = np.array([pos[col == k].mean(0) for k in range(K)])
        surf_groups = np.where(has_surf)[0]
        for k in np.where(~has_surf)[0]:
            d = ((cent[surf_groups] - cent[k]) ** 2).sum(1)
            target = surf_groups[d.argmin()]
            col[col == k] = target
        # relabel again to compact 0..K'-1
        uniq2 = np.unique(col)
        remap2 = {int(c): i for i, c in enumerate(uniq2)}
        col = np.array([remap2[int(c)] for c in col], dtype=np.int64)
        K = len(uniq2)
    spg = np.array([int(((col == k) & surf_mask).sum()) for k in range(K)])
    info = dict(K=K, surf_per_group_mean=float(spg.mean()),
                surf_per_group_min=int(spg.min()),
                groups_touching_surface=int((spg > 0).sum()))
    return col, info


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
    import warp as wp; wp.init()
    from tools.day2_solver_wrapper import setup_solver_with_rest

    cfg_path = str(REPO / "resources/cube/cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(cfg_path, fix_threshold_fraction=0.20)
    N = solver.N
    pos = mesh_info["vertices"]
    bb0, bb1 = pos.min(0), pos.max(0); bs = bb1 - bb0
    top_axis = 1
    surf = pos[:, top_axis] > (bb1[top_axis] - 0.10 * bs[top_axis])
    print(f"N={N}, surface={surf.sum()} ({100*surf.mean():.1f}%)")

    print("\n=== column_grid_assignment (GRID=8) ===")
    col, info = column_grid_assignment(pos, surf, top_axis, grid_n=8)
    print(f"  K={info['K']}  surf/group mean={info['surf_per_group_mean']:.1f} "
          f"min={info['surf_per_group_min']}  touching_surface={info['groups_touching_surface']}/{info['K']}")
    assert info["groups_touching_surface"] == info["K"], "some group has no surface!"
    assert info["surf_per_group_min"] >= 1
    print("  ✓ every group touches surface")

    print("\n=== surface_bfs_assignment ===")
    ea = solver.e["a"].numpy().astype(np.int64)
    eb = solver.e["b"].numpy().astype(np.int64)
    col2, info2 = surface_bfs_assignment(pos, ea, eb, surf)
    print(f"  K={info2['K']}  surf/group mean={info2['surf_per_group_mean']:.2f} "
          f"min={info2['surf_per_group_min']}  touching={info2['groups_touching_surface']}/{info2['K']}  "
          f"isolated_fixed={info2['isolated_fixed']}")
    assert info2["groups_touching_surface"] == info2["K"]
    print("  ✓ every BFS group has a surface seed (by construction)")
    # BFS K = number of surface particles (each is its own seed) — too fine. Note for caller.
    print(f"  NOTE: raw BFS gives K={info2['K']} (= #surface particles, too fine).")

    print("\n=== coarse_surface_bfs_assignment (n_seeds=64, production general) ===")
    col3, info3 = coarse_surface_bfs_assignment(pos, ea, eb, surf, n_seeds=64)
    print(f"  K={info3['K']}  surf/group mean={info3['surf_per_group_mean']:.1f} "
          f"min={info3['surf_per_group_min']}  touching={info3['groups_touching_surface']}/{info3['K']}  "
          f"isolated_fixed={info3['isolated_fixed']}")
    assert info3["groups_touching_surface"] == info3["K"]
    assert info3["surf_per_group_min"] >= 1
    print("  ✓ every group touches surface, K controllable")
    print("\nSTEP 1 DONE — surface_projection module verified (grid + raw-BFS + coarse-BFS).")
