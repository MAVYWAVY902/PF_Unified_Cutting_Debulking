"""Between-frame cluster split for the Warp solver — numpy/scipy port of cpp_sim's
cpu_bfs_split() (main_gpu_viewer.cpp) + process_fragments() (cutting_manager.h).

Runs DETACHED on the solver's current numpy-downloadable state (no autograd tape),
mirroring how cpp_sim runs the split on CPU around the GPU step. Rebuilds the solver's
cluster arrays (starts/sizes/particle_ids/rest_offsets/rest_cm/cluster_quat) +
num_clusters so the NEXT forward step uses the new topology. tau gradient unaffected
(per-frame, alpha channel; split is between frames like teacher-forcing).

Algorithm (verified against cpp_sim):
 1. broken edges: edge_damage > break_thresh; alive: alpha > alive_thresh
 2. connected components on alive particles via non-broken edges (scipy)
 3. drop small fragments (< min_frag, except largest) -> alpha=0 (CUSA aspirate)
 4. for each cluster spanning >1 component: split PER-COMPONENT — every component with
    >= min_new members becomes its own sub-cluster, smaller components are dropped from
    THIS cluster (cpp_sim check_and_split drops size<2). This is NOT all-or-nothing: a
    single un-split straddling cluster would weld the two sides of the cut together and
    keep the severed chunk floating. rest from CURRENT positions (cpp_sim cluster.init);
    unchanged clusters keep ORIGINAL rest_offsets.
 4b. orphan handling: alive particles that end up in ZERO clusters after the split are
    isolated debris -> removed (alpha=0). cpp leaves them clusterless (= free); we make
    them debris for a clean cavity. With min_new=2 this only hits truly isolated points.
 5. rebuild solver.c arrays + num_clusters; reset cluster_quat for split clusters.
"""
import numpy as np
import warp as wp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def split_clusters_inplace(solver, alive_thresh=0.5, break_thresh=0.5,
                           min_frag=5, min_new=2, drop_small=True,
                           mark_orphans_removed=True):
    """Mutates solver.c (cluster arrays) + solver.num_clusters + solver.p alpha/damage.
    Returns dict(n_comp, dropped, split_clusters, orphans, C_before, C_after).

    min_new=2 matches cpp_sim (drop components of size <2). Per-component split (not
    all-or-nothing) so every weld across a cut breaks. mark_orphans_removed=True turns
    alive-but-clusterless particles into debris (alpha=0)."""
    N = solver.N
    dev = solver.device
    ea = solver.e["a"].numpy().astype(np.int64)
    eb = solver.e["b"].numpy().astype(np.int64)
    ed = solver.e["damage"].numpy()
    alpha = solver.p["alpha"].numpy()
    pos = solver.p["pos"].numpy().astype(np.float32)
    starts = solver.c["starts"].numpy().astype(np.int64)
    sizes = solver.c["sizes"].numpy().astype(np.int64)
    pids = solver.c["particle_ids"].numpy().astype(np.int64)
    orig_roff = solver.c["rest_offsets"].numpy().astype(np.float32)
    orig_rcm = solver.c["rest_cm"].numpy().astype(np.float32)
    C0 = len(starts)

    alive = alpha > alive_thresh
    # adjacency from non-broken edges between two alive particles
    keep = (ed <= break_thresh) & alive[ea] & alive[eb]
    r = ea[keep]; c = eb[keep]
    rr = np.concatenate([r, c]); cc = np.concatenate([c, r])
    A = csr_matrix((np.ones(len(rr), np.int8), (rr, cc)), shape=(N, N))
    n_comp, labels = connected_components(A, directed=False)

    # component sizes counting ALIVE particles only
    comp_alive = np.bincount(labels, weights=alive.astype(np.float64), minlength=n_comp).astype(np.int64)
    largest = int(np.argmax(comp_alive))

    # 3. drop small alive fragments (except largest)
    newly_removed = np.zeros(N, bool)
    if drop_small:
        small = np.where((comp_alive > 0) & (comp_alive < min_frag))[0]
        small = small[small != largest]
        if len(small):
            newly_removed = np.isin(labels, small) & alive
    eff_alive = alive & ~newly_removed

    # 4. rebuild clusters
    new_pids, new_starts, new_sizes, new_roff, new_rcm, new_quat = [], [], [], [], [], []
    split_count = 0; dropped_clusters = 0; off = 0
    ident = np.array([0., 0., 0., 1.], np.float32)
    cq = solver.c["cluster_quat"].numpy().astype(np.float32)
    for cid in range(C0):
        s0 = starts[cid]; n = sizes[cid]
        members = pids[s0:s0 + n]
        am = members[eff_alive[members]]
        if len(am) < 2:
            dropped_clusters += 1; continue
        uniq = np.unique(labels[am])
        if len(uniq) == 1:
            # unchanged: keep ORIGINAL membership + ORIGINAL rest (dead members alpha-weighted out)
            new_starts.append(off); new_sizes.append(n)
            new_pids.append(members)
            new_roff.append(orig_roff[s0:s0 + n]); new_rcm.append(orig_rcm[cid])
            new_quat.append(cq[cid]); off += n
        else:
            # cpp check_and_split: PER-COMPONENT. Each component >= min_new becomes its
            # own sub-cluster; smaller components are dropped from THIS cluster. Not
            # all-or-nothing — otherwise one un-split straddling cluster welds the cut.
            split_count += 1
            for u in uniq:
                g = am[labels[am] == u]
                if len(g) < min_new:
                    continue                            # drop tiny component from this cluster
                cm = pos[g].mean(0)                      # rest from CURRENT pos (cpp_sim cluster.init)
                new_starts.append(off); new_sizes.append(len(g))
                new_pids.append(g); new_roff.append(pos[g] - cm); new_rcm.append(cm)
                new_quat.append(ident); off += len(g)

    # 4b. orphan handling: alive particles now in ZERO clusters (isolated debris left by
    #     aggressive splitting) -> remove (cpp leaves them free; we make them debris).
    n_orphans = 0
    if mark_orphans_removed:
        in_cluster = np.zeros(N, bool)
        if new_pids:
            in_cluster[np.concatenate(new_pids)] = True
        orphans = eff_alive & ~in_cluster
        n_orphans = int(orphans.sum())
        if n_orphans:
            newly_removed = newly_removed | orphans      # new_pids already exclude them

    changed = (split_count > 0) or (dropped_clusters > 0) or bool(newly_removed.any())
    if newly_removed.any():
        a = solver.p["alpha"].numpy(); a[newly_removed] = 0.0; solver.p["alpha"].assign(a)
        d = solver.p["damage"].numpy(); d[newly_removed] = np.maximum(d[newly_removed], 1.0)
        solver.p["damage"].assign(d)
    if not changed:
        return dict(n_comp=n_comp, dropped=0, split_clusters=0, orphans=n_orphans,
                    C_before=C0, C_after=C0)

    npids = np.concatenate(new_pids).astype(np.int32)
    nroff = np.concatenate(new_roff).astype(np.float32)
    nstarts = np.array(new_starts, np.int32)
    nsizes = np.array(new_sizes, np.int32)
    nrcm = np.stack(new_rcm).astype(np.float32)
    nquat = np.stack(new_quat).astype(np.float32)
    C = len(new_starts)

    solver.c["starts"] = wp.array(nstarts, dtype=wp.int32, device=dev)
    solver.c["sizes"] = wp.array(nsizes, dtype=wp.int32, device=dev)
    solver.c["particle_ids"] = wp.array(npids, dtype=wp.int32, device=dev)
    solver.c["rest_offsets"] = wp.array(nroff, dtype=wp.vec3, device=dev)
    solver.c["rest_cm"] = wp.array(nrcm, dtype=wp.vec3, device=dev)
    solver.c["cluster_quat"] = wp.array(nquat, dtype=wp.quat, device=dev, requires_grad=False)
    solver.c["num_clusters"] = C
    solver.c["total_entries"] = len(npids)
    solver.num_clusters = C
    return dict(n_comp=n_comp, dropped=int(newly_removed.sum()), split_clusters=split_count,
                orphans=n_orphans, C_before=C0, C_after=C)
