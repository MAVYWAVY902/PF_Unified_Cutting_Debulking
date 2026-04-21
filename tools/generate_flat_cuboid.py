"""Generate a flat cuboid tet mesh (1.0 x Ly x 1.0) with structured 6-tet decomposition.

Produces a .msh (gmsh 2.2 ASCII) with the same topology properties as cube_dense.msh
— dense enough for shape-matching PBD, preserves surface faces, uniform particle spacing.

Usage:
    python tools/generate_flat_cuboid.py --height 0.4 --spacing 0.03 \\
        --out resources/cuboid_flat/cuboid_h040.msh
"""
import argparse
import numpy as np
import meshio


def generate_cuboid(Lx: float, Ly: float, Lz: float, spacing: float):
    """Structured hex grid + Kuhn-Pennack 6-tet decomposition.

    Each hex corner v0..v7:
        v0=(i,  j,  k)    v4=(i,  j,  k+1)
        v1=(i+1,j,  k)    v5=(i+1,j,  k+1)
        v2=(i+1,j+1,k)    v6=(i+1,j+1,k+1)
        v3=(i,  j+1,k)    v7=(i,  j+1,k+1)

    All 6 tets share diagonal v0-v6, all have positive volume.
    """
    nx = int(np.round(Lx / spacing)) + 1
    ny = int(np.round(Ly / spacing)) + 1
    nz = int(np.round(Lz / spacing)) + 1

    xs = np.linspace(0.0, Lx, nx, dtype=np.float64)
    ys = np.linspace(0.0, Ly, ny, dtype=np.float64)
    zs = np.linspace(0.0, Lz, nz, dtype=np.float64)

    N = nx * ny * nz
    pts = np.zeros((N, 3), dtype=np.float64)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = k * ny * nx + j * nx + i
                pts[idx] = (xs[i], ys[j], zs[k])

    def v(i, j, k):
        return k * ny * nx + j * nx + i

    tets = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                v0 = v(i,     j,     k)
                v1 = v(i + 1, j,     k)
                v2 = v(i + 1, j + 1, k)
                v3 = v(i,     j + 1, k)
                v4 = v(i,     j,     k + 1)
                v5 = v(i + 1, j,     k + 1)
                v6 = v(i + 1, j + 1, k + 1)
                v7 = v(i,     j + 1, k + 1)
                # Kuhn-Pennack 6-tet subdivision sharing diagonal v0-v6
                tets.extend([
                    [v0, v1, v2, v6],
                    [v0, v2, v3, v6],
                    [v0, v3, v7, v6],
                    [v0, v7, v4, v6],
                    [v0, v4, v5, v6],
                    [v0, v5, v1, v6],
                ])
    tets = np.asarray(tets, dtype=np.int64)

    # Sanity: positive volumes
    p = pts[tets]  # (T, 4, 3)
    v_ab = p[:, 1] - p[:, 0]
    v_ac = p[:, 2] - p[:, 0]
    v_ad = p[:, 3] - p[:, 0]
    vol = np.einsum('ij,ij->i', np.cross(v_ab, v_ac), v_ad) / 6.0
    n_neg = int((vol <= 0).sum())
    if n_neg:
        # flip orientation to make positive
        tets[vol < 0] = tets[vol < 0][:, [0, 1, 3, 2]]
        print(f"[warn] Flipped {n_neg} negative-volume tets.")
    return pts, tets, (nx, ny, nz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lx", type=float, default=1.0, help="X extent")
    ap.add_argument("--Ly", "--height", type=float, default=0.4,
                    help="Y extent (HEIGHT, the short axis)")
    ap.add_argument("--Lz", type=float, default=1.0, help="Z extent")
    ap.add_argument("--spacing", type=float, default=0.03,
                    help="target particle spacing (cube_dense.msh = 0.031)")
    ap.add_argument("--out", required=True, help="output .msh path")
    args = ap.parse_args()

    print(f"[generate] cuboid {args.Lx} x {args.Ly} x {args.Lz} at spacing {args.spacing}")
    pts, tets, (nx, ny, nz) = generate_cuboid(args.Lx, args.Ly, args.Lz, args.spacing)
    print(f"[generate] grid {nx} x {ny} x {nz} = {len(pts)} vertices, {len(tets)} tets")

    mesh = meshio.Mesh(
        points=pts,
        cells=[("tetra", tets.astype(np.int32))],
    )
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # gmsh 2.2 ASCII format — matches cube_dense.msh
    meshio.write(args.out, mesh, file_format="gmsh22")
    size_kb = os.path.getsize(args.out) / 1024
    print(f"[write] {args.out} ({size_kb:.0f} KB)")

    # Quick stats
    print("\n=== Mesh stats ===")
    print(f"  bbox x: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
    print(f"  bbox y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]  ← HEIGHT")
    print(f"  bbox z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
    print(f"  vertices:    {len(pts):>7}")
    print(f"  tetrahedra:  {len(tets):>7}")
    # boundary faces count
    from collections import defaultdict
    fc = defaultdict(int)
    for t in tets:
        for face in [(t[0],t[1],t[2]), (t[0],t[1],t[3]),
                     (t[0],t[2],t[3]), (t[1],t[2],t[3])]:
            fc[tuple(sorted(face))] += 1
    boundary = sum(1 for c in fc.values() if c == 1)
    print(f"  surface tris: {boundary:>7}  (faces in only 1 tet)")


if __name__ == "__main__":
    main()
