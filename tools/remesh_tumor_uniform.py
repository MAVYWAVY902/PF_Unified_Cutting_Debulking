#!/usr/bin/env python3
"""
Re-tetrahedralize tumor surface with UNIFORM interior density using TetGen.

Why this exists:
  The shipped `tumor_7800.msh` / `tumor_7800_dense.msh` have dense surface but
  sparse interior (shell-mesh pattern). When `cpp_sim` debulks through the
  interior, the exposed cavity walls are coarse internal tets → visible
  "jagged" rendering.

  This script loads the surface .obj, runs TetGen with a volume constraint
  (-a) that forces every tet below a target volume, producing an interior
  that's as dense as the surface. Output is MSH 4.x format that `cpp_sim`
  can load directly.

Dependencies (one-time install):
    pip install tetgen meshio numpy

Usage:
    python tools/remesh_tumor_uniform.py \
        --input  resources/tumor/tumor_7800.obj \
        --output resources/tumor/tumor_uniform.msh \
        --edge   0.0007

    --edge (meters): target uniform edge length.
        Current mesh surface avg_spacing = 0.000646 m (0.65 mm).
        Recommended first try: 0.0007 (~0.7 mm) → ~15-20k verts, balanced.
        For finer: 0.0005 (0.5 mm) → ~40-60k verts, slower but smoother.
        For coarser: 0.001  (1.0 mm) → ~5-8k verts, faster, acceptable.

Validation after running:
    - Load new mesh in `cpp_sim`: update `config/tumor_debulk.yaml:mesh:`
    - Check startup log:
        * "平均粒子间距" should be close to target edge length (not smaller)
        * "cluster_membership" likely drops (less surface clustering)
    - Tuning: if new mesh slower than expected, bump `--edge` up.
"""
import argparse
import math
import sys
from pathlib import Path


def regular_tet_volume(edge_length: float) -> float:
    """Volume of a regular tetrahedron with given edge length: V = a³ * √2 / 12."""
    return (edge_length ** 3) * math.sqrt(2.0) / 12.0


def _load_obj_geometry_only(path):
    """Minimal .obj reader — only reads `v` (vertices) and `f` (faces), ignores vt/vn.

    Used as a fallback when meshio rejects the file due to UV seam mismatches
    (`len(points) != len(obj:vt)`). TetGen needs only geometry + topology, so we
    can safely discard texture coords and normals.

    Handles face formats: `f v`, `f v/vt`, `f v/vt/vn`, `f v//vn`.
    Fan-triangulates quads / n-gons.
    """
    import numpy as np
    points = []
    tris = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            tag = parts[0]
            if tag == 'v' and len(parts) >= 4:
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag == 'f' and len(parts) >= 4:
                # Face references: parts[1..] look like "12", "12/34", "12/34/56", "12//56"
                # Extract vertex index (1-indexed in .obj) → 0-indexed
                idx = []
                for p in parts[1:]:
                    v_str = p.split('/')[0]
                    idx.append(int(v_str) - 1)
                # Fan-triangulate (works for triangles and convex polygons)
                for i in range(1, len(idx) - 1):
                    tris.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(points, dtype=np.float64), np.asarray(tris, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='Input surface mesh (.obj / .stl / .ply)')
    ap.add_argument('--output', required=True, help='Output tet mesh (.msh, gmsh 4 format)')
    ap.add_argument('--edge', type=float, default=0.0007,
                    help='Target uniform edge length in mesh units (meters for this project). Default 0.0007 = 0.7mm.')
    ap.add_argument('--min-ratio', type=float, default=1.5,
                    help='TetGen quality ratio (default 1.5). Lower = worse quality, fewer extra points.')
    ap.add_argument('--preserve-surface', action='store_true',
                    help='Pass -Y to TetGen (no Steiner points on surface). Preserves input surface topology '
                         'but interior-near-boundary tets may be lower quality. Off by default — priority is '
                         'uniform interior, surface can be slightly resampled.')
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[err] input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Lazy imports so --help works without deps installed
    try:
        import meshio
        import numpy as np
        import tetgen
    except ImportError as e:
        print(f"[err] missing dep: {e}", file=sys.stderr)
        print("      pip install tetgen meshio numpy", file=sys.stderr)
        sys.exit(1)

    # ---- 1. Load surface ----
    print(f"[1/4] Loading surface: {input_path}")
    points, tris = None, None
    try:
        # First try meshio (handles .msh, .stl, .ply, clean .obj well)
        surf = meshio.read(str(input_path))
        points = np.asarray(surf.points, dtype=np.float64)
        for block in surf.cells:
            if block.type == 'triangle':
                tris = np.asarray(block.data, dtype=np.int32)
                break
    except Exception as e:
        # meshio can fail on .obj files with UV seams (len(points) != len(obj:vt)).
        # Fall back to a minimal reader that only cares about geometry (v + f lines).
        if str(input_path).lower().endswith('.obj'):
            print(f"      meshio failed ({e.__class__.__name__}), falling back to minimal .obj reader")
            points, tris = _load_obj_geometry_only(input_path)
        else:
            raise

    if tris is None or len(tris) == 0:
        print("[err] input has no triangle cells", file=sys.stderr)
        sys.exit(1)
    print(f"      {len(points)} surface verts, {len(tris)} triangles")
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    print(f"      bounding box: {bbox_min} → {bbox_max}")
    print(f"      extent: {bbox_max - bbox_min}")

    # ---- 2. Compute volume constraint from target edge ----
    target_vol = regular_tet_volume(args.edge)
    print(f"[2/4] Target edge length: {args.edge} → max tet volume {target_vol:.3e}")

    # TetGen switches breakdown:
    #   p - PLC (tetrahedralize piecewise linear complex from input faces)
    #   q - quality (min radius-edge ratio; 1.5 is good)
    #   a - max tet volume
    #   Y - preserve surface (no Steiner points on input boundary)
    switches = f'pq{args.min_ratio}a{target_vol:.3e}'
    if args.preserve_surface:
        switches += 'Y'
    print(f"      TetGen switches: -{switches}")

    # ---- 3. Run TetGen ----
    print("[3/4] Running TetGen...")
    nodes, elems = None, None
    try:
        tet = tetgen.TetGen(points, tris)
        # tetrahedralize() in pyvista/tetgen may return different shapes across versions
        # (tuple, grid, or None). Ignore its return and use attribute access below.
        tet.tetrahedralize(order=1, switches=switches)
    except Exception as e:
        print(f"[err] TetGen itself failed during tetrahedralize: {e}", file=sys.stderr)
        print("      Common causes:", file=sys.stderr)
        print("        - Surface not watertight (holes, self-intersections)", file=sys.stderr)
        print("        - Degenerate triangles (zero area, duplicate verts)", file=sys.stderr)
        print("      Try: MeshLab → Filters → Cleaning and Repairing → Merge Close Vertices / Close Holes,", file=sys.stderr)
        print("           then export .obj and rerun.", file=sys.stderr)
        sys.exit(1)

    # Extract nodes + elements by trying common attribute names across tetgen versions
    for attr in ('node', 'nodes', 'points'):
        if hasattr(tet, attr):
            nodes = np.asarray(getattr(tet, attr), dtype=np.float64)
            break
    for attr in ('elem', 'elements', 'tets'):
        if hasattr(tet, attr):
            elems = np.asarray(getattr(tet, attr), dtype=np.int64)
            break
    if nodes is None or elems is None:
        # Last resort: pull from .grid (pyvista UnstructuredGrid)
        if hasattr(tet, 'grid'):
            grid = tet.grid
            nodes = np.asarray(grid.points, dtype=np.float64)
            # VTK_TETRA cell type = 10
            cells_dict = grid.cells_dict if hasattr(grid, 'cells_dict') else {}
            elems = cells_dict.get(10)  # VTK tet type
            if elems is None:
                # Fallback: try keys that might map to tet cells
                for key, val in cells_dict.items():
                    if val.shape[-1] == 4:
                        elems = val
                        break
            elems = np.asarray(elems, dtype=np.int64) if elems is not None else None
    if nodes is None or elems is None:
        print(f"[err] Could not extract nodes/elems from tetgen object. Attributes: {dir(tet)}",
              file=sys.stderr)
        sys.exit(1)

    n_verts = len(nodes)
    n_tets = len(elems)
    print(f"      Done: {n_verts} verts, {n_tets} tets")

    # Sanity: average nearest-neighbor distance as a density check
    # Sample 100 random verts
    sample_idx = np.random.RandomState(42).choice(n_verts, size=min(100, n_verts), replace=False)
    sample = nodes[sample_idx]
    # O(100 × N) — cheap for 100 samples, still O(N) per sample
    min_dists = []
    for p in sample:
        d2 = np.sum((nodes - p) ** 2, axis=1)
        d2_sorted = np.partition(d2, 1)[:2]  # self + nearest
        nearest_sq = d2_sorted[1]  # skip self
        min_dists.append(math.sqrt(nearest_sq))
    avg_spacing = float(np.mean(min_dists))
    print(f"      Sampled avg_spacing: {avg_spacing:.6f} (target {args.edge:.6f})")
    ratio = avg_spacing / args.edge
    if ratio < 0.7:
        print(f"      [warn] avg_spacing {ratio:.1%} of target — surface may be MUCH denser than interior")
        print(f"             (still much better than shell-mesh, but consider finer --edge)")
    elif 0.8 <= ratio <= 1.3:
        print(f"      [ok] avg_spacing within 20% of target — uniform distribution confirmed")
    else:
        print(f"      [warn] avg_spacing = {ratio:.1%} of target — may indicate non-uniform output")

    # ---- 4. Save as gmsh 4 format ----
    print(f"[4/4] Writing: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = meshio.Mesh(
        points=np.asarray(nodes, dtype=np.float64),
        cells=[('tetra', np.asarray(elems, dtype=np.int32))],
    )
    # file_format='gmsh' → latest (4.x); cpp_sim parser expects MSH 4
    meshio.write(str(output_path), mesh, file_format='gmsh', binary=False)

    print(f"\nDone. Update config/tumor_debulk.yaml:")
    print(f"  mesh: \"{output_path.relative_to(input_path.parent.parent.parent) if str(output_path).startswith(str(input_path.parent.parent.parent)) else output_path}\"")
    print(f"\nExpected cpp_sim startup line:")
    print(f"  加载 tet mesh: {n_verts} 顶点, {n_tets} 四面体")
    print(f"  平均粒子间距: ~{avg_spacing:.6f}")


if __name__ == '__main__':
    main()
