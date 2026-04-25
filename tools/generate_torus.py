#!/usr/bin/env python3
"""Generate a volumetric tet mesh of a torus for the ICRA 2027 capsule task.

Why torus:
  The cuboid task (cuboid_h050.msh, 20k particles, 83% interior) lacks
  geometric discriminative power — any motion inside the block debulks
  interior particles, so "intracapsular carving" vs "random wandering" cannot
  be distinguished. A torus forces the policy to follow the centerline
  through a thin tube; any deviation hits the capsule (inner or outer tube
  wall). This exposes the L2 tactile gradient's benefit cleanly.

Geometry (default) — Z-UP convention (2026-04-24 switch from Y-up):
  Ring plane = XY, axis along Z (parallel to gravity so the doughnut lies
  flat and renders flat in matplotlib's default Z-up convention). Center
  (0.5, 0.5, 0.25) matches the new Z-up workspace.
    major_R = 0.35   (center-of-torus -> center-of-tube)
    minor_r = 0.12   (tube radius; ~2.4x knife_radius=0.05)
  Bounding box: [0.03, 0.03, 0.13] to [0.97, 0.97, 0.37].

Pipeline:
  trimesh.creation.torus  ->  surface (tri) mesh (watertight)
  tetgen                 ->  volumetric tet mesh (Delaunay + Steiner points)
  meshio                 ->  write .msh 2.2 ASCII (loader-compatible)
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def generate_torus_surface(major_R, minor_r, center, major_sec, minor_sec):
    """Build a watertight torus triangle mesh (axis = Z, ring in XY plane).

    trimesh's built-in torus already has axis along Z. We keep this convention
    for Z-up physics (gravity = -Z), so the ring lays flat in the XY ground
    plane and renders flat in matplotlib's default view.
    """
    import trimesh

    t = trimesh.creation.torus(
        major_radius=major_R,
        minor_radius=minor_r,
        major_sections=major_sec,
        minor_sections=minor_sec,
    )
    # No rotation — keep trimesh default axis=Z (Z-up convention).
    t.apply_translation(center)

    if not t.is_watertight:
        raise RuntimeError("trimesh torus not watertight — increase sections")

    return np.asarray(t.vertices, dtype=np.float64), np.asarray(t.faces, dtype=np.int32)


def run_tetgen(verts_surf, tris_surf, target_edge_length):
    """Volumetric tet mesh via tetgen Python binding."""
    import tetgen

    tet = tetgen.TetGen(verts_surf, tris_surf)

    # Target tet volume ≈ (edge^3 * sqrt(2) / 12) for a regular tet.
    max_vol = (target_edge_length ** 3) * np.sqrt(2.0) / 12.0

    # switches:
    #   p : input is PLC (surface)
    #   q1.2 : quality mesh (radius/edge ratio <= 1.2)
    #   a{VOL} : volume constraint per tet
    #   Y : preserve input surface (no Steiner points on boundary)
    switches = f"pq1.2a{max_vol:.8f}Y"
    print(f"  [tetgen] switches='{switches}'  (target edge={target_edge_length}, max_vol={max_vol:.2e})")

    # tetgen returns a 4-tuple (nodes, tet_indices, + 2 arrays for bg mesh).
    out = tet.tetrahedralize(switches=switches)
    tet_verts = np.asarray(out[0], dtype=np.float64)
    tet_tets  = np.asarray(out[1], dtype=np.int32)
    return tet_verts, tet_tets


def save_msh2(out_path, verts, tets):
    """Write a format-2.2 ASCII .msh (loader-compatible)."""
    import meshio
    cells = [("tetra", tets)]
    mesh = meshio.Mesh(points=verts, cells=cells)
    meshio.write(out_path, mesh, file_format="gmsh22", binary=False)


def verify_inside_torus(verts, R, r, center):
    """Z-up convention: ring in XY, axis Z. Inside means
       (sqrt((x-cx)^2 + (y-cy)^2) - R)^2 + (z-cz)^2 <= r^2."""
    dx = verts[:, 0] - center[0]
    dy = verts[:, 1] - center[1]
    dz = verts[:, 2] - center[2]
    ring_dist = np.sqrt(dx * dx + dy * dy) - R
    tube_dist = np.sqrt(ring_dist * ring_dist + dz * dz)
    inside = tube_dist <= r + 1e-3
    return inside.mean(), tube_dist.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="resources/torus/torus.msh")
    ap.add_argument("--major-R", type=float, default=0.35)
    ap.add_argument("--minor-r", type=float, default=0.12)
    ap.add_argument("--center", type=float, nargs=3, default=[0.5, 0.5, 0.25])
    ap.add_argument("--major-sections", type=int, default=64,
                    help="trimesh torus major circle resolution")
    ap.add_argument("--minor-sections", type=int, default=32,
                    help="trimesh torus minor circle resolution")
    ap.add_argument("--edge", type=float, default=0.025,
                    help="Target uniform interior tet edge length")
    ap.add_argument("--save-surface-obj", default=None)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Torus mesh generation ===")
    print(f"  major R     = {args.major_R}")
    print(f"  minor r     = {args.minor_r}")
    print(f"  center      = {args.center}")
    print(f"  target edge = {args.edge}  (knife radius=0.05 -> clearance={(args.minor_r-0.05):.3f})")

    # Step 1: surface mesh
    verts_s, tris_s = generate_torus_surface(
        args.major_R, args.minor_r, args.center,
        args.major_sections, args.minor_sections,
    )
    print(f"  [surface] verts={len(verts_s)}, tris={len(tris_s)}")
    bbox_min, bbox_max = verts_s.min(0), verts_s.max(0)
    print(f"  [surface] bbox = {bbox_min} -> {bbox_max}")

    if args.save_surface_obj:
        import trimesh
        trimesh.Trimesh(vertices=verts_s, faces=tris_s).export(args.save_surface_obj)
        print(f"  wrote surface: {args.save_surface_obj}")

    # Step 2: tet mesh
    verts_v, tets_v = run_tetgen(verts_s, tris_s, args.edge)
    print(f"  [volume ] verts={len(verts_v)}, tets={len(tets_v)}")

    # Step 3: sanity-check
    frac_inside, max_dist = verify_inside_torus(verts_v, args.major_R, args.minor_r, args.center)
    print(f"  [verify ] inside-torus fraction = {frac_inside*100:.2f}%  "
          f"(max tube-dist = {max_dist:.4f})")

    # Step 4: save
    save_msh2(str(out_path), verts_v, tets_v)
    print(f"  [save   ] wrote {out_path}")
    print(f"  [size   ] {out_path.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    sys.exit(main() or 0)
