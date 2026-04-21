"""Quick visualization of a tet mesh — surface faces + bbox + tissue layers.

Shows what the nerve / tumor / fixed regions would look like with the same
heuristics as DebulkEnv:
  - fixed: bottom 10% y
  - nerve: next 15% y  (fixed < y <= fixed + 15%)  → y_max = min_y + 25% height
  - tumor: everything else

Usage:
    python tools/visualize_mesh.py --mesh resources/cuboid_flat/cuboid_h040.msh
    python tools/visualize_mesh.py --mesh <...> --save out.png    # offscreen PNG
"""
import argparse
import os
import numpy as np
import meshio
from collections import defaultdict


def extract_surface_faces(tets: np.ndarray) -> np.ndarray:
    face_count = defaultdict(int)
    face_first = {}
    for tet in tets:
        a, b, c, d = int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])
        for face in [(a, c, b), (a, b, d), (a, d, c), (b, c, d)]:
            key = tuple(sorted(face))
            face_count[key] += 1
            if key not in face_first:
                face_first[key] = face
    boundary = [face_first[k] for k, cnt in face_count.items() if cnt == 1]
    return np.asarray(boundary, dtype=np.int32)


def classify_vertices(v: np.ndarray):
    """DebulkEnv-style classification."""
    y_min, y_max = v[:, 1].min(), v[:, 1].max()
    h = y_max - y_min
    fixed_thr = y_min + 0.10 * h
    nerve_thr = y_min + 0.25 * h
    fixed = v[:, 1] < fixed_thr
    nerve = (~fixed) & (v[:, 1] <= nerve_thr)
    tumor = (~fixed) & (~nerve)
    return fixed, nerve, tumor


def face_tissue_code(faces, fixed, nerve, tumor):
    """0=tumor, 1=nerve, 2=fixed-base."""
    codes = np.zeros(len(faces), dtype=np.int32)
    for i, f in enumerate(faces):
        is_n = nerve[f].sum()
        is_f = fixed[f].sum()
        if is_n >= 2:
            codes[i] = 1
        elif is_f >= 2:
            codes[i] = 2
        else:
            codes[i] = 0
    return codes


def _faces_flat(faces_idx):
    if len(faces_idx) == 0:
        return np.array([3, 0, 0, 0], dtype=np.int64)
    n = len(faces_idx)
    return np.hstack(
        [np.full((n, 1), 3, dtype=np.int64), faces_idx.astype(np.int64)]
    ).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--save", default=None, help="PNG path for offscreen render")
    ap.add_argument("--compare", default=None, help="optional 2nd mesh for side-by-side")
    ap.add_argument("--knife", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"), help="draw knife sphere at this position")
    ap.add_argument("--knife2", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"), help="knife position for --compare mesh")
    ap.add_argument("--knife-radius", type=float, default=0.1)
    args = ap.parse_args()

    import pyvista as pv
    if args.save:
        pv.OFF_SCREEN = True

    def load(path):
        m = meshio.read(path)
        v = m.points.astype(np.float32)
        tets = None
        for cb in m.cells:
            if cb.type == "tetra":
                tets = cb.data
                break
        surface = extract_surface_faces(tets)
        fx, nv, tm = classify_vertices(v)
        codes = face_tissue_code(surface, fx, nv, tm)
        return v, tets, surface, codes, (fx.sum(), nv.sum(), tm.sum())

    COLOR_TUMOR = (0.88, 0.45, 0.38)
    COLOR_NERVE = (0.95, 0.15, 0.15)
    COLOR_BASE = (0.55, 0.52, 0.50)
    BG = (0.12, 0.12, 0.15)

    def add_mesh_to(p, v, surface, codes, counts, label, knife_pos=None):
        ft = surface[codes == 0]
        fn = surface[codes == 1]
        fb = surface[codes == 2]
        if len(ft):
            p.add_mesh(pv.PolyData(v, _faces_flat(ft)),
                       color=COLOR_TUMOR, smooth_shading=True,
                       specular=0.35, specular_power=15, ambient=0.25,
                       label=f"tumor ({counts[2]} pts)")
        if len(fn):
            p.add_mesh(pv.PolyData(v, _faces_flat(fn)),
                       color=COLOR_NERVE, smooth_shading=True,
                       specular=0.4, ambient=0.3,
                       label=f"nerve ({counts[1]} pts)")
        if len(fb):
            p.add_mesh(pv.PolyData(v, _faces_flat(fb)),
                       color=COLOR_BASE, smooth_shading=True,
                       specular=0.1, ambient=0.2,
                       label=f"fixed base ({counts[0]} pts)")
        # Wireframe overlay
        wf = pv.PolyData(v, _faces_flat(surface))
        p.add_mesh(wf, color=(0.05, 0.05, 0.05), style='wireframe',
                   line_width=0.3, opacity=0.3)
        # Knife sphere at initial position
        if knife_pos is not None:
            kp = np.asarray(knife_pos, dtype=np.float32)
            knife = pv.Sphere(radius=args.knife_radius, center=kp,
                              theta_resolution=28, phi_resolution=24)
            p.add_mesh(knife, color=(0.25, 0.85, 0.35), opacity=0.55,
                       smooth_shading=True, specular=0.6, specular_power=30)
            # Marker for center
            p.add_mesh(pv.Sphere(radius=0.008, center=kp), color='yellow')
            # Drop a vertical line from knife to mesh top to show entry path
            y_top = v[:, 1].max()
            if kp[1] > y_top:
                entry_line = pv.Line(kp, (kp[0], y_top, kp[2]))
                p.add_mesh(entry_line, color=(0.9, 0.9, 0.2), line_width=2)

        info_line = (
            f"{os.path.basename(label)}\n"
            f"V={len(v):>6}  T={len(surface)}  "
            f"h={v[:,1].max()-v[:,1].min():.2f}"
        )
        if knife_pos is not None:
            kp = np.asarray(knife_pos)
            rel = "INSIDE" if (kp[1] <= v[:, 1].max() and kp[1] >= v[:, 1].min()) else "ABOVE mesh"
            info_line += f"\nknife@({kp[0]:.2f},{kp[1]:.2f},{kp[2]:.2f}) [{rel}]"
        p.add_text(info_line, position='upper_left', font_size=12, color='white')
        p.add_legend(size=(0.20, 0.15), loc="lower left", bcolor=(0.1, 0.1, 0.15))

    if args.compare:
        p = pv.Plotter(shape=(1, 2), window_size=(1800, 800))
        p.subplot(0, 0)
        p.background_color = BG
        v1, t1, s1, c1, cnt1 = load(args.mesh)
        add_mesh_to(p, v1, s1, c1, cnt1, args.mesh, knife_pos=args.knife)
        p.show_grid(color='gray')
        p.subplot(0, 1)
        p.background_color = BG
        v2, t2, s2, c2, cnt2 = load(args.compare)
        add_mesh_to(p, v2, s2, c2, cnt2, args.compare, knife_pos=args.knife2 or args.knife)
        p.show_grid(color='gray')
        p.link_views()
        p.camera_position = 'iso'
    else:
        p = pv.Plotter(window_size=(1280, 720))
        p.background_color = BG
        v1, t1, s1, c1, cnt1 = load(args.mesh)
        add_mesh_to(p, v1, s1, c1, cnt1, args.mesh, knife_pos=args.knife)
        p.show_grid(color='gray', font_size=10)
        p.camera_position = 'iso'

    if args.save:
        p.screenshot(args.save, transparent_background=False)
        print(f"[saved] {args.save}")
    else:
        print("[viewer] opening interactive window — close to exit")
        p.show()


if __name__ == "__main__":
    main()
