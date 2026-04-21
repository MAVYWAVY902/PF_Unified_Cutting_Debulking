"""Interactive 3D visualization of knife tip position relative to tumor + nerve.

Shows the spatial relationship between:
  - Tumor (semi-transparent salmon surface, so you can see nerve through it)
  - Nerve (bright red, fully opaque)
  - Fixed base (gray)
  - Knife sphere (translucent green, at `knife_start` position)

Plus three sliders to interactively move the knife and see where it ends up
relative to tumor / nerve. Info panel shows:
  - knife inside/outside mesh bbox
  - knife depth below top surface
  - min distance from knife-surface to nerve region
  - whether knife sphere overlaps nerve particles

This is a PURE visualization script — no warp, no gym, no env build.
Reads the .msh file directly and applies the same nerve/fixed heuristics
as DebulkEnv for consistency.

Usage:
    # Flat cuboid with suggested knife position
    python tools/visualize_knife_tissue.py \\
        --mesh resources/cuboid_flat/cuboid_h040.msh \\
        --knife 0.9 0.38 0.5 --knife-radius 0.1

    # Original cube with its config
    python tools/visualize_knife_tissue.py \\
        --mesh resources/cube/cube_dense.msh \\
        --knife 0.95 0.7 0.5 --knife-radius 0.1

    # Load knife start directly from a config YAML
    python tools/visualize_knife_tissue.py \\
        --mesh resources/cuboid_flat/cuboid_h040.msh \\
        --config warp_sim/config/debulk_ppo.yaml

    # Offscreen PNG (if no display)
    python tools/visualize_knife_tissue.py \\
        --mesh resources/cuboid_flat/cuboid_h040.msh \\
        --knife 0.9 0.38 0.5 --save out.png
"""
import argparse
import os
from collections import defaultdict
import numpy as np
import meshio
import yaml


# Colors (match cpp_sim/render/viewer.h palette)
COLOR_TUMOR = (0.88, 0.45, 0.38)
COLOR_NERVE = (0.95, 0.15, 0.15)
COLOR_BASE = (0.55, 0.52, 0.50)
COLOR_KNIFE = (0.25, 0.85, 0.35)
COLOR_BG = (0.12, 0.12, 0.15)


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
    return np.asarray(
        [face_first[k] for k, cnt in face_count.items() if cnt == 1],
        dtype=np.int32,
    )


def classify_vertices(v: np.ndarray, nerve_y_max: float = None):
    """Match DebulkEnv heuristics:
      fixed: y < min_y + 10% height
      nerve: non-fixed AND y <= min_y + 25% height (default)
      tumor: everything else
    """
    y_min, y_max = v[:, 1].min(), v[:, 1].max()
    h = y_max - y_min
    fixed_thr = y_min + 0.10 * h
    nerve_thr = nerve_y_max if nerve_y_max is not None else y_min + 0.25 * h
    fixed = v[:, 1] < fixed_thr
    nerve = (~fixed) & (v[:, 1] <= nerve_thr)
    tumor = (~fixed) & (~nerve)
    return fixed, nerve, tumor, (fixed_thr, nerve_thr)


def face_tissue_code(faces, fixed, nerve):
    codes = np.zeros(len(faces), dtype=np.int32)
    for i, f in enumerate(faces):
        is_n = nerve[f].sum()
        is_f = fixed[f].sum()
        if is_n >= 2:
            codes[i] = 1
        elif is_f >= 2:
            codes[i] = 2
    return codes


def _faces_flat(faces_idx):
    if len(faces_idx) == 0:
        return np.array([3, 0, 0, 0], dtype=np.int64)
    n = len(faces_idx)
    return np.hstack(
        [np.full((n, 1), 3, dtype=np.int64), faces_idx.astype(np.int64)]
    ).ravel()


def compute_knife_stats(kp, knife_radius, v, fixed, nerve, tumor, bbox):
    """Return a summary dict of knife spatial relations."""
    inside_bbox = (
        bbox[0] - 1e-6 <= kp[0] <= bbox[1] + 1e-6 and
        bbox[2] - 1e-6 <= kp[1] <= bbox[3] + 1e-6 and
        bbox[4] - 1e-6 <= kp[2] <= bbox[5] + 1e-6
    )
    y_top = bbox[3]
    y_bot = bbox[2]
    depth_from_top = y_top - kp[1]  # negative = above mesh
    # Distance knife center to nearest nerve / tumor particle
    if nerve.any():
        d_nerve = np.linalg.norm(v[nerve] - kp, axis=1)
        min_nerve_dist = float(d_nerve.min())
        nerve_particles_inside_knife = int((d_nerve <= knife_radius).sum())
    else:
        min_nerve_dist, nerve_particles_inside_knife = float('inf'), 0

    if tumor.any():
        d_tumor = np.linalg.norm(v[tumor] - kp, axis=1)
        min_tumor_dist = float(d_tumor.min())
        tumor_particles_inside_knife = int((d_tumor <= knife_radius).sum())
    else:
        min_tumor_dist, tumor_particles_inside_knife = float('inf'), 0

    return dict(
        inside_bbox=inside_bbox,
        depth_from_top=depth_from_top,
        min_nerve_dist=min_nerve_dist,
        nerve_in_knife=nerve_particles_inside_knife,
        min_tumor_dist=min_tumor_dist,
        tumor_in_knife=tumor_particles_inside_knife,
        knife_bottom_y=kp[1] - knife_radius,
        knife_top_y=kp[1] + knife_radius,
    )


def format_info(kp, r, stats):
    bot_rel = "(below mesh)" if stats['knife_bottom_y'] < 0 else ""
    nerve_warn = ""
    if stats['nerve_in_knife'] > 0:
        nerve_warn = f"  ⚠ {stats['nerve_in_knife']} NERVE particles inside knife!"
    elif stats['min_nerve_dist'] < r * 1.5:
        nerve_warn = f"  ⚠ nerve only {stats['min_nerve_dist']:.3f} from knife center"
    return (
        f"knife: ({kp[0]:.3f}, {kp[1]:.3f}, {kp[2]:.3f})  r={r:.3f}\n"
        f"  inside bbox: {stats['inside_bbox']}\n"
        f"  depth from top: {stats['depth_from_top']:+.3f}\n"
        f"  knife span y: [{stats['knife_bottom_y']:.3f}, {stats['knife_top_y']:.3f}] {bot_rel}\n"
        f"  nearest tumor: {stats['min_tumor_dist']:.3f}  ({stats['tumor_in_knife']} inside knife)\n"
        f"  nearest nerve: {stats['min_nerve_dist']:.3f}  ({stats['nerve_in_knife']} inside knife){nerve_warn}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help=".msh tet mesh file")
    ap.add_argument("--knife", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="knife position (x y z), overrides --config knife_start")
    ap.add_argument("--knife-radius", type=float, default=None,
                    help="knife radius, overrides --config knife_radius")
    ap.add_argument("--config", default=None,
                    help="optional yaml to read knife_start + knife_radius from")
    ap.add_argument("--save", default=None, help="offscreen PNG render path")
    ap.add_argument("--no-sliders", action="store_true",
                    help="disable knife position sliders (for screenshot)")
    ap.add_argument("--tumor-opacity", type=float, default=0.45,
                    help="tumor surface opacity (0 = invisible, 1 = solid)")
    args = ap.parse_args()

    # --- Resolve config-derived defaults ---
    knife_pos = args.knife
    knife_radius = args.knife_radius
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if knife_pos is None and 'knife_start' in cfg:
            knife_pos = list(cfg['knife_start'])
        if knife_radius is None and 'knife_radius' in cfg:
            knife_radius = float(cfg['knife_radius'])
    if knife_pos is None:
        knife_pos = [0.5, 0.5, 0.5]
    if knife_radius is None:
        knife_radius = 0.1
    knife_pos = np.array(knife_pos, dtype=np.float32)

    # --- Load mesh, classify ---
    print(f"[load] {args.mesh}")
    m = meshio.read(args.mesh)
    v = m.points.astype(np.float32)
    # Apply DebulkEnv's y-offset so floor alignment matches training
    v[:, 1] += -v[:, 1].min() + 0.001
    tets = None
    for cb in m.cells:
        if cb.type == "tetra":
            tets = cb.data
            break
    surface = extract_surface_faces(tets)
    fixed, nerve, tumor, (fixed_thr, nerve_thr) = classify_vertices(v)
    codes = face_tissue_code(surface, fixed, nerve)

    bbox = (v[:, 0].min(), v[:, 0].max(),
            v[:, 1].min(), v[:, 1].max(),
            v[:, 2].min(), v[:, 2].max())
    print(f"[mesh]  V={len(v)}  T={len(tets)}  surface_tri={len(surface)}")
    print(f"[mesh]  bbox y=[{bbox[2]:.3f}, {bbox[3]:.3f}]  "
          f"fixed<{fixed_thr:.3f}  nerve<={nerve_thr:.3f}")
    print(f"[mesh]  counts — fixed:{fixed.sum()}  nerve:{nerve.sum()}  tumor:{tumor.sum()}")

    # --- PyVista setup ---
    import pyvista as pv
    if args.save:
        pv.OFF_SCREEN = True

    p = pv.Plotter(window_size=(1400, 900))
    p.background_color = COLOR_BG
    p.enable_anti_aliasing()

    faces_tumor = surface[codes == 0]
    faces_nerve = surface[codes == 1]
    faces_base = surface[codes == 2]

    if len(faces_tumor):
        p.add_mesh(
            pv.PolyData(v, _faces_flat(faces_tumor)),
            color=COLOR_TUMOR, opacity=args.tumor_opacity,
            smooth_shading=True, specular=0.3, ambient=0.25,
            label=f"tumor ({int(tumor.sum())} particles)",
        )
    if len(faces_nerve):
        p.add_mesh(
            pv.PolyData(v, _faces_flat(faces_nerve)),
            color=COLOR_NERVE, opacity=0.95,
            smooth_shading=True, specular=0.4, ambient=0.3,
            label=f"nerve ({int(nerve.sum())} particles)",
        )
    if len(faces_base):
        p.add_mesh(
            pv.PolyData(v, _faces_flat(faces_base)),
            color=COLOR_BASE, opacity=0.85,
            smooth_shading=True, specular=0.1, ambient=0.2,
            label=f"fixed base ({int(fixed.sum())} particles)",
        )

    # Wireframe overlay for depth perception
    p.add_mesh(
        pv.PolyData(v, _faces_flat(surface)),
        color=(0.02, 0.02, 0.02), style='wireframe',
        line_width=0.2, opacity=0.25,
    )

    # Nerve boundary plane (y = nerve_thr) to show the transition
    x0, x1, y0, y1, z0, z1 = bbox
    plane = pv.Plane(center=((x0+x1)/2, nerve_thr, (z0+z1)/2),
                     direction=(0, 1, 0),
                     i_size=(x1-x0), j_size=(z1-z0))
    p.add_mesh(plane, color=(0.9, 0.3, 0.3), opacity=0.12,
               style='surface', show_edges=False)

    # --- Knife actor (dynamic) ---
    knife_actor_name = "knife"
    traj_actor_name = "knife_center"

    def _rebuild_knife():
        sphere = pv.Sphere(radius=knife_radius, center=knife_pos,
                           theta_resolution=32, phi_resolution=28)
        p.add_mesh(sphere, color=COLOR_KNIFE, opacity=0.55,
                   smooth_shading=True, specular=0.6, specular_power=30,
                   name=knife_actor_name,
                   label=f"knife tip (r={knife_radius:.3f})")
        # Center marker (small yellow ball)
        p.add_mesh(pv.Sphere(radius=max(0.005, knife_radius*0.08),
                             center=knife_pos),
                   color='yellow', name=traj_actor_name)

    def _update_info():
        stats = compute_knife_stats(knife_pos, knife_radius, v, fixed, nerve, tumor, bbox)
        # Update the info text
        txt = format_info(knife_pos, knife_radius, stats)
        p.add_text(txt, position='lower_left', font_size=10,
                   color='white', font='courier', name='info')

    _rebuild_knife()
    _update_info()

    # --- Sliders for knife X/Y/Z ---
    if not args.no_sliders and not args.save:
        def _make_slider_callback(axis):
            def cb(value):
                knife_pos[axis] = float(value)
                _rebuild_knife()
                _update_info()
            return cb

        slider_margin = 0.05
        p.add_slider_widget(
            callback=_make_slider_callback(0),
            rng=[bbox[0] - slider_margin, bbox[1] + slider_margin],
            value=float(knife_pos[0]), title="knife X",
            pointa=(0.05, 0.08), pointb=(0.28, 0.08),
            style='modern', fmt="%.3f",
        )
        p.add_slider_widget(
            callback=_make_slider_callback(1),
            rng=[bbox[2] - slider_margin, bbox[3] + slider_margin],
            value=float(knife_pos[1]), title="knife Y (height)",
            pointa=(0.36, 0.08), pointb=(0.59, 0.08),
            style='modern', fmt="%.3f",
        )
        p.add_slider_widget(
            callback=_make_slider_callback(2),
            rng=[bbox[4] - slider_margin, bbox[5] + slider_margin],
            value=float(knife_pos[2]), title="knife Z",
            pointa=(0.67, 0.08), pointb=(0.90, 0.08),
            style='modern', fmt="%.3f",
        )

    # --- Header text + legend + grid ---
    header = (
        f"{os.path.basename(args.mesh)}   "
        f"bbox y=[{bbox[2]:.2f},{bbox[3]:.2f}]  "
        f"nerve y≤{nerve_thr:.2f}"
    )
    p.add_text(header, position='upper_left', font_size=12, color='white')
    help_txt = (
        "Mouse: drag=orbit  right=zoom  mid=pan    "
        "Tissue: salmon=tumor, red=nerve, gray=base    "
        "Red plane = nerve top"
    )
    p.add_text(help_txt, position='upper_right', font_size=10,
               color=(0.7, 0.7, 0.7))
    p.add_legend(size=(0.22, 0.18), loc='lower right',
                 bcolor=(0.08, 0.08, 0.12))
    p.show_grid(color='gray', font_size=9)
    p.camera_position = 'iso'

    if args.save:
        p.screenshot(args.save, transparent_background=False)
        print(f"[saved] {args.save}")
    else:
        print("[viewer] interactive window open — drag sliders to move knife; close to exit")
        p.show()


if __name__ == "__main__":
    main()
