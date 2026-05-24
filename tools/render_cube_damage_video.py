"""Render cube mesh video with LIVE DAMAGE coloring (vs static α).

Shows how α field affects DAMAGE PROPAGATION (halo width around knife):
  - HIGH α: sharp narrow damage, damage stays close to knife
  - LOW α: wide diffuse damage, damage spreads further from knife

Both per-edge and global cases use the SAME damage colormap → direct visual
comparison shows how α gradient creates ASYMMETRIC HALO shape.

Frame format updated: pos + removed + knife + damage (live per-particle).

Usage: python render_cube_damage_video.py <input_dir> <output.mp4> [--fps 25]
"""
import struct
from pathlib import Path
import argparse
import numpy as np
import pyvista as pv
import imageio
import imageio_ffmpeg  # noqa


pv.global_theme.window_size = [1280, 960]


def load_meta(meta_path):
    with open(meta_path, "rb") as f:
        N, n_frames, per_edge, n_faces = struct.unpack("iiii", f.read(16))
        alpha_low, alpha_high, knife_r = struct.unpack("fff", f.read(12))
        bbox_min = struct.unpack("fff", f.read(12))
        bbox_max = struct.unpack("fff", f.read(12))
    return dict(N=N, n_frames=n_frames, per_edge=bool(per_edge), n_faces=n_faces,
                 alpha_low=alpha_low, alpha_high=alpha_high, knife_r=knife_r,
                 bbox_min=np.array(bbox_min), bbox_max=np.array(bbox_max))


def load_frame_with_damage(frame_path, N):
    raw = np.fromfile(frame_path, dtype=np.uint8)
    pos = np.frombuffer(raw[:N * 12], dtype=np.float32).reshape(N, 3).copy()
    off = N * 12
    removed = np.frombuffer(raw[off:off + N * 4], dtype=np.int32).copy()
    off += N * 4
    knife = np.frombuffer(raw[off:off + 12], dtype=np.float32).copy()
    off += 12
    damage = np.frombuffer(raw[off:off + N * 4], dtype=np.float32).copy()
    return pos, removed, knife, damage


def render(input_dir, output_path, fps=25):
    input_dir = Path(input_dir)
    meta = load_meta(input_dir / "meta.bin")
    print(f"meta: N={meta['N']}, frames={meta['n_frames']}, "
          f"per_edge={meta['per_edge']}, faces={meta['n_faces']}")

    faces = np.fromfile(input_dir / "surface_faces.bin", dtype=np.int32,
                          count=meta["n_faces"] * 3).reshape(meta["n_faces"], 3)
    static_alpha = np.fromfile(input_dir / "alpha.bin", dtype=np.float32, count=meta["N"])

    N = meta["N"]; n_frames = meta["n_frames"]

    # Preload all frames (with damage)
    all_pos = np.zeros((n_frames, N, 3), dtype=np.float32)
    all_removed = np.zeros((n_frames, N), dtype=np.int32)
    all_knife = np.zeros((n_frames, 3), dtype=np.float32)
    all_damage = np.zeros((n_frames, N), dtype=np.float32)
    for f in range(n_frames):
        pos, rem, kn, dmg = load_frame_with_damage(input_dir / f"frame_{f:04d}.bin", N)
        all_pos[f] = pos
        all_removed[f] = rem
        all_knife[f] = kn
        all_damage[f] = dmg
    print(f"loaded {n_frames} frames (max damage across all frames: "
          f"{all_damage.max():.3f})")

    bbox_min = meta["bbox_min"]; bbox_max = meta["bbox_max"]
    cube_center = (bbox_min + bbox_max) / 2
    cube_size = float(np.max(bbox_max - bbox_min))

    plotter = pv.Plotter(off_screen=True, window_size=[1280, 960])
    plotter.set_background("white")
    title_text = (f"PER-EDGE α [{meta['alpha_low']:.1f}, {meta['alpha_high']:.1f}]"
                  if meta["per_edge"]
                  else f"GLOBAL α (uniform)")
    plotter.add_text(title_text, position="upper_edge", font_size=14, color="black")

    # Camera: perspective from front-right-up
    plotter.camera.position = (
        cube_center[0] + cube_size * 1.4,
        cube_center[2] - cube_size * 1.4,
        cube_center[1] + cube_size * 1.0,
    )
    plotter.camera.focal_point = (cube_center[0], cube_center[2], cube_center[1])
    plotter.camera.up = (0, 0, 1)
    plotter.camera.view_angle = 32

    writer = imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
    )

    mesh_actor = None
    knife_actor = None
    arrow_actor = None

    # Damage colormap range: 0 (intact) → 0.05 (faint halo) → 0.5 (broken)
    # Use TIGHT range to visualize sub-threshold halo (where α range matters)
    DMG_MIN, DMG_MAX = 0.0, 0.05

    for f_idx in range(n_frames):
        pos = all_pos[f_idx]
        rem = all_removed[f_idx]
        kn = all_knife[f_idx]
        dmg = all_damage[f_idx]

        verts_pv = np.stack([pos[:, 0], pos[:, 2], pos[:, 1]], axis=1)

        face_mask = (rem[faces[:, 0]] == 0) & (rem[faces[:, 1]] == 0) & (rem[faces[:, 2]] == 0)
        active_faces = faces[face_mask]

        if mesh_actor is not None:
            plotter.remove_actor(mesh_actor)
        if knife_actor is not None:
            plotter.remove_actor(knife_actor)

        if len(active_faces) > 0:
            faces_pv = np.hstack([np.full((len(active_faces), 1), 3, dtype=np.int32),
                                   active_faces]).flatten()
            mesh = pv.PolyData(verts_pv, faces_pv)
            mesh["damage"] = dmg  # color by LIVE damage
            mesh_actor = plotter.add_mesh(
                mesh, scalars="damage", cmap="hot",   # black→red→yellow→white
                clim=[DMG_MIN, DMG_MAX],
                show_edges=True, edge_color="gray", line_width=0.3,
                scalar_bar_args={"title": "damage (live)", "color": "black",
                                   "vertical": True, "position_x": 0.88, "position_y": 0.25,
                                   "width": 0.05, "height": 0.5},
            )

        knife_pv = (kn[0], kn[2], kn[1])
        knife_actor = plotter.add_mesh(
            pv.Sphere(radius=meta["knife_r"], center=knife_pv,
                       theta_resolution=28, phi_resolution=28),
            color="blue", opacity=0.4,
        )

        img = plotter.screenshot(return_img=True)
        writer.append_data(img)

        if f_idx % 20 == 0 or f_idx == n_frames - 1:
            n_active = int((rem == 0).sum())
            n_damaged = int((dmg > 0.1).sum())
            max_dmg = float(dmg.max())
            print(f"  frame {f_idx+1:>4d}/{n_frames}: knife_x={kn[0]:.3f}  "
                  f"active={n_active}/{N}  damaged>0.1 = {n_damaged}  max_dmg={max_dmg:.3f}")

    writer.close()
    plotter.close()
    print(f"✓ saved: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_dir")
    p.add_argument("output_path")
    p.add_argument("--fps", type=int, default=25)
    args = p.parse_args()
    render(args.input_dir, args.output_path, fps=args.fps)
