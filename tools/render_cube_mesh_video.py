"""Render cube as 3D perspective mesh video with PyVista.

Loads binary frames + surface_faces from record_trajectory, renders triangulated
cube surface mesh with edges visible, jet colormap on α (per_edge_alpha case)
or solid gray (baseline), knife as red sphere.

Usage: python render_cube_mesh_video.py <input_dir> <output.mp4> [--fps 20]
"""
import struct
from pathlib import Path
import argparse
import numpy as np
import pyvista as pv
import imageio
import imageio_ffmpeg  # noqa: F401 — ensures imageio uses bundled ffmpeg

# Headless rendering using existing display
pv.global_theme.window_size = [1024, 768]


def load_meta(meta_path):
    with open(meta_path, "rb") as f:
        N, n_frames, per_edge, n_faces = struct.unpack("iiii", f.read(16))
        alpha_low, alpha_high, knife_r = struct.unpack("fff", f.read(12))
        bbox_min = struct.unpack("fff", f.read(12))
        bbox_max = struct.unpack("fff", f.read(12))
    return {
        "N": N, "n_frames": n_frames, "per_edge": bool(per_edge), "n_faces": n_faces,
        "alpha_low": alpha_low, "alpha_high": alpha_high, "knife_r": knife_r,
        "bbox_min": np.array(bbox_min), "bbox_max": np.array(bbox_max),
    }


def load_alpha(alpha_path, N):
    return np.fromfile(alpha_path, dtype=np.float32, count=N)


def load_faces(faces_path, n_faces):
    raw = np.fromfile(faces_path, dtype=np.int32, count=n_faces * 3)
    return raw.reshape(n_faces, 3)


def load_frame(frame_path, N):
    raw = np.fromfile(frame_path, dtype=np.uint8)
    pos = np.frombuffer(raw[:N * 12], dtype=np.float32).reshape(N, 3).copy()
    removed = np.frombuffer(raw[N * 12:N * 12 + N * 4], dtype=np.int32).copy()
    knife = np.frombuffer(raw[N * 12 + N * 4:], dtype=np.float32).copy()
    return pos, removed, knife


def render_video(input_dir, output_path, fps=20):
    input_dir = Path(input_dir)
    meta = load_meta(input_dir / "meta.bin")
    print(f"meta: N={meta['N']}, frames={meta['n_frames']}, faces={meta['n_faces']}, "
          f"per_edge={meta['per_edge']}")

    alpha = load_alpha(input_dir / "alpha.bin", meta["N"])
    faces = load_faces(input_dir / "surface_faces.bin", meta["n_faces"])
    N = meta["N"]
    n_frames = meta["n_frames"]

    # Pre-load all frames
    all_pos = np.zeros((n_frames, N, 3), dtype=np.float32)
    all_removed = np.zeros((n_frames, N), dtype=np.int32)
    all_knife = np.zeros((n_frames, 3), dtype=np.float32)
    for f in range(n_frames):
        pos, rem, kn = load_frame(input_dir / f"frame_{f:04d}.bin", N)
        all_pos[f] = pos
        all_removed[f] = rem
        all_knife[f] = kn
    print(f"loaded all {n_frames} frames")

    # PyVista expects faces in format: [3, i0, i1, i2, 3, i0, i1, i2, ...]
    faces_pv = np.hstack([np.full((meta["n_faces"], 1), 3, dtype=np.int32),
                            faces]).flatten()

    bbox_min = meta["bbox_min"]; bbox_max = meta["bbox_max"]
    cube_center = (bbox_min + bbox_max) / 2
    cube_size = float(np.max(bbox_max - bbox_min))

    # Setup off-screen plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1280, 960])
    plotter.set_background("white")
    title = "PER-EDGE α (gradient)" if meta["per_edge"] else "GLOBAL α (uniform)"
    plotter.add_text(title, position="upper_edge", font_size=16, color="black")

    # Camera: perspective view from front-right-up
    cam_pos = (
        cube_center[0] + cube_size * 1.3,
        cube_center[2] - cube_size * 1.3,   # front
        cube_center[1] + cube_size * 0.9,   # above
    )
    plotter.camera.position = cam_pos
    plotter.camera.focal_point = (cube_center[0], cube_center[2], cube_center[1])
    plotter.camera.up = (0, 0, 1)
    plotter.camera.view_angle = 35
    # Note: we re-map (x, y, z) → (x_world, z_world, y_world) so y points UP

    # Frame writer (manual loop)
    out_frames = []
    writer = imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p",
    )

    # Mesh + knife actors (will be replaced each frame)
    mesh_actor = None
    knife_actor = None

    for f_idx in range(n_frames):
        pos = all_pos[f_idx]
        rem = all_removed[f_idx]
        kn = all_knife[f_idx]

        # Remap coords: y is height in sim, but pyvista uses z as up
        # So vertex.xyz_pyvista = (sim.x, sim.z, sim.y)
        verts_pyvista = np.stack([pos[:, 0], pos[:, 2], pos[:, 1]], axis=1)

        # Filter out faces with any removed particle
        face_mask = (rem[faces[:, 0]] == 0) & \
                     (rem[faces[:, 1]] == 0) & \
                     (rem[faces[:, 2]] == 0)
        active_faces = faces[face_mask]
        if len(active_faces) > 0:
            faces_active_pv = np.hstack([
                np.full((len(active_faces), 1), 3, dtype=np.int32),
                active_faces
            ]).flatten()
            mesh = pv.PolyData(verts_pyvista, faces_active_pv)
            mesh["alpha"] = alpha
        else:
            mesh = None

        # Remove old actors
        if mesh_actor is not None:
            plotter.remove_actor(mesh_actor)
        if knife_actor is not None:
            plotter.remove_actor(knife_actor)

        # Add mesh
        if mesh is not None:
            if meta["per_edge"]:
                mesh_actor = plotter.add_mesh(
                    mesh, scalars="alpha", cmap="jet",
                    clim=[meta["alpha_low"], meta["alpha_high"]],
                    show_edges=True, edge_color="black", line_width=0.4,
                    scalar_bar_args={"title": "α (low=hard cut → high=easy cut)",
                                       "color": "black"},
                )
            else:
                mesh_actor = plotter.add_mesh(
                    mesh, color="lightgray",
                    show_edges=True, edge_color="black", line_width=0.4,
                )

        # Knife sphere (red), remap coords
        knife_pyvista = (kn[0], kn[2], kn[1])
        knife_mesh = pv.Sphere(radius=meta["knife_r"], center=knife_pyvista,
                                 theta_resolution=24, phi_resolution=24)
        knife_actor = plotter.add_mesh(knife_mesh, color="red", opacity=0.85)

        # Render and grab image
        img = plotter.screenshot(return_img=True)
        writer.append_data(img)

        if f_idx % 20 == 0 or f_idx == n_frames - 1:
            n_active = int((rem == 0).sum())
            print(f"  frame {f_idx+1:>4d}/{n_frames}: knife_x={kn[0]:.3f}  "
                  f"active particles={n_active}/{N}")

    writer.close()
    plotter.close()
    print(f"✓ saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_path")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    render_video(args.input_dir, args.output_path, fps=args.fps)
