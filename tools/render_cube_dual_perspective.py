"""Dual side-by-side PERSPECTIVE view: per_edge vs global, mesh + edges + damage colormap.

Same as render_cube_dual_topdown.py but with 3D perspective camera
(angled view from front-right-above) instead of straight-down.

Usage: python render_cube_dual_perspective.py <per_edge_dir> <global_dir> <output.mp4>
"""
import struct
from pathlib import Path
import argparse
import numpy as np
import pyvista as pv
import imageio
import imageio_ffmpeg  # noqa


def load_meta(meta_path):
    with open(meta_path, "rb") as f:
        N, n_frames, flags, n_faces = struct.unpack("iiii", f.read(16))
        alpha_low, alpha_high, knife_r = struct.unpack("fff", f.read(12))
        tau_low, tau_high = struct.unpack("ff", f.read(8))
        bbox_min = struct.unpack("fff", f.read(12))
        bbox_max = struct.unpack("fff", f.read(12))
    per_edge = bool(flags & 1)
    per_tau = bool(flags & 2)
    return dict(N=N, n_frames=n_frames, per_edge=per_edge, per_tau=per_tau, n_faces=n_faces,
                 alpha_low=alpha_low, alpha_high=alpha_high, knife_r=knife_r,
                 tau_low=tau_low, tau_high=tau_high,
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


def load_dataset(input_dir):
    input_dir = Path(input_dir)
    meta = load_meta(input_dir / "meta.bin")
    faces = np.fromfile(input_dir / "surface_faces.bin", dtype=np.int32,
                          count=meta["n_faces"] * 3).reshape(meta["n_faces"], 3)
    N = meta["N"]; n_frames = meta["n_frames"]
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
    return dict(meta=meta, faces=faces, pos=all_pos, removed=all_removed,
                 knife=all_knife, damage=all_damage)


def setup_subplot_perspective(plotter, subplot_idx, ds, title):
    """3D perspective view from front-right-above."""
    plotter.subplot(0, subplot_idx)
    plotter.set_background("white")
    plotter.add_text(title, position="upper_edge", font_size=13, color="black")

    bbox_min = ds["meta"]["bbox_min"]; bbox_max = ds["meta"]["bbox_max"]
    cube_center = (bbox_min + bbox_max) / 2
    cube_size = float(np.max(bbox_max - bbox_min))

    # Perspective camera: front-right-above
    # In remapped pyvista coords (x_sim, z_sim, y_sim with y_sim up):
    plotter.camera.position = (
        cube_center[0] + cube_size * 1.3,    # right
        cube_center[2] - cube_size * 1.3,    # front (toward viewer)
        cube_center[1] + cube_size * 1.1,    # above
    )
    plotter.camera.focal_point = (cube_center[0], cube_center[2], cube_center[1])
    plotter.camera.up = (0, 0, 1)            # y_sim points UP in view
    plotter.camera.view_angle = 32


def render_subplot_frame(plotter, subplot_idx, ds, frame_idx, mesh_ref, knife_ref):
    plotter.subplot(0, subplot_idx)

    pos = ds["pos"][frame_idx]
    rem = ds["removed"][frame_idx]
    kn = ds["knife"][frame_idx]
    dmg = ds["damage"][frame_idx]
    faces = ds["faces"]
    knife_r = ds["meta"]["knife_r"]

    if mesh_ref[0] is not None:
        plotter.remove_actor(mesh_ref[0])
    if knife_ref[0] is not None:
        plotter.remove_actor(knife_ref[0])

    face_mask = (rem[faces[:, 0]] == 0) & (rem[faces[:, 1]] == 0) & (rem[faces[:, 2]] == 0)
    active_faces = faces[face_mask]
    verts_pv = np.stack([pos[:, 0], pos[:, 2], pos[:, 1]], axis=1)

    if len(active_faces) > 0:
        faces_pv = np.hstack([np.full((len(active_faces), 1), 3, dtype=np.int32),
                                active_faces]).flatten()
        mesh = pv.PolyData(verts_pv, faces_pv)
        mesh["damage"] = dmg
        mesh_ref[0] = plotter.add_mesh(
            mesh, scalars="damage", cmap="hot",
            clim=[0.0, 0.05],
            show_edges=True, edge_color="gray", line_width=0.3,
            show_scalar_bar=(subplot_idx == 1),
            scalar_bar_args=dict(title="damage (live)", color="black",
                                  vertical=True, position_x=0.92, position_y=0.25,
                                  width=0.04, height=0.5) if subplot_idx == 1 else {},
        )

    knife_pv = (kn[0], kn[2], kn[1])
    knife_ref[0] = plotter.add_mesh(
        pv.Sphere(radius=knife_r, center=knife_pv, theta_resolution=24, phi_resolution=24),
        color="cyan", opacity=0.5,
    )


def render(per_edge_dir, global_dir, output_path, fps=30):
    print("loading per_edge dataset...")
    ds_pe = load_dataset(per_edge_dir)
    print("loading global dataset...")
    ds_gl = load_dataset(global_dir)
    n_frames = min(ds_pe["meta"]["n_frames"], ds_gl["meta"]["n_frames"])
    print(f"both loaded, frames={n_frames}")

    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=[1280, 640])
    # Dynamic title based on what kind of gradient is active
    if ds_pe['meta']['per_tau']:
        title_left = (f"PER-PARTICLE τ [{ds_pe['meta']['tau_low']:.2f} → "
                       f"{ds_pe['meta']['tau_high']:.2f}]  (toughness)")
    elif ds_pe['meta']['per_edge']:
        title_left = (f"PER-EDGE α [{ds_pe['meta']['alpha_low']:.1f} → "
                       f"{ds_pe['meta']['alpha_high']:.1f}]")
    else:
        title_left = "(uniform)"
    setup_subplot_perspective(plotter, 0, ds_pe, title_left)
    setup_subplot_perspective(plotter, 1, ds_gl, "GLOBAL uniform baseline")

    # Phone-compatible: h.264 baseline + faststart for streaming + lower bitrate
    writer = imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", quality=6,
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "+faststart",
                        "-profile:v", "baseline", "-level", "3.0"],
    )

    mesh_pe = [None]; knife_pe = [None]
    mesh_gl = [None]; knife_gl = [None]

    for f_idx in range(n_frames):
        render_subplot_frame(plotter, 0, ds_pe, f_idx, mesh_pe, knife_pe)
        render_subplot_frame(plotter, 1, ds_gl, f_idx, mesh_gl, knife_gl)
        img = plotter.screenshot(return_img=True)
        writer.append_data(img)

        if f_idx % 20 == 0 or f_idx == n_frames - 1:
            kn_pe = ds_pe["knife"][f_idx]
            n_pe = int((ds_pe["removed"][f_idx] == 0).sum())
            n_gl = int((ds_gl["removed"][f_idx] == 0).sum())
            print(f"  frame {f_idx+1:>4d}/{n_frames}: knife_x={kn_pe[0]:.3f}  "
                  f"per_edge_active={n_pe}/{ds_pe['meta']['N']}  "
                  f"global_active={n_gl}")

    writer.close()
    plotter.close()
    print(f"✓ saved: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("per_edge_dir")
    p.add_argument("global_dir")
    p.add_argument("output_path")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    render(args.per_edge_dir, args.global_dir, args.output_path, fps=args.fps)
