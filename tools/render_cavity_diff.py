"""Render cavity DIFFERENCE between per_edge and global.

Static side-by-side final-frame snapshot:
  LEFT  panel: per_edge cavity (only its removed particles colored red)
  MID   panel: global cavity (only its removed particles colored red)
  RIGHT panel: DIFF — particles only-in-global (= per_edge resisted) in BLUE
                       particles only-in-per_edge in GREEN (~none expected)
                       particles removed by both in GRAY

The 3rd panel is the key visualization: BLUE clusters show where toughness
mode (per_edge) prevented cuts.
"""
import struct
from pathlib import Path
import argparse
import numpy as np
import pyvista as pv


def load_final(dir_path):
    d = Path(dir_path)
    with open(d/"meta.bin", "rb") as f:
        N, nf, pe, nfaces = struct.unpack("iiii", f.read(16))
    raw0 = (d/"frame_0000.bin").read_bytes()
    pos_init = np.frombuffer(raw0[:N*12], dtype=np.float32).reshape(N, 3).copy()
    raw = (d/f"frame_{nf-1:04d}.bin").read_bytes()
    pos_final = np.frombuffer(raw[:N*12], dtype=np.float32).reshape(N, 3).copy()
    rem = np.frombuffer(raw[N*12:N*12+N*4], dtype=np.int32).copy()
    return pos_init, pos_final, rem, N


def main():
    p = argparse.ArgumentParser()
    p.add_argument("per_edge_dir")
    p.add_argument("global_dir")
    p.add_argument("output_path")
    args = p.parse_args()

    pos_pe, pos_pe_f, rem_pe, N = load_final(args.per_edge_dir)
    pos_gl, pos_gl_f, rem_gl, _ = load_final(args.global_dir)

    set_pe = np.where(rem_pe == 1)[0]
    set_gl = np.where(rem_gl == 1)[0]
    only_pe = np.setdiff1d(set_pe, set_gl)
    only_gl = np.setdiff1d(set_gl, set_pe)
    both = np.intersect1d(set_pe, set_gl)
    print(f"per_edge: {len(set_pe)}, global: {len(set_gl)}, "
          f"both: {len(both)}, only_pe: {len(only_pe)}, only_gl: {len(only_gl)}")

    # Pyvista coords: (x_sim, z_sim, y_sim)
    def pv_coords(positions):
        return np.stack([positions[:, 0], positions[:, 2], positions[:, 1]], axis=1)

    plotter = pv.Plotter(off_screen=True, shape=(1, 3), window_size=[1920, 700])

    for i in range(3):
        plotter.subplot(0, i)
        plotter.set_background("white")
        bb_min = pos_pe.min(axis=0); bb_max = pos_pe.max(axis=0)
        center = (bb_min + bb_max) / 2
        size = float((bb_max - bb_min).max())
        plotter.camera.position = (center[0] + size * 1.2,
                                     center[2] - size * 1.2,
                                     center[1] + size * 1.0)
        plotter.camera.focal_point = (center[0], center[2], center[1])
        plotter.camera.up = (0, 0, 1)
        plotter.camera.view_angle = 32

    # PANEL 1: per_edge — show only removed particles in red, rest gray small
    plotter.subplot(0, 0)
    plotter.add_text(f"PER-EDGE α [0.5→20]\n{len(set_pe)} removed",
                      position="upper_edge", font_size=14, color="black")
    intact_pe = np.where(rem_pe == 0)[0]
    if len(intact_pe) > 0:
        plotter.add_mesh(pv.PolyData(pv_coords(pos_pe[intact_pe])),
                          color="lightgray", point_size=2, render_points_as_spheres=False)
    if len(set_pe) > 0:
        plotter.add_mesh(pv.PolyData(pv_coords(pos_pe[set_pe])),
                          color="red", point_size=8, render_points_as_spheres=True)

    # PANEL 2: global
    plotter.subplot(0, 1)
    plotter.add_text(f"GLOBAL α=3\n{len(set_gl)} removed",
                      position="upper_edge", font_size=14, color="black")
    intact_gl = np.where(rem_gl == 0)[0]
    if len(intact_gl) > 0:
        plotter.add_mesh(pv.PolyData(pv_coords(pos_pe[intact_gl])),
                          color="lightgray", point_size=2, render_points_as_spheres=False)
    if len(set_gl) > 0:
        plotter.add_mesh(pv.PolyData(pv_coords(pos_pe[set_gl])),
                          color="red", point_size=8, render_points_as_spheres=True)

    # PANEL 3: DIFF
    plotter.subplot(0, 2)
    plotter.add_text(
        f"DIFF\nBLUE = per_edge RESISTED (only global cut, n={len(only_gl)})\n"
        f"GREEN = only per_edge (n={len(only_pe)})\n"
        f"GRAY = both cut (n={len(both)})",
        position="upper_edge", font_size=11, color="black"
    )
    # Intact-in-both: very faint
    intact_both = np.where((rem_pe == 0) & (rem_gl == 0))[0]
    if len(intact_both) > 0:
        plotter.add_mesh(pv.PolyData(pv_coords(pos_pe[intact_both])),
                          color=(0.9, 0.9, 0.9), point_size=1, render_points_as_spheres=False,
                          opacity=0.3)
    # Both removed: gray spheres
    if len(both) > 0:
        plotter.add_mesh(pv.PolyData(pv_coords(pos_pe[both])),
                          color="gray", point_size=6, render_points_as_spheres=True)
    # Only per_edge: green
    if len(only_pe) > 0:
        plotter.add_mesh(pv.PolyData(pv_coords(pos_pe[only_pe])),
                          color="green", point_size=12, render_points_as_spheres=True)
    # ⭐ Only global (= per_edge resisted): BLUE
    if len(only_gl) > 0:
        plotter.add_mesh(pv.PolyData(pv_coords(pos_pe[only_gl])),
                          color="blue", point_size=12, render_points_as_spheres=True)

    img = plotter.screenshot(return_img=True)
    import imageio
    imageio.imwrite(args.output_path, img)
    plotter.close()
    print(f"saved: {args.output_path}")


if __name__ == "__main__":
    main()
