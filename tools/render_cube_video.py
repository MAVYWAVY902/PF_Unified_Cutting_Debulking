"""Render binary frames from record_trajectory into MP4 video.

2D top-down + side view to show debulk pattern clearly.
"""
import sys
import struct
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation, FFMpegWriter
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def load_meta(meta_path):
    with open(meta_path, "rb") as f:
        N, n_frames, per_edge = struct.unpack("iii", f.read(12))
        alpha_low, alpha_high, knife_r = struct.unpack("fff", f.read(12))
        bbox_min = struct.unpack("fff", f.read(12))
        bbox_max = struct.unpack("fff", f.read(12))
    return {
        "N": N, "n_frames": n_frames, "per_edge": bool(per_edge),
        "alpha_low": alpha_low, "alpha_high": alpha_high, "knife_r": knife_r,
        "bbox_min": np.array(bbox_min), "bbox_max": np.array(bbox_max),
    }


def load_alpha(alpha_path, N):
    return np.fromfile(alpha_path, dtype=np.float32, count=N)


def load_frame(frame_path, N):
    raw = np.fromfile(frame_path, dtype=np.uint8)
    expected = N * 3 * 4 + N * 4 + 3 * 4
    assert len(raw) == expected, f"{frame_path}: size {len(raw)} != expected {expected}"
    pos = np.frombuffer(raw[:N * 12], dtype=np.float32).reshape(N, 3).copy()
    removed = np.frombuffer(raw[N * 12:N * 12 + N * 4], dtype=np.int32).copy()
    knife = np.frombuffer(raw[N * 12 + N * 4:], dtype=np.float32).copy()
    return pos, removed, knife


def render_video(input_dir, output_path, fps=20):
    input_dir = Path(input_dir)
    meta = load_meta(input_dir / "meta.bin")
    print(f"meta: N={meta['N']}, frames={meta['n_frames']}, "
          f"per_edge={meta['per_edge']}, α=[{meta['alpha_low']}, {meta['alpha_high']}]")

    alpha = load_alpha(input_dir / "alpha.bin", meta["N"])
    N = meta["N"]
    n_frames = meta["n_frames"]

    all_pos = np.zeros((n_frames, N, 3), dtype=np.float32)
    all_removed = np.zeros((n_frames, N), dtype=np.int32)
    all_knife = np.zeros((n_frames, 3), dtype=np.float32)
    for f in range(n_frames):
        pos, rem, kn = load_frame(input_dir / f"frame_{f:04d}.bin", N)
        all_pos[f] = pos
        all_removed[f] = rem
        all_knife[f] = kn
    print(f"loaded all {n_frames} frames")

    bbox_min = meta["bbox_min"]; bbox_max = meta["bbox_max"]
    knife_r = meta["knife_r"]

    # 2x1 figure: top view (xz) + side view (xy)
    fig, (ax_top, ax_side) = plt.subplots(2, 1, figsize=(10, 11))
    fig.subplots_adjust(left=0.10, right=0.92, top=0.93, bottom=0.06, hspace=0.30)

    # Top view (xz, looking down -y axis): shows debulk left-right pattern
    ax_top.set_xlim(bbox_min[0] - 0.02, bbox_max[0] + 0.02)
    ax_top.set_ylim(bbox_min[2] - 0.02, bbox_max[2] + 0.02)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("X (gradient axis)")
    ax_top.set_ylabel("Z")
    ax_top.set_title("TOP VIEW (looking down)")
    ax_top.add_patch(Rectangle((bbox_min[0], bbox_min[2]),
                                 bbox_max[0] - bbox_min[0],
                                 bbox_max[2] - bbox_min[2],
                                 fill=False, edgecolor="black", linewidth=1))

    # Side view (xy): shows knife penetration depth
    ax_side.set_xlim(bbox_min[0] - 0.02, bbox_max[0] + 0.02)
    ax_side.set_ylim(bbox_min[1] - 0.02, bbox_max[1] + 0.05)
    ax_side.set_aspect("equal")
    ax_side.set_xlabel("X (gradient axis)")
    ax_side.set_ylabel("Y (height)")
    ax_side.set_title("SIDE VIEW (knife penetration visible)")
    ax_side.add_patch(Rectangle((bbox_min[0], bbox_min[1]),
                                  bbox_max[0] - bbox_min[0],
                                  bbox_max[1] - bbox_min[1],
                                  fill=False, edgecolor="black", linewidth=1))

    # Colors
    if meta["per_edge"]:
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=meta["alpha_low"], vmax=meta["alpha_high"])
        colors_full = cmap(norm(alpha))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax_top, ax_side], shrink=0.85,
                             label="α (low=hard to cut → high=easy to cut)",
                             pad=0.02)
        title_prefix = "PER-EDGE α (linear gradient)"
    else:
        colors_full = np.tile([0.5, 0.5, 0.5, 1.0], (N, 1))
        title_prefix = "GLOBAL α (baseline, uniform)"

    # Scatter handles
    top_scatter = ax_top.scatter([], [], s=2, alpha=0.5)
    side_scatter = ax_side.scatter([], [], s=2, alpha=0.5)
    # Knife circles (top + side)
    top_knife = Circle((0, 0), knife_r, fill=True, color="red", alpha=0.6,
                        edgecolor="black", linewidth=1.5)
    side_knife = Circle((0, 0), knife_r, fill=True, color="red", alpha=0.6,
                         edgecolor="black", linewidth=1.5)
    ax_top.add_patch(top_knife)
    ax_side.add_patch(side_knife)

    suptitle = fig.suptitle("", fontsize=13, fontweight="bold")

    def update(frame_idx):
        pos = all_pos[frame_idx]
        rem = all_removed[frame_idx]
        kn = all_knife[frame_idx]
        active = rem == 0
        n_active = int(active.sum())

        suptitle.set_text(
            f"{title_prefix}  |  frame {frame_idx+1}/{n_frames}  |  "
            f"knife_x={kn[0]:.3f}  |  active particles {n_active}/{N} "
            f"({100*n_active/N:.1f}%)"
        )

        # Top view: x vs z
        p_active = pos[active]
        c_active = colors_full[active]
        top_scatter.set_offsets(p_active[:, [0, 2]])
        top_scatter.set_color(c_active)
        top_knife.center = (kn[0], kn[2])

        # Side view: x vs y
        side_scatter.set_offsets(p_active[:, [0, 1]])
        side_scatter.set_color(c_active)
        side_knife.center = (kn[0], kn[1])

        return top_scatter, side_scatter, top_knife, side_knife, suptitle

    anim = FuncAnimation(fig, update, frames=n_frames,
                          interval=1000.0 / fps, blit=False)

    writer = FFMpegWriter(fps=fps, bitrate=4000,
                           extra_args=["-pix_fmt", "yuv420p"])
    print(f"rendering MP4 to {output_path} ...")
    anim.save(str(output_path), writer=writer, dpi=110)
    plt.close(fig)
    print(f"✓ saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_path")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    render_video(args.input_dir, args.output_path, fps=args.fps)
