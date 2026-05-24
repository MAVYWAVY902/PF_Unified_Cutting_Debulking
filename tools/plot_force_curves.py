"""Plot knife reaction force time series for per_edge vs global.

Loads dumped binary frames (with fake force at end), generates dual-axis chart:
  - Top: force magnitude vs knife_x (the trajectory position)
  - Bottom: number of particles in contact vs knife_x

Both per_edge and global plotted together for direct comparison.

Hypothesis: toughness (per_edge α gradient) shows asymmetric force profile —
high α (right side) → more resistance → higher force.
"""
import struct
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_force_series(dir_path):
    d = Path(dir_path)
    with open(d/"meta.bin", "rb") as f:
        N, n_frames, per_edge, n_faces = struct.unpack("iiii", f.read(16))

    knife_x = np.zeros(n_frames, dtype=np.float32)
    force_mag = np.zeros(n_frames, dtype=np.float32)
    force_xyz = np.zeros((n_frames, 3), dtype=np.float32)
    n_contact = np.zeros(n_frames, dtype=np.int32)
    n_removed = np.zeros(n_frames, dtype=np.int32)

    for f_idx in range(n_frames):
        raw = (d/f"frame_{f_idx:04d}.bin").read_bytes()
        # Layout: pos[N*12] | removed[N*4] | knife[12] | damage[N*4] | force_buf[16] | n_contact[4]
        off = N*12
        rem = np.frombuffer(raw[off:off+N*4], dtype=np.int32)
        off += N*4
        kn = np.frombuffer(raw[off:off+12], dtype=np.float32)
        off += 12
        off += N*4   # skip damage
        force_buf = np.frombuffer(raw[off:off+16], dtype=np.float32)
        off += 16
        nc = struct.unpack("i", raw[off:off+4])[0]

        knife_x[f_idx] = kn[0]
        force_xyz[f_idx] = force_buf[:3]
        force_mag[f_idx] = force_buf[3]
        n_contact[f_idx] = nc
        n_removed[f_idx] = (rem == 1).sum()

    return dict(knife_x=knife_x, force_mag=force_mag, force_xyz=force_xyz,
                 n_contact=n_contact, n_removed=n_removed, N=N, per_edge=bool(per_edge))


def plot(per_edge_dir, global_dir, output_path):
    pe = load_force_series(per_edge_dir)
    gl = load_force_series(global_dir)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.93, bottom=0.07, hspace=0.20)

    # Force magnitude
    ax = axes[0]
    ax.plot(pe["knife_x"], pe["force_mag"], color="red", linewidth=2,
             label="per-edge α (toughness, LEFT soft → RIGHT tough)")
    ax.plot(gl["knife_x"], gl["force_mag"], color="blue", linewidth=2,
             label="global α (uniform baseline)")
    ax.set_ylabel("|F| on knife (N, fake)", fontsize=11)
    ax.set_title("Knife reaction force along sweep — toughness manifests in FORCE",
                  fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvspan(0, 0.10, alpha=0.05, color="green", label="LOW α region")
    ax.axvspan(0.20, 0.30, alpha=0.05, color="orange", label="HIGH α region")

    # Force components (y = vertical, x = lateral)
    ax = axes[1]
    ax.plot(pe["knife_x"], -pe["force_xyz"][:, 1], color="red", linestyle="-",
             linewidth=1.5, label="per-edge: -F_y (upward reaction)")
    ax.plot(gl["knife_x"], -gl["force_xyz"][:, 1], color="blue", linestyle="-",
             linewidth=1.5, label="global: -F_y")
    ax.plot(pe["knife_x"], pe["force_xyz"][:, 0], color="red", linestyle=":",
             linewidth=1.5, alpha=0.5, label="per-edge: F_x (lateral, sweep direction)")
    ax.plot(gl["knife_x"], gl["force_xyz"][:, 0], color="blue", linestyle=":",
             linewidth=1.5, alpha=0.5, label="global: F_x")
    ax.set_ylabel("Force component (N, fake)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    # Contact count + removed count
    ax = axes[2]
    ax.plot(pe["knife_x"], pe["n_contact"], color="red", linewidth=2,
             label="per-edge: particles in contact")
    ax.plot(gl["knife_x"], gl["n_contact"], color="blue", linewidth=2,
             label="global: particles in contact")
    ax.set_ylabel("# particles in knife sphere", fontsize=11)
    ax.set_xlabel("knife_x (sweep position, m)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate alpha values at left/right
    for ax in axes:
        ax.axvline(0.075, color="green", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.axvline(0.225, color="orange", linestyle="--", alpha=0.3, linewidth=0.8)

    axes[0].text(0.075, axes[0].get_ylim()[1] * 0.95, "α≈5 (soft)",
                  color="green", ha="center", fontsize=10)
    axes[0].text(0.225, axes[0].get_ylim()[1] * 0.95, "α≈15 (tough)",
                  color="orange", ha="center", fontsize=10)

    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output_path}")

    # Print summary stats
    print(f"\n=== Force statistics ===")
    print(f"{'metric':>30s}  {'per_edge':>12s}  {'global':>12s}  {'ratio':>8s}")
    for label, f_pe, f_gl in [
        ("max |F|",                pe["force_mag"].max(), gl["force_mag"].max()),
        ("mean |F| (knife active)", pe["force_mag"][pe["force_mag"] > 0].mean(),
                                     gl["force_mag"][gl["force_mag"] > 0].mean()),
        ("max contact count",      float(pe["n_contact"].max()), float(gl["n_contact"].max())),
        ("total particles removed", float(pe["n_removed"][-1]), float(gl["n_removed"][-1])),
    ]:
        ratio = f_pe / max(f_gl, 1e-9)
        print(f"{label:>30s}  {f_pe:>12.4f}  {f_gl:>12.4f}  {ratio:>7.3f}×")

    # Force on left half vs right half (per_edge only)
    left_mask_pe = (pe["knife_x"] > 0.02) & (pe["knife_x"] < 0.15)
    right_mask_pe = (pe["knife_x"] > 0.15) & (pe["knife_x"] < 0.28)
    left_force_pe = pe["force_mag"][left_mask_pe].mean()
    right_force_pe = pe["force_mag"][right_mask_pe].mean()
    print(f"\n=== Per-edge force asymmetry (left α≈low vs right α≈high) ===")
    print(f"  LEFT half mean |F|: {left_force_pe:.3f}")
    print(f"  RIGHT half mean |F|: {right_force_pe:.3f}")
    print(f"  RIGHT/LEFT ratio: {right_force_pe/max(left_force_pe, 1e-9):.2f}× "
          "(predict > 1 for toughness model)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("per_edge_dir")
    p.add_argument("global_dir")
    p.add_argument("output_path")
    args = p.parse_args()
    plot(args.per_edge_dir, args.global_dir, args.output_path)
