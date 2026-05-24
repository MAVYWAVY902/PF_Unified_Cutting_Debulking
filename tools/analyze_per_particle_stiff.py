"""Quantitative analysis: cube_per_particle_stiffness vs cube_global_stiffness_baseline.

Forward sensitivity probe (2026-05-24):
   Do per-particle stiffness heterogeneity (LEFT soft 0.3, RIGHT stiff 1.5)
   visibly change the cavity formation vs uniform baseline (stiffness=0.85)?

Metric:
   1. Per-frame removed-particle count along x-axis bins
   2. Spatial cavity distribution (which x-bins have more / fewer removed)
   3. Final-frame asymmetry: do RIGHT-side particles survive more (because stiff)?
"""
import sys
from pathlib import Path
import struct
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_meta(meta_path):
    with open(meta_path, "rb") as f:
        N, n_frames, flags, n_faces = struct.unpack("iiii", f.read(16))
        alpha_low, alpha_high, knife_r = struct.unpack("fff", f.read(12))
        tau_low, tau_high = struct.unpack("ff", f.read(8))
        bbox_min = struct.unpack("fff", f.read(12))
        bbox_max = struct.unpack("fff", f.read(12))
    return dict(N=N, n_frames=n_frames, n_faces=n_faces, knife_r=knife_r,
                 bbox_min=np.array(bbox_min), bbox_max=np.array(bbox_max))


def load_frame(frame_path, N):
    raw = np.fromfile(frame_path, dtype=np.uint8)
    pos = np.frombuffer(raw[:N * 12], dtype=np.float32).reshape(N, 3).copy()
    off = N * 12
    removed = np.frombuffer(raw[off:off + N * 4], dtype=np.int32).copy()
    return pos, removed


def analyze_dir(dirpath, label):
    dirpath = Path(dirpath)
    meta = load_meta(dirpath / "meta.bin")
    N = meta["N"]
    n_frames = meta["n_frames"]
    bbox_min, bbox_max = meta["bbox_min"], meta["bbox_max"]
    print(f"\n=== {label} ===")
    print(f"  N={N}  frames={n_frames}  bbox=[{bbox_min[0]:.3f}, {bbox_max[0]:.3f}] along x")

    # Load first frame to get x positions of all particles (rest)
    pos0, _ = load_frame(dirpath / "frame_0000.bin", N)
    x_rest = pos0[:, 0].copy()

    # Final frame: which particles got removed?
    pos_final, rem_final = load_frame(dirpath / f"frame_{n_frames-1:04d}.bin", N)
    n_removed = int(rem_final.sum())
    print(f"  Final removed: {n_removed}/{N}  ({n_removed/N*100:.2f}%)")

    # Bin removed-counts along x (10 bins)
    n_bins = 10
    x_min, x_max = bbox_min[0], bbox_max[0]
    bins = np.linspace(x_min, x_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(x_rest, bins) - 1, 0, n_bins - 1)
    counts_removed = np.zeros(n_bins, dtype=int)
    counts_total = np.zeros(n_bins, dtype=int)
    for i in range(N):
        counts_total[bin_idx[i]] += 1
        if rem_final[i]:
            counts_removed[bin_idx[i]] += 1
    print(f"  Removed per x-bin (LEFT → RIGHT):")
    print(f"    {'x_range':<16s} {'removed':>10s}  {'total':>8s}  {'pct':>6s}")
    for b in range(n_bins):
        pct = counts_removed[b] / max(counts_total[b], 1) * 100
        print(f"    [{bins[b]:.2f}, {bins[b+1]:.2f}]  "
              f"{counts_removed[b]:10d}  {counts_total[b]:8d}  {pct:5.2f}%")
    return counts_removed, counts_total, n_removed


def main():
    baseline_dir = REPO_ROOT / "tools" / "viz_stiff_baseline"
    perpart_dir  = REPO_ROOT / "tools" / "viz_stiff_perpart"

    c_b, t_b, n_b = analyze_dir(baseline_dir, "BASELINE (uniform stiffness 0.85)")
    c_p, t_p, n_p = analyze_dir(perpart_dir, "PER-PARTICLE (LEFT 0.3 → RIGHT 1.5)")

    print(f"\n{'='*72}")
    print(f"COMPARISON")
    print(f"{'='*72}")
    print(f"\n  Total removed: baseline={n_b}, per-particle={n_p}, "
          f"diff = {n_p - n_b:+d}")
    print(f"\n  Per-bin diff (per_particle - baseline, normalized by bin total %):")
    print(f"    {'x_bin':<5s}  {'baseline %':>11s}  {'per-part %':>11s}  {'Δ %':>8s}  notes")
    n_bins = len(c_b)
    asym_signal = 0
    for b in range(n_bins):
        pct_b = c_b[b] / max(t_b[b], 1) * 100
        pct_p = c_p[b] / max(t_p[b], 1) * 100
        delta = pct_p - pct_b
        note = ""
        if b < n_bins // 3:
            note = "(SOFT side — expect MORE removal)"
        elif b >= 2 * n_bins // 3:
            note = "(STIFF side — expect LESS removal)"
        # Asymmetric signal: positive on soft (left), negative on stiff (right)
        if b < n_bins // 3:
            asym_signal += delta
        elif b >= 2 * n_bins // 3:
            asym_signal -= delta
        print(f"    bin {b}  {pct_b:10.2f}%  {pct_p:10.2f}%  {delta:+7.2f}%  {note}")

    print(f"\n  ASYMMETRY SIGNAL (SOFT-extra + STIFF-deficit): {asym_signal:+.2f}%")
    print(f"  ({'POSITIVE' if asym_signal > 1 else 'NEGATIVE'} = "
          f"{'CLEAR signal' if abs(asym_signal) > 1 else 'WEAK signal'})")

    # Verdict
    print(f"\n{'='*72}")
    print(f"VERDICT")
    print(f"{'='*72}")
    if asym_signal > 1.0:
        print(f"  ✓ Per-particle stiffness produces VISIBLE asymmetric cavity")
        print(f"    Soft side removes more, stiff side resists. Forward signal CONFIRMED.")
        print(f"    → sys-ID for per-particle stiff is worth attempting in Warp.")
    elif abs(asym_signal) < 0.5:
        print(f"  ✗ No visible asymmetric signal.")
        print(f"    Per-particle stiff barely affects cavity in this setup.")
        print(f"    → sys-ID would have no signal. Don't proceed to Warp.")
    else:
        print(f"  ⚠ Marginal signal ({asym_signal:.2f}%).")
        print(f"    May work but expect slow / fragile sys-ID.")


if __name__ == "__main__":
    main()
