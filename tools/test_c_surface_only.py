"""Test C: re-run Test α + β with SURFACE-ONLY cavity observation.

Phantom-realistic scenario: camera sees surface, not interior particles.
Question: do monotonicity + discriminability still hold when we only look
at surface particles' removal?

Method:
   Use 7 existing sim outputs (4 gradient + 3 pattern from test α/β).
   Extract surface vertex indices from surface_faces.bin.
   Re-compute cavity metrics using ONLY surface particles' removed flags.

Compare to full-obs results to see how much signal degrades.
"""
import sys
from pathlib import Path
import struct
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load_meta(meta_path):
    with open(meta_path, "rb") as f:
        N, n_frames, flags, n_faces = struct.unpack("iiii", f.read(16))
    return dict(N=N, n_frames=n_frames, n_faces=n_faces)


def load_surface_verts(out_dir):
    meta = load_meta(out_dir / "meta.bin")
    n_faces = meta["n_faces"]
    faces = np.fromfile(out_dir / "surface_faces.bin", dtype=np.int32,
                          count=n_faces * 3).reshape(n_faces, 3)
    surface_verts = np.unique(faces.flatten())
    return surface_verts, meta


def load_final_cavity(out_dir):
    meta = load_meta(out_dir / "meta.bin")
    N = meta["N"]
    n_frames = meta["n_frames"]
    final_path = out_dir / f"frame_{n_frames-1:04d}.bin"
    raw = np.fromfile(final_path, dtype=np.uint8)
    pos = np.frombuffer(raw[:N * 12], dtype=np.float32).reshape(N, 3).copy()
    off = N * 12
    removed = np.frombuffer(raw[off:off + N * 4], dtype=np.int32).copy()
    return pos, removed.astype(np.float32), N


def cavity_per_bin_surface(pos, removed, surface_idx, axis_idx=0, n_bins=10):
    """Bin removed-fractions along axis, restricted to surface particles."""
    pos_s = pos[surface_idx]
    rem_s = removed[surface_idx]
    x = pos_s[:, axis_idx]
    x_min, x_max = x.min(), x.max()
    bins = np.linspace(x_min, x_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(x, bins) - 1, 0, n_bins - 1)
    rem_per_bin = np.zeros(n_bins)
    tot_per_bin = np.zeros(n_bins)
    np.add.at(rem_per_bin, bin_idx, rem_s)
    np.add.at(tot_per_bin, bin_idx, 1.0)
    return rem_per_bin / np.maximum(tot_per_bin, 1.0)


def cavity_3d_voxel_surface(pos, removed, surface_idx, grid_n=5):
    """3D voxel grid of surface-only cavity for pattern discrimination."""
    pos_s = pos[surface_idx]
    rem_s = removed[surface_idx]
    cavity_3d = np.zeros((grid_n, grid_n, grid_n))
    count_3d = np.zeros((grid_n, grid_n, grid_n))
    bins_3d = [np.linspace(pos_s[:,ax].min(), pos_s[:,ax].max(), grid_n + 1) for ax in range(3)]
    for i in range(len(pos_s)):
        ix = min(int((pos_s[i,0] - bins_3d[0][0]) / (bins_3d[0][-1] - bins_3d[0][0]) * grid_n), grid_n-1)
        iy = min(int((pos_s[i,1] - bins_3d[1][0]) / (bins_3d[1][-1] - bins_3d[1][0]) * grid_n), grid_n-1)
        iz = min(int((pos_s[i,2] - bins_3d[2][0]) / (bins_3d[2][-1] - bins_3d[2][0]) * grid_n), grid_n-1)
        cavity_3d[ix, iy, iz] += rem_s[i]
        count_3d[ix, iy, iz] += 1
    return (cavity_3d / np.maximum(count_3d, 1)).flatten()


def main():
    print("="*72)
    print("Test C: surface-only re-analysis of Test α + β")
    print("="*72)

    alpha_dirs = [("a_uniform", 0.00), ("a_mild", 0.40),
                   ("a_medium", 0.80), ("a_steep", 1.20)]
    beta_dirs = ["b_xgrad", "b_ygrad", "b_incl"]

    # Pre-load surface verts (same for all — cube mesh)
    sample_out = REPO / "tools" / "_test_out_a_uniform"
    surface_idx, meta = load_surface_verts(sample_out)
    n_surface = len(surface_idx)
    N = meta["N"]
    print(f"\nSurface particles: {n_surface}/{N} ({n_surface/N*100:.1f}%)")

    # ------------------------------------------------------------------
    # Test α surface-only
    # ------------------------------------------------------------------
    print(f"\n[Test α surface-only] Gradient strength sweep:")
    print(f"  {'tag':<12s}  {'grad Δ':>8s}  {'surf rem':>9s}  {'L1 vs base':>11s}  {'asym':>9s}")
    alpha_results = {}
    for tag, gd in alpha_dirs:
        out_dir = REPO / "tools" / f"_test_out_{tag}"
        pos, rem, N = load_final_cavity(out_dir)
        per_bin = cavity_per_bin_surface(pos, rem, surface_idx, axis_idx=0, n_bins=10)
        alpha_results[tag] = dict(per_bin=per_bin, surf_rem=int(rem[surface_idx].sum()),
                                    gradient=gd)
        # placeholder; print after base computed
    base_bin = alpha_results["a_uniform"]["per_bin"]
    L1s = []
    for tag, gd in alpha_dirs:
        r = alpha_results[tag]
        bin_diff = r["per_bin"] - base_bin
        L1 = float(np.sum(np.abs(bin_diff)))
        nb = len(bin_diff)
        asym = float(bin_diff[-nb//3:].mean() - bin_diff[:nb//3].mean())
        L1s.append((gd, L1))
        print(f"  {tag:<12s}  {gd:8.2f}  {r['surf_rem']:9d}  {L1:11.4f}  {asym:+9.4f}")

    L1s.sort()
    monotonic = all(L1s[i][1] <= L1s[i+1][1] + 1e-6 for i in range(len(L1s)-1))
    print(f"\n  Monotonicity: {'✓ YES' if monotonic else '✗ NO'}")

    # ------------------------------------------------------------------
    # Test β surface-only
    # ------------------------------------------------------------------
    print(f"\n[Test β surface-only] Pattern discrimination:")
    beta_cavity_maps = {}
    for tag in beta_dirs:
        out_dir = REPO / "tools" / f"_test_out_{tag}"
        pos, rem, N = load_final_cavity(out_dir)
        cm = cavity_3d_voxel_surface(pos, rem, surface_idx, grid_n=5)
        beta_cavity_maps[tag] = cm
        print(f"  {tag}: total surface removed = {int(rem[surface_idx].sum())}")

    print(f"\n  Pairwise L1 (surface-only 5×5×5 voxel cavity):")
    pair_L1 = {}
    for i in range(len(beta_dirs)):
        for j in range(i+1, len(beta_dirs)):
            L1 = float(np.sum(np.abs(beta_cavity_maps[beta_dirs[i]]
                                       - beta_cavity_maps[beta_dirs[j]])))
            pair_L1[f"{beta_dirs[i]} vs {beta_dirs[j]}"] = L1
            print(f"    {beta_dirs[i]:<10s} vs {beta_dirs[j]:<10s}  {L1:8.4f}")
    THRESH = 0.5
    distinct = sum(1 for v in pair_L1.values() if v > THRESH)
    print(f"\n  Distinct pairs (L1 > {THRESH}): {distinct}/{len(pair_L1)}")

    # Compare to full-obs (original Test α/β numbers)
    print(f"\n{'─'*72}")
    print(f"FULL-OBS vs SURFACE-ONLY comparison")
    print(f"{'─'*72}")
    print(f"\n  Test α L1 comparison (gradient Δ → L1):")
    full_L1 = {0.00: 0.0000, 0.40: 0.0134, 0.80: 0.0296, 1.20: 0.0608}
    print(f"    {'grad Δ':>8s}  {'full-obs L1':>13s}  {'surface L1':>12s}  {'degrade':>9s}")
    for gd, L1 in L1s:
        full = full_L1.get(gd, 0)
        ratio = L1 / max(full, 1e-12) if full > 0 else float('inf')
        print(f"    {gd:8.2f}  {full:13.4f}  {L1:12.4f}  ×{ratio:5.2f}")

    print(f"\n  Test β L1 comparison:")
    full_β = {"b_xgrad vs b_ygrad": 0.7200, "b_xgrad vs b_incl": 0.7254,
                "b_ygrad vs b_incl": 0.7791}
    print(f"    {'pair':<24s}  {'full L1':>9s}  {'surf L1':>9s}  {'degrade':>9s}")
    for pair, surf_L1 in pair_L1.items():
        full = full_β[pair]
        ratio = surf_L1 / full
        print(f"    {pair:<24s}  {full:9.4f}  {surf_L1:9.4f}  ×{ratio:5.2f}")

    print(f"\n{'='*72}")
    print(f"VERDICT (Test C — surface-only)")
    print(f"{'='*72}")
    print(f"  Monotonicity: {'✓ PASS' if monotonic else '✗ FAIL'}")
    print(f"  Discriminability: {'✓ PASS' if distinct == len(pair_L1) else f'✗ FAIL ({distinct}/{len(pair_L1)})'}")
    if monotonic and distinct == len(pair_L1):
        print(f"\n  ✓✓ Surface-only obs still discriminates patterns.")
        print(f"     → phantom-realistic forward signal CONFIRMED.")
        print(f"     → safe to proceed to Warp per-particle implementation.")
    else:
        print(f"\n  ⚠ Surface-only loses some signal vs full obs.")
        print(f"    May need L_E for interior propagation, or more surface obs.")


if __name__ == "__main__":
    main()
