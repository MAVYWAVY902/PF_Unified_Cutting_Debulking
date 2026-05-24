"""Test α + β: rigorous probing of per-particle stiffness sys-ID viability.

Test α (Monotonicity):
   4 configs with progressively narrower stiff gradient (0.85=baseline, 0.7/1.1, 0.5/1.3, 0.3/1.5).
   Quantity: |asymmetry| of cavity (deviation from baseline removed-per-bin).
   Pass: monotonically grows with gradient width.

Test β (Pattern discriminability):
   3 spatial configs, same magnitude (0.3, 1.5):
     x-gradient, y-gradient, center inclusion
   Quantity: pairwise L1 distance between cavity maps.
   Pass: ≥ 2/3 pairs are clearly distinguishable (L1 >> baseline noise).

If Test α monotonic AND Test β discriminable → forward map is invertible-looking,
sys-ID viable. Otherwise → don't proceed to Warp per-particle.
"""
import sys
import subprocess
from pathlib import Path
import struct
import shutil
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "cpp_sim" / "build" / "record_trajectory"
N_FRAMES = 200

CONFIG_TEMPLATE = """## Auto-generated for Test α/β
name: "test_{tag}"
mesh: "../../resources/cube/cube_tet.msh"

simulation:
  dt: 0.005
  gravity: [0, -9.81, 0]
  num_iterations: 6
  stiffness: 0.85
  damping: 0.95
  substeps: 8

boundary:
  floor_y: 0.0
  fix_bottom: true
  fixed_ratio: 0.05

clusters:
  rings: 2

knife:
  radius: 0.025
  cutting_interval: 4
  alpha_min: 7.0
  aspiration_threshold: 0.45
  damage_growth_rate: 0.15
  per_edge_alpha: false
  damage_model: "sharpness"
  per_particle_tau: false
  per_particle_stiffness: {pps}
  stiff_low: {sl}
  stiff_high: {sh}
  stiff_gradient_axis: "{axis}"
  stiff_pattern: "{pattern}"
  stiff_inclusion_radius_frac: 0.25

self_collision:
  enabled: true
  radius: 0.012
  stiffness: 0.8

viewer:
  draw_surface: false
  draw_wireframe: true
  draw_points: true
"""


def write_config(tag, pps, sl, sh, axis="x", pattern="gradient"):
    cfg_path = REPO / "config" / f"_test_{tag}.yaml"
    with open(cfg_path, "w") as f:
        f.write(CONFIG_TEMPLATE.format(
            tag=tag, pps="true" if pps else "false",
            sl=sl, sh=sh, axis=axis, pattern=pattern))
    return cfg_path


def run_sim(cfg_path, out_tag):
    out_dir = REPO / "tools" / f"_test_out_{out_tag}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cmd = [str(RECORD), str(cfg_path.relative_to(REPO / "cpp_sim" / "build" / "..").as_posix()
                              if False else cfg_path), str(out_dir), str(N_FRAMES)]
    # record_trajectory expects relative paths from its cwd (build dir); use absolute for safety
    cmd = [str(RECORD), str(cfg_path), str(out_dir), str(N_FRAMES)]
    result = subprocess.run(cmd, cwd=str(REPO / "cpp_sim" / "build"),
                              capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  ✗ sim failed for {out_tag}:")
        print(result.stderr[-500:])
        return None
    return out_dir


def load_final_cavity(out_dir):
    """Return per-particle removed flag (final frame)."""
    meta_path = out_dir / "meta.bin"
    with open(meta_path, "rb") as f:
        N, n_frames, flags, n_faces = struct.unpack("iiii", f.read(16))
    final_path = out_dir / f"frame_{n_frames-1:04d}.bin"
    raw = np.fromfile(final_path, dtype=np.uint8)
    pos = np.frombuffer(raw[:N * 12], dtype=np.float32).reshape(N, 3).copy()
    off = N * 12
    removed = np.frombuffer(raw[off:off + N * 4], dtype=np.int32).copy()
    return pos, removed.astype(np.float32), N


def cavity_per_bin(pos, removed, axis_idx=0, n_bins=10):
    """Bin removed-fractions along given spatial axis."""
    x = pos[:, axis_idx]
    x_min, x_max = x.min(), x.max()
    bins = np.linspace(x_min, x_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(x, bins) - 1, 0, n_bins - 1)
    rem_per_bin = np.zeros(n_bins)
    tot_per_bin = np.zeros(n_bins)
    np.add.at(rem_per_bin, bin_idx, removed)
    np.add.at(tot_per_bin, bin_idx, 1.0)
    return rem_per_bin / np.maximum(tot_per_bin, 1.0)


def main():
    print("="*72)
    print("Test α + β: per-particle stiffness sys-ID viability")
    print("="*72)

    # ------------------------------------------------------------------
    # Test α: 4 gradient strengths, all x-axis
    # ------------------------------------------------------------------
    alpha_configs = [
        ("a_uniform",   False, 0.85, 0.85, "x", "gradient"),  # baseline
        ("a_mild",      True,  0.70, 1.10, "x", "gradient"),
        ("a_medium",    True,  0.50, 1.30, "x", "gradient"),
        ("a_steep",     True,  0.30, 1.50, "x", "gradient"),
    ]
    print(f"\n[Test α] Gradient strength sweep (4 configs)")
    alpha_results = {}
    for tag, pps, sl, sh, axis, pattern in alpha_configs:
        cfg = write_config(tag, pps, sl, sh, axis, pattern)
        print(f"  Running {tag}: stiff [{sl:.2f}, {sh:.2f}] {axis}-gradient …")
        out = run_sim(cfg, tag)
        if out is None:
            return
        pos, rem, N = load_final_cavity(out)
        per_bin = cavity_per_bin(pos, rem, axis_idx=0, n_bins=10)
        alpha_results[tag] = dict(per_bin=per_bin, total=int(rem.sum()), gradient=sh - sl)

    base_bin = alpha_results["a_uniform"]["per_bin"]
    print(f"\n[Test α] Results: deviation from uniform baseline")
    print(f"  {'tag':<12s}  {'grad Δ':>8s}  {'total rem':>10s}  {'L1 vs base':>11s}  {'asym signal':>12s}")
    for tag, _, sl, sh, _, _ in alpha_configs:
        r = alpha_results[tag]
        bin_diff = r["per_bin"] - base_bin
        L1 = float(np.sum(np.abs(bin_diff)))
        # Signed asymmetry: sum of (left bins diff) - (right bins diff)
        nb = len(bin_diff)
        left_mean = bin_diff[:nb//3].mean()
        right_mean = bin_diff[-nb//3:].mean()
        asym = float(right_mean - left_mean)
        print(f"  {tag:<12s}  {sh-sl:8.2f}  {r['total']:10d}  {L1:11.4f}  {asym:+12.4f}")

    # Monotonicity verdict
    L1s = []
    for tag, _, sl, sh, _, _ in alpha_configs:
        bin_diff = alpha_results[tag]["per_bin"] - base_bin
        L1s.append((sh - sl, float(np.sum(np.abs(bin_diff)))))
    L1s.sort()
    monotonic = all(L1s[i][1] <= L1s[i+1][1] + 1e-6 for i in range(len(L1s)-1))
    print(f"\n  Monotonicity (L1 distance grows with gradient width): "
          f"{'✓ YES' if monotonic else '✗ NO'}")

    # ------------------------------------------------------------------
    # Test β: 3 spatial patterns, same magnitude
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"[Test β] 3 different spatial patterns (same magnitude 0.3-1.5)")
    print(f"{'='*72}")
    beta_configs = [
        ("b_xgrad",     True, 0.30, 1.50, "x", "gradient"),
        ("b_ygrad",     True, 0.30, 1.50, "y", "gradient"),
        ("b_incl",      True, 0.30, 1.50, "x", "inclusion"),
    ]
    beta_results = {}
    for tag, pps, sl, sh, axis, pattern in beta_configs:
        cfg = write_config(tag, pps, sl, sh, axis, pattern)
        print(f"  Running {tag}: stiff [{sl:.2f}, {sh:.2f}] {pattern} ({axis} if grad) …")
        out = run_sim(cfg, tag)
        if out is None:
            return
        pos, rem, N = load_final_cavity(out)
        # 3D voxel grid for cavity (5x5x5 = 125 voxels)
        grid_n = 5
        bins_3d = []
        for ax in range(3):
            x = pos[:, ax]
            bins_3d.append(np.linspace(x.min(), x.max(), grid_n + 1))
        cavity_3d = np.zeros((grid_n, grid_n, grid_n))
        count_3d = np.zeros((grid_n, grid_n, grid_n))
        for i in range(N):
            ix = min(int((pos[i,0] - bins_3d[0][0]) / (bins_3d[0][-1] - bins_3d[0][0]) * grid_n), grid_n-1)
            iy = min(int((pos[i,1] - bins_3d[1][0]) / (bins_3d[1][-1] - bins_3d[1][0]) * grid_n), grid_n-1)
            iz = min(int((pos[i,2] - bins_3d[2][0]) / (bins_3d[2][-1] - bins_3d[2][0]) * grid_n), grid_n-1)
            cavity_3d[ix, iy, iz] += rem[i]
            count_3d[ix, iy, iz] += 1
        cavity_map = cavity_3d / np.maximum(count_3d, 1)
        beta_results[tag] = dict(cavity_map=cavity_map.flatten(), total=int(rem.sum()))

    # Pairwise L1 distances on 3D cavity maps
    tags = [c[0] for c in beta_configs]
    print(f"\n[Test β] Pairwise L1 distances between 3D cavity maps (5×5×5 voxels):")
    print(f"  {'pair':<24s}  {'L1':>8s}")
    pair_L1 = {}
    for i in range(len(tags)):
        for j in range(i+1, len(tags)):
            L1 = float(np.sum(np.abs(beta_results[tags[i]]["cavity_map"]
                                       - beta_results[tags[j]]["cavity_map"])))
            pair_L1[f"{tags[i]} vs {tags[j]}"] = L1
            print(f"  {tags[i]:<10s} vs {tags[j]:<10s}  {L1:8.4f}")
    # Noise floor: re-load same baseline config twice would give 0 (deterministic).
    # If all pairs > some threshold (e.g., 0.1 = >10% voxels differ), patterns are discriminable.
    THRESH = 0.5  # 5×5×5=125 voxels, threshold means avg per-voxel cavity diff > 0.004
    distinct = sum(1 for v in pair_L1.values() if v > THRESH)
    print(f"\n  Distinct pairs (L1 > {THRESH}): {distinct}/{len(pair_L1)}")

    # ------------------------------------------------------------------
    # Combined verdict
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"COMBINED VERDICT")
    print(f"{'='*72}")
    print(f"  Test α (monotonicity): {'✓ PASS' if monotonic else '✗ FAIL'}")
    print(f"  Test β (discriminable patterns): {'✓ PASS' if distinct == len(pair_L1) else '✗ FAIL'}"
          f" ({distinct}/{len(pair_L1)})")
    print()
    if monotonic and distinct == len(pair_L1):
        print(f"  ✓✓ STRONG forward signal — per-particle stiffness IS recoverable.")
        print(f"     → Proceed to Warp per-particle implementation.")
    elif monotonic or distinct >= 2:
        print(f"  ⚠ PARTIAL signal — recoverable but may need more obs / regularization.")
    else:
        print(f"  ✗ WEAK signal — sys-ID likely won't work in this setup.")


if __name__ == "__main__":
    main()
