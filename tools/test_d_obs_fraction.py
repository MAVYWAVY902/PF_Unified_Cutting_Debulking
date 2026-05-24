"""Test D: obs fraction ablation — is surface-only really best?

Re-analyzes existing 7 sim outputs (4 gradient + 3 pattern) with varying
particle observation fractions:
   100%  (all particles)
    80%  (random sample)
    60%  (random sample)
    40%  (random sample)
    20%  (random sample)
   surface only (~13%, deterministic — only boundary vertices)
   random 13% (matched count to surface, but random selection)

Fair comparison needs:
   (1) Multiple random samples per fraction (to estimate variance)
   (2) Compare signal magnitude AND robustness across fractions
   (3) Compare surface-only vs random-with-same-count (controlled test)

Pass criteria for sys-ID viability:
   ✓ Monotonicity still holds (Test α)
   ✓ All 3 pairs still distinguishable (Test β)

Fair "best" criterion:
   max(min pairwise L1) across configs = which obs strategy gives most robust
   discrimination from minimum required obs count.
"""
import sys
from pathlib import Path
import struct
import numpy as np

REPO = Path(__file__).resolve().parents[1]
N_RANDOM_SAMPLES = 5  # per fraction


def load_meta(meta_path):
    with open(meta_path, "rb") as f:
        N, n_frames, flags, n_faces = struct.unpack("iiii", f.read(16))
    return dict(N=N, n_frames=n_frames, n_faces=n_faces)


def load_surface_verts(out_dir):
    meta = load_meta(out_dir / "meta.bin")
    n_faces = meta["n_faces"]
    faces = np.fromfile(out_dir / "surface_faces.bin", dtype=np.int32,
                          count=n_faces * 3).reshape(n_faces, 3)
    return np.unique(faces.flatten()), meta


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


def cavity_per_bin(pos, removed, obs_idx, axis_idx=0, n_bins=10):
    pos_o = pos[obs_idx]; rem_o = removed[obs_idx]
    x = pos_o[:, axis_idx]
    x_min, x_max = x.min(), x.max()
    bins = np.linspace(x_min, x_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(x, bins) - 1, 0, n_bins - 1)
    rem_per_bin = np.zeros(n_bins); tot_per_bin = np.zeros(n_bins)
    np.add.at(rem_per_bin, bin_idx, rem_o)
    np.add.at(tot_per_bin, bin_idx, 1.0)
    return rem_per_bin / np.maximum(tot_per_bin, 1.0)


def cavity_3d(pos, removed, obs_idx, grid_n=5):
    pos_o = pos[obs_idx]; rem_o = removed[obs_idx]
    cavity = np.zeros((grid_n, grid_n, grid_n))
    count = np.zeros((grid_n, grid_n, grid_n))
    bins3 = [np.linspace(pos_o[:,ax].min(), pos_o[:,ax].max(), grid_n + 1) for ax in range(3)]
    for i in range(len(pos_o)):
        ix = min(int((pos_o[i,0] - bins3[0][0]) / (bins3[0][-1] - bins3[0][0]) * grid_n), grid_n-1)
        iy = min(int((pos_o[i,1] - bins3[1][0]) / (bins3[1][-1] - bins3[1][0]) * grid_n), grid_n-1)
        iz = min(int((pos_o[i,2] - bins3[2][0]) / (bins3[2][-1] - bins3[2][0]) * grid_n), grid_n-1)
        cavity[ix, iy, iz] += rem_o[i]
        count[ix, iy, iz] += 1
    return (cavity / np.maximum(count, 1)).flatten()


def evaluate_obs(alpha_dirs, beta_dirs, obs_idx_fn_factory, label, seeds=None):
    """Evaluate Test α monotonicity + Test β pairwise L1 for given obs subset.
    obs_idx_fn_factory(seed) returns indices for that seed."""
    if seeds is None: seeds = [None]
    α_L1_per_seed = []
    β_pairs_per_seed = []
    for seed in seeds:
        obs_idx = obs_idx_fn_factory(seed)
        # Test α
        per_bins = {}
        for tag, gd in alpha_dirs:
            out = REPO / "tools" / f"_test_out_{tag}"
            pos, rem, N = load_final_cavity(out)
            per_bins[tag] = cavity_per_bin(pos, rem, obs_idx, axis_idx=0, n_bins=10)
        base = per_bins["a_uniform"]
        L1s = []
        for tag, gd in alpha_dirs:
            diff = per_bins[tag] - base
            L1s.append((gd, float(np.sum(np.abs(diff)))))
        L1s.sort()
        α_L1_per_seed.append([l for _, l in L1s])
        # Test β
        cms = {}
        for tag in beta_dirs:
            out = REPO / "tools" / f"_test_out_{tag}"
            pos, rem, N = load_final_cavity(out)
            cms[tag] = cavity_3d(pos, rem, obs_idx, grid_n=5)
        pairs = {}
        for i in range(len(beta_dirs)):
            for j in range(i+1, len(beta_dirs)):
                k = f"{beta_dirs[i]} vs {beta_dirs[j]}"
                pairs[k] = float(np.sum(np.abs(cms[beta_dirs[i]] - cms[beta_dirs[j]])))
        β_pairs_per_seed.append(pairs)
    return α_L1_per_seed, β_pairs_per_seed


def main():
    print("="*72)
    print("Test D: obs fraction ablation")
    print("="*72)
    alpha_dirs = [("a_uniform", 0.00), ("a_mild", 0.40),
                   ("a_medium", 0.80), ("a_steep", 1.20)]
    beta_dirs = ["b_xgrad", "b_ygrad", "b_incl"]

    sample_out = REPO / "tools" / "_test_out_a_uniform"
    surface_idx, meta = load_surface_verts(sample_out)
    N = meta["N"]
    n_surf = len(surface_idx)
    surf_frac = n_surf / N
    print(f"\n  Mesh N={N}, surface verts={n_surf} ({surf_frac*100:.1f}%)")

    # Define obs configs
    configs = []
    for frac in [1.0, 0.8, 0.6, 0.4, 0.2]:
        n_obs = int(frac * N)
        def factory(seed, n=n_obs):
            rng = np.random.default_rng(seed if seed is not None else 42)
            return rng.choice(N, size=n, replace=False)
        configs.append((f"random_{int(frac*100)}%", factory, list(range(5))))
    # Surface-only (deterministic)
    surf_factory = lambda seed: surface_idx
    configs.append((f"surface_only ({surf_frac*100:.0f}%)", surf_factory, [None]))
    # Random with same count as surface (controlled comparison)
    def random_match_surface(seed, n=n_surf):
        rng = np.random.default_rng(seed if seed is not None else 42)
        return rng.choice(N, size=n, replace=False)
    configs.append((f"random_{surf_frac*100:.0f}% (match surface count)",
                     random_match_surface, list(range(5))))

    # Evaluate each
    print(f"\n[Running] {len(configs)} obs configurations × {N_RANDOM_SAMPLES} seeds (if random) …\n")

    all_results = {}
    for label, factory, seeds in configs:
        α_L1, β_pairs = evaluate_obs(alpha_dirs, beta_dirs, factory, label, seeds)
        all_results[label] = dict(α=α_L1, β=β_pairs)

    # ------------------------------------------------------------------
    # Test α: monotonicity per config
    # ------------------------------------------------------------------
    print(f"{'='*72}")
    print(f"Test α — Monotonicity check per obs config")
    print(f"{'='*72}\n")
    print(f"  {'config':<30s}  {'grad Δ→L1 (mean over seeds)':>52s}")
    print(f"  {'':30s}  {'0.00':>10s}  {'0.40':>10s}  {'0.80':>10s}  {'1.20':>10s}  {'mono':>5s}")
    for label, _, _ in configs:
        α_L1 = all_results[label]["α"]
        mean_L1 = np.mean(α_L1, axis=0)
        mono = all(mean_L1[i] <= mean_L1[i+1] + 1e-6 for i in range(len(mean_L1)-1))
        print(f"  {label:<30s}  "
              f"{mean_L1[0]:10.4f}  {mean_L1[1]:10.4f}  {mean_L1[2]:10.4f}  {mean_L1[3]:10.4f}  "
              f"{'✓' if mono else '✗':>5s}")

    # ------------------------------------------------------------------
    # Test β: pairwise discrimination per config
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"Test β — Pairwise L1 distances (mean over seeds, ± std)")
    print(f"{'='*72}\n")
    pair_names = list(all_results[configs[0][0]]["β"][0].keys())
    print(f"  {'config':<30s}  {pair_names[0]:>16s}  {pair_names[1]:>16s}  {pair_names[2]:>16s}  {'min':>8s}")
    for label, _, _ in configs:
        β = all_results[label]["β"]
        means = {k: np.mean([d[k] for d in β]) for k in pair_names}
        stds = {k: np.std([d[k] for d in β]) for k in pair_names}
        min_pair = min(means.values())
        line = f"  {label:<30s}  "
        for k in pair_names:
            if len(β) > 1:
                line += f"{means[k]:7.3f}±{stds[k]:6.3f}  "
            else:
                line += f"{means[k]:>15.3f}  "
        line += f"{min_pair:8.3f}"
        print(line)

    # ------------------------------------------------------------------
    # Fair "best" comparison
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"FAIR COMPARISON — surface-only vs random-with-same-count")
    print(f"{'='*72}\n")
    surf_label = f"surface_only ({surf_frac*100:.0f}%)"
    rand_label = f"random_{surf_frac*100:.0f}% (match surface count)"
    surf_β = all_results[surf_label]["β"][0]
    rand_β_means = {k: np.mean([d[k] for d in all_results[rand_label]["β"]]) for k in pair_names}
    rand_β_stds = {k: np.std([d[k] for d in all_results[rand_label]["β"]]) for k in pair_names}
    print(f"  Both observing ~{n_surf} particles (={surf_frac*100:.0f}%):\n")
    print(f"  {'pair':<24s}  {'surface (1 sample)':>20s}  {'random (5 samples mean)':>26s}  {'surface ranked':>14s}")
    for k in pair_names:
        ranked = "TOP" if surf_β[k] > rand_β_means[k] + rand_β_stds[k] else (
                  "BELOW" if surf_β[k] < rand_β_means[k] - rand_β_stds[k] else "WITHIN noise")
        print(f"  {k:<24s}  {surf_β[k]:20.3f}  "
              f"{rand_β_means[k]:7.3f}±{rand_β_stds[k]:6.3f}        {ranked:>14s}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"VERDICT")
    print(f"{'='*72}\n")
    print(f"  All configs pass monotonicity & discriminability?")
    for label, _, _ in configs:
        α_L1 = all_results[label]["α"]
        mean_L1 = np.mean(α_L1, axis=0)
        mono = all(mean_L1[i] <= mean_L1[i+1] + 1e-6 for i in range(len(mean_L1)-1))
        β = all_results[label]["β"]
        β_means = {k: np.mean([d[k] for d in β]) for k in pair_names}
        all_distinct = all(v > 0.5 for v in β_means.values())
        ok = mono and all_distinct
        print(f"    {label:<30s}  mono={'✓' if mono else '✗'}  "
              f"distinct={'✓' if all_distinct else '✗'}  → {'PASS' if ok else 'FAIL'}")

    # Best by minimum pairwise L1 (most robust discrimination)
    best_label = max(configs, key=lambda c: min(
        np.mean([d[k] for d in all_results[c[0]]["β"]]) for k in pair_names
    ))[0]
    print(f"\n  Best obs strategy (max min-pairwise-L1): {best_label}")


if __name__ == "__main__":
    main()
