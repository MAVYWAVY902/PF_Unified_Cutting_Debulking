"""Visualize Stage A-4 inclusion recovery results.

Loads stageA_inclusion_recovery.npz and produces multi-panel figure:
   - 3D scatter: truth vs recovered, colored by stiff
   - 2D cross-section at z = knife z, colored by stiff
   - Histogram: recovered stiff inside vs outside inclusion
   - Knife trajectory marker

Saves as stageA_inclusion_recovery.png.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parents[1]


def main():
    data = np.load(REPO / "tools" / "stageA_inclusion_recovery.npz")
    pos = data["positions"]
    stiff_truth = data["stiff_truth"]
    stiff_recovered = data["stiff_recovered"]
    stiff_init = data["stiff_init"]
    inclusion_mask = data["inclusion_mask"]

    N = len(pos)
    n_incl = int(inclusion_mask.sum())
    print(f"Loaded: N={N}, inclusion={n_incl}")
    print(f"  truth: in={stiff_truth[inclusion_mask].mean():.2f} "
          f"out={stiff_truth[~inclusion_mask].mean():.2f}")
    print(f"  recovered: in={stiff_recovered[inclusion_mask].mean():.3f} "
          f"out={stiff_recovered[~inclusion_mask].mean():.3f}")

    # Knife metadata (from script)
    knife_y = 0.30
    knife_z = 0.15
    knife_x_min, knife_x_max = 0.12, 0.18
    knife_r = 0.04

    # Inclusion center
    incl_center = pos[inclusion_mask].mean(axis=0)
    print(f"  inclusion center: {incl_center}")

    # Cross-section: particles within ±0.02 of z = knife_z
    z_slice_mask = np.abs(pos[:, 2] - knife_z) < 0.02
    pos_slice = pos[z_slice_mask]
    truth_slice = stiff_truth[z_slice_mask]
    rec_slice = stiff_recovered[z_slice_mask]
    incl_slice = inclusion_mask[z_slice_mask]

    fig = plt.figure(figsize=(15, 11))
    cmap = "viridis"
    vmin = min(stiff_truth.min(), stiff_recovered.min(), 0.0)
    vmax = max(stiff_truth.max(), stiff_recovered.max())

    # Panel 1: Cross-section TRUTH (x-y plane at z = knife_z)
    ax1 = fig.add_subplot(2, 3, 1)
    sc1 = ax1.scatter(pos_slice[:, 0], pos_slice[:, 1], c=truth_slice,
                        cmap=cmap, vmin=vmin, vmax=vmax, s=3, alpha=0.8)
    # Knife trajectory marker
    ax1.plot([knife_x_min, knife_x_max], [knife_y, knife_y], 'r-', lw=4, label='knife')
    circle_t = plt.Circle((incl_center[0], incl_center[1]), 0.04,
                            color='cyan', fill=False, lw=2, ls='--', label='inclusion')
    ax1.add_patch(circle_t)
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    ax1.set_title(f"TRUTH — stiff field (z={knife_z}±0.02 slice)\n"
                  f"inclusion: {stiff_truth[inclusion_mask].mean():.2f}, "
                  f"outside: {stiff_truth[~inclusion_mask].mean():.2f}")
    ax1.set_aspect("equal"); ax1.set_xlim(0, 0.3); ax1.set_ylim(0, 0.32)
    ax1.legend(loc='lower right')
    plt.colorbar(sc1, ax=ax1, label="stiff")

    # Panel 2: Cross-section RECOVERED
    ax2 = fig.add_subplot(2, 3, 2)
    sc2 = ax2.scatter(pos_slice[:, 0], pos_slice[:, 1], c=rec_slice,
                        cmap=cmap, vmin=vmin, vmax=vmax, s=3, alpha=0.8)
    ax2.plot([knife_x_min, knife_x_max], [knife_y, knife_y], 'r-', lw=4)
    circle_r = plt.Circle((incl_center[0], incl_center[1]), 0.04,
                            color='cyan', fill=False, lw=2, ls='--')
    ax2.add_patch(circle_r)
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
    ax2.set_title(f"RECOVERED — stiff field\n"
                  f"inclusion: {stiff_recovered[inclusion_mask].mean():.3f}, "
                  f"outside: {stiff_recovered[~inclusion_mask].mean():.3f}")
    ax2.set_aspect("equal"); ax2.set_xlim(0, 0.3); ax2.set_ylim(0, 0.32)
    plt.colorbar(sc2, ax=ax2, label="stiff")

    # Panel 3: Difference map (RECOVERED - TRUTH)
    ax3 = fig.add_subplot(2, 3, 3)
    diff_slice = rec_slice - truth_slice
    abs_max = max(abs(diff_slice.min()), abs(diff_slice.max()))
    sc3 = ax3.scatter(pos_slice[:, 0], pos_slice[:, 1], c=diff_slice,
                        cmap="coolwarm", vmin=-abs_max, vmax=abs_max,
                        s=3, alpha=0.8)
    ax3.plot([knife_x_min, knife_x_max], [knife_y, knife_y], 'r-', lw=4)
    circle_d = plt.Circle((incl_center[0], incl_center[1]), 0.04,
                            color='black', fill=False, lw=2, ls='--')
    ax3.add_patch(circle_d)
    ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)")
    ax3.set_title(f"DIFFERENCE (recovered - truth)\n"
                  f"L1 inside={np.mean(np.abs(rec_slice[incl_slice] - truth_slice[incl_slice])):.3f}, "
                  f"L1 outside={np.mean(np.abs(rec_slice[~incl_slice] - truth_slice[~incl_slice])):.3f}")
    ax3.set_aspect("equal"); ax3.set_xlim(0, 0.3); ax3.set_ylim(0, 0.32)
    plt.colorbar(sc3, ax=ax3, label="Δstiff")

    # Panel 4: Histogram of stiff values, by group
    ax4 = fig.add_subplot(2, 3, 4)
    bins = np.linspace(0, 2.0, 50)
    ax4.hist(stiff_truth[inclusion_mask], bins=bins, alpha=0.5, color='cyan',
              label=f'truth inside ({stiff_truth[inclusion_mask].mean():.2f})', density=True)
    ax4.hist(stiff_recovered[inclusion_mask], bins=bins, alpha=0.5, color='blue',
              label=f'recovered inside ({stiff_recovered[inclusion_mask].mean():.2f})',
              density=True)
    ax4.hist(stiff_recovered[~inclusion_mask], bins=bins, alpha=0.3, color='orange',
              label=f'recovered outside ({stiff_recovered[~inclusion_mask].mean():.2f})',
              density=True)
    ax4.axvline(0.3, color='red', ls=':', lw=1, label='truth outside (0.3)')
    ax4.axvline(1.5, color='cyan', ls=':', lw=1, label='truth inside (1.5)')
    ax4.axvline(0.85, color='gray', ls=':', lw=1, label='init (0.85)')
    ax4.set_xlabel("stiff value")
    ax4.set_ylabel("density")
    ax4.set_title("Stiff distribution by region")
    ax4.legend(fontsize=8)

    # Panel 5: Per-particle scatter (recovered vs truth)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(stiff_truth[~inclusion_mask], stiff_recovered[~inclusion_mask],
                 s=1, c='orange', alpha=0.3, label=f'outside ({(~inclusion_mask).sum()})')
    ax5.scatter(stiff_truth[inclusion_mask], stiff_recovered[inclusion_mask],
                 s=10, c='blue', alpha=0.7, label=f'inside ({inclusion_mask.sum()})')
    lo = min(stiff_truth.min(), stiff_recovered.min())
    hi = max(stiff_truth.max(), stiff_recovered.max())
    ax5.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='perfect (y=x)')
    ax5.set_xlabel("truth stiff")
    ax5.set_ylabel("recovered stiff")
    ax5.set_title("Per-particle recovery quality\n(closer to y=x = better)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Panel 6: 3D scatter with knife position
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    # Subsample for clarity
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(N, size=3000, replace=False)
    # Include all inclusion particles to highlight them
    incl_idx = np.where(inclusion_mask)[0]
    plot_idx = np.unique(np.concatenate([sub_idx, incl_idx]))
    sc6 = ax6.scatter(pos[plot_idx, 0], pos[plot_idx, 1], pos[plot_idx, 2],
                        c=stiff_recovered[plot_idx], cmap=cmap,
                        vmin=vmin, vmax=vmax, s=3, alpha=0.6)
    # Knife sphere positions
    knife_x_pts = np.linspace(knife_x_min, knife_x_max, 5)
    for kx in knife_x_pts:
        ax6.scatter([kx], [knife_y], [knife_z], c='red', s=200, alpha=0.5,
                    marker='o', edgecolors='black')
    ax6.set_xlabel("x"); ax6.set_ylabel("y"); ax6.set_zlabel("z")
    ax6.set_title("3D recovered stiff map (red = knife sweep)")
    plt.colorbar(sc6, ax=ax6, label="stiff", shrink=0.6)

    fig.suptitle(f"Stage A-4 per-particle stiffness recovery\n"
                 f"Inclusion at (0.15, 0.27, 0.15), r=0.04 — directly below knife at y=0.30",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    out_path = REPO / "tools" / "stageA_inclusion_recovery.png"
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
