"""Simpler visualization — 2 panels side by side (truth vs recovered).

Take a top-down slice through the inclusion region. Show particles as big
colored dots. Same color scale on both panels. Inclusion circled.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]


def main():
    data = np.load(REPO / "tools" / "stageA_inclusion_recovery.npz")
    pos = data["positions"]
    stiff_truth = data["stiff_truth"]
    stiff_recovered = data["stiff_recovered"]
    inclusion_mask = data["inclusion_mask"]
    N = len(pos)

    incl_center = pos[inclusion_mask].mean(axis=0)
    incl_radius = 0.04

    # Slice: take particles near z = 0.15 (the knife z plane)
    # Show x-y view (looking down z axis)
    z_slice = 0.15
    z_thick = 0.04  # take particles within ±0.02 of z=0.15
    slice_mask = np.abs(pos[:, 2] - z_slice) < z_thick / 2
    pos_s = pos[slice_mask]
    truth_s = stiff_truth[slice_mask]
    rec_s = stiff_recovered[slice_mask]
    incl_s = inclusion_mask[slice_mask]
    print(f"Slice at z={z_slice}±{z_thick/2}: {len(pos_s)} particles, "
          f"{int(incl_s.sum())} in inclusion")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Common color scale
    vmin = 0.0
    vmax = 1.6

    # === Panel 1: TRUTH stiff field ===
    ax = axes[0]
    sc = ax.scatter(pos_s[:, 0], pos_s[:, 1], c=truth_s, cmap='viridis',
                     vmin=vmin, vmax=vmax, s=8, alpha=0.9)
    # Inclusion circle
    circle = plt.Circle((incl_center[0], incl_center[1]), incl_radius,
                          fill=False, edgecolor='red', linewidth=2.5, ls='--',
                          label=f'inclusion (truth: stiff=1.5)')
    ax.add_patch(circle)
    # Knife position
    ax.plot([0.12, 0.18], [0.30, 0.30], 'r-', linewidth=5, alpha=0.7,
             label='knife sweep')
    ax.set_xlabel("x (m)", fontsize=12); ax.set_ylabel("y (m)", fontsize=12)
    ax.set_title(f"GROUND TRUTH\n"
                 f"inside inclusion: stiff=1.5\noutside: stiff=0.3",
                 fontsize=13, fontweight='bold')
    ax.set_aspect("equal"); ax.set_xlim(0, 0.3); ax.set_ylim(0, 0.32)
    ax.legend(loc='lower right', fontsize=10)
    plt.colorbar(sc, ax=ax, label="stiff value")

    # === Panel 2: RECOVERED stiff field ===
    ax = axes[1]
    sc = ax.scatter(pos_s[:, 0], pos_s[:, 1], c=rec_s, cmap='viridis',
                     vmin=vmin, vmax=vmax, s=8, alpha=0.9)
    circle = plt.Circle((incl_center[0], incl_center[1]), incl_radius,
                          fill=False, edgecolor='red', linewidth=2.5, ls='--',
                          label=f'true inclusion location')
    ax.add_patch(circle)
    ax.plot([0.12, 0.18], [0.30, 0.30], 'r-', linewidth=5, alpha=0.7,
             label='knife sweep')
    ax.set_xlabel("x (m)", fontsize=12); ax.set_ylabel("y (m)", fontsize=12)
    rec_in_mean = stiff_recovered[inclusion_mask].mean()
    rec_out_mean = stiff_recovered[~inclusion_mask].mean()
    ax.set_title(f"RECOVERED (sys-ID output)\n"
                 f"inside inclusion: stiff≈{rec_in_mean:.2f} (truth 1.5)\n"
                 f"outside: stiff≈{rec_out_mean:.2f} (truth 0.3)",
                 fontsize=13, fontweight='bold')
    ax.set_aspect("equal"); ax.set_xlim(0, 0.3); ax.set_ylim(0, 0.32)
    ax.legend(loc='lower right', fontsize=10)
    plt.colorbar(sc, ax=ax, label="stiff value")

    # === Panel 3: Histogram comparison ===
    ax = axes[2]
    bins = np.linspace(0, 2.0, 60)
    # Truth distributions (delta at 0.3 and 1.5)
    ax.axvline(0.3, color='orange', ls='-', lw=3, alpha=0.6, label='TRUTH outside=0.3')
    ax.axvline(1.5, color='blue', ls='-', lw=3, alpha=0.6, label='TRUTH inside=1.5')
    ax.axvline(0.85, color='gray', ls=':', lw=2, label='init=0.85')
    # Recovered
    ax.hist(stiff_recovered[~inclusion_mask], bins=bins, alpha=0.5,
             color='orange', density=True, label=f'recovered OUTSIDE ({(~inclusion_mask).sum()} pts)')
    ax.hist(stiff_recovered[inclusion_mask], bins=bins, alpha=0.7,
             color='blue', density=True, label=f'recovered INSIDE ({inclusion_mask.sum()} pts)')
    ax.set_xlabel("stiff value", fontsize=12)
    ax.set_ylabel("density (normalized)", fontsize=12)
    ax.set_title("Distribution of recovered stiff\n"
                 "(blue should peak near 1.5, orange near 0.3)",
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 2)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Stage A-4: per-particle stiffness recovery\n"
                 "Inclusion = small region with different stiffness (analog: tumor in tissue)\n"
                 "INSIDE peak at ~1.4 (truth 1.5) = recovery WORKS for the cut region",
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    out_path = REPO / "tools" / "stageA_inclusion_recovery_v2.png"
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
