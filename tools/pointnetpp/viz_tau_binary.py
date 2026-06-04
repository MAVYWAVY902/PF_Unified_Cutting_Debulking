"""Binary tau classification viz — what tau_IoU actually measures.

The continuous tau field viz failed because both Liang and Hier have mean
tau ≈ 0.300 (= init), variation is sub-millesimal, invisible at any
reasonable color range.

But tau_IoU > 0 measurement says SOME particles do cross threshold τ<0.3.
This viz shows them directly:
   - Truth: τ<0.3 inside inclusion (target)
   - Method: predicted τ<0.3 anywhere (= "predicted inclusion")
   - Diff: TP green / FN red / FP orange (vs truth)

Also report HISTOGRAM of actual tau values per method to expose how tight
the distributions are.
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))

import warp as wp  # noqa: E402
wp.init()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS  # noqa: E402
from tools.pointnetpp.viz_tau_field import calibrate_for_mesh, MEDIUM_INCL  # noqa: E402


def render(data, mesh_name, out_path):
    pos = data["pos_rest"]
    incl_center = data["incl_center"]; incl_r = data["incl_radius"]
    bbox_min = data["bbox_min"]; bbox_max = data["bbox_max"]
    top_axis = data["top_axis"]
    other = [i for i in range(3) if i != top_axis]
    ax_h, ax_v = other[0], other[1]
    inclusion_mask = data["inclusion_mask"]

    tau_GT = data["tau_truth"]
    tau_L = data["tau_L"]
    tau_H = data["tau_H"]

    print(f"\n[{mesh_name}] tau distribution:")
    for name, tau in [("GT", tau_GT), ("Liang", tau_L), ("Hier", tau_H)]:
        print(f"  {name}: min={tau.min():.6f} max={tau.max():.6f} mean={tau.mean():.6f} std={tau.std():.6f}")
        n_below = (tau < 0.3).sum()
        print(f"        n(τ<0.3)={n_below}/{len(tau)} ({n_below/len(tau)*100:.2f}%)")

    threshold = 0.3
    truth_inside = tau_GT < threshold     # truth inclusion particles
    L_inside = tau_L < threshold           # Liang predicts inclusion-like
    H_inside = tau_H < threshold           # Hier predicts inclusion-like

    # 2D cross-section slab through inclusion center
    slice_val = incl_center[top_axis]
    slice_tol = incl_r * 1.5
    slab = np.abs(pos[:, top_axis] - slice_val) < slice_tol

    fig = plt.figure(figsize=(20, 12))

    # Row 1: cross-section binary classification (TP/FN/FP/TN)
    panels_binary = [
        ("Truth", truth_inside, truth_inside),
        ("Liang vs Truth", L_inside, truth_inside),
        ("Hier+Liang vs Truth", H_inside, truth_inside),
    ]
    for j, (label, pred, truth) in enumerate(panels_binary):
        ax = fig.add_subplot(2, 3, j + 1)
        slab_pos = pos[slab]
        p = pred[slab]
        t = truth[slab]
        if j == 0:
            # Truth panel: red = inclusion, lightblue = healthy
            ax.scatter(slab_pos[~p, ax_h], slab_pos[~p, ax_v], c="lightblue",
                        s=20, edgecolors="none", alpha=0.5, label=f"τ≥0.3 ({(~p).sum()})")
            ax.scatter(slab_pos[p, ax_h], slab_pos[p, ax_v], c="red",
                        s=30, edgecolors="darkred", linewidth=0.5,
                        label=f"τ<0.3 ({p.sum()})")
        else:
            tp = p & t
            fn = ~p & t
            fp = p & ~t
            tn = ~p & ~t
            ax.scatter(slab_pos[tn, ax_h], slab_pos[tn, ax_v], c="lightgray",
                        s=15, edgecolors="none", alpha=0.4, label=f"TN ({tn.sum()})")
            ax.scatter(slab_pos[tp, ax_h], slab_pos[tp, ax_v], c="green",
                        s=28, edgecolors="none", label=f"TP ({tp.sum()})")
            ax.scatter(slab_pos[fn, ax_h], slab_pos[fn, ax_v], c="red",
                        s=30, edgecolors="darkred", linewidth=0.5,
                        label=f"FN ({fn.sum()})")
            ax.scatter(slab_pos[fp, ax_h], slab_pos[fp, ax_v], c="orange",
                        s=30, edgecolors="darkorange", linewidth=0.5,
                        label=f"FP ({fp.sum()})")
            inter = int(tp.sum())
            union = int((p | t).sum())
            iou = 0.0 if union == 0 else inter / union
            ax.text(0.5, 1.05, f"IoU = {iou:.3f}", transform=ax.transAxes,
                     ha="center", fontsize=11, fontweight="bold",
                     bbox=dict(boxstyle="round", fc="yellow", alpha=0.7))
        ax.add_patch(Circle((incl_center[ax_h], incl_center[ax_v]), incl_r,
                              fill=False, color="green", linewidth=2.0, linestyle="--"))
        ax.set_xlim(bbox_min[ax_h] - 0.02 * (bbox_max[ax_h] - bbox_min[ax_h]),
                     bbox_max[ax_h] + 0.02 * (bbox_max[ax_h] - bbox_min[ax_h]))
        ax.set_ylim(bbox_min[ax_v] - 0.02 * (bbox_max[ax_v] - bbox_min[ax_v]),
                     bbox_max[ax_v] + 0.02 * (bbox_max[ax_v] - bbox_min[ax_v]))
        ax.set_xlabel(["x", "y", "z"][ax_h])
        ax.set_ylabel(["x", "y", "z"][ax_v])
        ax.set_aspect("equal")
        ax.set_title(f"{label}\nslab through inclusion {['x','y','z'][top_axis]}={slice_val:.3f}",
                       fontsize=10)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    # Row 2: histograms of tau values
    for j, (tau, name) in enumerate([(tau_GT, "GT"), (tau_L, "Liang"), (tau_H, "Hier")]):
        ax = fig.add_subplot(2, 3, 4 + j)
        # Two histograms: inside vs outside truth inclusion
        inside_taus = tau[inclusion_mask]
        outside_taus = tau[~inclusion_mask]
        # Adaptive bin range
        all_min = min(inside_taus.min(), outside_taus.min())
        all_max = max(inside_taus.max(), outside_taus.max())
        # Stretch slightly
        rng = max(all_max - all_min, 1e-4)
        all_min -= rng * 0.05
        all_max += rng * 0.05
        bins = np.linspace(all_min, all_max, 60)
        ax.hist(outside_taus, bins=bins, alpha=0.6, color="lightblue",
                 label=f"outside inclusion (truth τ={data['tau_truth'][~inclusion_mask].mean():.2f})",
                 density=True)
        ax.hist(inside_taus, bins=bins, alpha=0.6, color="red",
                 label=f"inside inclusion (truth τ={data['tau_truth'][inclusion_mask].mean():.2f})",
                 density=True)
        ax.axvline(0.3, color="black", linestyle="--", linewidth=1.5,
                     label="threshold τ=0.3")
        ax.set_xlabel("τ value"); ax.set_ylabel("density")
        ax.set_title(f"{name} tau distribution\n"
                       f"in_mean={inside_taus.mean():.4f}  out_mean={outside_taus.mean():.4f}  "
                       f"in_std={inside_taus.std():.5f}",
                       fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"{mesh_name}  ·  BINARY tau classification (τ<0.3 vs τ≥0.3) + tau histograms\n"
                  f"Row 1: cross-section slab through inclusion (green dashed circle)\n"
                  f"Row 2: how concentrated are predicted tau values per method",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    device = "cuda:0"
    t0 = time.time()
    for cfg in MESH_CONFIGS:
        print(f"\n{'='*80}\n{cfg['name']}\n{'='*80}", flush=True)
        data = calibrate_for_mesh(cfg, MEDIUM_INCL[cfg["name"]], device,
                                       T_calib=10, N_epochs=15)
        out_path = REPO / "tools" / "pointnetpp" / f"tau_binary_{cfg['name']}.png"
        render(data, cfg["name"], out_path)
        print(f"  saved {out_path}", flush=True)
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
