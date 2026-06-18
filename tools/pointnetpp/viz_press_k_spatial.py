"""Top-down visualization of recovered k field vs truth (press k spatial recovery).
Loads press_k_spatial.npz; plots surface particles colored by k: truth | recovered | error,
with the indenter scan positions marked. Confirms the left-stiff/right-soft pattern is localized.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
d = np.load(REPO / "tools/pointnetpp/viz_out/press_k_spatial.npz")
pos, surf = d["pos"], d["surf"]; sweep, la = int(d["sweep"]), int(d["la"])
k_truth, k_rec = d["k_truth"], d["k_rec"]; KS, KF = float(d["K_STIFF"]), float(d["K_SOFT"])
sweep_locs = d["sweep_locs"]; pr = float(d["press_r"])
m = surf  # show top surface (where k is observable / column-shared)
x = pos[m, sweep]; y = pos[m, la]
clat = 0.5 * (pos[:, la].min() + pos[:, la].max())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
vmin, vmax = KF - 0.05, KS + 0.05
for ax, vals, title, cmap, vlo, vhi in [
    (axes[0], k_truth[m], "TRUTH k (left stiff / right soft)", "viridis", vmin, vmax),
    (axes[1], k_rec[m], "RECOVERED k (multi-press + grouping)", "viridis", vmin, vmax),
    (axes[2], (k_rec - k_truth)[m], "ERROR (recovered - truth)", "coolwarm", -0.25, 0.25),
]:
    sc = ax.scatter(x, y, c=vals, s=6, cmap=cmap, vmin=vlo, vmax=vhi, linewidths=0)
    ax.set_title(title, fontsize=11); ax.set_aspect("equal")
    ax.set_xlabel(f"sweep axis {sweep}"); ax.set_ylabel(f"lateral axis {la}")
    plt.colorbar(sc, ax=ax, fraction=0.046)
    # mark indenter scan positions (footprints)
    for sx in sweep_locs:
        ax.add_patch(plt.Circle((sx, clat), pr, fill=False, color="red", lw=1.2, ls="--"))
    mid = 0.5 * (pos[:, sweep].min() + pos[:, sweep].max())
    ax.axvline(mid, color="k", lw=0.8, ls=":")
ks_r = k_rec[k_truth > 0.5 * (KS + KF)].mean(); kf_r = k_rec[k_truth < 0.5 * (KS + KF)].mean()
fig.suptitle(f"Press k spatial recovery — stiff {ks_r:.2f} (truth {KS}) | soft {kf_r:.2f} (truth {KF})"
             f"  | red dashed = indenter scan footprints", fontsize=12)
fig.tight_layout()
out = REPO / "tools/pointnetpp/viz_out/press_k_spatial.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved {out}")
