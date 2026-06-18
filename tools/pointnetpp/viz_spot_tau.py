"""Top-down viz of recovered SPOT τ field vs truth (corr-free point-cloud occupancy).
Confirms the positive contrast number is a genuine spatial DISK, not an artifact."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
d = np.load(REPO / "tools/pointnetpp/viz_out/spot_tau_recovery.npz")
pr, surf = d["pr"], d["surf"]; sweep, la = int(d["sweep"]), int(d["la"])
tau_truth, tau_rec, hard = d["tau_truth"], d["tau_rec"], d["hard"]
m = surf
x = pr[m, sweep]; y = pr[m, la]
ks_r = tau_rec[hard].mean(); kf_r = tau_rec[~hard].mean()
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, vals, title, cmap, vlo, vhi in [
    (axes[0], tau_truth[m], "TRUTH τ (SPOT: hard disk 0.55 / soft 0.15)", "viridis", 0.10, 0.60),
    (axes[1], tau_rec[m], "RECOVERED τ (corr-free point-cloud occupancy)", "viridis", 0.10, 0.60),
    (axes[2], (tau_rec - tau_truth)[m], "ERROR (recovered - truth)", "coolwarm", -0.3, 0.3),
]:
    sc = ax.scatter(x, y, c=vals, s=7, cmap=cmap, vmin=vlo, vmax=vhi, linewidths=0)
    ax.set_title(title, fontsize=11); ax.set_aspect("equal")
    ax.set_xlabel(f"sweep axis {sweep}"); ax.set_ylabel(f"lateral axis {la}")
    plt.colorbar(sc, ax=ax, fraction=0.046)
fig.suptitle(f"SPOT τ corr-free recovery — disk τ={ks_r:.3f} (truth 0.55) | bg τ={kf_r:.3f} (truth 0.15)", fontsize=12)
fig.tight_layout()
out = REPO / "tools/pointnetpp/viz_out/spot_tau_recovery.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved {out}  disk={ks_r:.3f} bg={kf_r:.3f}")
