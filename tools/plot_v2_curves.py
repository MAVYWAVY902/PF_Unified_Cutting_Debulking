#!/usr/bin/env python3
"""3-way training curves comparison for Arm C torus runs.

  • v2 zup (Z-up + NaN guards + stiffness rolled back to 0.6) — current best, 500k
  • v1 zup (Z-up first attempt)                              — crashed @ ~422k with NaN
  • Y-up   (cap_armC_torus_seed0)                             — known-successful baseline

Layout: same 3×3 grid as plot_torus_curves.py for visual continuity.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(logdir, tags):
    ea = EventAccumulator(logdir, size_guidance={"scalars": 100000})
    ea.Reload()
    out = {}
    avail = set(ea.Tags()["scalars"])
    for t in tags:
        if t not in avail:
            out[t] = (np.array([]), np.array([]))
            continue
        events = ea.Scalars(t)
        out[t] = (np.array([e.step for e in events]),
                  np.array([e.value for e in events]))
    return out


TAGS = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/explained_variance",
    "train/value_loss",
    "train/std",
    "train/approx_kl",
    "train/entropy_loss",
    "train/clip_fraction",
    "time/fps",
]

TITLES = {
    "rollout/ep_rew_mean":      "ep_rew_mean (higher = better)",
    "rollout/ep_len_mean":      "ep_len_mean (1024 = no force-term; drop = catastrophic OR NaN-guard)",
    "train/explained_variance": "explained_variance (closer to 1 = better critic)",
    "train/value_loss":         "value_loss (log scale, lower = better)",
    "train/std":                "action std (premature drop to 0 = under-exploration)",
    "train/approx_kl":          "approx_kl (target_kl=0.02; spikes = unstable updates)",
    "train/entropy_loss":       "entropy_loss (more negative = more exploration)",
    "train/clip_fraction":      "clip_fraction (PPO action-clip ratio)",
    "time/fps":                 "fps (sim speed)",
}

REPO = "/home/yunxin/pf-shape-matching"

RUNS = [
    {
        "label": "v2 zup (NaN-guarded, Z-up, stiff=0.6) — 500k DONE",
        "logdir": f"{REPO}/tb_logs/cap_armC_torus_zup_v2_seed0_1",
        "color": "crimson",
        "ls":    "-",
        "lw":    1.6,
        "alpha": 0.95,
    },
    {
        "label": "v1 zup (Z-up first attempt) — CRASHED @ ~422k",
        "logdir": f"{REPO}/tb_logs/cap_armC_torus_zup_seed0_1",
        "color": "darkorange",
        "ls":    "--",
        "lw":    1.2,
        "alpha": 0.65,
    },
    {
        "label": "Y-up torus (baseline)        — 500k DONE",
        "logdir": f"{REPO}/tb_logs/cap_armC_torus_seed0_1",
        "color": "royalblue",
        "ls":    "-",
        "lw":    1.2,
        "alpha": 0.65,
    },
]

# Load all
for r in RUNS:
    if not os.path.isdir(r["logdir"]):
        print(f"  [warn] missing tb dir: {r['logdir']}")
        r["data"] = {t: (np.array([]), np.array([])) for t in TAGS}
    else:
        r["data"] = load_scalars(r["logdir"], TAGS)
        first_xs = next((xs for xs, _ in r["data"].values() if len(xs)), None)
        last_step = first_xs[-1] if first_xs is not None and len(first_xs) else "n/a"
        n_pts = len(first_xs) if first_xs is not None else 0
        print(f"  {r['label']:60s} → {n_pts} pts, last step={last_step}")

# 3×3 grid
fig, axes = plt.subplots(3, 3, figsize=(17, 12))
axes = axes.flatten()

for i, tag in enumerate(TAGS):
    ax = axes[i]
    for r in RUNS:
        xs, ys = r["data"][tag]
        if len(xs) == 0:
            continue
        ax.plot(xs, ys, color=r["color"], linestyle=r["ls"],
                linewidth=r["lw"], alpha=r["alpha"],
                label=r["label"].split(" — ")[0])

    ax.set_title(TITLES.get(tag, tag), fontsize=10)
    ax.set_xlabel("training step", fontsize=8)
    ax.grid(alpha=0.25)
    if tag == "train/value_loss":
        ax.set_yscale("log")
    if tag == "rollout/ep_len_mean":
        ax.axhline(y=1024, c="gray", linestyle=":", alpha=0.5, label="ep_length=1024")
        ax.axhline(y=1024 * 0.95, c="gray", linestyle=":", alpha=0.3)
    if tag == "train/approx_kl":
        ax.axhline(y=0.02, c="gray", linestyle=":", alpha=0.5, label="target_kl=0.02")
    ax.legend(fontsize=7, loc="best", framealpha=0.85)
    ax.tick_params(labelsize=8)

# Mark each run's terminal point in panel 0 (ep_rew_mean)
for r in RUNS:
    xs, ys = r["data"]["rollout/ep_rew_mean"]
    if len(xs):
        axes[0].scatter([xs[-1]], [ys[-1]], c=r["color"], marker="*",
                        s=140, zorder=10, edgecolors="black", linewidths=0.7)

fig.suptitle(
    "Arm C Torus — Training Curves Comparison\n"
    "v2 (red, current 500k) vs v1 zup (orange, crashed) vs Y-up baseline (blue, 500k)\n"
    "All: γ=0.995, λ=0.97, ep_length=1024, PBRS α=50, n_steps=2048",
    fontsize=12, y=1.00,
)
plt.tight_layout()
out = f"{REPO}/tb_logs/v2_vs_v1_vs_yup_curves.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nWrote {out}")
print(f"  size: {os.path.getsize(out) / 1e6:.2f} MB")
