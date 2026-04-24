#!/usr/bin/env python3
"""Plot PPO training curves for torus Arm C — cross-compare with cuboid Arm C."""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(logdir, tags):
    ea = EventAccumulator(logdir, size_guidance={"scalars": 100000})
    ea.Reload()
    out = {}
    for t in tags:
        if t not in ea.Tags()["scalars"]:
            out[t] = ([], [])
            continue
        events = ea.Scalars(t)
        xs = np.array([e.step for e in events])
        ys = np.array([e.value for e in events])
        out[t] = (xs, ys)
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
    "rollout/ep_rew_mean":      "ep_rew_mean — 越高越好",
    "rollout/ep_len_mean":      "ep_len_mean — 1024 = 健康；下降 = 灾难破口",
    "train/explained_variance": "explained_variance — 越接近 1 越好",
    "train/value_loss":         "value_loss — 越低越好（log 尺度）",
    "train/std":                "std — 太快 →0 = 过早收敛",
    "train/approx_kl":          "approx_kl — <0.02 = 安全",
    "train/entropy_loss":       "entropy_loss — 绝对值大 = 探索足",
    "train/clip_fraction":      "clip_fraction — PPO 动作截断比例",
    "time/fps":                 "fps — 训练速度",
}

torus = load_scalars("/home/yunxin/pf-shape-matching/tb_logs/cap_armC_torus_seed0_1", TAGS)
cuboid = load_scalars("/home/yunxin/pf-shape-matching/tb_logs/cap_armC_diffM_seed0_1", TAGS)

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, tag in enumerate(TAGS):
    ax = axes[i]
    xs_t, ys_t = torus[tag]
    xs_c, ys_c = cuboid[tag]

    if len(xs_t) > 0:
        ax.plot(xs_t, ys_t, c="crimson", linewidth=1.4, alpha=0.85,
                label=f"torus (live, {xs_t[-1]/1000:.0f}k)")
    if len(xs_c) > 0:
        ax.plot(xs_c, ys_c, c="royalblue", linewidth=1.4, alpha=0.55,
                label=f"cuboid (done, 500k)")

    ax.set_title(TITLES.get(tag, tag), fontsize=10)
    ax.set_xlabel("step", fontsize=8)
    ax.grid(alpha=0.25)
    if tag == "train/value_loss":
        ax.set_yscale("log")
    if tag in ("rollout/ep_rew_mean", "rollout/ep_len_mean"):
        ax.axhline(y=1024 if "len" in tag else 0, c="gray", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.tick_params(labelsize=8)

# Mark the torus live endpoint with a star
xs_t, ys_t = torus["rollout/ep_rew_mean"]
if len(xs_t) > 0:
    axes[0].scatter([xs_t[-1]], [ys_t[-1]], c="black", marker="*",
                    s=130, zorder=10, label=f"now: rew={ys_t[-1]:.0f}")
    axes[0].legend(fontsize=8, loc="best")

fig.suptitle(
    "PPO Training Curves — Arm C (L2 tactile, l2_analytic)\n"
    "torus (red, LIVE) vs cuboid (blue, DONE 500k) — torus has γ=0.995, λ=0.97, ep=1024, PBRS α=50",
    fontsize=12, y=1.00,
)
plt.tight_layout()
out = "/home/yunxin/pf-shape-matching/tb_logs/torus_vs_cuboid_curves.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"Wrote {out}")
print(f"  torus latest: step={xs_t[-1] if len(xs_t) else 'n/a'}, rew={ys_t[-1]:.1f if len(ys_t) else 0}")
