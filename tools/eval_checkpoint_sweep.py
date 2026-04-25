#!/usr/bin/env python3
"""Sweep-evaluate all intermediate checkpoints of a PPO run + plot trajectory.

For a training run with run-name {run}, checkpoints are saved as
  checkpoints/{run}/{run}_{N}_steps.zip      + {run}_vecnormalize_{N}_steps.pkl
  checkpoints/{run}/{run}_final.zip          + {run}_final_vecnormalize.pkl

This tool:
  1. Finds every intermediate checkpoint (sorted by step).
  2. Runs N_EPS deterministic eval episodes per checkpoint, calling the same
     `evaluate()` fn as eval_policy.py (so metrics + info-dict path match).
  3. Writes per-step CSVs to results/sweep/{run}/{run}_step{N:06d}.csv.
  4. Writes summary CSV results/sweep/{run}/{run}_summary.csv with
     (step, interior_mean, interior_std, cap_mean, catastrophic_rate, return).
  5. Saves matplotlib multi-panel plot showing task-level progress vs training step.

Use this AFTER training is complete (or any time you want a curve mapping
training progress to actual task performance — interior coverage vs reward).

Note: each checkpoint eval takes ~(episodes × episode_length / fps_eval) seconds.
For torus (ep=1024, fps≈50), 20 eps ≈ 7 min per checkpoint. 10 checkpoints ≈ 70 min.
"""
import argparse
import csv
import glob
import os
import re
import sys
import numpy as np


def find_checkpoints(ckpt_dir, run_name):
    """Return list of (step, ckpt_path, vec_path) sorted by step; also include 'final' last."""
    results = []
    # Intermediate: {run_name}_{N}_steps.zip + {run_name}_vecnormalize_{N}_steps.pkl
    intermediate_pat = re.compile(rf"^{re.escape(run_name)}_(\d+)_steps\.zip$")
    for fn in sorted(os.listdir(ckpt_dir)):
        m = intermediate_pat.match(fn)
        if m:
            step = int(m.group(1))
            ck = os.path.join(ckpt_dir, fn)
            vn = os.path.join(ckpt_dir, f"{run_name}_vecnormalize_{step}_steps.pkl")
            if not os.path.exists(vn):
                print(f"  [warn] no vecnormalize for step {step}, skipping", flush=True)
                continue
            results.append((step, ck, vn))
    results.sort(key=lambda x: x[0])

    # Final
    final_ck = os.path.join(ckpt_dir, f"{run_name}_final.zip")
    final_vn = os.path.join(ckpt_dir, f"{run_name}_final_vecnormalize.pkl")
    if os.path.exists(final_ck) and os.path.exists(final_vn):
        # Label final as (last_step + 1) sentinel OR reuse last step if already present
        last_step = results[-1][0] if results else 0
        # Use last_step as the x coord but tag it distinctly for plot
        results.append((last_step, final_ck, final_vn))  # sentinel — will be dedup'd

    return results


def summarize_csv(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return {
        "n": len(df),
        "interior_mean": df["final_interior_cov"].mean(),
        "interior_std":  df["final_interior_cov"].std(ddof=0),
        "interior_median": df["final_interior_cov"].median(),
        "cap_mean":      df["capsule_dmg_at_80"].mean(),
        "cap_median":    df["capsule_dmg_at_80"].median(),
        "return_mean":   df["return"].mean(),
        "return_std":    df["return"].std(ddof=0),
        "episode_len_mean": df["episode_len"].mean(),
        "catastrophic_rate": df["catastrophic"].mean(),
    }


def plot_trajectory(summary_rows, out_png, title):
    import matplotlib.pyplot as plt

    steps = np.array([r["step"] for r in summary_rows])
    int_m = np.array([r["interior_mean"] for r in summary_rows])
    int_s = np.array([r["interior_std"]  for r in summary_rows])
    cap_m = np.array([r["cap_mean"]      for r in summary_rows])
    ret_m = np.array([r["return_mean"]   for r in summary_rows])
    ret_s = np.array([r["return_std"]    for r in summary_rows])
    epl_m = np.array([r["episode_len_mean"] for r in summary_rows])
    cata  = np.array([r["catastrophic_rate"] for r in summary_rows])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.errorbar(steps / 1000, int_m * 100, yerr=int_s * 100, marker="o",
                c="crimson", capsize=3, label="interior_done (mean ± std)")
    ax.axhline(y=80, c="gray", linestyle="--", alpha=0.5, label="target 80%")
    ax.set_xlabel("training step (k)")
    ax.set_ylabel("interior coverage (%)")
    ax.set_title("Task metric: interior debulk coverage")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(steps / 1000, cap_m * 100, marker="o", c="orange",
            label="capsule breach at end")
    ax.axhline(y=5,  c="gray", linestyle="--", alpha=0.5, label="budget 5%")
    ax.axhline(y=15, c="red",  linestyle="--", alpha=0.5, label="catastrophic 15%")
    ax.set_xlabel("training step (k)")
    ax.set_ylabel("capsule breach (%)")
    ax.set_title("Safety metric: capsule preservation")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.errorbar(steps / 1000, ret_m, yerr=ret_s, marker="o", c="royalblue",
                capsize=3, label="episode return (mean ± std)")
    ax.set_xlabel("training step (k)")
    ax.set_ylabel("episode return")
    ax.set_title("Training proxy: ep_return (also visible in TB)")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(steps / 1000, epl_m, marker="o", c="purple", label="ep_len_mean")
    ax2 = ax.twinx()
    ax2.plot(steps / 1000, cata * 100, marker="s", c="darkred", linestyle=":",
             label="catastrophic rate (%)")
    ax.set_xlabel("training step (k)")
    ax.set_ylabel("episode length (mean)", color="purple")
    ax2.set_ylabel("catastrophic breach rate (%)", color="darkred")
    ax.axhline(y=1024, c="gray", linestyle="--", alpha=0.5)
    ax.set_title("Aggression: ep_len drop ↔ catastrophic breach")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="lower right", fontsize=9)

    fig.suptitle(title, fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"  [plot] wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True, help="e.g. cap_armC_torus_seed0")
    ap.add_argument("--cfg", required=True, help="config yaml used for training")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="default: checkpoints/{run_name}")
    ap.add_argument("--out-dir", default=None,
                    help="default: results/sweep/{run_name}")
    ap.add_argument("--episodes", type=int, default=20,
                    help="Eval episodes per checkpoint (default 20)")
    ap.add_argument("--seed", type=str, default="0")
    ap.add_argument("--arm", default="C")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Don't re-eval a checkpoint if its CSV exists")
    ap.add_argument("--skip-final", action="store_true",
                    help="Skip the *_final.zip checkpoint (if 100-ep eval already done)")
    args = ap.parse_args()

    ckpt_dir = args.checkpoint_dir or f"checkpoints/{args.run_name}"
    out_dir  = args.out_dir or f"results/sweep/{args.run_name}"
    os.makedirs(out_dir, exist_ok=True)

    ckpts = find_checkpoints(ckpt_dir, args.run_name)
    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir}", file=sys.stderr)
        return 1

    # Dedup: if final file duplicates a step, drop the final (intermediate is already there)
    seen_steps = set()
    deduped = []
    for step, ck, vn in ckpts:
        is_final = "final" in os.path.basename(ck)
        if is_final and args.skip_final:
            continue
        deduped.append((step, ck, vn, is_final))

    print(f"=== Sweep-eval {len(deduped)} checkpoints ({args.episodes} eps each) ===", flush=True)
    for step, ck, vn, is_final in deduped:
        label = f"step={step:06d}" + (" (final)" if is_final else "")
        print(f"\n--- {label} ---", flush=True)

    # Lazy import of evaluate (pulls in Warp + SB3 — slow)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from warp_sim.eval_policy import evaluate

    summary = []
    for step, ck, vn, is_final in deduped:
        label = f"{step:06d}" + ("_final" if is_final else "")
        csv_out = os.path.join(out_dir, f"{args.run_name}_step{label}.csv")
        if args.skip_existing and os.path.exists(csv_out):
            print(f"  [skip] {csv_out} exists", flush=True)
        else:
            print(f"\n>>> evaluating {label} ({args.episodes} eps)", flush=True)
            try:
                evaluate(
                    cfg_path=args.cfg,
                    ckpt_path=ck,
                    vecnorm_path=vn,
                    episodes=args.episodes,
                    out_path=csv_out,
                    arm_label=args.arm,
                    seed_label=args.seed,
                )
            except Exception as e:
                print(f"  [error] {e}", flush=True)
                continue
        try:
            s = summarize_csv(csv_out)
        except Exception as e:
            print(f"  [skip summary] {e}", flush=True)
            continue
        s["step"] = step
        s["is_final"] = is_final
        s["csv"] = csv_out
        summary.append(s)
        print(f"  ↳ interior={s['interior_mean']*100:.2f}±{s['interior_std']*100:.2f}%  "
              f"cap={s['cap_mean']*100:.2f}%  return={s['return_mean']:.1f}  "
              f"ep_len={s['episode_len_mean']:.0f}  cata={s['catastrophic_rate']*100:.1f}%",
              flush=True)

    # Write summary CSV
    sum_csv = os.path.join(out_dir, f"{args.run_name}_summary.csv")
    with open(sum_csv, "w", newline="") as f:
        fieldnames = ["step", "is_final", "n", "interior_mean", "interior_std",
                      "interior_median", "cap_mean", "cap_median",
                      "return_mean", "return_std", "episode_len_mean",
                      "catastrophic_rate"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print(f"\n[summary] wrote {sum_csv}")

    # Plot
    if summary:
        plot_trajectory(
            summary,
            os.path.join(out_dir, f"{args.run_name}_trajectory.png"),
            title=f"Checkpoint sweep eval — {args.run_name}\n"
                  f"{args.episodes} deterministic eps each, {len(summary)} checkpoints",
        )

    # Terminal best-checkpoint hint
    if summary:
        best_by_interior = max(summary, key=lambda r: r["interior_mean"])
        print(f"\n>>> best checkpoint by interior_mean: "
              f"step={best_by_interior['step']}, "
              f"interior={best_by_interior['interior_mean']*100:.2f}%, "
              f"cap={best_by_interior['cap_mean']*100:.2f}%, "
              f"return={best_by_interior['return_mean']:.1f}")
        print(f">>> CSV: {best_by_interior['csv']}")


if __name__ == "__main__":
    sys.exit(main() or 0)
