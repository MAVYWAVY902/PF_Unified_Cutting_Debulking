"""Sprint 1 — Statistical analysis of M4.1 4-arm ablation CSVs.

Consumes per-episode CSVs emitted by `eval_policy.py` (one per arm × seed),
computes Welch's t-test + Cohen's d for Arm C vs each other arm, and writes
Table 2 LaTeX + markdown summary.

Primary metric (pre-registered per ICRA plan):
  • **capsule_dmg@80-coverage** — lower is better
  • Threshold: **Cohen's d > 0.8** → claim L2 advantage of Arm C

Usage
-----
  # Default: look for results/cap_arm{A,X,Y,C}_seed{0,1,2}.csv
  python warp_sim/run_stats.py --dir results

  # Or explicit:
  python warp_sim/run_stats.py \
      --armA results/cap_armA_seed*.csv \
      --armC results/cap_armC_seed*.csv \
      --out   results/table2.md
"""
import os
import sys
import glob
import argparse
import csv
import re
from collections import defaultdict

import numpy as np
from scipy import stats


def _load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _collect_by_arm(file_list):
    """Return dict[arm_label] = list of float values of primary metric per row."""
    by_arm = defaultdict(list)
    secondary = defaultdict(lambda: defaultdict(list))  # arm -> metric -> values
    for path in file_list:
        for row in _load_csv(path):
            arm = row["arm"]
            by_arm[arm].append(float(row["capsule_dmg_at_80"]))
            secondary[arm]["final_interior_cov"].append(float(row["final_interior_cov"]))
            secondary[arm]["steps_to_80"].append(float(row["steps_to_80"]))
            secondary[arm]["catastrophic"].append(int(row["catastrophic"]))
            secondary[arm]["return"].append(float(row["return"]))
            secondary[arm]["episode_len"].append(float(row["episode_len"]))
    return by_arm, secondary


def _cohens_d(x, y):
    """Pooled-variance Cohen's d. x = target (Arm C), y = baseline (A/X/Y)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0.0:
        return float("inf") if np.mean(x) != np.mean(y) else 0.0
    # Convention: d > 0 means x > y. For capsule_dmg we want C < others, so d
    # computed as `(mean_C - mean_other)`. We report absolute + sign.
    return (np.mean(x) - np.mean(y)) / pooled


def _welch(x, y):
    """Welch's unequal-variance t-test. Returns (t, p, df)."""
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan"), float("nan")
    t, p = stats.ttest_ind(x, y, equal_var=False)
    # Welch df (Satterthwaite approximation)
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    if vx == 0 and vy == 0:
        df = nx + ny - 2
    else:
        df = (vx / nx + vy / ny) ** 2 / (
            (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1) + 1e-12
        )
    return t, p, df


def _summarize(values, decimals=4):
    if not values:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "min": float("nan"), "max": float("nan")}
    v = np.asarray(values, dtype=np.float64)
    return {
        "n":      len(v),
        "mean":   round(float(np.mean(v)), decimals),
        "median": round(float(np.median(v)), decimals),
        "std":    round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, decimals),
        "min":    round(float(np.min(v)), decimals),
        "max":    round(float(np.max(v)), decimals),
    }


def _format_pval(p):
    if np.isnan(p):
        return "n/a"
    if p < 1e-4:
        return f"<1e-4"
    if p < 0.01:
        return f"{p:.4f}"
    return f"{p:.3f}"


def run_stats(by_arm, secondary, out_path):
    if "C" not in by_arm:
        raise RuntimeError("Missing Arm C data — cannot compute L2 advantage.")

    print("=" * 78)
    print("  M4.1 4-arm ablation statistics")
    print("=" * 78)

    primary_summary = {arm: _summarize(vals) for arm, vals in by_arm.items()}
    print("\nPrimary metric: capsule_dmg@80-coverage (lower better)")
    print(f"  {'arm':<5} {'n':>4}  {'mean':>8}  {'median':>8}  {'std':>8}  {'max':>8}")
    for arm in sorted(primary_summary.keys()):
        s = primary_summary[arm]
        print(f"  {arm:<5} {s['n']:>4}  {s['mean']:>8.4f}  {s['median']:>8.4f}  "
              f"{s['std']:>8.4f}  {s['max']:>8.4f}")

    # C vs A / X / Y comparisons
    print("\nArm C vs baselines (Welch's t-test, Cohen's d on capsule_dmg@80):")
    print(f"  {'comparison':<12}  {'Δmean':>8}  {'d':>7}  {'t':>7}  {'df':>6}  "
          f"{'p':>8}  {'verdict':<20}")
    comparisons = {}
    for other in ("A", "X", "Y"):
        if other not in by_arm:
            continue
        c_vals = by_arm["C"]
        o_vals = by_arm[other]
        t, p, df = _welch(c_vals, o_vals)
        d = _cohens_d(c_vals, o_vals)
        delta_mean = np.mean(c_vals) - np.mean(o_vals)

        # Verdict
        if np.isnan(d):
            verdict = "insufficient data"
        elif d < -0.8 and p < 0.05:
            verdict = "✓ C strongly < other"
        elif d < -0.5 and p < 0.05:
            verdict = "◇ C moderately <"
        elif d < 0 and p < 0.05:
            verdict = "~ C slightly <"
        elif p < 0.05:
            verdict = "✗ C ≥ other (!)"
        else:
            verdict = "no sig diff"

        comparisons[other] = {
            "delta": delta_mean, "d": d, "t": t, "df": df, "p": p,
            "verdict": verdict,
        }
        print(f"  C vs {other:<5}  {delta_mean:>+8.4f}  {d:>+7.3f}  {t:>+7.3f}  "
              f"{df:>6.1f}  {_format_pval(p):>8}  {verdict:<20}")

    # Secondary metrics (just summaries; statistical tests optional for now)
    print("\nSecondary metrics (summary only):")
    for metric_name in ("final_interior_cov", "steps_to_80", "catastrophic", "return"):
        print(f"\n  {metric_name}:")
        for arm in sorted(by_arm.keys()):
            vals = secondary[arm][metric_name]
            s = _summarize(vals, decimals=3)
            print(f"    {arm:<5}  mean={s['mean']:>7.3f}  std={s['std']:>7.3f}")

    # Write LaTeX Table 2 + markdown
    lines = []
    lines.append("# Table 2 — ICRA 2027 M4.1 4-arm ablation\n")
    lines.append("**Primary metric**: capsule_dmg @ 80% interior coverage "
                 "(lower is better)\n")
    lines.append("| arm | n | mean | median | std | vs C Δ | Cohen's d | p-value | verdict |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for arm in sorted(by_arm.keys()):
        s = primary_summary[arm]
        if arm == "C":
            row = (f"| **C** | {s['n']} | **{s['mean']:.4f}** | {s['median']:.4f} | "
                   f"{s['std']:.4f} | — | — | — | target |")
        elif arm in comparisons:
            c = comparisons[arm]
            row = (f"| {arm} | {s['n']} | {s['mean']:.4f} | {s['median']:.4f} | "
                   f"{s['std']:.4f} | {c['delta']:+.4f} | {c['d']:+.3f} | "
                   f"{_format_pval(c['p'])} | {c['verdict']} |")
        else:
            row = (f"| {arm} | {s['n']} | {s['mean']:.4f} | {s['median']:.4f} | "
                   f"{s['std']:.4f} | — | — | — | |")
        lines.append(row)
    lines.append("\n**Pre-registered threshold**: Cohen's d < -0.8 on (C vs A) "
                 "→ claim L2 advantage. Welch's t-test, p < 0.05.\n")

    # Also LaTeX form
    lines.append("\n## LaTeX (paste into paper)\n")
    lines.append("```latex")
    lines.append("\\begin{table}")
    lines.append("\\caption{4-arm ablation on capsule-preservation task. Primary metric: "
                 "capsule\\_dmg@80\\% interior coverage.}")
    lines.append("\\begin{tabular}{l|rrrrr|rrr|r}")
    lines.append("Arm & $n$ & Mean & Median & Std & Catastrophic & $\\Delta$ vs C & "
                 "Cohen's $d$ & $p$ & Verdict \\\\")
    lines.append("\\hline")
    for arm in sorted(by_arm.keys()):
        s = primary_summary[arm]
        cat_rate = float(np.mean(secondary[arm]["catastrophic"]))
        if arm == "C":
            row = (f"C & {s['n']} & \\textbf{{{s['mean']:.4f}}} & {s['median']:.4f} & "
                   f"{s['std']:.4f} & {cat_rate:.3f} & -- & -- & -- & target \\\\")
        elif arm in comparisons:
            c = comparisons[arm]
            row = (f"{arm} & {s['n']} & {s['mean']:.4f} & {s['median']:.4f} & "
                   f"{s['std']:.4f} & {cat_rate:.3f} & "
                   f"${c['delta']:+.4f}$ & ${c['d']:+.3f}$ & "
                   f"${_format_pval(c['p'])}$ & {c['verdict']} \\\\")
        else:
            row = (f"{arm} & {s['n']} & {s['mean']:.4f} & {s['median']:.4f} & "
                   f"{s['std']:.4f} & {cat_rate:.3f} & -- & -- & -- &  \\\\")
        lines.append(row)
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("```")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  → wrote {out_path}")


def _gather_files(args):
    files = []
    if args.armA:
        files += [p for pat in args.armA for p in glob.glob(pat)]
    if args.armX:
        files += [p for pat in args.armX for p in glob.glob(pat)]
    if args.armY:
        files += [p for pat in args.armY for p in glob.glob(pat)]
    if args.armC:
        files += [p for pat in args.armC for p in glob.glob(pat)]
    if args.dir:
        files += glob.glob(os.path.join(args.dir, "cap_arm*_seed*.csv"))
    files = sorted(set(files))
    if not files:
        raise RuntimeError(f"No CSV files found. Use --armA/--armX/--armY/--armC or --dir.")
    print(f"Reading {len(files)} CSV files:")
    for p in files:
        print(f"  {p}")
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=None, help="directory with cap_arm*_seed*.csv")
    p.add_argument("--armA", nargs="+", default=None, help="glob patterns for Arm A CSVs")
    p.add_argument("--armX", nargs="+", default=None)
    p.add_argument("--armY", nargs="+", default=None)
    p.add_argument("--armC", nargs="+", default=None)
    p.add_argument("--out", default="results/table2.md", help="output markdown+LaTeX path")
    args = p.parse_args()

    files = _gather_files(args)
    by_arm, secondary = _collect_by_arm(files)
    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    run_stats(by_arm, secondary, args.out)


if __name__ == "__main__":
    main()
