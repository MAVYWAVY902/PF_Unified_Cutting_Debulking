#!/usr/bin/env python3
"""
Stage 2.3.5 sharpness sensitivity sweep — runs the full protocol for the
F/T-sensor purchase decision.

Protocol (per project_phantom_experiment_design.md Stage 2.3.5):
  1. Control run: sharpness=20, 1 ckpt (200k), 50 episodes — measure σ
  2. Full sweep:  sharpness ∈ {14, 17, 20, 23, 26}
                  × ckpts ∈ {200k, 350k, 500k}
                  × N episodes (auto-scaled from σ)
  3. Aggregate + apply asymmetric verdict:
        Δ < 2%  → don't buy F/T  (strong: even mediocre v3 robust)
        2-5%     → defer to v4 retrain
        Δ > 5%  → defer (likely buy F/T but verify with v4)

Usage:
  python tools/run_sharpness_sensitivity.py --mode control      # just control run
  python tools/run_sharpness_sensitivity.py --mode sweep        # full sweep (after control)
  python tools/run_sharpness_sensitivity.py --mode analyze      # just re-analyze existing CSVs

Outputs:
  results/stage2_3_5_sensitivity/
    control_sharp20_ckpt200k.csv         (50 eps, σ baseline)
    sweep_sharp{14,17,20,23,26}_ckpt{200k,350k,500k}.csv
    SUMMARY.md                            (analysis output + verdict)
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PYTHON = "/home/yunxin/miniconda3/envs/debulk_warp/bin/python"
CFG = REPO / "warp_sim/config/debulk_cap_armC.yaml"
CKPT_DIR = REPO / "checkpoints/cap_armC_torus_seed0"
OUT_DIR = REPO / "results/stage2_3_5_sensitivity"

CKPTS = {
    "200k": "cap_armC_torus_seed0_200000_steps",
    "350k": "cap_armC_torus_seed0_350000_steps",
    "500k": "cap_armC_torus_seed0_500000_steps",
}

SHARPNESS_VALUES = [14.0, 17.0, 20.0, 23.0, 26.0]


def run_eval(ckpt_label, sharpness, n_episodes, out_csv):
    """Invoke eval_policy.py as a subprocess with the override flag."""
    ckpt_base = CKPTS[ckpt_label]
    ckpt_path = CKPT_DIR / f"{ckpt_base}.zip"
    vecnorm_path = CKPT_DIR / f"cap_armC_torus_seed0_vecnormalize_{ckpt_base.split('_')[-2]}_steps.pkl"

    cmd = [
        PYTHON, str(REPO / "warp_sim/eval_policy.py"),
        "--checkpoint", str(ckpt_path),
        "--vecnorm", str(vecnorm_path),
        "--cfg", str(CFG),
        "--episodes", str(n_episodes),
        "--out", str(out_csv),
        "--arm", f"armC_torus_sharp{sharpness:.0f}_ckpt{ckpt_label}",
        "--seed", "0",
        "--sharpness_override", str(sharpness),
    ]
    print(f"[run] sharp={sharpness}, ckpt={ckpt_label}, N={n_episodes} → {out_csv.name}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.0f}s):\n  STDERR: {result.stderr[-500:]}")
        return False
    print(f"  ok ({elapsed:.0f}s, {elapsed/n_episodes:.1f}s/ep)")
    return True


def load_csv(path):
    """Read eval CSV, return numpy structured array."""
    import csv
    rows = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append({
                "interior":   float(r["final_interior_cov"]),
                "cap_at_80":  float(r["capsule_dmg_at_80"]),
                "catastrophic": int(r["catastrophic"]),
                "ep_len":     int(r["episode_len"]),
                "return":     float(r["return"]),
            })
    return rows


def summarize(rows):
    """Return dict of mean/std for key metrics."""
    interior = [r["interior"] for r in rows]
    cap = [r["cap_at_80"] for r in rows]
    cata = [r["catastrophic"] for r in rows]
    ep_len = [r["ep_len"] for r in rows]
    return {
        "n":              len(rows),
        "interior_mean":  float(np.mean(interior)),
        "interior_std":   float(np.std(interior, ddof=1) if len(interior) > 1 else 0.0),
        "cap_mean":       float(np.mean(cap)),
        "cap_std":        float(np.std(cap, ddof=1) if len(cap) > 1 else 0.0),
        "catastrophic_rate": float(np.mean(cata)),
        "ep_len_mean":    float(np.mean(ep_len)),
    }


def control_run(n_eps=50):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "control_sharp20_ckpt200k.csv"
    print("=" * 72)
    print(f"  CONTROL RUN: sharp=20, ckpt=200k, N={n_eps}")
    print("=" * 72)
    ok = run_eval("200k", 20.0, n_eps, out)
    if not ok:
        return None
    rows = load_csv(out)
    s = summarize(rows)
    print(f"\n  Summary:")
    print(f"    interior  mean={s['interior_mean']:.4f}  std={s['interior_std']:.4f}")
    print(f"    cap@80    mean={s['cap_mean']:.4f}  std={s['cap_std']:.4f}")
    print(f"    cata rate {s['catastrophic_rate']:.3f}  ({int(s['catastrophic_rate']*s['n'])}/{s['n']})")
    print(f"\n  σ check (signal-detection capability):")
    se_50 = s['interior_std'] / np.sqrt(50)
    print(f"    SE(50 eps) interior = {se_50:.4f} ≈ {se_50*100:.2f}%")
    if se_50 < 0.01:
        print(f"    → 50 eps per condition resolves <1% effects. Proceed with N=50.")
    elif se_50 < 0.025:
        print(f"    → 50 eps per condition resolves ~2.5% effects. Borderline; consider N=100.")
    else:
        print(f"    → 50 eps per condition only resolves >5%. Need N=100-200.")
    return s


def sweep_run(n_eps=50):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f"  FULL SWEEP: 5 sharpness × 3 checkpoints × N={n_eps} eps")
    print(f"  Total runs: {5 * 3} = 15 conditions × {n_eps} eps = {15 * n_eps} episodes")
    print("=" * 72)
    failed = []
    for ckpt_label in CKPTS:
        for sharp in SHARPNESS_VALUES:
            out = OUT_DIR / f"sweep_sharp{int(sharp)}_ckpt{ckpt_label}.csv"
            # Skip control run if already done (sharp=20, ckpt=200k same data exists)
            if sharp == 20.0 and ckpt_label == "200k" and (OUT_DIR / "control_sharp20_ckpt200k.csv").exists():
                # symlink so analyze step finds it under sweep name
                if not out.exists():
                    out.symlink_to(OUT_DIR / "control_sharp20_ckpt200k.csv")
                continue
            ok = run_eval(ckpt_label, sharp, n_eps, out)
            if not ok:
                failed.append((ckpt_label, sharp))
    if failed:
        print(f"\n  FAILED runs: {failed}")
    return failed


def analyze():
    """Aggregate all sweep CSVs into a comparison table + apply asymmetric verdict."""
    print("=" * 72)
    print("  ANALYSIS")
    print("=" * 72)

    results = {}  # ckpt_label → sharp → summary
    for ckpt_label in CKPTS:
        results[ckpt_label] = {}
        for sharp in SHARPNESS_VALUES:
            f = OUT_DIR / f"sweep_sharp{int(sharp)}_ckpt{ckpt_label}.csv"
            if not f.exists() and sharp == 20.0 and ckpt_label == "200k":
                f = OUT_DIR / "control_sharp20_ckpt200k.csv"
            if not f.exists():
                print(f"  MISSING: {f.name}")
                continue
            rows = load_csv(f)
            results[ckpt_label][sharp] = summarize(rows)

    # Per-ckpt sensitivity (max - min across sharpness)
    print(f"\n  {'ckpt':>6}  {'metric':>14}  {'min':>8}  {'max':>8}  {'Δ':>7}")
    deltas = {}  # ckpt → metric → delta
    for ckpt_label in CKPTS:
        if not results[ckpt_label]:
            continue
        deltas[ckpt_label] = {}
        for metric in ["interior_mean", "cap_mean", "catastrophic_rate"]:
            vals = [results[ckpt_label][s][metric] for s in SHARPNESS_VALUES if s in results[ckpt_label]]
            if not vals:
                continue
            d = max(vals) - min(vals)
            deltas[ckpt_label][metric] = d
            print(f"  {ckpt_label:>6}  {metric:>14}  {min(vals):>8.4f}  {max(vals):>8.4f}  {d*100:>6.2f}%")

    # Cross-checkpoint consistency: is the max-min pattern similar across ckpts?
    print("\n  Cross-checkpoint consistency check:")
    for metric in ["interior_mean", "cap_mean", "catastrophic_rate"]:
        ds = [deltas[c][metric] for c in deltas if metric in deltas[c]]
        if ds:
            print(f"    {metric}: deltas across ckpts = {[f'{d*100:.2f}%' for d in ds]}, "
                  f"mean = {np.mean(ds)*100:.2f}%")

    # Asymmetric verdict
    print("\n" + "=" * 72)
    print("  VERDICT (asymmetric — biased toward 'don't buy F/T' given v3 mediocrity)")
    print("=" * 72)

    interior_deltas = [deltas[c].get("interior_mean", 0) for c in deltas]
    cap_deltas = [deltas[c].get("cap_mean", 0) for c in deltas]
    cata_deltas = [deltas[c].get("catastrophic_rate", 0) for c in deltas]
    max_delta = max(interior_deltas + cap_deltas + cata_deltas, default=0)

    if max_delta < 0.02:
        print("  ✅ ROBUST (max Δ < 2% across all metrics, all checkpoints)")
        print("     → DON'T BUY F/T sensor.")
        print("     → Sharpness ±15% calibration tolerance is acceptable.")
        print("     → Stage 2.2 uses cavity-only loss.")
        print("     → Save $2k + 4-6w lead time.")
    elif max_delta < 0.05:
        print(f"  🟡 BORDERLINE (max Δ = {max_delta*100:.2f}%)")
        print("     → DEFER F/T decision to v4 retrain.")
        print("     → Re-run this test with v4 best ckpt before ordering.")
    else:
        print(f"  🟠 SENSITIVE (max Δ = {max_delta*100:.2f}%)")
        print("     → DEFER F/T decision (BUT lean toward purchase) to v4 retrain.")
        print("     → v3 mediocrity may cause false-positive sensitivity.")
        print("     → Verify with v4 before committing to $2k purchase.")

    # Write SUMMARY.md
    md_path = OUT_DIR / "SUMMARY.md"
    with open(md_path, "w") as f:
        f.write("# Stage 2.3.5 sharpness sensitivity sweep results\n\n")
        f.write("## Protocol\n")
        f.write(f"- Sharpness values: {SHARPNESS_VALUES} (default 20)\n")
        f.write(f"- Checkpoints: {list(CKPTS.keys())}\n")
        f.write(f"- N episodes per condition: see CSVs\n\n")
        f.write("## Per-checkpoint deltas (max-min across 5 sharpness values)\n\n")
        f.write("| ckpt | metric | min | max | Δ |\n|---|---|---|---|---|\n")
        for ckpt_label in deltas:
            for metric in deltas[ckpt_label]:
                vals = [results[ckpt_label][s][metric] for s in SHARPNESS_VALUES if s in results[ckpt_label]]
                f.write(f"| {ckpt_label} | {metric} | {min(vals):.4f} | "
                        f"{max(vals):.4f} | {deltas[ckpt_label][metric]*100:.2f}% |\n")
        f.write(f"\n## Verdict\nMax Δ = {max_delta*100:.2f}%. ")
        if max_delta < 0.02:
            f.write("**ROBUST** — don't buy F/T sensor.\n")
        elif max_delta < 0.05:
            f.write("**BORDERLINE** — defer to v4.\n")
        else:
            f.write("**SENSITIVE** — defer to v4 (lean toward F/T purchase).\n")
    print(f"\n  Summary written to {md_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["control", "sweep", "analyze"], required=True)
    p.add_argument("--n_eps", type=int, default=50)
    args = p.parse_args()

    t0 = time.time()
    if args.mode == "control":
        control_run(args.n_eps)
    elif args.mode == "sweep":
        sweep_run(args.n_eps)
        analyze()
    elif args.mode == "analyze":
        analyze()
    print(f"\n[total elapsed] {time.time() - t0:.0f}s")
