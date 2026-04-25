#!/usr/bin/env python3
"""Post-training NaN-guard monitor.

Diagnoses whether the env.step NaN guard (warp_sim/envs/debulk_env.py:428-436)
is firing during a trained policy's rollouts, and whether it correlates with
specific state regions (high capsule breach in particular). The guard is a
defensive boundary, NOT a root-cause fix — without this monitor you cannot
tell if it is silently masking an upstream physics bug.

The training-time `info["nan_in_obs"] / nan_in_reward` flags are NOT persisted
to TB or CSV anywhere — they live in the runtime info dict for one step then
get discarded. So we recover frequency two ways:

  Part 1 — Log parser (cheap, indirect)
    Reads training stdout log for proxies:
      • `RuntimeWarning` count                — divide-by-zero / overflow in env
      • `Early stopping at step 0` count      — KL spikes (a NaN-ish symptom)
      • `ep_len_mean` curve                   — sub-episode_length means force-term firing
      • `ep_rew_mean` mid-training drops      — possible NaN→0 reward replacement
    Doesn't tell us per-step NaN rate.

  Part 2 — Eval profiler (expensive, definitive)
    Runs N deterministic episodes wrapping the env and reading info per step:
      • `nan_in_obs / nan_in_reward`
      • `interior_done_frac / capsule_breached_frac` (state at NaN time)
      • `knife_pos`                                  (spatial pattern)
      • episode index, step-in-episode               (temporal pattern)
    Aggregates into:
      • Per-step + per-episode NaN rate
      • mean(breach | nan)   vs mean(breach | ¬nan)
      • mean(interior | nan) vs mean(interior | ¬nan)
      • Histogram of breach@nan / interior@nan / step@nan
      • Knife-pos scatter at NaN events
    Verdict (printed):
      0%       — guard never fired; clean
      < 0.1%   — guard rare, accept as insurance
      0.1-1%   — investigate; may bias policy near NaN regions
      > 1%     — guard hides upstream bug; root-cause before publishing data

Usage
-----
  # Full diagnostic (log + eval) on currently-running run after it finishes:
  python tools/monitor_nan.py --run-name cap_armC_torus_zup_v2_seed0 --episodes 50

  # Only parse the training log (cheap, while training is still running):
  python tools/monitor_nan.py --run-name cap_armC_torus_zup_v2_seed0 --log-only

  # Use a specific intermediate checkpoint instead of final:
  python tools/monitor_nan.py --run-name cap_armC_torus_zup_v2_seed0 \\
                              --step 400000 --episodes 30
"""
import argparse
import json
import os
import re
import sys
import csv
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "pf_shape_matching_warp"))


# ---------------------------------------------------------------------------
# Part 1 — Log parser
# ---------------------------------------------------------------------------

ITER_TABLE_RE = re.compile(
    r"\|\s*ep_len_mean\s*\|\s*([\d.e+\-]+)\s*\|.*?"
    r"\|\s*ep_rew_mean\s*\|\s*([\d.e+\-]+)\s*\|.*?"
    r"\|\s*total_timesteps\s*\|\s*([\d.e+\-]+)\s*\|",
    re.DOTALL,
)


def _to_float(s):
    try:
        return float(s)
    except Exception:
        return float("nan")


def parse_training_log(log_path):
    """Read training stdout log, return dict of indirect NaN-signal stats."""
    if not os.path.exists(log_path):
        return {"log_exists": False, "log_path": log_path}

    with open(log_path) as f:
        text = f.read()

    runtime_warnings = len(re.findall(r"RuntimeWarning", text))
    invalid_divide = len(re.findall(r"invalid value encountered in divide", text))
    overflow = len(re.findall(r"overflow encountered", text))
    # SB3 emits `Early stopping at step N due to reaching max kl` for any N
    # (N=0 = killed before first update; N>0 = halted partway). All N values
    # signal KL-clip kicking in; N=0 is the most aggressive subset.
    early_stops_total = len(re.findall(r"Early stopping at step \d+ due to reaching max kl", text))
    early_stops_step0 = len(re.findall(r"Early stopping at step 0 due to reaching max kl", text))
    nan_traceback = len(re.findall(r"loc \(Tensor of shape.*?nan", text, re.DOTALL))
    crashed_with_nan = "nan, nan, nan" in text or "nan]], device" in text

    # ep_len / ep_rew / step trajectory
    iter_rows = []
    for m in ITER_TABLE_RE.finditer(text):
        ep_len  = _to_float(m.group(1))
        ep_rew  = _to_float(m.group(2))
        steps   = _to_float(m.group(3))
        iter_rows.append((steps, ep_len, ep_rew))
    iter_rows.sort(key=lambda r: r[0])

    # Episode-length signal: any iteration with ep_len_mean < 95% of nominal episode_length
    # suggests force-terminations are occurring (catastrophic OR NaN-guard).
    episode_length_nominal = 1024  # we only run torus configs now
    short_episodes = [r for r in iter_rows if r[1] < 0.95 * episode_length_nominal]

    return {
        "log_exists": True,
        "log_path": log_path,
        "runtime_warnings_total": runtime_warnings,
        "invalid_divide": invalid_divide,
        "overflow": overflow,
        "early_stops_kl_total": early_stops_total,
        "early_stops_kl_step0": early_stops_step0,
        "nan_traceback_seen": nan_traceback,
        "crashed_with_nan": crashed_with_nan,
        "iter_count": len(iter_rows),
        "short_episode_iters": len(short_episodes),
        "short_episode_iter_examples": short_episodes[:5],
        "iter_rows": iter_rows,  # full curve, in case caller wants to plot
    }


def print_log_summary(stats):
    if not stats["log_exists"]:
        print(f"[log] not found: {stats['log_path']}")
        return
    print(f"[log] parsed {stats['iter_count']} iteration tables from {stats['log_path']}")
    print(f"[log] RuntimeWarning           : {stats['runtime_warnings_total']}")
    print(f"[log]   invalid divide         : {stats['invalid_divide']}")
    print(f"[log]   overflow               : {stats['overflow']}")
    print(f"[log] KL early-stop (total)    : {stats['early_stops_kl_total']}  "
          f"(step-0 subset: {stats['early_stops_kl_step0']})")
    print(f"[log] NaN traceback seen       : {stats['nan_traceback_seen']}")
    print(f"[log] training crashed w/ NaN  : {stats['crashed_with_nan']}")
    print(f"[log] iters with ep_len < 0.95×1024 (force-term firing): "
          f"{stats['short_episode_iters']} / {stats['iter_count']}")
    if stats["short_episode_iter_examples"]:
        print(f"[log]   sample short-iter rows (steps, ep_len, ep_rew):")
        for s, l, r in stats["short_episode_iter_examples"]:
            print(f"          step={int(s):>7d}  ep_len={l:7.1f}  ep_rew={r:8.2f}")


# ---------------------------------------------------------------------------
# Part 2 — Eval-time NaN profiler
# ---------------------------------------------------------------------------

def find_checkpoint(ckpt_dir, run_name, step="final"):
    """Locate (ckpt_path, vecnorm_path). step='final' or integer."""
    if step == "final":
        ck = os.path.join(ckpt_dir, f"{run_name}_final.zip")
        vn = os.path.join(ckpt_dir, f"{run_name}_final_vecnormalize.pkl")
    else:
        ck = os.path.join(ckpt_dir, f"{run_name}_{int(step)}_steps.zip")
        vn = os.path.join(ckpt_dir, f"{run_name}_vecnormalize_{int(step)}_steps.pkl")
    return (ck if os.path.exists(ck) else None,
            vn if os.path.exists(vn) else None)


def derive_cfg_path(run_name):
    """Map run-name like cap_armC_torus_zup_v2_seed0 → debulk_cap_armC.yaml."""
    m = re.search(r"arm([A-Za-z])", run_name)
    if not m:
        return None
    arm = m.group(1).upper()
    candidate = os.path.join(REPO, "warp_sim", "config", f"debulk_cap_arm{arm}.yaml")
    return candidate if os.path.exists(candidate) else None


def run_eval_profile(cfg_path, ck_path, vn_path, episodes, device="cuda:0", verbose=True):
    """Roll N deterministic episodes; record every per-step info row. Return list of dicts."""
    import warp as wp
    import yaml
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    from warp_sim.envs.debulk_env import DebulkEnv

    wp.init()
    cfg = yaml.safe_load(open(cfg_path))
    env_fn = lambda: DebulkEnv(cfg, device=device)
    env_raw = DummyVecEnv([env_fn])
    env_mon = VecMonitor(env_raw)

    if vn_path:
        env_norm = VecNormalize.load(vn_path, env_mon)
        env_norm.training = False
        env_norm.norm_reward = False
    else:
        env_norm = env_mon

    model = PPO.load(ck_path, env=env_norm, device="cpu")
    underlying = env_raw.envs[0]
    max_steps = int(underlying.episode_length) + 2

    rows = []
    for ep in range(episodes):
        obs = env_norm.reset()
        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info_list = env_norm.step(action)
            info = info_list[0] if info_list else {}
            kp = info.get("knife_pos", np.zeros(3))
            rows.append({
                "ep": ep,
                "step": t,
                "nan_in_obs":            int(bool(info.get("nan_in_obs", False))),
                "nan_in_reward":         int(bool(info.get("nan_in_reward", False))),
                "interior_done_frac":    float(info.get("interior_done_frac", 0.0)),
                "capsule_breached_frac": float(info.get("capsule_breached_frac", 0.0)),
                "reward":                float(reward[0]),
                "knife_x": float(kp[0]) if hasattr(kp, "__len__") else 0.0,
                "knife_y": float(kp[1]) if hasattr(kp, "__len__") else 0.0,
                "knife_z": float(kp[2]) if hasattr(kp, "__len__") else 0.0,
                "done":                  int(bool(done[0])),
            })
            if done[0]:
                break
        if verbose and (ep + 1) % 5 == 0:
            recent = rows[-min(len(rows), 5 * max_steps):]
            recent_nan = sum(r["nan_in_obs"] | r["nan_in_reward"] for r in recent)
            print(f"  ep {ep+1}/{episodes}  recent NaN steps={recent_nan}/{len(recent)}",
                  flush=True)

    env_norm.close()
    return rows


def aggregate_nan_stats(rows):
    """Compute frequency + correlation aggregates from per-step rows."""
    if not rows:
        return {"total_steps": 0, "verdict": "NO_DATA"}
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    total = len(rows)
    nan_mask = (arr["nan_in_obs"] | arr["nan_in_reward"]).astype(bool)
    nan_count = int(nan_mask.sum())
    rate_step = nan_count / total

    eps_with_nan = len(set(int(r["ep"]) for r in rows if r["nan_in_obs"] | r["nan_in_reward"]))
    eps_total = len(set(int(r["ep"]) for r in rows))
    rate_ep = eps_with_nan / eps_total if eps_total else 0.0

    out = {
        "total_steps":       total,
        "total_episodes":    eps_total,
        "nan_steps":         nan_count,
        "eps_with_any_nan":  eps_with_nan,
        "rate_per_step":     rate_step,
        "rate_per_episode":  rate_ep,
    }

    if nan_count > 0:
        out["breach_at_nan_mean"]    = float(arr["capsule_breached_frac"][nan_mask].mean())
        out["breach_at_nan_median"]  = float(np.median(arr["capsule_breached_frac"][nan_mask]))
        out["breach_clean_mean"]     = float(arr["capsule_breached_frac"][~nan_mask].mean())
        out["interior_at_nan_mean"]  = float(arr["interior_done_frac"][nan_mask].mean())
        out["interior_clean_mean"]   = float(arr["interior_done_frac"][~nan_mask].mean())
        out["step_at_nan_mean"]      = float(arr["step"][nan_mask].mean())
        out["step_at_nan_median"]    = float(np.median(arr["step"][nan_mask]))
        out["knife_at_nan_mean"]     = [
            float(arr["knife_x"][nan_mask].mean()),
            float(arr["knife_y"][nan_mask].mean()),
            float(arr["knife_z"][nan_mask].mean()),
        ]
        # Spearman-ish indicator: does breach predict NaN? Compare top-decile breach NaN rate
        # against the bottom-decile breach NaN rate.
        breach = arr["capsule_breached_frac"]
        q90 = np.quantile(breach, 0.90)
        q10 = np.quantile(breach, 0.10)
        rate_high_breach = nan_mask[breach >= q90].mean() if (breach >= q90).any() else 0.0
        rate_low_breach  = nan_mask[breach <= q10].mean() if (breach <= q10).any() else 0.0
        out["nan_rate_top_decile_breach"]    = float(rate_high_breach)
        out["nan_rate_bottom_decile_breach"] = float(rate_low_breach)
    else:
        # When zero NaN, all correlation stats are undefined.
        out["breach_at_nan_mean"] = None
        out["breach_clean_mean"]  = float(arr["capsule_breached_frac"].mean())
        out["interior_clean_mean"] = float(arr["interior_done_frac"].mean())

    # Verdict
    if rate_step == 0.0:
        verdict = "CLEAN — guard never fired."
    elif rate_step < 1e-3:
        verdict = "OK — < 0.1% rate, treat guard as insurance."
    elif rate_step < 1e-2:
        verdict = "INVESTIGATE — 0.1–1% rate; check breach correlation below."
    else:
        verdict = "BAD — > 1% rate; guard is hiding upstream bug. Root-cause before shipping data."
    out["verdict"] = verdict
    return out


def print_eval_summary(stats):
    print("\n" + "=" * 72)
    print(f"NaN guard frequency over {stats['total_steps']} steps "
          f"({stats['total_episodes']} eps):")
    print(f"  rate per step    = {stats['rate_per_step']*100:.4f}%   "
          f"({stats['nan_steps']} steps)")
    print(f"  rate per episode = {stats['rate_per_episode']*100:.2f}%   "
          f"({stats['eps_with_any_nan']}/{stats['total_episodes']} eps with any NaN)")
    if stats["nan_steps"] > 0:
        print()
        print(f"  Breach correlation:")
        print(f"    mean breach @ NaN     = {stats['breach_at_nan_mean']:.4f}")
        print(f"    mean breach @ clean   = {stats['breach_clean_mean']:.4f}")
        if stats["breach_clean_mean"] > 0:
            ratio = stats["breach_at_nan_mean"] / stats["breach_clean_mean"]
            print(f"    ratio                 = {ratio:.2f}×  "
                  f"({'NaN ENRICHED in high-breach states' if ratio > 1.5 else 'no strong correlation'})")
        print(f"    NaN rate | top-10% breach = "
              f"{stats['nan_rate_top_decile_breach']*100:.4f}%")
        print(f"    NaN rate | bot-10% breach = "
              f"{stats['nan_rate_bottom_decile_breach']*100:.4f}%")
        print()
        print(f"  Interior correlation:")
        print(f"    mean interior @ NaN   = {stats['interior_at_nan_mean']:.4f}")
        print(f"    mean interior @ clean = {stats['interior_clean_mean']:.4f}")
        print()
        print(f"  Temporal pattern:")
        print(f"    mean step-in-ep @ NaN = {stats['step_at_nan_mean']:.1f}  "
              f"(median {stats['step_at_nan_median']:.0f})")
        print(f"  Spatial:")
        print(f"    mean knife pos @ NaN  = "
              f"({stats['knife_at_nan_mean'][0]:.3f}, "
              f"{stats['knife_at_nan_mean'][1]:.3f}, "
              f"{stats['knife_at_nan_mean'][2]:.3f})")
    print()
    print(f"  VERDICT: {stats['verdict']}")
    print("=" * 72)


def maybe_plot(rows, agg, out_path):
    """4-panel plot: NaN-time, breach@nan vs all, interior@nan vs all, knife-pos scatter."""
    if agg["total_steps"] == 0:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib unavailable; skipping diagnostic figure")
        return

    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    nan_mask = (arr["nan_in_obs"] | arr["nan_in_reward"]).astype(bool)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # (a) NaN events vs episode-step
    if nan_mask.any():
        ax[0, 0].hist(arr["step"][nan_mask], bins=30, color="crimson", alpha=0.85)
    ax[0, 0].set_title(f"NaN events by step-in-episode "
                       f"({int(nan_mask.sum())} total)")
    ax[0, 0].set_xlabel("step in episode")
    ax[0, 0].set_ylabel("count")

    # (b) breach distribution: at-NaN vs all
    if nan_mask.any():
        ax[0, 1].hist(arr["capsule_breached_frac"], bins=40, alpha=0.5,
                      label="all steps", color="steelblue", density=True)
        ax[0, 1].hist(arr["capsule_breached_frac"][nan_mask], bins=20,
                      alpha=0.85, label="@ NaN steps", color="crimson", density=True)
    else:
        ax[0, 1].hist(arr["capsule_breached_frac"], bins=40, color="steelblue")
    ax[0, 1].axvline(0.05, color="k", linestyle="--", alpha=0.4, label="budget=0.05")
    ax[0, 1].axvline(0.15, color="r", linestyle="--", alpha=0.4, label="catastrophic=0.15")
    ax[0, 1].set_title("capsule_breached_frac distribution")
    ax[0, 1].set_xlabel("breach")
    ax[0, 1].legend(fontsize=8)

    # (c) interior distribution
    if nan_mask.any():
        ax[1, 0].hist(arr["interior_done_frac"], bins=40, alpha=0.5,
                      label="all steps", color="steelblue", density=True)
        ax[1, 0].hist(arr["interior_done_frac"][nan_mask], bins=20,
                      alpha=0.85, label="@ NaN steps", color="crimson", density=True)
    else:
        ax[1, 0].hist(arr["interior_done_frac"], bins=40, color="steelblue")
    ax[1, 0].set_title("interior_done_frac distribution")
    ax[1, 0].set_xlabel("interior")
    ax[1, 0].legend(fontsize=8)

    # (d) knife-pos scatter (XZ for Z-up: ring plane)
    ax[1, 1].scatter(arr["knife_x"][~nan_mask], arr["knife_y"][~nan_mask],
                     s=2, alpha=0.10, c="steelblue", label="clean")
    if nan_mask.any():
        ax[1, 1].scatter(arr["knife_x"][nan_mask], arr["knife_y"][nan_mask],
                         s=14, alpha=0.95, c="crimson", marker="x", label="@ NaN")
    ax[1, 1].set_title("knife XY pos (ring plane)")
    ax[1, 1].set_xlabel("knife_x")
    ax[1, 1].set_ylabel("knife_y")
    ax[1, 1].set_aspect("equal", adjustable="box")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle(f"NaN-guard diagnostic — rate {agg['rate_per_step']*100:.4f}% per step",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[plot] wrote {out_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True,
                   help="Run name (e.g. cap_armC_torus_zup_v2_seed0)")
    p.add_argument("--ckpt-dir", default=None,
                   help="Defaults to checkpoints/{run_name}/")
    p.add_argument("--log-path", default=None,
                   help="Defaults to checkpoints/{run_name}.log")
    p.add_argument("--cfg", default=None,
                   help="Defaults to derived from run_name (armA/X/Y/C → debulk_cap_arm*.yaml)")
    p.add_argument("--step", default="final",
                   help="Checkpoint step (integer or 'final', default final)")
    p.add_argument("--episodes", type=int, default=50,
                   help="Eval episodes for Part 2 (default 50)")
    p.add_argument("--out-dir", default=None,
                   help="Defaults to results/nan_monitor/{run_name}/")
    p.add_argument("--log-only", action="store_true",
                   help="Skip Part 2 (eval). Useful while training is still running.")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    ckpt_dir = args.ckpt_dir or os.path.join(REPO, "checkpoints", args.run_name)
    log_path = args.log_path or os.path.join(REPO, "checkpoints", f"{args.run_name}.log")
    cfg_path = args.cfg or derive_cfg_path(args.run_name)
    out_dir  = args.out_dir or os.path.join(REPO, "results", "nan_monitor", args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Part 1 — log parser
    print(f"\n=== Part 1 — Training log indirect signals ===")
    log_stats = parse_training_log(log_path)
    print_log_summary(log_stats)

    # Part 2 — eval profiler
    eval_stats = None
    rows = None
    if args.log_only:
        print("\n[skip] Part 2 (--log-only)")
    else:
        print(f"\n=== Part 2 — Deterministic eval NaN profile ===")
        if not cfg_path:
            print(f"[error] cannot derive cfg from run-name={args.run_name}; pass --cfg.")
            sys.exit(2)
        ck_path, vn_path = find_checkpoint(ckpt_dir, args.run_name, args.step)
        if ck_path is None:
            print(f"[skip] no checkpoint at step={args.step} under {ckpt_dir}.")
            print(f"[hint] run is probably still training; try --log-only or wait for "
                  f"final.zip / {args.step}_steps.zip to land.")
        else:
            print(f"  cfg          = {cfg_path}")
            print(f"  checkpoint   = {ck_path}")
            print(f"  vecnormalize = {vn_path}")
            print(f"  episodes     = {args.episodes}")
            rows = run_eval_profile(cfg_path, ck_path, vn_path, args.episodes,
                                    device=args.device)
            eval_stats = aggregate_nan_stats(rows)
            print_eval_summary(eval_stats)

            # Persist artifacts
            csv_path = os.path.join(out_dir, "nan_events.csv")
            fields = list(rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print(f"\n[csv] wrote {csv_path}  ({len(rows)} rows)")

            plot_path = os.path.join(out_dir, "nan_diagnostic.png")
            maybe_plot(rows, eval_stats, plot_path)

    # Combined JSON summary
    summary = {
        "run_name": args.run_name,
        "step":     args.step,
        "log":      {k: v for k, v in log_stats.items() if k != "iter_rows"},
        "eval":     eval_stats,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[json] wrote {summary_path}")


if __name__ == "__main__":
    main()
