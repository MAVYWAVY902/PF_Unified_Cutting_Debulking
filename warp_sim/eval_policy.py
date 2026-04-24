"""Sprint 1 — Per-checkpoint deterministic policy evaluation.

ICRA 2027 plan: after M4.1 12-run training, run 100 deterministic eval episodes
per (arm, seed) checkpoint and emit a CSV of primary + secondary metrics.
Feeds `run_stats.py` for Welch's t-test + Cohen's d.

Usage
-----
  python warp_sim/eval_policy.py \
      --checkpoint checkpoints/cap_armC_seed0/final.zip \
      --vecnorm    checkpoints/cap_armC_seed0/vecnormalize_final.pkl \
      --cfg        warp_sim/config/debulk_cap_armC.yaml \
      --episodes 100 \
      --out results/cap_armC_seed0.csv

Primary metric (pre-registered per plan §4.3):
  • capsule_damage @ 80% interior coverage  — lower is better

Secondary metrics:
  • final interior coverage %
  • steps to 80% interior coverage
  • catastrophic breach rate (R terminated with breach > 3×budget)
  • episode return (sum of rewards)
  • episode length

Output CSV schema (one row per episode):
  arm,seed,ep,final_interior_cov,capsule_dmg_at_80,steps_to_80,catastrophic,
  return,episode_len,terminated_by_completion,terminated_by_catastrophe

The per-episode CSV is the unit of statistical analysis in run_stats.py.
"""
import os
import sys
import argparse
import csv
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pf_shape_matching_warp"))

import warp as wp


def _run_single_episode(env_raw, env_norm, model, seed_rng, max_steps=512):
    """Run one deterministic episode via `model.predict(obs, deterministic=True)`.
    Track per-step interior coverage progression to compute step-to-80 and
    capsule_damage@80-coverage metrics. Return a dict of per-episode metrics."""
    obs = env_norm.reset()
    ep_return = 0.0
    ep_len = 0
    terminated_by_completion = False
    terminated_by_catastrophe = False

    # Track the first step where interior_done_frac >= 0.80 (for @80%-coverage metric)
    step_first_80 = None
    capsule_dmg_at_first_80 = None
    final_interior_cov = 0.0
    final_capsule_dmg = 0.0

    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info_list = env_norm.step(action)
        ep_return += float(reward[0])
        ep_len += 1

        # Read metrics from info dict — survives VecEnv auto-reset on done=True.
        # Direct env-attribute reads would hit reset(0.0) after episode end.
        info = info_list[0]
        interior_done = float(info.get("interior_done_frac", 0.0))
        capsule_breach = float(info.get("capsule_breached_frac", 0.0))
        underlying = env_raw.envs[0]
        final_interior_cov = interior_done
        final_capsule_dmg = capsule_breach

        if step_first_80 is None and interior_done >= 0.80:
            step_first_80 = t + 1
            capsule_dmg_at_first_80 = capsule_breach

        if done[0]:
            infos = info_list[0]
            if interior_done >= underlying.completion_interior_frac:
                terminated_by_completion = True
            if capsule_breach >= underlying.catastrophic_breach_mult * underlying.capsule_budget:
                terminated_by_catastrophe = True
            break

    # If never reached 80% coverage, capsule_dmg@80 uses end-of-episode capsule damage
    # (worst-case assumption: flagged as high). Mark with sentinel step=max_steps+1.
    if step_first_80 is None:
        step_first_80 = ep_len + 1
        capsule_dmg_at_first_80 = final_capsule_dmg

    return {
        "final_interior_cov":       round(final_interior_cov, 4),
        "capsule_dmg_at_80":        round(capsule_dmg_at_first_80, 4),
        "steps_to_80":              step_first_80,
        "catastrophic":             int(terminated_by_catastrophe),
        "return":                   round(ep_return, 3),
        "episode_len":              ep_len,
        "terminated_by_completion": int(terminated_by_completion),
        "terminated_by_catastrophe": int(terminated_by_catastrophe),
    }


def evaluate(cfg_path, ckpt_path, vecnorm_path, episodes, out_path,
             arm_label, seed_label, device="cuda:0"):
    wp.init()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    from warp_sim.envs.debulk_env import DebulkEnv

    cfg = yaml.safe_load(open(cfg_path))
    env_fn = lambda: DebulkEnv(cfg, device=device)
    env_raw = DummyVecEnv([env_fn])
    env_mon = VecMonitor(env_raw)

    if vecnorm_path and os.path.exists(vecnorm_path):
        env_norm = VecNormalize.load(vecnorm_path, env_mon)
        env_norm.training = False
        env_norm.norm_reward = False
    else:
        print("WARN: no vecnormalize pkl — running without normalization (wrong for trained policies)")
        env_norm = env_mon

    model = PPO.load(ckpt_path, env=env_norm, device="cpu")

    print(f"Evaluating {arm_label} seed={seed_label}: {episodes} episodes ...", flush=True)
    rng = np.random.default_rng(int(seed_label) if seed_label.isdigit() else 0)

    rows = []
    for ep in range(episodes):
        # Per-episode varied reset via default RNG in env.reset (no per-ep seed set here
        # — env uses internal rng; we rely on it for variation)
        metrics = _run_single_episode(env_raw, env_norm, model, rng)
        metrics["arm"] = arm_label
        metrics["seed"] = seed_label
        metrics["ep"] = ep
        rows.append(metrics)
        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1}/{episodes}  "
                  f"cap_dmg@80={metrics['capsule_dmg_at_80']:.3f}  "
                  f"interior={metrics['final_interior_cov']:.3f}  "
                  f"catastrophic={metrics['catastrophic']}", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    fields = ["arm", "seed", "ep", "final_interior_cov", "capsule_dmg_at_80",
              "steps_to_80", "catastrophic", "return", "episode_len",
              "terminated_by_completion", "terminated_by_catastrophe"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"Wrote {len(rows)} rows → {out_path}")

    # Quick summary
    cap_dmgs = [r["capsule_dmg_at_80"] for r in rows]
    interiors = [r["final_interior_cov"] for r in rows]
    catas = [r["catastrophic"] for r in rows]
    print(f"\n=== Summary: arm={arm_label} seed={seed_label} over {episodes} eps ===")
    print(f"  capsule_dmg@80:    mean={np.mean(cap_dmgs):.4f}  "
          f"median={np.median(cap_dmgs):.4f}  std={np.std(cap_dmgs):.4f}")
    print(f"  final_interior:    mean={np.mean(interiors):.4f}  "
          f"median={np.median(interiors):.4f}")
    print(f"  catastrophic rate: {np.mean(catas):.3f}  ({sum(catas)}/{len(catas)})")
    env_norm.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="PPO checkpoint .zip")
    p.add_argument("--vecnorm",    default=None,  help="VecNormalize .pkl")
    p.add_argument("--cfg",        required=True, help="arm YAML config")
    p.add_argument("--episodes",   type=int, default=100)
    p.add_argument("--out",        required=True, help="output CSV path")
    p.add_argument("--arm",        default=None, help="arm label (else derived from cfg filename)")
    p.add_argument("--seed",       default="0",  help="seed label")
    p.add_argument("--device",     default="cuda:0")
    args = p.parse_args()

    arm_label = args.arm or os.path.basename(args.cfg).replace(".yaml", "").split("_")[-1]
    evaluate(args.cfg, args.checkpoint, args.vecnorm, args.episodes,
             args.out, arm_label, args.seed, args.device)


if __name__ == "__main__":
    main()
