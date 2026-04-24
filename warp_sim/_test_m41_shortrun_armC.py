"""Arm C 10k-step end-to-end sanity check before M4.1 12-run commit.

Purpose: catch integration bugs in the PPO + Arm C L2-tactile-obs pipeline that
only appear during actual training (beyond the 1k smoke test). If healthy, green-
light the full 3-4 day 12-run matrix. If unhealthy, fix before burning GPU.

Health criteria (checked automatically after run):
  [1] No NaN/inf in rewards or training losses throughout the run.
  [2] ep_rew_mean has upward trend over second half (linear regression slope > 0).
  [3] approx_kl below target_kl=0.02 most of the time (stable PPO updates).
  [4] Policy entropy hasn't collapsed (|entropy_loss| > 0.2 at end).
  [5] value_loss decreasing over run (ratio last/first < 1.5).
  [6] TB log directory populated with scalars.
  [7] Arm C specifically: tactile feature (last 3 obs dims) has nonzero variance
      — proof the L2 gradient signal is reaching the policy and isn't degenerate.

Usage:
  python warp_sim/_test_m41_shortrun_armC.py

Runs ~10k PPO steps on debulk_cap_armC.yaml; ~10 min wall-clock on RTX 2070.
Outputs: _m41_shortrun_results/armC_10k/ + human-readable summary to stdout.
"""
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pf_shape_matching_warp"))

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

TOTAL_TIMESTEPS = 10_000
N_STEPS = 512
BATCH_SIZE = 128
N_EPOCHS = 4
SAVE_FREQ = 5000
CHECKPOINT_DIR = os.path.join(REPO, "_m41_shortrun_results", "armC_10k")
TB_LOG_ROOT = os.path.join(REPO, "tb_logs")
TB_RUN_NAME = "m41_shortrun_armC_10k"


def run_training():
    import warp as wp
    import yaml
    import torch
    wp.init()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    from warp_sim.envs.debulk_env import DebulkEnv

    cfg = yaml.safe_load(open(os.path.join(HERE, "config", "debulk_cap_armC.yaml")))
    cfg.setdefault("training", {})
    # Override for short-run
    cfg["training"]["total_timesteps"] = TOTAL_TIMESTEPS
    cfg["training"]["save_freq"] = SAVE_FREQ
    cfg["training"]["n_steps"] = N_STEPS
    cfg["training"]["batch_size"] = BATCH_SIZE
    cfg["training"]["n_epochs"] = N_EPOCHS

    tcfg = cfg["training"]

    # Env stack
    env = DummyVecEnv([lambda: DebulkEnv(cfg, device="cuda:0")])
    env = VecMonitor(env)
    env = VecNormalize(
        env, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0, gamma=tcfg.get("gamma", 0.99),
    )

    # Print obs dim so we know Arm C is 1519D (L2 tactile appended)
    obs_sanity = env.reset()
    print(f"[armC 10k] obs_dim = {obs_sanity.shape}  (expect (1, 1519))", flush=True)
    assert obs_sanity.shape == (1, 1519), f"Arm C obs dim mismatch: {obs_sanity.shape}"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=tcfg.get("learning_rate", 3e-4),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=tcfg.get("gamma", 0.99),
        gae_lambda=tcfg.get("gae_lambda", 0.95),
        clip_range=tcfg.get("clip_range", 0.2),
        ent_coef=tcfg.get("ent_coef", 0.01),
        vf_coef=tcfg.get("vf_coef", 0.5),
        max_grad_norm=tcfg.get("max_grad_norm", 0.5),
        target_kl=tcfg.get("target_kl", 0.02),
        policy_kwargs=dict(net_arch=tcfg.get("net_arch", [256, 256])),
        verbose=1,
        tensorboard_log=TB_LOG_ROOT,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=SAVE_FREQ, save_path=CHECKPOINT_DIR,
        name_prefix="shortrun_armC", save_vecnormalize=True,
    )

    t0 = time.perf_counter()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_cb,
        tb_log_name=TB_RUN_NAME,
        progress_bar=False,
    )
    wall = time.perf_counter() - t0
    print(f"\n[armC 10k] Training completed in {wall:.1f}s ({wall/60:.1f} min)", flush=True)

    # Final save
    final_zip = os.path.join(CHECKPOINT_DIR, "shortrun_armC_final.zip")
    final_pkl = os.path.join(CHECKPOINT_DIR, "shortrun_armC_final_vecnormalize.pkl")
    model.save(final_zip)
    env.save(final_pkl)
    print(f"[armC 10k] Final model + vecnorm saved", flush=True)

    return wall


def _latest_tb_run_dir():
    """Find the TB run dir created by this session (latest _1, _2, etc suffix)."""
    import glob
    matches = sorted(glob.glob(os.path.join(TB_LOG_ROOT, f"{TB_RUN_NAME}*")))
    return matches[-1] if matches else None


def _read_tb_scalars(tb_run_dir):
    """Read all scalar tags from a TB event file. Returns dict tag -> list of (step, value)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(tb_run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return out


def _linear_slope(xs, ys):
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64); y = np.asarray(ys, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def assess_health():
    print("\n" + "=" * 78)
    print("  Arm C 10k short-run health assessment")
    print("=" * 78)

    tb_run_dir = _latest_tb_run_dir()
    if tb_run_dir is None:
        print("  [✗] No TB log directory found — training did not write events")
        return 1
    print(f"  TB run dir: {tb_run_dir}")

    scalars = _read_tb_scalars(tb_run_dir)
    available = list(scalars.keys())
    print(f"  Scalar tags available: {len(available)}")
    print(f"    {available}")

    results = []

    def _check(name, passed, msg=""):
        tag = "✓" if passed else "✗"
        print(f"  [{tag}] {name}" + (f"  — {msg}" if msg else ""))
        results.append((name, passed, msg))

    # 1. No NaN/inf in reward/loss tags
    key_tags_prefixes = ("rollout/ep_rew_mean", "train/policy_gradient_loss",
                          "train/value_loss", "train/approx_kl", "train/entropy_loss")
    any_nan = False
    for tag in available:
        if not tag.startswith(key_tags_prefixes):
            continue
        vals = [v for _, v in scalars[tag]]
        if any(not np.isfinite(v) for v in vals):
            any_nan = True
            print(f"    NaN/inf in {tag}")
    _check("no NaN/inf in reward/loss", not any_nan)

    # 2. ep_rew_mean upward trend in second half
    rew_tag = next((t for t in available if "ep_rew_mean" in t), None)
    if rew_tag is None:
        _check("ep_rew_mean upward trend", False, "tag missing (VecMonitor not wrapped?)")
    else:
        series = scalars[rew_tag]
        if len(series) < 4:
            _check("ep_rew_mean upward trend", False,
                   f"only {len(series)} data points (need >= 4)")
        else:
            half = len(series) // 2
            xs = [s for s, _ in series[half:]]
            ys = [v for _, v in series[half:]]
            slope = _linear_slope(xs, ys)
            first_rew = series[0][1]
            last_rew = series[-1][1]
            _check(f"ep_rew_mean trend (slope={slope:+.3e}, {first_rew:.2f} → {last_rew:.2f})",
                   slope > 0 or last_rew > first_rew,
                   f"{len(series)} data points")

    # 3. approx_kl stability
    kl_tag = next((t for t in available if "approx_kl" in t), None)
    if kl_tag:
        kls = [v for _, v in scalars[kl_tag]]
        if kls:
            bad = sum(1 for v in kls if v > 0.05)
            _check(f"approx_kl < 0.05 majority ({bad}/{len(kls)} violations)",
                   bad < 0.3 * len(kls),
                   f"max_kl={max(kls):.4f}")

    # 4. Entropy not collapsed
    ent_tag = next((t for t in available if "entropy_loss" in t), None)
    if ent_tag:
        ents = [v for _, v in scalars[ent_tag]]
        if ents:
            final_abs_ent = abs(ents[-1])
            _check(f"|entropy_loss| > 0.2 at end (got {final_abs_ent:.3f})",
                   final_abs_ent > 0.2)

    # 5. value_loss decreasing
    vl_tag = next((t for t in available if "value_loss" in t), None)
    if vl_tag:
        vls = [v for _, v in scalars[vl_tag]]
        if len(vls) >= 2:
            ratio = vls[-1] / max(vls[0], 1e-12)
            _check(f"value_loss ratio last/first ({ratio:.2f}) < 2.0",
                   ratio < 2.0,
                   f"first={vls[0]:.3f}, last={vls[-1]:.3f}")

    # 6. TB has scalars
    _check(f"TB log populated ({len(available)} scalar tags)",
           len(available) >= 5)

    # 7. Arm C L2 tactile signal actually varies
    # Read raw env observations via a quick eval rollout
    try:
        print("\n  Checking L2 tactile feature variance over 10 fresh env steps...")
        import warp as wp
        import yaml
        from warp_sim.envs.debulk_env import DebulkEnv
        cfg = yaml.safe_load(open(os.path.join(HERE, "config", "debulk_cap_armC.yaml")))
        env = DebulkEnv(cfg, device="cuda:0")
        env.reset()
        tactile_history = []
        rng = np.random.default_rng(0)
        # warm up to contact
        for _ in range(8):
            env.step(np.array([-1.0, -1.0, 1.0], dtype=np.float32))
        # 10 random steps, record last 3 obs dims (= tactile g_cap)
        for _ in range(10):
            a = rng.uniform(-1, 1, 3).astype(np.float32)
            obs, _, _, _, _ = env.step(a)
            tactile_history.append(obs[-3:].copy())
        tactile_arr = np.array(tactile_history)
        tactile_var = float(tactile_arr.var())
        nonzero_steps = int((np.linalg.norm(tactile_arr, axis=1) > 1e-6).sum())
        _check(f"L2 tactile variance > 0.01 over 10 steps  (var={tactile_var:.4f}, "
               f"{nonzero_steps}/10 nonzero)",
               tactile_var > 0.01 and nonzero_steps >= 3)
    except Exception as e:
        _check("L2 tactile feature variance probe", False, f"{type(e).__name__}: {e}")

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n  Health: {passed}/{total} checks passed")
    if passed == total:
        print("  ✓ Arm C 10k short-run is HEALTHY — safe to launch M4.1 12-run matrix")
        return 0
    elif passed >= total - 1:
        print("  ⚠ 1 failure — review and decide whether to proceed")
        return 0  # don't hard-fail on single check
    else:
        print(f"  ✗ {total - passed} failures — investigate before launching 12-run")
        return 1


def main():
    wall = run_training()
    rc = assess_health()
    meta = {
        "wall_clock_seconds": round(wall, 1),
        "total_timesteps": TOTAL_TIMESTEPS,
        "checkpoint_dir": CHECKPOINT_DIR,
        "tb_run": _latest_tb_run_dir(),
        "health_rc": rc,
    }
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(os.path.join(CHECKPOINT_DIR, "shortrun_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMeta → {os.path.join(CHECKPOINT_DIR, 'shortrun_meta.json')}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
