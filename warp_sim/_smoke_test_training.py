"""M3.6 bridge — smoke-test the PPO training pipeline on all 4 capsule arms.

PURPOSE: catch integration bugs BEFORE committing 3-4 days of GPU to M4.1 12-run
matrix. Phase 6 validated env correctness; this validates the PPO → VecNormalize →
CheckpointCallback → TB stack actually flows for each arm's obs/compute shape.

USAGE:
    python warp_sim/_smoke_test_training.py              # orchestrator: runs 4 arms
    python warp_sim/_smoke_test_training.py --arm C      # single-arm worker
    python warp_sim/_smoke_test_training.py --quick      # 1k steps/arm (~5 min total)

Per-arm integration checks performed:
  [1] PPO MlpPolicy accepts obs_dim (1516 for A, 1519 for X/Y/C)
  [2] VecNormalize running stats applied without nan/inf
  [3] tb_logs/cap_smoke_arm{A,X,Y,C}/ scalar tags present (ep_rew_mean etc.)
  [4] Checkpoint zip + vecnormalize pkl both saved at save_freq
  [5] Checkpoint round-trip: reload → 5 deterministic eval steps → reward finite
  [6] GPU memory stable (Arm C: wp.Tape doesn't leak over N_STEPS)
  [7] Wall-clock per arm (benchmark: confirm Plan estimates A/X≈1x, C≈2x, Y≈6x)

Each arm runs in a subprocess for CUDA isolation (one arm's OOM can't poison the next).
"""
import os
import sys
import argparse
import json
import subprocess
import time
import shutil
import gc

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONFIG_DIR = os.path.join(HERE, "config")
ARMS = ["A", "X", "Y", "C"]

# Smoke-test overrides: small total_timesteps to keep suite under ~15 min.
# Must be > n_steps × n_epochs to trigger at least 1 full PPO training iteration.
# n_steps=512, n_epochs=4 → 2048 ≥ 1 iter; use 3000 to get ~6 rollouts + 3 checkpoints.
SMOKE_TIMESTEPS_FULL = 3000
SMOKE_TIMESTEPS_QUICK = 1000
SMOKE_SAVE_FREQ = 1000     # at 3k steps, 2-3 checkpoints per arm


# ============================================================================
# Single-arm worker (spawned as subprocess)
# ============================================================================

def run_one_arm(arm, total_timesteps, json_out):
    """Run one arm's smoke training + integration checks; dump JSON result."""
    result = {
        "arm": arm,
        "total_timesteps": total_timesteps,
        "checks": {},
        "metrics": {},
        "passed": False,
        "error": None,
    }
    t0 = time.perf_counter()

    try:
        sys.path.insert(0, REPO)
        sys.path.insert(0, os.path.join(REPO, "pf_shape_matching_warp"))

        import warp as wp
        import yaml
        import numpy as np
        wp.init()

        # Stash GPU mem before import (after wp.init so CUDA ctx already up)
        import torch
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1e6
        result["metrics"]["gpu_mem_before_MB"] = round(mem_before, 1)

        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
        from warp_sim.envs.debulk_env import DebulkEnv

        # Load arm config, inject smoke overrides
        cfg_path = os.path.join(CONFIG_DIR, f"debulk_cap_arm{arm}.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("training", {})
        cfg["training"]["total_timesteps"] = total_timesteps
        cfg["training"]["save_freq"] = SMOKE_SAVE_FREQ
        cfg["training"]["n_steps"] = 512
        cfg["training"]["batch_size"] = 128

        tcfg = cfg["training"]
        expected_obs_dim = 1519 if cfg["obs_mode"] != "baseline" else 1516

        # --- Env stack ---
        env = DummyVecEnv([lambda: DebulkEnv(cfg, device="cuda:0")])
        env = VecMonitor(env)
        env = VecNormalize(env,
                           norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0,
                           gamma=tcfg.get("gamma", 0.99))

        obs = env.reset()
        result["checks"]["obs_dim_matches"] = (obs.shape == (1, expected_obs_dim))
        result["metrics"]["obs_dim_actual"] = int(obs.shape[1])
        result["metrics"]["obs_dim_expected"] = expected_obs_dim

        # --- PPO with per-arm TB log & checkpoint paths ---
        tb_dir = os.path.join(REPO, "tb_logs")
        ckpt_dir = os.path.join(REPO, "checkpoints", f"smoke_arm{arm}")
        os.makedirs(ckpt_dir, exist_ok=True)

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=tcfg.get("learning_rate", 3e-4),
            n_steps=tcfg["n_steps"],
            batch_size=tcfg["batch_size"],
            n_epochs=tcfg.get("n_epochs", 4),
            gamma=tcfg.get("gamma", 0.99),
            gae_lambda=tcfg.get("gae_lambda", 0.95),
            clip_range=tcfg.get("clip_range", 0.2),
            ent_coef=tcfg.get("ent_coef", 0.01),
            vf_coef=tcfg.get("vf_coef", 0.5),
            max_grad_norm=tcfg.get("max_grad_norm", 0.5),
            target_kl=tcfg.get("target_kl", 0.02),
            policy_kwargs=dict(net_arch=tcfg.get("net_arch", [256, 256])),
            verbose=0,
            tensorboard_log=tb_dir,
        )

        # Verify PPO's MlpPolicy first-layer shape matches obs_dim
        first_layer = model.policy.mlp_extractor.policy_net[0]
        result["checks"]["ppo_first_layer_shape_matches"] = (
            first_layer.in_features == expected_obs_dim
        )
        result["metrics"]["ppo_first_layer_in"] = int(first_layer.in_features)

        # --- Train ---
        t_train = time.perf_counter()
        checkpoint_cb = CheckpointCallback(
            save_freq=SMOKE_SAVE_FREQ, save_path=ckpt_dir,
            name_prefix=f"smoke_arm{arm}", save_vecnormalize=True,
        )
        model.learn(total_timesteps=total_timesteps,
                    callback=checkpoint_cb,
                    tb_log_name=f"cap_smoke_arm{arm}",
                    progress_bar=False)
        result["metrics"]["train_seconds"] = round(time.perf_counter() - t_train, 1)

        # --- Post-train mem ---
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / 1e6
        result["metrics"]["gpu_mem_after_MB"] = round(mem_after, 1)
        result["metrics"]["gpu_mem_delta_MB"] = round(mem_after - mem_before, 1)

        # --- Check 4: artifacts exist ---
        import glob
        zips = sorted(glob.glob(os.path.join(ckpt_dir, f"smoke_arm{arm}_*_steps.zip")))
        pkls = sorted(glob.glob(os.path.join(ckpt_dir, f"smoke_arm{arm}_vecnormalize_*_steps.pkl")))
        result["checks"]["checkpoints_saved"] = (len(zips) >= 1 and len(pkls) >= 1)
        result["metrics"]["n_checkpoints"] = len(zips)
        result["metrics"]["n_vecnorm_pkls"] = len(pkls)

        # --- Check 3: TB log dir populated ---
        tb_run_glob = glob.glob(os.path.join(tb_dir, f"cap_smoke_arm{arm}*"))
        if tb_run_glob:
            tb_files = []
            for d in tb_run_glob:
                tb_files.extend(glob.glob(os.path.join(d, "events.out.tfevents*")))
            result["checks"]["tb_log_written"] = (
                len(tb_files) > 0 and
                all(os.path.getsize(f) > 100 for f in tb_files)
            )
            result["metrics"]["tb_event_files"] = len(tb_files)
        else:
            result["checks"]["tb_log_written"] = False

        # --- Save final artifacts for round-trip test ---
        final_zip = os.path.join(ckpt_dir, f"smoke_arm{arm}_final.zip")
        final_pkl = os.path.join(ckpt_dir, f"smoke_arm{arm}_final_vecnormalize.pkl")
        model.save(final_zip)
        env.save(final_pkl)

        # Tear down training stack
        env.close()
        del model, env
        gc.collect()
        torch.cuda.empty_cache()

        # --- Check 5: round-trip (reload + eval 5 steps, reward finite) ---
        eval_env = DummyVecEnv([lambda: DebulkEnv(cfg, device="cuda:0")])
        eval_env = VecMonitor(eval_env)
        eval_env = VecNormalize.load(final_pkl, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        reloaded_model = PPO.load(final_zip, env=eval_env, device="cpu")
        obs = eval_env.reset()
        eval_rewards = []
        for _ in range(5):
            action, _ = reloaded_model.predict(obs, deterministic=True)
            obs, r, dones, infos = eval_env.step(action)
            eval_rewards.append(float(r[0]))
        result["metrics"]["eval_rewards"] = [round(r, 3) for r in eval_rewards]
        result["checks"]["roundtrip_eval_finite"] = all(
            np.isfinite(r) for r in eval_rewards
        )

        eval_env.close()
        del reloaded_model, eval_env
        gc.collect()
        torch.cuda.empty_cache()

        # --- Check 6: GPU mem delta small (< 500 MB residual after train)
        # Baseline of ~400 MB is Warp kernel cache + PyTorch CUDA ctx. Anything
        # > 800 MB delta suggests Warp.Tape arrays aren't being freed.
        result["checks"]["gpu_mem_stable"] = (
            result["metrics"]["gpu_mem_delta_MB"] < 800
        )

        # --- Aggregate pass/fail ---
        result["passed"] = all(result["checks"].values())
        result["metrics"]["total_seconds"] = round(time.perf_counter() - t0, 1)

    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        result["passed"] = False

    # Dump JSON
    with open(json_out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[arm {arm}] result written to {json_out}")
    return 0 if result["passed"] else 1


# ============================================================================
# Orchestrator (spawns one subprocess per arm)
# ============================================================================

def orchestrate(total_timesteps):
    print("="*70)
    print(f"  PPO smoke test — 4 arms × {total_timesteps} steps (subprocess mode)")
    print("="*70)

    results_dir = os.path.join(REPO, "_smoke_results")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    # Clean out previous smoke TB runs and checkpoint dirs so this test is clean
    import glob
    for d in glob.glob(os.path.join(REPO, "tb_logs", "cap_smoke_*")):
        shutil.rmtree(d, ignore_errors=True)
    for d in glob.glob(os.path.join(REPO, "checkpoints", "smoke_arm*")):
        shutil.rmtree(d, ignore_errors=True)

    results = {}
    for arm in ARMS:
        json_out = os.path.join(results_dir, f"arm{arm}.json")
        print(f"\n>>> Spawning arm {arm} subprocess...")
        t0 = time.perf_counter()
        rc = subprocess.call(
            [sys.executable, os.path.abspath(__file__),
             "--arm", arm,
             "--total-timesteps", str(total_timesteps),
             "--json-out", json_out],
            cwd=REPO,
        )
        elapsed = time.perf_counter() - t0
        if os.path.exists(json_out):
            with open(json_out) as f:
                results[arm] = json.load(f)
        else:
            results[arm] = {
                "arm": arm, "passed": False,
                "error": f"subprocess exited rc={rc} without writing JSON",
                "checks": {}, "metrics": {},
            }
        print(f"<<< Arm {arm} subprocess done (rc={rc}, {elapsed:.1f}s)")

    # Summary table
    print("\n" + "="*70)
    print("  Smoke test summary")
    print("="*70)
    header = f"{'Arm':<4} {'obs_dim':<8} {'ppo_in':<7} {'train_s':<8} {'mem_Δ_MB':<10} {'ckpts':<6} {'pass':<5}"
    print(header)
    print("-" * len(header))
    all_passed = True
    for arm in ARMS:
        r = results[arm]
        m = r.get("metrics", {})
        passed = r.get("passed", False)
        all_passed = all_passed and passed
        row = (
            f"{arm:<4} "
            f"{m.get('obs_dim_actual', '-'):<8} "
            f"{m.get('ppo_first_layer_in', '-'):<7} "
            f"{str(m.get('train_seconds', '-')):<8} "
            f"{str(m.get('gpu_mem_delta_MB', '-')):<10} "
            f"{m.get('n_checkpoints', '-'):<6} "
            f"{'✓' if passed else '✗':<5}"
        )
        print(row)
        if not passed:
            failed = [k for k, v in r.get("checks", {}).items() if not v]
            if failed:
                print(f"     failed checks: {', '.join(failed)}")
            if r.get("error"):
                print(f"     ERROR: {r['error']}")

    # Compute ratios for benchmark
    if all(r.get("metrics", {}).get("train_seconds") for r in results.values()):
        tA = results["A"]["metrics"]["train_seconds"]
        print(f"\nWall-clock ratios (Arm A = 1.0×):")
        for arm in ARMS:
            t = results[arm]["metrics"]["train_seconds"]
            print(f"  Arm {arm}: {t/tA:.2f}×  ({t}s)")
        print("  Plan estimates: A/X ≈ 1×, C ≈ 2×, Y ≈ 6×")

    print("\n" + "="*70)
    print(f"  Result: {'ALL PASSED' if all_passed else 'FAILED'}")
    print("="*70)
    return 0 if all_passed else 1


# ============================================================================
# Entry
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, choices=ARMS,
                    help="Internal: run one arm as subprocess.")
    ap.add_argument("--total-timesteps", type=int, default=SMOKE_TIMESTEPS_FULL)
    ap.add_argument("--quick", action="store_true",
                    help=f"Use {SMOKE_TIMESTEPS_QUICK}-step runs (~5 min total)")
    ap.add_argument("--json-out", default=None,
                    help="Internal: where worker writes its result JSON.")
    args = ap.parse_args()

    if args.arm is not None:
        assert args.json_out is not None, "worker mode requires --json-out"
        return run_one_arm(args.arm, args.total_timesteps, args.json_out)

    timesteps = SMOKE_TIMESTEPS_QUICK if args.quick else args.total_timesteps
    return orchestrate(timesteps)


if __name__ == "__main__":
    sys.exit(main())
