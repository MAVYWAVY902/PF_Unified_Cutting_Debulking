"""PPO training for P(F) surgical debulking.

Uses stable-baselines3 PPO with standard Gymnasium interface.
No differentiable physics needed — P(F) provides the physics model and reward.

Usage:
    python warp_sim/train_ppo.py [--config warp_sim/config/debulk_ppo.yaml]
"""
import os
import sys
import argparse
import yaml
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pf_shape_matching_warp"))


def train(cfg, seed=None, run_name=None):
    wp.init()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    from warp_sim.envs.debulk_env import DebulkEnv

    if seed is not None:
        set_random_seed(int(seed))
        print(f"[seed] Set SB3/torch/numpy random seed = {seed}")

    # PPO config
    tcfg = cfg.get("training", {})

    # Create env, then stack wrappers in correct order:
    #   DummyVecEnv → VecMonitor → VecNormalize → PPO
    # VecMonitor BEFORE VecNormalize: Monitor must see RAW rewards to log episode stats.
    # Without VecMonitor, rollout/ep_rew_mean tag is silently omitted from TB logs
    # (PPO_12 bug — fixed here).
    base_env_fn = lambda: DebulkEnv(cfg, device="cuda:0")
    env = DummyVecEnv([base_env_fn])
    env = VecMonitor(env)
    print("[env] VecMonitor enabled (ep_rew_mean, ep_len_mean will be logged)")

    # VecNormalize: running-avg normalize obs + reward. FIXES the PPO_11
    # value-function collapse (reward ±7000 → critic couldn't fit).
    use_vecnorm = tcfg.get("use_vecnormalize", True)
    if use_vecnorm:
        env = VecNormalize(
            env,
            norm_obs=tcfg.get("norm_obs", True),
            norm_reward=tcfg.get("norm_reward", True),
            clip_obs=tcfg.get("clip_obs", 10.0),
            clip_reward=tcfg.get("clip_reward", 10.0),
            gamma=tcfg.get("gamma", 0.99),
        )
        print(f"[env] VecNormalize enabled: norm_obs={env.norm_obs}, "
              f"norm_reward={env.norm_reward}, clip_reward={env.clip_reward}")
    else:
        print("[env] VecNormalize DISABLED (config use_vecnormalize=false)")

    # Per-run paths — if run_name provided, use dedicated TB + checkpoint dirs
    # so multi-seed / multi-arm runs don't collide.
    import os
    if run_name:
        ckpt_dir = f"./checkpoints/{run_name}"
        os.makedirs(ckpt_dir, exist_ok=True)
        tb_log_name = run_name
    else:
        ckpt_dir = "./checkpoints/"
        tb_log_name = None  # SB3 auto-names (PPO_1, PPO_2, ...)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=tcfg.get("learning_rate", 3e-4),
        n_steps=tcfg.get("n_steps", 256),
        batch_size=tcfg.get("batch_size", 64),
        n_epochs=tcfg.get("n_epochs", 10),
        gamma=tcfg.get("gamma", 0.99),
        gae_lambda=tcfg.get("gae_lambda", 0.95),
        clip_range=tcfg.get("clip_range", 0.2),
        ent_coef=tcfg.get("ent_coef", 0.01),
        vf_coef=tcfg.get("vf_coef", 0.5),
        max_grad_norm=tcfg.get("max_grad_norm", 0.5),
        target_kl=tcfg.get("target_kl", None),   # FIXES PPO_12 KL blow-up (default None = disabled)
        seed=int(seed) if seed is not None else None,
        policy_kwargs=dict(
            net_arch=tcfg.get("net_arch", [256, 256]),
        ),
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    total_timesteps = tcfg.get("total_timesteps", 100000)

    # Callbacks — also save vecnormalize stats so intermediate checkpoints replayable
    name_prefix = run_name if run_name else "ppo_debulk"
    checkpoint_cb = CheckpointCallback(
        save_freq=tcfg.get("save_freq", 10000),
        save_path=ckpt_dir,
        name_prefix=name_prefix,
        save_vecnormalize=use_vecnorm,   # writes vecnormalize_<step>.pkl alongside zip
    )

    print(f"\nPPO Training: {total_timesteps} timesteps")
    print(f"  Run name: {run_name or '(default: SB3 auto-numbered)'}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print(f"  LR={model.learning_rate}, n_steps={model.n_steps}")
    print(f"  batch_size={model.batch_size}, n_epochs={model.n_epochs}")
    print(f"  gamma={model.gamma}, clip={model.clip_range}")
    print(f"  target_kl={model.target_kl}  (None = no early-stop; 0.02 = MuJoCo-like safety)")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        tb_log_name=tb_log_name,
        progress_bar=True,
    )

    # Final save under run_name prefix so multi-run output isn't clobbered
    final_zip = os.path.join(ckpt_dir, f"{name_prefix}_final.zip")
    final_pkl = os.path.join(ckpt_dir, f"{name_prefix}_final_vecnormalize.pkl")
    model.save(final_zip)
    if use_vecnorm:
        env.save(final_pkl)
        print(f"\nTraining complete. Saved: {final_zip} + {final_pkl}")
        print("  IMPORTANT: rollout/eval must load BOTH files together (see rollout_viz_mesh.py).")
    else:
        print(f"\nTraining complete. Model saved: {final_zip}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="warp_sim/config/debulk_ppo.yaml")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (SB3 + torch + numpy). If None, unseeded.")
    parser.add_argument("--run-name", default=None,
                        help="Run identifier (e.g. cap_armC_diffM_seed0). Used for "
                             "TB subdir, checkpoint dir, and final file prefix. "
                             "If None, SB3 auto-numbers.")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, seed=args.seed, run_name=args.run_name)


if __name__ == "__main__":
    main()
