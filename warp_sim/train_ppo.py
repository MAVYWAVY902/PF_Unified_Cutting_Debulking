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


def train(cfg):
    wp.init()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    from warp_sim.envs.debulk_env import DebulkEnv

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
        policy_kwargs=dict(
            net_arch=tcfg.get("net_arch", [256, 256]),
        ),
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    total_timesteps = tcfg.get("total_timesteps", 100000)

    # Callbacks — also save vecnormalize stats so intermediate checkpoints replayable
    checkpoint_cb = CheckpointCallback(
        save_freq=tcfg.get("save_freq", 10000),
        save_path="./checkpoints/",
        name_prefix="ppo_debulk",
        save_vecnormalize=use_vecnorm,   # writes vecnormalize_<step>.pkl alongside zip
    )

    print(f"\nPPO Training: {total_timesteps} timesteps")
    print(f"  LR={model.learning_rate}, n_steps={model.n_steps}")
    print(f"  batch_size={model.batch_size}, n_epochs={model.n_epochs}")
    print(f"  gamma={model.gamma}, clip={model.clip_range}")
    print(f"  target_kl={model.target_kl}  (None = no early-stop; 0.02 = MuJoCo-like safety)")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save("ppo_debulk_final")
    if use_vecnorm:
        env.save("vecnormalize_final.pkl")
        print("\nTraining complete. Saved: ppo_debulk_final.zip + vecnormalize_final.pkl")
        print("  IMPORTANT: rollout/eval must load BOTH files together (see rollout_viz_mesh.py).")
    else:
        print("\nTraining complete. Model saved: ppo_debulk_final.zip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="warp_sim/config/debulk_ppo.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
