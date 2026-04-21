"""PPO training with vectorized P(F) debulking environment.

N=4 parallel envs in one batched GPU solver (FF-SRL pattern).

Usage:
    python warp_sim/train_ppo_vec.py [--config warp_sim/config/debulk_ppo_vec.yaml]
"""
import os
import sys
import argparse
import yaml
import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pf_shape_matching_warp"))


def train(cfg):
    wp.init()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from warp_sim.envs.vec_debulk_sb3 import SB3VecDebulkEnv

    num_envs = cfg.get("num_envs", 4)
    env = SB3VecDebulkEnv(cfg, num_envs=num_envs, device="cuda:0")

    tcfg = cfg.get("training", {})

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=tcfg.get("learning_rate", 3e-4),
        n_steps=tcfg.get("n_steps", 128),
        batch_size=tcfg.get("batch_size", 128),
        n_epochs=tcfg.get("n_epochs", 10),
        gamma=tcfg.get("gamma", 0.99),
        gae_lambda=tcfg.get("gae_lambda", 0.95),
        clip_range=tcfg.get("clip_range", 0.2),
        ent_coef=tcfg.get("ent_coef", 0.01),
        vf_coef=tcfg.get("vf_coef", 0.5),
        max_grad_norm=tcfg.get("max_grad_norm", 0.5),
        policy_kwargs=dict(net_arch=tcfg.get("net_arch", [256, 256])),
        verbose=1,
        tensorboard_log="./tb_logs/",
        device="cpu",  # PPO on CPU, physics on GPU
    )

    total_timesteps = tcfg.get("total_timesteps", 500000)

    checkpoint_cb = CheckpointCallback(
        save_freq=tcfg.get("save_freq", 50000),
        save_path="./checkpoints/",
        name_prefix="ppo_vec_debulk",
    )

    print(f"\nPPO Vectorized Training: {total_timesteps} steps, {num_envs} envs")
    print(f"  LR={model.learning_rate}, n_steps={model.n_steps}")
    print(f"  batch_size={model.batch_size}, n_epochs={model.n_epochs}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save("ppo_vec_debulk_final")
    print(f"\nTraining complete. Model saved: ppo_vec_debulk_final.zip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="warp_sim/config/debulk_ppo_vec.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
