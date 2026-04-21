"""SAPO training for P(F) surgical debulking.

Selective differentiation (AHAC-style):
- Full physics runs WITHOUT tape (includes SVD shape matching)
- Collision + reward runs WITH internal tape (stable gradient)
- env.step() returns action_grad directly — no outer tape needed
- Critic provides long-horizon value via TD(lambda) advantages

Usage:
    python warp_sim/train_sapo.py [--config warp_sim/config/debulk_sapo.yaml]
"""
import os
import sys
import copy
import argparse
import math
import yaml
import numpy as np

import warp as wp

torch = None


def _ensure_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch


# ============== Actor-Critic ==============

def build_actor_critic(obs_dim, act_dim, hidden_dim):
    _ensure_torch()
    nn = torch.nn
    actor_mean = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim, act_dim),
    )
    actor_log_std = nn.Parameter(torch.zeros(act_dim))
    critic = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim, 1),
    )
    return actor_mean, actor_log_std, critic


def _compute_gae(rewards, values, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
    return advantages, returns


# ============== Training ==============

def train(cfg):
    _ensure_torch()
    wp.init()
    device = "cuda:0"

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from warp_sim.envs.debulk_env import DebulkEnv
    env = DebulkEnv(cfg, device=device)

    tcfg = cfg.get("training", {})
    num_epochs = tcfg.get("num_epochs", 500)
    horizon = cfg.get("horizon_len", 8)
    gamma = tcfg.get("gamma", 0.99)
    lam = tcfg.get("lam", 0.95)
    max_grad_norm = tcfg.get("max_grad_norm", 0.5)
    actor_lr = tcfg.get("actor_lr", 2e-3)
    critic_lr = tcfg.get("critic_lr", 5e-4)
    actor_beta1 = tcfg.get("actor_beta1", 0.7)
    hidden_dim = tcfg.get("hidden_dim", 256)
    critic_iterations = tcfg.get("critic_iterations", 16)
    log_interval = tcfg.get("log_interval", 10)
    save_interval = tcfg.get("save_interval", 100)
    init_alpha = tcfg.get("init_alpha", 1.0)
    target_entropy = tcfg.get("target_entropy", -1.5)
    tau = tcfg.get("target_critic_tau", 0.005)
    sim_grad_clip = tcfg.get("sim_grad_clip", 10.0)

    obs_dim = env.num_obs
    actor_mean, actor_log_std, critic = build_actor_critic(obs_dim, env.num_act, hidden_dim)
    actor_mean.cuda()
    critic.cuda()
    actor_log_std = torch.nn.Parameter(actor_log_std.detach().cuda())
    critic_target = copy.deepcopy(critic)
    critic_target.eval()

    actor_params = list(actor_mean.parameters()) + [actor_log_std]
    actor_opt = torch.optim.AdamW(actor_params, lr=actor_lr, betas=(actor_beta1, 0.95))
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=critic_lr, betas=(actor_beta1, 0.95))
    log_alpha = torch.tensor(math.log(init_alpha), requires_grad=True, device="cuda")
    alpha_opt = torch.optim.Adam([log_alpha], lr=5e-3)

    print(f"\nSAPO Training (selective differentiation, H={horizon})")
    print(f"  {num_epochs} epochs, obs_dim={obs_dim}")
    print(f"  Actor LR={actor_lr}, Critic LR={critic_lr}")
    print(f"  gamma={gamma}, lam={lam}, grad_clip={max_grad_norm}")

    best_reward = -float("inf")

    for epoch in range(num_epochs):
        actor_mean.train()

        # ============================================================
        # Phase 1: Rollout — env.step() manages its own internal tape
        # ============================================================
        step_data = []
        obs_dict = env.initialize_trajectory()

        for t in range(horizon):
            wp.synchronize()
            obs_np = env.get_obs_flat_np(obs_dict)
            obs_torch = torch.tensor(obs_np, dtype=torch.float32, device="cuda").unsqueeze(0)

            # Actor forward
            with torch.no_grad():
                act_mean_out = actor_mean(obs_torch)
                std = actor_log_std.exp().clamp(min=0.01, max=1.0)
                dist = torch.distributions.Normal(act_mean_out, std)
                action_torch = dist.sample().squeeze(0)
                action_torch = torch.tanh(action_torch)

            # Convert to Warp
            action_np = action_torch.cpu().numpy().astype(np.float32)
            action_wp = wp.array(
                [wp.vec3(action_np[0], action_np[1], action_np[2])],
                dtype=wp.vec3, device=device, requires_grad=True,
            )

            # env.step() runs full physics + internal tape for collision gradient
            obs_dict, rew_wp, done_wp, extras = env.step(action_wp)

            wp.synchronize()
            reward_val = float(rew_wp.numpy()[0])

            # Get stable action gradient from env (collision-only, no SVD)
            action_grad = extras.get("action_grad", np.zeros(3, dtype=np.float32))

            # Clip gradient norm (preserve direction)
            gnorm = np.linalg.norm(action_grad)
            if gnorm > sim_grad_clip:
                action_grad = action_grad * (sim_grad_clip / gnorm)

            # Critic value
            with torch.no_grad():
                value = critic_target(obs_torch).item()

            step_data.append({
                "obs_np": obs_np.copy(),
                "action_grad": action_grad,
                "reward": reward_val,
                "value": value,
            })

        # ============================================================
        # Phase 2: Advantages
        # ============================================================
        rewards_np = np.array([sd["reward"] for sd in step_data])
        values_np = np.array([sd["value"] for sd in step_data] +
                             [step_data[-1]["value"]])
        advantages, returns = _compute_gae(rewards_np, values_np, gamma, lam)

        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std

        # ============================================================
        # Phase 3: Actor update
        # ============================================================
        actor_opt.zero_grad()
        policy_loss = torch.tensor(0.0, device="cuda")

        for t in range(horizon):
            sd = step_data[t]
            obs_torch = torch.tensor(sd["obs_np"], dtype=torch.float32,
                                     device="cuda").unsqueeze(0)
            act_mean_out = actor_mean(obs_torch)
            std = actor_log_std.exp().clamp(min=0.01, max=1.0)
            dist = torch.distributions.Normal(act_mean_out, std)
            action_sampled = dist.rsample()
            action_bounded = torch.tanh(action_sampled)

            # Sim gradient direction (from env's internal collision tape)
            sim_grad = torch.tensor(sd["action_grad"], device="cuda")
            adv_t = torch.tensor(advantages[t], device="cuda")

            # Policy gradient = advantage × (action · sim_gradient_direction)
            sg_norm = sim_grad.norm()
            if sg_norm > 1e-8:
                sg_dir = sim_grad / sg_norm
            else:
                sg_dir = sim_grad

            policy_loss = policy_loss - adv_t * torch.sum(
                action_bounded.squeeze() * sg_dir.detach())

            # Entropy
            alpha = log_alpha.exp()
            log_prob = dist.log_prob(action_sampled).sum()
            policy_loss = policy_loss + alpha.detach() * log_prob

        policy_loss = policy_loss / float(horizon)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_params, max_grad_norm)
        actor_opt.step()

        # Alpha update
        alpha_opt.zero_grad()
        alpha = log_alpha.exp()
        with torch.no_grad():
            obs_last = torch.tensor(step_data[-1]["obs_np"], dtype=torch.float32,
                                    device="cuda").unsqueeze(0)
            mu = actor_mean(obs_last)
            std = actor_log_std.exp().clamp(min=0.01, max=1.0)
            d = torch.distributions.Normal(mu, std)
            lp = d.log_prob(d.rsample()).sum()
        alpha_loss = -alpha * (lp + target_entropy)
        alpha_loss.backward()
        alpha_opt.step()

        # ============================================================
        # Phase 4: Critic update
        # ============================================================
        for _ in range(critic_iterations):
            critic_opt.zero_grad()
            critic_loss = torch.tensor(0.0, device="cuda")
            for t in range(horizon):
                obs_t = torch.tensor(step_data[t]["obs_np"], dtype=torch.float32,
                                     device="cuda").unsqueeze(0)
                v_pred = critic(obs_t).squeeze()
                target = torch.tensor(returns[t], dtype=torch.float32, device="cuda")
                critic_loss = critic_loss + (v_pred - target) ** 2
            critic_loss = critic_loss / float(horizon)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_opt.step()

        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_target.parameters()):
                pt.data.lerp_(p.data, tau)

        # ============================================================
        # Logging
        # ============================================================
        ep_reward = sum(sd["reward"] for sd in step_data)
        if epoch % log_interval == 0:
            grad_norms = [np.linalg.norm(sd["action_grad"]) for sd in step_data]
            avg_grad = np.mean(grad_norms)
            max_grad = max(grad_norms) if grad_norms else 0
            nonzero = sum(1 for n in grad_norms if n > 1e-8)
            print(f"  epoch {epoch:4d} | reward={ep_reward:+.4f} | "
                  f"avg_grad={avg_grad:.6f} max_grad={max_grad:.6f} | "
                  f"grads>0: {nonzero}/{horizon} | "
                  f"alpha={log_alpha.exp().item():.3f} | "
                  f"critic={critic_loss.item():.4f}")

        if ep_reward > best_reward:
            best_reward = ep_reward

        if save_interval > 0 and epoch % save_interval == 0 and epoch > 0:
            torch.save({
                "actor": actor_mean.state_dict(),
                "actor_log_std": actor_log_std.data,
                "critic": critic.state_dict(),
                "epoch": epoch, "best_reward": best_reward,
            }, f"checkpoint_epoch{epoch}.pt")
            print(f"  Saved checkpoint_epoch{epoch}.pt")

    print(f"\nTraining complete. Best reward: {best_reward:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="warp_sim/config/debulk_sapo.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
