"""SAPO training for P(F) surgical debulking.

Short-Horizon Analytic Policy Optimization:
- Actor gradient comes from differentiable sim (wp.Tape), not reward sampling
- Critic trained via TD(lambda) on collected trajectory data
- Max-entropy formulation for exploration stability

Usage:
    python warp_sim/train_sapo.py [--config warp_sim/config/debulk_sapo.yaml]
"""
import os
import sys
import argparse
import math
import yaml
import numpy as np

import warp as wp

# Defer torch import to allow running env tests without torch
torch = None


def _ensure_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch


# ============== Actor-Critic Network ==============

def build_actor_critic(obs_dim, act_dim, hidden_dim):
    """Build SAPO actor-critic with LayerNorm + SiLU (matching SAPO paper)."""
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


# ============== Training Loop ==============

def train(cfg):
    _ensure_torch()
    nn = torch.nn

    wp.init()
    device = "cuda:0"

    # Build env
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from warp_sim.envs.debulk_env import DebulkEnv
    env = DebulkEnv(cfg, device=device)

    # Training config
    tcfg = cfg.get("training", {})
    num_epochs = tcfg.get("num_epochs", 1000)
    horizon = cfg.get("horizon_len", 32)
    gamma = tcfg.get("gamma", 0.99)
    lam = tcfg.get("lam", 0.95)
    max_grad_norm = tcfg.get("max_grad_norm", 0.5)
    actor_lr = tcfg.get("actor_lr", 2e-3)
    critic_lr = tcfg.get("critic_lr", 5e-4)
    actor_beta1 = tcfg.get("actor_beta1", 0.7)
    hidden_dim = tcfg.get("hidden_dim", 64)
    critic_iterations = tcfg.get("critic_iterations", 16)
    log_interval = tcfg.get("log_interval", 10)
    save_interval = tcfg.get("save_interval", 100)
    init_alpha = tcfg.get("init_alpha", 1.0)
    target_entropy = tcfg.get("target_entropy", -1.5)

    # Build networks (obs_dim = N_SAMPLE*4 + 3, act_dim = 3)
    obs_dim = env.num_obs  # 250*4 + 3 = 1003
    actor_mean, actor_log_std, critic = build_actor_critic(
        obs_dim, env.num_act, hidden_dim)
    actor_mean.cuda()
    critic.cuda()
    actor_log_std = torch.nn.Parameter(actor_log_std.detach().cuda())

    actor_params = list(actor_mean.parameters()) + [actor_log_std]
    actor_opt = torch.optim.AdamW(actor_params, lr=actor_lr,
                                   betas=(actor_beta1, 0.95))
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=critic_lr,
                                    betas=(actor_beta1, 0.95))

    # Entropy temperature (learnable)
    log_alpha = torch.tensor(math.log(init_alpha), requires_grad=True, device="cuda")
    alpha_opt = torch.optim.Adam([log_alpha], lr=5e-3)

    print(f"\nSAPO Training: {num_epochs} epochs, horizon={horizon}")
    print(f"  Actor LR={actor_lr}, Critic LR={critic_lr}, beta1={actor_beta1}")
    print(f"  gamma={gamma}, lambda={lam}, grad_clip={max_grad_norm}")

    best_reward = -float("inf")

    for epoch in range(num_epochs):
        # ---- Actor update (differentiable rollout) ----
        actor_mean.train()

        # Collect trajectory data for critic
        obs_list = []
        reward_list = []
        value_list = []
        action_grads = []

        # Reset env (cuts gradient)
        obs_dict = env.initialize_trajectory()

        # Record differentiable rollout
        tape = wp.Tape()
        actions_wp_list = []

        # Accumulate total reward inside tape for backprop
        total_reward_buf = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        @wp.kernel
        def accumulate_reward(src: wp.array(dtype=wp.float32),
                              dst: wp.array(dtype=wp.float32),
                              discount: float):
            tid = wp.tid()
            if tid > 0:
                return
            wp.atomic_add(dst, 0, src[0] * discount)

        with tape:
            for t in range(horizon):
                # obs dict -> flat numpy -> torch
                wp.synchronize()
                obs_np = env.get_obs_flat_np(obs_dict)
                obs_list.append(obs_np.copy())
                obs_torch = torch.tensor(obs_np, dtype=torch.float32, device="cuda").unsqueeze(0)

                # Actor forward
                with torch.no_grad():
                    act_mean_out = actor_mean(obs_torch)
                    std = actor_log_std.exp().clamp(min=0.01, max=1.0)
                    dist = torch.distributions.Normal(act_mean_out, std)
                    action_torch = dist.sample().squeeze(0)  # (3,)
                    action_torch = torch.tanh(action_torch)  # bound to [-1, 1]

                # torch -> warp action
                action_np = action_torch.cpu().numpy().astype(np.float32)
                action_wp = wp.array(
                    [wp.vec3(action_np[0], action_np[1], action_np[2])],
                    dtype=wp.vec3, device=device, requires_grad=True,
                )
                actions_wp_list.append(action_wp)

                # Env step (differentiable physics)
                obs_dict, rew_wp, done_wp, _ = env.step(action_wp)

                # Accumulate discounted reward into total (inside tape!)
                discount_t = gamma ** t
                wp.launch(accumulate_reward, dim=1,
                          inputs=[rew_wp, total_reward_buf, discount_t],
                          device=device)

                # Track data
                wp.synchronize()
                reward_list.append(float(rew_wp.numpy()[0]))

                # Critic value estimate (detached)
                with torch.no_grad():
                    v = critic(obs_torch).item()
                    value_list.append(v)

        # Backward through ALL steps via accumulated reward
        try:
            tape.backward(loss=total_reward_buf)

            # Collect action gradients from Warp
            for t, a_wp in enumerate(actions_wp_list):
                if a_wp.grad is not None:
                    g = a_wp.grad.numpy().copy()
                    action_grads.append(g)
                else:
                    action_grads.append(np.zeros((1, 3), dtype=np.float32))
        except Exception as e:
            print(f"  [epoch {epoch}] tape.backward error: {e}")
            action_grads = [np.zeros((1, 3), dtype=np.float32)] * horizon

        env.clear_grad()

        # ---- Policy gradient via analytic gradients ----
        # Rerun actor forward and use Warp's action.grad as the policy gradient signal
        actor_opt.zero_grad()
        policy_loss = torch.tensor(0.0, device="cuda")

        for t in range(horizon):
            obs_torch = torch.tensor(obs_list[t], dtype=torch.float32, device="cuda").unsqueeze(0)
            act_mean_out = actor_mean(obs_torch)
            std = actor_log_std.exp().clamp(min=0.01, max=1.0)
            dist = torch.distributions.Normal(act_mean_out, std)
            action_sampled = dist.rsample()
            action_bounded = torch.tanh(action_sampled)

            # Use Warp's analytic gradient as the "reward gradient" w.r.t. action
            grad_from_sim = torch.tensor(
                action_grads[t].reshape(3), dtype=torch.float32, device="cuda")

            # gamma-discounted contribution
            discount = gamma ** t
            # Policy gradient: action * grad (stop gradient on grad_from_sim)
            policy_loss = policy_loss - discount * torch.sum(action_bounded.squeeze() * grad_from_sim.detach())

            # Entropy bonus
            alpha = log_alpha.exp()
            entropy = dist.entropy().sum()
            policy_loss = policy_loss - alpha.detach() * entropy * 0.01

        policy_loss = policy_loss / float(horizon)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_params, max_grad_norm)
        actor_opt.step()

        # ---- Alpha (entropy temperature) update ----
        alpha_opt.zero_grad()
        alpha = log_alpha.exp()
        alpha_loss = -alpha * (torch.tensor(target_entropy, device="cuda") -
                               torch.tensor(float(sum(reward_list)) / horizon, device="cuda")).detach()
        alpha_loss.backward()
        alpha_opt.step()

        # ---- Critic update (TD(lambda) targets) ----
        rewards_np = np.array(reward_list)
        values_np = np.array(value_list + [value_list[-1]])  # bootstrap last
        returns = _compute_gae_returns(rewards_np, values_np, gamma, lam)

        for _ in range(critic_iterations):
            critic_opt.zero_grad()
            critic_loss = torch.tensor(0.0, device="cuda")

            for t in range(horizon):
                obs_torch = torch.tensor(obs_list[t], dtype=torch.float32, device="cuda").unsqueeze(0)
                v_pred = critic(obs_torch).squeeze()
                target = torch.tensor(returns[t], dtype=torch.float32, device="cuda")
                critic_loss = critic_loss + (v_pred - target) ** 2

            critic_loss = critic_loss / float(horizon)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_opt.step()

        # ---- Logging ----
        ep_reward = sum(reward_list)
        if epoch % log_interval == 0:
            grad_norms = [np.linalg.norm(g) for g in action_grads]
            avg_grad = np.mean(grad_norms) if grad_norms else 0
            print(f"  epoch {epoch:4d} | reward={ep_reward:+8.3f} | "
                  f"avg_grad={avg_grad:.4f} | alpha={log_alpha.exp().item():.3f} | "
                  f"critic_loss={critic_loss.item():.4f}")

        if ep_reward > best_reward:
            best_reward = ep_reward

        if save_interval > 0 and epoch % save_interval == 0 and epoch > 0:
            save_path = f"checkpoint_epoch{epoch}.pt"
            torch.save({
                "actor": actor_mean.state_dict(),
                "actor_log_std": actor_log_std.data,
                "critic": critic.state_dict(),
                "log_alpha": log_alpha.data,
                "epoch": epoch,
            }, save_path)
            print(f"  Saved checkpoint: {save_path}")

    print(f"\nTraining complete. Best reward: {best_reward:.3f}")


def _compute_gae_returns(rewards, values, gamma, lam):
    """Compute GAE (Generalized Advantage Estimation) returns."""
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        returns[t] = gae + values[t]
    return returns


# ============== Entry Point ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="warp_sim/config/debulk_sapo.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
