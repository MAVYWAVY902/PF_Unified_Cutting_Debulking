# Arm A Baseline — PPO_11 peak@step_150000

Training run: tb_logs/PPO_11 (2026-04-19)
Source checkpoint: checkpoints/ppo_debulk_150000_steps.zip

## Why this checkpoint (not final)
PPO_11 trained to 500k steps but reward COLLAPSED after step 210k
(+7871 → -7014) due to value function never learning
(explained_variance=0 throughout, value_loss=1e5-1e6).

The 150k checkpoint is at reward peak, with verified good behavior:
- Tumor coverage: 35.4%
- Nerve damage: 0.0%
- Monotonic coverage growth (not reward-hack)
- Knife path length 0.96 (mesh diameter scale)

## Known limitations
- value function never learned (explained_variance=0)
- Future trainings MUST add VecNormalize to prevent this
- Coverage 35% is baseline-acceptable but not compelling; may need
  longer episode or better reward shaping for stronger baseline

## Arm A config
1516D obs, fixed nerve at y<0.25, 7-term coverage+surface reward,
episode_length=256, workspace_margin=-0.02, no gradient in training loop.

## How to replay
python warp_sim/rollout_interactive.py --replay baselines/armA_PPO_11_peak150k/rollout_data/trajectory.npz
