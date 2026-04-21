"""SB3-compatible VecEnv wrapper for batched P(F) debulking.

Inherits from stable_baselines3's VecEnv so PPO/SAC can use it directly
without DummyVecEnv wrapping.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pf_shape_matching_warp"))

from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from warp_sim.envs.vec_debulk_env import VectorizedDebulkEnv


class SB3VecDebulkEnv(VecEnv):
    """SB3 VecEnv wrapping our batched GPU solver.

    SB3's VecEnv expects:
    - reset() → obs (num_envs, obs_dim)
    - step_async(actions) + step_wait() → obs, rewards, dones, infos
    - num_envs, observation_space, action_space
    """

    def __init__(self, cfg=None, num_envs=4, device="cuda:0"):
        self.inner = VectorizedDebulkEnv(cfg, num_envs=num_envs, device=device)

        observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.inner.obs_dim,), dtype=np.float32,
        )
        action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(3,), dtype=np.float32,
        )

        super().__init__(num_envs, observation_space, action_space)

        self._actions = None

    def reset(self):
        obs, _ = self.inner.reset()
        return obs  # (num_envs, obs_dim)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rewards, terminated, truncated, infos = self.inner.step(self._actions)
        # SB3 VecEnv uses 'dones' = terminated | truncated
        dones = terminated | truncated
        # SB3 expects list of info dicts, one per env
        info_list = [{"TimeLimit.truncated": bool(truncated[i])} for i in range(self.num_envs)]
        return obs, rewards, dones, info_list

    def close(self):
        pass

    def env_method(self, method_name, *args, **kwargs):
        return [getattr(self.inner, method_name)(*args, **kwargs)]

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.inner, attr_name)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.inner, attr_name, value)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        pass
