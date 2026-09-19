"""JugglingEnv(torch, batched)を Stable-Baselines3 の VecEnv インターフェースに
橋渡しするラッパー。学習アルゴリズム自体はSB3のPPOをそのまま使う。"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from .env import ACTION_DIM, JugglingEnv, OBS_DIM


class JugglingVecEnv(VecEnv):
    def __init__(self, n_envs: int, dt: float = 0.02, device=None):
        self.env = JugglingEnv(n_envs=n_envs, dt=dt, device=device)

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        super().__init__(n_envs, observation_space, action_space)

        self._pending_actions = None

    def reset(self):
        obs = self.env.reset()
        return self._to_numpy(obs)

    def step_async(self, actions):
        self._pending_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.env.device)

    def step_wait(self):
        obs, reward, done, _ = self.env.step(self._pending_actions)

        infos = [{} for _ in range(self.num_envs)]
        done_idx = done.nonzero(as_tuple=True)[0]
        if len(done_idx) > 0:
            terminal_obs_np = self._to_numpy(obs)
            for i in done_idx.tolist():
                infos[i]["terminal_observation"] = terminal_obs_np[i]

            reset_obs = self.env.reset(envs_idx=done_idx)
            obs = obs.clone()
            obs[done_idx] = reset_obs[done_idx]

        return self._to_numpy(obs), self._to_numpy(reward), done.detach().cpu().numpy(), infos

    def close(self):
        pass

    @staticmethod
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    # --- SB3 VecEnv boilerplate (このプロジェクトでは未使用の機能) ---
    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)] * self._n_indices(indices)

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("JugglingVecEnv does not support per-env method calls.")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self._n_indices(indices)

    def _n_indices(self, indices) -> int:
        if indices is None:
            return self.num_envs
        if isinstance(indices, int):
            return 1
        return len(indices)
