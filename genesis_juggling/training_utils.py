"""学習中に「動画 + 報酬推移グラフ + ハイパーパラメータ」を1セットにして
定期的に書き出すためのSB3コールバック。

Colab側でこのコールバックを使うと、ユーザーが出力フォルダをそのまま
Claudeにアップロードするだけで、パラメータ調整の相談がしやすくなる。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from stable_baselines3.common.callbacks import BaseCallback

from .env import ACTION_DIM, JugglingEnv


class CheckpointVideoCallback(BaseCallback):
    """`save_freq`ステップごとに、以下をタイムスタンプ付きフォルダへまとめて保存する。

    - `model_<step>.zip`            : SB3チェックポイント
    - `reward_curve.png`            : これまでの平均報酬の推移
    - `rollout_<step>.mp4`          : 現在の方策による1エピソードのロールアウト動画
    - `hyperparams.json`            : このコールバック作成時に渡されたハイパーパラメータ
    """

    def __init__(
        self,
        save_freq: int,
        output_dir: str,
        hyperparams: dict,
        rollout_steps: int = 300,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.output_dir = output_dir
        self.hyperparams = hyperparams
        self.rollout_steps = rollout_steps

        self._reward_history: list[tuple[int, float]] = []
        self._eval_env: JugglingEnv | None = None
        self._last_save_step = 0

    def _on_training_start(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "hyperparams.json"), "w") as f:
            json.dump(self.hyperparams, f, indent=2, ensure_ascii=False, default=str)

    def _on_rollout_end(self) -> None:
        mean_reward = float(self.model.rollout_buffer.rewards.mean())
        self._reward_history.append((self.num_timesteps, mean_reward))

        if self.num_timesteps - self._last_save_step >= self.save_freq:
            self._last_save_step = self.num_timesteps
            self._dump(self.num_timesteps)

    def _on_step(self) -> bool:
        return True

    # ------------------------------------------------------------------
    def _dump(self, step: int) -> None:
        run_dir = os.path.join(self.output_dir, f"step_{step:09d}")
        os.makedirs(run_dir, exist_ok=True)

        self.model.save(os.path.join(run_dir, f"model_{step}.zip"))
        self._save_reward_curve(os.path.join(run_dir, "reward_curve.png"))
        self._save_rollout_video(os.path.join(run_dir, f"rollout_{step}.mp4"))

        if self.verbose:
            print(f"[CheckpointVideoCallback] saved checkpoint/video/curve to {run_dir}")

    def _save_reward_curve(self, filename: str) -> None:
        if not self._reward_history:
            return
        steps, rewards = zip(*self._reward_history)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(steps, rewards)
        ax.set_xlabel("timesteps")
        ax.set_ylabel("mean rollout reward")
        ax.set_title("Training reward curve")
        fig.tight_layout()
        fig.savefig(filename, dpi=120)
        plt.close(fig)

    def _save_rollout_video(self, filename: str) -> None:
        if self._eval_env is None:
            self._eval_env = JugglingEnv(n_envs=1, camera=True)

        def policy_fn(obs_t):
            obs_np = obs_t.detach().cpu().numpy()
            action_np, _ = self.model.predict(obs_np, deterministic=True)
            return torch.as_tensor(action_np, dtype=torch.float32, device=self._eval_env.device).reshape(
                1, ACTION_DIM
            )

        self._eval_env.record_rollout(policy_fn, n_steps=self.rollout_steps, filename=filename)
