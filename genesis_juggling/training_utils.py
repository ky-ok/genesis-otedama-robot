"""学習中に「動画 + 報酬推移グラフ + ハイパーパラメータ」を1セットにして
定期的に書き出すためのSB3コールバック。

Colab側でこのコールバックを使うと、ユーザーが出力フォルダをそのまま
Claudeにアップロードするだけで、パラメータ調整の相談がしやすくなる。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

import genesis as gs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CheckpointVideoCallback(BaseCallback):
    """`save_freq`ステップごとに、以下をステップ別フォルダへまとめて保存する。

    - `model_<step>.zip`   : SB3チェックポイント
    - `reward_curve.png`   : これまでの平均報酬の推移
    - `rollout_<step>.mp4` : 現在の方策による1エピソードのロールアウト動画
    - `video_error.log`    : 動画撮影に失敗した場合のみ。エラー内容

    動画は別プロセス(`genesis_juggling.record_video`)で撮る。ヘッドレス描画で
    ネイティブクラッシュが起きても学習プロセスは巻き込まれず、学習は続行される。
    `hyperparams.json` は output_dir 直下に1つ保存する。
    """

    def __init__(
        self,
        save_freq: int,
        output_dir: str,
        hyperparams: dict,
        rollout_steps: int = 300,
        record_video: bool = True,
        video_timeout: int = 600,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.output_dir = output_dir
        self.hyperparams = hyperparams
        self.rollout_steps = rollout_steps
        self.record_video = record_video
        self.video_timeout = video_timeout

        self._reward_history: list[tuple[int, float]] = []
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

        model_path = os.path.join(run_dir, f"model_{step}.zip")
        self.model.save(model_path)
        self._save_reward_curve(os.path.join(run_dir, "reward_curve.png"))
        if self.record_video:
            self._save_rollout_video(model_path, run_dir, os.path.join(run_dir, f"rollout_{step}.mp4"))

        if self.verbose:
            print(f"[CheckpointVideoCallback] saved checkpoint/curve/video to {run_dir}", flush=True)

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

    def _save_rollout_video(self, model_path: str, run_dir: str, filename: str) -> None:
        cmd = [
            sys.executable,
            "-m",
            "genesis_juggling.record_video",
            "--model",
            model_path,
            "--out",
            filename,
            "--steps",
            str(self.rollout_steps),
            "--backend",
            "cpu" if gs.backend == gs.cpu else "gpu",
            "--dt",
            str(self.hyperparams.get("dt", 0.02)),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

        try:
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=self.video_timeout, cwd=REPO_ROOT
            )
            ok = proc.returncode == 0 and os.path.exists(filename)
            detail = f"returncode={proc.returncode}\n--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
        except subprocess.TimeoutExpired:
            ok = False
            detail = f"timed out after {self.video_timeout}s"

        if not ok:
            with open(os.path.join(run_dir, "video_error.log"), "w") as f:
                f.write(detail)
            print(
                f"[CheckpointVideoCallback] WARNING: video recording failed (training continues). "
                f"See {os.path.join(run_dir, 'video_error.log')}",
                flush=True,
            )
