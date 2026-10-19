"""学習中に「動画 + 学習曲線 + ハイパーパラメータ」を1セットにして
定期的に書き出すためのSB3コールバック。

Colab側でこのコールバックを使うと、ユーザーが出力フォルダをそのまま
Claudeにアップロードするだけで、パラメータ調整の相談がしやすくなる。
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import genesis as gs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CheckpointVideoCallback(BaseCallback):
    """`save_freq`ステップごとに、以下をステップ別フォルダへまとめて保存する。

    - `model_<step>.zip`   : SB3チェックポイント
    - `reward_curve.png`   : 学習曲線(1ステップあたり平均報酬 / エピソード長 / 1エピソードあたり着地成功数)
    - `metrics.csv`        : 上記曲線の数値
    - `rollout_<step>.mp4` : 現在の方策によるロールアウト動画
    - `video_error.log`    : 動画撮影に失敗した場合のみ。エラー内容

    動画は別プロセス(`genesis_juggling.record_video`)で撮る。ヘッドレス描画で
    ネイティブクラッシュが起きても学習プロセスは巻き込まれず、学習は続行される。
    `hyperparams.json` は output_dir 直下に1つ保存する。

    エピソード長はVecMonitorで包んだ環境でのみ記録される。
    """

    def __init__(
        self,
        save_freq: int,
        output_dir: str,
        hyperparams: dict,
        rollout_steps: int = 300,
        record_video: bool = True,
        video_timeout: int = 900,
        gl_platform: str | None = "auto",
        evaluate: bool = True,
        eval_envs: int = 64,
        eval_steps: int = 3000,
        eval_timeout: int = 1800,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.output_dir = output_dir
        self.hyperparams = hyperparams
        self.rollout_steps = rollout_steps
        self.record_video = record_video
        self.video_timeout = video_timeout
        # Linuxのヘッドレス環境ではEGL初期化がSegfaultしうるため、動画撮影はソフトウェア描画のOSMesaを使う
        if gl_platform == "auto":
            gl_platform = "osmesa" if sys.platform == "linux" else None
        self.gl_platform = gl_platform
        # チェックポイントごとに、別プロセスで評価(描画なし)を行い、eval.json に保存する。
        # 「60秒(eval_steps)活動的に続く割合」の推移が、学習がどこまで進んだかの一番の目安になる
        self.evaluate = evaluate
        self.eval_envs = eval_envs
        self.eval_steps = eval_steps
        self.eval_timeout = eval_timeout
        self._eval_history: list[tuple[int, float, float]] = []  # (timesteps, 活動的に続いた割合, 落とすまでの平均秒)

        # (timesteps, step_reward, ep_len, ep_catches, ep_catches_red, ep_catches_blue)
        self._history: list[tuple[int, float, float, float, float, float]] = []
        self._ep_lens: list[float] = []
        self._ep_catches: list[float] = []
        self._ep_catches_a: list[float] = []
        self._ep_catches_b: list[float] = []
        self._last_save_step = 0

    def _on_training_start(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "hyperparams.json"), "w") as f:
            json.dump(self.hyperparams, f, indent=2, ensure_ascii=False, default=str)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self._ep_lens.append(float(episode["l"]))
            if "episode_catches" in info:
                self._ep_catches.append(float(info["episode_catches"]))
                self._ep_catches_a.append(float(info.get("episode_catches_a", float("nan"))))
                self._ep_catches_b.append(float(info.get("episode_catches_b", float("nan"))))
        return True

    def _on_rollout_end(self) -> None:
        step_reward = float(self.model.rollout_buffer.rewards.mean())
        ep_len = float(np.mean(self._ep_lens)) if self._ep_lens else float("nan")
        ep_catches = float(np.mean(self._ep_catches)) if self._ep_catches else float("nan")
        ep_a = float(np.mean(self._ep_catches_a)) if self._ep_catches_a else float("nan")
        ep_b = float(np.mean(self._ep_catches_b)) if self._ep_catches_b else float("nan")
        self._ep_lens.clear()
        self._ep_catches.clear()
        self._ep_catches_a.clear()
        self._ep_catches_b.clear()
        self._history.append((self.num_timesteps, step_reward, ep_len, ep_catches, ep_a, ep_b))

        if not np.isnan(ep_catches):
            self.logger.record("juggling/ep_catches", ep_catches)
            self.logger.record("juggling/ep_catches_red", ep_a)
            self.logger.record("juggling/ep_catches_blue", ep_b)

        if self.num_timesteps - self._last_save_step >= self.save_freq:
            self._last_save_step = self.num_timesteps
            self._dump(self.num_timesteps)

    # ------------------------------------------------------------------
    def _dump(self, step: int) -> None:
        run_dir = os.path.join(self.output_dir, f"step_{step:09d}")
        os.makedirs(run_dir, exist_ok=True)

        model_path = os.path.join(run_dir, f"model_{step}.zip")
        self.model.save(model_path)
        self._save_curves(run_dir)
        if self.evaluate:
            self._run_evaluation(model_path, run_dir, step)
            self._save_curves(run_dir)  # 評価の点も含めて描き直す
        if self.record_video:
            self._save_rollout_video(model_path, run_dir, os.path.join(run_dir, f"rollout_{step}.mp4"))

        if self.verbose:
            print(f"[CheckpointVideoCallback] saved checkpoint/curves/video to {run_dir}", flush=True)

    def _run_evaluation(self, model_path: str, run_dir: str, step: int) -> None:
        cmd = [
            sys.executable, "-m", "genesis_juggling.evaluation",
            "--model", model_path,
            "--n-balls", str(self.hyperparams.get("n_balls", 2)),
            "--n-envs", str(self.eval_envs),
            "--steps", str(self.eval_steps),
            "--backend", "cpu" if gs.backend == gs.cpu else "gpu",
            "--json",
        ]
        if self.hyperparams.get("focus_obs"):
            cmd.append("--focus-obs")
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        result = None
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=self.eval_timeout, cwd=REPO_ROOT)
            for line in proc.stdout.splitlines():
                if line.startswith("EVAL_JSON "):
                    result = json.loads(line[len("EVAL_JSON "):])
            detail = f"returncode={proc.returncode}\n--- stdout ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-3000:]}"
        except subprocess.TimeoutExpired:
            detail = f"timed out after {self.eval_timeout}s"
        if result is None:
            with open(os.path.join(run_dir, "eval_error.log"), "w") as f:
                f.write(detail)
            print(f"[CheckpointVideoCallback] WARNING: evaluation failed (training continues). See {run_dir}/eval_error.log", flush=True)
            return
        with open(os.path.join(run_dir, "eval.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        self._eval_history.append((step, result["active_survive_rate"], result["mean_seconds"]))
        print(
            f"[評価 {step:,} steps] {result['max_seconds']:.0f}秒 活動的に続いた割合 {result['active_survive_rate'] * 100:.0f}% / "
            f"落とすまで平均 {result['mean_seconds']:.1f}秒 / 成功 {result['catches_per_10s']:.1f}回/10秒 / "
            f"2個が同じお盆 {result['stacked_ratio'] * 100:.0f}% / じっとしている時間 {result['idle_ratio'] * 100:.0f}%",
            flush=True,
        )

    def _save_curves(self, run_dir: str) -> None:
        if not self._history:
            return
        data = np.array(self._history, dtype=float)

        with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timesteps", "mean_step_reward", "mean_episode_length", "mean_catches_per_episode", "mean_catches_red", "mean_catches_blue"]
            )
            writer.writerows(self._history)

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        titles = ["mean reward per step", "mean episode length (max 1500)", "mean catches per episode (total / red / blue)"]
        for ax, col, title in zip(axes[:3], (1, 2, 3), titles):
            ax.plot(data[:, 0], data[:, col], color="black" if col == 3 else None, label="total")
            ax.set_xlabel("timesteps")
            ax.set_title(title)
        axes[2].plot(data[:, 0], data[:, 4], color="red", label="red ball")
        axes[2].plot(data[:, 0], data[:, 5], color="blue", label="blue ball")
        axes[2].legend()
        ax = axes[3]
        if self._eval_history:
            ev = np.array(self._eval_history, dtype=float)
            ax.plot(ev[:, 0], ev[:, 1] * 100, marker="o")
        ax.set_ylim(0, 100)
        ax.set_xlabel("timesteps")
        ax.set_title("eval: % of envs staying active for the whole window")
        fig.suptitle("Training curves")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "reward_curve.png"), dpi=110)
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
            "--n-balls",
            str(self.hyperparams.get("n_balls", 2)),
        ]
        if self.hyperparams.get("focus_obs"):
            cmd.append("--focus-obs")
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        if self.gl_platform:
            env["PYOPENGL_PLATFORM"] = self.gl_platform

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


class GapCurriculumCallback(BaseCallback):
    """着地成功に必要な「お盆同士の最低間隔」を、学習の進み具合に応じて段階的に広げる。

    最初から広い間隔(0.42m)を要求すると、成功がほとんど出ず学習が進まなかった。
    近い間隔(start)から始めて、学習全体の`fraction`の時点で最終値(end)になるよう線形に広げる。
    """

    def __init__(self, total_timesteps: int, start: float = 0.20, end: float = 0.42, fraction: float = 0.6):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.start = start
        self.end = end
        self.fraction = fraction

    def _jugglng_env(self):
        env = self.training_env
        while hasattr(env, "venv"):
            env = env.venv
        return env.env

    def _current_gap(self) -> float:
        progress = min(1.0, self.num_timesteps / (self.fraction * self.total_timesteps))
        return self.start + (self.end - self.start) * progress

    def _on_training_start(self) -> None:
        self._jugglng_env().min_tray_gap = self._current_gap()

    def _on_rollout_end(self) -> None:
        gap = self._current_gap()
        self._jugglng_env().min_tray_gap = gap
        self.logger.record("juggling/min_tray_gap", gap)

    def _on_step(self) -> bool:
        return True
