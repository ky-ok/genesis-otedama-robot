"""
Mac(CPU)上でPPO学習を実行するスクリプト。

M4 Proで256並列・毎秒約1.6万ステップ出るため、数百万〜1千万ステップなら数分〜十数分で回る。
Colab(GPU)と同じ環境・同じコールバックを使うので、出力フォルダの中身(チェックポイント・学習曲線・
動画・ハイパーパラメータ)もColabと同じ。

    python mac_dev/train_local.py --n-balls 1 --steps 10000000
    python mac_dev/train_local.py --n-balls 2 --steps 5000000 --gap-start 0.42 --resume runs/<前の実行>/model_final.zip
"""

import argparse
import datetime
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import genesis_juggling  # noqa: F401,E402  (最初にimportする: TensorFlow抑止)

import genesis as gs  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--n-balls", type=int, default=1, choices=[1, 2])
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--save-freq", type=int, default=1_000_000)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--log-std-init", type=float, default=-1.0)
    p.add_argument("--gamma", type=float, default=0.995, help="割引率。0.995だと約200ステップ(4秒)先まで見る")
    p.add_argument("--gap-start", type=float, default=0.20, help="着地成功に必要なお盆の最低間隔の初期値[m](カリキュラム)")
    p.add_argument("--gap-end", type=float, default=0.42, help="同・最終値[m]")
    p.add_argument("--gap-fraction", type=float, default=0.6, help="学習全体のこの割合の時点で最終値にする")
    p.add_argument("--reset-std", type=float, default=None, help="続きから学習するとき、行動ノイズの標準偏差をこの値に戻す(探索が枯れたとき用。例 0.15)")
    p.add_argument("--settled-prob", type=float, default=0.5, help="静止状態(お盆を動かして玉を載せた状態)から開始する割合")
    p.add_argument("--ball-rolling-friction", type=float, default=None, help="玉の転がり摩擦(既存モデルの引き継ぎでは変えない方がよい)")
    p.add_argument("--resume", default=None, help="続きから学習するチェックポイント(.zip)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-eval", action="store_true", help="チェックポイントごとの自動評価を行わない(短い動作確認用)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    out = a.out or os.path.join(
        os.path.dirname(__file__), "..", "runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out, exist_ok=True)

    gs.init(backend=gs.cpu)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor

    from genesis_juggling.checkpoint import apply_resume_settings, load_ppo_for_env
    from genesis_juggling.training_utils import CheckpointVideoCallback, GapCurriculumCallback
    from genesis_juggling.vec_env import JugglingVecEnv

    hyper = {
        "n_envs": a.n_envs,
        "dt": 0.02,
        "n_balls": a.n_balls,
        "learning_rate": a.learning_rate,
        "n_steps": a.n_steps,
        "batch_size": a.n_envs * a.n_steps // 4,
        "n_epochs": 10,
        "gamma": a.gamma,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": a.ent_coef,
        "log_std_init": a.log_std_init,
        "reset_std": a.reset_std,
        "settled_prob": a.settled_prob,
        "ball_rolling_friction": a.ball_rolling_friction,
        "net_arch": [256, 256],
        "gap_start": a.gap_start,
        "gap_end": a.gap_end,
        "gap_fraction": a.gap_fraction,
        "total_timesteps": a.steps,
        "save_freq": a.save_freq,
        "resume": a.resume,
    }
    print(json.dumps(hyper, indent=2, ensure_ascii=False))

    env_kwargs = {"settled_prob": a.settled_prob}
    if a.ball_rolling_friction is not None:
        env_kwargs["ball_rolling_friction"] = a.ball_rolling_friction
    vec_env = VecMonitor(JugglingVecEnv(n_envs=a.n_envs, dt=hyper["dt"], n_balls=a.n_balls, **env_kwargs))
    if a.resume:
        model = load_ppo_for_env(a.resume, vec_env, device="cpu")  # 観測の次元が増えていれば、ゼロで埋めて引き継ぐ
        apply_resume_settings(
            model, gamma=a.gamma, ent_coef=a.ent_coef, reset_std=a.reset_std, learning_rate=a.learning_rate
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=hyper["learning_rate"],
            n_steps=hyper["n_steps"],
            batch_size=hyper["batch_size"],
            n_epochs=hyper["n_epochs"],
            gamma=hyper["gamma"],
            gae_lambda=hyper["gae_lambda"],
            clip_range=hyper["clip_range"],
            ent_coef=hyper["ent_coef"],
            policy_kwargs=dict(log_std_init=hyper["log_std_init"], net_arch=hyper["net_arch"]),
            verbose=1,
            device="cpu",
        )

    callback = CheckpointVideoCallback(
        save_freq=a.save_freq,
        output_dir=out,
        hyperparams=hyper,
        rollout_steps=300,
        record_video=not a.no_video,
        evaluate=not a.no_eval,
    )
    curriculum = GapCurriculumCallback(a.steps, start=a.gap_start, end=a.gap_end, fraction=a.gap_fraction)
    model.learn(total_timesteps=a.steps, callback=[callback, curriculum])
    model.save(os.path.join(out, "model_final.zip"))
    print("Saved to", os.path.abspath(out))


if __name__ == "__main__":
    main()
