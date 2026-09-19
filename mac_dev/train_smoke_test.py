"""
genesis_juggling の環境・VecEnv・学習コールバックが最後まで壊れず動くかを
Mac(CPU、極小規模)で確認するためのスモークテスト。

本番の学習ではない。PPOが数十ステップ回って、チェックポイント・報酬曲線・
ロールアウト動画が出力フォルダにちゃんと書き出されることだけを確認する。
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import genesis as gs

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "videos", "smoke_test_run")


def main():
    gs.init(backend=gs.cpu)

    from genesis_juggling.training_utils import CheckpointVideoCallback
    from genesis_juggling.vec_env import JugglingVecEnv
    from stable_baselines3 import PPO

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    vec_env = JugglingVecEnv(n_envs=2)
    model = PPO("MlpPolicy", vec_env, n_steps=16, batch_size=16, n_epochs=1, verbose=1)

    callback = CheckpointVideoCallback(
        save_freq=32,
        output_dir=OUTPUT_DIR,
        hyperparams={"n_steps": 16, "n_envs": 2, "note": "mac smoke test, not real training"},
        rollout_steps=20,
    )

    model.learn(total_timesteps=64, callback=callback)
    print(f"Smoke test finished. Check outputs under {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
