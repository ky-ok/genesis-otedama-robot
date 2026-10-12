"""
学習済み方策を、ランダムな初期状態から決定論的に走らせて評価する。

各環境で「最初に玉を落とすまで」(最大 --steps ステップ、既定3000=60秒)を追い、以下を集計する:
- 落とさずに最後まで続いた環境の割合
- 落とすまでの時間(秒)の平均・中央値
- 10秒あたりの着地成功数
- 2個が同じお盆に載っていた時間の割合(「1個ずつ交差」できているかの指標)

    python mac_dev/evaluate.py --model checkpoints/stage2_2balls_30M.zip --n-balls 2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import genesis_juggling  # noqa: F401,E402

import genesis as gs  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n-balls", type=int, default=2, choices=[1, 2])
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=3000)
    a = p.parse_args()

    gs.init(backend=gs.cpu, logging_level="error")

    import numpy as np
    import torch
    from stable_baselines3 import PPO

    from genesis_juggling.env import JugglingEnv

    model = PPO.load(a.model, device="cpu")
    env = JugglingEnv(n_envs=a.n_envs, n_balls=a.n_balls, max_episode_steps=10**9)
    obs = env.reset()

    alive = torch.ones(a.n_envs, dtype=torch.bool)
    steps_alive = torch.zeros(a.n_envs)
    catches = torch.zeros(a.n_envs)
    stack_steps = torch.zeros(a.n_envs)
    for _ in range(a.steps):
        action, _ = model.predict(obs.cpu().numpy(), deterministic=True)
        obs, _, done, info = env.step(torch.as_tensor(action))
        alive_now = alive & ~done.cpu()
        steps_alive += alive.float()
        stack_steps += (alive & env.last_stacked.cpu()).float()
        catches = torch.where(alive, info["catches"].cpu(), catches)
        alive = alive_now
        if not alive.any():
            break

    sec = steps_alive.numpy() * env.dt
    full = (steps_alive >= a.steps - 1).float().mean().item()
    print(f"環境数 {a.n_envs}, 最大 {a.steps * env.dt:.0f} 秒")
    print(f"落とさず最後まで続いた割合: {full * 100:.0f}%")
    print(f"落とすまでの時間: 平均 {sec.mean():.1f} 秒 / 中央値 {np.median(sec):.1f} 秒")
    print(f"着地成功: {(catches.sum() / steps_alive.sum() * 10 / env.dt).item():.2f} 回 / 10秒")
    if a.n_balls == 2:
        print(f"2個が同じお盆に載っていた時間の割合: {(stack_steps.sum() / steps_alive.sum() * 100).item():.0f}%")


if __name__ == "__main__":
    main()
