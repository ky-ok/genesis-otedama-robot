"""チェックポイントを読み込み、1エピソード分のロールアウト動画を保存する。

学習プロセスとは別プロセスで実行するための入口。ヘッドレス描画(EGL等)で
ネイティブクラッシュが起きても、学習側のカーネルを巻き込まないようにする。

    python -m genesis_juggling.record_video --model model.zip --out rollout.mp4
"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()

    import genesis as gs

    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu)

    import torch
    from stable_baselines3 import PPO

    from genesis_juggling.env import ACTION_DIM, JugglingEnv

    model = PPO.load(args.model, device="cpu")
    env = JugglingEnv(n_envs=1, dt=args.dt, camera=True)

    def policy_fn(obs_t):
        action_np, _ = model.predict(obs_t.detach().cpu().numpy(), deterministic=True)
        return torch.as_tensor(action_np, dtype=torch.float32, device=env.device).reshape(1, ACTION_DIM)

    env.record_rollout(policy_fn, n_steps=args.steps, filename=args.out)


if __name__ == "__main__":
    main()
