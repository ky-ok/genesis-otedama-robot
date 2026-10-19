"""学習済み方策を、ランダムな初期状態から走らせて評価する(描画なし)。

各環境で「最初に落とす(または動かなくなって打ち切られる)まで」を追い、最大 `steps` ステップ(既定3000=60秒)で集計する。
「落とさず続いた」だけだと、じっとしているだけの試行が混ざるため、成功が一定数以上ある「活動的」な試行の割合も出す。

    python -m genesis_juggling.evaluation --model model.zip --n-balls 2 [--json]

学習中は `CheckpointVideoCallback` が、チェックポイントごとにこれを別プロセスで呼び、`eval.json` として保存する。
"""

from __future__ import annotations

import argparse
import json


def evaluate_policy(
    model, n_balls: int, n_envs: int = 64, steps: int = 3000, min_rate: float = 3.0, stochastic: bool = False, env_kwargs=None
):
    import numpy as np
    import torch

    from .checkpoint import predict
    from .env import IDLE_GRACE, JugglingEnv

    env = JugglingEnv(n_envs=n_envs, n_balls=n_balls, max_episode_steps=10**9, **(env_kwargs or {}))
    obs = env.reset()

    alive = torch.ones(n_envs, dtype=torch.bool)
    steps_alive = torch.zeros(n_envs)
    catches = torch.zeros(n_envs)
    catches_a = torch.zeros(n_envs)
    catches_b = torch.zeros(n_envs)
    stack_steps = torch.zeros(n_envs)
    idle_steps = torch.zeros(n_envs)
    for _ in range(steps):
        action = predict(model, obs.cpu().numpy(), deterministic=not stochastic)
        # 環境がGPU上にあるので、行動も同じデバイスに置く(CPUのままだとColabで 'two devices' エラーになった)
        obs, _, done, info = env.step(torch.as_tensor(action, dtype=torch.float32, device=env.device))
        alive_now = alive & ~done.cpu()
        steps_alive += alive.float()
        stack_steps += (alive & env.last_stacked.cpu()).float()
        idle_steps += (alive & (env._since_catch.cpu() > IDLE_GRACE)).float()
        catches = torch.where(alive, info["catches"].cpu(), catches)
        catches_a = torch.where(alive, info["catches_a"].cpu(), catches_a)
        catches_b = torch.where(alive, info["catches_b"].cpu(), catches_b)
        alive = alive_now
        if not alive.any():
            break

    sec = steps_alive.numpy() * env.dt
    full = (steps_alive >= steps - 1).numpy()
    rate = (catches / (steps_alive * env.dt) * 10).numpy()  # 各環境の成功数[回/10秒]
    total = steps_alive.sum().item()
    return {
        "n_envs": n_envs,
        "max_seconds": steps * env.dt,
        "survive_rate": float(full.mean()),
        "active_survive_rate": float((full & (rate >= min_rate)).mean()),  # <- 本当の指標: じっとしているだけの試行を除く
        "idle_survivors": int((full & (rate < min_rate)).sum()),
        "mean_seconds": float(sec.mean()),
        "median_seconds": float(np.median(sec)),
        "catches_per_10s": float(catches.sum().item() / total * 10 / env.dt),
        "red_catches_per_10s": float(catches_a.sum().item() / total * 10 / env.dt),
        "blue_catches_per_10s": float(catches_b.sum().item() / total * 10 / env.dt),
        # 青い玉の成功の割合。0%に近いと、青を交換できていない(赤だけ往復している)
        "blue_share": float(catches_b.sum().item() / max(catches.sum().item(), 1e-9)) if n_balls == 2 else 0.0,
        # 両方の玉が、このエピソードで3回以上成功した環境の割合
        "both_balls_rate": float(((catches_a >= 3) & (catches_b >= 3)).float().mean().item()) if n_balls == 2 else 0.0,
        "stacked_ratio": float(stack_steps.sum().item() / total) if n_balls == 2 else 0.0,
        "idle_ratio": float(idle_steps.sum().item() / total),
        "min_rate": min_rate,
        "idle_seconds": IDLE_GRACE * env.dt,
    }


def format_report(r: dict, n_balls: int) -> str:
    lines = [
        f"環境数 {r['n_envs']}, 最大 {r['max_seconds']:.0f} 秒",
        f"落とさず最後まで続いた割合: {r['survive_rate'] * 100:.0f}%",
        f"  うち活動的(成功が{r['min_rate']:.0f}回/10秒以上)に続いた割合: {r['active_survive_rate'] * 100:.0f}%  <- じっとしているだけの環境を除いた本当の指標",
        f"じっとしていた(落とさず最後まで続いたが成功が少ない)環境: {r['idle_survivors']} / {r['n_envs']}",
        f"落とすまでの時間: 平均 {r['mean_seconds']:.1f} 秒 / 中央値 {r['median_seconds']:.1f} 秒",
        f"じっとしていた時間の割合(最後の成功から{r['idle_seconds']:.0f}秒超): {r['idle_ratio'] * 100:.0f}%",
        f"着地成功: {r['catches_per_10s']:.2f} 回 / 10秒",
    ]
    if n_balls == 2:
        lines.append(
            f"  玉ごとの成功: 赤 {r['red_catches_per_10s']:.2f} / 青 {r['blue_catches_per_10s']:.2f} 回/10秒"
            f"  (青の割合 {r['blue_share'] * 100:.0f}%, 両方の玉が3回以上成功した環境 {r['both_balls_rate'] * 100:.0f}%)  <- 青が0%に近いなら、青を交換できていない"
        )
        lines.append(f"2個が同じお盆に載っていた時間の割合: {r['stacked_ratio'] * 100:.0f}%")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n-balls", type=int, default=2, choices=[1, 2])
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--min-rate", type=float, default=3.0, help="「活動的」とみなす成功数の下限[回/10秒]")
    p.add_argument("--stochastic", action="store_true", help="学習時と同じ探索ノイズありで動かす(既定は決定論的)")
    p.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--focus-obs", action="store_true", help="観測の玉の順番を並べ替える設定(学習時と同じにする)")
    p.add_argument("--ball-rolling-friction", type=float, default=None, help="玉の転がり摩擦(物理パラメータの確認用)")
    p.add_argument("--json", action="store_true", help="結果をJSON1行で出力する(学習中の自動評価用)")
    a = p.parse_args()

    import genesis as gs

    gs.init(backend=gs.cpu if a.backend == "cpu" else gs.gpu, logging_level="error")

    from stable_baselines3 import PPO

    model = PPO.load(a.model, device="cpu")
    env_kwargs = {} if a.ball_rolling_friction is None else {"ball_rolling_friction": a.ball_rolling_friction}
    if a.focus_obs:
        env_kwargs["focus_obs"] = True
    result = evaluate_policy(model, a.n_balls, a.n_envs, a.steps, a.min_rate, a.stochastic, env_kwargs)
    if a.json:
        print("EVAL_JSON " + json.dumps(result))
    else:
        print(format_report(result, a.n_balls))


if __name__ == "__main__":
    main()
