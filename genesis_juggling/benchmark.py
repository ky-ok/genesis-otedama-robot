"""環境のスループット(ステップ/秒)を測る。学習の規模(並列数・ステップ数)を決める目安にする。

環境のステップ(物理 + 観測 + 報酬)だけを、ランダムな行動で回して測る。PPOの更新時間は含まないので、
実際の学習速度はこれより遅くなる。並列数ごとに別プロセスで測ること(GPUメモリを解放するため)。

    python -m genesis_juggling.benchmark --n-envs 1024 --backend gpu
"""

from __future__ import annotations

import argparse
import json
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    a = p.parse_args()

    import torch

    import genesis as gs

    gs.init(backend=gs.cpu if a.backend == "cpu" else gs.gpu, logging_level="error")

    from .env import ACTION_DIM, JugglingEnv

    t0 = time.time()
    env = JugglingEnv(n_envs=a.n_envs, n_balls=2)
    build_seconds = time.time() - t0

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(10):  # ウォームアップ(カーネルのコンパイル等を除く)
        env.step(torch.zeros(a.n_envs, ACTION_DIM, device=env.device))
    sync()
    t0 = time.time()
    for _ in range(a.steps):
        action = (torch.rand(a.n_envs, ACTION_DIM, device=env.device) * 2 - 1) * 0.3
        _, _, done, _ = env.step(action)
        if done.any():
            env.reset(envs_idx=done.nonzero(as_tuple=True)[0])
    sync()
    elapsed = time.time() - t0
    result = {
        "n_envs": a.n_envs,
        "env_steps_per_sec": a.n_envs * a.steps / elapsed,
        "build_seconds": build_seconds,
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        result["gpu_mem_used_gb"] = (total - free) / 1e9
    print("BENCH_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
