"""
学習済み方策で、課題提出用の動画(1〜3分、2倍速再生可)を撮影する。

動画の長さ = --video-seconds、再生速度 = --speed(2.0なら実時間の2倍速で再生される)。
例: --video-seconds 120 --speed 2 なら、シミュレーション上の240秒分を、120秒の動画にする。

途中で玉を落としてリセットが入らないよう、初期状態を変えながら撮影を試み、落としたものや、
じっとしているだけ(成功が少ない)のものは捨てて、最後まで落とさず活動的に続いたものを採用する(試行回数内に見つからなければ、リセットを含めて最後まで撮る)。
(探索と撮影を分けて「同じ初期状態で撮り直す」方式は、撮り直しで動きが再現されず失敗した)

    python mac_dev/make_video.py --model checkpoints/xxx.zip --n-balls 2 --video-seconds 120 --speed 2
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import genesis_juggling  # noqa: F401,E402

import genesis as gs  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stochastic", action="store_true", help="学習時と同じ探索ノイズありで動かす(既定は決定論的)。決定論的だと静止状態にはまることがある")
    p.add_argument("--n-balls", type=int, default=2, choices=[1, 2])
    p.add_argument("--video-seconds", type=float, default=120.0, help="出来上がる動画の長さ[秒]。課題の条件は60〜180秒")
    p.add_argument("--speed", type=float, default=2.0, help="再生速度。課題の条件は2倍速まで")
    p.add_argument("--montage", action="store_true", help="連続で撮れないとき用: 活動的に続いた区間(落とす直前まで)を複数つないで、指定の長さにする")
    p.add_argument("--clip-min", type=float, default=12.0, help="--montage で採用する区間の最短[シミュレーション秒]")
    p.add_argument("--tries", type=int, default=100, help="条件を満たす初期状態を探す回数")
    p.add_argument("--max-idle", type=float, default=8.0, help="採用する試行で、成功と成功の間に空いてよい最大秒数。じっとしている時間が長い試行を除く")
    p.add_argument("--min-rate", type=float, default=3.0, help="採用する試行の成功数の下限[回/10秒]。じっとしているだけの試行を除く")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "videos", "final.mp4"))
    a = p.parse_args()

    if not 60 <= a.video_seconds <= 180:
        print(f"注意: 動画の長さ {a.video_seconds:.0f}秒 は、課題の条件(60〜180秒)の範囲外です。")
    if a.speed > 2.0:
        print(f"注意: 再生速度 {a.speed}倍 は、課題の条件(2倍速まで)を超えています。")

    gs.init(backend=gs.cpu, logging_level="error")

    import torch
    from stable_baselines3 import PPO

    from genesis_juggling.env import JugglingEnv

    dt = 0.02
    steps = int(round(a.video_seconds * a.speed / dt))
    # Genesisの録画は1ステップに最大1フレーム。倍速は realtime_factor で指定し、フレーム間隔(ステップ数)を整数にする
    steps_per_frame = max(1, round(a.speed))
    fps = a.speed / (steps_per_frame * dt)
    print(f"撮影: シミュレーション {steps * dt:.0f}秒分 → 動画 約{steps * dt / a.speed:.0f}秒 ({a.speed}倍速, {fps:.0f}fps)")

    model = PPO.load(a.model, device="cpu")
    env = JugglingEnv(n_envs=1, dt=dt, n_balls=a.n_balls, camera=True, max_episode_steps=10**9, realtime_factor=a.speed)

    def act(obs):
        action, _ = model.predict(obs.cpu().numpy(), deterministic=not a.stochastic)
        return torch.as_tensor(action, dtype=torch.float32).reshape(1, -1)

    def attempt(path):
        """撮影しながら走らせ、最初に落とした時点で打ち切る。最後まで続けば steps を返す。"""
        obs = env.reset()
        env.cam.start_recording(save_to_filename=path, fps=fps)
        survived = steps
        max_gap = 0
        for t in range(steps):
            obs, _, done, _ = env.step(act(obs))
            max_gap = max(max_gap, int(env._since_catch[0]))
            if bool(done[0]):
                survived = t
                break
        env.cam.stop_recording()
        return survived, float(env._episode_catches[0]), max_gap * dt

    def fallback(path):
        """最後まで続く試行が無かったとき用: 落としてもリセットして最後まで撮る。"""
        obs = env.reset()
        env.cam.start_recording(save_to_filename=path, fps=fps)
        first_drop = None
        for t in range(steps):
            obs, _, done, _ = env.step(act(obs))
            if bool(done[0]):
                first_drop = t if first_drop is None else first_drop
                obs = env.reset()
        env.cam.stop_recording()
        return first_drop

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".try.mp4"

    if a.montage:
        import imageio.v3 as iio
        import numpy as np

        def attempt_clip(path, cap_steps):
            """撮影しながら走らせ、落とした時点で止める。(使える区間のステップ数, 成功数, 成功の最大間隔[秒])を返す。"""
            obs = env.reset()
            env.cam.start_recording(save_to_filename=path, fps=fps)
            catches_at = []
            gap_at = []
            for t in range(cap_steps):
                obs, _, done, _ = env.step(act(obs))
                catches_at.append(float(env._episode_catches[0]))
                gap_at.append(int(env._since_catch[0]))
                if bool(done[0]):
                    break
            env.cam.stop_recording()
            n = len(catches_at)
            usable = max(0, n - 75) if n < cap_steps else n  # 落とした場合は、落とす直前の1.5秒を削る
            if usable == 0:
                return 0, 0.0, 0.0
            return usable, catches_at[usable - 1], max(gap_at[:usable]) * dt

        target_frames = int(round(a.video_seconds * fps))
        torch.manual_seed(a.seed)
        t0 = time.time()
        clips = []  # (フレーム列, 説明)
        total = 0
        for i in range(a.tries):
            usable, catches, max_gap = attempt_clip(tmp, int(60 / dt))
            ok = (
                usable * dt >= a.clip_min
                and catches >= a.min_rate * usable * dt / 10
                and max_gap <= a.max_idle
            )
            print(f"  試行 {i + 1}/{a.tries}: 使える区間 {usable * dt:.1f}秒, 成功 {catches:.0f}回, 最大間隔 {max_gap:.0f}秒" + ("  <- 採用" if ok else ""), flush=True)
            if not ok:
                continue
            frames = iio.imread(tmp)[: usable // steps_per_frame]
            remaining = target_frames - total
            frames = frames[:remaining]
            clips.append((frames, usable * dt))
            total += len(frames)
            if total >= target_frames:
                break
        if os.path.exists(tmp):
            os.remove(tmp)
        if total < target_frames:
            print(f"{a.tries}回試しても必要な長さに届きませんでした({total / fps:.0f}秒 / 必要 {a.video_seconds:.0f}秒)。--tries を増やしてください。")
        import imageio

        with imageio.get_writer(a.out, fps=fps, codec="libx264", macro_block_size=1) as writer:
            for frames, _ in clips:
                for fr in frames:
                    writer.append_data(np.asarray(fr))
        print(f"撮影 {time.time() - t0:.0f}秒。{len(clips)}本の区間をつないで保存: {os.path.abspath(a.out)}")
        print(f"できあがった動画: {total / fps:.1f}秒 ({fps:.0f}fps, {total}フレーム, 区間のシミュレーション秒: " + ", ".join(f"{c[1]:.0f}" for c in clips) + ")")
        return

    tmp = a.out + ".try.mp4"
    torch.manual_seed(a.seed)
    t0 = time.time()
    clean = False
    need_catches = a.min_rate * steps * dt / 10
    for i in range(a.tries):
        length, catches, max_gap = attempt(tmp)
        active = catches >= need_catches and max_gap <= a.max_idle
        note = "" if length < steps else ("  <- 採用" if active else f"  <- 落とさなかったが成功が少ない(じっとしている)ので不採用(じっとしている時間が長い)")
        print(f"  試行 {i + 1}/{a.tries}: {length * dt:.1f}秒 続いた, 成功 {catches:.0f}回, 成功の最大間隔 {max_gap:.0f}秒{note}", flush=True)
        if length >= steps and active:
            os.replace(tmp, a.out)
            clean = True
            break
    if os.path.exists(tmp):
        os.remove(tmp)

    def report_length():
        import imageio

        reader = imageio.v2.get_reader(a.out)
        n = reader.count_frames()
        video_fps = reader.get_meta_data()["fps"]
        print(f"できあがった動画: {n / video_fps:.1f}秒 ({video_fps:.0f}fps, {n}フレーム)")

    if clean:
        print(f"撮影 {time.time() - t0:.0f}秒。保存: {os.path.abspath(a.out)}")
        print("途中で落とさず最後まで続きました。")
        report_length()
    else:
        print(f"{a.tries}回試しても、最後まで落とさず続く試行がありませんでした。落としたらリセットして最後まで撮ります。")
        first_drop = fallback(a.out)
        print(f"撮影 {time.time() - t0:.0f}秒。保存: {os.path.abspath(a.out)}  (最初に落としたのは {first_drop * dt:.1f}秒)")
        report_length()


if __name__ == "__main__":
    main()
