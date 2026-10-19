"""colab/train_ppo.ipynb を生成する。ノートブックはJSONで直接編集しづらいので、ここを正とする。

    python colab/make_notebook.py
"""

import json
import os

OUT = os.path.join(os.path.dirname(__file__), "train_ppo.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
# Genesis お手玉ロボット: PPO学習 (Google Colab)

2本のFranka Pandaアーム(手首にすり鉢型のお盆)が、お手玉に見立てた球2個を、1個ずつ交差させて投げ合う方策を、PPOで学習します。

**このノートブックの役割**: 学習(GPU)と、チェックポイントごとの自動評価。動画の撮影はMacで行うのが確実です（下の「学習のあと」を参照）。

**目標**: 赤と青の**両方の玉**を左右に受け渡し、落とさず、じっとせず、**60秒間続ける**こと。進み具合は、学習中に自動で出る
`[評価 ... steps] ... 玉ごとの成功 赤 ○ / 青 ○ (青の割合 ○%)` と `60秒 活動的に続いた割合 ○%` が目安です。
**青の割合が0%に近い**と、青を交換できていません(2026-10-19にそうなっていた)。動画や報酬だけだと、赤だけ往復している状態に気づけません。

**使い方**
1. メニューの「ランタイム」→「ランタイムのタイプを変更」で **GPU** を選択してください
2. 上から順にセルを実行してください。**「2.5 速度測定」の結果を見て、4.の `N_ENVS` と `total_timesteps` を決めます**
3. 再開元は、4.の `RESUME_SOURCE` で選びます(既定は、リポジトリに入っている、青の交換を学習し始めたモデル)
4. Google Driveの出力フォルダ(`genesis_otedama/runs/<日時>/`)に、`step_*/`ごとに
   `model_*.zip`・`reward_curve.png`(4枚組)・`eval.json`(自動評価)・`rollout_*.mp4`・`metrics.csv` が保存されます
5. 相談するときは、**そのフォルダをまるごとClaudeにアップロード**してください

**セッションが「不明な理由でクラッシュ」したとき**: 直後に「ランタイム」→「ランタイムログを表示」(または app.log)と、
落ちたセル・直前の「[メモリ]」の行を共有してください。`faulthandler`により、Segfaultならスタックが残ります。
(過去の原因: EGLの初期化でSegfault → 学習用環境はGL描画を使わない / TensorFlowのLLVM衝突 → 読み込みを抑止)
"""
    ),
    code(
        """
!nvidia-smi
!free -h
"""
    ),
    md("## 1. リポジトリの取得と依存関係のインストール"),
    code(
        """
import os

REPO_URL = "https://github.com/ky-ok/genesis-otedama-robot.git"
REPO_DIR = "/content/genesis-otedama-robot"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull
"""
    ),
    code(
        """
# 動画撮影(別プロセス)で使うソフトウェア描画(OSMesa)。EGLはColabのヘッドレス環境でSegfaultしたため使わない。
!apt-get -qq update && apt-get -qq install -y libosmesa6 libgl1 > /dev/null || true
"""
    ),
    code(
        """
!pip install -q -r {REPO_DIR}/requirements.txt
"""
    ),
    md(
        """
## 2. Genesisの初期化

- `faulthandler`: ネイティブクラッシュ時にPythonのスタックをランタイムログへ出力します
- `genesis_juggling` を最初に読み込みます。SB3は内部でTensorFlowを読み込み、そのLLVMがGenesisのLLVMと衝突して異常終了するため、このパッケージがTensorFlowの読み込みを抑止します
"""
    ),
    code(
        """
import faulthandler

faulthandler.enable()

import sys

sys.path.append(REPO_DIR)

import genesis_juggling  # noqa: F401  (最初に読み込む: TensorFlowの読み込みを抑止し、GenesisのLLVMとの衝突を防ぐ)

import json
import subprocess

import numpy as np
import psutil
import torch
import stable_baselines3  # noqa: F401

import genesis as gs

gs.init(backend=gs.gpu)


def show_memory(tag=""):
    ram = psutil.virtual_memory()
    msg = f"[メモリ:{tag}] RAM {ram.used / 1e9:.1f}/{ram.total / 1e9:.1f} GB"
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        msg += f" | GPU {(total - free) / 1e9:.1f}/{total / 1e9:.1f} GB"
    print(msg, flush=True)


show_memory("gs.init直後")
"""
    ),
    md(
        """
## 2.5 速度測定（学習の規模を決める）

並列数ごとの、環境のステップ速度を**別プロセスで**測ります（PPOの更新は含まないので、実際の学習はこれより遅くなります）。
参考: **Mac(M4 Pro、CPU) は、256並列で毎秒約4.9万ステップ**（PPOを含む実際の学習は毎秒約1.6万ステップ）でした。
Colabがこれより明らかに速い並列数があれば、その並列数を4.の `N_ENVS` に使います。
"""
    ),
    code(
        """
bench_env = dict(os.environ, PYTHONPATH=REPO_DIR)
print(f"{'並列数':>6} {'環境ステップ/秒':>16} {'構築[秒]':>9} {'GPUメモリ[GB]':>14}")
for n in (256, 1024, 4096):
    r = subprocess.run(
        [sys.executable, "-m", "genesis_juggling.benchmark", "--n-envs", str(n), "--backend", "gpu"],
        env=bench_env, capture_output=True, text=True, cwd=REPO_DIR, timeout=1200,
    )
    line = [l for l in r.stdout.splitlines() if l.startswith("BENCH_JSON ")]
    if not line:
        print(f"{n:>6}  失敗: {r.stderr[-300:]}")
        continue
    b = json.loads(line[0][len("BENCH_JSON "):])
    print(f"{b['n_envs']:>6} {b['env_steps_per_sec']:>16,.0f} {b['build_seconds']:>9.0f} {b.get('gpu_mem_used_gb', float('nan')):>14.1f}")
"""
    ),
    md(
        """
## 3. Google Driveのマウント

チェックポイント・評価・動画・曲線をここに保存する。Colabのランタイムが切れても消えない。
"""
    ),
    code(
        """
from google.colab import drive

drive.mount("/content/drive")

import datetime

RUN_NAME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"/content/drive/MyDrive/genesis_otedama/runs/{RUN_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Output dir:", OUTPUT_DIR)
"""
    ),
    md(
        """
## 3.5 動画撮影の事前チェック（任意）

学習中のチェックポイントごとの動画が撮れるかを、**別プロセスで**確認します。ここで失敗しても学習側のカーネルは落ちません。
失敗しても、学習・自動評価は問題なく動きます（動画はMacで撮れます）。失敗したら、4.の `record_video` を `False` にしてください。
"""
    ),
    code(
        """
check_env = dict(os.environ, PYOPENGL_PLATFORM="osmesa", PYTHONPATH=REPO_DIR)
r = subprocess.run(
    [sys.executable, "-m", "genesis_juggling.record_video", "--out", "/content/video_check.mp4", "--steps", "20", "--backend", "gpu"],
    env=check_env, capture_output=True, text=True, cwd=REPO_DIR, timeout=900,
)
print("returncode:", r.returncode)
print(r.stdout[-1500:])
print(r.stderr[-1500:])
print("動画あり" if os.path.exists("/content/video_check.mp4") else "動画なし(撮影失敗)")
"""
    ),
    md(
        """
## 4. ハイパーパラメータ

調整したいパラメータはこのセルにまとめている。ここを変えて再実行すれば別の設定で学習し直せる。

- **既定は「Macのモデルの続きから、玉2個」** です。`START_FRESH = True` にすると、最初から(玉1個から)学習し直します。
- `N_ENVS` は2.5の速度測定を見て決めます。`batch_size` は `N_ENVS × n_steps` の約1/4が目安で、自動で計算します。
- 続きから学習するときは、探索ノイズが枯れていることが多いので、`reset_std`(ノイズの大きさを戻す)と`ent_coef`(探索の促し)を入れ、学習率は低め(1e-4)にしてあります。
  (Macでの比較: 学習率1e-4・ノイズ0.10の設定が、3e-4・0.15より、成功回数・持続とも良かった)
- `focus_obs`: 方策に見せる玉の順番を「飛んでいる玉・放置されている玉が先頭」に並べ替えます。赤と青は物理的に同じなので、覚えた「先頭の玉の交換」を青にも使わせるための工夫です(これで青の割合が1%→約40%に上がりました)。**学習・評価・動画で同じ設定にします**(自動で揃います)。
- 玉ごとに「最後の成功から10秒」成功がないとエピソードが打ち切られ(減点)、青を置きっぱなしにすると損をします。
- `settled_prob`: 静止状態(お盆を様々な位置へ動かして玉を載せた状態)から開始する割合。じっとして動かなくなった状態から、再び投げ始める練習になります。
"""
    ),
    code(
        """
START_FRESH = False   # True: 最初から学習し直す(玉1個から)。False: Macで学習したモデルの続きから(玉2個)
N_ENVS = 1024         # 並列シミュレーション数。2.5の速度測定を見て決める。GPUメモリが足りなければ減らす

HYPERPARAMS = {
    # 環境
    "n_envs": N_ENVS,
    "dt": 0.02,
    "n_balls": 1 if START_FRESH else 2,    # 最初から学習するときは、玉1個→うまくいったら2個の段階学習
    "settled_prob": 0.5,    # 静止状態から開始する割合(0で無効)
    "focus_obs": True,      # 玉の順番を並べ替えて方策に見せる(青の交換のために必要。学習・評価・動画で同じにする)
    "ball_mix": None,       # エピソードごとに使う玉を混ぜる割合 (赤だけ, 青だけ, 両方)。試したが効果なし(既存のスキルが崩れた)ので使わない
    "ball_rolling_friction": None,  # 玉の転がり摩擦。引き継ぎ学習では変えない(変えると既存モデルが崩れた)。新規学習で試すなら 0.01
    # PPO
    "learning_rate": 3e-4 if START_FRESH else 1e-4,  # 続きから学習するときは低め(穏やかに). 3e-4だと引き継いだ方策が崩れやすかった
    "n_steps": 64,          # 1回の更新までに各envが踏むステップ数
    "batch_size": N_ENVS * 64 // 4,
    "n_epochs": 10,
    "gamma": 0.995,         # 約4秒先まで見る(長く続けることを学習させるため)
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0 if START_FRESH else 0.001,
    "log_std_init": -1.0,   # 新規学習のときの、行動ノイズの初期値
    "reset_std": None if START_FRESH else 0.10,  # 続きから学習するとき、行動ノイズの大きさをこの値に戻す
    "net_arch": [256, 256],
    # カリキュラム: 着地成功に必要な「お盆同士の最低間隔」を gap_start から gap_end まで段階的に広げる
    # (最初から0.42を要求すると成功が出ず学習が進まなかった)。続きから学習するときは gap_start も 0.42 にする
    "gap_start": 0.20 if START_FRESH else 0.42,
    "gap_end": 0.42,
    "gap_fraction": 0.6,    # 学習全体のこの割合の時点で gap_end にする
    # 学習の長さ。速度測定を見て決める(目安: 所要時間 ≈ total_timesteps ÷ 実際の毎秒ステップ数)
    "total_timesteps": 30_000_000,
    # チェックポイントごとに、保存・自動評価・動画を行う間隔(タイムステップ数)
    "save_freq": 5_000_000,
    "rollout_steps": 300,
    "record_video": True,   # 別プロセスで撮影。失敗しても学習は続行され、video_error.log が残る
    "evaluate": True,       # 別プロセスで自動評価(描画なし)。eval.json に保存される
}
HYPERPARAMS
"""
    ),
    md(
        """
## 5. 学習

`RESUME_FROM` は、`START_FRESH=False` のとき、リポジトリに入っているMacのモデルになります(別のチェックポイントから再開したいときは、ここを書き換えます)。
"""
    ),
    code(
        """
from genesis_juggling.checkpoint import apply_resume_settings, load_ppo_for_env
from genesis_juggling.vec_env import JugglingVecEnv
from genesis_juggling.training_utils import CheckpointVideoCallback, GapCurriculumCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

import glob

# 再開元の選択:
#   "repo"         : リポジトリに入っている、青の交換を学習し始めたモデル(既定)
#   "drive_latest" : Google Drive上の、直近の学習(runs/*)のモデル。前回の続きから学習したいとき
#   "path"         : RESUME_PATH に書いたチェックポイント
RESUME_SOURCE = "repo"
RESUME_PATH = ""
REPO_CKPT = f"{REPO_DIR}/checkpoints/stage2_focus_adapted.zip"

if START_FRESH:
    RESUME_FROM = None
elif RESUME_SOURCE == "drive_latest":
    runs = sorted(glob.glob("/content/drive/MyDrive/genesis_otedama/runs/*/model_final.zip")
                  or glob.glob("/content/drive/MyDrive/genesis_otedama/runs/*/step_*/model_*.zip"))
    if not runs:
        raise FileNotFoundError("Drive上に前回の学習(runs/*)が見つかりません。RESUME_SOURCE を変えてください。")
    RESUME_FROM = runs[-1]
elif RESUME_SOURCE == "path":
    RESUME_FROM = RESUME_PATH
else:
    RESUME_FROM = REPO_CKPT
if RESUME_FROM is not None and not os.path.exists(RESUME_FROM):
    raise FileNotFoundError(f"{RESUME_FROM} がありません。`git pull` を実行したか、パスを確認してください。")
print("続きから:" if RESUME_FROM else "最初から学習します", RESUME_FROM or "")

env_kwargs = {"settled_prob": HYPERPARAMS["settled_prob"], "focus_obs": HYPERPARAMS["focus_obs"]}
if HYPERPARAMS["ball_mix"] is not None:
    env_kwargs["ball_mix"] = tuple(HYPERPARAMS["ball_mix"])
if HYPERPARAMS["ball_rolling_friction"] is not None:
    env_kwargs["ball_rolling_friction"] = HYPERPARAMS["ball_rolling_friction"]

show_memory("環境構築前")
vec_env = JugglingVecEnv(n_envs=HYPERPARAMS["n_envs"], dt=HYPERPARAMS["dt"], n_balls=HYPERPARAMS["n_balls"], **env_kwargs)
vec_env = VecMonitor(vec_env)
show_memory("環境構築後")

if RESUME_FROM is not None:
    model = load_ppo_for_env(RESUME_FROM, vec_env)  # 観測の次元が増えていれば、ゼロで埋めて引き継ぐ
    apply_resume_settings(  # 保存時の設定を、今の設定で上書きする(割引率・探索・学習率)
        model,
        gamma=HYPERPARAMS["gamma"],
        ent_coef=HYPERPARAMS["ent_coef"],
        reset_std=HYPERPARAMS["reset_std"],
        learning_rate=HYPERPARAMS["learning_rate"],
    )
else:
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=HYPERPARAMS["learning_rate"],
        n_steps=HYPERPARAMS["n_steps"],
        batch_size=HYPERPARAMS["batch_size"],
        n_epochs=HYPERPARAMS["n_epochs"],
        gamma=HYPERPARAMS["gamma"],
        gae_lambda=HYPERPARAMS["gae_lambda"],
        clip_range=HYPERPARAMS["clip_range"],
        ent_coef=HYPERPARAMS["ent_coef"],
        policy_kwargs=dict(log_std_init=HYPERPARAMS["log_std_init"], net_arch=HYPERPARAMS["net_arch"]),
        verbose=1,
    )
show_memory("PPO構築後")

callback = CheckpointVideoCallback(
    save_freq=HYPERPARAMS["save_freq"],
    output_dir=OUTPUT_DIR,
    hyperparams=HYPERPARAMS,
    rollout_steps=HYPERPARAMS["rollout_steps"],
    record_video=HYPERPARAMS["record_video"],
    evaluate=HYPERPARAMS["evaluate"],
)
curriculum = GapCurriculumCallback(
    HYPERPARAMS["total_timesteps"],
    start=HYPERPARAMS["gap_start"],
    end=HYPERPARAMS["gap_end"],
    fraction=HYPERPARAMS["gap_fraction"],
)

print("学習開始", flush=True)
model.learn(total_timesteps=HYPERPARAMS["total_timesteps"], callback=[callback, curriculum])
show_memory("学習終了")
"""
    ),
    md(
        """
## 6. 最終モデルの保存と、最終評価

学習が正常に終わった場合の最終モデルも明示的に保存し、評価します（64環境・60秒、描画なし）。
`活動的に続いた割合` が、「60秒続けられる確率」です。
"""
    ),
    code(
        """
model.save(f"{OUTPUT_DIR}/model_final.zip")
print("Saved final model to", f"{OUTPUT_DIR}/model_final.zip")

r = subprocess.run(
    [sys.executable, "-m", "genesis_juggling.evaluation", "--model", f"{OUTPUT_DIR}/model_final.zip",
     "--n-balls", str(HYPERPARAMS["n_balls"]), "--n-envs", "128", "--steps", "3000", "--backend", "gpu"]
    + (["--focus-obs"] if HYPERPARAMS["focus_obs"] else []),
    env=dict(os.environ, PYTHONPATH=REPO_DIR), capture_output=True, text=True, cwd=REPO_DIR, timeout=3600,
)
print(r.stdout[-2000:] if r.returncode == 0 else r.stderr[-2000:])
"""
    ),
    md(
        """
## 7. 学習のあと（Macで動画を撮る）

1. Google Driveの `genesis_otedama/runs/<日時>/model_final.zip` (または評価が一番良かった `step_*/model_*.zip`) をMacの `checkpoints/` に置く
2. Macで、60秒・等速・連続の動画を撮る（描画を伴う撮影はMacが確実です）:

(`--focus-obs` は、学習で `focus_obs=True` にしたときに付けます)

```bash
python mac_dev/make_video.py --model checkpoints/model_final.zip --n-balls 2 --focus-obs \\
    --video-seconds 60 --speed 1 --max-idle 5 --min-rate 4 --tries 200 --out videos/final.mp4
```
"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "train_ppo.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
print("wrote", OUT)
