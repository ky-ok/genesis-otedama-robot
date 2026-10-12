# mac_dev

ローカルMac（GPUなし）上で、Genesisのシーン（2アーム + お手玉2個）の見た目や物理挙動を確認するためのスクリプトを置く場所です。学習は行わず、固定/手動の動作確認のみを行います。

## セットアップ

プロジェクトルートで以下を実行し、`uv`で仮想環境を作成します（システム環境には影響しません）。

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## scene_check.py

Franka Panda（URDF、Genesis同梱）を2台向かい合わせに配置し、お手玉に見立てた球体2個を間に置いて、腕を単純な周期運動で動かすスクリプトです。学習は行わず、ロボットモデルの読み込み・物理演算・カメラ録画が正しく動くことを確認するためのものです。

```bash
source ../.venv/bin/activate  # プロジェクトルートで有効化していれば不要
python scene_check.py
```

実行すると `videos/scene_check.mp4` が生成されます。動作確認用のため、まだ実際にボールをキャッチする制御にはなっていません（トス&キャッチの獲得は次のステップでColab上のRLで学習します）。

## train_smoke_test.py

`genesis_juggling`（環境・VecEnv・学習コールバック）が最後まで壊れず動くかを、Mac上のCPUで極小規模(2並列・64ステップ)に確認するスモークテストです。本番の学習ではありません。

```bash
python train_smoke_test.py
```

`videos/smoke_test_run/` にチェックポイント・報酬曲線・ロールアウト動画が出力されれば、Colabに移植する前提のコードが壊れていないことが分かります。

## train_local.py

Mac(CPU)上でPPO学習を実行します。M4 Proで256並列・毎秒約1.6万ステップ出るため、数百万〜1千万ステップなら数分〜十数分で回ります。Colabと同じ環境・コールバックを使うので、出力フォルダ(`runs/<日時>/`)の中身(チェックポイント・学習曲線・動画・ハイパーパラメータ)もColabと同じです。

```bash
python train_local.py --n-balls 1 --steps 5000000
python train_local.py --n-balls 2 --steps 5000000 --resume ../runs/<前の実行>/model_final.zip   # 段階2: 玉2個へ引き継ぎ
```

報酬を変えたら、学習曲線だけでなく**動画で挙動を確認**してください(報酬の抜け道を使った「不正な」高スコアが出ることがあります)。

## evaluate.py

学習済み方策を、ランダムな初期状態から決定論的に走らせて評価します(落とさず続いた割合、落とすまでの時間、成功数、2個が同じお盆に載っている割合、じっとしていた時間の割合)。「落とさず続いた」だけだと、じっとしているだけの試行が混ざるため、成功が一定数以上ある「活動的」な試行の割合も出します。

```bash
python evaluate.py --model ../checkpoints/stage2_funnel_idle_130M.zip --n-balls 2
```

## make_video.py

課題提出用の動画(1〜3分、2倍速まで)を撮影します。`--montage`を付けると、活動的に続いた区間を複数つないで指定の長さにします(連続で長く続く試行が見つからないとき用。つないだ動画であることを提出時に明記してください)。

```bash
python make_video.py --model ../checkpoints/stage2_funnel_idle_130M.zip --n-balls 2 \
    --video-seconds 60 --speed 2 --montage --max-idle 4 --min-rate 4 --tries 400 --out ../videos/final.mp4
```
