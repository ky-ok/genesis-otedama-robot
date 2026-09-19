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
