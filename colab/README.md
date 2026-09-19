# colab

Google Colab（GPU）上でRL学習を実行するためのノートブックを置く場所です。

## train_ppo.ipynb

`genesis_juggling`環境（2アームでお手玉2個を交換する課題）をStable-Baselines3のPPOで学習するノートブック。

- Colabで開き、ランタイムをGPUに設定して上から実行する
- リポジトリを`git clone`し、`requirements.txt`から依存関係をインストールする
- Google Driveをマウントし、`runs/<timestamp>/step_*/`以下に一定間隔で以下をまとめて保存する:
  - `model_*.zip`: チェックポイント
  - `reward_curve.png`: 報酬推移グラフ
  - `rollout_*.mp4`: 現在の方策によるロールアウト動画
  - `hyperparams.json`: そのときのハイパーパラメータ
- ハイパーパラメータ調整の相談をする際は、動画だけでなくこのフォルダ一式をアップロードすると的確な助言を得やすい（動画だけでは学習が収束途中なのか報酬設計の問題なのか判断しにくいため）

**注意**: このノートブックはローカル(CPU、小規模)での動作確認のみ済んでおり、Colab GPU上での実行はまだ検証していない。エラーが出た場合はエラーメッセージを共有して調整する。
