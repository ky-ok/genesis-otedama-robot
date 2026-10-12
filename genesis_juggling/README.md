# genesis_juggling

Mac（`mac_dev/`）とColab（`colab/`）の両方から共通で使うコードを置く場所です。

- `env.py`: 向かい合った2本のFranka Pandaアーム(手首にお盆を装着)が球を投げ合うGenesisバッチ環境（`JugglingEnv`、`n_balls=1/2`）。観測・行動・報酬はすべてtorchテンソル(先頭次元がn_envs)
- `robot_asset.py`: 元のPanda URDFを実行時に書き換えて、手首に縁付きのお盆を取り付けたURDFを生成する
- `vec_env.py`: `JugglingEnv`をStable-Baselines3の`VecEnv`インターフェースに橋渡しする`JugglingVecEnv`
- `training_utils.py`: 学習中に「チェックポイント + 報酬推移グラフ + ロールアウト動画 + ハイパーパラメータ」を一式まとめて保存する`CheckpointVideoCallback`（SB3コールバック）。動画は`record_video.py`を別プロセスで呼んで撮る
- `record_video.py`: チェックポイントを読み込んで1エピソード分の動画を保存するCLI（`python -m genesis_juggling.record_video --model ... --out ...`）。`--model`を省略するとランダム方策で撮影する（描画の動作確認用）
- 学習用環境(`camera=False`)は、GenesisがbuildでGLコンテキストを作るのを抑止している（Colab EGLのSegfault対策）

ローカルでの動作確認とColabでの本番学習で、同じこのコードを使い回す。
