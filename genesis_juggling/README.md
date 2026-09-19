# genesis_juggling

Mac（`mac_dev/`）とColab（`colab/`）の両方から共通で使うコードを置く場所です。

- `env.py`: 向かい合った2本のFranka Pandaアームが球2個を交換するGenesisバッチ環境（`JugglingEnv`）。観測・行動・報酬はすべてtorchテンソル(先頭次元がn_envs)
- `vec_env.py`: `JugglingEnv`をStable-Baselines3の`VecEnv`インターフェースに橋渡しする`JugglingVecEnv`
- `training_utils.py`: 学習中に「チェックポイント + 報酬推移グラフ + ロールアウト動画 + ハイパーパラメータ」を一式まとめて保存する`CheckpointVideoCallback`（SB3コールバック）

ローカルでの動作確認とColabでの本番学習で、同じこのコードを使い回す。
