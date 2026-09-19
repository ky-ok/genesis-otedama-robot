# CLAUDE.md

このファイルは、本プロジェクトで今後Claude Code（このセッション含む）が作業する際の開発方針をまとめたものです。新しいセッションで作業を再開する際は、まずこの内容を前提として動いてください。

## プロジェクト概要

物理シミュレータ **Genesis** を使い、ロボットに「お手玉（複数個の玉を交互にトス&キャッチする動作）」をさせ、その様子を録画した動画（1〜5分、倍速再生可）を作成する。Physical AI Advanced課題の一環。AI要素として**強化学習（RL）**を用いる。

## タスク仕様（現時点で確定している内容）

- お手玉の個数: **2個**からスタート（将来的に3個以上への拡張は検討事項）
- ロボット構成: **2アーム構成**（両腕、または2台のロボットアーム）で、左右の腕の間で玉を交互にトスし合う
- 本格的な連続juggling（3個以上の同時ジャグリング）はRLの学習難易度が高いため、まずは上記スコープで確実に完走させることを優先する

## AI活用方針

- **強化学習（PPO）** でトス&キャッチの制御方策を学習する
- Genesisの並列シミュレーション（GPU上で多数envを同時実行）を活かす設計にする
- 報酬設計の方針（初期案、学習しながら調整する）:
  - 玉が受け手側の手/グリッパー付近に来たら加点（キャッチ成功）
  - 玉が一定の高さまでトスされたら加点
  - 玉を落とさず継続できた時間・往復回数に応じて加点
  - 動きが暴れすぎないようエネルギー/滑らかさにペナルティ

## ローカル環境構築の方針

- Mac上でPythonパッケージをインストールする際は、システム環境を汚さないよう**必ず`uv`で仮想環境（`.venv/`）を作成してから**作業する
- 仮想環境は `uv venv .venv --python 3.11`、依存関係は `uv pip install -r requirements.txt` で導入する
- `.venv/` はリポジトリには含めない（`.gitignore`済み）

## 開発環境と役割分担

ユーザーの開発機はMac（GPUなし）。GenesisのRL学習はGPU前提のため、以下のように役割を分ける。

| 作業 | 場所 | 理由 |
|---|---|---|
| シーン作成・デバッグ・報酬設計の確認 | Mac（ローカル、`mac_dev/`） | 軽い動作確認、GUIビューアで挙動を目視できる |
| RL学習（本番、並列env） | **Google Colab（GPU）**（`colab/`） | CUDA + 並列シミュレーションが必要 |
| 最終方策のロールアウト録画 | Colab（学習環境と同一構成） | 学習時と評価時で物理パラメータがずれるのを防ぐ |
| 動画の倍速編集・テロップ | Mac（ffmpeg等） | 単純な後処理なのでローカルで十分 |

## 開発ロードマップ

1. **[完了]** Mac上でGenesisをインストールし、ロボット(2アーム)+玉(2個)のシーンのみ確認する（学習なし、固定/手動の動作で見た目を確認する段階）。`mac_dev/scene_check.py`
   - 2アーム構成は、Genesis同梱の**Franka Panda（URDF, `urdf/panda_bullet/panda.urdf`）を2台、向かい合わせに配置**する方式を採用した
   - Genesisには「bi-franka_panda」という胴体+両腕一体型のMJCFアセットも同梱されているが、このバージョン(Genesis 1.4.1)のMJCFパーサは`<worldbody>`配下にネストした`<include>`タグを解決できずロードに失敗したため不採用とした（body直下に置かれたincludeまでは解決するが、更に深い階層のincludeは非対応）
2. **[完了]** シーン構築コードをColabに移植し、RL学習ループ（PPO、並列env、Google Driveへのチェックポイント保存）を追加する
   - `genesis_juggling/env.py`: `JugglingEnv`（Genesisのn_envsによるバッチ並列環境。観測・行動・報酬はtorchテンソル）
   - `genesis_juggling/vec_env.py`: `JugglingVecEnv`（Stable-Baselines3のVecEnvへのラッパー）
   - `genesis_juggling/training_utils.py`: `CheckpointVideoCallback`（後述の「3点セット」を定期保存するSB3コールバック）
   - `colab/train_ppo.ipynb`: 上記を組み合わせたPPO学習ノートブック
   - Mac上のCPUで`mac_dev/train_smoke_test.py`により、環境構築→PPO学習→チェックポイント/動画/報酬曲線の出力までのパイプライン全体が壊れていないことを極小規模で確認済み。ただしColab GPU上での実運用はまだ未検証
3. ユーザー自身がColab上でPPO学習を実行し、Claudeと一緒にハイパーパラメータを調整しながら報酬設計を洗練させる
4. 最終方策でロールアウトを録画する
5. Macに動画を持ち帰り、倍速編集・テロップ付けを行い、1〜5分の最終動画に仕上げる

## Colab利用時の注意点

- Colabにはディスプレイがないため `show_viewer=False` とし、カメラを追加して `camera.render()` でフレームを取得するヘッドレスレンダリング構成にする
- EGL/OpenGL関連の依存関係が不足する場合は `apt-get` 等で追加インストールする
- Colabのランタイムは揮発性（切断でファイル・変数が消える）なので、Google Driveをマウントして学習チェックポイントを定期保存し、再開可能にする
- 生成した動画・学習済み重みはGoogle Drive経由でダウンロードし、Macに持ち帰る

## パラメータ調整の相談の受け方

RL学習の実行はユーザー自身がGoogle Colab上で行う。Claudeへの相談は以下の一式をアップロードしてもらう形を基本とする（動画だけだと、探索不足なのか報酬設計の問題なのか学習途中なだけなのかの判断が難しいため）。

- `rollout_*.mp4`（現在の方策の挙動）
- `reward_curve.png`（報酬の推移）
- `hyperparams.json`（そのときのハイパーパラメータ）

これらは`colab/train_ppo.ipynb`の`CheckpointVideoCallback`がGoogle Drive上に自動でまとめて出力する（`genesis_juggling/training_utils.py`）。

## リポジトリ構成

```
CLAUDE.md           このファイル
README.md           プロジェクト概要（GitHub公開用）
requirements.txt    共通の依存関係（genesis-world, torch, stable-baselines3等）
mac_dev/            Mac上でのシーン確認・デバッグ用スクリプト
colab/              Google Colab用の学習ノートブック(train_ppo.ipynb)
genesis_juggling/   Mac/Colab共通で使うコード（env.py, vec_env.py, training_utils.py）
videos/             出力動画
checkpoints/         学習済みモデルの重み（.gitignore対象、サイズが大きいため）
```

## GitHub

- リポジトリ: `ky-ok/genesis-otedama-robot`（public、他者が閲覧可能）
- 学習済み重み・生成動画などサイズの大きいファイルはリポジトリに含めず `.gitignore` で除外する（必要ならGoogle Drive等で共有）
