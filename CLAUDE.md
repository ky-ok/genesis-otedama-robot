# CLAUDE.md

このファイルは、本プロジェクトで今後Claude Code（このセッション含む）が作業する際の開発方針をまとめたものです。新しいセッションで作業を再開する際は、まずこの内容を前提として動いてください。

## プロジェクト概要

物理シミュレータ **Genesis** を使い、ロボットに「お手玉（複数個の玉を交互にトス&キャッチする動作）」をさせ、その様子を録画した動画（1〜5分、倍速再生可）を作成する。Physical AI Advanced課題の一環。AI要素として**強化学習（RL）**を用いる。

## タスク仕様（現時点で確定している内容）

- お手玉の個数: **2個**が最終目標。学習は**段階的**に行う（段階1: 玉1個で左右のお盆を往復 → 段階2: 2個で受け渡し。観測の形は同じなので`RESUME_FROM`で引き継げる）
- ロボット構成: **Franka Panda 2台を向かい合わせ**に配置し、**手首に縁付きのお盆(トレイ)を装着**して、玉をお盆に載せて空中に投げ合う
  - Pandaの指を上に向けて玉を載せる方式は、姿勢が不自然になるうえ玉が安定して載らなかったため不採用。お盆は`genesis_juggling/robot_asset.py`が元のURDFを実行時に書き換えて取り付ける
  - 操作する関節は投げに必要な縦平面の3関節(q2,q4,q6)のみ。他は基準姿勢で固定(RLの探索を現実的にするため)
  - お盆中心は基準姿勢で左右 ±0.25m(間隔0.5m)
- 本格的な連続juggling（3個以上の同時ジャグリング）はRLの学習難易度が高いため、まずは上記スコープで確実に完走させることを優先する

## 環境・報酬設計の要点(`genesis_juggling/env.py`)

- 玉は**お盆に載った状態からスタート**。宙に浮かせて始めると腕が何もできずに落ちる(2026-10-12に実際に起きた)
- **毎ステップの負の報酬を置かない**。負の報酬 + 落下で終了、だと「早く落としたほうが得」になる。進捗(相手のお盆への接近)・到達高さ・着地成功の正の報酬で作り、落下には一度きりのペナルティ
- 着地成功の条件: 目標のお盆の上でほぼ静止 + **両方のお盆より10cm以上高く上がった** + **前回の位置から水平に30cm以上移動**。成功すると目標が相手側へ切り替わり、受け渡しが往復して続く
- 時間切れ(500ステップ)は失敗ではないので`TimeLimit.truncated`としてSB3に伝える
- 学習の指標は報酬だけでなく**エピソードあたり着地成功数**と**エピソード長**も出す(`reward_curve.png`の3枚組、`metrics.csv`)

### 学習の進め方(カリキュラム)と、ローカルで確認できた結果
- 玉1個では、左右どちらのお盆から始めるかをランダムにする(左固定だと「右→左」の練習が足りず、片道しか覚えなかった)
- 着地成功には「お盆同士が`min_tray_gap`(最終0.42m)以上離れている」ことを要求する。最初から0.42を要求すると成功が出ず学習が進まなかったため、`GapCurriculumCallback`で0.20→0.42へ、学習の前半60%で段階的に広げる
- Mac CPU・256並列で、玉1個・10Mステップ(約7分)で、お盆の間隔0.46mの長い受け渡しを、左→右・右→左の両方向で約2.7回/エピソード学習できた(動画でも、玉が高く上がって空中を渡り、反対側のお盆に着地している)。ただし最終的には全エピソードで落とす
- 玉2個への引き継ぎ(段階2)は、3Mステップで約1.2回/エピソードで、玉1個より明らかに難しい(2個が空中ですれ違う)。長く回す必要がある

### 報酬の抜け道(過去の事例)
- 高さを「2つのお盆の平均」から測ると、お盆を下げるだけで玉が高く上がったことになり、玉に触れずに着地成功を量産された(1エピソード30回)。高さは「高い方のお盆」から測る
- お盆同士を接触寸前(0.15〜0.19m)まで近づけて、玉を短くポンと渡す抜け道も使われた。着地成功に最低間隔を要求して防ぐ(上のカリキュラム)
- いずれも、ローカルで実際に学習させて動画と数値(成功位置・お盆の間隔)を見たことで発覚した。**報酬を変えたら、学習曲線だけでなく動画と数値で挙動を確認すること**

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

**ローカル学習も実用的**: Mac(M4 Pro, CPU)で256並列・毎秒約1.6万ステップ出るため、数百万〜1千万ステップは数分〜十数分で回る。`mac_dev/train_local.py`で、Colabと同じ環境・コールバックで学習できる(Colabのヘッドレス描画の問題も無い)。まずローカルで報酬や挙動を確認し、規模を上げたいときにColabを使う、という使い分けができる。

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

## Colabで「不明な理由によりセッションがクラッシュ」したとき

2026-10-12に発生。`faulthandler`のスタックから**原因を特定済み**: Colab(Python 3.13, L4)で`JugglingEnv.__init__`の`scene.build()`中、Genesisのビジュアライザが作るオフスクリーン描画用のEGLコンテキスト初期化(`genesis/ext/pyrender/platforms/egl.py` `init_context`)でSegfaultしていた。Genesisはカメラもビューアもないシーンでもbuild時にGLコンテキストを作る。

対策(いずれもMac CPUでの動作確認のみ。Colab上での再確認は未了):
- 学習用環境(camera=False)は、`scene.build()`の間だけ`pyrender.OffscreenRenderer`を無効化し、GLに一切触れない(`env.py`の`_without_offscreen_renderer`)。Macで`_renderer is None`で学習が回ることを確認済み
- 動画撮影は別プロセス(`genesis_juggling/record_video.py`)で、EGLではなくソフトウェア描画のOSMesa(`PYOPENGL_PLATFORM=osmesa`, `apt: libosmesa6`)で行う。失敗しても学習は続行し、`video_error.log`が残る
- ノートブックに、学習前の動画撮影事前チェックセル(別プロセス)を追加
- `faulthandler.enable()`と`[メモリ:...]`出力、既定の小さい設定(`n_envs=64`)は診断用に維持
- OSMesaでの撮影がColabで動くかは未検証。失敗したらエラーを共有してもらい、`record_video: False`で学習だけ先に進める

続報(同日): 動画の事前チェックで別の異常終了 `LLVM ERROR: inconsistency in registered CommandLine options` が発生。直前にTensorFlowの起動ログが出ており、`stable_baselines3` → `torch.utils.tensorboard` → `tensorboard.compat.tf` → `import tensorflow` の経路(使い捨てvenvで確認)でTensorFlowが読み込まれ、そのLLVMがGenesis(quadrants)のLLVMと1プロセス内で衝突したと判断。`genesis_juggling/__init__.py`で`sys.modules["tensorflow"] = None`にして読み込みを抑止した(tensorboardは自前スタブにフォールバックし、SB3は問題なく動くことを確認済み)。このパッケージはSB3・Genesisより先にimportすること。Colab上での再確認は未了。

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
