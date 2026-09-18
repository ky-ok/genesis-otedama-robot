# genesis-otedama-robot

物理シミュレータ [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) を使い、2本のロボットアームが強化学習（PPO）で学習した方策により、お手玉2個を左右に交互にトス&キャッチする様子を動画化するプロジェクトです。

Physical AI Advanced課題の一環として作成しています。

## 概要

- **お手玉の個数**: 2個
- **ロボット構成**: 2アーム構成で左右に玉をトスし合う
- **AI手法**: 強化学習（PPO）。GenesisのGPU並列シミュレーションを利用して学習
- **開発環境**: シーン開発・デバッグはローカルMac、RL学習はGoogle Colab（GPU）

開発方針の詳細は [CLAUDE.md](./CLAUDE.md) を参照してください。

## ディレクトリ構成

```
mac_dev/            Mac上でのシーン確認・デバッグ用スクリプト
colab/              Google Colab用の学習ノートブック
genesis_juggling/   共通コード（シーン定義、報酬関数、環境ラッパー等）
videos/             出力動画
checkpoints/        学習済みモデルの重み
```

## セットアップ（ローカル）

```bash
pip install genesis-world
```

学習はGoogle Colab上のGPUランタイムで実行します。詳細は `colab/` 以下のノートブックを参照してください。
