"""観測の次元が増える前に学習したPPOチェックポイントを、今の環境で使うための道具。

観測の末尾に特徴量(お盆の傾き・速度)を足したので、それ以前のチェックポイントは入力の次元が足りない。
- 推論(評価・動画): 観測の末尾を切り捨てて、そのチェックポイントの次元に合わせる(`predict`)
- 続きの学習: 新しい次元のモデルを作り、足した入力の重みをゼロで埋めて引き継ぐ(`load_ppo_for_env`)。
  ゼロで埋めるので、最初は元の方策とまったく同じ動作で、そこから新しい特徴量を使うように学習が進む
"""

from __future__ import annotations

import math

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_schedule_fn

# 入力に最も近い層の重み。これらだけ入力の次元に依存する
_INPUT_LAYERS = ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")


def predict(model: PPO, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
    """チェックポイントの観測次元に合わせて末尾を切り捨ててから、行動を返す。"""
    dim = model.observation_space.shape[0]
    action, _ = model.predict(obs[..., :dim], deterministic=deterministic)
    return action


def load_ppo_for_env(path: str, env, device="auto") -> PPO:
    """チェックポイントを読み込み、envの観測次元が増えていれば、新しい入力の重みをゼロで埋めて引き継ぐ。"""
    old = PPO.load(path, device="cpu")
    old_dim = old.observation_space.shape[0]
    new_dim = env.observation_space.shape[0]
    if old_dim == new_dim:
        model = PPO.load(path, env=env, device=device)
        model.verbose = 1
        return model
    if old_dim > new_dim:
        raise ValueError(f"チェックポイントの観測次元({old_dim})が、環境({new_dim})より大きいです。")

    new = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=old.policy_kwargs,
        learning_rate=old.learning_rate,
        n_steps=old.n_steps,
        batch_size=old.batch_size,
        n_epochs=old.n_epochs,
        gamma=old.gamma,
        gae_lambda=old.gae_lambda,
        clip_range=old.clip_range(1.0),
        ent_coef=old.ent_coef,
        verbose=1,  # 学習の進捗表の表示(評価指標の確認に必要)
        device=device,
    )
    state = new.policy.state_dict()
    for key, value in old.policy.state_dict().items():
        if value.shape == state[key].shape:
            state[key] = value.clone().to(state[key].device)
        elif key in _INPUT_LAYERS:
            padded = torch.zeros_like(state[key])
            padded[:, :old_dim] = value.to(padded.device)
            state[key] = padded
        else:
            raise ValueError(f"想定外の形の違い: {key} {tuple(value.shape)} -> {tuple(state[key].shape)}")
    new.policy.load_state_dict(state)
    return new


def apply_resume_settings(model: PPO, gamma=None, ent_coef=None, reset_std=None, learning_rate=None) -> PPO:
    """続きから学習するときの設定を、読み込んだモデルに確実に反映する。

    PPO.loadしたモデルは、保存時の設定(割引率・学習率など)を引き継ぐので、変えたいものはここで上書きする。
    割引率は、ロールアウトバッファが別に持っているのでそちらも更新する。学習率はスケジュール関数ごと差し替える。
    """
    if gamma is not None:
        model.gamma = gamma
        model.rollout_buffer.gamma = gamma
    if ent_coef is not None:
        model.ent_coef = ent_coef
    if reset_std is not None:
        model.policy.log_std.data.fill_(math.log(reset_std))
    if learning_rate is not None:
        model.learning_rate = learning_rate
        model.lr_schedule = get_schedule_fn(learning_rate)
    return model
