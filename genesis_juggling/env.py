"""
向かい合った2本のFranka Pandaアーム(手首に縁付きのお盆を装着)が、お手玉に見立てた
球を空中に投げ上げて、相手のお盆に受け渡しし合う課題のGenesis環境。

Mac(CPU、動作確認用の小規模並列)とGoogle Colab(GPU、本番の大規模並列学習)の
両方から同じコードを使う。学習アルゴリズム(PPO等)には依存しない、素の
リセット/ステップAPIのみを提供する。SB3等のラッパーは vec_env.py 側で行う。

設計の要点:
- 玉はお盆に載った状態からスタートする(宙に浮かせて始めると、腕が何もできないまま落ちる)
- 操作する関節は投げに必要な縦平面の3関節(q2, q4, q6)のみ。他は基準姿勢で固定
- 報酬は「進捗(相手のお盆への接近)」「到達高さ」「着地成功」で作り、毎ステップの負の報酬は
  置かない(負の報酬 + 落下で終了、だと「早く落としたほうが得」になるため)
- 着地成功した玉は、次は元のお盆へ戻す番になる(受け渡しが往復して続く)
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.ext import pyrender

from .robot_asset import TRAY_XYZ, panda_with_tray_urdf

# --- 配置 ---------------------------------------------------------------
# 基準姿勢でのお盆中心は、アーム土台から+0.623m前方。お盆同士の間隔 = 2*(ARM_OFFSET-0.623)
ARM_OFFSET = 0.873
READY_POSE = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79]
ACTIVE_JOINTS = (1, 3, 5)  # q2, q4, q6 (0始まりの添字)
N_ARM_DOFS = 7
N_ACTIVE = len(ACTIVE_JOINTS)
BALL_RADIUS = 0.03
BALL_REST_DZ = 0.04  # お盆中心から玉中心までの、載せたときの高さ
PARKED_BALL_POS = (3.0, 3.0, 0.5)  # 使わない玉の置き場所(遠くへ退避)

# --- 空間 ---------------------------------------------------------------
OBS_PER_BALL = 10  # お盆(左)からの相対位置3 + お盆(右)からの相対位置3 + 速度3 + 目標フラグ1
OBS_DIM = N_ARM_DOFS * 2 * 2 + 6 + OBS_PER_BALL * 2
ACTION_DIM = N_ACTIVE * 2  # 左腕3 + 右腕3 の目標角度に対する増分

MAX_DELTA = 0.1  # 1制御ステップあたりの関節目標角の最大変化量[rad]
MAX_OFFSET = 0.9  # 基準姿勢からの関節目標角の最大ずれ[rad]

# --- エピソード ----------------------------------------------------------
MAX_EPISODE_STEPS = 500
TRAY_READY_Z = 0.602  # 基準姿勢でのお盆中心の高さ
DROP_Z = 0.35  # 玉がこの高さより低くなったら「落とした」
WORKSPACE_X = 1.2
WORKSPACE_Y = 0.8

# --- 報酬 ---------------------------------------------------------------
W_PROGRESS = 5.0  # 相手のお盆へ1m近づくごとの報酬
W_APEX = 20.0  # お盆より高く上がった分(最大 APEX_CAP)の報酬。新記録を出したときだけ
APEX_CAP = 0.30
MIN_APEX = 0.10  # 着地成功とみなすために必要な、両方のお盆より高く上がった最低高さ[m]
MIN_TRAVEL = 0.30  # 着地成功とみなすために必要な、前回の位置からの水平移動距離[m]
MIN_TRAY_GAP = 0.42  # お盆同士がこれ以上離れているときだけ着地成功とみなす[m](基準は0.50)。近づけて短く渡す抜け道を防ぐ
CLOSE_PENALTY = 0.1  # お盆がMIN_TRAY_GAPより近づいたとき、近づいた距離[m]あたりの1ステップの減点
CATCH_BONUS = 10.0
CATCH_XY = 0.07  # 着地成功とみなすお盆中心からの水平距離[m]
CATCH_SPEED = 0.6  # 着地成功とみなす玉の速さの上限[m/s]
DROP_PENALTY = 2.0
ACTION_PENALTY = 0.002


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q: (N,4) wxyz, v: (3,) を回転させる。"""
    w = q[:, :1]
    u = q[:, 1:]
    t = 2.0 * torch.cross(u, v.expand_as(u), dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


@contextlib.contextmanager
def _without_offscreen_renderer():
    """scene.build()中だけ、オフスクリーン描画用のGLコンテキスト作成を省略する。

    Genesisは、カメラもビューアも無いシーンでもbuild時にEGL等のGLコンテキストを作る。
    Colabのヘッドレス環境ではこのEGL初期化でSegfaultしたため、描画しない学習用環境では
    GLに一切触れないようにする。
    """
    original = pyrender.OffscreenRenderer
    pyrender.OffscreenRenderer = lambda *args, **kwargs: None
    try:
        yield
    finally:
        pyrender.OffscreenRenderer = original


class JugglingEnv:
    """向かい合った2本のPandaアームが、お盆の上の玉を投げ合うバッチ環境。

    Genesisのn_envs機能でN個の同一シーンを並列シミュレーションする。
    観測・行動・報酬はすべて shape の先頭次元が n_envs のtorchテンソル。

    n_balls=1 のときは玉aだけを使う(左→右→左…と往復)。n_balls=2 のときは玉bも使い、
    右の玉は左へ、左の玉は右へ、それぞれ投げる。観測の次元は n_balls によらず同じなので、
    1個で学習した方策を2個の学習へ引き継げる。
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.02,
        n_balls: int = 2,
        show_viewer: bool = False,
        device=None,
        camera: bool = False,
    ):
        assert n_balls in (1, 2)
        self.n_envs = n_envs
        self.dt = dt
        self.n_balls = n_balls
        self.device = device if device is not None else gs.device

        self.scene = gs.Scene(
            show_viewer=show_viewer,
            sim_options=gs.options.SimOptions(dt=dt),
        )
        self.scene.add_entity(gs.morphs.Plane())

        urdf = panda_with_tray_urdf()
        # 重力補償により、PD制御で腕が自重でたわまず、お盆が水平に保たれる
        self.robot_left = self.scene.add_entity(
            gs.morphs.URDF(file=urdf, pos=(-ARM_OFFSET, 0.0, 0.0), fixed=True),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        self.robot_right = self.scene.add_entity(
            gs.morphs.URDF(file=urdf, pos=(ARM_OFFSET, 0.0, 0.0), euler=(0.0, 0.0, 180.0), fixed=True),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

        self.ball_a = self.scene.add_entity(
            gs.morphs.Sphere(radius=BALL_RADIUS, pos=(-0.25, 0.0, 0.64)),
            material=gs.materials.Rigid(rho=300.0),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2)),
        )
        self.ball_b = self.scene.add_entity(
            gs.morphs.Sphere(radius=BALL_RADIUS, pos=(0.25, 0.0, 0.64)),
            material=gs.materials.Rigid(rho=300.0),
            surface=gs.surfaces.Default(color=(0.2, 0.4, 0.9)),
        )

        self.cam = None
        if camera:
            self.cam = self.scene.add_camera(
                res=(960, 540),
                pos=(0.0, -1.7, 0.95),
                lookat=(0.0, 0.0, 0.5),
                fov=55,
                GUI=False,
            )

        if camera or show_viewer:
            self.scene.build(n_envs=n_envs)
        else:
            with _without_offscreen_renderer():
                self.scene.build(n_envs=n_envs)

        self.left_dofs = [self.robot_left.get_joint(f"panda_joint{i}").dofs_idx_local[0] for i in range(1, 8)]
        self.right_dofs = [self.robot_right.get_joint(f"panda_joint{i}").dofs_idx_local[0] for i in range(1, 8)]
        self.left_finger_dofs = [
            self.robot_left.get_joint(f"panda_finger_joint{i}").dofs_idx_local[0] for i in (1, 2)
        ]
        self.right_finger_dofs = [
            self.robot_right.get_joint(f"panda_finger_joint{i}").dofs_idx_local[0] for i in (1, 2)
        ]
        self.left_link7 = self.robot_left.get_link("panda_link7").idx_local
        self.right_link7 = self.robot_right.get_link("panda_link7").idx_local

        for robot, dofs, fdofs in (
            (self.robot_left, self.left_dofs, self.left_finger_dofs),
            (self.robot_right, self.right_dofs, self.right_finger_dofs),
        ):
            robot.set_dofs_kp(np.full(len(dofs), 800.0), dofs_idx_local=dofs)
            robot.set_dofs_kv(np.full(len(dofs), 80.0), dofs_idx_local=dofs)
            robot.set_dofs_kp(np.full(2, 200.0), dofs_idx_local=fdofs)
            robot.set_dofs_kv(np.full(2, 20.0), dofs_idx_local=fdofs)

        self._ready = torch.tensor(READY_POSE, device=self.device, dtype=torch.float32)
        self._active = list(ACTIVE_JOINTS)
        self._tray_local = torch.tensor(TRAY_XYZ, device=self.device, dtype=torch.float32)
        self._ball_active = torch.tensor([True, n_balls == 2], device=self.device)
        # カリキュラムで学習中に変更する(training_utils.GapCurriculumCallback)。評価・動画撮影では既定値のまま
        self.min_tray_gap = MIN_TRAY_GAP

        n = n_envs
        self._left_off = torch.zeros((n, N_ACTIVE), device=self.device)
        self._right_off = torch.zeros((n, N_ACTIVE), device=self.device)
        self._finger_zero = torch.zeros((n, 2), device=self.device)
        # 各玉が次に向かうお盆: 1.0=右のお盆, 0.0=左のお盆 (玉aは左から右へ、玉bは右から左へ)
        self._tgt = torch.tensor([[1.0, 0.0]], device=self.device).repeat(n, 1)
        self._apex = torch.zeros((n, 2), device=self.device)
        self._prev_dist = torch.zeros((n, 2), device=self.device)
        self._launch_x = torch.zeros((n, 2), device=self.device)
        self._step_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self._episode_catches = torch.zeros(n, device=self.device)

        self.reset()

    # ------------------------------------------------------------------
    def _tray_center(self, robot, link7, envs_idx=None) -> torch.Tensor:
        p7 = robot.get_links_pos(links_idx_local=[link7], envs_idx=envs_idx, relative=False)[:, 0]
        q7 = robot.get_links_quat(links_idx_local=[link7], envs_idx=envs_idx, relative=False)[:, 0]
        return p7 + _quat_rotate(q7, self._tray_local)

    def _targets(self, off: torch.Tensor) -> torch.Tensor:
        target = self._ready.unsqueeze(0).repeat(off.shape[0], 1)
        target[:, self._active] += off
        return target

    # ------------------------------------------------------------------
    def reset(self, envs_idx=None) -> torch.Tensor:
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs, device=self.device)
        n = len(envs_idx)
        if n == 0:
            return self._get_obs()

        zeros_off = torch.zeros((n, N_ACTIVE), device=self.device)
        self._left_off[envs_idx] = zeros_off
        self._right_off[envs_idx] = zeros_off
        pose = self._targets(zeros_off)

        for robot, dofs, fdofs in (
            (self.robot_left, self.left_dofs, self.left_finger_dofs),
            (self.robot_right, self.right_dofs, self.right_finger_dofs),
        ):
            robot.set_dofs_position(pose, dofs_idx_local=dofs, envs_idx=envs_idx)
            robot.set_dofs_position(self._finger_zero[:n], dofs_idx_local=fdofs, envs_idx=envs_idx)
            robot.control_dofs_position(pose, dofs_idx_local=dofs, envs_idx=envs_idx)
            robot.control_dofs_position(self._finger_zero[:n], dofs_idx_local=fdofs, envs_idx=envs_idx)

        tray_l = self._tray_center(self.robot_left, self.left_link7, envs_idx)
        tray_r = self._tray_center(self.robot_right, self.right_link7, envs_idx)
        rest = torch.tensor([0.0, 0.0, BALL_REST_DZ], device=self.device)
        tgt = torch.tensor([1.0, 0.0], device=self.device).repeat(n, 1)  # 玉aは左→右、玉bは右→左
        if self.n_balls == 2:
            self.ball_a.set_pos(tray_l + rest, envs_idx=envs_idx)
            self.ball_b.set_pos(tray_r + rest, envs_idx=envs_idx)
        else:
            # 玉1個のときは、左右どちらのお盆から始めるかをランダムにして、両方向の投げを同じだけ練習させる
            # (左固定だと「右→左へ返す」練習機会が少なく、片道しか覚えなかった)
            start_right = torch.rand(n, device=self.device) < 0.5
            self.ball_a.set_pos(torch.where(start_right.unsqueeze(-1), tray_r, tray_l) + rest, envs_idx=envs_idx)
            tgt[:, 0] = torch.where(start_right, 0.0, 1.0)
            self.ball_b.set_pos(
                torch.tensor(PARKED_BALL_POS, device=self.device).repeat(n, 1), envs_idx=envs_idx
            )

        self._tgt[envs_idx] = tgt
        self._apex[envs_idx] = 0.0
        self._step_count[envs_idx] = 0
        self._episode_catches[envs_idx] = 0.0

        tray_l_all = self._tray_center(self.robot_left, self.left_link7)
        tray_r_all = self._tray_center(self.robot_right, self.right_link7)
        tray_tgt = self._target_tray(tray_l_all, tray_r_all)
        ball_pos_now = self._ball_pos()
        self._prev_dist[envs_idx] = torch.linalg.norm(ball_pos_now - tray_tgt, dim=-1)[envs_idx]
        self._launch_x[envs_idx] = ball_pos_now[:, :, 0][envs_idx]

        return self._get_obs()

    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        """action: shape (n_envs, ACTION_DIM), 各成分は[-1, 1]を想定"""
        action = torch.clamp(action, -1.0, 1.0)
        self._left_off = torch.clamp(self._left_off + action[:, :N_ACTIVE] * MAX_DELTA, -MAX_OFFSET, MAX_OFFSET)
        self._right_off = torch.clamp(self._right_off + action[:, N_ACTIVE:] * MAX_DELTA, -MAX_OFFSET, MAX_OFFSET)

        self.robot_left.control_dofs_position(self._targets(self._left_off), dofs_idx_local=self.left_dofs)
        self.robot_right.control_dofs_position(self._targets(self._right_off), dofs_idx_local=self.right_dofs)
        self.robot_left.control_dofs_position(self._finger_zero, dofs_idx_local=self.left_finger_dofs)
        self.robot_right.control_dofs_position(self._finger_zero, dofs_idx_local=self.right_finger_dofs)

        self.scene.step()
        self._step_count += 1

        reward, dropped = self._compute_reward(action)
        truncated = (self._step_count >= MAX_EPISODE_STEPS) & ~dropped
        done = dropped | truncated
        obs = self._get_obs()
        info = {"truncated": truncated, "catches": self._episode_catches.clone()}
        return obs, reward, done, info

    # ------------------------------------------------------------------
    def _ball_pos(self) -> torch.Tensor:
        return torch.stack([self.ball_a.get_pos(relative=False), self.ball_b.get_pos(relative=False)], dim=1)

    def _ball_vel(self) -> torch.Tensor:
        return torch.stack([self.ball_a.get_vel(relative=False), self.ball_b.get_vel(relative=False)], dim=1)

    def _target_tray(self, tray_l: torch.Tensor, tray_r: torch.Tensor) -> torch.Tensor:
        """各玉の目標のお盆中心。shape (n_envs, 2, 3)"""
        return torch.where(self._tgt.unsqueeze(-1) > 0.5, tray_r.unsqueeze(1), tray_l.unsqueeze(1))

    def _get_obs(self) -> torch.Tensor:
        left_q = self.robot_left.get_dofs_position(dofs_idx_local=self.left_dofs)
        left_dq = self.robot_left.get_dofs_velocity(dofs_idx_local=self.left_dofs)
        right_q = self.robot_right.get_dofs_position(dofs_idx_local=self.right_dofs)
        right_dq = self.robot_right.get_dofs_velocity(dofs_idx_local=self.right_dofs)

        tray_l = self._tray_center(self.robot_left, self.left_link7)
        tray_r = self._tray_center(self.robot_right, self.right_link7)

        ball_pos = self._ball_pos()
        ball_vel = self._ball_vel()
        per_ball = torch.cat(
            [
                ball_pos - tray_l.unsqueeze(1),
                ball_pos - tray_r.unsqueeze(1),
                ball_vel,
                self._tgt.unsqueeze(-1),
            ],
            dim=-1,
        )
        per_ball = per_ball * self._ball_active.view(1, 2, 1)

        return torch.cat([left_q, left_dq, right_q, right_dq, tray_l, tray_r, per_ball.flatten(1)], dim=-1)

    def _compute_reward(self, action: torch.Tensor):
        tray_l = self._tray_center(self.robot_left, self.left_link7)
        tray_r = self._tray_center(self.robot_right, self.right_link7)
        ball_pos = self._ball_pos()
        ball_vel = self._ball_vel()
        active = self._ball_active.view(1, 2).float()

        # 到達高さ: 「高い方のお盆」からの最高到達高さを更新したときだけ、その増分を報酬にする。
        # (お盆を下げたり、玉を載せたままお盆ごと持ち上げたりしても、高く上がったことにならない)
        tray_gap = torch.linalg.norm(tray_r[:, :2] - tray_l[:, :2], dim=-1)
        tray_z_top = torch.maximum(tray_l[:, 2], tray_r[:, 2])
        height = ball_pos[:, :, 2] - tray_z_top.unsqueeze(-1) - BALL_REST_DZ
        new_apex = torch.clamp(torch.maximum(self._apex, height), max=APEX_CAP)
        apex_gain = (new_apex - self._apex) * active
        self._apex = new_apex

        # 進捗: 目標のお盆へ近づいた分
        tray_tgt = self._target_tray(tray_l, tray_r)
        dist = torch.linalg.norm(ball_pos - tray_tgt, dim=-1)
        progress = torch.clamp(self._prev_dist - dist, -0.15, 0.15) * active

        # 着地成功: 目標のお盆の上で、十分に高く上げた後、ほぼ静止した
        rel = ball_pos - tray_tgt
        horiz = torch.linalg.norm(rel[:, :, :2], dim=-1)
        speed = torch.linalg.norm(ball_vel, dim=-1)
        caught = (
            (horiz < CATCH_XY)
            & (rel[:, :, 2] > 0.0)
            & (rel[:, :, 2] < 0.08)
            & (speed < CATCH_SPEED)
            & (self._apex >= MIN_APEX)
            & ((ball_pos[:, :, 0] - self._launch_x).abs() >= MIN_TRAVEL)
            & (tray_gap >= self.min_tray_gap).unsqueeze(-1)
            & self._ball_active.view(1, 2)
        )

        # 成功した玉は目標を元のお盆へ切り替える。距離の基準も切り替え後に取り直し、報酬が飛ばないようにする
        self._tgt = torch.where(caught, 1.0 - self._tgt, self._tgt)
        self._apex = torch.where(caught, torch.zeros_like(self._apex), self._apex)
        self._launch_x = torch.where(caught, ball_pos[:, :, 0], self._launch_x)
        tray_tgt = self._target_tray(tray_l, tray_r)
        dist_after = torch.linalg.norm(ball_pos - tray_tgt, dim=-1)
        self._prev_dist = dist_after
        self._episode_catches += caught.float().sum(-1)

        out = (
            (ball_pos[:, :, 2] < DROP_Z)
            | (ball_pos[:, :, 0].abs() > WORKSPACE_X)
            | (ball_pos[:, :, 1].abs() > WORKSPACE_Y)
        ) & self._ball_active.view(1, 2)
        dropped = out.any(dim=-1)

        reward = (
            W_PROGRESS * progress.sum(-1)
            + W_APEX * apex_gain.sum(-1)
            + CATCH_BONUS * caught.float().sum(-1)
            - DROP_PENALTY * dropped.float()
            - CLOSE_PENALTY * torch.clamp(self.min_tray_gap - tray_gap, min=0.0)
            - ACTION_PENALTY * (action**2).sum(-1)
        )
        return reward, dropped

    # ------------------------------------------------------------------
    def record_rollout(self, policy_fn, n_steps: int, filename: str, fps: int | None = None):
        """学習済み方策(または任意の関数)をn_steps分実行して動画に保存する。

        camera=True かつ n_envs==1 で作った環境でのみ使用できる。
        policy_fn は obs(shape=(1, OBS_DIM)のtorch.Tensor)を受け取り、
        action(shape=(1, ACTION_DIM))を返す呼び出し可能オブジェクト。
        """
        if self.cam is None:
            raise RuntimeError("record_rollout requires the env to be built with camera=True.")
        if self.n_envs != 1:
            raise RuntimeError("record_rollout only supports n_envs=1.")

        obs = self.reset()
        self.cam.start_recording(save_to_filename=filename, fps=fps or int(round(1.0 / self.dt)))
        for _ in range(n_steps):
            action = policy_fn(obs)
            obs, _, done, _ = self.step(action)
            if bool(done[0]):
                obs = self.reset()
        self.cam.stop_recording()
