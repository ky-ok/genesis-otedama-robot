"""
2本のFranka Pandaアームが、向かい合った相手の手にお手玉(球体)2個を
トスし合う課題のGenesis環境。

Mac(CPU、動作確認用の小規模並列)とGoogle Colab(GPU、本番の大規模並列学習)の
両方から同じコードを使う。学習アルゴリズム(PPO等)には依存しない、素の
リセット/ステップAPIのみを提供する。SB3等のラッパーは vec_env.py 側で行う。
"""

from __future__ import annotations

import numpy as np
import torch

import genesis as gs

ARM_OFFSET = 0.5
READY_POSE = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79]
FINGER_OPEN = 0.03  # 指を軽く開いた「受け皿」姿勢で固定する

N_ARM_DOFS = 7
N_OBS_PER_BALL = 6  # 相手側の手から見た相対位置(3) + 速度(3)
N_OBS_PER_ARM = N_ARM_DOFS * 2  # 関節角(7) + 関節角速度(7)
OBS_DIM = N_OBS_PER_ARM * 2 + N_OBS_PER_BALL * 2
ACTION_DIM = N_ARM_DOFS * 2  # 左腕7 + 右腕7 の目標角度に対する増分

MAX_DELTA = 0.08  # 1制御ステップあたりの関節目標角の最大変化量[rad]
DROP_HEIGHT = 0.05  # これより低くなったら「落とした」とみなす床上の高さ
BALL_START_HEIGHT = 0.55


class JugglingEnv:
    """向かい合った2本のPandaアームが2個の球を交換するバッチ環境。

    Genesisのn_envs機能でN個の同一シーンを並列シミュレーションする。
    観測・行動・報酬はすべて shape の先頭次元が n_envs のtorchテンソル。
    """

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.02,
        show_viewer: bool = False,
        device=None,
        camera: bool = False,
    ):
        self.n_envs = n_envs
        self.dt = dt
        self.device = device if device is not None else gs.device

        self.scene = gs.Scene(
            show_viewer=show_viewer,
            sim_options=gs.options.SimOptions(dt=dt),
        )

        self.scene.add_entity(gs.morphs.Plane())

        self.robot_left = self.scene.add_entity(
            gs.morphs.URDF(file="urdf/panda_bullet/panda.urdf", pos=(-ARM_OFFSET, 0.0, 0.0), fixed=True),
        )
        self.robot_right = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/panda_bullet/panda.urdf",
                pos=(ARM_OFFSET, 0.0, 0.0),
                euler=(0.0, 0.0, 180.0),
                fixed=True,
            ),
        )

        self.ball_a = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.03, pos=(-ARM_OFFSET * 0.6, 0.0, BALL_START_HEIGHT)),
            material=gs.materials.Rigid(rho=300.0),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2)),
        )
        self.ball_b = self.scene.add_entity(
            gs.morphs.Sphere(radius=0.03, pos=(ARM_OFFSET * 0.6, 0.0, BALL_START_HEIGHT)),
            material=gs.materials.Rigid(rho=300.0),
            surface=gs.surfaces.Default(color=(0.2, 0.4, 0.9)),
        )

        self.cam = None
        if camera:
            self.cam = self.scene.add_camera(
                res=(960, 540),
                pos=(0.0, -2.2, 1.4),
                lookat=(0.0, 0.0, 0.5),
                fov=45,
                GUI=False,
            )

        self.scene.build(n_envs=n_envs)

        self.left_dofs = [self.robot_left.get_joint(f"panda_joint{i}").dofs_idx_local[0] for i in range(1, 8)]
        self.right_dofs = [self.robot_right.get_joint(f"panda_joint{i}").dofs_idx_local[0] for i in range(1, 8)]
        self.left_finger_dofs = [
            self.robot_left.get_joint("panda_finger_joint1").dofs_idx_local[0],
            self.robot_left.get_joint("panda_finger_joint2").dofs_idx_local[0],
        ]
        self.right_finger_dofs = [
            self.robot_right.get_joint("panda_finger_joint1").dofs_idx_local[0],
            self.robot_right.get_joint("panda_finger_joint2").dofs_idx_local[0],
        ]

        # 固定関節(panda_hand, panda_grasptarget等)はURDF読み込み時にpanda_link7へ
        # 統合される(merge_fixed_links=True)ため、実在する指リンクの中点を
        # 「手の位置」として使う
        self.left_finger_links = [
            self.robot_left.get_link("panda_leftfinger").idx_local,
            self.robot_left.get_link("panda_rightfinger").idx_local,
        ]
        self.right_finger_links = [
            self.robot_right.get_link("panda_leftfinger").idx_local,
            self.robot_right.get_link("panda_rightfinger").idx_local,
        ]

        for robot, dofs in ((self.robot_left, self.left_dofs), (self.robot_right, self.right_dofs)):
            robot.set_dofs_kp(np.full(len(dofs), 800.0), dofs_idx_local=dofs)
            robot.set_dofs_kv(np.full(len(dofs), 80.0), dofs_idx_local=dofs)
        for robot, dofs in (
            (self.robot_left, self.left_finger_dofs),
            (self.robot_right, self.right_finger_dofs),
        ):
            robot.set_dofs_kp(np.full(len(dofs), 200.0), dofs_idx_local=dofs)
            robot.set_dofs_kv(np.full(len(dofs), 20.0), dofs_idx_local=dofs)

        self._left_target = torch.zeros((n_envs, N_ARM_DOFS), device=self.device)
        self._right_target = torch.zeros((n_envs, N_ARM_DOFS), device=self.device)

        self.reset()

    # ------------------------------------------------------------------
    def reset(self, envs_idx=None) -> torch.Tensor:
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs, device=self.device)
        n = len(envs_idx)
        if n == 0:
            return self._get_obs()

        ready = torch.tensor(READY_POSE, device=self.device, dtype=torch.float32)
        left_pose = ready.unsqueeze(0).repeat(n, 1)
        right_pose = ready.unsqueeze(0).repeat(n, 1)

        self.robot_left.set_dofs_position(left_pose, dofs_idx_local=self.left_dofs, envs_idx=envs_idx)
        self.robot_right.set_dofs_position(right_pose, dofs_idx_local=self.right_dofs, envs_idx=envs_idx)
        self.robot_left.set_dofs_position(
            torch.full((n, 2), FINGER_OPEN, device=self.device), dofs_idx_local=self.left_finger_dofs, envs_idx=envs_idx
        )
        self.robot_right.set_dofs_position(
            torch.full((n, 2), FINGER_OPEN, device=self.device), dofs_idx_local=self.right_finger_dofs, envs_idx=envs_idx
        )

        self._left_target[envs_idx] = ready
        self._right_target[envs_idx] = ready

        ball_a_pos = torch.tensor([-ARM_OFFSET * 0.6, 0.0, BALL_START_HEIGHT], device=self.device).repeat(n, 1)
        ball_b_pos = torch.tensor([ARM_OFFSET * 0.6, 0.0, BALL_START_HEIGHT], device=self.device).repeat(n, 1)
        self.ball_a.set_pos(ball_a_pos, envs_idx=envs_idx)
        self.ball_b.set_pos(ball_b_pos, envs_idx=envs_idx)

        return self._get_obs()

    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        """action: shape (n_envs, ACTION_DIM), 各成分は[-1, 1]を想定"""
        action = torch.clamp(action, -1.0, 1.0)
        left_delta = action[:, :N_ARM_DOFS] * MAX_DELTA
        right_delta = action[:, N_ARM_DOFS:] * MAX_DELTA

        self._left_target = self._left_target + left_delta
        self._right_target = self._right_target + right_delta

        self.robot_left.control_dofs_position(self._left_target, dofs_idx_local=self.left_dofs)
        self.robot_right.control_dofs_position(self._right_target, dofs_idx_local=self.right_dofs)

        self.scene.step()

        obs = self._get_obs()
        reward, dropped = self._compute_reward()
        done = dropped

        return obs, reward, done, {}

    # ------------------------------------------------------------------
    def _get_obs(self) -> torch.Tensor:
        left_q = self.robot_left.get_dofs_position(dofs_idx_local=self.left_dofs)
        left_dq = self.robot_left.get_dofs_velocity(dofs_idx_local=self.left_dofs)
        right_q = self.robot_right.get_dofs_position(dofs_idx_local=self.right_dofs)
        right_dq = self.robot_right.get_dofs_velocity(dofs_idx_local=self.right_dofs)

        left_hand_pos = self._hand_pos(self.robot_left, self.left_finger_links)
        right_hand_pos = self._hand_pos(self.robot_right, self.right_finger_links)

        ball_a_pos = self.ball_a.get_pos(relative=False)
        ball_a_vel = self.ball_a.get_vel(relative=False)
        ball_b_pos = self.ball_b.get_pos(relative=False)
        ball_b_vel = self.ball_b.get_vel(relative=False)

        # ball_a は左腕から右腕(相手)へ、ball_b は右腕から左腕へトスするのが目標なので、
        # それぞれの「行き先の手」から見た相対位置を観測にする
        ball_a_rel = ball_a_pos - right_hand_pos
        ball_b_rel = ball_b_pos - left_hand_pos

        obs = torch.cat(
            [left_q, left_dq, right_q, right_dq, ball_a_rel, ball_a_vel, ball_b_rel, ball_b_vel],
            dim=-1,
        )
        return obs

    @staticmethod
    def _hand_pos(robot, finger_links):
        pos = robot.get_links_pos(links_idx_local=finger_links, relative=False)
        return pos.mean(dim=1)

    def _compute_reward(self):
        left_hand_pos = self._hand_pos(self.robot_left, self.left_finger_links)
        right_hand_pos = self._hand_pos(self.robot_right, self.right_finger_links)

        ball_a_pos = self.ball_a.get_pos(relative=False)
        ball_b_pos = self.ball_b.get_pos(relative=False)

        # ball_a -> 右手, ball_b -> 左手 への到達を促す
        dist_a = torch.linalg.norm(ball_a_pos - right_hand_pos, dim=-1)
        dist_b = torch.linalg.norm(ball_b_pos - left_hand_pos, dim=-1)
        catch_reward = -(dist_a + dist_b)

        toss_height_reward = torch.clamp(ball_a_pos[:, 2] - BALL_START_HEIGHT, min=0.0) + torch.clamp(
            ball_b_pos[:, 2] - BALL_START_HEIGHT, min=0.0
        )

        dropped_a = ball_a_pos[:, 2] < DROP_HEIGHT
        dropped_b = ball_b_pos[:, 2] < DROP_HEIGHT
        dropped = dropped_a | dropped_b

        alive_reward = (~dropped).float() * 0.1
        drop_penalty = dropped.float() * -5.0

        reward = catch_reward + 0.5 * toss_height_reward + alive_reward + drop_penalty
        return reward, dropped

    # ------------------------------------------------------------------
    def record_rollout(self, policy_fn, n_steps: int, filename: str, fps: int | None = None):
        """学習済み方策(または任意の関数)を1エピソード分実行して動画に保存する。

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
