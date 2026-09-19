"""
Mac上でのシーン確認用スクリプト。

Genesis同梱のFranka Panda（URDF）を2台、向かい合わせに配置し、お手玉に見立てた
球体2個を間に置いて、腕を単純な周期運動で動かしてモデル・物理・カメラ録画が
正しく動くことを確認する。

学習は行わない（強化学習によるトス&キャッチの獲得はGoogle Colab側で行う）。
"""

import os

import numpy as np

import genesis as gs

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "videos", "scene_check.mp4")

# Frankaの一般的な「レディポーズ」（肘を曲げて前に構えた姿勢）
READY_POSE = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79]

ARM_OFFSET = 0.5  # 2台のアームを原点からどれだけ離して向かい合わせに置くか
DT = 0.02


def main():
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=DT),
    )

    scene.add_entity(gs.morphs.Plane())

    robot_left = scene.add_entity(
        gs.morphs.URDF(file="urdf/panda_bullet/panda.urdf", pos=(-ARM_OFFSET, 0.0, 0.0), fixed=True),
    )
    robot_right = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/panda_bullet/panda.urdf",
            pos=(ARM_OFFSET, 0.0, 0.0),
            euler=(0.0, 0.0, 180.0),
            fixed=True,
        ),
    )

    ball_left = scene.add_entity(
        gs.morphs.Sphere(radius=0.03, pos=(-0.15, 0.0, 0.6)),
        material=gs.materials.Rigid(rho=300.0),
        surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2)),
    )
    ball_right = scene.add_entity(
        gs.morphs.Sphere(radius=0.03, pos=(0.15, 0.0, 0.6)),
        material=gs.materials.Rigid(rho=300.0),
        surface=gs.surfaces.Default(color=(0.2, 0.4, 0.9)),
    )

    cam = scene.add_camera(
        res=(960, 540),
        pos=(0.0, -2.2, 1.4),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
        GUI=False,
    )

    scene.build()

    left_dofs = [robot_left.get_joint(f"panda_joint{i}").dofs_idx_local[0] for i in range(1, 8)]
    right_dofs = [robot_right.get_joint(f"panda_joint{i}").dofs_idx_local[0] for i in range(1, 8)]

    def set_gains(robot, dofs):
        robot.set_dofs_kp(np.full(len(dofs), 800.0), dofs_idx_local=dofs)
        robot.set_dofs_kv(np.full(len(dofs), 80.0), dofs_idx_local=dofs)

    set_gains(robot_left, left_dofs)
    set_gains(robot_right, right_dofs)

    robot_left.set_dofs_position(READY_POSE, dofs_idx_local=left_dofs)
    robot_right.set_dofs_position(READY_POSE, dofs_idx_local=right_dofs)

    n_steps = 150  # dt=0.02 -> 3秒
    fps = int(round(1.0 / DT))

    cam.start_recording(save_to_filename=VIDEO_PATH, fps=fps)
    for step in range(n_steps):
        t = step * DT
        swing = 0.4 * np.sin(2.0 * np.pi * 0.5 * t)

        left_target = list(READY_POSE)
        left_target[1] += swing  # 肩(joint2)を上下に揺らす
        right_target = list(READY_POSE)
        right_target[1] -= swing  # 逆位相で右腕を揺らす

        robot_left.control_dofs_position(left_target, dofs_idx_local=left_dofs)
        robot_right.control_dofs_position(right_target, dofs_idx_local=right_dofs)

        scene.step()
    cam.stop_recording()

    print(f"Saved video to {os.path.abspath(VIDEO_PATH)}")


if __name__ == "__main__":
    main()
