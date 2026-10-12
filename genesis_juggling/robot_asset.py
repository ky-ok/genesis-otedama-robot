"""手首に縁付きのお盆(トレイ)を取り付けたFranka PandaのURDFを実行時に生成する。

Genesis同梱の panda.urdf はメッシュを `package://meshes/...` と相対指定しているため、
コピーを置かずに、元のURDFの文字列を書き換えて一時ディレクトリに出力する。
"""

from __future__ import annotations

import hashlib
import os
import tempfile

import genesis as gs

# 基準姿勢(env.READY_POSE)でお盆が水平になるように求めた、手首リンク(panda_link7)座標系での
# お盆の取り付け位置と姿勢。基準姿勢でのワールド座標では、link7原点から(+0.16, 0, -0.02)の位置。
TRAY_XYZ = (0.1106, -0.1117, 0.0359)
TRAY_RPY = (3.1416, -0.1, -0.79)

TRAY_SIZE = 0.16  # 一辺[m]
TRAY_WALL_H = 0.03  # 縁の高さ[m]
TRAY_THICK = 0.01
TRAY_WALL_T = 0.01


def _box(size, xyz, rgba_name):
    sx, sy, sz = size
    x, y, z = xyz
    geom = f'<geometry><box size="{sx} {sy} {sz}"/></geometry>'
    visual = f'<visual><origin xyz="{x} {y} {z}" rpy="0 0 0"/>{geom}<material name="{rgba_name}"/></visual>'
    collision = f'<collision><origin xyz="{x} {y} {z}" rpy="0 0 0"/>{geom}</collision>'
    return visual + collision


def _tray_xml() -> str:
    half = TRAY_SIZE / 2 - TRAY_WALL_T / 2
    wall_z = TRAY_THICK / 2 + TRAY_WALL_H / 2
    parts = [
        _box((TRAY_SIZE, TRAY_SIZE, TRAY_THICK), (0, 0, 0), "tray_mat"),
        _box((TRAY_WALL_T, TRAY_SIZE, TRAY_WALL_H), (half, 0, wall_z), "tray_mat"),
        _box((TRAY_WALL_T, TRAY_SIZE, TRAY_WALL_H), (-half, 0, wall_z), "tray_mat"),
        _box((TRAY_SIZE, TRAY_WALL_T, TRAY_WALL_H), (0, half, wall_z), "tray_mat"),
        _box((TRAY_SIZE, TRAY_WALL_T, TRAY_WALL_H), (0, -half, wall_z), "tray_mat"),
    ]
    xyz = " ".join(str(v) for v in TRAY_XYZ)
    rpy = " ".join(str(v) for v in TRAY_RPY)
    return (
        '<material name="tray_mat"><color rgba="0.85 0.65 0.35 1"/></material>'
        '<link name="tray">'
        '<inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.2"/>'
        '<inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>'
        + "".join(parts)
        + "</link>"
        f'<joint name="tray_joint" type="fixed"><origin xyz="{xyz}" rpy="{rpy}"/>'
        '<parent link="panda_link7"/><child link="tray"/></joint>'
    )


def panda_with_tray_urdf() -> str:
    """お盆付きPandaのURDFを(必要なら)生成し、そのファイルの絶対パスを返す。"""
    src_dir = os.path.join(gs.utils.get_assets_dir(), "urdf", "panda_bullet")
    with open(os.path.join(src_dir, "panda.urdf")) as f:
        text = f.read()

    text = text.replace("package://meshes/", f"package://{src_dir}/meshes/")
    assert "</robot>" in text
    text = text.replace("</robot>", _tray_xml() + "</robot>")

    digest = hashlib.md5(text.encode()).hexdigest()[:10]
    out_dir = os.path.join(tempfile.gettempdir(), "genesis_juggling")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"panda_tray_{digest}.urdf")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(text)
    return path
