"""手首に縁付きのお盆(トレイ)を取り付けたFranka PandaのURDFを実行時に生成する。

Genesis同梱の panda.urdf はメッシュを `package://meshes/...` と相対指定しているため、
コピーを置かずに、元のURDFの文字列を書き換えて一時ディレクトリに出力する。
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile

import genesis as gs

# 基準姿勢(env.READY_POSE)でお盆が水平になるように求めた、手首リンク(panda_link7)座標系での
# お盆の取り付け位置と姿勢。基準姿勢でのワールド座標では、link7原点から(+0.16, 0, -0.02)の位置。
TRAY_XYZ = (0.1106, -0.1117, 0.0359)
TRAY_RPY = (3.1416, -0.1, -0.79)

# すり鉢型のお盆: 平らな底(一辺TRAY_BASE)と、外側へ向かって立ち上がる4枚の斜面からなる。
# 玉が少しずれて着地しても、斜面を転がって中央に集まり、縁から飛び出しにくい。
TRAY_BASE = 0.07  # 底の一辺[m](玉の直径6cmより少し広いだけ。玉が中央に集まる)
TRAY_RIM = 0.19  # 上端の一辺[m]
TRAY_HEIGHT = 0.05  # 底から上端までの高さ[m]
TRAY_THICK = 0.01


def _box(size, xyz, rpy, rgba_name):
    sx, sy, sz = size
    x, y, z = xyz
    r, p, yw = rpy
    geom = f'<geometry><box size="{sx} {sy} {sz}"/></geometry>'
    origin = f'<origin xyz="{x} {y} {z}" rpy="{r} {p} {yw}"/>'
    visual = f"<visual>{origin}{geom}<material name=\"{rgba_name}\"/></visual>"
    collision = f"<collision>{origin}{geom}</collision>"
    return visual + collision


def _tray_xml() -> str:
    half_base = TRAY_BASE / 2
    half_rim = TRAY_RIM / 2
    run = half_rim - half_base  # 斜面の水平方向の広がり
    length = math.hypot(run, TRAY_HEIGHT)  # 斜面の長さ
    tilt = math.atan2(TRAY_HEIGHT, run)  # 水平からの傾き
    mid = (half_base + half_rim) / 2  # 斜面の中心の、中心軸からの距離
    z = TRAY_THICK / 2 + TRAY_HEIGHT / 2
    parts = [
        _box((TRAY_BASE, TRAY_BASE, TRAY_THICK), (0, 0, 0), (0, 0, 0), "tray_mat"),
        # +x / -x の斜面(y方向に長い)
        _box((length, TRAY_RIM, TRAY_THICK), (mid, 0, z), (0, -tilt, 0), "tray_mat"),
        _box((length, TRAY_RIM, TRAY_THICK), (-mid, 0, z), (0, tilt, 0), "tray_mat"),
        # +y / -y の斜面(x方向に長い)
        _box((TRAY_RIM, length, TRAY_THICK), (0, mid, z), (tilt, 0, 0), "tray_mat"),
        _box((TRAY_RIM, length, TRAY_THICK), (0, -mid, z), (-tilt, 0, 0), "tray_mat"),
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
