"""用 MjSpec 把 URDF 装配成带场景元素的 MuJoCo 模型。

URDF 的 `<mujoco>` 块里只能放 compiler 选项（实测 worldbody/light/asset
会被静默忽略），所以地板、灯光、天空盒一律在 MjSpec 上程序化添加。
天空盒用纯色贴图：与主卡底色同色，3D 区才不会露出一个异色矩形。
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from mujoco import MjSpec

from .urdf2mjcf import prepare_model_file

# ── 关节契约（与 ros2_control / MoveIt 完全一致）────────────────
LEFT_ARM = [f"openarm_left_joint{i}" for i in range(1, 8)]
RIGHT_ARM = [f"openarm_right_joint{i}" for i in range(1, 8)]
LEFT_GRIP = ["openarm_left_finger_joint1"]
RIGHT_GRIP = ["openarm_right_finger_joint1"]
ALL_ACTUATED = LEFT_ARM + LEFT_GRIP + RIGHT_ARM + RIGHT_GRIP

GRIP_OPEN = 0.044   # m，完全张开
GRIP_CLOSED = 0.0


def _hex_to_rgb(h: str) -> list[float]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def _write_solid_png(path: Path, rgb: list[float], size: int = 32) -> Path:
    """写一张纯色 PNG 当天空盒（MuJoCo 只认图片文件，不认真空颜色）。"""
    from PySide6.QtGui import QColor, QImage

    color = QColor(*[max(0, min(255, int(c * 255))) for c in rgb])
    img = QImage(size, size, QImage.Format_RGB888)
    img.fill(color)
    if not img.save(str(path), "PNG"):
        raise RuntimeError(f"天空盒贴图写入失败: {path}")
    return path


def _write_grid_png(path: Path, base_rgb: list[float], line_factor: float,
                    size: int = 512, cells: int = 8) -> Path:
    """地板网格贴图：底色 + 细网格线（line_factor<1 更暗，>1 更亮）。"""
    from PySide6.QtCore import QLineF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPen

    def c(v: float) -> int:
        return max(0, min(255, int(v * 255)))

    base = [c(v) for v in base_rgb]
    # 网格线必须与底色拉开足够差别，否则缩到 3D 里完全看不见
    line = []
    for v in base:
        target = int(v * line_factor)
        if abs(target - v) < 14:
            target = v + 14 if line_factor >= 1 else v - 14
        line.append(max(0, min(255, target)))

    img = QImage(size, size, QImage.Format_RGB888)
    img.fill(QColor(*base))
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing, False)
    pen = QPen(QColor(*line))
    pen.setWidth(2)
    pen.setCosmetic(True)
    painter.setPen(pen)
    step = size / cells
    for i in range(cells + 1):
        p = i * step
        painter.drawLine(QLineF(p, 0, p, size))
        painter.drawLine(QLineF(0, p, size, p))
    painter.end()
    if not img.save(str(path), "PNG"):
        raise RuntimeError(f"地板网格贴图写入失败: {path}")
    return path


def build_spec(urdf_path: Path, *, sky_rgb: list[float], floor_rgb: list[float],
               light_scale: float, haze: float,
               sky_png: Path, grid_png: Path, grid_line: float,
               floor_reflect: float) -> str:
    """返回最终 MJCF 文本。

    先让 MjSpec 把 URDF 读成 MJCF（这一步它做得好），再把地板/灯光/
    天空盒以文本形式注入——MjSpec.add_texture 实测不会写入 texture 类型，
    而手写标签这条路径完全可控。
    """
    spec = MjSpec.from_file(str(urdf_path))
    xml = spec.to_xml()

    s = max(0.4, float(light_scale))
    sky = " ".join(f"{v:.4f}" for v in sky_rgb)
    assets = (
        '<texture name="grid" type="2d" file="%s"/>'
        # 内置 gradient 天空盒：两端同色＝纯色天空，且不依赖外部图片尺寸
        '<texture name="sky" type="skybox" builtin="gradient" rgb1="%s" '
        'rgb2="%s" width="256" height="256"/>'
        '<material name="floor_mat" texture="grid" texrepeat="16 16" '
        'rgba="1 1 1 1" reflectance="%g" specular="0.02" shininess="0.05"/>'
        % (grid_png, sky, sky, floor_reflect)
    )
    scene = (
        '<light name="key" pos="1.1 -1.3 1.9" dir="-0.45 0.55 -1" '
        'diffuse="%g %g %g" castshadow="false"/>'
        '<light name="fill" pos="-1.5 0.9 1.1" dir="0.6 -0.4 -1" '
        'diffuse="%g %g %g" castshadow="false"/>'
        '<light name="rim" pos="-0.2 1.8 1.4" dir="0 -0.9 -0.6" '
        'diffuse="%g %g %g" castshadow="false"/>'
        '<geom name="floor" type="plane" size="4 4 0.05" pos="0 0 0" '
        'material="floor_mat" contype="0" conaffinity="0"/>'
        % (0.42 * s, 0.42 * s, 0.42 * s,
           0.20 * s, 0.21 * s, 0.23 * s,
           0.14 * s, 0.14 * s, 0.15 * s)
    )
    visual = (
        '<visual><headlight ambient="0.30 0.30 0.31" diffuse="0.30 0.30 0.31" '
        'specular="0 0 0"/>'
        '<global offwidth="1600" offheight="1200" ipd="0"/>'
        '<map haze="%g" fogstart="1.2" fogend="6"/>'
        '</visual>' % haze
    )

    xml = _inject_into(xml, "asset", assets)
    xml = _inject_into(xml, "worldbody", scene)
    # visual 块：有就合并，没有就插到 </mujoco> 前
    if "<visual>" in xml:
        xml = xml.replace("<visual>", "<visual>", 1)
        xml = _inject_into(xml, "visual", visual[len("<visual>"):-len("</visual>")])
    else:
        xml = xml.replace("</mujoco>", visual + "</mujoco>", 1)
    return xml


def _inject_into(xml: str, tag: str, payload: str) -> str:
    """把 payload 插到 <tag> 的末尾（标签不存在则新建在 </mujoco> 前）。"""
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    if open_tag in xml:
        idx = xml.index(close_tag, xml.index(open_tag))
        return xml[:idx] + payload + xml[idx:]
    return xml.replace("</mujoco>", f"{open_tag}{payload}{close_tag}</mujoco>", 1)


def build_model(cache_dir: Path, theme_scene: dict, *,
                bimanual: bool = True, force_rebuild: bool = False) -> mujoco.MjModel:
    """按主题配色构建模型（主题切换时重建，耗时 ~0.2s）。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    urdf = prepare_model_file(cache_dir, bimanual=bimanual, force=force_rebuild)

    sky_rgb = [float(v) for v in theme_scene["sky_top"].split()]
    floor_rgb = [float(v) for v in theme_scene["sky_floor"].split()] \
        if "sky_floor" in theme_scene else [float(v) for v in theme_scene["sky_horizon"].split()]

    sky_png = cache_dir / f"sky_{theme_scene['name']}.png"
    grid_png = cache_dir / f"grid_{theme_scene['name']}.png"
    _write_solid_png(sky_png, sky_rgb)
    _write_grid_png(grid_png, floor_rgb, float(theme_scene.get("grid_line", 0.9)))

    xml = build_spec(
        urdf, sky_rgb=sky_rgb, floor_rgb=floor_rgb,
        light_scale=theme_scene.get("light_scale", 1.0),
        haze=theme_scene.get("haze", 0.35),
        sky_png=sky_png,
        grid_png=grid_png,
        grid_line=float(theme_scene.get("grid_line", 0.9)),
        floor_reflect=float(theme_scene.get("floor_reflect", 0.1)),
    )
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.global_.ipd = 0.0
    return model


# ── 关节映射 ────────────────────────────────────────────────
def joint_qpos_map(model: mujoco.MjModel) -> dict[str, int]:
    """关节名 → qpos 下标。"""
    out: dict[str, int] = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name:
            out[name] = int(model.jnt_qposadr[j])
    return out


def apply_joint_positions(model: mujoco.MjModel, data: mujoco.MjData,
                          positions: dict[str, float]) -> int:
    """把 关节名→角度 写进 qpos（自动跳过不存在的关节），返回写入数。"""
    qmap = joint_qpos_map(model)
    n = 0
    for name, val in positions.items():
        idx = qmap.get(name)
        if idx is None:
            continue
        data.qpos[idx] = float(val)
        n += 1
    return n


def home_pose(model: mujoco.MjModel) -> np.ndarray:
    """零位（双臂所有关节 0，夹爪闭合）。"""
    data = mujoco.MjData(model)
    return np.array(data.qpos, copy=True)
