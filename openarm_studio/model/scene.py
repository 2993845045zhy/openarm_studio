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

# 外观配色不再人工指定：dae 里自带每种部件的材质颜色（银色金属件 /
# 3D 打印灰黑 / 电机黑），urdf2mjcf 转换时按颜色拆分并记录在
# ~/.cache/openarm_studio/meshes/colors.json，_apply_finishes 还原。


def _apply_finishes(xml: str) -> str:
    """按 dae 提取的原始配色给 visual geom 绑材质；collision 挪到第 3 组。

    颜色来自 urdf2mjcf 转换时写的 colors.json（分片名 → rgb）。
    亮度高的（银色金属件）给金属质感，深色（打印件/电机）给哑光。
    """
    import json
    import re as _re
    from pathlib import Path as _P

    meta_path = _P.home() / ".cache/openarm_studio/meshes/colors.json"
    colors = {}
    if meta_path.exists():
        colors = {k: tuple(v) for k, v in json.loads(meta_path.read_text()).items()}

    # mesh name → 分片名（MjSpec 导出的 mesh 名 = 文件 stem，colors.json
    # 的键同样是无后缀 stem）
    mesh_file = {}
    for m in _re.finditer(r'<mesh\s+name="([^"]+)"[^>]*file="([^"]+)"', xml):
        mesh_file[m.group(1)] = _P(m.group(2)).stem

    mat_names = {}       # rgb -> 材质名
    mats_xml = []
    seen_mats = set()

    def material_for(rgb: tuple) -> str:
        key = f"mat_c{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        if key not in mat_names:
            r, g, b = (v / 255.0 for v in rgb)
            mean = (r + g + b) / 3.0
            if mean > 0.45:
                # 银色机加件：金属高光
                spec, shin, metal, rough = 0.85, 0.6, 1.0, 0.35
            elif mean > 0.15:
                # 3D 打印塑料件（灰黑 63）：半哑光，留塑料的柔和高光
                spec, shin, metal, rough = 0.35, 0.45, 0.0, 0.55
            else:
                # 电机/深色件（24/0）：硬质光泽，比打印件亮一档
                spec, shin, metal, rough = 0.30, 0.5, 0.1, 0.45
            if key not in seen_mats:
                mats_xml.append(
                    f'<material name="{key}" rgba="{r:.3f} {g:.3f} {b:.3f} 1" '
                    f'specular="{spec}" shininess="{shin}" metallic="{metal}" '
                    f'roughness="{rough}"/>')
                seen_mats.add(key)
            mat_names[rgb] = key
        return mat_names[rgb]

    def repl(m: _re.Match[str]) -> str:
        tag = m.group(0)
        name = m.group(1)
        if name.endswith("_collision"):
            if 'group="' in tag:
                return tag
            return tag[:-2] + ' group="3"/>'
        mesh_m = _re.search(r'mesh="([^"]+)"', tag)
        if mesh_m and (name.endswith("_visual") or "_visual_c" in name):
            shard = mesh_file.get(mesh_m.group(1), "")
            rgb = colors.get(shard)
            if rgb is None:
                return tag
            mat = material_for(rgb)
            if 'material="' in tag:
                return tag
            return tag[:-2] + f' material="{mat}"/>'
        return tag

    xml = _re.sub(r'<geom name="([^"]+)"[^>]*/>', repl, xml)
    if mats_xml:
        xml = _inject_into(xml, "asset", "".join(mats_xml))
    return xml


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

    def mat_xml(name: str, p: dict) -> str:
        attrs = " ".join(f'{k}="{v}"' for k, v in p.items())
        return f'<material name="{name}" {attrs}/>'

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
        # 灯光照 pbvs 场景标准：暖调主光(key) + 冷调补光(fill) + 顶光(top)
        # + 工作区聚光(work_spot)，方向光为主、spot 补层次
        '<light name="key" directional="true" pos="0.5 -1.5 2.5" dir="-0.15 0.45 -0.87" '
        'diffuse="%s %s %s" specular="0.12 0.12 0.12" castshadow="false"/>'
        '<light name="fill" directional="true" pos="-1.5 0.5 2.0" dir="0.55 -0.2 -0.81" '
        'diffuse="%s %s %s" specular="0.03 0.03 0.03" castshadow="false"/>'
        '<light name="top" directional="true" pos="0 0 3" dir="0 0 -1" '
        'diffuse="%s %s %s" specular="0.02 0.02 0.02" castshadow="false"/>'
        '<light name="work_spot" pos="0.18 -0.32 2.0" dir="0 0 -1" cutoff="55" '
        'exponent="1.5" diffuse="%s %s %s" specular="0.05 0.05 0.05" castshadow="false"/>'
        '<geom name="floor" type="plane" size="4 4 0.05" pos="0 0 0" '
        'material="floor_mat" contype="0" conaffinity="0"/>'
        % (0.60 * s, 0.58 * s, 0.53 * s,
           0.20 * s, 0.22 * s, 0.27 * s,
           0.15 * s, 0.16 * s, 0.18 * s,
           0.45 * s, 0.45 * s, 0.45 * s)
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
    xml = _apply_finishes(xml)
    # 位置伺服 actuator：仿真模式下由指令驱动关节，而不是直接改 qpos
    xml = _inject_into(xml, "actuator", _actuators())
    # URDF 的 <mimic>（finger2 跟随 finger1）MuJoCo 导入时会丢弃，
    # 用 joint equality 补回来，否则第二根手指永远不动
    xml = _inject_into(xml, "equality", _equalities())
    # visual 块：有就合并，没有就插到 </mujoco> 前
    if "<visual>" in xml:
        xml = xml.replace("<visual>", "<visual>", 1)
        xml = _inject_into(xml, "visual", visual[len("<visual>"):-len("</visual>")])
    else:
        xml = xml.replace("</mujoco>", visual + "</mujoco>", 1)
    return xml


def _actuators() -> str:
    """16 个位置伺服（左右各 7 关节 + 夹爪）。

    kp 按"能顶住自重"标定：肩部大关节给到 200，肘/腕逐级降低；
    dampratio=1 让伺服临界阻尼，不会来回荡。
    夹爪力很小，避免仿真里"啪"地撞死。
    """
    arm_kp = [200.0, 200.0, 150.0, 120.0, 45.0, 35.0, 30.0]
    out = []
    for side in ("left", "right"):
        for i, kp in enumerate(arm_kp, start=1):
            out.append(f'<position name="{side}_joint{i}_act" '
                       f'joint="openarm_{side}_joint{i}" kp="{kp}" '
                       f'dampratio="1" ctrllimited="true" ctrlrange="-3.5 3.5"/>')
        out.append(f'<position name="{side}_grip_act" '
                   f'joint="openarm_{side}_finger_joint1" kp="60" '
                   f'dampratio="1" ctrllimited="true" ctrlrange="0 0.044"/>')
    return "".join(out)


def _equalities() -> str:
    """两对手指的 mimic：finger2 = finger1（URDF <mimic> 的等价物）。"""
    out = []
    for side in ("left", "right"):
        j1 = f"openarm_{side}_finger_joint1"
        j2 = f"openarm_{side}_finger_joint2"
        out.append(f'<joint joint1="{j1}" joint2="{j2}" polycoef="0 1 0 0 0"/>')
    return "".join(out)


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
    """把 关节名→角度 写进 qpos（自动跳过不存在的关节），返回写入数。

    finger_joint2 是 finger_joint1 的 mimic：equality 约束只在物理步进时
    求解，直接写 qpos 的孪生路径必须手动同步第二根手指。
    """
    qmap = joint_qpos_map(model)
    n = 0
    for name, val in positions.items():
        idx = qmap.get(name)
        if idx is None:
            continue
        data.qpos[idx] = float(val)
        n += 1
        if name.endswith("finger_joint1"):
            twin = qmap.get(name.replace("joint1", "joint2"))
            if twin is not None:
                data.qpos[twin] = float(val)
    return n


def home_pose(model: mujoco.MjModel) -> np.ndarray:
    """零位（双臂所有关节 0，夹爪闭合）。"""
    data = mujoco.MjData(model)
    return np.array(data.qpos, copy=True)
