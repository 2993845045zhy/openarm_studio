"""把 openarm_description 的 xacro 转成 MuJoCo 能吃的 URDF 文本。

MuJoCo 的 URDF 支持有三个硬性限制，这里逐条处理：

1. **不支持 COLLADA (.dae)** —— 视觉网格全部改成同名 `.stl`；若视觉目录
   没有对应 stl，退到 `collision/` 下同名 stl（手部 `hand.dae` 就是这种情况）。
2. **不认 `package://`** —— 换成绝对文件路径。
3. **惯量校验严格** —— 手指连杆的惯量主矩不满足三角不等式，需要
   `balanceinertia="true"`。

另外 xacro 里被注释掉的 `<joint name="${prefix}joint8">` 会留下未展开的
`${...}` 占位符，必须先剥掉 XML 注释，否则干扰解析。
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

DESC_PACKAGE = "openarm_description"


def _ws_prefixes() -> list[str]:
    """colcon 默认是隔离安装：每个包在 install/<pkg> 下自成前缀。"""
    out: list[str] = []
    install = Path.home() / "openarm_ws/install"
    if install.is_dir():
        for child in install.iterdir():
            if (child / "share/ament_index/resource_index/packages").is_dir():
                out.append(str(child))
    return out


def _ros_env() -> dict:
    """构造能跑 xacro 的环境（系统 python + ROS 2 Jazzy）。

    已经 source 过环境时（run_studio.sh 会做）不动 AMENT/CMAKE 路径，
    只把 anaconda 的影响隔离掉；否则自己拼一套可用的前缀链。
    """
    env = dict(os.environ)
    base = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    env["PATH"] = base + ":/opt/ros/jazzy/bin:/opt/ros/jazzy/opt/gz_tools_vendor/bin"
    for key in ("CONDA_PREFIX", "PYTHONHOME"):
        env.pop(key, None)
    extra = [
        "/opt/ros/jazzy/lib/python3.12/site-packages",
        "/opt/ros/jazzy/local/lib/python3.12/dist-packages",
    ]
    env["PYTHONPATH"] = ":".join(extra + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    prefixes = [p for p in (env.get("AMENT_PREFIX_PATH") or "").split(":") if p]
    for p in _ws_prefixes() + ["/opt/ros/jazzy"]:
        if p not in prefixes:
            prefixes.append(p)
    env["AMENT_PREFIX_PATH"] = ":".join(prefixes)
    if not env.get("CMAKE_PREFIX_PATH"):
        env["CMAKE_PREFIX_PATH"] = ":".join(prefixes)
    return env


def find_description_root() -> Path | None:
    """定位 openarm_description 包根目录（源码树或 install 目录都行）。"""
    candidates = [
        Path.home() / "openarm_ws/src/openarm_description",
        Path.home() / "openarm_ws/install/openarm_description/share/openarm_description",
    ]
    for c in candidates:
        if (c / "urdf").is_dir():
            return c
    # 退一步：按 ament 索引猜
    for c in (Path("/opt/ros/jazzy/share/" + DESC_PACKAGE),):
        if (c / "urdf").is_dir():
            return c
    return None


def generate_urdf(out_path: Path, *, bimanual: bool = True,
                  arm_type: str = "v10", force: bool = False) -> Path:
    """跑 xacro 生成 URDF 并缓存到 out_path。"""
    if out_path.exists() and not force:
        return out_path
    root = find_description_root()
    if root is None:
        raise FileNotFoundError(
            "找不到 openarm_description 包；请在 ~/openarm_ws 下先 colcon build")
    xacro_file = root / "urdf/robot/v10.urdf.xacro"
    args = [f"bimanual:={'true' if bimanual else 'false'}", f"arm_type:={arm_type}"]
    env = _ros_env()
    proc = subprocess.run(
        ["/opt/ros/jazzy/bin/xacro", str(xacro_file), *args],
        capture_output=True, text=True, env=env, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"xacro 失败: {proc.stderr[:800]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(proc.stdout)
    return out_path


def _mesh_rewrite(text: str, desc_root: Path) -> str:
    text = text.replace(f"package://{DESC_PACKAGE}/", f"{desc_root}/")

    def repl(m: re.Match[str]) -> str:
        src = Path(m.group(1))
        if src.suffix.lower() != ".dae":
            return m.group(0)
        for cand in (src.with_suffix(".stl"),
                     Path(str(src).replace("/visual/", "/collision/")).with_suffix(".stl")):
            if cand.exists():
                return f'filename="{cand}"'
        return m.group(0)

    return re.sub(r'filename="([^"]*)"', repl, text)


COMPILER = (
    '<mujoco><compiler balanceinertia="true" fusestatic="false" '
    'strippath="false" angle="radian" discardvisual="false"/></mujoco>'
)


def preprocess(raw: str, desc_root: Path) -> str:
    """注释剥离 → 网格路径改写 → 注入 compiler 选项。"""
    text = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    text = _mesh_rewrite(text, desc_root)
    return re.sub(r"(<robot[^>]*>)", r"\1\n" + COMPILER, text, count=1)


def prepare_model_file(cache_dir: Path, *, bimanual: bool = True,
                       force: bool = False) -> Path:
    """生成的中间 URDF（已预处理），供 MjSpec 载入。"""
    stem = "openarm_bimanual" if bimanual else "openarm_single"
    src = generate_urdf(cache_dir / f"{stem}.raw.urdf", bimanual=bimanual, force=force)
    root = find_description_root()
    assert root is not None
    out = cache_dir / f"{stem}.mujoco.urdf"
    out.write_text(preprocess(src.read_text(), root))
    return out
