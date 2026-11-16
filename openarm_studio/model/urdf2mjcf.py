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
    # hand:=true 必须显式传：缺了它手部装配段整段不展开，
    # hand/finger link 没有装配关节，会被 MuJoCo 焊死在世界原点
    args = [
        f"bimanual:={'true' if bimanual else 'false'}",
        f"arm_type:={arm_type}",
        "hand:=true",
    ]
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
    """只做路径改写与 dae→stl 转换，**scale 原样保留**。

    MuJoCo 的 URDF 导入器本来就支持负 scale（官方 openarm_mujoco 的
    mesh 声明里就是 scale="1 -1 1"）。早前以为它丢负号、生成镜像文件
    去补，结果和 MuJoCo 自己的负 scale 叠加成双重镜像，左右手指又回到
    同一侧——必须交回给 MuJoCo 处理。
    """
    text = text.replace(f"package://{DESC_PACKAGE}/", f"{desc_root}/")

    def repl(m: re.Match[str]) -> str:
        fname = m.group(1)
        src = Path(fname)
        out: Path | None = None
        if src.suffix.lower() == ".dae":
            try:
                out = _convert_dae(src)[0]
            except Exception:  # noqa: BLE001
                out = None
        if out is None:
            for cand in (src.with_suffix(".stl"),
                         Path(str(src).replace("/visual/", "/collision/")).with_suffix(".stl")):
                if cand.exists():
                    out = cand
                    break
        if out is None:
            return m.group(0)
        return m.group(0).replace(fname, str(out))

    return re.sub(r'filename="([^"]+)"', repl, text)


COMPILER = (
    '<mujoco><compiler balanceinertia="true" fusestatic="false" '
    'strippath="false" angle="radian" discardvisual="false"/></mujoco>'
)

_MESH_CACHE: Path | None = None


def _mesh_cache() -> Path:
    global _MESH_CACHE
    if _MESH_CACHE is None:
        _MESH_CACHE = Path.home() / ".cache/openarm_studio/meshes"
        _MESH_CACHE.mkdir(parents=True, exist_ok=True)
    return _MESH_CACHE


def _convert_dae(src: Path) -> list[Path]:
    """DAE → 按材质颜色拆分的多片 STL（返回清单，首片代表整体）。

    MuJoCo 不支持 COLLADA，而 STL 不带颜色。dae 里每种部件颜色
    （银色金属件/3D 打印灰黑/电机黑）各是一组 geometry——这里按颜色
    分组各自导出 STL，scene.py 再把它们复制成多个 geom 并绑上
    对应颜色的材质，还原 RViz 里的着色。

    分片命名 {stem}_{idx}_{hash}.stl；同前缀即同源。
    """
    import hashlib

    key = hashlib.sha256(str(src).encode()).hexdigest()[:10]
    prefix = f"{src.stem}"
    out_dir = _mesh_cache()
    marker = out_dir / f"{prefix}_0_{key}.stl"
    if marker.exists() and marker.stat().st_size > 0 and list(out_dir.glob(f"{prefix}_*_{key}.stl")):
        return sorted(out_dir.glob(f"{prefix}_*_{key}.stl"))

    import numpy as np
    import trimesh

    scene = trimesh.load(str(src), force="scene")
    # dae 的每个子网格带自己的装配变换（在 scene graph 里），
    # 必须先烘焙再合并——直接取 scene.geometry 拿到的是局部坐标，
    # 分片会各自错位（手指/手掌散架的根源）
    groups: dict[tuple, list] = {}
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name].copy()
        geom.apply_transform(transform)
        color = None
        mat = getattr(geom.visual, "material", None)
        try:
            if hasattr(mat, "main_color"):
                color = tuple(int(c) for c in mat.main_color[:3])
            elif hasattr(mat, "baseColor"):
                color = tuple(int(c) for c in mat.baseColor[:3])
        except Exception:  # noqa: BLE001
            color = None
        if color is None:
            color = (160, 160, 160)
        groups.setdefault(color, []).append(geom)

    outs = []
    meta_path = _mesh_cache() / "colors.json"
    import json
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for idx, (color, geoms) in enumerate(sorted(groups.items(), key=lambda kv: -sum(len(g.faces) for g in kv[1]))):
        verts = np.vstack([np.asarray(g.vertices, dtype=np.float64) for g in geoms])
        faces = []
        offset = 0
        for g in geoms:
            f = np.asarray(g.faces, dtype=np.int64) + offset
            faces.append(f)
            offset += len(g.vertices)
        mesh = trimesh.Trimesh(vertices=verts, faces=np.vstack(faces), process=False)
        out = out_dir / f"{prefix}_{idx}_{key}.stl"
        mesh.export(str(out))
        outs.append(out)
        meta[f"{prefix}_{idx}_{key}"] = list(color)
    meta_path.write_text(json.dumps(meta))
    return outs


def dae_color_map() -> dict[str, tuple]:
    """分片文件名前缀（含 hash）→ 各分片的颜色（从 STL 头? 不存——重建映射）。

    颜色信息在拆分时已知，但 STL 不带色。为避免二次解析，转换时把
    prefix+idx → rgb 写进 sidecar JSON。
    """
    import json

    meta = _mesh_cache() / "colors.json"
    if meta.exists():
        return {k: tuple(v) for k, v in json.loads(meta.read_text()).items()}
    return {}


_SHARD_RE = re.compile(r"^(?P<stem>.+)_(?P<idx>\d+)_(?P<hash>[0-9a-f]{10})(?P<mirror>_m)?\.stl$")


def _expand_visual_shards(text: str) -> str:
    """把指向分片 0 的 <visual> 复制成 N 份（每色一片）。

    URDF 允许一个 link 多个 visual；MjSpec 会为每个 visual 生成一个
    geom。分片 idx 与 colors.json 的颜色一一对应。镜像分片（_m 后缀）
    一起展开，镜像版的颜色沿用同 hash 正版的记录。
    """
    cache = _mesh_cache()
    import re as _re

    def repl(m: _re.Match[str]) -> str:
        block = m.group(0)
        fname = _re.search(r'filename="([^"]+)"', block)
        if not fname:
            return block
        shard = Path(fname.group(1)).name
        hit = _SHARD_RE.match(shard)
        if not hit or hit.group("idx") != "0":
            return block
        prefix, key, mirror = hit.group("stem"), hit.group("hash"), hit.group("mirror") or ""
        # 只展开**同一镜像态**的分片：_m 与非 _m 是同一部件的两种朝向，
        # 混在一起展开会让一根手指拿到 4 份几何（正常 2 色 + 镜像 2 色），
        # 鬼影伸出掌外
        siblings = sorted(cache.glob(f"{prefix}_*_{key}{mirror}.stl"))
        if len(siblings) <= 1:
            return block
        extra = []
        for sib in siblings[1:]:
            sh = _SHARD_RE.match(sib.name)
            idx, mir = sh.group("idx"), ("m" if sh.group("mirror") else "")
            new_block = block.replace(fname.group(1), str(sib))
            name_m = _re.search(r'name="([^"]+)"', new_block)
            if name_m:
                # 后缀带 hash 短版保证全局唯一（左右臂/正镜像复用同 dae）
                new_block = new_block.replace(
                    f'name="{name_m.group(1)}"',
                    f'name="{name_m.group(1)}_c{key[:4]}{idx}{mir}"')
            extra.append(new_block)
        return block + "".join(extra)

    return _re.sub(r"<visual\b[^>]*>.*?</visual>", repl, text, flags=_re.S)


def preprocess(raw: str, desc_root: Path) -> str:
    """注释剥离 → 网格路径改写 → 多色 visual 展开 → 注入 compiler 选项。"""
    text = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    text = _mesh_rewrite(text, desc_root)
    text = _expand_visual_shards(text)
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
