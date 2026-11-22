"""把 OpenArm 双臂模型转换成 MuJoCo 能读取的 URDF。

MuJoCo 的 URDF 支持有三个硬性限制，这里逐条处理：

1. **不支持 COLLADA (.dae)** —— 视觉网格全部改成同名 `.stl`；若视觉目录
   没有对应 stl，退到 `collision/` 下同名 stl（手部 `hand.dae` 就是这种情况）。
2. **不认 `package://`** —— 换成绝对文件路径。
3. **惯量校验严格** —— 手指连杆的惯量主矩不满足三角不等式，需要
   `balanceinertia="true"`。

"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..paths import mesh_dirs

BIMANUAL_URDF = Path(__file__).with_name("assets") / "openarm_v10_bimanual.urdf"


def _layout(root: Path) -> str | None:
    """openarm_description 两代布局：meshes/（记作 legacy）和 assets/（记作 main）。"""
    if (root / "arm/v10").is_dir():
        return "legacy"
    if (root / "arm/openarm_v1.0").is_dir():
        return "main"
    return None


def _to_main_layout(fname: str) -> str:
    """随附 URDF 里的路径（按旧版 meshes/ 布局写的）→ main 分支 assets/ 下的路径。"""
    parts = fname.split("/")
    name = parts[-1]
    if parts[:2] == ["arm", "v10"]:
        if name.startswith("link7"):
            # 腕部末节归在末端执行器目录下，碰撞网格也不带 _symp 后缀
            name = "link7.stl" if name == "link7_symp.stl" else name
            return f"end_effector/parallel_link/meshes/{parts[2]}/{name}"
        return f"arm/openarm_v1.0/meshes/{parts[2]}/{name}"
    if parts[:2] == ["body", "v10"]:
        return f"body/v1.0/meshes/{parts[2]}/{name}"
    if parts[:2] == ["ee", "openarm_hand"]:
        return f"end_effector/parallel_link/meshes/{parts[2]}/{name}"
    return fname


def mesh_dir_available() -> bool:
    try:
        _mesh_dir()
    except FileNotFoundError:
        return False
    return True


def _mesh_dir() -> Path:
    """能认出布局的第一个网格根目录。"""
    for root in mesh_dirs():
        if _layout(root):
            return root.resolve()
    first = mesh_dirs()[0]
    raise FileNotFoundError(f"找不到 OpenArm v1.0 网格（在 {first} 及其 meshes/、assets/ 下都没找到）。"
                            "请按 README「安装」克隆 openarm_description，"
                            "或设置 OPENARM_MESH_DIR 指向它。")


def _mirror_path(src: Path) -> Path:
    """镜像副本的位置。dae 分片本来就在缓存里，原地加 _m 后缀；取自 openarm_description
    的碰撞网格镜像到缓存，不往别人的仓库里写生成文件。"""
    if src.parent == _mesh_cache():
        return src.with_name(src.stem + "_m.stl")
    key = hashlib.sha256(str(src).encode()).hexdigest()[:10]
    return _mesh_cache() / f"{src.stem}_m_{key}.stl"


def _mirror_shards(shards: list[Path]) -> list[Path]:
    """Y 镜像分片（URDF 用负 scale 复用左右臂/手指网格）。

    MuJoCo 的 URDF 导入器在**同一文件被正负 scale 混用**时，去重会把
    镜像 geom 错配成非镜像网格（实测右臂 right_finger 被绑到非镜像
    版本、整体外移 8.8cm，而左臂正常）。所以这里把镜像直接烘进几何、
    scale 写成全正，每个朝向各有唯一文件名，去重无从出错。
    """
    import numpy as np
    import trimesh

    outs = []
    for src in shards:
        dst = _mirror_path(src)
        if not dst.exists():
            g = trimesh.load(str(src), force="mesh")
            g.apply_transform(np.diag([1.0, -1.0, 1.0, 1.0]))
            g.export(str(dst))
        outs.append(dst)
    return outs


def _mesh_rewrite(text: str, mesh_dir: Path) -> str:
    """路径改写 + dae→stl + 负 scale 烘焙成镜像文件。"""
    layout = _layout(mesh_dir)

    def locate(fname: str) -> Path:
        rel = _to_main_layout(fname) if layout == "main" else fname
        return mesh_dir / rel

    def repl(m: re.Match[str]) -> str:
        fname, scale = m.group(1), m.group(2)
        negate = scale is not None and "-" in scale
        src = locate(fname)
        shards: list[Path] | None = None
        if src.suffix.lower() == ".dae":
            try:
                shards = _convert_dae(src)
            except Exception:  # noqa: BLE001
                shards = None
        if shards is None:
            stem = fname.rsplit(".", 1)[0]
            for cand in (locate(stem + ".stl"),
                         locate(stem.replace("/visual/", "/collision/") + ".stl")):
                if cand.exists():
                    shards = [cand]
                    break
        if shards is None:
            raise FileNotFoundError(f"缺少 OpenArm 网格：{fname}（网格根目录 {mesh_dir}）")
        if negate:
            shards = _mirror_shards(shards)
        pos_scale = scale.replace("-", "") if scale else scale
        out = f'filename="{shards[0]}"'
        if pos_scale:
            out += f' scale="{pos_scale}"'
        return out

    return re.sub(r'filename="([^"]+)"(?:\s+scale="([^"]+)")?', repl, text)


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
    对应颜色的材质，还原原模型配色。

    分片命名 {stem}_{idx}_{hash}.stl；同前缀即同源。
    """
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


def preprocess(raw: str, mesh_dir: Path) -> str:
    """注释剥离 → 网格路径改写 → 多色 visual 展开 → 注入 compiler 选项。"""
    text = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    text = _mesh_rewrite(text, mesh_dir)
    text = _expand_visual_shards(text)
    return re.sub(r"(<robot[^>]*>)", r"\1\n" + COMPILER, text, count=1)


def prepare_model_file(cache_dir: Path) -> Path:
    """将静态双臂模型和网格转换成 MuJoCo 使用的模型文件。"""
    out = cache_dir / "openarm_bimanual.mujoco.urdf"
    out.write_text(preprocess(BIMANUAL_URDF.read_text(), _mesh_dir()))
    return out
