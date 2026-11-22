"""Studio 依赖的外部资源放在哪。

渲染网格、重力前馈和 IK 用的官方模型不随本仓库分发，运行时从 openarm_description
和 openarm_mujoco 两个官方仓库读取。默认按 README「安装」里的目录布局，
三个位置都可用环境变量覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def workspace_dir() -> Path:
    """官方仓库的存放目录。"""
    return _env_path("OPENARM_WS", Path.home() / "openarm_ws")


def mesh_dirs() -> list[Path]:
    """依次尝试的网格根目录：OPENARM_MESH_DIR 指向的位置，以及它下面的 meshes/ 和 assets/。

    openarm_description 有两代目录布局，网格分别在仓库根目录的 meshes/（1.0.4 及更早）
    和 assets/（main 分支）下，所以三种写法都接受。
    """
    base = _env_path("OPENARM_MESH_DIR",
                     workspace_dir() / "src/openarm_description")
    roots: list[Path] = []
    for cand in (base, base / "meshes", base / "assets"):
        if cand not in roots:
            roots.append(cand)
    return roots


def mujoco_v1_xml() -> Path:
    """openarm_mujoco 仓库的 v1 双臂 MJCF：重力前馈与 IK 的模型。"""
    return _env_path("OPENARM_MUJOCO_V1_XML",
                     workspace_dir() / "official_openarm_mujoco/v1/openarm_bimanual.xml")
