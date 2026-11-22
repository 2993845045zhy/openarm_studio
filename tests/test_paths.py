"""外部资源路径解析（README「安装」里的三个环境变量）。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openarm_studio import paths  # noqa: E402

ENV_NAMES = ("OPENARM_WS", "OPENARM_MESH_DIR", "OPENARM_MUJOCO_V1_XML")


def _clear_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_follow_recommended_layout(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(Path, "home", lambda: Path("/home/tester"))
    ws = paths.workspace_dir()
    assert ws == Path("/home/tester/openarm_ws")
    assert paths.mesh_dirs()[0] == ws / "src/openarm_description"
    assert paths.mujoco_v1_xml() == ws / "official_openarm_mujoco/v1/openarm_bimanual.xml"


def test_workspace_override_moves_both_defaults(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("OPENARM_WS", "/opt/arm")
    assert paths.mesh_dirs()[0] == Path("/opt/arm/src/openarm_description")
    assert paths.mujoco_v1_xml() == Path("/opt/arm/official_openarm_mujoco/v1/openarm_bimanual.xml")


def test_individual_overrides_win_and_expand_tilde(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("OPENARM_WS", "/opt/arm")
    monkeypatch.setenv("OPENARM_MESH_DIR", "~/desc")
    monkeypatch.setenv("OPENARM_MUJOCO_V1_XML", "~/mujoco/v1/openarm_bimanual.xml")
    assert paths.mesh_dirs()[0] == Path.home() / "desc"
    assert paths.mujoco_v1_xml() == Path.home() / "mujoco/v1/openarm_bimanual.xml"


def test_mesh_root_candidates_cover_both_description_layouts(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("OPENARM_MESH_DIR", "~/desc")
    assert paths.mesh_dirs() == [Path.home() / "desc", Path.home() / "desc/meshes",
                                 Path.home() / "desc/assets"]


def test_ik_and_gravity_read_the_same_model_path(monkeypatch):
    _clear_env(monkeypatch)
    from openarm_studio.backend import gravity, ik
    assert ik.V1_XML == paths.mujoco_v1_xml()
    assert gravity.GravityModel.MUJOCO_CANDIDATES == (paths.mujoco_v1_xml(),)


def test_mesh_dir_override_selects_layout_roots(monkeypatch, tmp_path):
    from openarm_studio.model import urdf2mjcf
    monkeypatch.setenv("OPENARM_MESH_DIR", str(tmp_path / "description"))
    # 目录还不存在，或存在但认不出网格布局：都算不可用，报错里给出环境变量名
    assert not urdf2mjcf.mesh_dir_available()
    (tmp_path / "description").mkdir()
    assert not urdf2mjcf.mesh_dir_available()
    with pytest.raises(FileNotFoundError, match="OPENARM_MESH_DIR"):
        urdf2mjcf._mesh_dir()  # noqa: SLF001
    # 旧版 meshes/ 布局：有 arm/v10 就能认出来
    (tmp_path / "description/meshes/arm/v10").mkdir(parents=True)
    assert urdf2mjcf.mesh_dir_available()
    assert urdf2mjcf._mesh_dir() == (tmp_path / "description/meshes").resolve()  # noqa: SLF001
