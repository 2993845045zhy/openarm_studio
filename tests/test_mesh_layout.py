"""网格路径解析：兼容 openarm_description 的两代目录布局。

随附 URDF 按旧版 meshes/ 布局写路径；上游 main 分支把网格挪到了 assets/ 下并改了目录名。
这里用占位文件搭出两种目录树，验证每一个被引用的网格都能定位到真实文件。
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openarm_studio.model import urdf2mjcf as u  # noqa: E402

ARM = [f"arm/v10/{kind}/link{i}{'_symp' if kind == 'collision' else ''}.{'stl' if kind == 'collision' else 'dae'}"
       for kind in ("collision", "visual") for i in range(8)]
BODY = ["body/v10/collision/body_link0_symp.stl", "body/v10/visual/body_link0.dae"]
HAND = ["ee/openarm_hand/collision/finger.stl", "ee/openarm_hand/collision/hand.stl",
        "ee/openarm_hand/visual/finger.dae", "ee/openarm_hand/visual/hand.dae"]
URDF_MESHES = ARM + BODY + HAND

MAIN_PATHS = {
    "arm/v10/visual/link0.dae": "arm/openarm_v1.0/meshes/visual/link0.dae",
    "arm/v10/visual/link6.dae": "arm/openarm_v1.0/meshes/visual/link6.dae",
    "arm/v10/collision/link3_symp.stl": "arm/openarm_v1.0/meshes/collision/link3_symp.stl",
    "arm/v10/visual/link7.dae": "end_effector/parallel_link/meshes/visual/link7.dae",
    "arm/v10/collision/link7_symp.stl": "end_effector/parallel_link/meshes/collision/link7.stl",
    "body/v10/visual/body_link0.dae": "body/v1.0/meshes/visual/body_link0.dae",
    "body/v10/collision/body_link0_symp.stl": "body/v1.0/meshes/collision/body_link0_symp.stl",
    "ee/openarm_hand/visual/hand.dae": "end_effector/parallel_link/meshes/visual/hand.dae",
    "ee/openarm_hand/collision/finger.stl": "end_effector/parallel_link/meshes/collision/finger.stl",
}


@pytest.fixture(scope="module")
def tiny_stl():
    import trimesh
    return trimesh.creation.box().export(file_type="stl")


def _fake_tree(root: Path, layout: str, stl: bytes) -> None:
    """按指定布局摆出占位网格：.stl 是真实的小方块，.dae 是坏文件（转换失败后退到同名 .stl）。"""
    for fname in URDF_MESHES:
        rel = u._to_main_layout(fname) if layout == "main" else fname  # noqa: SLF001
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".stl":
            target.write_bytes(stl)
        else:
            target.write_text("not a collada file")
            # 旧版布局里手部没有 visual/*.stl，必须退到 collision/；上游 main 分支两处都有
            if layout == "main" or "openarm_hand" not in rel:
                target.with_suffix(".stl").write_bytes(stl)


@pytest.mark.parametrize("fname,expected", sorted(MAIN_PATHS.items()))
def test_urdf_paths_map_onto_the_upstream_assets_layout(fname, expected):
    assert u._to_main_layout(fname) == expected  # noqa: SLF001


def test_every_mesh_used_by_the_bundled_urdf_has_a_mapping_rule():
    used = sorted(set(re.findall(r'filename="([^"]+)"', u.BIMANUAL_URDF.read_text())))
    assert used == sorted(set(URDF_MESHES))
    for fname in used:
        assert u._to_main_layout(fname) != fname  # noqa: SLF001


def _use_cache(tmp_path, monkeypatch) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(u, "_MESH_CACHE", cache)
    return cache


@pytest.mark.parametrize("layout", ["legacy", "main"])
def test_preprocess_resolves_every_mesh_in_both_layouts(tmp_path, monkeypatch, tiny_stl, layout):
    cache = _use_cache(tmp_path, monkeypatch)
    root = tmp_path / layout / ("meshes" if layout == "legacy" else "assets")
    _fake_tree(root, layout, tiny_stl)
    refs = re.findall(r'filename="([^"]+)"', u.preprocess(u.BIMANUAL_URDF.read_text(), root))
    assert len(refs) > len(URDF_MESHES), "左右臂各引用一遍，还应包含按颜色展开的分片"
    for ref in refs:
        path = Path(ref)
        assert path.is_file(), ref
        assert path.is_relative_to(root) or path.is_relative_to(cache), ref


def test_hand_without_visual_stl_falls_back_to_its_collision_mesh(tmp_path, monkeypatch, tiny_stl):
    """旧版布局的手掌只有 hand.dae：转换不了时用 ee/.../collision/hand.stl。"""
    root = tmp_path / "meshes"
    (root / "ee/openarm_hand/collision").mkdir(parents=True)
    (root / "ee/openarm_hand/collision/hand.stl").write_bytes(tiny_stl)
    (root / "ee/openarm_hand/visual").mkdir(parents=True)
    (root / "ee/openarm_hand/visual/hand.dae").write_text("not a collada file")
    text = u._mesh_rewrite('<mesh filename="ee/openarm_hand/visual/hand.dae" />', root)  # noqa: SLF001
    assert 'collision/hand.stl"' in text


def test_missing_mesh_reports_the_urdf_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="arm/v10/visual/link4.dae"):
        u._mesh_rewrite('<mesh filename="arm/v10/visual/link4.dae" />', tmp_path / "meshes")  # noqa: SLF001


def test_env_accepts_the_checkout_root_and_either_subdir(tmp_path, monkeypatch, tiny_stl):
    checkout = tmp_path / "openarm_description"
    monkeypatch.setenv("OPENARM_MESH_DIR", str(checkout))
    assert not u.mesh_dir_available()
    with pytest.raises(FileNotFoundError, match="OPENARM_MESH_DIR"):
        u._mesh_dir()  # noqa: SLF001
    _fake_tree(checkout / "assets", "main", tiny_stl)
    assert u.mesh_dir_available()
    assert u._mesh_dir() == (checkout / "assets").resolve()  # noqa: SLF001


def test_mirror_copy_goes_to_the_cache_not_into_the_mesh_repo(tmp_path, monkeypatch, tiny_stl):
    _use_cache(tmp_path, monkeypatch)
    src = tmp_path / "openarm_description" / "link3_symp.stl"
    src.parent.mkdir()
    src.write_bytes(tiny_stl)
    out = u._mirror_shards([src])[0]  # noqa: SLF001
    assert out.parent == tmp_path / "cache"
    assert out.is_file()
    assert list(src.parent.iterdir()) == [src], "不能往别人的仓库里写生成文件"
