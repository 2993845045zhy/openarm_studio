#!/usr/bin/env python3
"""官方 openarm_control IK 接入测试（纯计算，不碰硬件）。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("openarm_control")
from openarm_studio.backend.ik import ArmIK  # noqa: E402


@pytest.fixture(scope="module")
def ik():
    if not ArmIK.available():
        pytest.skip("缺少官方 v1 MuJoCo 模型")
    return ArmIK.shared()


def test_fk_returns_seven_dim_pose(ik):
    pose = ik.fk("right", np.zeros(8, dtype=np.float32))
    assert len(pose) == 7
    assert all(np.isfinite(pose))


def test_ik_roundtrip_position(ik):
    """抬起 5cm 后逆解，末端位置误差应小于 1cm。"""
    q = np.zeros(8, dtype=np.float32)
    q[3] = 1.2
    p = ik.fk("right", q)
    target = list(p)
    target[2] += 0.05
    sol = ik.ik("right", target, q)
    assert sol is not None, "IK 无解"
    back = ik.fk("right", sol)
    err = float(np.linalg.norm(np.array(back[:3]) - np.array(target[:3])))
    assert err < 0.01, f"末端误差 {err:.4f} m 过大"


def test_ik_returns_eight_dims(ik):
    q = np.zeros(8, dtype=np.float32)
    p = ik.fk("left", q)
    sol = ik.ik("left", p, q)
    assert sol is not None and len(sol) == 8
