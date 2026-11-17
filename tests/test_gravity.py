#!/usr/bin/env python3
"""重力模型（官方 KDL 栈）单元测试。

验证：链构造正确、重力矩随姿态平滑变化、左右臂镜像互补。
不接触硬件；首次运行会跑一次 xacro（约 1~2s）。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append("/usr/lib/python3/dist-packages")   # 系统 PyKDL（ROS 2 同源）

pytest.importorskip("PyKDL")
pytest.importorskip("urdf_parser_py")

from openarm_studio.backend.gravity import GravityModel  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return GravityModel.shared()


def test_chains_have_seven_joints(model):
    for side in ("left", "right"):
        names = model.basis(side)
        assert len(names) == 7
        assert names[0].endswith("joint1") and names[-1].endswith("joint7")


def test_gravity_shape_and_finite(model):
    q = [0.0] * 7
    for side in ("left", "right"):
        g = model.gravity(side, q)
        assert g.shape == (7,)
        assert np.all(np.isfinite(g))


def test_gravity_is_pose_dependent(model):
    """J2 变化时 J2 的重力矩必须随之变化（不能是常数/全零）。"""
    vals = [float(model.gravity("right", [0.0, a, 0.0, 0.6, 0, 0, 0])[1])
            for a in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    assert len(set(np.round(vals, 3))) >= 4, f"重力矩几乎不变：{vals}"
    assert max(vals) - min(vals) > 1.0, f"变化幅度过小：{vals}"


def test_gravity_magnitude_reasonable(model):
    """单关节重力矩应在几十 Nm 以内（物理合理性）。"""
    for side in ("left", "right"):
        g = model.gravity(side, [0.3, -0.4, 0.2, 0.8, 0.1, -0.2, 0.1])
        assert np.abs(g).max() < 50.0


def test_left_right_mirrored(model):
    """J2（肩部偏航）在镜像姿态下应反号；J4（肘部）两臂同向，不在此列。

    注：openarm 两臂的肘关节绕同一方向，J4 重力矩同号是正确的。
    """
    qr = [0.2, -0.5, 0.3, 0.9, 0.1, -0.1, 0.2]
    ql = [-0.2, 0.5, -0.3, 0.9, -0.1, 0.1, -0.2]
    gr = model.gravity("right", qr)
    gl = model.gravity("left", ql)
    assert gr[1] * gl[1] < 0, f"J2 未镜像：right={gr[1]:.2f} left={gl[1]:.2f}"
    # J4 同向
    assert gr[3] * gl[3] > 0, f"J4 应同向：right={gr[3]:.2f} left={gl[3]:.2f}"


# ── 摩擦补偿（官方 tanh 模型）──────────────────────────────
def test_friction_zero_at_rest(model):
    """静止时只允许纯粘滞项，库仑项为 0；默认不计恒定偏置 Fo。"""
    f = model.friction("right", [0.0] * 7)
    assert np.allclose(f, 0.0, atol=1e-9)


def test_friction_sign_follows_velocity(model):
    """ω>0 的补偿力矩为正（驱动需同向出力），反向同理。"""
    pos = model.friction("right", [0.5] * 7)
    neg = model.friction("right", [-0.5] * 7)
    assert np.all(pos > 0.0)
    assert np.all(neg < 0.0)
    assert np.allclose(pos, -neg)


def test_friction_magnitude_reasonable(model):
    """1 rad/s 时的补偿量：末端小电机 <0.3 Nm，大关节 ~1 Nm（物理合理）。"""
    f = model.friction("right", [1.0] * 7)
    assert np.abs(f).max() < 2.0
    assert abs(f[4]) < 0.3 and abs(f[5]) < 0.3


def test_friction_bias_is_optional(model):
    f0 = model.friction("right", [0.0] * 7)
    fb = model.friction("right", [0.0] * 7, include_bias=True)
    assert np.allclose(f0, 0.0, atol=1e-9)
    assert np.abs(fb).max() > 0.05      # Fo 生效
    assert fb[0] == pytest.approx(0.088, abs=1e-6)


# ── 导纳用的雅可比 ────────────────────────────────────────
def test_jacobian_shape(model):
    for side in ("left", "right"):
        jac = model.jacobian(side, [0.0] * 7)
        assert jac.shape == (3, 7)
        assert np.all(np.isfinite(jac))


def test_jacobian_matches_numeric(model):
    """雅可比每一列 = 该关节微扰下的末端位置变化率（有限差分校验）。"""
    q = np.array([0.1, -0.4, 0.2, 0.9, 0.1, -0.2, 0.3])
    jac = model.jacobian("right", q)
    eps = 1e-5
    for i in range(7):
        qp, qm = q.copy(), q.copy()
        qp[i] += eps
        qm[i] -= eps
        num = (model.ee_position("right", qp)
               - model.ee_position("right", qm)) / (2 * eps)
        assert np.allclose(jac[:, i], num, atol=1e-3), f"J{i + 1} 列不匹配"


def test_ee_position_moves_with_joint(model):
    p0 = model.ee_position("right", [0.0] * 7)
    p1 = model.ee_position("right", [0.0, -0.6, 0.0, 0.9, 0.0, 0.0, 0.0])
    assert np.linalg.norm(p1 - p0) > 0.05
