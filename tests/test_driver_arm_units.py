#!/usr/bin/env python3
"""DriverArm 纯逻辑单测（无硬件）：部分目标的零填充与单步限幅。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openarm_studio.backend.driver_arm import GRIP_OPEN_M, IMP_KP_MIN, DriverArm


def _soft_arm(side="right"):
    """`_soft_gains` 所需的最小状态（不跑 __init__）。"""
    a = _arm_with_state(side)
    a._grav_kd_scale = 2.0
    a._imp_on = False
    a._imp_stiff = 0.35
    a._imp_zeta = 1.0
    a._imp_kp_cur = None
    a._imp_kd_cur = None
    a._imp_t = None
    a._last_kp = None
    a._last_kd = None
    return a


class _FakeDrv:
    kps = (70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0)
    kds = (2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5)


def _arm_with_state(side="right", positions=None):
    a = DriverArm.__new__(DriverArm)          # 不跑 __init__（不碰 CAN）
    a.side = side
    a._lock = __import__("threading").Lock()
    a._delta_limits = np.array([1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 3.14])
    a._positions = positions or {}
    a._targets = None
    a._last_send_t = 0.0
    a._last_poll_t = 0.0
    return a


def test_to_array_seeds_missing_from_measured_once():
    """从未下发过目标（_targets=None）时，缺省关节用实测角播种。"""
    measured = {f"openarm_right_joint{i}": 1.0 for i in range(1, 8)}
    measured["openarm_right_finger_joint1"] = 0.02
    a = _arm_with_state("right", measured)
    arr = a._to_array({"openarm_right_joint1": 0.2})
    assert arr[0] == 0.2
    for i in range(1, 7):
        assert arr[i] == 1.0, f"joint{i+1} 应取实测 1.0，得到 {arr[i]}"
    # 夹爪 0.02 m → 负 rad（右臂取反）
    assert arr[7] == np.float64(-0.02 / GRIP_OPEN_M * 1.0472)


def test_to_array_holds_last_command_not_measured():
    """下发过目标后，缺省关节保持上一拍指令（恒值），不再追实测角——
    追实测角会让 PD 伺服追自己的编码器形成正反馈震荡。"""
    measured = {f"openarm_right_joint{i}": 0.9 for i in range(1, 8)}
    a = _arm_with_state("right", measured)
    a._targets = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, -0.5])
    arr = a._to_array({"openarm_right_joint1": 0.3})
    assert arr[0] == 0.3                       # 显式给定的用新值
    assert arr[1:7].tolist() == [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]  # 保持指令
    assert arr[7] == -0.5                      # 夹爪：last_cmd 已是电机弧度，直用
    assert arr[1] != 0.9                       # 绝不能追实测角


def test_to_array_empty_state_falls_back_zero():
    a = _arm_with_state("left", {})
    arr = a._to_array({"openarm_left_joint3": 0.5})
    assert arr[2] == 0.5
    assert arr[0] == 0.0


def test_step_toward_clamps_big_jump():
    """目标跳 3.0 rad 时，单步只允许走限幅 1.0。"""

    class FakeDrv:
        last_command = np.array([0.0] * 8)

    a = _arm_with_state()
    out = a._step_toward(FakeDrv(), np.array([3.0] + [0.0] * 7))
    assert out[0] == 1.0 and out[1] == 0.0


def test_step_toward_small_target_untouched():
    class FakeDrv:
        last_command = np.array([0.5] + [0.0] * 7)

    a = _arm_with_state()
    out = a._step_toward(FakeDrv(), np.array([0.9] + [0.0] * 7))
    assert out[0] == 0.9


def test_step_toward_walk_converges():
    """50Hz 重发全量目标，逐拍逼近最终收敛到目标。"""
    tgt = np.array([2.0, -2.0] + [0.0] * 6)

    class FakeDrv:
        last_command = np.zeros(8)

    a = _arm_with_state()
    drv = FakeDrv()
    for _ in range(10):
        drv.last_command = a._step_toward(drv, tgt)
    assert np.allclose(drv.last_command, tgt)


# ── 摩擦补偿 ──────────────────────────────────────────────
def test_friction_tau_disabled_is_zero():
    a = _arm_with_state()
    a._fric_on = False
    a._grav_model = None
    a._fric_tau = None
    assert np.allclose(a._friction_tau(["openarm_right_joint1"], {}), 0.0)


def test_friction_tau_lowpass_and_clamp():
    """前馈要过一阶低通（编码器速度噪声），并且单项限幅 3 Nm。"""

    class FakeModel:
        def friction(self, side, v):
            return np.array([10.0] * 7)      # 故意给个超大值

    a = _arm_with_state()
    a._fric_on = True
    a._grav_model = FakeModel()
    a._fric_tau = None
    first = a._friction_tau([f"j{i}" for i in range(7)], {})
    assert np.allclose(first, 3.0)           # 第一次就被限幅
    assert a._fric_tau is not None


# ── 关节空间阻抗（整臂）───────────────────────────────────
def test_soft_gains_teach_is_limp_without_velocity_feedforward():
    a = _soft_arm()
    kp, kd, dq = a._soft_gains(_FakeDrv(), True)
    assert dq is False                      # 立即停发速度前馈
    assert kp[0] > 1.0                      # 但刚度是**斜坡**降下去的
    for _ in range(400):                    # 每次按 dt=5ms 推进 → 2s
        kp, kd, dq = a._soft_gains(_FakeDrv(), True)
    assert np.allclose(kp, 0.5)
    assert dq is False


def test_soft_gains_gravity_uses_official_gains_and_kd_clamp():
    a = _soft_arm()
    for _ in range(400):
        kp, kd, dq = a._soft_gains(_FakeDrv(), False)
    assert np.allclose(kp, _FakeDrv.kps)
    # kd 基准 = 官方 × 2，J1 的 5.5 必须被裁到 MIT 编码上限 5.0
    assert np.allclose(kd, [5.0, 5.0, 4.0, 4.0, 1.4, 1.2, 1.0])
    assert dq is True


def test_soft_gains_impedance_scales_with_params():
    """Kp = 官方 × 刚度，Kd = 官方×2 × ζ × √刚度，都受限幅。"""
    base_kp = np.asarray(_FakeDrv.kps)
    base_kd = np.asarray(_FakeDrv.kds) * 2.0
    for stiff, zeta in ((0.10, 1.0), (0.35, 0.6), (1.00, 1.0), (0.01, 2.0)):
        a = _soft_arm()
        a._imp_on = True
        a._imp_stiff, a._imp_zeta = stiff, zeta
        for _ in range(400):                 # 每次调用按 dt=5ms 推进
            kp, kd, dq = a._soft_gains(_FakeDrv(), False)
        want_kp = np.maximum(base_kp * stiff, IMP_KP_MIN)
        want_kd = np.clip(base_kd * zeta * np.sqrt(stiff), 0.02, 5.0)
        assert np.allclose(kp, want_kp, atol=1e-6), (stiff, zeta, kp)
        assert np.allclose(kd, want_kd, atol=1e-6), (stiff, zeta, kd)


def test_soft_gains_full_stiffness_equals_gravity_comp():
    """刚度 100% + 阻尼比 1.0 必须与重力补偿逐参数相同。"""
    g = _soft_arm()
    for _ in range(400):
        g_kp, g_kd, _ = g._soft_gains(_FakeDrv(), False)
    i = _soft_arm()
    i._imp_on = True
    i._imp_stiff, i._imp_zeta = 1.0, 1.0
    for _ in range(400):
        i_kp, i_kd, _ = i._soft_gains(_FakeDrv(), False)
    assert np.allclose(g_kp, i_kp) and np.allclose(g_kd, i_kd)


def test_soft_gains_ramp_prevents_stiffness_step():
    """示教(kp=0.5) 切到满刚度(kp=70) 时，第一帧不能一步到位。"""
    a = _soft_arm()
    for _ in range(400):
        a._soft_gains(_FakeDrv(), True)      # 示教：稳定在 kp=0.5
    a._imp_on = True
    a._imp_stiff, a._imp_zeta = 1.0, 1.0
    kp0, _, _ = a._soft_gains(_FakeDrv(), False)
    assert kp0[0] < 5.0, f"刚度阶跃过大：{kp0[0]}"


def test_reset_gain_ramp_clears_state():
    """离开软模式后必须清掉斜坡状态，否则下次进入会从旧中间值起步。"""
    a = _soft_arm()
    a._soft_gains(_FakeDrv(), False)
    assert a._last_kp is not None
    a._reset_gain_ramp()
    assert a._imp_kp_cur is None and a._last_kp is None
