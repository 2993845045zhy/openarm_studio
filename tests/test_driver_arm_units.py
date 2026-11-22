#!/usr/bin/env python3
"""DriverArm 纯逻辑单测（无硬件）：部分目标的零填充与单步限幅。"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openarm_studio.backend.driver_arm import (
    GRIP_OPEN_M, IMP_KP_MIN, MANUAL_JOINT_SPEED_RAD_S, SLIDER_JOINT_SPEED_LIMITS_RAD_S, DriverArm,
)
from openarm_studio.backend.gripper import GRASP_MIN_LIMIT, GripperFeedback


def _soft_arm(side="right"):
    """`_soft_gains` 所需的最小状态（不跑 __init__）。"""
    a = _arm_with_state(side)
    a._grav_kd_scale = 2.0
    a._imp_on = False
    a._power_motion = None
    a._teleop_role = None
    a._imp_stiff = 0.35
    a._imp_zeta = 1.0
    a._imp_kp_cur = None
    a._imp_kd_cur = None
    a._imp_t = None
    a._last_kp = None
    a._last_kd = None
    a._soft_prev_tgt = None
    a._vel_limits = None
    a._lim_lo = None
    a._lim_hi = None
    return a


class _FakeDrv:
    kps = (70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0)
    kds = (2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5)


def _arm_with_state(side="right", positions=None):
    a = DriverArm.__new__(DriverArm)          # 不跑 __init__（不碰 CAN）
    a.side = side
    a._lock = __import__("threading").Lock()
    a._delta_limits = np.array([1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 3.14])
    a._vel_limits = None
    a._slider_target = False
    a._positions = positions or {}
    a._targets = None
    a._last_send_t = 0.0
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
    out = a._step_toward(FakeDrv(), np.array([3.0] + [0.0] * 7), 0.02, manual=False)
    assert out[0] == 1.0 and out[1] == 0.0


def test_step_toward_small_target_untouched():
    class FakeDrv:
        last_command = np.array([0.5] + [0.0] * 7)

    a = _arm_with_state()
    out = a._step_toward(FakeDrv(), np.array([0.9] + [0.0] * 7), 0.02, manual=False)
    assert out[0] == 0.9


def test_step_toward_walk_converges():
    """50Hz 重发全量目标，逐拍逼近最终收敛到目标。"""
    tgt = np.array([2.0, -2.0] + [0.0] * 6)

    class FakeDrv:
        last_command = np.zeros(8)

    a = _arm_with_state()
    drv = FakeDrv()
    for _ in range(10):
        drv.last_command = a._step_toward(drv, tgt, 0.02, manual=False)
    assert np.allclose(drv.last_command, tgt)


def test_connect_does_not_mistake_a_full_can_queue_for_motor_silence(monkeypatch):
    from types import SimpleNamespace

    import openarm_driver

    logs = []
    arm = DriverArm("left", "can1", on_log=logs.append)
    monkeypatch.setattr("openarm_studio.backend.driver_arm.can_ifstate",
                        lambda iface: {"exists": True, "up": True, "fd": True})
    axes = SimpleNamespace(get_link_stats=lambda _i: SimpleNamespace(responses=0))
    bus = SimpleNamespace(write_no_buffer=SimpleNamespace(count=20))
    reads = []
    drv = SimpleNamespace(
        openarm=SimpleNamespace(get_arm=lambda: axes, get_bus_status=lambda: bus),
        set_latest_state=lambda *, timeout_us: reads.append(timeout_us),
    )
    monkeypatch.setattr(openarm_driver, "Config", lambda _name: SimpleNamespace())
    monkeypatch.setattr(openarm_driver, "SingleArmDriver", lambda *args, **kwargs: drv)
    assert not arm.connect()
    assert not arm.connected
    assert reads == [arm.RECV_US]
    assert "can1 发送队列满 20 次" in logs[-1]


def test_disabled_can_poll_backs_off_until_send_queue_accepts_frames():
    from types import SimpleNamespace

    logs = []
    arm = DriverArm("left", "can1", on_log=logs.append)
    bus = SimpleNamespace(write_no_buffer=SimpleNamespace(count=0))
    blocked = True

    def refresh():
        if blocked:
            bus.write_no_buffer.count += 8

    axes = SimpleNamespace(
        get_link_stats=lambda _i: SimpleNamespace(seconds_since_response=lambda: 0.0))
    drv = SimpleNamespace(
        openarm=SimpleNamespace(refresh_all=refresh, get_arm=lambda: axes,
                                get_bus_status=lambda: bus),
        set_latest_state=lambda **_kwargs: None,
        latest_state={"qpos": np.zeros(8), "qvel": np.zeros(8)},
    )
    arm._poll_feedback(drv, 1.0)
    assert arm._next_poll_t == pytest.approx(1.25)
    arm._poll_feedback(drv, 1.25)
    assert arm._next_poll_t == pytest.approx(1.50)
    assert len(logs) == 1 and "发送队列满" in logs[0]

    blocked = False
    arm._poll_feedback(drv, 1.5)
    assert arm._next_poll_t == pytest.approx(1.54)
    assert not arm._idle_poll_blocked

    arm._enabled = True
    arm._poll_feedback(drv, 2.0)
    assert arm._next_poll_t == pytest.approx(2.01)


# ── 摩擦补偿 ──────────────────────────────────────────────
def test_friction_torque_ramps_at_switch_and_stays_within_limit():
    from types import SimpleNamespace

    a = DriverArm("right", "unused")
    a._grav_model = SimpleNamespace(friction=lambda side, velocity: 100.0 * velocity)
    a._fric_feedback = (np.ones(7), np.full(7, 10.0))

    a.set_friction_comp(False)
    a._apply_force_mode(None)
    assert a._friction_tau(1.0, 0.02) == pytest.approx(np.zeros(7))

    a.set_friction_comp(True)
    a._apply_force_mode(None)
    first = a._friction_tau(1.0, 0.02)
    assert first == pytest.approx(np.full(7, 0.2))
    for _ in range(50):
        steady = a._friction_tau(1.0, 0.02)
    assert steady == pytest.approx(np.full(7, 3.0))
    assert steady.max() <= 3.0

    a.set_friction_comp(False)
    a._apply_force_mode(None)
    first_off = a._friction_tau(1.0, 0.02)
    assert first_off == pytest.approx(np.full(7, 2.8))
    for _ in range(50):
        off = a._friction_tau(1.0, 0.02)
    assert off == pytest.approx(np.zeros(7))


def test_friction_ignores_rest_noise_and_releases_old_direction():
    from openarm_studio.backend.gravity import GravityModel

    arm = DriverArm("left", "unused")
    arm._grav_model = GravityModel.__new__(GravityModel)
    velocity = np.full(7, 0.5)
    arm._fric_feedback = velocity, np.full(7, 10.0)
    for _ in range(50):
        output = arm._friction_tau(1.0, 0.02)
    assert output == pytest.approx(arm._grav_model.friction("left", velocity))

    velocity[0] = -0.5
    velocity[1] = 0.04
    raw = arm._grav_model.friction("left", velocity)
    output = arm._friction_tau(1.0, 0.02)
    assert raw[0] < output[0] < 0.0
    assert 0.0 < output[1] < raw[1]
    assert np.all(output[2:] > 0.0)

    quantum = np.array([0.022, 0.022, 0.005, 0.005, 0.015, 0.015, 0.015])
    for sign in (-1, 1, -1, 0):
        velocity[:] = sign * quantum
        assert arm._friction_tau(1.0, 0.02) == pytest.approx(np.zeros(7))


def test_teach_wrist_friction_helps_earlier_without_cancelling_damping():
    from openarm_studio.backend.gravity import GravityModel

    arm = DriverArm("right", "unused")
    arm._grav_model = GravityModel.__new__(GravityModel)
    velocity = np.full(7, 0.03)
    arm._fric_feedback = velocity, np.full(7, 10.0)
    wrist_kd = (1.0, 1.0, 0.9)
    for _ in range(40):
        baseline = arm._friction_tau(1.0, 0.02)
    for _ in range(40):
        assist = arm._friction_tau(1.0, 0.02, wrist_kd=wrist_kd)

    assert assist[:4] == pytest.approx(baseline[:4])
    assert np.all(assist[4:] > baseline[4:])
    assert np.all(velocity[4:] * wrist_kd - assist[4:] < velocity[4:] * 1.5 - baseline[4:])

    velocity[4:] = [-0.05, -0.05, 0.015]
    ramp_kd = (0.7, 0.6, 0.5)
    output = arm._friction_tau(1.0, 0.02, wrist_kd=ramp_kd)
    assert np.all(output[4:6] < 0.0)
    assert np.all(np.abs(output[4:]) <= 0.8 * np.abs(velocity[4:]) * ramp_kd)
    assert output[6] == 0.0
    for _ in range(40):
        output = arm._friction_tau(1.0, 0.02, wrist_kd=ramp_kd)
    assert np.all(np.abs(output[4:]) <= 0.8 * np.abs(velocity[4:]) * ramp_kd)
    assert np.all(arm._friction_tau(10.0, 0.02, wrist_kd=ramp_kd) == 0.0)


def test_teach_wrist_uses_coherent_position_steps_below_velocity_deadband(monkeypatch):
    from types import SimpleNamespace

    from openarm_studio.backend.gravity import GravityModel

    clock = [1.0]
    monkeypatch.setattr("openarm_studio.backend.driver_arm.time.monotonic", lambda: clock[0])
    arm = DriverArm("right", "unused")
    arm._grav_on = True
    arm._grav_kp = 0.5
    arm._grav_model = GravityModel.__new__(GravityModel)
    angle = np.zeros(8)
    velocity = np.full(8, 0.0073)
    ages = np.zeros(7)
    axes = SimpleNamespace(get_link_stats=lambda i: SimpleNamespace(
        seconds_since_response=lambda: ages[i]))
    drv = SimpleNamespace(openarm=SimpleNamespace(get_arm=lambda: axes),
                          latest_state={"qpos": angle, "qvel": velocity})
    wrist_kd = (1.0, 1.0, 0.9)
    tick = 25 / 65535

    for i in range(4):
        clock[0] = 1.0 + i * 0.030
        angle[4:7] = np.array([i, -i, i % 2]) * tick
        arm._sample_joint_feedback(drv)
        output = arm._friction_tau(clock[0], 0.02, wrist_kd=wrist_kd)
        if i < 3:
            assert not output.any()
    assert 0.0 < output[4] <= 0.025 and -0.035 <= output[5] < 0.0
    assert output[6] == 0.0

    # 慢速转动时速度包在临近的量化格之间跳动。相同的位置轨迹应得到相同的助力。
    slow_outputs = []
    for reported_speed in (-0.0073, 0.0073, 0.022):
        velocity[4] = reported_speed
        arm._sample_joint_feedback(drv)
        arm._fric_tau = None
        slow_outputs.append(arm._friction_tau(clock[0], 0.02, wrist_kd=wrist_kd)[4])
    assert slow_outputs[0] > 0.0
    assert slow_outputs == pytest.approx([slow_outputs[0]] * 3)

    velocity[4] = 0.0073
    # 倒过来一格，立即撤掉旧方向的力矩。另一轴长时间不动也撤掉。
    clock[0] = 1.11
    angle[4] -= tick
    arm._sample_joint_feedback(drv)
    assert arm._friction_tau(clock[0], 0.02, wrist_kd=wrist_kd)[4] == 0.0
    clock[0] = 1.15
    arm._sample_joint_feedback(drv)
    assert not arm._friction_tau(clock[0], 0.02, wrist_kd=wrist_kd).any()

    # 之前跨过反馈中断的位移不能为恢复后的第一帧提供起转证据。
    clock[0] = 1.25
    angle[4:7] += [5 * tick, -5 * tick, 0.0]
    arm._sample_joint_feedback(drv)
    assert not arm._friction_tau(clock[0], 0.02, wrist_kd=wrist_kd).any()


def test_teach_wrist_assist_fades_at_speed_and_joint_limits():
    arm = DriverArm("right", "unused")
    arm._lim_lo = np.full(8, -1.0)
    arm._lim_hi = np.full(8, 1.0)
    tick = 25 / 65535
    for i in range(4):
        q = np.array([1.0 - (3 - i) * tick, i * tick, i * tick])
        arm._teach_wrist_positions.append((q, np.full(3, i * 0.02)))

    corrected, assist = arm._teach_wrist_friction_inputs(
        0.06, np.array([0.0, 0.10, 0.0]), np.ones(3, dtype=bool))
    assert assist == pytest.approx([0.0, 0.0, 0.05])
    assert corrected == pytest.approx([0.0, 0.10, tick / 0.02])

    fresh = np.array([True, True, False])
    corrected, assist = arm._teach_wrist_friction_inputs(0.06, np.array([0.0, -0.04, 0.0]), fresh)
    assert not assist.any()
    assert corrected == pytest.approx([0.0, -0.04, 0.0])

    _, partial = arm._teach_wrist_friction_inputs(
        0.06, np.array([0.0, 0.05, 0.06]), np.ones(3, dtype=bool))
    assert 0.0 < partial[2] < 0.05


def test_soft_frame_keeps_gravity_when_one_axis_loses_friction_feedback(monkeypatch):
    from types import SimpleNamespace

    clock = [1.0]
    monkeypatch.setattr("openarm_studio.backend.driver_arm.time.monotonic", lambda: clock[0])
    arm = DriverArm("right", "unused")
    arm._grav_on = True
    names = [f"openarm_right_joint{i}" for i in range(1, 8)]
    arm._grav_model = SimpleNamespace(
        basis=lambda side: names,
        gravity=lambda side, q: np.full(7, 2.0),
        friction=lambda side, velocity: velocity,
    )
    ages = np.zeros(7)
    ages[1] = 0.05
    velocity = np.full(8, 0.5)
    velocity[2] = np.nan
    frames = []
    axes = SimpleNamespace(
        get_link_stats=lambda i: SimpleNamespace(seconds_since_response=lambda: ages[i]),
        mit_control_all=frames.append,
    )

    class Driver(_FakeDrv):
        last_command = np.zeros(8)
        joint_offsets = np.zeros(8)
        gripper_posforce_limits = (1.0, 1.0)
        latest_state = {"qpos": np.zeros(8), "qvel": velocity}
        openarm = SimpleNamespace(
            get_arm=lambda: axes,
            get_gripper=lambda: SimpleNamespace(set_position=lambda *args, **kwargs: None),
            recv_all=lambda *_: None,
        )

    drv = Driver()
    arm._sample_joint_feedback(drv)
    arm._send_soft_frame(drv, 0.02)
    assert [frame.tau for frame in frames[-1]][:3] == pytest.approx([2.196735, 2.196735, 2.0])

    clock[0] = 1.02
    arm._send_soft_frame(drv, 0.02)
    assert frames[-1][0].tau > frames[-2][0].tau
    assert [frame.tau for frame in frames[-1]][1:3] == pytest.approx([2.0, 2.0])

    clock[0] = 1.07
    arm._send_soft_frame(drv, 0.02)
    assert [frame.tau for frame in frames[-1]] == pytest.approx([2.0] * 7)


# ── 关节空间阻抗（整臂）───────────────────────────────────
@pytest.mark.parametrize(
    ("role", "expected_wrist_kd"),
    [(None, [1.0, 1.0, 0.9]), ("leader", [1.5, 1.5, 1.5])],
)
def test_soft_gains_teach_is_limp_without_velocity_feedforward(role, expected_wrist_kd):
    a = _soft_arm()
    a._teleop_role = role
    kp, kd, dq = a._soft_gains(_FakeDrv(), True)
    assert dq is False                      # 立即停发速度前馈
    assert kp[0] > 1.0                      # 但刚度是**斜坡**降下去的
    for _ in range(400):                    # 每次按 dt=5ms 推进 → 2s
        kp, kd, dq = a._soft_gains(_FakeDrv(), True)
    assert np.allclose(kp, 0.5)
    assert kd == pytest.approx([2.7, 2.7, 2.2, 2.2, *expected_wrist_kd])
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


def test_manual_target_limits_steps_smooths_arrival_and_never_catches_up():
    arm = DriverArm("right", "unused")
    arm._vel_limits = np.array([0.10, 2, 2, 2, 2, 2, 2])
    target = np.array([0.12, -0.35, 0, 0, 0, 0, 0, -0.4])

    class Driver:
        last_command = np.zeros(8)

    drv = Driver()
    steps = []
    for frame in range(250):
        previous = drv.last_command.copy()
        drv.last_command = arm._step_toward(drv, target, 8.0 if frame == 3 else 0.02, manual=True)
        steps.append(drv.last_command - previous)
    assert np.max(np.abs(np.array(steps)[:, 0])) <= 0.10 * 0.02 + 1e-10
    assert np.max(np.abs(np.array(steps)[:, 1])) == pytest.approx(MANUAL_JOINT_SPEED_RAD_S * 0.02)
    assert abs(steps[160][1]) < abs(steps[0][1])
    assert np.allclose(drv.last_command, target)


@pytest.mark.parametrize("mode", ["position", "gravity", "impedance"])
def test_slider_gets_faster_peak_and_slows_at_each_end(monkeypatch, mode):
    from types import SimpleNamespace

    clock = [1.0]
    monkeypatch.setattr("openarm_studio.backend.driver_arm.time.monotonic", lambda: clock[0])
    arm = DriverArm("right", "unused")
    arm._enabled = True
    arm._vel_limits = np.array([2.0, 2.0, 0.3, 2.0, 2.0, 2.0, 2.0])
    arm._grav_on = mode == "gravity"
    arm._imp_on = mode == "impedance"
    arm._fric_on = False
    names = [f"openarm_right_joint{i}" for i in range(1, 8)]
    arm._grav_model = SimpleNamespace(basis=lambda side: names, gravity=lambda side, q: np.ones(7))
    arm._positions = dict.fromkeys(names, 0.0)
    frames = [np.zeros(7)]

    class Driver(_FakeDrv):
        last_command = np.zeros(8)
        safety_stop_reason = None
        joint_offsets = np.zeros(8)
        gripper_posforce_limits = (1.0, 1.0)

        def send_position(self, target):
            self.last_command = target
            frames.append(target[:7])
            return True

    drv = Driver()
    drv.openarm = SimpleNamespace(
        recv_all=lambda *_: None,
        get_arm=lambda: SimpleNamespace(
            mit_control_all=lambda params: frames.append(np.array([param.q for param in params]))),
        get_gripper=lambda: SimpleNamespace(set_position=lambda *args, **kwargs: None),
    )
    goal = [1.3, 0.08, 0.5, 1.3, 1.3, 0.65, 1.3]
    for frame in range(240):
        clock[0] += 0.5 if frame == 3 else 0.021
        arm.set_targets(dict(zip(names, goal)), slider=True)
        arm._send_current_target(drv)

    velocity = np.diff(frames, axis=0) / 0.02
    assert velocity[0] == pytest.approx([0.072] * 4 + [0.1, 0.108, 0.116])
    assert frames[17][1] == pytest.approx(0.08)
    assert frames[78][0] == pytest.approx(1.3)
    peaks = np.max(velocity, axis=0)
    assert peaks[[0, 2, 3, 4, 5, 6]] == pytest.approx([1.10, 0.30, 1.10, 1.18, 1.20, 1.25])
    assert 0.0 < peaks[1] < SLIDER_JOINT_SPEED_LIMITS_RAD_S[1]
    assert np.min(velocity) >= -1e-10
    velocity_changes = np.diff(velocity, axis=0)
    accel_steps = np.array([0.072] * 4 + [0.1, 0.108, 0.116])
    brake_steps = np.array([0.084] * 4 + [0.12, 0.128, 0.136])
    assert np.all(np.max(velocity_changes, axis=0) <= accel_steps + 1e-9)
    assert np.all(np.min(velocity_changes, axis=0) >= -brake_steps - 1e-9)
    assert np.all(velocity[-1] == 0)
    assert frames[-1] == pytest.approx(goal)


def test_wrist_slider_settles_faster_on_short_moves():
    arm = DriverArm("right", "unused")
    arm._slider_target = True
    position = np.zeros(7)
    goal = np.full(7, 0.08)
    first_stopped_frame = np.zeros(7, dtype=int)

    for frame in range(1, 40):
        position = arm._manual_joint_step(position, goal, 0.02)
        stopped = (position == goal) & (arm._slider_velocity == 0.0)
        first_stopped_frame[(first_stopped_frame == 0) & stopped] = frame

    assert first_stopped_frame[:4].tolist() == [16] * 4
    assert np.all((0 < first_stopped_frame[4:]) & (first_stopped_frame[4:] <= 13))


@pytest.mark.parametrize(("axis", "accel_step", "brake_step"), [(0, 0.072, 0.084), (6, 0.116, 0.136)])
def test_slider_changes_direction_without_restarting_velocity(axis, accel_step, brake_step):
    arm = DriverArm("right", "unused")
    arm._slider_target = True
    position = np.zeros(7)
    arm._lim_lo = np.full(8, -1.0)
    arm._lim_hi = np.full(8, 1.0)
    goal = np.ones(7)
    for _ in range(20):
        position = arm._manual_joint_step(position, goal, 0.02)
    assert arm._slider_velocity[axis] == pytest.approx(SLIDER_JOINT_SPEED_LIMITS_RAD_S[axis])

    goal = -goal
    speeds = [arm._slider_velocity[axis]]
    for _ in range(240):
        previous = position
        position = arm._manual_joint_step(previous, goal, 0.02)
        speeds.append((position[axis] - previous[axis]) / 0.02)
        assert np.all(np.abs(position) <= 1.0)
    assert speeds[1] == pytest.approx(speeds[0] - brake_step)
    first_reverse = next(index for index, speed in enumerate(speeds) if speed < -1e-9)
    assert speeds[first_reverse - 1] == pytest.approx(0.0)
    assert speeds[first_reverse] == pytest.approx(-accel_step)
    assert np.max(np.abs(np.diff(speeds))) <= brake_step + 1e-9
    assert position == pytest.approx(goal)


@pytest.mark.parametrize("compensation", [False, True, "impedance"])
def test_preset_uses_rate_limit_and_preserves_compensation(monkeypatch, compensation):
    from types import SimpleNamespace

    arm = DriverArm("right", "unused")
    arm._enabled = True
    arm._grav_on = compensation is True
    arm._imp_on = compensation == "impedance"
    arm._grav_kp = 70.0
    arm._grav_model = SimpleNamespace(
        basis=lambda side: [f"openarm_right_joint{i}" for i in range(1, 8)],
        gravity=lambda side, q: np.full(7, 2.0),
    )
    arm._fric_on = False
    clock = [1.0]
    monkeypatch.setattr("openarm_studio.backend.driver_arm.time.monotonic", lambda: clock[0])

    frames = []
    completed = []
    goal = np.array([0.18, 0, 0, 0, 0, 0, 0, 0])

    class Driver(_FakeDrv):
        arm_side = "right_arm"
        last_command = np.zeros(8)
        safety_stop_reason = None
        joint_offsets = np.zeros(8)
        gripper_posforce_limits = (1.0, 1.0)
        config = SimpleNamespace(
            get_start_config=lambda: {"moves": [{"position": {"right_arm": goal}}]},
        )

        def send_position(self, target):
            self.last_command = target
            frames.append(float(target[0]))
            return True

    drv = Driver()

    def send_mit(params):
        frames.append(params[0].q)
        assert params[0].tau == pytest.approx(2.0)

    drv.openarm = SimpleNamespace(
        recv_all=lambda *_: None,
        get_arm=lambda: SimpleNamespace(mit_control_all=send_mit),
        get_gripper=lambda: SimpleNamespace(set_position=lambda *args, **kwargs: None),
    )
    arm.move_start(done=completed.append)
    for _ in range(180):
        clock[0] += 0.021
        arm._drain_cmds(drv)
        arm._send_current_target(drv)
        if completed:
            break
    assert completed == [True]
    assert arm._hold_mode is bool(compensation)
    assert frames[-1] == pytest.approx(goal[0])
    assert np.max(np.diff([0.0, *frames])) <= MANUAL_JOINT_SPEED_RAD_S * 0.02 + 1e-9



@pytest.mark.parametrize("result", ["settled", "unstable", "stale", "estop"])
def test_compensated_disable_keeps_support_until_stationary_at_park(monkeypatch, result):
    from types import SimpleNamespace

    arm = DriverArm("right", "unused")
    arm._enabled = True
    arm._grav_on = True
    arm._grav_kp = 70.0
    names = [f"openarm_right_joint{i}" for i in range(1, 8)]
    arm._positions = dict.fromkeys(names, 0.0)
    arm._velocities = dict.fromkeys(names, 0.0)
    arm._grav_model = SimpleNamespace(
        basis=lambda side: names,
        gravity=lambda side, q: np.full(7, 2.0),
    )
    arm._fric_on = False

    clock = [1.0]
    monkeypatch.setattr("openarm_studio.backend.driver_arm.time.monotonic", lambda: clock[0])
    goal = np.array([0.1, 0, 0, 0, 0, 0, 0, 0])
    frames = []
    outcomes = []
    shutdowns = []

    def send_mit(params):
        assert arm._grav_on and not shutdowns
        assert all(frame.tau == pytest.approx(2.0) for frame in params)
        frames.append((clock[0], params[0].q))
        if result != "unstable":
            arm._positions[names[0]] = params[0].q

    def disable_motors():
        assert arm._grav_on
        shutdowns.append(clock[0])

    class Driver(_FakeDrv):
        arm_side = "right_arm"
        last_command = np.zeros(8)
        joint_offsets = np.zeros(8)
        safety_stop_reason = None
        gripper_posforce_limits = (1.0, 1.0)
        config = SimpleNamespace(
            get_stop_config=lambda: {"moves": [{"position": {"right_arm": goal}}]},
        )

        def _on_stop(self):
            pytest.fail("software must not replay an uncompensated return")

        def stop(self):
            assert self._on_stop()
            self.openarm.disable_all()

    drv = Driver()
    drv.openarm = SimpleNamespace(
        recv_all=lambda *_: None,
        get_arm=lambda: SimpleNamespace(
            mit_control_all=send_mit,
            get_link_stats=lambda i: SimpleNamespace(is_stale=lambda timeout: result == "stale"),
        ),
        get_gripper=lambda: SimpleNamespace(set_position=lambda *args, **kwargs: None),
        disable_all=disable_motors,
    )
    assert arm.disable(done=outcomes.append)
    for i in range(240):
        clock[0] += 0.021
        if i == 5 and result == "estop":
            arm.estop()
        if not arm._drain_cmds(drv):
            arm._send_current_target(drv)
        if outcomes:
            break

    assert len(frames) > 1
    positions = [position for _, position in frames]
    assert np.max(np.diff([0.0, *positions])) == pytest.approx(0.25 * 0.02)
    if result == "settled":
        assert outcomes == [True] and not arm.enabled and not arm._grav_on
        assert positions[-1] == pytest.approx(goal[0])
        first_at_goal = next(t for t, position in frames if position == pytest.approx(goal[0]))
        assert len(shutdowns) == 1 and shutdowns[0] - first_at_goal >= 0.25
    elif result == "estop":
        assert outcomes == [False] and not arm.enabled
        assert shutdowns == [clock[0]] and positions[-1] < goal[0]
    else:
        assert outcomes == [False] and not shutdowns
        assert arm.enabled and arm._grav_on
        assert positions[-1] == pytest.approx(goal[0])



@pytest.mark.parametrize("mode", [
    "gravity", "impedance", "missing_model", "missing_feedback", "delayed_feedback",
    "sdk_feedback", "disabled_motor", "motor_fault", "estop",
])
def test_enable_starts_with_measured_compensated_frame_and_keeps_it(monkeypatch, mode):
    from types import SimpleNamespace
    from openarm_driver.driver import SingleArmDriver

    logs = []
    arm = DriverArm("right", "can0", on_log=logs.append)
    names = [f"openarm_right_joint{i}" for i in range(1, 8)]
    arm._positions = dict.fromkeys(names, -1.0)  # 上一次连接的姿态
    arm._fric_on = False
    if mode == "impedance":
        arm.set_joint_impedance(True, stiffness=0.1)
    if mode == "missing_model":
        monkeypatch.setattr(arm, "_ensure_gravity", lambda: False)
    else:
        arm._grav_model = SimpleNamespace(
            basis=lambda side: names,
            gravity=lambda side, q: np.full(7, 2.0 + q[0]),
        )

    clock = [5.0]
    monkeypatch.setattr("openarm_studio.backend.driver_arm.time.monotonic", lambda: clock[0])
    measured = np.array([0.24, 0, 0, 0, 0, 0, 0, -0.10])
    goal = np.array([0.36, 0, 0, 0, 0, 0, 0, -0.10])
    events, frames, outcomes = [], [], []
    reads = [30] * 7  # 连接期应答不能用于本次使能确认
    refreshes = 0

    def refresh():
        nonlocal refreshes
        refreshes += 1
        for index in range(6):
            reads[index] += 1
        if (mode != "missing_feedback"
                and (mode != "delayed_feedback" or refreshes >= 4)
                and (mode != "sdk_feedback" or refreshes == 1)):
            reads[6] += 1

    def motor(index):
        code = {"disabled_motor": 0, "motor_fault": 0xA}.get(mode, 1) if index == 6 else 1
        return SimpleNamespace(is_enabled=lambda: code == 1, has_error=lambda: code >= 0x8,
                               get_error_code=lambda: code)

    def send_mit(params):
        assert events == ["enable"]
        frames.append(params)
        measured[:7] = [frame.q for frame in params]

    class Driver(_FakeDrv):
        arm_side = "right_arm"
        joint_offsets = np.zeros(8)
        gripper_posforce_limits = (1.0, 1.0)
        config = SimpleNamespace(get_start_config=lambda: {
            "moves": [{"position": {"right_arm": goal}}],
        })
        start = SingleArmDriver.start

        @property
        def safety_stop_reason(self):
            return self._safety_stop_reason

        def set_latest_state(self, timeout_us):
            self.latest_state = {"qpos": measured.copy(), "qvel": np.zeros(8)}

    drv = Driver()
    drv.openarm = SimpleNamespace(
        set_callback_mode_all=lambda mode: None,
        enable_all=lambda: events.append("enable"),
        disable_all=lambda: events.append("disable"),
        refresh_all=refresh,
        recv_all=lambda *_: None,
        get_arm=lambda: SimpleNamespace(
            get_link_stats=lambda i: SimpleNamespace(responses=reads[i], is_stale=lambda _: False),
            get_motors=lambda: [motor(i) for i in range(7)], mit_control_all=send_mit,
        ),
        get_gripper=lambda: SimpleNamespace(set_position=lambda *args, **kwargs: None),
    )
    assert arm.enable(done=outcomes.append)
    arm._drain_cmds(drv)
    failure = {
        "missing_model": None, "missing_feedback": "J7 无新反馈",
        "disabled_motor": "J7 DISABLED", "motor_fault": "J7 OVERCURRENT",
    }
    if mode in failure:
        assert outcomes == [False] and not arm.enabled and not frames
        assert events == ([] if mode == "missing_model" else ["enable", "disable"])
        if failure[mode]:
            assert any(f"[right can0] 使能中止，{failure[mode]}" in log for log in logs)
        return

    assert arm.enabled and not outcomes
    if mode in ("delayed_feedback", "sdk_feedback"):
        assert refreshes == (4 if mode == "delayed_feedback" else 1)
    assert frames[0][0].q == pytest.approx(0.24)
    assert frames[0][0].dq == 0.0
    assert frames[0][0].tau == pytest.approx(2.24)
    for i in range(150):
        clock[0] += 0.021
        if mode == "estop" and i == 4:
            arm.estop()
        drv.set_latest_state(0)
        arm._publish(drv.latest_state)
        if not arm._drain_cmds(drv):
            arm._send_current_target(drv)
        if outcomes:
            break
    assert np.max(np.diff([p[0].q for p in frames])) == pytest.approx(0.40 * 0.02)
    if mode == "estop":
        assert outcomes == [False] and not arm.enabled and events == ["enable", "disable"]
        return
    assert outcomes == [True] and arm.gravity_active and events == ["enable"]
    if mode == "impedance":
        assert arm.joint_impedance_active
    clock[0] += 0.021
    assert arm._send_current_target(drv)
    assert frames[-1][0].q == pytest.approx(goal[0])
    assert frames[-1][0].tau > 2.0



def test_teleop_follower_limits_error_and_holds_when_commands_expire(monkeypatch):
    from types import SimpleNamespace
    from openarm_studio.backend.teleop import JointFeedback, TeleopCommand

    arm = DriverArm("right", "unused")
    arm._enabled = arm._running = True
    arm._vel_limits = np.array([2.0, 2.0, 0.8, 2.0, 2.0, 2.0, 2.0])
    q = np.array([0.3] * 7)
    names = [f"openarm_right_joint{i}" for i in range(1, 8)]
    arm._positions = dict(zip(names, q))
    arm._grav_model = SimpleNamespace(
        basis=lambda side: names,
        gravity=lambda side, joints: np.full(7, 2.0),
    )
    arm._fric_on = False
    clock = [1.0]
    monkeypatch.setattr("openarm_studio.backend.driver_arm.time.monotonic", lambda: clock[0])
    frames = []

    class Driver(_FakeDrv):
        last_command = np.concatenate((q, [-0.2]))
        joint_offsets = np.zeros(8)
        gripper_posforce_limits = (1.0, 1.0)

    drv = Driver()
    gripper = SimpleNamespace(set_position=lambda *args, **kwargs: None)
    motors = SimpleNamespace(mit_control_all=frames.append)
    drv.openarm = SimpleNamespace(
        get_arm=lambda: motors, get_gripper=lambda: gripper, recv_all=lambda *_: None,
    )
    arm._joint_feedback = JointFeedback(tuple(q), (0.0,) * 7, 1.15)
    arm.set_teleop(TeleopCommand("follower", 1.10, tuple(q + 1.5)))
    arm._send_current_target(drv)
    assert arm._teleop_role == "follower" and arm.gravity_active
    limits = np.array([1.05, 1.05, 0.8, 1.05, 1.25, 1.25, 1.25])
    first = np.array([param.q for param in frames[-1]])
    assert first - q == pytest.approx(0.02 * limits)
    assert arm._grav_hold_q == pytest.approx(q + 0.08)

    # 从臂有位移后，目标窗口随实测更新，位置步长和速度前馈仍受各轴限速约束。
    clock[0] = 1.021
    arm._joint_feedback = JointFeedback(tuple(first), tuple(limits), 1.15)
    arm._send_current_target(drv)
    second = np.array([param.q for param in frames[-1]])
    assert second - first == pytest.approx(0.02 * limits)
    assert np.abs([param.dq for param in frames[-1]]) == pytest.approx(limits * (0.02 / 0.021))
    sent = second[0]

    clock[0] = 1.16
    arm._send_current_target(drv)
    assert arm.gravity_active and arm._teleop_role is None
    assert frames[-1][0].q == pytest.approx(sent)
    assert frames[-1][0].tau == pytest.approx(2.0)

    clock[0] = 1.18
    arm._joint_feedback = JointFeedback(tuple(q), (0.0,) * 7, 1.33)
    arm._send_current_target(drv)
    assert arm._teleop_role is None
    assert frames[-1][0].q == pytest.approx(sent)



def test_teleop_watchdog_stops_both_arms_until_a_new_session(monkeypatch):
    from types import SimpleNamespace

    from openarm_studio.backend.teleop import BimanualTeleop, JointFeedback

    clock = [1.0]
    monkeypatch.setattr("openarm_studio.backend.driver_arm.time.monotonic", lambda: clock[0])
    left, right = DriverArm("left", "unused"), DriverArm("right", "unused")
    drivers = [SimpleNamespace(last_command=np.zeros(8)) for _ in range(2)]
    for arm in (left, right):
        arm._running = arm._enabled = True
        arm._grav_model = object()
    origin = (0.0,) * 7
    session = BimanualTeleop("left", origin, origin, mirror=False)

    def submit_commands(session, deadline):
        local_feedback = JointFeedback(origin, origin, deadline)
        for arm, command in zip((left, right), session.commands(local_feedback, local_feedback)):
            arm._joint_feedback = JointFeedback(origin, origin, clock[0] + 0.15)
            arm.set_teleop(command)

    submit_commands(session, 1.10)
    for arm, driver in zip((left, right), drivers):
        arm._update_teleop(driver, clock[0])
    assert (left._teleop_role, right._teleop_role) == ("leader", "follower")

    clock[0] = 1.12
    left._update_teleop(drivers[0], clock[0])
    assert "协同指令超时" in session.watchdog.reason
    submit_commands(session, 1.25)
    for arm, driver in zip((left, right), drivers):
        arm._update_teleop(driver, clock[0])
        assert arm._teleop_role is None and arm.gravity_active

    submit_commands(BimanualTeleop("left", origin, origin, mirror=False), 1.25)
    for arm, driver in zip((left, right), drivers):
        arm._update_teleop(driver, clock[0])
    assert (left._teleop_role, right._teleop_role) == ("leader", "follower")


@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("scenario", [None, "start", "holding", "low_current", "blocked"])
def test_grasp_closes_on_current_limit_and_holds_during_arm_motion(monkeypatch, soft, scenario):
    from types import SimpleNamespace

    arm = DriverArm("right", "unused")
    arm._enabled = arm._running = True
    arm._grav_on = soft
    arm._grip_config_limits = (50.0, 1.0)
    names = [f"openarm_right_joint{i}" for i in range(1, 8)]
    finger = "openarm_right_finger_joint1"
    arm._grav_model = SimpleNamespace(basis=lambda side: names, gravity=lambda side, q: np.ones(7))
    arm._fric_on = False
    clock = [1.0]
    monkeypatch.setattr("openarm_studio.backend.driver_arm.time.monotonic", lambda: clock[0])
    commands = []
    motor = SimpleNamespace(position=-0.75, velocity=0.0, torque=0.0)
    stats = SimpleNamespace(is_stale=lambda _: False, responses=1)

    def get_motor():
        if motor.read_fails:
            raise RuntimeError("gripper feedback unavailable")
        return SimpleNamespace(
            is_enabled=lambda: True, get_position=lambda: motor.position,
            get_velocity=lambda: motor.velocity, get_torque=lambda: motor.torque)

    motor.read_fails = scenario == "start"
    gripper = SimpleNamespace(
        get_link_stats=lambda _: stats,
        get_motor=get_motor,
        set_position=lambda position, **limits: commands.append(
            (position, limits["speed_rad_s"], limits["torque_pu"])),
    )

    class Driver(_FakeDrv):
        last_command = np.r_[np.zeros(7), -0.80]
        joint_offsets = np.r_[np.zeros(7), 0.05]
        gripper_posforce = True
        safety_stop_reason = None
        openarm = SimpleNamespace(
            get_arm=lambda: SimpleNamespace(mit_control_all=lambda _: None),
            get_gripper=lambda: gripper, recv_all=lambda _: None,
        )

        def send_position(self, target):
            self.last_command = target
            speed, nominal = self.gripper_posforce_limits
            gripper.set_position(target[7] + self.joint_offsets[7], speed_rad_s=speed,
                                 torque_pu=nominal / 4.5)
            return True

    def tick(position=None, velocity=None, torque=None, frames=1):
        for _ in range(frames):
            clock[0] += 0.021
            stats.responses += 1
            if position is not None:
                motor.position = position
            if velocity is not None:
                motor.velocity = velocity
            if torque is not None:
                motor.torque = torque
            arm._send_current_target(drv)

    drv = Driver()
    arm._driver = drv
    arm._set_motion_target(drv.last_command.copy())
    arm.set_gripper_force(10 if scenario == "low_current" else 50)
    arm.set_targets({finger: 0.02})
    assert arm.request_grasp()
    arm._send_current_target(drv)
    if scenario == "start":
        assert arm.grasp_status == "feedback_lost"
        assert arm.enabled and arm.connected
        assert commands[-1] == pytest.approx((-0.75, 1.0, 0.5 / 4.5))
        return
    # 面板 10% 低于机构摩擦所需，抬到下限；50% 原样使用。目标直接是全闭（0.0 + 零偏 0.05）
    limit = GRASP_MIN_LIMIT if scenario == "low_current" else 0.5
    command = (0.05, 0.5, limit / 4.5)
    assert arm.grasp_status == "closing"
    assert commands[-1] == pytest.approx(command)

    # 真机起步：力矩先涨到 0.15 N·m（超过旧的接触阈值），40–50 ms 后才动起来
    startup = [(-0.75, 0.0, 0.10), (-0.75, 0.05, 0.15)]
    startup += [(-0.75, 0.0, 0.10)] if scenario == "blocked" else [(-0.748, 0.28, 0.30)]
    for frame in startup:
        tick(*frame)
    assert arm.grasp_status == "closing"
    assert commands[-1] == pytest.approx(command)

    # 物体挡住夹爪：力矩顶到电流上限并保持，目标不回退
    tick(-0.50 if scenario != "blocked" else -0.75, 0.0, limit / 4.5 * 9.5, frames=20)
    assert arm.grasp_status == "holding"
    assert arm.grasp_torque_limit == pytest.approx(limit / 4.5 * 9.5)
    assert commands[-1] == pytest.approx(command)

    arm.set_targets({names[0]: 0.3}, slider=True)
    tick()
    assert arm.grasp_status == "holding"
    assert commands[-1] == pytest.approx(command)

    if scenario == "holding":
        motor.read_fails = True
        clock[0] += 0.021
        arm._send_current_target(drv)
        assert arm.grasp_status == "feedback_lost"
        assert arm.enabled and arm.connected
        assert commands[-1] == pytest.approx(command)  # 反馈丢失不改指令，电机自带保持

    motor.read_fails = False
    arm.set_gripper_force(40)
    arm.set_targets({finger: 0.044})
    tick()
    assert arm.grasp_status == "idle"
    assert arm.grasp_torque_limit is None
    assert commands[-1] == pytest.approx((0.05 - 1.0472, 1.0, 0.4 / 4.5))


# ── 真机录制回放（2026-11-22，右爪 DM4310 / 左爪；数据说明见 fixtures/gripper_posforce/README.md）──
_GRIP_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gripper_posforce"


def _recorded(name):
    with open(_GRIP_FIXTURES / name) as f:
        return list(csv.DictReader(f))


def _frame(row, index):
    from openarm_studio.backend.gripper import GripperFeedback

    return GripperFeedback(float(row["q_rad"]), float(row["v_rad_s"]), float(row["tau_nm"]), index)


@pytest.mark.parametrize("name", ["click_right_1541.csv", "click_left_1549.csv"])
def test_recorded_smart_grasp_clicks_are_not_contact(name):
    """点击「智能抓取」后的前 64 ms：夹爪全开、两指间无物体，旧逻辑据此判定"估计夹稳"。"""
    from openarm_studio.backend.gripper import GRASP_STILL_RAD_S, Grasp

    rows = [row for row in _recorded(name) if 0.0 <= float(row["t_s"]) <= 0.064]
    assert len(rows) >= 8
    # 速度还没超过"静止"门限，力矩却已是旧接触阈值（0.05 N·m）的两倍以上，位移不足 0.1 mm
    assert all(abs(_frame(row, 0).velocity) <= GRASP_STILL_RAD_S for row in rows)
    assert max(_frame(row, 0).torque for row in rows) >= 0.10
    assert (max(_frame(row, 0).position for row in rows)
            - min(_frame(row, 0).position for row in rows)) <= 0.002

    grasp = Grasp(float(rows[0]["t_s"]), _frame(rows[0], 0), 0.2)
    for index, row in enumerate(rows[1:], 1):
        grasp.update(float(row["t_s"]), _frame(row, index))
    assert (grasp.phase, grasp.target) == ("closing", 0.0)


def test_replayed_close_from_open_ends_empty_without_retreating():
    """同一条指令（全开、标幺 0.0444、0.35 rad/s）把目标给到全闭：3.4 s 走完全程。"""
    from openarm_studio.backend.gripper import Grasp

    rows = _recorded("right_close_from_open_speed035_pu0444.csv")
    grasp = Grasp(0.0, _frame(rows[0], 0), 0.2)
    seen = []
    for index, row in enumerate(rows[1:], 1):
        grasp.update(float(row["t_s"]), _frame(row, index))
        seen.append((grasp.phase, grasp.target, float(row["t_s"])))
    assert seen[0][0] == "closing" and seen[-1][0] == "empty"
    assert {target for _, target, _ in seen} == {0.0}
    last_closing = max(index for index, (phase, _, _) in enumerate(seen) if phase == "closing")
    assert seen[last_closing][2] > 2.0
    assert all(phase == "closing" for phase, _, _ in seen[:last_closing + 1])


def test_replayed_hold_keeps_target_and_reports_slip():
    """物体挡在 −0.50：位置固定，速度/力矩噪声取自真机堵转段。"""
    from openarm_studio.backend.gripper import Grasp, GripperFeedback

    approach = _recorded("right_close_from_open_speed035_pu0444.csv")
    stall = [row for row in _recorded("right_stall_vs_current_limit.csv")
             if int(row["pu_x1000"]) == 44]
    object_q = -0.50
    grasp = Grasp(0.0, _frame(approach[0], 0), 0.2)
    now = 0.0
    for index, row in enumerate(approach[1:], 1):
        if float(row["q_rad"]) >= object_q:
            break
        now = float(row["t_s"])
        grasp.update(now, _frame(row, index))
    assert grasp.phase == "closing"

    frame_number = 10000
    for _ in range(100):            # 约 5 分钟的保持（真实帧间隔约 10 ms）
        for row in stall:
            now += 0.02
            frame_number += 1
            grasp.update(now, GripperFeedback(object_q, float(row["v_rad_s"]),
                                              float(row["tau_nm"]), frame_number))
        assert (grasp.phase, grasp.target) == ("holding", 0.0)

    for index, row in enumerate(row for row in approach if float(row["q_rad"]) >= object_q):
        frame_number += 1
        grasp.update(now + 0.05 * index, GripperFeedback(
            float(row["q_rad"]), float(row["v_rad_s"]), float(row["tau_nm"]), frame_number))
    assert grasp.phase == "lost"            # 物体脱出后夹爪走到全闭
    assert grasp.target == 0.0


def test_recorded_tight_spots_justify_the_minimum_current_limit():
    """左爪标幺值 0.0444 闭合时在 −0.814 处卡住：力矩顶满上限却 1 s 多不动；0.08 能走完全程。"""
    from openarm_studio.backend.gripper import (
        GRASP_CLOSED_RAD, GRASP_MIN_PU, STALL_NM_PER_PU, Grasp,
    )

    stuck = _recorded("left_close_from_open_speed035_pu0444.csv")
    stalled = [row for row in stuck
               if -0.815 <= float(row["q_rad"]) <= -0.8125 and abs(float(row["v_rad_s"])) < 0.05]
    assert len(stalled) >= 100                     # 约 1.3 s
    stall_torque = max(float(row["tau_nm"]) for row in stalled)
    assert GRASP_MIN_PU * STALL_NM_PER_PU >= 1.5 * stall_torque

    rows = _recorded("left_close_from_open_speed05_pu08.csv")
    grasp = Grasp(0.0, _frame(rows[0], 0), 0.2)
    seen = []
    for index, row in enumerate(rows[1:], 1):
        grasp.update(float(row["t_s"]), _frame(row, index))
        seen.append((grasp.phase, float(row["q_rad"])))
    assert seen[-1][0] == "empty"
    assert {phase for phase, _ in seen} == {"closing", "empty"}     # 中途没有被当成停住
    assert next(q for phase, q in seen if phase == "empty") >= GRASP_CLOSED_RAD


@pytest.mark.parametrize("side", ["left", "right"])
def test_replayed_end_to_end_grasp_matches_the_recorded_status(side):
    """16:30–16:31 真机：Studio 生产代码路径点智能抓取（面板 60%，两指间无物体），工具的原始记录。"""
    from openarm_studio.backend.gripper import Grasp

    with open(_GRIP_FIXTURES / f"e2e_{side}_grasp_panel60.csv") as f:
        rows = [row for row in csv.DictReader(f) if row["label"] == "grasp"]
    closing = [row for row in rows if row["grasp"] in ("closing", "empty")]
    assert {(row["target"], row["speed"], row["pu"]) for row in closing} == {
        ("0.000000", "0.500", "0.13333")}          # 目标全闭、0.5 rad/s、面板 60% 的电流上限
    recorded = []
    for row in rows:
        if not recorded or recorded[-1] != row["grasp"]:
            recorded.append(row["grasp"])
    assert recorded == ["pending", "closing", "empty"]
    stall = [float(row["tau"]) for row in rows if row["grasp"] == "empty"]
    assert sum(stall) / len(stall) == pytest.approx(1.27, abs=0.03)    # 全闭后按电流上限保持
    assert min(stall) > 1.2

    grasp = Grasp(0.0, GripperFeedback(float(rows[0]["q"]), float(rows[0]["v"]),
                                      float(rows[0]["tau"]), 0), 0.6)
    replayed = []
    for index, row in enumerate(rows[1:], 1):
        grasp.update(index * 0.01, GripperFeedback(float(row["q"]), float(row["v"]),
                                                   float(row["tau"]), index))
        if not replayed or replayed[-1] != grasp.phase:
            replayed.append(grasp.phase)
    assert replayed == ["closing", "empty"]


def test_grasp_feedback_loss_and_cancel_keep_or_stop_the_command_safely():
    from openarm_studio.backend.gripper import Grasp, GripperFeedback

    grasp = Grasp(1.0, GripperFeedback(-0.80, 0.0, 0.0, 1), 0.6)
    grasp.update(1.05, GripperFeedback(-0.70, 0.3, 0.2, 2))
    grasp.update(1.10, None)
    assert (grasp.phase, grasp.target) == ("feedback_lost", -0.70)  # 闭合途中：停在最后已知位置

    grasp = Grasp(1.0, GripperFeedback(-0.80, 0.0, 0.0, 1), 0.6)
    for index in range(1, 12):
        grasp.update(1.0 + 0.05 * index, GripperFeedback(-0.50, 0.0, 1.2, 1 + index))
    assert grasp.phase == "holding"
    grasp.update(2.0, None)
    assert (grasp.phase, grasp.target) == ("feedback_lost", 0.0)    # 已夹住：指令不变

    grasp = Grasp(1.0, GripperFeedback(-0.80, 0.0, 0.0, 1), 0.6)
    grasp.update(1.05, GripperFeedback(-0.70, 0.3, 0.2, 2))
    grasp.finish("cancelled")
    grasp.update(1.10, GripperFeedback(-0.60, 0.3, 0.2, 3))
    assert (grasp.phase, grasp.target) == ("cancelled", -0.70)      # 终态不再响应

    grasp = Grasp(1.0, GripperFeedback(-0.50, 0.0, 0.0, 1), 0.6)
    for index in range(1, 20):
        grasp.update(1.0 + 0.03 * index, GripperFeedback(-0.50, 0.0, 1.2, index + 1))
    assert grasp.phase == "holding"
    grasp.finish("cancelled")
    assert (grasp.phase, grasp.target) == ("cancelled", 0.0)        # 已夹住后被外部停止：继续夹


def test_grasp_reports_timeout_but_keeps_closing_and_ignores_repeated_frames():
    from openarm_studio.backend.gripper import Grasp, GripperFeedback

    grasp = Grasp(0.0, GripperFeedback(-1.0, 0.0, 0.0, 1), 0.6)
    for now in (0.5, 1.0, 1.5):
        grasp.update(now, GripperFeedback(-1.0, 0.0, 0.0, 1))
    assert grasp.phase == "closing"            # 同一帧重复到达，不累计"静止"时间
    for index in range(1, 300):
        grasp.update(index * 0.02, GripperFeedback(-1.0 + 0.001 * index, 0.3, 0.3, index + 1))
    assert (grasp.phase, grasp.target) == ("timeout", 0.0)


def test_current_limit_conversion_matches_recorded_stall_torque():
    from openarm_studio.backend.gripper import GRASP_MIN_LIMIT, STALL_NM_PER_PU, Grasp

    initial = GripperFeedback(-0.8, 0.0, 0.0, 1)
    assert Grasp(0.0, initial, 0.10).nominal_limit == pytest.approx(GRASP_MIN_LIMIT)
    assert Grasp(0.0, initial, 0.50).torque_limit == pytest.approx(0.5 / 4.5 * 9.5, abs=1e-3)
    # 真机：两指互压时力矩恒等于电流上限，斜率 ≈ 9.5 N·m / 标幺值（驱动按 4.5 换算，偏小 2.1 倍）
    rows = _recorded("right_stall_vs_current_limit.csv")
    for pu in (33, 44, 67, 89, 111, 133):
        values = [float(row["tau_nm"]) for row in rows if int(row["pu_x1000"]) == pu]
        tail = values[len(values) // 2:]       # 前半段还在从上一档的力矩过渡
        assert tail and max(tail) - min(tail) < 0.08      # 半秒以上波动不超过 ±0.04 N·m
        assert sum(tail) / len(tail) == pytest.approx(STALL_NM_PER_PU * pu / 1000, rel=0.05)


def test_left_gripper_config_follows_the_right_hand_direction():
    config_module = pytest.importorskip("openarm_driver.config")
    from openarm_studio.backend.driver_arm import _fix_left_gripper_config

    cfg = config_module.Config("openarm_pedestal")
    assert cfg.get_joint_limits("left_arm")[7] == pytest.approx([-0.4, 1.0472])  # 官方按镜像写
    _fix_left_gripper_config(cfg)
    assert cfg.get_joint_limits("left_arm")[7] == pytest.approx([-1.0472, 0.4])
    for section in ("start", "stop"):
        for move in getattr(cfg, f"get_{section}_config")()["moves"]:
            assert move["position"]["left_arm"][7] == move["position"]["right_arm"][7]
    assert cfg.get_start_config()["moves"][0]["position"]["left_arm"][7] == pytest.approx(-0.4)
    assert cfg.get_stop_config()["moves"][0]["position"]["left_arm"][7] == pytest.approx(0.0)

def test_emergency_stop_preempts_full_motion_queue():
    from types import SimpleNamespace

    arm = DriverArm("right", "unused")
    arm._enabled = True
    outcomes, events = [], []
    for _ in range(64):
        assert arm.move_start(done=outcomes.append)
    arm.set_targets({"openarm_right_joint1": 1.0})
    drv = SimpleNamespace(openarm=SimpleNamespace(disable_all=lambda: events.append("disable")))

    assert arm.estop()
    assert arm._drain_cmds(drv)
    assert events == ["disable"] and outcomes == [False] * 64
    assert not arm.enabled and arm._pending_targets is None


def test_emergency_stop_interrupts_position_segment_and_cancels_queued_smooth_move():
    from types import SimpleNamespace

    arm = DriverArm("right", "unused")
    arm._enabled = True
    events, outcomes = [], []
    assert arm.smooth_move({"openarm_right_joint1": 0.5}, done=outcomes.append)
    assert arm.smooth_move({"openarm_right_joint1": 0.7}, done=outcomes.append)

    def send_position(_):
        events.append("position")
        arm.estop()
        return True

    drv = SimpleNamespace(
        last_command=np.zeros(8), latest_state={"qpos": np.zeros(8)},
        safety_stop_reason=None, send_position=send_position,
        openarm=SimpleNamespace(recv_all=lambda _: None, disable_all=lambda: events.append("disable")),
    )
    assert arm._drain_cmds(drv)
    assert events == ["position", "disable"]
    assert outcomes == [False, False] and not arm.enabled and arm._cmds.empty()


def test_queued_force_modes_use_latest_selection_and_measured_pose():
    from types import SimpleNamespace

    arm = DriverArm("right", "unused")
    arm._enabled = arm._grav_on = True
    arm._positions = {"openarm_right_joint1": 0.2, "openarm_right_finger_joint1": 0.02}
    drv = SimpleNamespace(last_command=np.zeros(8))
    arm.set_control_mode("position")
    arm.set_control_mode("impedance", stiffness=0.2)
    assert arm._grav_on and not arm._imp_on

    arm._drain_cmds(drv)
    assert not arm._grav_on and arm._imp_on and arm._imp_stiff == pytest.approx(0.2)
    assert arm._grav_hold_q[0] == pytest.approx(0.2)
    assert arm._targets[-1] < 0


def test_live_commands_are_latest_only_and_playback_timeout_holds_feedback():
    from types import SimpleNamespace

    arm = DriverArm("right", "unused")
    arm._enabled = arm._grav_on = True
    arm._positions = {"openarm_right_joint1": 0.2}
    drv = SimpleNamespace(last_command=np.zeros(8))
    for value in (0.3, 0.6, 0.9):
        arm.set_targets({"openarm_right_joint1": value}, manual=False)
    assert arm._cmds.empty()
    arm._apply_live_targets(drv, 1.0)
    assert arm._targets[0] == pytest.approx(0.9)
    arm._apply_live_targets(drv, 1.16)
    assert arm._targets[0] == pytest.approx(0.2)
    assert arm._manual_target



@pytest.mark.parametrize("mirror", [True, False])
def test_teleop_signs_match_bundled_urdf_axes(mirror):
    import xml.etree.ElementTree as ET

    from openarm_studio.backend.teleop import BimanualTeleop, JointFeedback

    rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    urdf = ROOT / "openarm_studio/model/assets/openarm_v10_bimanual.urdf"
    joints = ET.parse(urdf).getroot().findall("joint")
    by_child = {joint.find("child").get("link"): joint for joint in joints}
    by_name = {joint.get("name"): joint for joint in joints}

    def zero_rotation(link):
        joint = by_child.get(link)
        if joint is None:
            return np.eye(3)
        origin = joint.find("origin")
        angles = tuple(map(float, origin.get("rpy", "0 0 0").split()))
        return zero_rotation(joint.find("parent").get("link")) @ rotation.from_euler(
            "xyz", angles).as_matrix()

    def joint_axis(side, index):
        joint = by_name[f"openarm_{side}_joint{index}"]
        axis = tuple(map(float, joint.find("axis").get("xyz").split()))
        return zero_rotation(joint.find("child").get("link")) @ axis

    origin = (0.0,) * 7
    leader = JointFeedback((0.1,) * 7, origin, 1.0)
    follower = JointFeedback(origin, origin, 1.0)
    _, command = BimanualTeleop("left", origin, origin, mirror).commands(leader, follower)
    for index, signed_angle in enumerate(command.joints, 1):
        left_axis = joint_axis("left", index)
        if mirror:
            # 角速度是轴向量，关于双臂中间的 XZ 平面镜像时乘 det(R)·R。
            left_axis *= (-1, 1, -1)
        right_axis = joint_axis("right", index)
        assert signed_angle * right_axis == pytest.approx(0.1 * left_axis, abs=1e-6)
