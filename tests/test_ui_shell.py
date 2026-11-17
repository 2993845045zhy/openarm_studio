#!/usr/bin/env python3
"""UI 行为测试：导航索引 / 全部回零 / 分段滑块几何 / 悬浮胶囊。

离屏（offscreen platform）跑 StudioWindow 的真实构建路径，
不连硬件、不依赖 X（Qt offscreen + EGL 渲染线程照常工作）。
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QPushButton
from PySide6.QtTest import QTest

from openarm_studio.ui.shell import StudioWindow
from openarm_studio.ui.switch import ModeSwitch


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture()
def win(app):
    w = StudioWindow(theme_name="dark")
    yield w
    w.close()
    QTest.qWait(50)


def _nav_item(win, label):
    for item in win.nav_items:
        if item._label == label:  # noqa: SLF001
            return item
    return None


def test_nav_pages_in_range(win):
    """每个导航行的 page_index 必须落在页面栈里（日志页越界是空壳根因）。"""
    n = win.pages.count()
    assert n == 5, f"应为 5 页（概览/机器人/场景/日志/示教），实际 {n}"
    for item in win.nav_items:
        assert 0 <= item.page_index < n, f"{item._label} 越界"  # noqa: SLF001


def test_log_nav_switches_page(win, app):
    """点「日志」必须真的切到日志页且能看到日志内容。"""
    QTest.mouseClick(_nav_item(win, "日志"), Qt.LeftButton)
    assert win.pages.currentIndex() == 3
    assert win.log_view.toPlainText().strip(), "日志页应有启动期内容"


def test_zero_pose_moves_twin(win, app):
    """滑块抬高关节后，全部回零必须把孪生姿态清回 0（旧版 pop 导致不动）。"""
    win._on_slider("openarm_left_joint2", 120)   # 1.2 rad
    win._on_slider("openarm_left_finger_joint1", 30)
    assert win._local_pose["openarm_left_joint2"] == pytest.approx(1.2)
    QTest.mouseClick(win.btn_zero, Qt.LeftButton)
    for i in range(1, 8):
        assert win._local_pose[f"openarm_left_joint{i}"] == 0.0
        assert win._local_pose[f"openarm_right_joint{i}"] == 0.0
    assert win._local_pose["openarm_left_finger_joint1"] == 0.0
    # 孪生姿态推给渲染线程后，滑块标签也要归零显示
    name = "openarm_left_joint2"
    sld, val = win._sliders[name]
    assert sld.value() == 0
    assert val.text() == "+0.000"
    _, gval = win._sliders["openarm_left_finger_joint1"]
    assert gval.text() == "0 mm"


def test_grip_sliders_both_sides(win):
    """夹爪滑块左右各一，名字齐全。"""
    for side in ("left", "right"):
        assert f"openarm_{side}_finger_joint1" in win._sliders
    assert len([n for n in win._sliders if "finger" in n]) == 2


def test_arm_switch_keeps_grip_sliders(win, app):
    """切换左右臂只重建关节滑块，夹爪滑块保留。"""
    QTest.keyClicks(win.combo_arm, "")
    win.combo_arm.setCurrentIndex(1)
    QTest.qWait(30)
    assert f"openarm_right_joint7" in win._sliders
    assert f"openarm_left_joint1" not in win._sliders
    assert "openarm_left_finger_joint1" in win._sliders
    assert "openarm_right_finger_joint1" in win._sliders


def test_mode_switch_geometry(app, win):
    """分段滑块：自然高度、按钮无裁剪、thumb 与选中按钮几何一致、文字居中。"""
    for sw in (win.sw_mode, win.sw_theme):
        btn_h = sw._btns[0].height()
        assert btn_h >= 24, f"按钮过矮 {btn_h}"
        # thumb 覆盖在选中按钮上（允许 1px 误差）
        th, bh = sw._thumb.geometry(), sw._btns[sw._index].geometry()
        assert abs(th.x() - bh.x()) <= 1 and abs(th.y() - bh.y()) <= 1
        assert abs(th.height() - bh.height()) <= 1


def test_no_view_chip_overlay(win):
    """视口左下角的视角胶囊已按用户要求删除；预设改走 Inspector 场景页。"""
    assert not hasattr(win, "view_chips")
    assert not win.view.findChildren(QPushButton, "viewChip")
    # 场景页的按钮仍然有效（相机是阻尼跟随 τ=0.1s，等它收敛）
    win._set_view_preset("top")
    QTest.qWait(600)
    cam = win.view.cam_state()
    assert cam["elevation"] == pytest.approx(-70.0, abs=2.0)
    assert cam["distance"] == pytest.approx(1.9, abs=0.05)


def test_estop_and_enable_guard(win):
    """未连接时使能按钮不可见；急停在无臂时只记日志不崩。"""
    assert not win.btn_enable.isVisible()
    win._on_estop()
    assert "未连接" in win.log_view.toPlainText()


def test_bus_probe_result_updates_rows(win):
    """体检结果回填：在线/无应答/部分在线三态与日志。"""
    win._on_bus_result({
        "can0": {"alive": 7, "axes": [(5, 5)] * 7, "positions": [0.0] * 7},
        "can1": {"alive": 0, "axes": [(0, 5)] * 7, "positions": [0.0] * 7},
    })
    assert win.row_bus_r.val.text() == "在线 7/7"
    assert win.row_bus_l.val.text() == "❌ 0/7 无应答"
    log = win.log_view.toPlainText()
    assert "右臂(can0)" in log and "0/7 应答" in log
    assert not win._probing and win.btn_bus_check.isEnabled()

    win._on_bus_result({
        "can0": {"alive": 5, "axes": [(5, 5)] * 5 + [(0, 5)] * 2,
                 "positions": [0.0] * 7},
        "can1": {"error": "OSError: no such device"},
    })
    assert win.row_bus_r.val.text() == "⚠️ 5/7 部分在线"
    assert "失联轴 ['J6', 'J7']" in win.log_view.toPlainText()
    assert "打不开" in win.row_bus_l.val.text()


class _FakeArm:
    connected = True
    rate = 200.0
    enabled = True
    safety_stop_reason = None

    def disconnect(self):
        pass

    def temps(self):
        return {"openarm_right_joint1": 45, "openarm_right_joint4": 72}

    def rotor_temps(self):
        return {"openarm_right_joint4": 60}

    def max_mos_temp(self):
        return 72

    health = (["OK"] * 8, {"carrier": True, "tx_overflow": 3})


def test_diag_rows_temperature_and_health(win):
    """机器人页温度/健康卡：超温变色、总线错误计数、安全锁存清空。"""
    arm = _FakeArm()
    win.arms["right"] = arm
    win._poll_controllers()
    assert "72°" in win.row_temp_r.val.text() and "60°" in win.row_temp_r.val.text()
    assert "#e07b00" in win.row_temp_r.val.styleSheet()   # ≥60 橙色
    assert "总线 3" in win.row_health_r.val.text()
    assert "72°" in win.ctrl_rows["right"].text()          # 导航行带温度
    assert win.row_safety.val.text() == "—"
    # 超过 80℃ 转红
    arm.temps = lambda: {"openarm_right_joint1": 85}
    arm.rotor_temps = lambda: {"openarm_right_joint1": 70}
    win._poll_controllers()
    assert "#e03131" in win.row_temp_r.val.styleSheet()
    # 安全锁存显示原因
    arm.safety_stop_reason = "Joint 3 delta exceeds limit"
    win._poll_controllers()
    assert "delta" in win.row_safety.val.text()
    del win.arms["right"]


# ── 力控相关 UI ────────────────────────────────────────────
def test_torque_labels_present(win):
    """7 个关节 + 2 个夹爪都要有力矩读数，未连接时显示占位符。"""
    assert len(win._torque_labels) == 9
    assert all(lbl.text() == "--" for lbl in win._torque_labels.values())


def test_torque_labels_fill_and_warn(win):
    """有力矩数据时填数值；超过额定 60% 打上 warn 标记（QSS 据此转橙）。"""

    class _TorqueArm:
        connected = True
        enabled = True
        rate = 0.0
        safety_stop_reason = None
        admittance_active = False
        health = (["OK"] * 8, {})

        def torques(self):
            return {"openarm_left_joint1": 10.0,     # 54Nm 的 18% → 正常
                    "openarm_left_joint5": 8.0,      # 10Nm 的 80% → warn
                    "openarm_left_finger_joint1": 0.2}

        def positions(self):
            return {}

        def temps(self):
            return {}

        def rotor_temps(self):
            return {}

        def max_mos_temp(self):
            return 0

        def disconnect(self):
            pass

    win.arms["left"] = _TorqueArm()
    win._update_torque_labels()
    assert win._torque_labels["openarm_left_joint1"].text() == "+10.0"
    assert win._torque_labels["openarm_left_joint1"].property("warn") is False
    assert win._torque_labels["openarm_left_joint5"].property("warn") is True
    assert win._torque_labels["openarm_left_finger_joint1"].text() == "+0.2"
    del win.arms["left"]


def test_friction_switch_defaults_on(win):
    """摩擦补偿默认开（官方参数已标定，开箱即用）。"""
    assert win.sw_friction.isChecked()


class _CtrlArm:
    """只记录力控调用序列的假臂。"""
    connected = True
    enabled = True
    rate = 0.0
    safety_stop_reason = None
    health = (["OK"] * 8, {})

    def __init__(self):
        self.calls = []

    def positions(self):
        return {}

    def torques(self):
        return {}

    def temps(self):
        return {}

    def rotor_temps(self):
        return {}

    def max_mos_temp(self):
        return 0

    def disconnect(self):
        pass

    def set_gravity_comp(self, on, scale=1.0, kp_floor=30.0):
        self.calls.append(("grav", bool(on)))

    def set_joint_impedance(self, on, stiffness=0.35, zeta=1.0):
        self.calls.append(("jimp", bool(on), round(float(stiffness), 3),
                           round(float(zeta), 3)))

    def set_friction_comp(self, on):
        self.calls.append(("fric", bool(on)))


def test_joint_impedance_requires_connection(win):
    """未连接时开关节阻抗要自动回弹到关，并写日志。"""
    win.sw_jimp.setChecked(True)
    assert not win.sw_jimp.isChecked()
    assert "关节阻抗" in win.log_view.toPlainText()


def test_joint_impedance_mutual_exclusion(win):
    """关节阻抗与重力补偿互斥（同一个软模式帧）。"""
    arm = _CtrlArm()
    win.arms["right"] = arm
    win.sw_gravity.setChecked(True)
    win.sw_jimp.setChecked(True)
    assert win.sw_jimp.isChecked() and not win.sw_gravity.isChecked()
    assert ("jimp", True, 0.35, 1.0) in arm.calls

    win.sw_gravity.setChecked(True)
    assert win.sw_gravity.isChecked() and not win.sw_jimp.isChecked()
    assert ("jimp", False, 0.35, 1.0) in arm.calls
    del win.arms["right"]


def test_joint_impedance_sliders_apply_live(win):
    """刚度/阻尼比滑块变化要立即下发，并且关掉时禁用滑块。"""
    arm = _CtrlArm()
    win.arms["right"] = arm
    win.sw_jimp.setChecked(True)
    assert not win.sld_stiff.isEnabled()
    win.sld_stiff.setValue(10)               # 10%
    win.sld_zeta.setValue(60)                # 0.60
    assert ("jimp", True, 0.10, 0.60) in arm.calls
    assert win.lbl_stiff.text() == "10%"
    assert win.lbl_zeta.text() == "0.60"
    win.sw_jimp.setChecked(False)
    assert win.sld_stiff.isEnabled() and win.sld_zeta.isEnabled()
    del win.arms["right"]


# ── 示教录制 / 回放 ────────────────────────────────────────
def test_teach_page_has_record_button(win):
    """录制必须是显式按钮（在「示教」页），不是开关状态。"""
    assert win.btn_record.text() == "开始录制"
    assert win.lbl_rec.text().startswith("空闲")
    # 录制按钮不在状态列里
    from openarm_studio.ui.widgets import GroupCard
    titles = [c.lbl_title.text() for c in win.findChildren(GroupCard)]
    assert "示教录制" in titles


def test_record_requires_connection(win):
    win._on_record_toggle()                    # 未连接
    assert not win._recording
    assert win.btn_record.text() == "开始录制"
    assert "需要先连接" in win.log_view.toPlainText()


def test_record_toggle_and_save(win, tmp_path, monkeypatch):
    """开始→采样→停止：写 CSV 并把文件列表刷新出来。"""
    monkeypatch.setattr(win, "_teach_dir", lambda: tmp_path)
    win.arms["right"] = _CtrlArm()             # 只需 connected 判定
    win._start_record()
    assert win._recording and win.btn_record.text() == "停止并保存"
    win._rec_traj = [_teach_row(0.0, 0.1) + [0.02, 0.03]]
    win._stop_record()
    assert not win._recording and win.btn_record.text() == "开始录制"
    files = list(tmp_path.glob("teach_*.csv"))
    assert len(files) == 1
    assert len(win._load_teach_csv(files[0])[0]) == 17
    del win.arms["right"]


def test_teach_toggle_no_longer_records(win):
    """零重力开关只管变软，不再隐式录制。"""
    arm = _CtrlArm()
    win.arms["right"] = arm
    win.sw_teach.setChecked(True)
    assert not win._recording
    win.sw_teach.setChecked(False)


def _teach_row(t, joint1, n=15, grip=None):
    row = [t] + [0.0] * 14
    row[1] = joint1
    if grip is not None:
        row = row + list(grip)
    return row


def test_teach_file_list_refreshes(win, tmp_path, monkeypatch):
    (tmp_path / "teach_a.csv").write_text("t\n0\n")
    (tmp_path / "teach_b.csv").write_text("t\n0\n")
    monkeypatch.setattr(win, "_teach_dir", lambda: tmp_path)
    win._refresh_teach_files()
    names = [win.combo_teach.itemText(i) for i in range(win.combo_teach.count())]
    assert names == ["teach_b.csv", "teach_a.csv"]      # 新的在前
    assert win.btn_teach_play.isEnabled()


def test_teach_csv_parses_old_and_new_format(win, tmp_path):
    def _line(vals):
        return ",".join(str(v) for v in vals)

    old = tmp_path / "old.csv"
    old.write_text("t\n" + _line(_teach_row(0.0, 0.1))
                   + "\n" + _line(_teach_row(0.02, 0.2)) + "\n")
    rows = win._load_teach_csv(old)
    assert len(rows) == 2 and len(rows[0]) == 15

    new = tmp_path / "new.csv"
    new.write_text("t\n" + _line(_teach_row(0.0, 0.1) + [0.03, 0.04]) + "\n")
    rows = win._load_teach_csv(new)
    assert len(rows) == 1 and len(rows[0]) == 17


def test_apply_teach_frame_maps_joints_and_gripper(win):
    win._local_pose.clear()
    frame = _teach_row(0.0, 0.5) + [0.031, 0.012]
    win._apply_teach_frame(frame)
    assert win._local_pose["openarm_left_joint1"] == pytest.approx(0.5)
    assert win._local_pose["openarm_left_finger_joint1"] == pytest.approx(0.031)
    assert win._local_pose["openarm_right_finger_joint1"] == pytest.approx(0.012)
    # 旧格式（无夹爪列）不能抛异常，且不动夹爪
    win._local_pose.pop("openarm_left_finger_joint1")
    win._apply_teach_frame(_teach_row(0.0, 0.9))
    assert win._local_pose["openarm_left_joint1"] == pytest.approx(0.9)
    assert "openarm_left_finger_joint1" not in win._local_pose


def test_playback_advances_then_stops_at_end(win):
    """播到末帧自动停，并把最后一帧留在目标里（保持姿态，不失能）。"""
    win._play_rows = [_teach_row(0.0, 0.1), _teach_row(0.2, 0.4)]
    win._play_i = 0
    win._play_dur = 0.2
    win._local_pose.clear()
    win._play_timer.start()
    win._play_t0 = time.monotonic() - 1.0      # 已经超过总时长
    win._play_tick()
    assert not win._play_timer.isActive()
    assert win.btn_teach_play.text() == "播放"
    assert win._local_pose["openarm_left_joint1"] == pytest.approx(0.4)


def test_playback_interpolates_between_frames(win):
    win._play_rows = [_teach_row(0.0, 0.0), _teach_row(1.0, 1.0)]
    win._play_i = 0
    win._play_dur = 1.0
    win._local_pose.clear()
    win._play_t0 = time.monotonic() - 0.5      # 中段
    win._play_tick()
    v = win._local_pose["openarm_left_joint1"]
    assert 0.2 < v < 0.9, f"应插值到中间值，得到 {v}"
    win._play_timer.stop()
