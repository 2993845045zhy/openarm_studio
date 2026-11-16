#!/usr/bin/env python3
"""UI 自检：模拟点「连接」→ 等就绪 → 点「规划」，打印全过程。"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from openarm_studio.ui.shell import StudioWindow  # noqa: E402

app = QApplication(sys.argv)
win = StudioWindow(theme_name="dark")
win.sw_mode.setCurrentIndex(1, animate=False)   # 仿真
win.show()

state = {"step": "connect", "t0": time.time()}


def tick() -> None:
    el = time.time() - state["t0"]
    if state["step"] == "connect":
        print(f"[{el:5.1f}s] 点连接（conn_state={win._conn_state}）")
        win._on_connect_toggle()
        state["step"] = "wait"
        state["t0"] = time.time()
        return
    if state["step"] == "wait":
        if win._connected:
            print(f"[{el:5.1f}s] 已连接（耗时 {el:.1f}s）")
            win.combo_group.setCurrentText("left_arm")
            for i, (s, _v) in enumerate(win._goal_sliders):
                s.setValue([30, 20, 0, 100, 0, 20, 0][i])
            print("[plan] goal =", win._goal_values())
            win._do_plan(execute=False)
            state["step"] = "plan"
            state["t0"] = time.time()
        elif el > 60:
            print(f"[{el:5.1f}s] 连接超时，状态={win._conn_state}")
            print("日志尾部：", win.log_view.toPlainText().splitlines()[-4:])
            win.close(); app.quit()
        return
    if state["step"] == "plan":
        if el > 25:
            tail = win.log_view.toPlainText().splitlines()[-4:]
            print("[plan] 日志尾部：")
            for line in tail:
                print("   ", line)
            win.close(); app.quit()


timer = QTimer()
timer.timeout.connect(tick)
timer.start(400)
app.exec()
