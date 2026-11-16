#!/usr/bin/env python3
"""端到端验证：栈启动 → 控制器查询 → MoveIt 规划（不执行）。

用假硬件栈（use_fake_hardware:=true），不碰真电机。
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rclpy  # noqa: E402

from openarm_studio.ros.control import ControlClient  # noqa: E402
from openarm_studio.ros.moveit_client import MoveItClient  # noqa: E402
from openarm_studio.ros.stack import StackManager  # noqa: E402


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("e2e_probe")
    logs = []
    stack = StackManager(on_log=logs.append)
    ctl = ControlClient(node, on_log=logs.append)
    mc = MoveItClient(node, on_log=logs.append)

    print("probe before start:", stack.probe())
    stack.start(fake=True)

    deadline = time.time() + 60
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        if ctl.managers_ready():
            rows = ctl.list_controllers()
            active = [n for n, _t, s in rows if s == "active"]
            if active:
                print("controllers active:", active)
                break
    else:
        print("TIMEOUT waiting for controllers")
        print("logs:", logs[-3:])
        stack.stop()
        rclpy.shutdown()
        return 1

    # MoveIt 规划（plan_only）
    ok_server = mc.ready(timeout=15.0)
    print("/move_action ready:", ok_server)
    result = {}

    def done(ok: bool, text: str) -> None:
        result["ok"] = ok
        result["text"] = text

    mc.plan_named("left_arm", "hands_up", done, execute=False, vel=0.1, acc=0.1)
    t0 = time.time()
    while time.time() - t0 < 30 and "ok" not in result:
        rclpy.spin_once(node, timeout_sec=0.2)
    print("plan result:", result.get("ok"), "|", result.get("text"))

    stack.stop()
    node.destroy_node()
    rclpy.shutdown()
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
