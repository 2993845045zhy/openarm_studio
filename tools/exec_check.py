#!/usr/bin/env python3
"""e2e：规划 + 执行一条轨迹，并用 /joint_states 确认关节真的到位。

假硬件下执行是"指令直接到位"，但走的仍是 MoveIt→控制器→硬件的完整链路。
最后把关节留在 home，便于重复测试。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from openarm_studio.ros.moveit_client import MoveItClient  # noqa: E402

TARGET = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]     # left_arm hands_up
NAMES = [f"openarm_left_joint{i}" for i in range(1, 8)]


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("exec_probe")
    latest = {}

    def cb(msg):
        latest.update({n: p for n, p in zip(msg.name, msg.position)})

    node.create_subscription(JointState, "/joint_states", cb, 20)
    mc = MoveItClient(node, on_log=lambda m: print("  log:", m))
    if not mc.ready(timeout=25.0):
        print("/move_action 未就绪")
        node.destroy_node(); rclpy.shutdown(); return 1

    t0 = time.time()
    while time.time() - t0 < 5:
        rclpy.spin_once(node, timeout_sec=0.2)
    print("before:", {n.replace("openarm_left_", ""): round(v, 3)
                      for n, v in latest.items() if n in NAMES})

    res = {}

    def done(ok, text):
        res["ok"], res["text"] = ok, text

    mc.plan_joint("left_arm", TARGET, done, execute=True, vel=0.3, acc=0.3)
    t0 = time.time()
    while time.time() - t0 < 40 and "ok" not in res:
        rclpy.spin_once(node, timeout_sec=0.2)
    print("execute:", res.get("ok"), "|", res.get("text"))

    # 等轨迹跑完（假硬件即时，但控制器有插值）
    t0 = time.time()
    while time.time() - t0 < 6:
        rclpy.spin_once(node, timeout_sec=0.2)
    err = max(abs(latest.get(n, 99) - t) for n, t in zip(NAMES, TARGET))
    print("final:", {n.replace("openarm_left_", ""): round(latest.get(n, 0), 3)
                     for n in NAMES})
    print(f"max |target-actual| = {err:.4f} rad")

    # 回 home，保持可重复
    res.clear()
    mc.plan_named("left_arm", "home", done, execute=True, vel=0.3, acc=0.3)
    t0 = time.time()
    while time.time() - t0 < 30 and "ok" not in res:
        rclpy.spin_once(node, timeout_sec=0.2)
    print("reset home:", res.get("ok"))

    node.destroy_node(); rclpy.shutdown()
    return 0 if (res.get("ok") and err < 0.15) else 1


if __name__ == "__main__":
    raise SystemExit(main())
