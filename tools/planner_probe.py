#!/usr/bin/env python3
"""定位规划失败：对比不同 planner_id 的规划结果。"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rclpy  # noqa: E402

from openarm_studio.ros.moveit_client import MoveItClient  # noqa: E402

GOAL = [0.3, 0.2, 0.0, 1.0, 0.0, 0.2, 0.0]


def try_plan(mc, node, planner: str) -> None:
    res = {}

    def done(ok, text):
        res["ok"], res["text"] = ok, text

    mc.plan_joint("left_arm", GOAL, done, execute=False, vel=0.2, acc=0.2,
                  planner_id=planner, planning_time=5.0, attempts=5)
    t0 = time.time()
    while time.time() - t0 < 20 and "ok" not in res:
        rclpy.spin_once(node, timeout_sec=0.2)
    print(f"planner_id={planner!r:34s} -> {res.get('ok')} | {res.get('text')}")


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("planner_probe")
    mc = MoveItClient(node, on_log=lambda m: print("  log:", m))
    if not mc.ready(timeout=20.0):
        print("/move_action 未就绪")
        return 1
    for pid in ("", "RRTConnect", "geometric::RRTConnect"):
        try_plan(mc, node, pid)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
