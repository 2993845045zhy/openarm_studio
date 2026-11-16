#!/usr/bin/env python3
"""只验证 MoveItClient：连上 move_group 规划一条轨迹（不执行）。"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rclpy  # noqa: E402

from openarm_studio.ros.moveit_client import MoveItClient  # noqa: E402


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("plan_probe")
    mc = MoveItClient(node, on_log=lambda m: print("  log:", m))
    print("named states:", {g: sorted(v.keys()) for g, v in mc.named_states.items()})

    ok = mc.ready(timeout=25.0)
    print("/move_action ready:", ok)
    if not ok:
        node.destroy_node()
        rclpy.shutdown()
        return 1

    for group, state in (("left_arm", "hands_up"), ("right_arm", "home")):
        res = {}

        def done(ok_: bool, text: str) -> None:
            res["ok"], res["text"] = ok_, text

        mc.plan_named(group, state, done, execute=False, vel=0.2, acc=0.2)
        t0 = time.time()
        while time.time() - t0 < 25 and "ok" not in res:
            rclpy.spin_once(node, timeout_sec=0.2)
        print(f"{group} -> {state}: {res.get('ok')} | {res.get('text')}")

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
