#!/usr/bin/env python3
"""诊断：我们的客户端能否看到 controller_manager / move_action。"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rclpy  # noqa: E402

from openarm_studio.ros.bridge import RosBridge  # noqa: E402
from openarm_studio.ros.control import ControlClient  # noqa: E402
from openarm_studio.ros.moveit_client import MoveItClient  # noqa: E402


def main() -> int:
    br = RosBridge(on_log=lambda m: print("  bridge:", m))
    if not br.start():
        print("bridge 启动失败")
        return 1
    node = br._node  # noqa: SLF001
    ctl = ControlClient(node, on_log=lambda m: print("  ctl:", m))
    mc = MoveItClient(node, on_log=lambda m: print("  mc:", m))

    for i in range(30):
        names = [n for n, t in node.get_service_names_and_types()
                 if "controller_manager" in n or "move_action" in n][:6]
        print(f"[{i:02d}] services={names} "
              f"sw_ready={ctl._sw.service_is_ready()} "
              f"list_ready={ctl._list.service_is_ready()} "
              f"move_ready={mc._client.server_is_ready()}")
        if ctl.managers_ready() or mc._client.server_is_ready():  # noqa: SLF001
            rows = ctl.list_controllers()
            print("controllers:", [(r[0], r[2]) for r in rows][:6])
            break
        time.sleep(1.0)

    br.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
