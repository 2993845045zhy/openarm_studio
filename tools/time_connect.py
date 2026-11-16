#!/usr/bin/env python3
"""测连接耗时：起栈到「控制器 active + /move_action 就绪」用了多久。"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rclpy  # noqa: E402

from openarm_studio.ros.bridge import RosBridge  # noqa: E402
from openarm_studio.ros.control import ControlClient  # noqa: E402
from openarm_studio.ros.moveit_client import MoveItClient  # noqa: E402

LAUNCH = ROOT / "tools/launch_real.sh"


def main() -> int:
    br = RosBridge()
    br.start()
    node = br._node  # noqa: SLF001
    ctl = ControlClient(node)
    mc = MoveItClient(node)

    t0 = time.time()
    proc = subprocess.Popen(["bash", str(LAUNCH)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True, shell=False)

    ctl_t = mi_t = None
    while time.time() - t0 < 120:
        if ctl_t is None:
            rows = ctl.list_controllers(0.4)
            act = [r for r in rows if r[2] == "active"]
            if len(act) >= 5:
                ctl_t = time.time() - t0
                print(f"controllers active: {ctl_t:.1f}s ({len(act)} 个)")
        if mi_t is None and mc.ready(timeout=0.3):
            mi_t = time.time() - t0
            print(f"/move_action ready: {mi_t:.1f}s")
        if ctl_t is not None and mi_t is not None:
            print(f"=== 可连接总耗时: {max(ctl_t, mi_t):.1f}s ===")
            break
        time.sleep(0.3)
    else:
        print(f"超时：ctl={ctl_t} moveit={mi_t}")

    # 收尾
    try:
        subprocess.run(["pkill", "-INT", "-f", "openarm.bimanual.launch.py"],
                       timeout=5, shell=False)
        subprocess.run(["pkill", "-INT", "-f", "move_group.launch.py"],
                       timeout=5, shell=False)
    except Exception:  # noqa: BLE001
        pass
    br.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
