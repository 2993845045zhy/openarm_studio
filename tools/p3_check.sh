#!/usr/bin/env bash
# P3 验证：仿真模式是否真的在发 /joint_states（50Hz 左右）。
set -e
cd "$(dirname "$0")/.."

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
[ -f "$HOME/openarm_ws/install/setup.bash" ] && source "$HOME/openarm_ws/install/setup.bash"

PY="$HOME/anaconda3/envs/ros2ws/bin/python"

MUJOCO_GL=egl DISPLAY="${DISPLAY:-:1}" PYTHONPATH="$PWD:$PYTHONPATH" \
  "$PY" -m openarm_studio.app --mode sim > /tmp/p3_app.log 2>&1 &
APP=$!
trap 'kill $APP 2>/dev/null || true' EXIT

sleep 6

"$PY" - <<'PY'
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

rclpy.init()
node = rclpy.create_node("p3_probe")
stats = {"n": 0, "names": 0, "j4": None, "t0": time.time()}


def cb(msg):
    stats["n"] += 1
    stats["names"] = len(msg.name)
    idx = msg.name.index("openarm_left_joint4") if "openarm_left_joint4" in msg.name else None
    if idx is not None:
        stats["j4"] = msg.position[idx]


node.create_subscription(JointState, "/joint_states", cb, 20)
t0 = time.time()
while time.time() - t0 < 3.0:
    rclpy.spin_once(node, timeout_sec=0.1)
rate = stats["n"] / (time.time() - t0)
print(f"messages={stats['n']} rate={rate:.1f}Hz joints={stats['names']} left_j4={stats['j4']}")
print("P3:", "PASS" if stats["n"] > 50 and stats["names"] == 16 else "FAIL")
node.destroy_node()
rclpy.shutdown()
PY

kill $APP 2>/dev/null || true
