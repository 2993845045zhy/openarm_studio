#!/usr/bin/env python3
"""测试用：往 /joint_states 发布一组假关节角（验证孪生跟随）。

不碰硬件、不发布任何指令。
"""
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

NAMES = ([f"openarm_left_joint{i}" for i in range(1, 8)] + ["openarm_left_finger_joint1"] +
         [f"openarm_right_joint{i}" for i in range(1, 8)] + ["openarm_right_finger_joint1"])


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("fake_joint_states")
    pub = node.create_publisher(JointState, "/joint_states", 10)

    def pose(t: float) -> list[float]:
        # 缓慢摆动的姿势，肉眼可辨
        swing = 0.45 * (1.0 - abs(((t % 6.0) / 3.0) - 1.0))
        out = []
        for i in range(7):
            out.append(swing if i == 0 else 0.0)
        out.append(0.044)                       # 左夹爪张开
        for i in range(7):
            out.append(-swing if i == 0 else (1.1 if i == 3 else 0.0))
        out.append(0.0)                         # 右夹爪闭合
        return out

    t0 = time.time()
    try:
        while True:
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.name = NAMES
            msg.position = [float(v) for v in pose(time.time() - t0)]
            pub.publish(msg)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
