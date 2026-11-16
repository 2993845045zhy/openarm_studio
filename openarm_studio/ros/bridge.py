"""ROS 2 桥：订阅 /joint_states，把真机关节角喂给孪生。

设计取舍：
- 节点在 GUI 线程创建、在独立线程 spin —— rclpy 的 executor 不绑定线程，
  只要同一个进程里 init/shutdown 成对出现即可。
- 这里**只读**（订阅），不发布任何指令，所以可以在任何时候安全启动。
- 指令通道（控制器 / MoveIt）放在后面的 moveit_client / control 模块，
  与这个只读桥分开，避免"看一眼状态就把机器人动了"。
"""

from __future__ import annotations

import threading
import time

from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINT_TOPIC = "/joint_states"


class JointStatePublisher:
    """仿真模式下由上位机自己发 /joint_states（冒充 joint_state_broadcaster）。

    只在仿真模式使用：真机模式里由 ros2_control 的 broadcaster 发布，
    两边同时发会让 MoveIt/RViz 收到互相打架的状态。
    """

    def __init__(self, node: Node):
        self._pub = node.create_publisher(JointState, JOINT_TOPIC, 10)
        self._node = node

    def publish(self, positions: dict[str, float]) -> None:
        if not positions:
            return
        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(positions.keys())
        msg.position = [float(v) for v in positions.values()]
        self._pub.publish(msg)


class RosBridge:
    """持有 rclpy 节点与订阅，线程安全地对外暴露最新关节角。"""

    def __init__(self, on_log=None):
        self._log = on_log or (lambda *_: None)
        self._lock = threading.Lock()
        self._positions: dict[str, float] = {}
        self._stamps: dict[str, float] = {}
        self._last_rx = 0.0
        self._count = 0
        self._rate = 0.0
        self._rate_t0 = time.monotonic()
        self._rate_n = 0
        self._connected = False
        self._ok = False
        self._error: str | None = None

        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._thread: threading.Thread | None = None

    # ── 生命周期 ──────────────────────────────────────
    def start(self) -> bool:
        if self._ok:
            return True
        try:
            import rclpy

            if not rclpy.ok():
                rclpy.init(args=None)
            self._node = rclpy.create_node("openarm_studio_bridge")
            self._node.create_subscription(JointState, JOINT_TOPIC,
                                           self._on_joint_state, 20)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._thread = threading.Thread(target=self._spin, daemon=True,
                                            name="openarm-ros")
            self._thread.start()
            self._ok = True
            self._log(f"ROS 2 桥已启动，订阅 {JOINT_TOPIC}")
            return True
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"
            self._log(f"ROS 2 桥启动失败：{self._error}")
            return False

    def _spin(self) -> None:
        try:
            assert self._executor is not None
            self._executor.spin()
        except Exception as exc:  # noqa: BLE001
            self._log(f"ROS 2 桥 spin 结束：{type(exc).__name__}: {exc}")

    def stop(self) -> None:
        try:
            if self._executor is not None:
                self._executor.shutdown()
            if self._node is not None:
                self._node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        self._node = None
        self._executor = None
        self._ok = False

    # ── 回调 ─────────────────────────────────────────
    def _on_joint_state(self, msg: JointState) -> None:
        now = time.monotonic()
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self._positions[name] = float(pos)
            self._last_rx = now
            self._count += 1
            self._rate_n += 1
            if now - self._rate_t0 >= 1.0:
                self._rate = self._rate_n / (now - self._rate_t0)
                self._rate_t0 = now
                self._rate_n = 0
            self._connected = True

    # ── 查询 ─────────────────────────────────────────
    def positions(self) -> dict[str, float]:
        with self._lock:
            return dict(self._positions)

    @property
    def connected(self) -> bool:
        with self._lock:
            # 1 秒没新消息就认为断开（broadcaster 正常 50Hz）
            return self._connected and (time.monotonic() - self._last_rx) < 1.0

    @property
    def rate(self) -> float:
        with self._lock:
            return self._rate

    @property
    def message_count(self) -> int:
        with self._lock:
            return self._count
