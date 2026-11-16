"""真机控制面：控制器切换、夹爪、急停、关节指令。

契约（与 openarm_bringup 的控制器配置一致）：
- 轨迹控制：/left|right_joint_trajectory_controller/follow_joint_trajectory
- 位置流：  /left|right_forward_position_controller/commands (Float64MultiArray)
- 夹爪：    /left|right_gripper_controller/gripper_cmd (GripperCommand action)
- 控制器管理：/controller_manager/switch_controller
"""

from __future__ import annotations

import threading

from control_msgs.action import GripperCommand
from controller_manager_msgs.srv import ListControllers, SwitchController
from std_msgs.msg import Float64MultiArray

TRAJ = ("left_joint_trajectory_controller", "right_joint_trajectory_controller")
FWD = ("left_forward_position_controller", "right_forward_position_controller")
GRIP = ("left_gripper_controller", "right_gripper_controller")

GRIP_OPEN = 0.044
GRIP_CLOSED = 0.0
GRIP_EFFORT = 40.0


class ControlClient:
    def __init__(self, node, on_log=None):
        from rclpy.action import ActionClient

        self._node = node
        self._log = on_log or (lambda *_: None)
        self._sw = node.create_client(SwitchController,
                                      "/controller_manager/switch_controller")
        self._list = node.create_client(ListControllers,
                                        "/controller_manager/list_controllers")
        self._fwd_pub = {
            "left": node.create_publisher(
                Float64MultiArray, "/left_forward_position_controller/commands", 5),
            "right": node.create_publisher(
                Float64MultiArray, "/right_forward_position_controller/commands", 5),
        }
        self._grip = {
            "left": ActionClient(node, GripperCommand,
                                 "/left_gripper_controller/gripper_cmd"),
            "right": ActionClient(node, GripperCommand,
                                  "/right_gripper_controller/gripper_cmd"),
        }
        self._lock = threading.Lock()

    # ── 查询 ─────────────────────────────────────────
    def managers_ready(self) -> bool:
        return self._sw.service_is_ready() and self._list.service_is_ready()

    def list_controllers(self, timeout: float = 0.6) -> list[tuple[str, str, str]]:
        """返回 [(name, type, state)]，失败返回空表。"""
        if not self._list.service_is_ready():
            return []
        fut = self._list.call_async(ListControllers.Request())
        t0 = threading.Event()
        fut.add_done_callback(lambda _f: t0.set())
        t0.wait(timeout)
        if not fut.done() or fut.result() is None:
            return []
        return [(c.name, c.type, c.state) for c in fut.result().controller]

    # ── 控制器切换 ───────────────────────────────────
    def switch(self, start: list[str], stop: list[str],
               strict: bool = False, timeout: float = 2.0) -> bool:
        if not self._sw.service_is_ready():
            self._log("controller_manager 不可用（真机栈未启动？）")
            return False
        req = SwitchController.Request()
        req.start_controllers = start
        req.stop_controllers = stop
        req.strictness = (SwitchController.Request.STRICT if strict
                          else SwitchController.Request.BEST_EFFORT)
        req.start_asap = False
        req.timeout = float(timeout)
        fut = self._sw.call_async(req)
        done = threading.Event()
        fut.add_done_callback(lambda _f: done.set())
        done.wait(timeout + 1.0)
        ok = bool(fut.done() and fut.result() is not None and fut.result().ok)
        self._log(("控制器切换成功：" if ok else "控制器切换失败：")
                  + f"start={start} stop={stop}")
        return ok

    def use_trajectory(self) -> bool:
        return self.switch(list(TRAJ), list(FWD))

    def use_forward(self) -> bool:
        return self.switch(list(FWD), list(TRAJ))

    def emergency_stop(self) -> bool:
        """急停：停掉全部运动控制器 → 硬件 on_deactivate → disable_all。"""
        stop = list(TRAJ) + list(FWD) + list(GRIP)
        return self.switch([], stop, strict=False, timeout=1.0)

    def enable_all(self) -> bool:
        return self.switch(list(TRAJ) + list(GRIP), [])

    # ── 位置流下发（直接控制，需先 use_forward）────────────────
    def send_positions(self, side: str, joints: list[float]) -> None:
        msg = Float64MultiArray()
        msg.data = [float(v) for v in joints]
        self._fwd_pub[side].publish(msg)

    # ── 夹爪 ─────────────────────────────────────────
    def gripper(self, side: str, position: float,
                effort: float = GRIP_EFFORT, done_cb=None) -> bool:
        cli = self._grip[side]
        if not cli.server_is_ready():
            self._log(f"{side} 夹爪 action 不可用")
            if done_cb:
                done_cb(False, "夹爪 action 不可用")
            return False
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(effort)
        fut = cli.send_goal_async(goal)
        if done_cb:
            def _res(rf):
                res = rf.result()
                ok = res is not None and res.status == 4   # STATUS_SUCCEEDED
                done_cb(ok, f"{side} 夹爪 {'完成' if ok else '失败'}"
                            f"（目标 {position:.3f} m）")
            fut.add_done_callback(
                lambda f: f.result().get_result_async().add_done_callback(_res)
                if f.result() is not None else None)
        self._log(f"{side} 夹爪 → {position:.3f} m")
        return True
