"""MoveIt 2 客户端：规划 / 执行 / 命名状态 / 关节目标 / 位姿目标。

走 MoveGroup action（/move_action），plan_only 做预览、否则规划+执行。
不做 moveit_py 进程内规划：move_group 已经在跑，复用它的规划器与
碰撞场景才是真机同一条链路。
"""

from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup

# 规划组（SRDF 里就这四个）
GROUPS = ("left_arm", "right_arm", "left_gripper", "right_gripper")
GROUP_JOINTS = {
    "left_arm": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right_arm": [f"openarm_right_joint{i}" for i in range(1, 8)],
    "left_gripper": ["openarm_left_finger_joint1"],
    "right_gripper": ["openarm_right_finger_joint1"],
}
# SRDF 的 group_state 名（左右夹爪命名不一致是原样）
FALLBACK_STATES = {
    "left_arm": {"home": [0.0] * 7, "hands_up": [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]},
    "right_arm": {"home": [0.0] * 7, "hands_up": [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]},
    "left_gripper": {"open": [0.044], "closed": [0.0], "half closed": [0.022]},
    "right_gripper": {"open": [0.044], "closed": [0.0], "half_closed": [0.022]},
}


def load_named_states(srdf: Path | None = None) -> dict[str, dict[str, list[float]]]:
    """从 SRDF 读 group_state（读不到就用内置表）。"""
    out = {g: {} for g in GROUPS}
    if srdf is None:
        srdf = Path.home() / "openarm_ws/src/openarm_ros2" / \
            "openarm_bimanual_moveit_config/config/openarm_bimanual.srdf"
    try:
        root = ET.parse(srdf).getroot()
        for st in root.findall("group_state"):
            grp = st.get("group")
            name = st.get("name")
            if grp not in out:
                continue
            vals = {}
            for j in st.findall("joint"):
                vals[j.get("name")] = float(j.get("value"))
            joints = GROUP_JOINTS[grp]
            out[grp][name] = [float(vals.get(j, 0.0)) for j in joints]
    except Exception:  # noqa: BLE001
        return {g: dict(v) for g, v in FALLBACK_STATES.items()}
    for g in GROUPS:
        if not out[g]:
            out[g] = dict(FALLBACK_STATES[g])
    return out


class MoveItClient:
    """MoveGroup action 客户端（非阻塞调用 + 回调回主线程）。"""

    def __init__(self, node, on_log=None):
        from rclpy.action import ActionClient

        self._node = node
        self._log = on_log or (lambda *_: None)
        self._client = ActionClient(node, MoveGroup, "/move_action")
        self.named_states = load_named_states()
        self._busy = False
        self._lock = threading.Lock()
        self.last_result: str = ""

    # ── 可用性 ────────────────────────────────────────
    def ready(self, timeout: float = 0.4) -> bool:
        return self._client.server_is_ready() if timeout <= 0 else \
            self._client.wait_for_server(timeout_sec=timeout)

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    # ── 请求构造 ──────────────────────────────────────
    def _build(self, group: str, *, joint_goal: list[float] | None = None,
               pose_goal: Pose | None = None, pose_reference: str = "",
               vel_scale: float = 0.1, acc_scale: float = 0.1,
               plan_only: bool = False) -> MoveGroup.Goal:
        req = MoveGroup.Goal()
        # planning_options 属于 Goal（不是 MotionPlanRequest）
        req.planning_options.plan_only = bool(plan_only)
        req.planning_options.replan = False
        req.planning_options.look_around = False
        req.planning_options.planning_scene_diff.is_diff = True

        r = req.request
        r.group_name = group
        r.num_planning_attempts = 10
        r.allowed_planning_time = 5.0
        r.max_velocity_scaling_factor = float(vel_scale)
        r.max_acceleration_scaling_factor = float(acc_scale)

        from moveit_msgs.msg import Constraints, JointConstraint, \
            OrientationConstraint, PositionConstraint

        c = Constraints()
        c.name = f"{group}_goal"
        if joint_goal is not None:
            for name, val in zip(GROUP_JOINTS[group], joint_goal):
                jc = JointConstraint()
                jc.joint_name = name
                jc.position = float(val)
                jc.tolerance_above = 0.01
                jc.tolerance_below = 0.01
                jc.weight = 1.0
                c.joint_constraints.append(jc)
        elif pose_goal is not None:
            pc = PositionConstraint()
            pc.header.frame_id = pose_reference or "world"
            pc.link_name = self._ee_link(group)
            pc.constraint_region.primitive_poses.append(pose_goal)
            pc.weight = 1.0
            c.position_constraints.append(pc)
            oc = OrientationConstraint()
            oc.header.frame_id = pc.header.frame_id
            oc.link_name = pc.link_name
            oc.orientation = pose_goal.orientation
            oc.absolute_x_axis_tolerance = 0.05
            oc.absolute_y_axis_tolerance = 0.05
            oc.absolute_z_axis_tolerance = 0.05
            oc.weight = 1.0
            c.orientation_constraints.append(oc)
        r.goal_constraints.append(c)
        return req

    @staticmethod
    def _ee_link(group: str) -> str:
        return {
            "left_arm": "openarm_left_hand",
            "right_arm": "openarm_right_hand",
        }.get(group, "openarm_left_hand")

    # ── 调用 ─────────────────────────────────────────
    def send(self, goal: MoveGroup.Goal, done_cb) -> bool:
        """发送目标；done_cb(ok: bool, text: str) 在结果回来时调用。"""
        with self._lock:
            if self._busy:
                self._log("MoveIt 正忙，忽略本次请求")
                return False
            self._busy = True
        if not self.ready():
            with self._lock:
                self._busy = False
            self._log("MoveIt 未就绪：/move_action 无服务端（move_group 未启动？）")
            done_cb(False, "move_group 未启动")
            return False

        plan_only = bool(goal.planning_options.plan_only)
        fut = self._client.send_goal_async(goal)
        fut.add_done_callback(
            lambda f: self._on_goal(f, plan_only, done_cb))
        return True

    def _on_goal(self, fut, plan_only: bool, done_cb) -> None:
        gh = fut.result()
        if gh is None or not gh.accepted:
            with self._lock:
                self._busy = False
            done_cb(False, "目标被拒绝")
            return
        gh.get_result_async().add_done_callback(
            lambda f: self._on_result(f, plan_only, done_cb))

    def _on_result(self, fut, plan_only: bool, done_cb) -> None:
        with self._lock:
            self._busy = False
        res = fut.result()
        if res is None:
            done_cb(False, "无结果")
            return
        code = res.status
        r = res.result
        ok = code == GoalStatus.STATUS_SUCCEEDED
        if ok:
            n = len(r.planned_trajectory.joint_trajectory.points)
            dur = 0.0
            if n:
                pts = r.planned_trajectory.joint_trajectory.points[-1]
                dur = pts.time_from_start.sec + pts.time_from_start.nanosec / 1e9
            txt = (f"规划成功：{n} 个轨迹点，时长 {dur:.2f}s"
                   if plan_only
                   else f"执行完成（{n} 个轨迹点）")
        else:
            txt = f"失败（status={code}）{r.error_code.message if hasattr(r, 'error_code') else ''}"
        self.last_result = txt
        done_cb(ok, txt)

    # ── 便捷封装 ──────────────────────────────────────
    def plan_joint(self, group: str, joints: list[float], done_cb, *,
                   execute: bool, vel: float = 0.1, acc: float = 0.1) -> bool:
        return self.send(self._build(group, joint_goal=joints,
                                     vel_scale=vel, acc_scale=acc,
                                     plan_only=not execute), done_cb)

    def plan_named(self, group: str, state: str, done_cb, *,
                   execute: bool, vel: float = 0.1, acc: float = 0.1) -> bool:
        vals = self.named_states.get(group, {}).get(state)
        if vals is None:
            done_cb(False, f"未知命名状态 {state}")
            return False
        return self.plan_joint(group, vals, done_cb, execute=execute, vel=vel, acc=acc)

    def plan_pose(self, group: str, pose: Pose, done_cb, *,
                  execute: bool, frame: str = "world",
                  vel: float = 0.1, acc: float = 0.1) -> bool:
        return self.send(self._build(group, pose_goal=pose, pose_reference=frame,
                                     vel_scale=vel, acc_scale=acc,
                                     plan_only=not execute), done_cb)
