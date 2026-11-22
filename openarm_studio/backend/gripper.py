"""夹爪抓取判定。使用电机位置、速度和回报力矩。

夹持力来自电机 POS_FORCE 模式的电流上限，前提是目标位置始终停在物体另一侧。
上位机只下发「闭合到全闭」并判定状态，不根据力矩回退目标。

真机实测（2026-11-22，左右爪 DM4310；原始数据在 tests/fixtures/gripper_posforce/）：
- 被挡住时输出力矩恒等于电流上限：堵转力矩 ≈ 9.5 N·m × 电流标幺值，
  两指互压 11.4 s 波动不到 1%，温度不变。
- 让夹爪动起来本身就需要 0.24–0.37 N·m（机构摩擦），与是否碰到物体无关。
- 两爪各有一处"紧点"需要超过 0.43 N·m：右爪在 −0.68 附近张开、左爪在 −0.81 附近
  闭合，标幺值 0.0444 时都会半途卡住，力矩顶满却不动，0.08 能顺利越过。
- 换目标后约 40–50 ms 才起步；到位后静止时仍带有 ±0.1–0.3 N·m 的力矩，可持续数秒以上。
- 两指胶垫互压时的位置随力矩在 −0.059…−0.033 rad 之间。

所以「力矩偏大 + 没在动」不能当作接触。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GripperFeedback:
    position: float  # 电机弧度，负值为张开
    velocity: float
    torque: float
    frame: int


GRASP_SPEED_RAD_S = 0.50     # 约 21 mm/s，全行程约 2.5 s
GRASP_CLOSED_RAD = -0.075    # 开口约 3 mm 以内算全闭（实测全闭 −0.059…−0.033）
GRASP_STILL_RAD_S = 0.08     # 静止时速度反馈 −0.022…+0.007 rad/s
GRASP_TIMEOUT_S = 5.0        # 全行程 1.1 rad 约 2.2 s，留一倍余量
# openarm_driver 把上限按 N·m / 4.5 换算成电流标幺值，而实测堵转力矩是
# 9.5 N·m × 标幺值（电机固件按最大电流取标幺，不是额定力矩）。
DRIVER_NM_PER_PU = 4.5
STALL_NM_PER_PU = 9.5
# 机构摩擦 0.24–0.37 N·m，两爪的"紧点"超过 0.43 N·m：标幺值 0.0444 时实测半途卡住。
# 0.08（约 0.76 N·m）在两爪上都能越过紧点，夹住后还剩约 0.2–0.5 N·m 用于夹持。
GRASP_MIN_PU = 0.08
GRASP_MIN_LIMIT = GRASP_MIN_PU * DRIVER_NM_PER_PU
_STILL_S = 0.22
_MIN_CLOSE_S = 0.30   # 起步要 40–50 ms，静止判定不能早于它
_LOST_S = 0.18


class Grasp:
    """闭合到全闭，由电流上限限力；判定全闭、夹持、脱出，目标不回退。"""

    def __init__(
        self, now: float, feedback: GripperFeedback, nominal_limit: float, *,
        current_ceiling: float | None = None, current_step: float = 0.0,
    ):
        self.phase = "closing"
        self.speed_limit = GRASP_SPEED_RAD_S
        self.target = 0.0
        self.nominal_limit = max(nominal_limit, GRASP_MIN_LIMIT)
        self.started_at = now
        self.last = feedback
        self.last_at = now
        self.still_since: float | None = None
        self.closed_since: float | None = None

    @property
    def torque_limit(self) -> float:
        """电流上限对应的实测堵转力矩（N·m）。"""
        return self.nominal_limit / DRIVER_NM_PER_PU * STALL_NM_PER_PU

    def finish(self, phase: str) -> None:
        """被外部停止：闭合途中停在最后已知位置，已经夹住的继续夹。"""
        if self.phase == "closing":
            self.target = min(0.0, self.last.position)
        self.phase = phase

    def update(self, now: float, feedback: GripperFeedback | None) -> None:
        if self.phase not in ("closing", "holding"):
            return
        if feedback is None:
            self.finish("feedback_lost")
            return
        if feedback.frame == self.last.frame:
            return
        if now - self.last_at > 0.10:
            self.still_since = self.closed_since = None
        self.last, self.last_at = feedback, now
        still = abs(feedback.velocity) <= GRASP_STILL_RAD_S
        closed = feedback.position >= GRASP_CLOSED_RAD

        if self.phase == "closing":
            elapsed = now - self.started_at
            if not still:
                self.still_since = None
                if elapsed >= GRASP_TIMEOUT_S:
                    self.phase = "timeout"
            elif self.still_since is None:
                self.still_since = now
            elif (elapsed >= _MIN_CLOSE_S and now - self.still_since >= _STILL_S):
                self.phase = "empty" if closed else "holding"
            return

        if not closed:
            self.closed_since = None
        elif self.closed_since is None:
            self.closed_since = now
        elif now - self.closed_since >= _LOST_S:
            self.phase = "lost"
