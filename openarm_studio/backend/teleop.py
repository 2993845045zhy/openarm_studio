"""基于 OpenArm v1.0 左右关节轴的单向遥操。"""

from dataclasses import dataclass, field
from threading import Lock
from typing import Literal

# 按双臂 URDF 的轴方向关于 y=0 镜像。J4 同向，其余六轴反向。
MIRROR_SIGNS = (-1, -1, -1, 1, -1, -1, -1)
# 将零位关节轴换算到共同的机器人底座坐标系。J1、J7 的轴相反。
SAME_DIRECTION_SIGNS = (-1, 1, 1, 1, 1, 1, -1)
FEEDBACK_TIMEOUT_S = 0.15
_MAX_START_SPEED_RAD_S = 0.15


@dataclass(frozen=True)
class JointFeedback:
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    expires_at: float

    def stationary(self) -> bool:
        return all(abs(speed) <= _MAX_START_SPEED_RAD_S for speed in self.velocities)


class TeleopWatchdog:
    """让两条臂共享第一次停止原因，直到重新开始协同。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._reason: str | None = None

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def stop(self, reason: str) -> None:
        with self._lock:
            if self._reason is None:
                self._reason = reason


@dataclass(frozen=True)
class TeleopCommand:
    role: Literal["leader", "follower"]
    expires_at: float
    joints: tuple[float, ...] = ()
    watchdog: TeleopWatchdog = field(default_factory=TeleopWatchdog, repr=False, compare=False)


@dataclass(frozen=True)
class BimanualTeleop:
    leader_side: str
    leader_origin: tuple[float, ...]
    follower_origin: tuple[float, ...]
    mirror: bool = True
    watchdog: TeleopWatchdog = field(
        default_factory=TeleopWatchdog, init=False, repr=False, compare=False)

    @property
    def follower_side(self) -> str:
        return "right" if self.leader_side == "left" else "left"

    def commands(
        self, leader: JointFeedback, follower: JointFeedback,
    ) -> tuple[TeleopCommand, TeleopCommand]:
        deadline = min(leader.expires_at, follower.expires_at)
        directions = MIRROR_SIGNS if self.mirror else SAME_DIRECTION_SIGNS
        position = tuple(
            start + sign * (current - origin)
            for start, sign, current, origin in zip(
                self.follower_origin, directions, leader.positions, self.leader_origin)
        )
        return (TeleopCommand("leader", deadline, watchdog=self.watchdog),
                TeleopCommand("follower", deadline, position, watchdog=self.watchdog))
