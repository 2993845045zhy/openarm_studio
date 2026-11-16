"""OpenArm Studio 的直连后端：openarm_sdk / openarm_can，不依赖 ROS。

每条臂一个 SdkArm：
- 连接 = 打开 CAN 口（真机 can0/can1；可配）
- 后台线程轮询电机状态（refresh_all + recv_all）→ 关节名字典
- 控制 = MIT 力位混合帧（与 ros2_control 硬件层同一套底层）
- 急停/失能 = disable_all

安全注意（飞臂事故的教训）：
- enable 会立刻上力矩；本后端**连接时不自动 enable**，由 UI 显式触发
- 断开时先 disable_all 再关总线
"""

from __future__ import annotations

import threading
import time

# 关节名（与 URDF/ROS 侧完全一致，UI 直接用）
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
GRIP_JOINT = "finger_joint1"

# MIT 直发的默认增益（照官方 openarm_mujoco v2 的 kp/kv 分档）
_DEFAULT_KP = (230.0, 230.0, 190.0, 190.0, 30.0, 30.0, 30.0)
_DEFAULT_KD = (2.7, 2.7, 2.2, 2.2, 1.5, 1.5, 1.5)
GRIP_KP, GRIP_KD = 10.0, 0.2
GRIP_OPEN = 0.044      # m，完全张开
GRIP_CLOSED = 0.0


class SdkArm:
    """单臂直连（含夹爪，共 8 电机）。线程安全。"""

    def __init__(self, side: str, can_interface: str, on_log=None):
        self.side = side                      # "left" / "right"
        self.can = can_interface
        self._log = on_log or (lambda *_: None)
        self._sdk = None
        self._lock = threading.Lock()
        self._positions: dict[str, float] = {}
        self._velocities: dict[str, float] = {}
        self._torques: dict[str, float] = {}
        self._targets: dict[str, float] | None = None   # None=不下发
        self._poll_thread = None
        self._running = False
        self._rx_hz = 0.0
        self._n = 0
        self._t0 = time.monotonic()
        self._enabled = False

    # ── 生命周期 ──────────────────────────────────────
    def connect(self) -> bool:
        from openarm_sdk import OpenArmSDK

        try:
            sdk = OpenArmSDK(can_interface=self.can, enable_can_fd=True)
            sdk.init_arm_motors()
            sdk.init_gripper_motor()
        except Exception as exc:  # noqa: BLE001
            self._log(f"[{self.side}] 连接 {self.can} 失败：{exc}")
            return False
        with self._lock:
            self._sdk = sdk
            self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True,
                                             name=f"sdk-{self.side}")
        self._poll_thread.start()
        self._log(f"[{self.side}] 已连接 {self.can}（8 电机，CAN-FD）")
        return True

    def disconnect(self) -> None:
        with self._lock:
            self._running = False
            sdk, self._sdk = self._sdk, None
            self._targets = None
            self._enabled = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None
        if sdk is not None:
            try:
                sdk.disable_all()
            except Exception:  # noqa: BLE001
                pass
            try:
                sdk.close()
            except Exception:  # noqa: BLE001
                pass
        self._log(f"[{self.side}] 已断开（电机已失能）")

    @property
    def connected(self) -> bool:
        return self._sdk is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── 状态轮询 + 控制下发（同一个循环，与硬件层 read/write 同构）────
    def _poll_loop(self) -> None:
        while True:
            with self._lock:
                run = self._running
                sdk = self._sdk
                targets = self._targets
            if not run or sdk is None:
                return
            try:
                if targets is not None:
                    self._send_mit(sdk, targets)
                sdk.refresh_all()
                sdk.recv_all(timeout_us=500)
                arm = sdk.get_arm().get_motors()
                grip = sdk.get_gripper().get_motors()
                pos, vel, tau = {}, {}, {}
                prefix = f"openarm_{self.side}_"
                for i, m in enumerate(arm):
                    name = f"{prefix}joint{i + 1}"
                    pos[name] = m.get_position()
                    vel[name] = m.get_velocity()
                    tau[name] = m.get_torque()
                if grip:
                    gn = f"{prefix}{GRIP_JOINT}"
                    # SDK 的夹爪电机读数是弧度；换算到 0-0.044 m 关节值
                    pos[gn] = max(0.0, -grip[0].get_position()) / 1.0472 * 0.044
                    vel[gn] = grip[0].get_velocity()
                    tau[gn] = grip[0].get_torque()
                with self._lock:
                    self._positions, self._velocities, self._torques = pos, vel, tau
                self._n += 1
                now = time.monotonic()
                if now - self._t0 >= 1.0:
                    self._rx_hz = self._n / (now - self._t0)
                    self._n = 0
                    self._t0 = now
            except Exception as exc:  # noqa: BLE001
                self._log(f"[{self.side}] 轮询异常：{type(exc).__name__}: {exc}")
                with self._lock:
                    self._running = False
                    self._sdk = None
                return
            time.sleep(0.002)   # ~500Hz 循环上限，实际由总线时间决定

    def _send_mit(self, sdk, targets: dict[str, float]) -> None:
        from openarm.can import MITParam

        prefix = f"openarm_{self.side}_"
        params = []
        for i in range(7):
            name = f"{prefix}joint{i + 1}"
            params.append(MITParam(_DEFAULT_KP[i], _DEFAULT_KD[i],
                                   float(targets.get(name, 0.0)), 0.0, 0.0))
        sdk.get_arm().mit_control_all(params)
        gn = f"{prefix}{GRIP_JOINT}"
        if gn in targets:
            # 关节值 0-0.044 m → 电机弧度（0 → 0；张开 0.044 → -1.0472）
            rad = -float(targets[gn]) / 0.044 * 1.0472
            sdk.get_gripper().mit_control_all(
                [MITParam(GRIP_KP, GRIP_KD, rad, 0.0, 0.0)])

    # ── 对外接口 ──────────────────────────────────────
    def positions(self) -> dict[str, float]:
        with self._lock:
            return dict(self._positions)

    def state(self, name: str) -> tuple[float, float, float]:
        with self._lock:
            return (self._positions.get(name, 0.0),
                    self._velocities.get(name, 0.0),
                    self._torques.get(name, 0.0))

    @property
    def rate(self) -> float:
        return self._rx_hz

    def enable(self) -> bool:
        sdk = self._sdk
        if sdk is None:
            return False
        try:
            sdk.enable_all()
            self._enabled = True
            self._log(f"[{self.side}] 电机已使能（上力矩）")
            return True
        except Exception as exc:  # noqa: BLE001
            self._log(f"[{self.side}] 使能失败：{exc}")
            return False

    def disable(self) -> bool:
        sdk = self._sdk
        if sdk is None:
            return False
        try:
            sdk.disable_all()
            self._enabled = False
            with self._lock:
                self._targets = None
            self._log(f"[{self.side}] 电机已失能（手臂会下垂）")
            return True
        except Exception as exc:  # noqa: BLE001
            self._log(f"[{self.side}] 失能失败：{exc}")
            return False

    def set_targets(self, targets: dict[str, float]) -> None:
        """MIT 目标（关节名→rad / 夹爪 m）。targets=None 停止下发（只读）。"""
        with self._lock:
            self._targets = dict(targets) if targets is not None else None
