"""OpenArm Studio 直连后端：官方 openarm_driver，无 ROS。

架构（driver 非线程安全，这是它的正确用法）：
- 每臂一个 DriverArm，**唯一控制线程**拥有 SingleArmDriver
- UI 只投递命令（enable/disable/targets/smooth/急停）+ 读状态缓存
- 状态缓存由控制线程以 ~500Hz 刷新（受 CAN 带宽限制实际略低）

官方安全链全程生效：send_position 经过 限位裁剪→单步跳变拒绝→速度限幅，
任何 force-stop 会锁存 safety_stop_reason 并挡掉后续指令。

夹爪单位换算：driver 第 8 维是弧度；cell 配置的 joint8 限位右臂
[-1.0472, 0.4]、左臂 [-0.4, 1.0472]（左右符号相反）。UI 统一用米
（0=闭合，0.044=全开），按侧换算。方向以真机验证为准（TODO 注释处）。
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

GRIP_OPEN_M = 0.044
_GRIP_RAD_SPAN = 1.0472     # 0.044 m 对应的电机弧度行程

# 官方配置（bundled）
CONFIGS = ("openarm_cell", "openarm_cell_higher_pd",
           "openarm_cell_higher_pd_zero_offset", "openarm_pedestal")


class _Cmd:
    __slots__ = ("kind", "payload")

    def __init__(self, kind: str, payload=None):
        self.kind = kind
        self.payload = payload


class DriverArm:
    """单臂：控制线程 + 命令队列 + 状态缓存。线程安全对外。"""

    def __init__(self, side: str, can_interface: str, config_name: str = "openarm_cell",
                 on_log=None):
        self.side = side                    # "left" / "right"
        self.can = can_interface
        self.config_name = config_name
        self._log = on_log or (lambda *_: None)

        self._driver = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._cmds: "queue.Queue[_Cmd]" = queue.Queue(maxsize=64)

        self._lock = threading.Lock()
        self._positions: dict[str, float] = {}
        self._torques: dict[str, float] = {}
        self._tmos: dict[str, int] = {}
        self._targets: np.ndarray | None = None     # 8 维 rad（含夹爪）
        self._enabled = False
        self._safety: str | None = None
        self._rate = 0.0
        self._n = 0
        self._t0 = time.monotonic()
        self._smooth_done = None      # callback(ok)

    # ── 生命周期 ──────────────────────────────────────
    def connect(self) -> bool:
        from openarm_driver import Config, SingleArmDriver

        try:
            cfg = Config(self.config_name)
            drv = SingleArmDriver(f"{self.side}_arm", config=cfg,
                                  can_interface=self.can)
        except Exception as exc:  # noqa: BLE001
            self._log(f"[{self.side}] 连接 {self.can} 失败：{exc}")
            return False
        with self._lock:
            self._driver = drv
            self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"driver-{self.side}")
        self._thread.start()
        self._log(f"[{self.side}] 已连接 {self.can}（openarm_driver/{self.config_name}）")
        return True

    def disconnect(self) -> None:
        with self._lock:
            self._running = False
            drv, self._driver = self._driver, None
            self._targets = None
            self._enabled = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if drv is not None:
            try:
                drv.stop()
            except Exception:  # noqa: BLE001
                try:
                    drv.openarm.disable_all()
                except Exception:  # noqa: BLE001
                    pass
        self._log(f"[{self.side}] 已断开（stop 失能）")

    @property
    def connected(self) -> bool:
        return self._driver is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def safety_stop_reason(self) -> str | None:
        return self._safety

    @property
    def rate(self) -> float:
        return self._rate

    # ── 命令（UI 线程调用，全部投递）─────────────────
    def _push(self, cmd: _Cmd) -> bool:
        try:
            self._cmds.put_nowait(cmd)
            return True
        except queue.Full:
            self._log(f"[{self.side}] 命令队列满，丢弃 {cmd.kind}")
            return False

    def enable(self) -> bool:
        """start()：使能并走起始轨迹（臂会动到安全起始位，真机注意）。"""
        return self._push(_Cmd("enable"))

    def disable(self) -> bool:
        """stop()：回程轨迹后失能。急停不要用这个（有轨迹）。"""
        return self._push(_Cmd("disable"))

    def estop(self) -> bool:
        """急停：立即 disable_all，无回程轨迹（手臂直接失去力矩）。"""
        return self._push(_Cmd("estop"))

    def set_targets(self, named: dict[str, float] | None) -> None:
        """关节名→值（rad / 夹爪 m）。None = 停止下发，转只读。"""
        if named is None:
            self._push(_Cmd("hold"))
            return
        self._push(_Cmd("targets", self._to_array(named)))

    def smooth_move(self, named: dict[str, float], hz: float = 50.0,
                    duration: float = 2.0, done=None) -> bool:
        """插值运动（官方 smooth_move，走完整安全链）。阻塞控制线程。"""
        self._smooth_done = done
        return self._push(_Cmd("smooth", (self._to_array(named), hz, duration)))

    # ── 单位换算 ──────────────────────────────────────
    def _to_array(self, named: dict[str, float]) -> np.ndarray:
        """关节名字典 → driver 的 8 维数组（夹爪 m→rad）。"""
        prefix = f"openarm_{self.side}_"
        arr = np.zeros(8)
        for i in range(1, 8):
            arr[i - 1] = float(named.get(f"{prefix}joint{i}", 0.0))
        gm = float(named.get(f"{prefix}finger_joint1", 0.0))
        gm = min(GRIP_OPEN_M, max(0.0, gm))
        # TODO 真机确认方向：按 cell 限位左右取反（右臂张开=负 rad）
        sign = -1.0 if self.side == "right" else 1.0
        arr[7] = sign * gm / GRIP_OPEN_M * _GRIP_RAD_SPAN
        return arr

    def _to_named(self, arr: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
        prefix = f"openarm_{self.side}_"
        pos, tau = {}, {}
        for i in range(1, 8):
            pos[f"{prefix}joint{i}"] = float(arr[i - 1])
            tau[f"{prefix}joint{i}"] = float(self._tau_arr[i - 1]) \
                if self._tau_arr is not None else 0.0
        sign = -1.0 if self.side == "right" else 1.0
        rad = float(arr[7])
        pos[f"{prefix}finger_joint1"] = min(
            GRIP_OPEN_M, max(0.0, sign * rad / _GRIP_RAD_SPAN * GRIP_OPEN_M))
        return pos, tau

    # ── 控制线程 ──────────────────────────────────────
    def _loop(self) -> None:
        while True:
            with self._lock:
                run = self._running
                drv = self._driver
            if not run or drv is None:
                return
            try:
                # 1) 先执行命令（可能阻塞：enable 起始轨迹 / smooth 插值）
                busy = self._drain_cmds(drv)
                # 2) 状态刷新（smooth 期间 send_position 内部已刷新，跳过）
                if not busy:
                    st = drv.fetch_state(refresh=True)
                    self._publish(st)
                self._tick_rate()
            except Exception as exc:  # noqa: BLE001
                self._log(f"[{self.side}] 控制线程异常退出：{type(exc).__name__}: {exc}")
                with self._lock:
                    self._running = False
                    self._driver = None
                return
            time.sleep(0.002)

    def _drain_cmds(self, drv) -> bool:
        """执行队列里的命令。返回 True=刚做过耗时操作（状态已随之刷新）。"""
        try:
            cmd = self._cmds.get_nowait()
        except queue.Empty:
            # 无新命令：若持有目标则持续下发（保持力矩）
            with self._lock:
                tgt = self._targets
            if tgt is not None and self._enabled:
                ok = drv.send_position(tgt)
                self._check_safety(ok, drv)
                return True
            return False

        if cmd.kind == "enable":
            ok = drv.start()
            with self._lock:
                self._enabled = bool(ok)
            self._log(f"[{self.side}] start() {'成功（已走起始轨迹）' if ok else '失败/被安全链中断'}")
            self._check_safety(ok, drv)
            return True
        if cmd.kind == "disable":
            drv.stop()
            with self._lock:
                self._enabled = False
                self._targets = None
            self._log(f"[{self.side}] stop() 完成（回程轨迹+失能）")
            return True
        if cmd.kind == "estop":
            try:
                drv.openarm.disable_all()
            finally:
                with self._lock:
                    self._enabled = False
                    self._targets = None
            self._log(f"[{self.side}] 急停：立即失能（无回程轨迹）")
            return True
        if cmd.kind == "hold":
            with self._lock:
                self._targets = None
            return False
        if cmd.kind == "targets":
            with self._lock:
                self._targets = cmd.payload
            if self._enabled:
                ok = drv.send_position(cmd.payload)
                self._check_safety(ok, drv)
            return True
        if cmd.kind == "smooth":
            tgt, hz, dur = cmd.payload
            ok = drv.smooth_move(tgt, hz=hz, duration=dur)
            self._check_safety(ok, drv)
            self._log(f"[{self.side}] smooth_move {'完成' if ok else '被安全链中断'}")
            if self._smooth_done is not None:
                try:
                    self._smooth_done(bool(ok))
                except Exception:  # noqa: BLE001
                    pass
                self._smooth_done = None
            with self._lock:
                self._targets = tgt if ok else None
            return True
        return False

    def _check_safety(self, ok: bool, drv) -> None:
        reason = drv.safety_stop_reason
        with self._lock:
            self._safety = reason
            if reason is not None:
                self._enabled = False
                self._targets = None
        if reason is not None:
            self._log(f"[{self.side}] 安全停止（锁存）：{reason}")
            try:
                drv.openarm.disable_all()
            except Exception:  # noqa: BLE001
                pass

    def _publish(self, st: dict) -> None:
        self._tau_arr = st.get("qtorque")
        pos, tau = self._to_named(st["qpos"])
        tmos = st.get("tmos")
        prefix = f"openarm_{self.side}_"
        tm = {}
        for i in range(1, 8):
            tm[f"{prefix}joint{i}"] = int(tmos[i - 1]) if tmos is not None else 0
        with self._lock:
            self._positions, self._torques, self._tmos = pos, tau, tm

    def _tick_rate(self) -> None:
        self._n += 1
        now = time.monotonic()
        if now - self._t0 >= 1.0:
            self._rate = self._n / (now - self._t0)
            self._n = 0
            self._t0 = now

    _tau_arr = None

    # ── 状态读取（UI 线程）────────────────────────────
    def positions(self) -> dict[str, float]:
        with self._lock:
            return dict(self._positions)

    def torques(self) -> dict[str, float]:
        with self._lock:
            return dict(self._torques)

    def temps(self) -> dict[str, int]:
        with self._lock:
            return dict(self._tmos)
