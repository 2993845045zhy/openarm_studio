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
from collections import deque
import threading
import time

import numpy as np

from .gripper import DRIVER_NM_PER_PU, Grasp, GripperFeedback
from .teleop import FEEDBACK_TIMEOUT_S, JointFeedback, TeleopCommand

_TELEOP_MAX_TRACKING_ERROR_RAD = 0.08
_TELEOP_FOLLOWER_SPEED_LIMITS_RAD_S = (1.05, 1.05, 1.05, 1.05, 1.25, 1.25, 1.25)
_TELEOP_FOLLOWER_APPROACH_RATE_S = 18.0
GRIP_OPEN_M = 0.044
MANUAL_JOINT_SPEED_RAD_S = 0.60
# J1–J4 带动较长臂段，腕部 J5–J7 允许稍高的角速度。
SLIDER_JOINT_SPEED_LIMITS_RAD_S = (1.10, 1.10, 1.10, 1.10, 1.18, 1.20, 1.25)
_SLIDER_ACCEL_RAD_S2 = np.array([3.6, 3.6, 3.6, 3.6, 5.0, 5.4, 5.8])
_SLIDER_BRAKE_RAD_S2 = np.array([4.2, 4.2, 4.2, 4.2, 6.0, 6.4, 6.8])
_ENABLE_JOINT_SPEED_RAD_S = 0.40
_DISABLE_JOINT_SPEED_RAD_S = 0.25
_POWER_POSITION_TOL_RAD = 0.05
_POWER_VELOCITY_TOL_RAD_S = 0.10
_POWER_STABLE_S = 0.25
_POWER_FEEDBACK_MAX_AGE_S = 0.10
_POWER_TIMEOUT_S = 3.0
_START_FEEDBACK_TIMEOUT_S = 0.08
_START_FEEDBACK_POLLS = 12
_START_FEEDBACK_POLL_INTERVAL_S = 0.003
_COMMAND_PERIOD_S = 0.02
_POLL_ACTIVE_PERIOD_S = 0.01
_POLL_IDLE_PERIOD_S = 0.04
_POLL_BLOCKED_PERIOD_S = 0.25
_FRIC_FEEDBACK_MAX_AGE_S = 0.06
_FRIC_FILTER_S = 0.04
_FRIC_SLEW_NM_S = 10.0
_FRIC_LIMIT_NM = 3.0
# 略大于各轴电机速度反馈的一个量化步长，按 J1–J7 排列。
_FRIC_DEADBAND_RAD_S = np.array([0.03, 0.03, 0.01, 0.01, 0.02, 0.02, 0.02])
_TEACH_WRIST_KD = (1.0, 1.0, 0.9)
_TEACH_WRIST_FRIC_FRACTION = 0.8
_TEACH_WRIST_CONFIRM_RAD = 0.0011  # DM4310 位置反馈约三个格
_TEACH_WRIST_WINDOW_S = 0.14
_TEACH_WRIST_STOP_S = 0.045
_TEACH_WRIST_ASSIST_NM = np.array([0.025, 0.035, 0.05])
_TEACH_WRIST_ASSIST_FULL_RAD_S = 0.03
_TEACH_WRIST_ASSIST_MAX_RAD_S = 0.08
_TEACH_WRIST_LIMIT_MARGIN_RAD = 0.02
_PLAYBACK_TIMEOUT_S = 0.15
_GRIP_RAD_SPAN = 1.0472     # 0.044 m 对应的电机弧度行程

# 夹爪方向：实测本机两臂**同为**「张开 = 负 rad、闭合 ≈ 0」。
# 官方配置假设左臂镜像（joint8 限位左右相反），与本机实测不符：
# 左臂发正 rad 时电机不动（超出机械行程），故不按侧取反。
_GRIP_SIGN = -1.0

def _gripper_rad(opening: float) -> float:
    return _GRIP_SIGN * min(GRIP_OPEN_M, max(0.0, opening)) / GRIP_OPEN_M * _GRIP_RAD_SPAN


def _fix_left_gripper_config(cfg) -> None:
    """本机两臂夹爪同向（张开 = 负 rad），官方配置却按"左臂镜像"来写。

    限位 [-0.4, 1.0472] 会把左爪张开行程裁掉 2/3；起始位的 +0.4 方向改正后
    等于"越过全闭"，每次使能左爪都会以电流上限夹死。两处都改成与右臂相同的值。
    """
    try:
        cfg._config["joint_limits"]["left_arm"][7] = [-_GRIP_RAD_SPAN, 0.4]  # noqa: SLF001
        for section in ("start", "stop"):
            for move in cfg._config[section]["moves"]:  # noqa: SLF001
                move["position"]["left_arm"][7] = move["position"]["right_arm"][7]
    except Exception:  # noqa: BLE001  配置结构不符时不能挡住连接
        pass


# 官方配置（bundled）。默认 openarm_pedestal：零偏移 + 标准增益
# （本机电机零位=URDF 零位；cell 的工装偏移会把姿态整体平移出假角度）。
CONFIGS = ("openarm_pedestal",
           "openarm_cell",
           "openarm_cell_higher_pd",
           "openarm_cell_higher_pd_zero_offset")

# MIT 帧是 12-bit 定点编码：kd ∈ [0, 5]、kp ∈ [0, 500]、tau ∈ [−tMax, tMax]
# （见 openarm_can :: pack_mit_control_data）。超出部分会被**静默截断**，
# 所以这里显式裁剪，避免"调了没反应还想不通"。
MIT_KD_MAX = 5.0
MIT_KP_MAX = 500.0

# ── 关节空间阻抗（整条臂）──────────────────────────────────
# 前馈 τ = G(q) + τ_fric 承担重力，PD 只负责抗扰动与回位。两个可调量：
#   刚度 s ∈ [0.01, 1]      Kp = 官方增益 × s（1.0 = 重力补偿那档）
#   阻尼比 ζ ∈ [0.2, 2]     Kd = 官方增益 × 2 × ζ × √s
# kd 取 √s 是为了让阻尼比与刚度解耦：无论多软，ζ 都直接对应"回弹时
# 过不过冲"，而不是像原来那样按固定倍数缩放导致软档过阻尼、硬档欠阻尼。
IMP_KP_MIN = 0.5
IMP_KD_MIN = 0.02
_IMP_SLEW_KP = 80.0     # Kp 每秒钟最大变化量（拖滑块不能一脚踩下去）
_IMP_SLEW_KD = 3.0


class _Cmd:
    __slots__ = ("kind", "payload")

    def __init__(self, kind: str, payload=None):
        self.kind = kind
        self.payload = payload


# ── 总线体检（只读，随时可调用，与已连接的臂共存）────────────
_PROBE_TYPES = None      # 延迟初始化（避免 import 时强依赖 openarm_can）

_UP_HINT = ("sudo ip link set dev {iface} down && sudo ip link set dev {iface} type can "
            "bitrate 1000000 sample-point 0.75 dbitrate 5000000 dsample-point 0.75 fd on "
            "&& sudo ip link set dev {iface} up")


def can_ifstate(iface: str) -> dict:
    """读 /sys/class/net 判断接口状态：是否存在 / 是否 UP / 是否 FD 模式。

    USB 重插后内核按默认重建接口（DOWN + 经典 CAN），此时电机再多应答
    也测不到——先报接口问题，别让人去查电机。
    """
    import os
    base = f"/sys/class/net/{iface}"
    if not os.path.isdir(base):
        return {"exists": False, "up": False, "fd": False}
    try:
        with open(f"{base}/flags") as f:
            up = bool(int(f.read().strip(), 16) & 0x1)      # IFF_UP
        with open(f"{base}/mtu") as f:
            fd = int(f.read().strip()) >= 72                 # 72=FD, 16=经典
    except (OSError, ValueError):
        return {"exists": True, "up": False, "fd": False}
    return {"exists": True, "up": up, "fd": fd}


def probe_bus(can_interface: str, rounds: int = 5) -> dict:
    """对单臂总线发 rounds 轮只读查询，返回应答统计。

    只发送状态查询。返回在线轴数、每轴应答与查询次数、位置和发送队列满的次数。
    零应答表示未收到匹配的状态帧，此时位置无效。
    接口不存在/未启用/非 FD 模式时抛 RuntimeError 并附修复命令。
    """
    st = can_ifstate(can_interface)
    if not st["exists"]:
        raise RuntimeError(f"接口 {can_interface} 不存在（适配器未插好？）")
    if not st["up"] or not st["fd"]:
        raise RuntimeError(
            f"接口 {can_interface} 未启用或非 CAN-FD 模式（USB 重插后的典型现象）"
            f"——运行：{_UP_HINT.format(iface=can_interface)}")
    global _PROBE_TYPES
    import openarm_can as oa
    if _PROBE_TYPES is None:
        _PROBE_TYPES = ([oa.MotorType.DM8009] * 2 + [oa.MotorType.DM4340] * 2
                        + [oa.MotorType.DM4310] * 3)
    o = oa.OpenArm(can_interface, True)
    try:
        o.init_arm_motors(_PROBE_TYPES,
                          [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07],
                          [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17])
        o.set_callback_mode_all(oa.CallbackMode.STATE)
        for _ in range(rounds):
            o.refresh_all()
            o.recv_all(first_timeout_us=100_000)
            time.sleep(0.1)
        o.recv_all(first_timeout_us=10_000)  # 补收最后一轮的其他关节回包
        axes, alive = [], 0
        for i in range(7):
            ls = o.get_arm().get_link_stats(i)
            axes.append((ls.responses, ls.commands_sent))
            alive += ls.responses > 0
        pos = [round(m.get_position(), 3) for m in o.get_arm().get_motors()]
        queue_full = o.get_bus_status().write_no_buffer.count
        return {"alive": alive, "axes": axes, "positions": pos, "tx_queue_full": queue_full}
    finally:
        del o


class DriverArm:
    """单臂：控制线程 + 命令队列 + 状态缓存。线程安全对外。"""

    # 接收超时（µs）。官方 send_position 内部固定 300µs，而 8 个电机的
    # 状态帧串行回传约需 1ms，按 CAN 优先级排在后面的轴（0x15~0x18，
    # 即 J5/J6/J7+夹爪）常常读不到 → 应答计数不涨 → 被误判 "静默"。
    # 下发/轮询后补读一次，把积压的帧读干净（参数本身保持官方原值）。
    RECV_US = 4000

    # ── 关节阻抗调参 ──────────────────────────────────
    # 刚度用 0~1 的比例系数（相对官方增益），阻尼用**阻尼比 ζ**直接给：
    #     Kp = base_kp · s,   Kd = base_kd · 2 · ζ · √s
    # 这样调节目标与物理感觉一致——ζ 就是"回弹时过不过冲"，
    # 拖动时常说的"太黏/太飘"直接对应 ζ 的大小，与刚度解耦。
    IMP_STIFF_MIN, IMP_STIFF_MAX = 0.01, 1.0
    IMP_ZETA_MIN, IMP_ZETA_MAX = 0.2, 2.0

    def __init__(self, side: str, can_interface: str,
                 config_name: str = CONFIGS[0], on_log=None):
        self.side = side                    # "left" / "right"
        self.can = can_interface
        self.config_name = config_name
        self._log = on_log or (lambda *_: None)

        self._driver = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._cmds: "queue.Queue[_Cmd]" = queue.Queue(maxsize=64)
        self._urgent_stop = threading.Event()
        self._pending_targets: tuple[dict[str, float], bool, bool] | None = None
        self._pending_hold = False
        self._pending_force_mode: tuple[str, float, float, float] | None = None
        self._pending_friction: bool | None = None
        self._playback_until: float | None = None
        self._delta_limits = None       # 单步限幅（connect 时从配置读）
        self._vel_limits = None
        self._last_send_t = 0.0         # 指令发送节流（≥50Hz 官方节奏）
        self._next_poll_t = 0.0
        self._idle_poll_blocked = False
        # 重力补偿（KDL 前馈力矩，见 backend/gravity.py）
        self._grav_model = None
        self._grav_on = False
        self._grav_scale = 1.0
        self._grav_kp = 30.0            # 保底刚度：前馈算错也不至失控
        self._grav_hold_q: list[float] | None = None
        self._grav_tau: np.ndarray | None = None   # 前馈低通状态
        # 阻尼倍数 2.0：本机是准直驱电机（低减速比、低阻尼），拖动时比 UR
        # 那类谐波减速臂更容易晃。实测 kd×2 把拖动晃动从 0.025 降到 0.016 rad，
        # ×3 收益递减（0.014）且更迟钝，取 2.0。
        self._grav_kd_scale = 2.0
        self._grav_prev_q: list[float] | None = None   # 速度前馈基准
        self._grav_prev_t: float | None = None
        self._grav_n = 0
        self._grav_t0 = None
        # 摩擦补偿（官方 teleop 的 tanh 模型，见 backend/gravity.py）
        self._fric_on = True
        self._fric_tau: np.ndarray | None = None
        self._fric_feedback: tuple[np.ndarray, np.ndarray] | None = None  # 七轴速度及各自的截止时间
        self._teach_wrist_positions: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=20)
        # 关节空间阻抗（整条臂）：与重力补偿共用软模式帧，只换增益
        self._imp_on = False
        self._imp_stiff = 0.35                  # Kp 相对官方增益的比例
        self._imp_zeta = 1.0                    # 阻尼比
        self._imp_kp_cur: np.ndarray | None = None
        self._imp_kd_cur: np.ndarray | None = None
        self._imp_t: float | None = None
        self._last_kp: np.ndarray | None = None   # 上一帧实际下发的增益
        self._last_kd: np.ndarray | None = None
        self._last_impedance_frame: tuple[float, tuple, tuple] | None = None
        self._soft_prev_tgt: list[float] | None = None  # hold 目标限速基准
        self._lim_lo: np.ndarray | None = None
        self._lim_hi: np.ndarray | None = None

        self._lock = threading.Lock()
        self._positions: dict[str, float] = {}
        self._velocities: dict[str, float] = {}
        self._torques: dict[str, float] = {}
        self._tmos: dict[str, int] = {}
        self._trot: dict[str, int] = {}
        self._health: tuple[list, dict] | None = None
        self._health_t = 0.0
        self._targets: np.ndarray | None = None     # 8 维 rad（含夹爪）
        self._enabled = False
        self._safety: str | None = None
        self._rate = 0.0
        self._n = 0
        self._t0 = time.monotonic()
        self._manual_target = True
        self._slider_target = False
        self._slider_velocity = np.zeros(7)
        self._preset_moves: deque[np.ndarray] = deque()
        self._goto_done = None
        self._power_motion: str | None = None
        self._power_wait_since: float | None = None
        self._power_still_since: float | None = None
        self._joint_feedback: JointFeedback | None = None
        self._teleop_command: TeleopCommand | None = None
        self._teleop_role: str | None = None
        self._grip_config_limits: tuple[float, float] | None = None
        self._grip_force_fraction = 0.60
        self._grasp_requested = False
        self._grasp: Grasp | None = None
        self._grasp_status = "idle"

    # ── 生命周期 ──────────────────────────────────────
    def connect(self) -> bool:
        from openarm_driver import Config, SingleArmDriver

        st = can_ifstate(self.can)
        if not st["exists"] or not st["up"] or not st["fd"]:
            self._log(f"[{self.side}] 接口 {self.can} 未就绪（未启用或非 FD 模式，"
                      f"USB 重插后常见）——先运行："
                      f"{_UP_HINT.format(iface=self.can)}")
            return False

        try:
            cfg = Config(self.config_name)
            _fix_left_gripper_config(cfg)
            drv = SingleArmDriver(f"{self.side}_arm", config=cfg,
                                  can_interface=self.can)
        except Exception as exc:  # noqa: BLE001
            self._log(f"[{self.side}] 连接 {self.can} 失败：{exc}")
            return False
        # 只有收到匹配的电机状态帧才能确认总线连接。
        try:
            drv.set_latest_state(timeout_us=self.RECV_US)  # 补收构造期最后一轮回包
            axes = drv.openarm.get_arm()
            alive = sum(axes.get_link_stats(i).responses > 0 for i in range(7))
            queue_full = drv.openarm.get_bus_status().write_no_buffer.count
        except Exception as exc:  # noqa: BLE001
            self._log(f"[{self.side}] {self.can} 读取连接反馈失败：{exc}")
            return False
        if alive == 0:
            if queue_full:
                self._log(f"[{self.side}] {self.can} 发送队列满 {queue_full} 次，"
                          "未收到关节状态回包（0/7 轴）")
            else:
                self._log(f"[{self.side}] {self.can} 未收到关节状态回包（0/7 轴），"
                          "请检查总线体检、供电、线缆和电机 ID")
            return False
        with self._lock:
            self._driver = drv
            self._running = True
            try:
                self._delta_limits = np.asarray(
                    drv.config.get_joint_delta_position_limits(), dtype=float)
            except Exception:  # noqa: BLE001
                self._delta_limits = None
            # 软模式帧直接走 mit_control_all，**绕过官方安全链**（限位/单步/
            # 速度检查全在 send_position 里），所以这里自己拿两样东西兜底：
            # 关节限位 + 逐关节速度限值（软模式直接据此计算每帧位移）。
            try:
                lim = np.asarray(
                    drv.config.get_joint_limits(f"{self.side}_arm"), dtype=float)
                self._lim_lo, self._lim_hi = lim[:, 0].copy(), lim[:, 1].copy()
            except Exception:  # noqa: BLE001
                self._lim_lo = self._lim_hi = None
            try:
                vl = drv.config.get_joint_velocity_limits()
                self._vel_limits = (np.asarray(vl, dtype=float)[:7]
                                    if vl is not None else None)
            except Exception:  # noqa: BLE001
                self._vel_limits = None
        if drv.gripper_posforce:
            self._grip_config_limits = tuple(drv.gripper_posforce_limits)
        self._apply_gripper_limits(drv)
        drv._on_stop = lambda: self._travel_to_stop(drv)
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"driver-{self.side}")
        self._thread.start()
        self._log(f"[{self.side}] 已连接 {self.can}（openarm_driver/{self.config_name}）")
        return True

    def disconnect(self) -> None:
        """由控制线程关闭总线，无额外的收臂或位置指令。"""
        with self._lock:
            self._running = False
        self._urgent_stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                self._log(f"[{self.side}] 控制线程尚未退出")
                return
            self._thread = None
        self._log(f"[{self.side}] 已断开")

    @property
    def connected(self) -> bool:
        return self._driver is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def gravity_active(self) -> bool:
        """重力补偿/关节阻抗/示教是否正在驱动这条臂（UI 据此停发滑块目标）。"""
        return self._hold_mode and self._enabled

    @property
    def _hold_mode(self) -> bool:
        """重力补偿 / 示教 / 关节阻抗 共用「软模式帧」：同一个前馈 τ=G+fric，
        只是 Kp/Kd 不同。三者互斥，由 UI 保证。"""
        return self._grav_on or self._imp_on

    @property
    def joint_impedance_active(self) -> bool:
        return self._imp_on and self._enabled

    @property
    def impedance_gains(self) -> tuple[tuple, tuple] | None:
        """最近一次交给 CAN 驱动的关节阻抗增益。"""
        frame = self._last_impedance_frame
        if not self.connected or not self.joint_impedance_active or frame is None:
            return None
        sent_at, kp, kd = frame
        if time.monotonic() - sent_at > 0.5:
            return None
        return kp, kd

    @property
    def supports_gripper_control(self) -> bool:
        return self.connected and self._grip_config_limits is not None

    @property
    def grasp_status(self) -> str:
        with self._lock:
            return self._grasp_status

    @property
    def grasp_torque_limit(self) -> float | None:
        """本次抓取电流上限对应的实测堵转力矩（N·m）；没有抓取时为 None。"""
        grasp = self._grasp
        return None if grasp is None else grasp.torque_limit

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

    def enable(self, done=None) -> bool:
        """使能并带重力补偿走起始轨迹，走位结束后调用 done(ok)。"""
        return self._push(_Cmd("enable", done))

    def disable(self, done=None) -> bool:
        """请求回程失能，在控制线程完成后调用 done(ok)。"""
        return self._push(_Cmd("disable", done))

    def estop(self) -> bool:
        """标记急停，控制线程在处理正常命令前优先切断力矩。"""
        self._urgent_stop.set()
        return True

    def set_targets(
        self, named: dict[str, float] | None, *, manual: bool = True, slider: bool = False,
    ) -> None:
        """只保留最新目标。None 取消目标并保持当前姿态。"""
        with self._lock:
            self._pending_targets = None if named is None else (named.copy(), manual, slider)
            if named is None:
                self._pending_hold = True

    def joint_feedback(self, *, require_enabled: bool = True) -> JointFeedback | None:
        """仅提供七轴均有近期实测反馈的状态。"""
        with self._lock:
            sample = self._joint_feedback
            if not self._running or (require_enabled and not self._enabled) or self._safety is not None:
                return None
        if sample is None or sample.expires_at <= time.monotonic():
            return None
        return sample

    def set_teleop(self, command: TeleopCommand | None) -> None:
        """刷新遥操命令。只保留最近一帧，None 表示停止跟随。"""
        with self._lock:
            self._teleop_command = command

    def set_gripper_force(self, percent: int) -> None:
        """设置配置电流上限的百分比，已开始的抓取沿用启动时的设置。"""
        with self._lock:
            self._grip_force_fraction = min(100, max(10, percent)) / 100.0

    def request_grasp(self) -> bool:
        with self._lock:
            if (not self.supports_gripper_control or not self._enabled or self._safety
                    or self._grasp_requested
                    or self._grasp_status in ("pending", "closing", "holding")):
                return False
            if self._pending_targets is not None:
                self._pending_targets[0].pop(f"openarm_{self.side}_finger_joint1", None)
            self._grasp_requested = True
            self._grasp_status = "pending"
            return True

    # ── 重力补偿（KDL 前馈，见 backend/gravity.py）──────────
    def set_gravity_comp(self, on: bool, scale: float = 1.0,
                         kp_floor: float = 30.0) -> None:
        mode = "teach" if kp_floor < 2.0 else "gravity"
        self.set_control_mode(mode if on else "position", scale=scale)

    def set_control_mode(
        self, mode: str, *, stiffness: float = 0.35, zeta: float = 1.0, scale: float = 1.0,
    ) -> None:
        """投递完整的力控模式。只在控制线程切换增益和保持目标。"""
        if mode not in ("position", "gravity", "teach", "impedance"):
            raise ValueError(f"未知力控模式：{mode}")
        with self._lock:
            self._pending_force_mode = (mode, stiffness, zeta, scale)

    def _apply_force_mode(self, drv) -> None:
        if self._power_motion is not None or self._preset_moves or self._teleop_role is not None:
            return
        with self._lock:
            request, self._pending_force_mode = self._pending_force_mode, None
            friction, self._pending_friction = self._pending_friction, None
        if friction is not None:
            self._fric_on = friction
        if request is None:
            return
        mode, stiffness, zeta, scale = request
        previous = ("impedance" if self._imp_on else "teach" if self._teach_mode
                    else "gravity" if self._grav_on else "position")
        self._imp_on = mode == "impedance"
        self._grav_on = mode in ("gravity", "teach")
        self._grav_kp = 0.5 if mode == "teach" else 70.0
        self._grav_scale = scale if self._grav_on else 1.0
        self._imp_stiff = float(np.clip(stiffness, self.IMP_STIFF_MIN, self.IMP_STIFF_MAX))
        self._imp_zeta = float(np.clip(zeta, self.IMP_ZETA_MIN, self.IMP_ZETA_MAX))
        if mode == previous:
            return
        self._teach_wrist_positions.clear()
        self._last_impedance_frame = None
        if mode == "position" or previous == "position":
            self._fric_tau = None
        if mode == "position":
            self._reset_gain_ramp()
        if self._enabled:
            self._hold_measured(drv)
        self._log(f"[{self.side}] 力控模式切换为 "
                  f"{ {'position': '位置保持', 'teach': '零重力示教', 'gravity': '重力补偿', 'impedance': '关节阻抗'}[mode]}")

    def _ensure_gravity(self) -> bool:
        """按需加载重力模型。"""
        if self._grav_model is not None:
            return True
        try:
            from .gravity import GravityModel
            self._grav_model = GravityModel.shared()
            self._log(f"[{self.side}] 重力模型就绪（{self._grav_model.backend}）")
            return True
        except Exception as exc:  # noqa: BLE001
            self._log(f"[{self.side}] 重力模型加载失败：{type(exc).__name__}: {exc}")
            return False

    # ── 关节空间阻抗（整条臂）────────────────────────────
    def set_joint_impedance(self, on: bool, stiffness: float = 0.35,
                            zeta: float = 1.0) -> None:
        self.set_control_mode("impedance" if on else "position", stiffness=stiffness, zeta=zeta)

    def _reset_gain_ramp(self) -> None:
        """离开软模式时清掉增益斜坡状态：回到位置控制后实际下发的是官方
        增益，下次再进软模式必须从官方值起步，不能拿旧的中间值当起点。"""
        self._imp_kp_cur = None
        self._imp_kd_cur = None
        self._imp_t = None
        self._last_kp = None
        self._last_kd = None
        self._soft_prev_tgt = None

    def _soft_gains(self, drv, teach: bool) -> tuple[tuple, tuple, bool]:
        """按当前软模式算 Kp/Kd，返回 (kps, kds, 是否发 dq 前馈)。

        三种模式共用一帧，只是目标增益不同：
        - 示教：极软刚度 + 纯阻尼，**不发速度前馈**（目标=实测时前馈会
          放大编码器噪声，见 _send_soft_frame 里的说明）
        - 关节阻抗：官方增益 × 档位系数，上下限裁剪
        - 重力补偿：官方增益 × kd 倍数（= 原有行为）

        所有模式的目标增益都经过**同一个斜坡**（_IMP_SLEW_*），从上一帧
        实际下发的值收敛过去。这样 示教→阻抗、很软→硬、补偿→阻抗 都不会
        在切档瞬间给臂一个刚度阶跃。
        """
        base_kp = np.asarray(getattr(drv, "kps", (70, 70, 70, 60, 10, 10, 10)),
                             dtype=float)[:7]
        base_kd = np.asarray(getattr(drv, "kds", (2.7, 2.7, 2.2, 2.2, 1.5, 1.5, 1.5)),
                             dtype=float)[:7]

        if teach:
            tgt_kp = np.full(7, 0.5)
            tgt_kd = np.array([2.7, 2.7, 2.2, 2.2, 1.5, 1.5, 1.5])
            if self._teleop_role is None:
                tgt_kd[4:] = _TEACH_WRIST_KD
            send_dq = False
        elif self._imp_on and self._power_motion is None:
            s = max(self.IMP_STIFF_MIN, min(self._imp_stiff, self.IMP_STIFF_MAX))
            z = max(self.IMP_ZETA_MIN, min(self._imp_zeta, self.IMP_ZETA_MAX))
            tgt_kp = np.clip(base_kp * s, IMP_KP_MIN, MIT_KP_MAX)
            # 阻尼比 ζ = Kd/(2√(Kp·J)) ⇒ 取 Kd ∝ √Kp，ζ 就只由 user 决定
            tgt_kd = np.clip(base_kd * self._grav_kd_scale * z * np.sqrt(s),
                             IMP_KD_MIN, MIT_KD_MAX)
            send_dq = True
        else:
            tgt_kp = np.minimum(base_kp, MIT_KP_MAX)
            tgt_kd = np.minimum(base_kd * self._grav_kd_scale, MIT_KD_MAX)
            send_dq = True

        if self._imp_kp_cur is None or self._imp_kd_cur is None:
            # 从上一帧实际下发的增益起步；没有历史（刚从位置控制进来）
            # 就用官方值——位置模式下发的是 base_kp/base_kd。
            src_kp = self._last_kp if self._last_kp is not None else base_kp
            src_kd = self._last_kd if self._last_kd is not None else base_kd
            self._imp_kp_cur = np.array(src_kp, dtype=float)
            self._imp_kd_cur = np.array(src_kd, dtype=float)
            self._imp_t = time.monotonic()
        now = time.monotonic()
        dt = min(max(now - (self._imp_t or now), 0.005), 0.05)
        self._imp_t = now
        self._imp_kp_cur += np.clip(tgt_kp - self._imp_kp_cur,
                                    -_IMP_SLEW_KP * dt, _IMP_SLEW_KP * dt)
        self._imp_kd_cur += np.clip(tgt_kd - self._imp_kd_cur,
                                    -_IMP_SLEW_KD * dt, _IMP_SLEW_KD * dt)
        self._last_kp = self._imp_kp_cur.copy()
        self._last_kd = self._imp_kd_cur.copy()
        return (tuple(self._imp_kp_cur), tuple(self._imp_kd_cur), send_dq)

    def set_friction_comp(self, on: bool) -> None:
        """投递摩擦补偿开关，由控制线程渐变输出力矩。"""
        with self._lock:
            self._pending_friction = bool(on)

    def _teach_wrist_friction_inputs(
        self, now: float, velocity: np.ndarray, fresh: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """用同向编码器位移稳定慢速估计，并给出附加助力的幅值。"""
        corrected = velocity.copy()
        assist = np.zeros(3)
        if len(self._teach_wrist_positions) < 3:
            return corrected, assist
        latest, last_at = self._teach_wrist_positions[-1]
        history = list(self._teach_wrist_positions)[:-1]

        for axis in range(3):
            if not fresh[axis] or not np.isfinite(latest[axis]):
                continue
            speed = abs(velocity[axis])
            if speed >= _TEACH_WRIST_ASSIST_MAX_RAD_S:
                continue
            newer_angle, newer_at = latest[axis], last_at[axis]
            travel, changes = 0.0, 0
            last_change_at = first_change_from = float("-inf")
            for angles, timestamps in reversed(history):
                angle, timestamp = angles[axis], timestamps[axis]
                if (not np.isfinite(angle) or not np.isfinite(timestamp)
                        or last_at[axis] - timestamp > _TEACH_WRIST_WINDOW_S):
                    break
                gap = newer_at - timestamp
                if gap < 0.001:
                    continue  # 同一电机回包被多次采样
                if gap > _FRIC_FEEDBACK_MAX_AGE_S:
                    break
                step = newer_angle - angle
                if step != 0:
                    if travel * step < 0:
                        break
                    if changes == 0:
                        last_change_at = newer_at
                    travel += step
                    changes += 1
                    first_change_from = timestamp
                newer_angle, newer_at = angle, timestamp
            if (changes < 2 or abs(travel) < _TEACH_WRIST_CONFIRM_RAD
                    or now - last_change_at >= _TEACH_WRIST_STOP_S):
                continue

            direction = np.sign(travel)
            if velocity[axis] * direction < -_FRIC_DEADBAND_RAD_S[axis + 4]:
                continue  # 速度反馈已经明确显示反向
            position_speed = travel / (last_change_at - first_change_from)
            if abs(position_speed) >= _TEACH_WRIST_ASSIST_MAX_RAD_S:
                continue
            fade_width = _TEACH_WRIST_ASSIST_MAX_RAD_S - _TEACH_WRIST_ASSIST_FULL_RAD_S
            raw_weight = np.clip((speed - _TEACH_WRIST_ASSIST_FULL_RAD_S) / fade_width, 0.0, 1.0)
            blended_speed = (1.0 - raw_weight) * position_speed + raw_weight * velocity[axis]
            fade = np.clip((abs(blended_speed) - _TEACH_WRIST_ASSIST_FULL_RAD_S) / fade_width,
                           0.0, 1.0)
            scale = 1.0 - fade ** 2 * (3.0 - 2.0 * fade)
            if self._lim_lo is not None and self._lim_hi is not None:
                joint = axis + 4
                clearance = (self._lim_hi[joint] - latest[axis] if direction > 0
                             else latest[axis] - self._lim_lo[joint])
                if not clearance > 0.0:
                    continue
                scale *= min(clearance / _TEACH_WRIST_LIMIT_MARGIN_RAD, 1.0)
            corrected[axis] = blended_speed
            assist[axis] = scale * _TEACH_WRIST_ASSIST_NM[axis]
        return corrected, assist

    def _friction_tau(
        self, now: float, dt: float, *, wrist_kd: tuple[float, ...] | None = None,
    ) -> np.ndarray:
        """由当前有效转速限制助力，再平滑建立或撤回补偿。"""
        if self._fric_feedback is None or self._grav_model is None:
            self._fric_tau = None
            return np.zeros(7)
        if not self._fric_on and self._fric_tau is None:
            return np.zeros(7)

        measured, expires_at = self._fric_feedback
        fresh = (now < expires_at) & np.isfinite(measured)
        velocity = np.where(fresh, measured, 0.0)
        wrist_assist = None
        if wrist_kd is not None:
            wrist_velocity, wrist_assist = self._teach_wrist_friction_inputs(now, velocity[4:], fresh[4:])
            velocity[4:] = wrist_velocity
        direction = np.sign(velocity)
        # 从死区边缘平滑过渡到标定模型，压低近零速度的量化噪声。
        speed = np.abs(velocity)
        ramp_width = 2.0 * _FRIC_DEADBAND_RAD_S
        if wrist_kd is not None:
            ramp_width[4:] *= 0.5
        motion = np.clip((speed - _FRIC_DEADBAND_RAD_S) / ramp_width, 0.0, 1.0)
        weight = motion ** 2 * (3.0 - 2.0 * motion)
        model_tau = self._grav_model.friction(self.side, velocity)
        magnitude = np.clip(np.where(np.isfinite(model_tau), direction * model_tau, 0.0),
                            0.0, _FRIC_LIMIT_NM) * weight
        if wrist_kd is not None:
            # 按本帧实际下发的阻尼限制助力，阻尼斜坡期间也保留制动力。
            wrist_damping = np.asarray(wrist_kd) * speed[4:]
            magnitude[4:] = np.minimum(magnitude[4:], _TEACH_WRIST_FRIC_FRACTION * wrist_damping)
            magnitude[4:] += wrist_assist

        previous = np.zeros(7) if self._fric_tau is None else self._fric_tau
        # 丢弃上一方向的助力。减速、静止或反馈失效时，旧力矩不能超过当前允许值。
        previous = direction * np.clip(direction * previous, 0.0, magnitude)
        target = direction * magnitude if self._fric_on else np.zeros(7)
        dt = min(max(dt, 0.0), _COMMAND_PERIOD_S)
        alpha = -np.expm1(-dt / _FRIC_FILTER_S)
        step_limit = _FRIC_SLEW_NM_S * dt
        output = previous + np.clip(alpha * (target - previous), -step_limit, step_limit)
        output = np.where((target == 0.0) & (np.abs(output) < 0.002), 0.0, output)
        self._fric_tau = output if output.any() else None
        return output

    @property
    def _joint_target_speed_limits(self) -> np.ndarray:
        if self._power_motion == "enable":
            speed = _ENABLE_JOINT_SPEED_RAD_S
        elif self._power_motion == "disable":
            speed = _DISABLE_JOINT_SPEED_RAD_S
        elif self._teleop_role == "follower":
            speed = _TELEOP_FOLLOWER_SPEED_LIMITS_RAD_S
        elif self._slider_target:
            speed = SLIDER_JOINT_SPEED_LIMITS_RAD_S
        else:
            speed = MANUAL_JOINT_SPEED_RAD_S
        limits = np.broadcast_to(speed, (7,))
        return np.minimum(limits, self._vel_limits) if self._vel_limits is not None else limits

    def _reset_slider_motion(self) -> None:
        self._slider_target = False
        self._slider_velocity.fill(0.0)

    def _slider_joint_step(
        self, previous: np.ndarray, delta: np.ndarray, dt: float, vmax: np.ndarray,
    ) -> np.ndarray:
        """用加速度上限起步，并按剩余制动距离减速。改目标时沿用当前速度。"""
        if dt == 0.0:
            return previous.copy()
        if self._delta_limits is not None:
            vmax = np.minimum(vmax, self._delta_limits[:7] / dt)
        accel_step = _SLIDER_ACCEL_RAD_S2 * dt
        brake_step = _SLIDER_BRAKE_RAD_S2 * dt
        # 为下一帧位移和后续制动留余量。
        braking_speed = np.sqrt(brake_step ** 2 + 2 * _SLIDER_BRAKE_RAD_S2 * np.abs(delta)) - brake_step
        desired = np.sign(delta) * np.minimum(vmax, braking_speed)
        current_velocity = self._slider_velocity
        reversing = current_velocity * desired < 0.0
        desired = np.where(reversing, 0.0, desired)
        slowing = current_velocity * (desired - current_velocity) < 0.0
        speed_change = np.where(slowing, brake_step, accel_step)
        velocity = np.clip(current_velocity + np.clip(
            desired - current_velocity, -speed_change, speed_change), -vmax, vmax)
        # 本帧能到位且下一帧能按制动上限停住时，直接走完最后一小段。
        arrival_velocity = delta / dt
        settled = (~reversing
                   & (np.abs(arrival_velocity) <= np.minimum(brake_step, vmax))
                   & (np.abs(arrival_velocity - current_velocity) <= accel_step))
        step = np.where(settled, delta, velocity * dt)
        position = previous + step
        if self._lim_lo is not None and self._lim_hi is not None:
            lower = np.minimum(previous, self._lim_lo[:7])
            upper = np.maximum(previous, self._lim_hi[:7])
            position = np.clip(position, lower, upper)
        self._slider_velocity = (position - previous) / dt
        return position

    def _manual_joint_step(self, previous, target, dt: float) -> np.ndarray:
        """限制单帧关节位移，接近目标时逐步减速。"""
        delta = np.asarray(target) - previous
        vmax = self._joint_target_speed_limits
        dt = min(max(dt, 0.0), _COMMAND_PERIOD_S)
        if self._slider_target:
            return self._slider_joint_step(previous, delta, dt, vmax)
        # 从臂目标受实测位置 ±0.08 rad 限制，手动走位的减速系数会压低追赶速度。
        approach_rate = _TELEOP_FOLLOWER_APPROACH_RATE_S if self._teleop_role == "follower" else 8.0
        rate = np.minimum(vmax, approach_rate * np.abs(delta))
        step = np.minimum(vmax * dt, np.maximum(rate * dt, 0.0001))
        return previous + np.clip(delta, -step, step)

    def _clamp_hold_target(self, target, previous, dt: float, *, manual: bool) -> np.ndarray:
        target = np.asarray(target, dtype=float)
        if self._lim_lo is not None and self._lim_hi is not None:
            target = np.clip(target, self._lim_lo[:7], self._lim_hi[:7])
        if manual:
            return self._manual_joint_step(previous, target, dt)
        vmax = self._vel_limits if self._vel_limits is not None else 2.0
        step = vmax * dt
        return previous + np.clip(target - previous, -step, step)

    def _apply_gripper_limits(self, drv) -> None:
        if self._grip_config_limits is None:
            return
        speed, nominal = self._grip_config_limits
        with self._lock:
            fraction = self._grip_force_fraction
        grasp = self._grasp
        if grasp is not None:
            drv.gripper_posforce_limits = (min(speed, grasp.speed_limit), grasp.nominal_limit)
        else:
            drv.gripper_posforce_limits = (min(speed, 1.0), nominal * fraction)

    def _read_gripper_feedback(self, drv) -> GripperFeedback | None:
        try:
            gripper = drv.openarm.get_gripper()
            stats = gripper.get_link_stats(0)
            if stats.is_stale(0.10):
                return None
            motor = gripper.get_motor()
            position = motor.get_position() - drv.joint_offsets[7]
            velocity = motor.get_velocity()
            torque = motor.get_torque()
            if (not motor.is_enabled() or not np.isfinite([position, velocity, torque]).all()
                    or not -_GRIP_RAD_SPAN - 0.05 <= position <= 0.10):
                return None
            return GripperFeedback(position, velocity, torque, stats.responses)
        except (AttributeError, IndexError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self._log(f"[{self.side}] 夹爪反馈读取失败：{type(exc).__name__}: {exc}")
            return None

    def _update_grasp(self, drv, now: float) -> None:
        if self._grip_config_limits is None:
            return
        with self._lock:
            requested, self._grasp_requested = self._grasp_requested, False
            fraction = self._grip_force_fraction
        busy = bool(self._preset_moves or self._power_motion or self._teleop_role)
        phase = self._grasp.phase if self._grasp is not None else "idle"
        monitor = phase == "holding" or (not busy and (requested or phase == "closing"))
        feedback = self._read_gripper_feedback(drv) if monitor else None
        if requested:
            if feedback is not None:
                # 闭合全程受面板电流上限约束，夹住后电机自己按这个上限保持。
                self._grasp = Grasp(now, feedback, self._grip_config_limits[1] * fraction)
                if self._targets is None:
                    self._hold_measured(drv)
            else:
                phase = "cancelled" if busy else "feedback_lost"
                if self._grasp is not None:
                    self._grasp.finish(phase)
                with self._lock:
                    self._grasp_status = phase
        grasp = self._grasp
        if grasp is not None:
            if busy and grasp.phase == "closing":
                grasp.finish("cancelled")
            elif grasp.phase in ("closing", "holding"):
                grasp.update(now, feedback)
            with self._lock:
                self._grasp_status = grasp.phase
        self._apply_gripper_limits(drv)

    def _with_grasp_target(self, target: np.ndarray) -> np.ndarray:
        if self._grasp is None:
            return target
        command = target.copy()
        command[7] = np.clip(self._grasp.target, -_GRIP_RAD_SPAN, 0.0)
        return self._bounded_target(command)

    def _send_gripper(self, drv, pos: dict) -> None:
        """夹爪（第 8 轴）单独走 POS_FORCE：重力帧/导纳帧都只发 7 个臂关节，
        不补发的话补偿期间夹爪滑块完全没响应。"""
        if not getattr(drv, "gripper_posforce", True):
            return
        prefix = f"openarm_{self.side}_"
        with self._lock:
            # _targets[7] 已经是**电机弧度**（_to_array 转换过），直接用，
            # 不能再按米换算——否则双重换算会得到 +24.9 rad 的荒谬目标。
            grip_tgt = None if self._targets is None else float(self._targets[7])
            grip_meas_m = float(pos.get(f"{prefix}finger_joint1", 0.0))
        rad = grip_tgt if grip_tgt is not None else _gripper_rad(grip_meas_m)
        if self._grasp is not None:
            rad = float(np.clip(self._grasp.target, -_GRIP_RAD_SPAN, 0.0))
            if self._lim_lo is not None and self._lim_hi is not None:
                rad = float(np.clip(rad, self._lim_lo[7], self._lim_hi[7]))
        # 与 openarm_driver::send_position 相同的换算：上限 1.0 对应电流标幺值 0.222，
        # 实测堵转力矩约 2.1 N·m（见 backend/gripper.py）。
        speed, limit = drv.gripper_posforce_limits
        drv.openarm.get_gripper().set_position(
            rad + drv.joint_offsets[7],
            speed_rad_s=float(speed), torque_pu=float(limit) / DRIVER_NM_PER_PU)

    def _send_soft_frame(self, drv, dt: float) -> np.ndarray:
        """发一帧「G(q)+τ_fric 前馈 + 可调 PD」MIT。

        重力补偿、示教、关节阻抗共用这一帧，区别只在 _soft_gains() 给的
        Kp/Kd：位置目标都取进入时锁存的姿态（拖动模式跟实测角）。
        """
        import openarm_can as oa

        prefix = f"openarm_{self.side}_"
        teach = self._teach_mode
        with self._lock:
            pos = dict(self._positions)
        baseline = drv.last_command
        measured = [float(pos.get(f"{prefix}joint{i}", baseline[i - 1]))
                    for i in range(1, 8)]
        q_tgt = np.asarray(measured)
        if not teach and self._grav_hold_q is not None:
            previous = (measured if self._soft_prev_tgt is None
                        else self._soft_prev_tgt)
            q_tgt = self._clamp_hold_target(
                self._grav_hold_q, np.asarray(previous), dt, manual=self._manual_target)
        names = self._grav_model.basis(self.side)
        q_chain = [float(pos.get(n, 0.0)) for n in names]
        # 符号：KDL/MuJoCo 的 JntToGravity/qfrc_bias 给出的是"平衡重力所需
        # 施加的关节力矩"，与电机输出同号（实测位置保持时 tau≈+G，逐项吻合）。
        # 官方 openarm_teleop 同样直接下发不取负。
        tau_raw = self._grav_model.gravity(self.side, q_chain) * self._grav_scale
        # 一阶低通：实测关节角带编码器噪声，直接当前馈会让电机持续抖动
        if self._grav_tau is None or self._grav_tau.shape != tau_raw.shape:
            self._grav_tau = tau_raw.copy()
        else:
            self._grav_tau += (tau_raw - self._grav_tau) * 0.25
        # 增益按模式取（示教 / 关节阻抗 / 重力补偿），见 _soft_gains()。
        # 关键：必须**按关节**取官方配置值（J1/J2=70、J3/J4=60、J5~J7=10）。
        # 之前统一发 70，末端三个 DM4310 小电机的刚度超标 7 倍——
        # 小惯量+大刚度的高频振荡，表现为"只有末端三关节震"。
        impedance = self._imp_on and not teach
        kps, kds, send_dq = self._soft_gains(drv, teach)
        wrist_kd = kds[4:] if teach and self._teleop_role is None else None
        now = time.monotonic()
        tau = self._grav_tau + self._friction_tau(now, dt, wrist_kd=wrist_kd)
        # 速度前馈：目标在移动时把"目标变化率"作为 MIT 的 dq 下发。
        # **零重力模式必须关闭**：那时目标就等于实测角，"目标速度"
        # 等于实测速度，kd*(dq_d - q̇)≈0 完全没有阻尼，还把编码器噪声
        # 直接放大成力矩——表现就是手推时剧烈抖动。
        dq = [0.0] * 7
        if send_dq and self._grav_prev_q is not None \
                and self._grav_prev_t is not None:
            dt = max(now - self._grav_prev_t, 0.005)
            vmax = self._joint_target_speed_limits if self._manual_target else 3.0
            dq = np.clip((q_tgt - self._grav_prev_q) / dt, -vmax, vmax)
        self._grav_prev_q = list(q_tgt)
        self._grav_prev_t = now
        # q_tgt 是**关节空间**（已减掉 joint_offsets），MIT 帧要电机空间，
        # 和官方 send_position 一样把偏移加回去。pedestal 配置偏移全 0，
        # 但切到 cell 配置（有 ±1.57 等工装偏移）时不加就会整体偏一个角度。
        off = drv.joint_offsets
        drv.openarm.get_arm().mit_control_all([
            oa.MITParam(kps[i], kds[i], q_tgt[i] + off[i], dq[i], float(tau[i]))
            for i in range(7)
        ])
        if impedance and self._imp_on:
            self._last_impedance_frame = (time.monotonic(), kps, kds)
        # 夹爪（第 8 轴）不能丢：重力帧只发 7 关节，若不在此处补发，
        # 重力补偿期间夹爪滑块完全无响应。
        self._send_gripper(drv, pos)
        # 重力模式下只用短超时读帧：4ms 会拖慢 50Hz 循环（变成 23ms 抖动），
        # 也是"臂一直在抖"的来源之一。状态另由 100Hz 轮询补齐。
        drv.openarm.recv_all(800)
        # 频率自检：UI 里若被其他指令挤占，这里能直接看出来
        self._grav_n += 1
        now = time.monotonic()
        if self._grav_t0 is None:
            self._grav_t0 = now
        elif now - self._grav_t0 >= 3.0:
            hz = self._grav_n / (now - self._grav_t0)
            self._log(f"[{self.side}] 软模式帧 {hz:.0f}Hz（目标 50Hz）")
            self._grav_n, self._grav_t0 = 0, now
        # driver 内部基准保持 8 维（7 关节 + 夹爪），否则切回位置控制时
        # _step_toward 的数组形状与 send_position 不匹配
        with self._lock:
            finger = float(pos.get(f"{prefix}finger_joint1", 0.0))
        finger_rad = _gripper_rad(finger)
        drv.last_command = np.array(measured + [finger_rad])
        drv.last_command_time_s = time.monotonic()
        self._soft_prev_tgt = None if teach else q_tgt.copy()
        return q_tgt

    def smooth_move(self, named: dict[str, float], hz: float = 50.0,
                    duration: float = 2.0, done=None) -> bool:
        """插值运动（官方 smooth_move，走完整安全链）。阻塞控制线程。"""
        return self._push(_Cmd("smooth", (self._to_array(named), hz, duration, done)))

    def move_start(self, done=None) -> bool:
        """以手动限速移动到配置中的起始位。"""
        return self._push(_Cmd("goto_start", done))

    def move_park(self, done=None) -> bool:
        """以手动限速移动到配置中的收臂位。"""
        return self._push(_Cmd("goto_stop", done))

    def _finish_preset(self, ok: bool) -> None:
        self._preset_moves.clear()
        self._power_motion = None
        self._power_wait_since = None
        self._power_still_since = None
        self._playback_until = None
        done, self._goto_done = self._goto_done, None
        if done is not None:
            try:
                done(ok)
            except Exception as exc:  # noqa: BLE001
                self._log(f"[{self.side}] 预设位置回调出错：{exc}")

    # ── 单位换算 ──────────────────────────────────────
    def _to_array(self, named: dict[str, float]) -> np.ndarray:
        """关节名字典 → driver 的 8 维数组（夹爪 m→rad）。

        缺省关节的填充顺序：**上一拍下发值 → 实测角 → 0**。
        不能用实测角持续填充：50Hz 重发部分目标时"目标=滞后的实测角"
        会构成正反馈（PD 伺服追自己的编码器），臂会持续震荡；
        实测角只在从未下发过目标时播种一次。
        """
        prefix = f"openarm_{self.side}_"
        with self._lock:
            measured = dict(self._positions)
            last_cmd = None if self._targets is None \
                else np.array(self._targets, dtype=float)

        def fill(name: str, idx: int) -> float:
            if name in named:
                return float(named[name])
            if last_cmd is not None:
                return float(last_cmd[idx])
            return float(measured.get(name, 0.0))

        arr = np.zeros(8)
        for i in range(1, 8):
            arr[i - 1] = fill(f"{prefix}joint{i}", i - 1)
        # 夹爪：named/实测是米（需换算）；last_cmd 已是电机弧度，直接用
        gname = f"{prefix}finger_joint1"
        if gname in named or last_cmd is None:
            gm = float(named.get(gname, measured.get(gname, 0.0)))
            arr[7] = _gripper_rad(gm)
        else:
            arr[7] = float(last_cmd[7])
        return arr

    def _to_named(self, arr: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
        prefix = f"openarm_{self.side}_"
        pos, tau = {}, {}
        for i in range(1, 8):
            pos[f"{prefix}joint{i}"] = float(arr[i - 1])
            tau[f"{prefix}joint{i}"] = float(self._tau_arr[i - 1]) \
                if self._tau_arr is not None else 0.0
        rad = float(arr[7])
        pos[f"{prefix}finger_joint1"] = min(
            GRIP_OPEN_M, max(0.0, _GRIP_SIGN * rad / _GRIP_RAD_SPAN * GRIP_OPEN_M))
        return pos, tau

    # ── 控制线程 ──────────────────────────────────────
    def _start_with_gravity(self, drv, before: tuple[int, ...], deadline: float) -> bool:
        """确认七轴在本次使能中应答并上力矩，再按实测姿态发第一帧。"""
        import openarm_can as oa

        axes = drv.openarm.get_arm()
        for attempt in range(_START_FEEDBACK_POLLS):
            if self._urgent_stop.is_set():
                return False
            issues = []
            faulted = False
            motors = axes.get_motors()
            for index, count in enumerate(before):
                link = axes.get_link_stats(index)
                if link.responses <= count or link.is_stale(_START_FEEDBACK_TIMEOUT_S):
                    issues.append(f"J{index + 1} 无新反馈")
                    continue
                motor = motors[index]
                if not motor.is_enabled():
                    status = oa.motor_error_to_string(motor.get_error_code())
                    issues.append(f"J{index + 1} {status}")
                    faulted = faulted or motor.has_error()
            if not issues:
                break
            if (faulted or attempt + 1 == _START_FEEDBACK_POLLS
                    or time.monotonic() >= deadline):
                self._log(f"[{self.side} {self.can}] 使能中止，" + "、".join(issues))
                return False
            drv.openarm.refresh_all()
            drv.set_latest_state(timeout_us=self.RECV_US)
            self._urgent_stop.wait(_START_FEEDBACK_POLL_INTERVAL_S)

        measured = np.asarray(drv.latest_state["qpos"], dtype=float)
        if measured.shape != (8,) or not np.isfinite(measured).all():
            self._log(f"[{self.side}] 使能后关节反馈无效，中止启动")
            return False
        self._publish(drv.latest_state)
        with self._lock:
            self._targets = measured.copy()
            self._grav_hold_q = None
            self._grav_prev_q = self._grav_prev_t = None
            self._grav_tau = self._fric_tau = None
            self._fric_feedback = None
            self._teach_wrist_positions.clear()
            self._reset_gain_ramp()
        drv.last_command = measured.copy()
        if self._urgent_stop.is_set():
            return False
        self._send_soft_frame(drv, _COMMAND_PERIOD_S)
        self._grav_hold_q = measured[:7].tolist()
        self._last_send_t = time.monotonic()
        return True

    def _travel_to_stop(self, drv) -> bool:
        """普通位置模式的失能回程，沿用驱动配置并逐帧补读反馈。"""
        moves = drv.config.get_stop_config().get("moves", [])
        if not moves:
            return True
        for move in moves:
            target = np.asarray(move["position"][drv.arm_side], dtype=float)
            if not self._position_segment(drv, target, float(move.get("hz", 50)),
                                          float(move.get("duration", 2))):
                return False
        return True

    def _position_segment(self, drv, target: np.ndarray, hz: float, duration: float) -> bool:
        base = np.asarray(drv.last_command, dtype=float)
        steps = max(1, int(hz * duration))
        for index in range(steps):
            if self._urgent_stop.is_set():
                return False
            point = base + (target - base) * ((index + 1) / steps)
            if drv.send_position(self._with_grasp_target(point)) is False:
                return False
            drv.openarm.recv_all(self.RECV_US)
            self._publish(drv.latest_state)
            self._urgent_stop.wait(1.0 / hz)
        return not self._urgent_stop.is_set()

    def _stop_now(self, drv) -> None:
        self._urgent_stop.clear()
        try:
            drv.openarm.disable_all()
        except Exception as exc:  # noqa: BLE001
            self._log(f"[{self.side}] 急停下发失败：{exc}")
        self._clear_motion_after_disable(drv)
        self._finish_preset(False)
        while True:
            try:
                cmd = self._cmds.get_nowait()
            except queue.Empty:
                break
            self._reject_motion_cmd(cmd)
        self._log(f"[{self.side}] 已请求立即失能")

    def _loop(self) -> None:
        while True:
            with self._lock:
                run = self._running
                drv = self._driver
            if drv is None:
                return
            if not run:
                self._stop_now(drv)
                with self._lock:
                    self._driver = None
                return
            try:
                # 1) 先执行命令（可能阻塞：enable 起始轨迹 / smooth 插值）
                busy = self._drain_cmds(drv) or self._send_current_target(drv)
                now = time.monotonic()
                if not busy and now >= self._next_poll_t:
                    self._poll_feedback(drv, now)
                if now - self._health_t > 1.0:
                    self._health_t = now
                    self._snapshot_health(drv)
                self._tick_rate()
            except Exception as exc:  # noqa: BLE001
                self._log(f"[{self.side}] 控制线程异常退出：{type(exc).__name__}: {exc}")
                self._stop_now(drv)
                with self._lock:
                    self._running = False
                    self._driver = None
                return
            time.sleep(0.002)

    def _poll_feedback(self, drv, now: float) -> None:
        idle = not self._enabled
        if idle:
            before = drv.openarm.get_bus_status().write_no_buffer.count
        drv.openarm.refresh_all()
        drv.set_latest_state(timeout_us=self.RECV_US)
        self._publish(drv.latest_state)
        self._sample_joint_feedback(drv)
        period = _POLL_ACTIVE_PERIOD_S
        if idle:
            after = drv.openarm.get_bus_status().write_no_buffer.count
            blocked = after > before
            if blocked and not self._idle_poll_blocked:
                self._log(f"[{self.side} {self.can}] 发送队列满，暂缓失能状态查询")
            self._idle_poll_blocked = blocked
            period = _POLL_BLOCKED_PERIOD_S if blocked else _POLL_IDLE_PERIOD_S
        self._next_poll_t = now + period

    def _load_preset(self, drv, config, done) -> None:
        self._preset_moves.extend(
            self._bounded_target(np.asarray(move["position"][drv.arm_side], dtype=float))
            for move in config.get("moves", [])
        )
        self._goto_done = done
        self._manual_target = True
        self._reset_slider_motion()
        self._playback_until = None
        if self._preset_moves:
            self._set_motion_target(self._preset_moves[0])

    def _finish_disable(self, drv, *, at_park: bool = False) -> None:
        if at_park:
            # 软模式已经走完回程。保留驱动的断力矩及状态收尾，跳过重复的普通位置回程。
            stop_hook = drv._on_stop
            drv._on_stop = lambda: True
            try:
                drv.stop()
            finally:
                drv._on_stop = stop_hook
        else:
            drv.stop()
        self._clear_motion_after_disable(drv)
        self._finish_preset(not self._urgent_stop.is_set())
        self._log(f"[{self.side}] 已失能")

    def _clear_motion_after_disable(self, drv) -> None:
        with self._lock:
            self._grasp = None
            self._grasp_requested = False
            self._grasp_status = "idle"
            self._enabled = False
            self._targets = None
            self._reset_slider_motion()
            self._pending_force_mode = self._pending_targets = None
            self._pending_hold = False
            self._grav_on = self._imp_on = False
            self._grav_hold_q = None
            self._last_impedance_frame = None
            self._joint_feedback = None
            if self._teleop_command is not None:
                label = "左臂" if self.side == "left" else "右臂"
                self._teleop_command.watchdog.stop(f"{label}已失能")
            self._teleop_role = self._teleop_command = None
            self._grav_prev_q = self._grav_prev_t = None
            self._grav_tau = self._fric_tau = None
            self._fric_feedback = None
            self._teach_wrist_positions.clear()
            self._reset_gain_ramp()

    def _settle_power_motion(self, drv, goal: np.ndarray, now: float) -> None:
        if self._power_wait_since is None:
            self._power_wait_since = now
        prefix = f"openarm_{self.side}_"
        names = [f"{prefix}joint{i}" for i in range(1, 8)]
        with self._lock:
            unsettled = [
                f"J{index + 1}" for index, (name, target) in enumerate(zip(names, goal[:7]))
                if not (abs(self._positions.get(name, float("inf")) - target) <= _POWER_POSITION_TOL_RAD
                        and abs(self._velocities.get(name, float("inf"))) <= _POWER_VELOCITY_TOL_RAD_S)
            ]
        axes = drv.openarm.get_arm()
        stale = [f"J{i + 1}" for i in range(7)
                 if axes.get_link_stats(i).is_stale(_POWER_FEEDBACK_MAX_AGE_S)]
        if not unsettled and not stale:
            if self._power_still_since is None:
                self._power_still_since = now
            if now - self._power_still_since >= _POWER_STABLE_S:
                if self._power_motion == "disable":
                    self._finish_disable(drv, at_park=True)
                else:
                    self._finish_preset(True)
                    self._log(f"[{self.side}] 已到起始位，继续发送重力前馈")
                return
        else:
            self._power_still_since = None
        if now - self._power_wait_since >= _POWER_TIMEOUT_S:
            name = "收臂位" if self._power_motion == "disable" else "起始位"
            details = []
            if stale:
                details.append(f"{'、'.join(stale)} 反馈超时")
            if unsettled:
                details.append(f"{'、'.join(unsettled)} 未到位或未静止")
            self._log(f"[{self.side} {self.can}] {name}反馈未稳定，"
                      f"{'，'.join(details) or '稳定时间不足'}，保留使能和重力前馈")
            self._finish_preset(False)

    def _enable_with_gravity(self, drv, done) -> bool:
        if self._enabled:
            if done is not None:
                done(True)
            return False
        self._goto_done = done
        if not self._ensure_gravity():
            self._clear_motion_after_disable(drv)
            self._finish_preset(False)
            return False

        if self._urgent_stop.is_set():
            self._stop_now(drv)
            return True

        if not self._hold_mode:
            self._grav_on = True
            self._grav_kp = 70.0
            self._grav_scale = 1.0
        self._apply_gripper_limits(drv)
        self._power_motion = "enable"
        self._load_preset(drv, drv.config.get_start_config(), done)
        try:
            axes = drv.openarm.get_arm()
            before = tuple(axes.get_link_stats(i).responses for i in range(7))
            deadline = time.monotonic() + _START_FEEDBACK_TIMEOUT_S
            drv._on_start = lambda: self._start_with_gravity(drv, before, deadline)
            ok = drv.start()
        except Exception:
            try:
                drv.openarm.disable_all()
            finally:
                self._clear_motion_after_disable(drv)
            raise

        if self._urgent_stop.is_set():
            self._stop_now(drv)
            return True
        if not ok:
            if drv.safety_stop_reason is None:
                drv.openarm.disable_all()
            self._check_safety(False, drv)
            self._clear_motion_after_disable(drv)
            self._finish_preset(False)
            return True

        with self._lock:
            self._enabled = True
        self._check_safety(True, drv)
        self._next_poll_t = 0.0
        self._idle_poll_blocked = False
        self._log(f"[{self.side}] 已使能并保持重力前馈")
        if not self._preset_moves:
            self._finish_preset(True)
        return True

    def _reject_motion_cmd(self, cmd: _Cmd) -> None:
        done = cmd.payload[3] if cmd.kind == "smooth" else cmd.payload
        if done is not None:
            try:
                done(False)
            except Exception as exc:  # noqa: BLE001
                self._log(f"[{self.side}] 取消 {cmd.kind} 后回调出错：{exc}")

    def _drain_cmds(self, drv) -> bool:
        """执行队列里的命令。返回 True=刚做过耗时操作（状态已随之刷新）。"""
        if self._urgent_stop.is_set():
            self._stop_now(drv)
            return True
        self._apply_force_mode(drv)
        try:
            cmd = self._cmds.get_nowait()
        except queue.Empty:
            return False

        if self._power_motion is not None:
            self._reject_motion_cmd(cmd)
            return False

        if cmd.kind == "enable":
            return self._enable_with_gravity(drv, cmd.payload)
        if cmd.kind == "disable":
            if self._grasp is not None and self._grasp.phase == "closing":
                self._grasp.finish("cancelled")
            with self._lock:
                self._grasp_requested = False
                self._pending_targets = None
                self._pending_hold = False
            self.set_teleop(None)
            self._update_teleop(drv, time.monotonic())
            self._finish_preset(False)
            self._goto_done = cmd.payload
            if not self._enabled:
                self._clear_motion_after_disable(drv)
                self._finish_preset(True)
                return False
            if self._hold_mode and drv.safety_stop_reason is None:
                if not self._ensure_gravity():
                    self._finish_preset(False)
                    return False
                self._load_preset(drv, drv.config.get_stop_config(), cmd.payload)
                if self._preset_moves:
                    self._power_motion = "disable"
                    self._log(f"[{self.side}] 保持重力前馈，缓慢走向收臂位")
                    return False
                self._finish_disable(drv, at_park=True)
            else:
                self._finish_disable(drv)
            return True
        if cmd.kind == "smooth":
            tgt, hz, dur, done = cmd.payload
            ok = False
            if self._enabled and not (self._preset_moves or self._hold_mode or self._teleop_role):
                ok = self._position_segment(drv, tgt, hz, dur)
            if self._urgent_stop.is_set():
                self._stop_now(drv)
                ok = False
            self._check_safety(ok, drv)
            self._log(f"[{self.side}] smooth_move {'完成' if ok else '被安全链中断'}")
            if done is not None:
                done(bool(ok))
            with self._lock:
                self._targets = tgt if ok else None
            return True
        if cmd.kind in ("goto_start", "goto_stop"):
            done = cmd.payload
            if not self._enabled or self._preset_moves or self._teach_mode or self._teleop_role is not None:
                if done is not None:
                    done(False)
                return False
            conf = (drv.config.get_start_config() if cmd.kind == "goto_start"
                    else drv.config.get_stop_config())
            with self._lock:
                self._pending_targets = None
                self._pending_hold = False
            self._load_preset(drv, conf, done)
            if not self._preset_moves:
                self._finish_preset(True)
        return False

    def _bounded_target(self, target: np.ndarray) -> np.ndarray:
        if self._lim_lo is not None and self._lim_hi is not None:
            return np.clip(target, self._lim_lo, self._lim_hi)
        return target

    @property
    def _teach_mode(self) -> bool:
        return self._grav_on and self._grav_kp < 2.0 and not self._imp_on and self._power_motion is None

    def _set_motion_target(self, target: np.ndarray) -> None:
        target = self._bounded_target(target)
        with self._lock:
            self._targets = target
            if self._hold_mode:
                self._grav_hold_q = target[:7].tolist()

    def _hold_measured(self, drv) -> None:
        target = np.asarray(drv.last_command, dtype=float).copy()
        prefix = f"openarm_{self.side}_"
        with self._lock:
            positions = dict(self._positions)
        for index in range(7):
            target[index] = positions.get(f"{prefix}joint{index + 1}", target[index])
        grip_name = f"{prefix}finger_joint1"
        if grip_name in positions:
            target[7] = _gripper_rad(positions[grip_name])
        if np.isfinite(target).all():
            self._grav_prev_q = self._grav_prev_t = None
            self._soft_prev_tgt = target[:7].copy()
            self._manual_target = True
            self._reset_slider_motion()
            self._playback_until = None
            if not self._hold_mode:
                drv.last_command = self._bounded_target(target)
            self._set_motion_target(target)

    def _apply_live_targets(self, drv, now: float) -> None:
        with self._lock:
            pending, self._pending_targets = self._pending_targets, None
            hold, self._pending_hold = self._pending_hold, False
        if self._preset_moves or self._teleop_role is not None:
            return
        expired = self._playback_until is not None and now >= self._playback_until
        if hold or expired:
            self._hold_measured(drv)
        if pending is None:
            return
        named, manual, slider = pending
        if self._teach_mode:
            named = {name: value for name, value in named.items() if "finger" in name}
        if named:
            target = self._to_array(named)
            if np.isfinite(target).all():
                if f"openarm_{self.side}_finger_joint1" in named:
                    self._grasp = None
                    with self._lock:
                        self._grasp_requested = False
                        self._grasp_status = "idle"
                self._manual_target = manual
                if manual and slider and not self._teach_mode:
                    self._slider_target = True
                elif self._slider_target:
                    self._reset_slider_motion()
                self._playback_until = None if manual else now + _PLAYBACK_TIMEOUT_S
                self._set_motion_target(target)

    def _update_teleop(self, drv, now: float) -> None:
        with self._lock:
            command = self._teleop_command
        if self._teleop_role is None and command is None:
            return
        if self._power_motion is not None or self._preset_moves or not self._enabled:
            return
        sample = self.joint_feedback()
        role = None
        if command is not None:
            watchdog = command.watchdog
            label = "左臂" if self.side == "left" else "右臂"
            if sample is None:
                watchdog.stop(f"{label}关节反馈超时")
            elif command.expires_at <= now:
                watchdog.stop(f"协同指令超时（{label}驱动检测）")
            if watchdog.reason is None:
                role = command.role
        if role != self._teleop_role:
            if not self._ensure_gravity():
                if command is not None:
                    command.watchdog.stop(f"{label}重力模型不可用")
                return
            target = np.asarray(drv.last_command, dtype=float).copy()
            if sample is not None:
                target[:7] = sample.positions
            elif self._soft_prev_tgt is not None:
                target[:7] = self._soft_prev_tgt
            self._teleop_role = role
            self._teach_wrist_positions.clear()
            self._playback_until = None
            self._grav_on, self._imp_on = True, False
            self._grav_scale = 1.0
            self._grav_kp = 0.5 if role == "leader" else 70.0
            self._manual_target = True
            self._reset_slider_motion()
            self._grav_prev_q = self._grav_prev_t = None
            self._soft_prev_tgt = target[:7].copy()
            self._set_motion_target(target)
        if role == "follower":
            measured = np.asarray(sample.positions)
            target = self._targets.copy()
            target[:7] = np.clip(
                command.joints,
                measured - _TELEOP_MAX_TRACKING_ERROR_RAD,
                measured + _TELEOP_MAX_TRACKING_ERROR_RAD,
            )
            self._set_motion_target(target)

    def _sample_joint_feedback(self, drv) -> None:
        axes = drv.openarm.get_arm()
        sampled_at = time.monotonic()
        try:
            ages = np.array([axes.get_link_stats(i).seconds_since_response() for i in range(7)])
        except AttributeError:
            self._fric_feedback = None
            self._teach_wrist_positions.clear()
            return
        oldest = ages.max()
        positions = np.asarray(drv.latest_state["qpos"][:7])
        velocities = np.asarray(drv.latest_state["qvel"][:7])
        self._fric_feedback = None
        if velocities.shape == (7,):
            expires_at = sampled_at + _FRIC_FEEDBACK_MAX_AGE_S - ages
            self._fric_feedback = velocities.copy(), expires_at
        if self._teach_mode and self._teleop_role is None and positions.shape == (7,):
            self._teach_wrist_positions.append((positions[4:].copy(), sampled_at - ages[4:]))
        else:
            self._teach_wrist_positions.clear()
        sample = None
        if (positions.shape == velocities.shape == (7,)
                and np.isfinite(positions).all() and np.isfinite(velocities).all()
                and oldest < FEEDBACK_TIMEOUT_S):
            sample = JointFeedback(tuple(positions), tuple(velocities),
                                   sampled_at + FEEDBACK_TIMEOUT_S - oldest)
        with self._lock:
            self._joint_feedback = sample

    def _send_current_target(self, drv) -> bool:
        """按固定节奏处理流式目标及预设位置，每帧之间可处理停止命令。"""
        now = time.monotonic()
        dt = now - self._last_send_t
        if not self._enabled or dt < _COMMAND_PERIOD_S:
            return False
        if self._urgent_stop.is_set():
            self._stop_now(drv)
            return True
        self._apply_force_mode(drv)
        self._update_teleop(drv, now)
        self._apply_live_targets(drv, now)
        if self._preset_moves:
            if self._teach_mode:
                self._finish_preset(False)
                with self._lock:
                    self._targets = None
            else:
                self._set_motion_target(self._preset_moves[0])
        self._update_grasp(drv, now)
        target = self._targets
        if target is not None:
            target = self._with_grasp_target(target)
        if self._hold_mode:
            if not self._ensure_gravity():
                if self._preset_moves:
                    self._finish_preset(False)
                    with self._lock:
                        self._targets = None
                return False
            self._last_send_t = now
            sent = self._send_soft_frame(drv, min(dt, _COMMAND_PERIOD_S))
        elif target is not None:
            self._last_send_t = now
            payload = self._step_toward(drv, target, dt, manual=self._manual_target)
            if payload is None:
                self._finish_preset(False)
                return False
            previous = drv.last_command[:7].copy() if self._slider_target else None
            ok = drv.send_position(payload)
            drv.openarm.recv_all(self.RECV_US)
            self._check_safety(ok, drv)
            if not ok:
                self._finish_preset(False)
                return True
            sent = drv.last_command
            if previous is not None and not np.array_equal(sent[:7], payload[:7]):
                self._slider_velocity = (sent[:7] - previous) / min(dt, _COMMAND_PERIOD_S)
        else:
            return False

        if self._preset_moves and np.allclose(sent, target[:len(sent)], rtol=0, atol=1e-6):
            if self._power_motion is not None and len(self._preset_moves) == 1:
                self._settle_power_motion(drv, target, now)
            else:
                self._preset_moves.popleft()
                if self._preset_moves:
                    self._set_motion_target(self._preset_moves[0])
                else:
                    self._finish_preset(True)
        return True

    def _step_toward(self, drv, target: np.ndarray, dt: float, *, manual: bool) -> np.ndarray | None:
        """限制实时目标的帧间变化，再交给驱动自身的安全检查。"""
        base = np.asarray(drv.last_command, dtype=float)
        if base.shape != target.shape or not np.isfinite(base).all():
            return None
        payload = target.copy()
        if manual:
            payload[:7] = self._manual_joint_step(base[:7], target[:7], dt)
        if self._delta_limits is not None:
            payload = base + np.clip(payload - base, -self._delta_limits, self._delta_limits)
        return payload

    def _check_safety(self, ok: bool, drv) -> None:
        reason = drv.safety_stop_reason
        with self._lock:
            self._safety = reason
            if reason is not None:
                self._enabled = False
                self._targets = None
                self._grasp = None
                self._grasp_requested = False
                self._grasp_status = "idle"
        if reason is not None:
            self._log(f"[{self.side}] 安全停止（锁存）：{reason}")
            try:
                drv.openarm.disable_all()
            except Exception:  # noqa: BLE001
                pass

    def _publish(self, st: dict) -> None:
        self._tau_arr = st.get("qtorque")
        pos, tau = self._to_named(st["qpos"])
        tmos, trot = st.get("tmos"), st.get("trotor")
        qvel = st.get("qvel")
        prefix = f"openarm_{self.side}_"
        tm, tr, vel = {}, {}, {}
        for i in range(1, 8):
            tm[f"{prefix}joint{i}"] = int(tmos[i - 1]) if tmos is not None else 0
            tr[f"{prefix}joint{i}"] = int(trot[i - 1]) if trot is not None else 0
            if qvel is not None and len(qvel) >= i:
                vel[f"{prefix}joint{i}"] = float(qvel[i - 1])
        with self._lock:
            self._positions, self._torques, self._tmos, self._trot = pos, tau, tm, tr
            if vel:
                self._velocities = vel

    def _snapshot_health(self, drv) -> None:
        """控制线程里 1Hz 采样健康快照（driver 非线程安全，只能在这里调）。"""
        try:
            self._health = drv.get_health()
        except Exception:  # noqa: BLE001
            pass

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

    def velocities(self) -> dict[str, float]:
        """关节角速度（rad/s），关节名→值。"""
        with self._lock:
            return dict(self._velocities)

    def temps(self) -> dict[str, int]:
        """MOS 温度（℃），关节名→值。"""
        with self._lock:
            return dict(self._tmos)

    def rotor_temps(self) -> dict[str, int]:
        """转子温度（℃），关节名→值。"""
        with self._lock:
            return dict(self._trot)

    def max_mos_temp(self) -> int:
        with self._lock:
            return max(self._tmos.values(), default=0)

    @property
    def health(self) -> tuple[list, dict] | None:
        """(各轴状态字符串, 总线错误计数) 快照；None=尚无样本。"""
        return self._health
