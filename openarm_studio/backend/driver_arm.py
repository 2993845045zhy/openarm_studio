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

# 夹爪方向：实测本机两臂**同为**「张开 = 负 rad、闭合 ≈ 0」。
# 官方配置假设左臂镜像（joint8 限位左右相反），与本机实测不符：
# 左臂发正 rad 时电机不动（超出机械行程），故不按侧取反。
_GRIP_SIGN = -1.0

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

_UP_HINT = ("sudo bash ~/openarm_ws/src/openarm_can/setup/"
            "configure_socketcan.sh {iface} -fd")


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

    不使能、不下发目标。返回 {"alive": n, "axes": [(应答,发送)×7],
    "positions": [rad×7]}。alive=0 表示总线上没有任何设备 ACK
    （臂断电/急停/线缆断），此时 positions 无效。
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
        axes, alive = [], 0
        for i in range(7):
            ls = o.get_arm().get_link_stats(i)
            axes.append((ls.responses, ls.commands_sent))
            alive += ls.responses > 0
        pos = [round(m.get_position(), 3) for m in o.get_arm().get_motors()]
        return {"alive": alive, "axes": axes, "positions": pos}
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
        self._delta_limits = None       # 单步限幅（connect 时从配置读）
        self._last_send_t = 0.0         # 指令发送节流（≥50Hz 官方节奏）
        self._last_poll_t = 0.0         # 状态轮询节流（≥100Hz）
        # 重力补偿（KDL 前馈力矩，见 backend/gravity.py）
        self._grav_model = None
        self._grav_on = False
        self._grav_scale = 1.0
        self._grav_kp = 30.0            # 保底刚度：前馈算错也不至失控
        self._grav_hold_q: list[float] | None = None
        self._grav_tau: np.ndarray | None = None   # 前馈低通状态
        self._grav_slew = 0.25          # 锁存姿态更新的一阶低通系数
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
        # 关节空间阻抗（整条臂）：与重力补偿共用软模式帧，只换增益
        self._imp_on = False
        self._imp_stiff = 0.35                  # Kp 相对官方增益的比例
        self._imp_zeta = 1.0                    # 阻尼比
        self._imp_kp_cur: np.ndarray | None = None
        self._imp_kd_cur: np.ndarray | None = None
        self._imp_t: float | None = None
        self._last_kp: np.ndarray | None = None   # 上一帧实际下发的增益
        self._last_kd: np.ndarray | None = None
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
        self._smooth_done = None      # callback(ok)

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
            # 本机实测：两臂夹爪同向（张开 = 负 rad），而官方配置按"左臂
            # 镜像"给了 left_arm joint8 限位 [-0.4, 1.0472]，会把张开行程
            # 裁掉 2/3。这里按实测把左臂 joint8 限位改成与右臂一致。
            try:
                cfg._config["joint_limits"]["left_arm"][7] = [-_GRIP_RAD_SPAN, 0.4]  # noqa: SLF001
            except Exception:  # noqa: BLE001
                pass
            drv = SingleArmDriver(f"{self.side}_arm", config=cfg,
                                  can_interface=self.can)
        except Exception as exc:  # noqa: BLE001
            self._log(f"[{self.side}] 连接 {self.can} 失败：{exc}")
            return False
        # 总线应答校验：构造期已轮询 20 轮，仍 0 应答 = 总线上没有设备
        # （臂断电/急停/CAN 线断）。这种情况不进 arms，避免"已连接"假象。
        try:
            alive = sum(1 for i in range(7)
                        if drv.openarm.get_arm().get_link_stats(i).responses > 0)
        except Exception:  # noqa: BLE001
            alive = 7   # 统计读不出来就放行，交给控制线程日志
        if alive == 0:
            self._log(f"[{self.side}] {self.can} 总线无应答（0/7 轴）——"
                      "查供电/急停/CAN 线；若都正常仍无应答，"
                      "断电重启该臂（单个电机故障可锁死整条总线）")
            return False
        with self._lock:
            self._driver = drv
            self._running = True
            try:
                self._delta_limits = np.asarray(
                    drv.config.get_joint_delta_position_limits(), dtype=float)
            except Exception:  # noqa: BLE001
                self._delta_limits = None
            # 导纳帧直接走 mit_control_all，**绕过官方的限位检查**，
            # 所以这里必须自己拿到关节限位做裁剪，否则会把臂顶到机械限位。
            try:
                lim = np.asarray(
                    drv.config.get_joint_limits(f"{self.side}_arm"), dtype=float)
                self._lim_lo, self._lim_hi = lim[:, 0].copy(), lim[:, 1].copy()
            except Exception:  # noqa: BLE001
                self._lim_lo = self._lim_hi = None
        # 起始/收臂轨迹：官方参数原样执行，仅每帧补读状态
        drv._on_start = lambda: self._travel(drv, "start")   # noqa: SLF001
        drv._on_stop = lambda: self._travel(drv, "stop")     # noqa: SLF001
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
        """关节名→值（rad / 夹爪 m）。None = 停止下发，转只读。

        速度/加速度由官方安全链约束（JointVelocityChecker 按配置
        限幅裁剪指令），本封装只做单步限幅（_step_toward）防跳变锁存。
        """
        if named is None:
            self._push(_Cmd("hold"))
            return
        self._push(_Cmd("targets", self._to_array(named)))

    def _flush_cmds(self) -> int:
        """清空命令队列（轨迹类操作前调用），返回丢弃条数。"""
        n = 0
        while True:
            try:
                self._cmds.get_nowait()
                n += 1
            except queue.Empty:
                return n

    # ── 重力补偿（KDL 前馈，见 backend/gravity.py）──────────
    def set_gravity_comp(self, on: bool, scale: float = 1.0,
                         kp_floor: float = 30.0) -> None:
        """开关重力补偿：MIT 前馈 tau=G(q)×scale，位置环用 kp_floor 保底。

        kp_floor 是安全网：即使模型符号/数值有偏差，臂也只是"变软/变沉"，
        不会失控。kp_floor ≥ 2 时锁存进入时的姿态作为位置目标（补偿保持）；
        kp_floor < 2 时位置目标跟随实测角，即零刚度拖动模式。
        """
        self._grav_on = bool(on)
        if on:
            self._imp_on = False        # 互斥：同一软模式帧
        self._grav_scale = float(scale)
        self._grav_kp = float(kp_floor)
        if on:
            # 已使能：锁存当前姿态作为位置目标。
            # 未使能（使能命令还在队列里）：清空锁存，等 enable 分支
            # 执行完用起始位重新锁存——否则这里会把下垂姿态锁进去。
            if self._enabled:
                with self._lock:
                    pos = dict(self._positions)
                prefix = f"openarm_{self.side}_"
                self._grav_hold_q = [float(pos.get(f"{prefix}joint{i}", 0.0))
                                     for i in range(1, 8)]
            else:
                self._grav_hold_q = None
            self._push(_Cmd("soft"))
        else:
            with self._lock:
                pos = dict(self._positions)
            prefix = f"openarm_{self.side}_"
            # 关闭时把位置目标设为当前实测，恢复普通位置保持（否则会一直软着）
            with self._lock:
                self._targets = np.array(
                    [float(pos.get(f"{prefix}joint{i}", 0.0)) for i in range(1, 8)]
                    + [0.0])
            self._grav_hold_q = None
            self._reset_gain_ramp()
        self._log(f"[{self.side}] 重力补偿 {'开启' if on else '关闭'}"
                  + (f"（前馈×{scale:.2f}，保底 kp={kp_floor:.1f}"
                     f"{'，拖动模式' if kp_floor < 2 else '，姿态锁存'}）"
                     if on else "，已恢复位置保持"))

    def _ensure_gravity(self) -> bool:
        """懒加载 KDL 重力模型（首次约 1~2s，会跑一次 xacro）。"""
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
        """开关整臂关节阻抗。

        stiffness ∈ [0.01, 1]：Kp 相对官方增益的比例（1.0 = 重力补偿那档）
        zeta      ∈ [0.2, 2]  ：阻尼比，<1 偏弹、=1 临界、>1 偏黏
        前馈仍是 G(q)+τ_fric；两个参数都可在线改，增益按速率限制过渡。
        """
        s = max(self.IMP_STIFF_MIN, min(float(stiffness), self.IMP_STIFF_MAX))
        z = max(self.IMP_ZETA_MIN, min(float(zeta), self.IMP_ZETA_MAX))
        was = self._imp_on
        self._imp_stiff, self._imp_zeta = s, z
        self._imp_on = bool(on)
        if on:
            self._grav_on = False        # 互斥：同一帧，不能两套增益
            if not was:
                with self._lock:
                    pos = dict(self._positions)
                prefix = f"openarm_{self.side}_"
                self._grav_hold_q = [float(pos.get(f"{prefix}joint{i}", 0.0))
                                     for i in range(1, 8)]
                self._imp_kp_cur = None
                self._imp_kd_cur = None
            self._push(_Cmd("soft"))
        else:
            with self._lock:
                pos = dict(self._positions)
            prefix = f"openarm_{self.side}_"
            with self._lock:
                self._targets = np.array(
                    [float(pos.get(f"{prefix}joint{i}", 0.0)) for i in range(1, 8)]
                    + [0.0])
            self._grav_hold_q = None
            self._reset_gain_ramp()
        if on:
            kp = getattr(self, "_last_kp", None)
            shown = (f"Kp J1={kp[0]:.1f} J5={kp[4]:.1f}" if kp is not None
                     else f"Kp×{s:.2f}")
            self._log(f"[{self.side}] 关节阻抗 开启（刚度 {s:.0%}，阻尼比 {z:.2f}，"
                      f"{shown}）")
        else:
            self._log(f"[{self.side}] 关节阻抗 关闭（已恢复位置保持）")

    def _reset_gain_ramp(self) -> None:
        """离开软模式时清掉增益斜坡状态：回到位置控制后实际下发的是官方
        增益，下次再进软模式必须从官方值起步，不能拿旧的中间值当起点。"""
        self._imp_kp_cur = None
        self._imp_kd_cur = None
        self._imp_t = None
        self._last_kp = None
        self._last_kd = None

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
            send_dq = False
        elif self._imp_on:
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
        """开关关节摩擦补偿（官方 tanh 模型，作用于重力帧/导纳帧的前馈）。"""
        self._fric_on = bool(on)
        self._fric_tau = None
        self._log(f"[{self.side}] 摩擦补偿 {'开启' if on else '关闭'}"
                  + ("（tanh 库仑+粘滞，官方标定参数）" if on else ""))

    def _friction_tau(self, names: list[str], vel: dict[str, float]) -> np.ndarray:
        """摩擦补偿力矩（带一阶低通 + 限幅）。

        速度来自编码器差分，噪声不小；直接当前馈会把噪声放大成力矩抖动，
        所以和重力一样过一阶低通，并对单项限幅 3 Nm 兜底。
        """
        if not self._fric_on or self._grav_model is None:
            return np.zeros(7)
        v = [float(vel.get(n, 0.0)) for n in names]
        f_raw = self._grav_model.friction(self.side, v)
        if self._fric_tau is None or self._fric_tau.shape != f_raw.shape:
            self._fric_tau = f_raw.copy()
        else:
            self._fric_tau += (f_raw - self._fric_tau) * 0.3
        return np.clip(self._fric_tau, -3.0, 3.0)

    def _send_gripper(self, drv, pos: dict) -> None:
        """夹爪（第 8 轴）单独走 POS_FORCE：重力帧/导纳帧都只发 7 个臂关节，
        不补发的话补偿期间夹爪滑块完全没响应。"""
        prefix = f"openarm_{self.side}_"
        with self._lock:
            # _targets[7] 已经是**电机弧度**（_to_array 转换过），直接用，
            # 不能再按米换算——否则双重换算会得到 +24.9 rad 的荒谬目标。
            grip_tgt = None if self._targets is None else float(self._targets[7])
            grip_meas_m = float(pos.get(f"{prefix}finger_joint1", 0.0))
        rad = grip_tgt if grip_tgt is not None else \
            _GRIP_SIGN * grip_meas_m / GRIP_OPEN_M * _GRIP_RAD_SPAN
        speed, torque_pu = drv.gripper_posforce_limits
        drv.openarm.get_gripper().set_position(
            rad + drv.joint_offsets[7],
            speed_rad_s=float(speed), torque_pu=float(torque_pu) / 4.5)

    def _send_soft_frame(self, drv) -> None:
        """发一帧「G(q)+τ_fric 前馈 + 可调 PD」MIT。

        重力补偿、示教、关节阻抗共用这一帧，区别只在 _soft_gains() 给的
        Kp/Kd：位置目标都取进入时锁存的姿态（拖动模式跟实测角）。
        """
        import openarm_can as oa

        prefix = f"openarm_{self.side}_"
        teach = self._grav_kp < 2.0 and not self._imp_on
        with self._lock:
            pos = dict(self._positions)
        measured = [float(pos.get(f"{prefix}joint{i}", 0.0)) for i in range(1, 8)]
        q_tgt = measured if (teach or self._grav_hold_q is None) \
            else list(self._grav_hold_q)
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
        with self._lock:
            vel = dict(self._velocities)
        # 重力 + 摩擦：官方 teleop 的 τ = G(q) + τ_fric(ω)
        tau = self._grav_tau + self._friction_tau(names, vel)
        # 增益按模式取（示教 / 关节阻抗 / 重力补偿），见 _soft_gains()。
        # 关键：必须**按关节**取官方配置值（J1/J2=70、J3/J4=60、J5~J7=10）。
        # 之前统一发 70，末端三个 DM4310 小电机的刚度超标 7 倍——
        # 小惯量+大刚度的高频振荡，表现为"只有末端三关节震"。
        kps, kds, send_dq = self._soft_gains(drv, teach)
        # 速度前馈：目标在移动时把"目标变化率"作为 MIT 的 dq 下发。
        # **零重力模式必须关闭**：那时目标就等于实测角，"目标速度"
        # 等于实测速度，kd*(dq_d - q̇)≈0 完全没有阻尼，还把编码器噪声
        # 直接放大成力矩——表现就是手推时剧烈抖动。
        now = time.monotonic()
        dq = [0.0] * 7
        if send_dq and self._grav_prev_q is not None \
                and self._grav_prev_t is not None:
            dt = max(now - self._grav_prev_t, 0.005)
            for i in range(7):
                v = (q_tgt[i] - self._grav_prev_q[i]) / dt
                dq[i] = max(-3.0, min(3.0, v))       # 限幅 3 rad/s
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
        drv.last_command = np.array(measured + [finger])
        drv.last_command_time_s = time.monotonic()

    def smooth_move(self, named: dict[str, float], hz: float = 50.0,
                    duration: float = 2.0, done=None) -> bool:
        """插值运动（官方 smooth_move，走完整安全链）。阻塞控制线程。"""
        self._smooth_done = done
        return self._push(_Cmd("smooth", (self._to_array(named), hz, duration)))

    def move_start(self) -> bool:
        """平滑走到配置的起始位（2s，需已使能；不发回程/不失能）。"""
        return self._push(_Cmd("goto_start"))

    def move_park(self) -> bool:
        """平滑走到配置的收臂位（2s，需已使能；不失能）。"""
        return self._push(_Cmd("goto_stop"))

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
            gm = min(GRIP_OPEN_M, max(0.0, gm))
            arr[7] = _GRIP_SIGN * gm / GRIP_OPEN_M * _GRIP_RAD_SPAN
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
    def _travel(self, drv, which: str) -> bool:
        """起始(start)/收臂(stop)轨迹：官方参数 + 每帧补读。

        时序、时长、目标位姿全部取自官方配置（hz/duration/position），
        与官方 move_to_*_position 等价；唯一差别是每帧下发后补读状态，
        避免官方 send_position 的 300µs 超时读不全 8 轴应答。
        """
        conf = (drv.config.get_start_config() if which == "start"
                else drv.config.get_stop_config())
        moves = conf.get("moves", [])
        if not moves:
            return True
        for move in moves:
            tgt = np.asarray(move["position"][drv.arm_side], dtype=float)
            hz = float(move.get("hz", 50))
            duration = float(move.get("duration", 2))
            base = np.asarray(drv.last_command, dtype=float)
            steps = max(1, int(hz * duration))
            for k in range(steps):
                a = (k + 1) / steps
                if drv.send_position(base + (tgt - base) * a) is False:
                    return False
                drv.openarm.recv_all(self.RECV_US)     # 补读，读全 8 轴
                # 逐帧发布姿态：轨迹跑在本控制线程里，若不在这里更新，
                # UI 在整个 2s 轨迹期间读到的都是旧姿态——表现为
                # "使能/失能过程中 3D 画面不动，结束后才跳过去"。
                self._publish(drv.latest_state)
                time.sleep(1.0 / hz)
            drv.set_latest_state(timeout_us=self.RECV_US)
        return True

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
                # 2) 状态刷新：≥10ms 一次（≈100Hz）。之前每个 2ms 循环
                #    都刷（≈400Hz×8 帧）会把总线打满——拖滑块时电机
                #    报错下线、整臂无应答的直接诱因。
                now = time.monotonic()
                if not busy and now - self._last_poll_t >= 0.010:
                    self._last_poll_t = now
                    drv.openarm.refresh_all()
                    drv.set_latest_state(timeout_us=self.RECV_US)  # 大超时读全
                    self._publish(drv.latest_state)
                if now - self._health_t > 1.0:
                    self._health_t = now
                    self._snapshot_health(drv)
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
            # 无新命令：软模式（重力/示教/关节阻抗）> 位置保持
            with self._lock:
                tgt = self._targets
                hold_on = self._hold_mode
            if hold_on and self._enabled and self._ensure_gravity():
                now = time.monotonic()
                if now - self._last_send_t >= 0.019:
                    self._last_send_t = now
                    self._send_soft_frame(drv)
                    return True
                return False
            if tgt is not None and self._enabled:
                now = time.monotonic()
                if now - self._last_send_t >= 0.019:
                    self._last_send_t = now
                    ok = drv.send_position(tgt)
                    drv.openarm.recv_all(self.RECV_US)     # 补读，读全 8 轴
                    self._check_safety(ok, drv)
                    return True
            return False

        if cmd.kind == "enable":
            ok = drv.start()
            with self._lock:
                self._enabled = bool(ok)
            self._log(f"[{self.side}] start() {'成功（已走起始轨迹）' if ok else '失败/被安全链中断'}")
            self._check_safety(ok, drv)
            # 使能完成后若软模式已挂起（重力补偿/示教/关节阻抗）：以起始位
            # 锁存目标并发第一帧，否则空闲分支会用旧的下垂姿态当锁存目标，
            # 把臂拽回去。
            if ok and self._hold_mode and self._ensure_gravity():
                with self._lock:
                    pos = dict(self._positions)
                prefix = f"openarm_{self.side}_"
                self._grav_hold_q = [float(pos.get(f"{prefix}joint{i}", 0.0))
                                     for i in range(1, 8)]
                self._send_soft_frame(drv)
                self._last_send_t = time.monotonic()
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
        if cmd.kind == "soft":
            # 开启时立即生效的"第一帧"，之后由空闲分支按 50Hz 续发。
            # 未使能时只挂起标志（使能命令还在队列里）：若此刻就发软模式帧，
            # 锁存目标会取当前的下垂姿态，等 start() 轨迹把臂抬到起始位后，
            # 位置环会立刻把臂拽回下垂位（"使能后立马下垂"）。
            # enable 分支执行完会自行调用 _send_soft_frame 并锁存起始位。
            if self._ensure_gravity():
                if self._enabled:
                    self._send_soft_frame(drv)
                    self._last_send_t = time.monotonic()
            return True
        if cmd.kind == "targets":
            if self._hold_mode:
                # 软模式（重力/示教/关节阻抗）期间收到滑块目标：关节部分
                # 写进"锁存姿态"，夹爪目标（第 8 维）存进 _targets 由软模式
                # 帧补发。锁存位必须**平滑**靠近新目标：滑块是 0.01 rad 的
                # 阶跃，目标直接阶跃 + 位置环会产生过冲振荡。
                with self._lock:
                    new = [float(cmd.payload[i]) for i in range(7)]
                    cur = self._grav_hold_q
                    if cur is None or len(cur) != 7:
                        self._grav_hold_q = new
                    else:
                        a = self._grav_slew
                        self._grav_hold_q = [cur[i] + (new[i] - cur[i]) * a
                                             for i in range(7)]
                    # ⚠️ 必须用 cmd.payload（已含正确的填充值）做基准。
                    # 若用 zeros(8)，下一拍 _to_array 会把 last_cmd 当
                    # 填充源 → 所有关节目标变 0 → 肘关节突然下垂。
                    self._targets = np.array(cmd.payload, dtype=float)
                return False
            payload = self._step_toward(drv, cmd.payload)
            with self._lock:
                self._targets = payload
            if self._enabled:
                self._last_send_t = time.monotonic()
                ok = drv.send_position(payload)
                drv.openarm.recv_all(self.RECV_US)         # 补读，读全 8 轴
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
        if cmd.kind in ("goto_start", "goto_stop"):
            if not self._enabled:
                self._log(f"[{self.side}] 未使能，无法平滑运动（先点「使能」）")
                return True
            # 丢弃轨迹期间 UI 积压的滑块目标：否则轨迹结束瞬间这些旧目标
            # 会被逐条执行，与当前位置的差值超过单步限幅 → 官方安全链
            # force-stop 锁存 → 自动失能（双臂下垂）。
            dropped = self._flush_cmds()
            name = "起始位" if cmd.kind == "goto_start" else "收臂位"
            self._log(f"[{self.side}] 平滑走向{name}（官方位姿，臂会动）"
                      + (f"，已丢弃 {dropped} 条积压指令" if dropped else ""))
            ok = self._travel(drv, "start" if cmd.kind == "goto_start" else "stop")
            self._flush_cmds()      # 轨迹期间到达的旧目标同样丢弃
            self._check_safety(ok, drv)
            with self._lock:
                # 完成后以实际指令位姿作为保持目标，避免回到旧目标
                self._targets = np.array(drv.last_command) if ok else None
            self._log(f"[{self.side}] {name}{'到达' if ok else '被安全链中断'}")
            return True
        return False

    def _step_toward(self, drv, tgt: np.ndarray) -> np.ndarray:
        """把目标按单步限幅裁到上条指令附近（大跳变分小步走）。

        官方安全链对超过 delta 限幅的目标直接 force_stop 锁存；滑块
        单击凹槽会瞬移 >1 rad，这里逐步逼近，UI 以 50Hz 重发全量目标，
        每拍走一限幅步即可收敛。
        """
        if self._delta_limits is None:
            return tgt
        base = getattr(drv, "last_command", None)
        if base is None or np.shape(base) != np.shape(tgt):
            return tgt
        base = np.asarray(base, dtype=float)
        return base + np.clip(np.asarray(tgt) - base,
                              -self._delta_limits, self._delta_limits)

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
