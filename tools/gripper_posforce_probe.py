#!/usr/bin/env python3
"""夹爪（J8）POS_FORCE 真机标定 / 智能抓取端到端自检。只驱动一条总线上的第 8 轴，臂关节不动。

为什么需要它：智能抓取的判定阈值（静止门限、全闭位置、电流下限、力矩换算）来自这里量出的
真机数据，见 tests/fixtures/gripper_posforce/README.md。改了 gripper.py 的常数后要重新量。

用法（真机模式必须先关闭 OpenArm Studio，且总线上所有电机处于失能状态）：
    # 只做前置检查，不使能电机（不加 --run 时的默认行为）
    python tools/gripper_posforce_probe.py --bus can0
    # 先在仿真里跑一遍（不碰硬件）
    python tools/gripper_posforce_probe.py --simulate --run
    python tools/gripper_posforce_probe.py --simulate --run --grasp-check
    # 标定：各档堵转力矩、全程摩擦、闭合/张开到限位
    python tools/gripper_posforce_probe.py --bus can0 --run --out records/gripper_probe
    # 端到端：走 Studio 生产代码（DriverArm.request_grasp -> Grasp）在真爪上点一次智能抓取
    python tools/gripper_posforce_probe.py --bus can0 --run --grasp-check --side right

保护：
- Studio 在运行、任一电机已使能、J8 不在 POS_FORCE（CTRL_MODE 4）时拒绝执行
- 目标裁剪到机械行程内，速度 ≤ 1 rad/s，电流标幺值 ≤ 0.1333（等于 Studio 面板 60%）
- 反馈丢失、电机报错、位置越界、过热时立即失能；正常结束、异常、Ctrl-C、SIGTERM 都会失能

OpenArm 固件的帧格式（与 openarm_can 1.4.1 相同）：0xFC 使能、0xFD 失能；
POS_FORCE 指令本身不带回包，要另外发状态查询帧才有反馈。
"""
import argparse
import csv
import math
import os
import signal
import socket
import struct
import sys
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

J8 = 8
PMAX, VMAX, TMAX = 12.5, 30.0, 10.0     # DM4310 反馈量程，与 openarm_can 一致
ARM_LIMITS = {1: (12.5, 45, 54), 2: (12.5, 45, 54), 3: (12.5, 8, 28), 4: (12.5, 8, 28),
              5: (12.5, 30, 10), 6: (12.5, 30, 10), 7: (12.5, 30, 10), 8: (PMAX, VMAX, TMAX)}
Q_OPEN, Q_CLOSED = -1.0472, 0.0         # 本机两臂同向：张开 = 负 rad
SPEED_CAP, PU_CAP = 1.0, 0.1333
COMMAND_PERIOD_S = 0.02                 # 50 Hz，与 Studio 相同
POLL_PERIOD_S = 0.01

# name: (label, target, speed, pu, 最长时长 s, 到位即停)
SEQUENCES = {
    "calibrate": [
        ("hold_low", "q0", 1.0, 0.0444, 0.5, None),
        ("close_067", Q_CLOSED, 0.35, 0.0667, 8.0, True),
        ("stall_033", Q_CLOSED, 0.35, 0.0333, 1.0, None),
        ("stall_044", Q_CLOSED, 0.35, 0.0444, 1.0, None),
        ("stall_067", Q_CLOSED, 0.35, 0.0667, 1.0, None),
        ("stall_089", Q_CLOSED, 0.35, 0.0889, 1.0, None),
        ("stall_111", Q_CLOSED, 0.35, 0.1111, 1.0, None),
        ("stall_133", Q_CLOSED, 0.35, 0.1333, 1.5, None),
        ("open_133", Q_OPEN, 0.5, 0.1333, 6.0, True),
        ("close_044_from_open", Q_CLOSED, 0.35, 0.0444, 12.0, True),
        ("open_067", Q_OPEN, 0.5, 0.0667, 10.0, True),
        ("close_067_from_open", Q_CLOSED, 0.5, 0.0667, 8.0, True),
        ("back_q0_133", "q0", 0.5, 0.1333, 6.0, True),
    ],
    "short": [
        ("hold_low", "q0", 1.0, 0.0444, 0.5, None),
        ("close_044", Q_CLOSED, 0.35, 0.0444, 10.0, True),
        ("stall_044", Q_CLOSED, 0.35, 0.0444, 1.0, None),
        ("stall_067", Q_CLOSED, 0.35, 0.0667, 1.0, None),
        ("stall_133", Q_CLOSED, 0.35, 0.1333, 1.5, None),
        ("open_067", Q_OPEN, 0.5, 0.0667, 10.0, True),
        ("back_q0_067", "q0", 0.5, 0.0667, 8.0, True),
    ],
    # 验证某个电流下限能否越过机构"紧点"（实测两爪各有一处需要 > 0.43 N·m 的位置）
    "minlimit": [
        ("hold", "q0", 1.0, 0.08, 0.5, None),
        ("close", Q_CLOSED, 0.5, 0.08, 8.0, True),
        ("open", Q_OPEN, 0.5, 0.08, 8.0, True),
        ("back_q0", "q0", 0.5, 0.08, 8.0, True),
    ],
}


# ── 帧 ───────────────────────────────────────────────────────────────────────
def frame(can_id, data):
    assert len(data) == 8
    return struct.pack("=IBBBB", can_id, 8, 0x01, 0, 0) + bytes(data)


def posforce(q, speed, pu):
    q = min(Q_CLOSED, max(Q_OPEN, q))
    speed = min(SPEED_CAP, max(0.0, speed))
    pu = min(PU_CAP, max(0.0, pu))
    return frame(0x300 + J8, struct.pack("<fHH", q, round(speed * 100), round(pu * 10000)))


def refresh(mid):
    return frame(0x7FF, [mid & 0xFF, (mid >> 8) & 0xFF, 0xCC, 0, 0, 0, 0, 0])


def query(mid, rid):
    return frame(0x7FF, [mid & 0xFF, (mid >> 8) & 0xFF, 0x33, rid, 0, 0, 0, 0])


ENABLE = frame(J8, [0xFF] * 7 + [0xFC])
DISABLE = frame(J8, [0xFF] * 7 + [0xFD])


def decode(d, limits=(PMAX, VMAX, TMAX)):
    pmax, vmax, tmax = limits
    q = int.from_bytes(d[1:3], "big") * 2 * pmax / 65535 - pmax
    v = (((d[3] << 4) | (d[4] >> 4)) & 0xFFF) * 2 * vmax / 4095 - vmax
    tau = (((d[4] & 0xF) << 8) | d[5]) * 2 * tmax / 4095 - tmax
    return {"status": d[0] >> 4, "q": q, "v": v, "tau": tau, "tmos": d[6], "trot": d[7]}


# ── 总线：真实 SocketCAN / 仿真 ──────────────────────────────────────────────
class CanBus:
    """SocketCAN-FD。send() 只发；poll() 取回已到达的帧（不阻塞）。"""

    def __init__(self, iface):
        self.iface = iface
        self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
        self.sock.bind((iface,))
        self.sock.settimeout(0.0)

    def send(self, fr):
        self.sock.send(fr)

    def poll(self):
        out = []
        while True:
            try:
                raw = self.sock.recv(72)
            except BlockingIOError:
                return out
            can_id, dlc, flags = struct.unpack("=IBB", raw[:6])
            out.append((time.time(), can_id, raw[8:8 + min(dlc, 64)]))

    def close(self):
        self.sock.close()


class SimBus:
    """仿真 J8：按速度上限走，先要克服约 0.27 N·m 摩擦，被挡住时力矩顶到电流上限。

    只用来跑通脚本流程和回归判定逻辑，不代表真机的精确动态。
    """
    OPEN_STOP, CLOSED_STOP = -1.0477, -0.045

    def __init__(self):
        self.q, self.v, self.tau, self.status = -0.7277, 0.0, 0.0, 0
        self.target = self.speed = self.pu = None
        self.pending = []
        self.clock = time.time()
        self.sent = 0

    def _pack(self):
        q_raw = round((self.q + PMAX) / (2 * PMAX) * 65535)
        v_raw = round((self.v + VMAX) / (2 * VMAX) * 4095) & 0xFFF
        t_raw = round((self.tau + TMAX) / (2 * TMAX) * 4095) & 0xFFF
        return bytes([(self.status & 0xF) << 4, q_raw >> 8, q_raw & 0xFF, v_raw >> 4,
                      ((v_raw & 0xF) << 4) | (t_raw >> 8), t_raw & 0xFF, 28, 26])

    def _step(self, dt):
        if self.status != 1 or self.target is None:
            self.v = self.tau = 0.0
            return
        err = self.target - self.q          # 目标可以在限位之外：被挡住时误差不消失，力矩顶到上限
        limit = 10.0 * self.pu
        blocked = (err > 0 and self.q >= self.CLOSED_STOP - 1e-9) or (err < 0 and self.q <= self.OPEN_STOP + 1e-9)
        v_ref = max(-self.speed, min(self.speed, 54.0 * err))
        if blocked:
            self.v, self.tau = 0.0, math.copysign(min(limit, 54.0 * abs(err)), err)
        elif v_ref == 0:
            self.v, self.tau = 0.0, 0.0
        elif limit > 0.27:
            self.v, self.tau = v_ref, math.copysign(0.27, v_ref)
        else:
            self.v, self.tau = 0.0, math.copysign(limit, err)
        self.q = max(self.OPEN_STOP, min(self.CLOSED_STOP, self.q + self.v * dt))

    def send(self, fr):
        can_id = struct.unpack_from("=I", fr)[0]
        d = fr[8:16]
        now = time.time()
        self._step(now - self.clock)
        self.clock = now
        self.sent += 1
        if can_id == 0x7FF and d[2] == 0xCC and d[0] == J8:
            self.pending.append((now + 0.005, 0x18, self._pack()))
        elif can_id == 0x7FF and d[2] == 0x33 and d[0] == J8:
            self.pending.append((now + 0.005, 0x18, bytes([d[0], d[1], 0x33, d[3]]) + struct.pack("<i", 4)))
        elif can_id == 0x308:
            q, speed, pu = struct.unpack("<fHH", d)
            self.target, self.speed, self.pu = q, speed / 100, pu / 10000
        elif can_id == J8 and d[7] == 0xFC:
            self.status = 1
        elif can_id == J8 and d[7] == 0xFD:
            self.status, self.target = 0, None

    def poll(self):
        now = time.time()
        due = [p for p in self.pending if p[0] <= now]
        self.pending = [p for p in self.pending if p[0] > now]
        return due

    def close(self):
        pass


def studio_running():
    """是否有 `python -m openarm_studio.app` 进程（按 argv 判断，pgrep -f 会误匹配 shell）。"""
    me = os.getpid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv = fh.read().split(b"\0")
        except OSError:
            continue
        if b"-m" in argv and b"openarm_studio.app" in argv:
            return True
    return False


class Aborted(RuntimeError):
    pass


# ── 探测主体 ─────────────────────────────────────────────────────────────────
class Probe:
    def __init__(self, bus, out_dir, simulate=False):
        self.bus = bus
        self.simulate = simulate
        self.rows = []
        self.all_rows = []
        self.dump = None
        self.moved = False            # 是否真的使能过 J8（决定退出时是否要确认失能）
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(out_dir, f"gripper_{getattr(bus, 'iface', 'sim')}_{stamp}.csv")
        self.enabled = False
        self.last_fb = None          # (t, state)
        self.frame_number = 0

    # ---- 收发 ----
    def send(self, fr):
        if not self.simulate:
            can_id = struct.unpack_from("=I", fr)[0]
            self.dump.write(f"({time.time():.6f}) {self.bus.iface} {can_id:03X}##5{fr[8:16].hex()}\n")
        self.bus.send(fr)

    def poll_states(self):
        """取回 J8 的状态回包，刷新 self.last_fb；返回是否有新帧。"""
        fresh = False
        for t, can_id, d in self.bus.poll():
            if can_id == 0x18 and len(d) >= 8:
                self.last_fb = (t, decode(d))
                self.frame_number += 1
                fresh = True
        return fresh

    def read_state(self, mid, wait=0.25):
        self.send(refresh(mid))
        self.send(refresh(mid))
        end = time.time() + wait
        while time.time() < end:
            for _, can_id, d in self.bus.poll():
                if can_id == 0x10 + mid and len(d) >= 8:
                    return decode(d, ARM_LIMITS[mid])
            time.sleep(0.005)
        return None

    def read_param(self, mid, rid, wait=0.25):
        self.send(query(mid, rid))
        end = time.time() + wait
        while time.time() < end:
            for _, can_id, d in self.bus.poll():
                if len(d) >= 8 and d[2] == 0x33 and d[3] == rid:
                    return d[4:8]
            time.sleep(0.005)
        return None

    # ---- 安全 ----
    def precheck(self):
        """返回 (各电机状态, J8 当前模式)。任何一项不满足就抛 Aborted，不使能任何电机。"""
        if self.simulate:
            self.send(refresh(J8))
            time.sleep(0.01)
            self.poll_states()
            return {J8: self.last_fb[1]}, 4
        if studio_running():
            raise Aborted("OpenArm Studio 正在运行：同一台电机不能有两个发送方，请先关闭 Studio")
        states = {}
        for mid in range(1, 9):
            try:
                state = self.read_state(mid)
            except OSError as exc:
                raise Aborted(f"{self.bus.iface} 发送失败（{exc}）：适配器通道卡死或未就绪") from exc
            if state is None:
                raise Aborted(f"{self.bus.iface} 上 J{mid} 不应答状态查询")
            if state["status"] != 0:
                raise Aborted(f"J{mid} 处于使能/报错状态 {state['status']}，应先失能")
            states[mid] = state
        raw = self.read_param(J8, 10)
        mode = None if raw is None else struct.unpack("<i", raw)[0]
        if mode != 4:
            raise Aborted(f"J8 CTRL_MODE={mode}，应为 4（POS_FORCE）")
        return states, mode

    def disable(self):
        for _ in range(3):
            try:
                self.send(DISABLE)
            except OSError:
                pass
            time.sleep(0.02)
        self.enabled = False

    def _guard(self, label):
        """每帧检查：反馈、状态、行程、温度。返回最近一帧的状态字典。"""
        if self.last_fb is None or time.time() - self.last_fb[0] > 0.4:
            raise Aborted(f"{label}: J8 反馈丢失")
        state = self.last_fb[1]
        if state["status"] != 1:
            raise Aborted(f"{label}: J8 状态 {state['status']}（掉使能或报错）")
        if not (Q_OPEN - 0.08 <= state["q"] <= Q_CLOSED + 0.12):
            raise Aborted(f"{label}: J8 位置 {state['q']:.3f} 越界")
        if state["tmos"] > 70 or state["trot"] > 85:
            raise Aborted(f"{label}: J8 过热 {state['tmos']}/{state['trot']} ℃")
        return state

    # ---- 动作 ----
    def run_segment(self, label, target, speed, pu, duration, settle):
        cmd = posforce(target, speed, pu)
        start = time.time()
        self.last_fb = None
        last_cmd = last_poll = -1.0
        window, q_start, settled_at = [], None, None
        while True:
            now = time.time()
            if now - last_cmd >= COMMAND_PERIOD_S:
                self.send(cmd)
                last_cmd = now
            if now - last_poll >= POLL_PERIOD_S:
                self.send(refresh(J8))
                last_poll = now
            fresh = self.poll_states()
            if fresh:
                t, state = self.last_fb
                self.rows.append([f"{t:.6f}", label, f"{target:.6f}", f"{speed:.3f}", f"{pu:.5f}",
                                  state["status"], f"{state['q']:.6f}", f"{state['v']:.5f}",
                                  f"{state['tau']:.5f}", state["tmos"], state["trot"], ""])
                window.append((t, state["q"]))
            elapsed = now - start
            if self.last_fb is None:
                if elapsed > 0.4:
                    raise Aborted(f"{label}: 0.4 s 内没有反馈")
                time.sleep(0.002)
                continue
            state = self._guard(label)
            # 起步要 40–50 ms，"已停住"只在实际动过之后才认，避免把起步当成到位
            moved = q_start is None or abs(state["q"] - q_start) >= 0.02
            if settle and elapsed > 0.4 and (moved or elapsed > 1.5):
                window = [w for w in window if now - w[0] <= 0.5]
                if q_start is None:
                    q_start = window[0][1] if window else state["q"]
                if window and max(w[1] for w in window) - min(w[1] for w in window) < 0.0012:
                    settled_at = settled_at or now
                else:
                    settled_at = None
            if elapsed >= duration or settled_at:
                break
            time.sleep(0.002)
        state = self._guard(label)
        print(f"  {label:22s} {time.time() - start:5.2f}s  q={state['q']:+.4f}  v={state['v']:+.3f}  "
              f"tau={state['tau']:+.3f}", flush=True)
        return state

    def write_rows(self):
        """把本次运行攒下的反馈行落到 CSV（每次追加，文件只写一遍表头）。"""
        if not self.rows:
            return
        new_file = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        with open(self.csv_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(["t", "label", "target", "speed", "pu", "status", "q", "v", "tau",
                                 "tmos", "trot", "grasp"])
            writer.writerows(self.rows)
        self.all_rows.extend(self.rows)
        self.rows = []


def calibrate(probe, sequence, q0):
    for label, target, speed, pu, duration, settle in SEQUENCES[sequence]:
        probe.run_segment(label, q0 if target == "q0" else float(target), speed, pu, duration, settle)
        probe.write_rows()
    print("标定汇总（每段后半段的力矩均值 / 位置范围）：")
    groups = {}
    for row in probe.all_rows:
        groups.setdefault(row[1], []).append(row)
    for label, rows in groups.items():
        tail = rows[len(rows) // 2:]
        taus = [float(r[8]) for r in tail]
        qs = [float(r[6]) for r in tail]
        print(f"  {label:22s} pu={rows[0][4]}  tau={sum(taus) / len(taus):+.3f} [{min(taus):+.3f}, {max(taus):+.3f}]"
              f"  q∈[{min(qs):+.4f}, {max(qs):+.4f}]")


# ── 端到端：Studio 生产代码 + 真实 J8 ─────────────────────────────────────────
class RealJ8Driver:
    """openarm_driver.SingleArmDriver 的最小替身：第 8 轴接到真实总线，其余 7 轴是空操作。"""

    kps = (70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0)
    kds = (2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5)
    joint_offsets = (0.0,) * 8
    gripper_posforce = True
    gripper_posforce_limits = (1.0, 0.6)
    safety_stop_reason = None

    def __init__(self, probe):
        import numpy as np
        self.np = np
        self.probe = probe
        self.last_command = None
        self.sent = []
        owner = self

        def link_stats(_index):
            return SimpleNamespace(
                is_stale=lambda timeout: time.time() - owner.probe.last_fb[0] > timeout,
                responses=owner.probe.frame_number)

        def get_motor():
            state = owner.probe.last_fb[1]
            return SimpleNamespace(
                is_enabled=lambda: state["status"] == 1, get_position=lambda: state["q"],
                get_velocity=lambda: state["v"], get_torque=lambda: state["tau"])

        self.gripper = SimpleNamespace(get_link_stats=link_stats, get_motor=get_motor,
                                       set_position=self.set_position)
        self.openarm = SimpleNamespace(
            get_arm=lambda: SimpleNamespace(mit_control_all=lambda _: None),
            get_gripper=lambda: self.gripper, recv_all=lambda *_: None)

    def set_position(self, position, speed_rad_s, torque_pu):
        self.probe.send(posforce(float(position), float(speed_rad_s), float(torque_pu)))
        self.sent.append((float(position), float(speed_rad_s), float(torque_pu)))

    def send_position(self, target):
        self.last_command = self.np.array(target, dtype=float)
        speed, limit = self.gripper_posforce_limits
        self.set_position(float(target[7]) + self.joint_offsets[7], speed, limit / 4.5)
        return True


def grasp_check(probe, side, panel_percent, q0):
    """张开 → request_grasp → 等判定结果并保持 1.5 s → 回到运行前的开口。返回最终抓取状态。"""
    import numpy as np
    from openarm_studio.backend.driver_arm import DriverArm

    drv = RealJ8Driver(probe)
    arm = DriverArm(side, "unused")
    arm._driver = drv
    arm._running = arm._enabled = True          # 控制线程与"已使能"由本脚本代替
    arm._grip_config_limits = (50.0, 1.0)
    arm.set_gripper_force(panel_percent)
    finger = f"openarm_{side}_finger_joint1"
    opening_before = round(-q0 / 1.0472 * 0.044, 4)
    trace = []
    drv.last_command = np.r_[np.zeros(7), q0]
    arm._set_motion_target(drv.last_command.copy())

    def pump(label, seconds, done):
        start = last_cmd = last_poll = time.time()
        next_note = start
        while time.time() - start < seconds:
            now = time.time()
            if now - last_poll >= POLL_PERIOD_S:
                probe.send(refresh(J8))
                last_poll = now
            if probe.poll_states():
                t, state = probe.last_fb
                sent = drv.sent[-1] if drv.sent else (float("nan"),) * 3
                probe.rows.append([f"{t:.6f}", label, f"{sent[0]:.6f}", f"{sent[1]:.3f}",
                                   f"{sent[2]:.5f}", state["status"], f"{state['q']:.6f}",
                                   f"{state['v']:.5f}", f"{state['tau']:.5f}", state["tmos"],
                                   state["trot"], arm.grasp_status])
            if now - start > 0.5:
                probe._guard(label)
            if now - last_cmd >= COMMAND_PERIOD_S:
                arm._send_current_target(drv)
                last_cmd = now
            state = probe.last_fb[1]
            if not trace or trace[-1][1] != arm.grasp_status:
                trace.append((round(now - trace_start, 3), arm.grasp_status, round(state["q"], 4),
                              round(state["v"], 3), round(state["tau"], 3)))
            if now >= next_note:
                print(f"  [{label}] t={now - start:5.2f}s status={arm.grasp_status:13s} q={state['q']:+.4f} "
                      f"v={state['v']:+.3f} tau={state['tau']:+.3f} "
                      f"cmd={tuple(round(x, 4) for x in drv.sent[-1]) if drv.sent else None}", flush=True)
                next_note = now + 0.5
            if done(now):
                return
            time.sleep(0.002)

    # 1) 张开（生产代码的滑块路径）
    arm.set_targets({finger: 0.044})
    trace_start = time.time()
    pump("open", 7.0, lambda now: probe.last_fb[1]["q"] < -1.04 and abs(probe.last_fb[1]["v"]) < 0.05)
    if probe.last_fb[1]["q"] > -1.03:
        raise Aborted("张开没有到位")
    open_at = time.time()

    # 2) 智能抓取
    if not arm.request_grasp():
        raise Aborted(f"request_grasp 被拒绝（status={arm.grasp_status}）")
    seen_since = {}

    def decided(now):
        if arm.grasp_status in ("pending", "closing"):
            return False
        seen_since.setdefault(arm.grasp_status, now)
        return now - seen_since[arm.grasp_status] > 1.5     # 判定后多保持 1.5 s，读堵转力矩

    pump("grasp", 8.0, decided)
    status = arm.grasp_status
    decided_at = next(t for t, s, *_ in trace if s not in ("idle", "pending", "closing"))
    print(f"  → 判定结果 {status}，用时 {decided_at - (open_at - trace_start):.2f}s，"
          f"界面显示的电流上限 {arm.grasp_torque_limit:.2f} N·m", flush=True)

    # 3) 滑块接管，回到运行前的开口
    arm.set_targets({finger: opening_before})
    pump("back", 6.0, lambda now: abs(probe.last_fb[1]["q"] - q0) < 0.012
         and abs(probe.last_fb[1]["v"]) < 0.05)
    print(f"  → 拖滑块后 status={arm.grasp_status}", flush=True)
    for row in trace:
        print("     状态轨迹", row)
    return status, arm.grasp_status


# ── 入口 ─────────────────────────────────────────────────────────────────────
def _on_sigterm(_signum, _frame):
    raise Aborted("SIGTERM")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bus", default="can0", help="can0=右臂，can1=左臂")
    parser.add_argument("--side", default="right", choices=["left", "right"])
    parser.add_argument("--out", default=os.path.join(ROOT, "records", "gripper_probe"))
    parser.add_argument("--sequence", default="calibrate", choices=sorted(SEQUENCES))
    parser.add_argument("--grasp-check", action="store_true", help="走 Studio 生产代码点一次智能抓取")
    parser.add_argument("--panel", type=int, default=60, help="抓取自检用的电流上限 %%（Studio 默认 60）")
    parser.add_argument("--run", action="store_true", help="真的使能 J8 并运动；不加只做前置检查")
    parser.add_argument("--simulate", action="store_true", help="用仿真电机，不碰硬件")
    args = parser.parse_args()

    try:
        bus = SimBus() if args.simulate else CanBus(args.bus)
    except OSError as exc:
        print(f"打不开 {args.bus}：{exc}（接口未启用？运行 tools/can_reset.sh）")
        return 2
    probe = Probe(bus, args.out, simulate=args.simulate)
    log_path = probe.csv_path[:-4] + ".log"
    if not args.simulate:
        probe.dump = open(log_path, "w")
    outcome = 1
    try:
        try:
            states, mode = probe.precheck()
        except Aborted as exc:
            print(f"拒绝执行：{exc}")
            return 2
        for mid, state in states.items():
            print(f"J{mid}: status={state['status']} q={state['q']:+.4f} tau={state['tau']:+.3f} "
                  f"tmos={state['tmos']} trot={state['trot']}")
        print(f"J8 CTRL_MODE={mode}")
        if not args.run:
            print("前置检查通过。未加 --run：没有使能任何电机。")
            return 0
        q0 = states[J8]["q"]
        print(f"使能 {getattr(bus, 'iface', '仿真')} 上的 J8（q={q0:+.4f}）")
        signal.signal(signal.SIGTERM, _on_sigterm)
        probe.send(ENABLE)
        probe.enabled = probe.moved = True
        time.sleep(0.05)
        state = probe.read_state(J8, 0.3)
        if state is None or state["status"] != 1:
            raise Aborted(f"J8 没有使能成功（status={None if state is None else state['status']}）")
        probe.last_fb = (time.time(), state)
        if args.grasp_check:
            status, after = grasp_check(probe, args.side, args.panel, q0)
            outcome = 0 if (status in ("empty", "holding") and after == "idle") else 1
        else:
            calibrate(probe, args.sequence, q0)
            outcome = 0
    except KeyboardInterrupt:
        print("被 Ctrl-C 中断")
    except Aborted as exc:
        print(f"已中止：{exc}")
    finally:
        if probe.enabled:
            probe.disable()
        probe.write_rows()
        if probe.moved and not args.simulate:
            state = None
            for _ in range(10):
                state = probe.read_state(J8, 0.3)
                if state is not None and state["status"] == 0:
                    break
            print(f"J8 已失能：status={None if state is None else state['status']} "
                  f"q={None if state is None else round(state['q'], 4)}")
        bus.close()
        if probe.dump:
            probe.dump.close()
        if not probe.moved:
            # 只做了前置检查：不留文件
            for path in (probe.csv_path, log_path):
                if os.path.exists(path):
                    os.remove(path)
            try:
                os.rmdir(os.path.dirname(probe.csv_path))
            except OSError:
                pass
            return outcome
        print(f"数据：{probe.csv_path}" + ("" if args.simulate else f"  帧记录：{log_path}"))
    return outcome


if __name__ == "__main__":
    raise SystemExit(main())
