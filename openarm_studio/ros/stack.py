"""ROS 2 真机栈生命周期：启动 / 停止 bringup + move_group。

两条铁律（上次事故的教训）：
1. 停止必须 pgrep 确认进程归零，不信任 launch 退出码
2. 启动前探测栈进程，已有实例就只连接不起栈

安全：启动用固定字面量参数列表（shell=False），命令行内容不可被外部
影响；停止用 pgrep/os.kill 按 pid 操作，不经过 shell。
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

LAUNCH_SCRIPT = str(Path.home() / "openarm_studio/tools/launch_real.sh")

# 停栈时按这个顺序杀（先停控制器宿主，再停 move_group）
_PATTERNS = ("ros2_control_node", "move_group", "spawner", "demo.launch.py")


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""


def _pids() -> list[int]:
    """栈进程 pid 列表。

    必须排除"执行 pgrep 的那个 shell"——`pgrep -f <pattern>` 会匹配到
    自身命令行（里面有 pattern 文本），否则永远判定"已有栈在跑"。
    """
    out: list[int] = []
    for pat in _PATTERNS:
        try:
            r = subprocess.run(["pgrep", "-f", pat], capture_output=True,
                               text=True, timeout=4, shell=False)
            for line in r.stdout.split():
                if not line.isdigit():
                    continue
                p = int(line)
                if p == os.getpid() or p in out:
                    continue
                cl = _cmdline(p)
                if not cl or "pgrep" in cl or "bash -c" in cl:
                    continue
                out.append(p)
        except Exception:  # noqa: BLE001
            pass
    return out


class StackManager:
    """管理真机栈子进程；同时支持"只连接已运行实例"。"""

    def __init__(self, on_log=None):
        self._log = on_log or (lambda *_: None)
        self._proc: subprocess.Popen | None = None

    @property
    def running(self) -> bool:
        return bool(_pids())

    def probe(self) -> int:
        return len(_pids())

    def start(self, *, fake: bool = False) -> bool:
        n = self.probe()
        if n:
            self._log(f"检测到已有 {n} 个栈进程在跑——只连接，不再启动")
            return True
        if not os.path.exists(LAUNCH_SCRIPT):
            self._log(f"缺少启动脚本 {LAUNCH_SCRIPT}")
            return False
        if fake:
            self._proc = subprocess.Popen(
                ["bash", LAUNCH_SCRIPT, "--fake"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True, shell=False)
        else:
            self._proc = subprocess.Popen(
                ["bash", LAUNCH_SCRIPT],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True, shell=False)
        self._log(f"已启动真机栈（pid {self._proc.pid}），等待 controller_manager…")
        return True

    def stop(self, timeout: float = 8.0) -> bool:
        self._log("停止栈：先发 SIGINT，超时后 SIGKILL")
        for pid in _pids():
            try:
                os.kill(pid, signal.SIGINT)
            except Exception:  # noqa: BLE001
                pass
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not _pids():
                break
            time.sleep(0.25)
        for pid in _pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(0.4)
        left = _pids()
        self._log("栈已停止（进程确认归零）" if not left
                  else f"仍有进程残留：{left}")
        if self._proc is not None:
            try:
                self._proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass
            self._proc = None
        return not left
