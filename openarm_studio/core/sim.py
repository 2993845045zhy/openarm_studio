"""仿真后端：MuJoCo 物理 + 位置伺服。

与数字孪生的区别只有一处 —— 孪生是"外部给 qpos"，仿真是"外部给目标，
物理自己算出 qpos"。渲染那条路径两者完全共用。
"""

from __future__ import annotations

import mujoco

from ..model.scene import ALL_ACTUATED, joint_qpos_map

PHYS_DT = 0.001       # 1kHz 物理步长（照官方 openarm_mujoco 的 option）
SUBSTEPS = 10         # 每帧推进 10ms


class SimBackend:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self._qmap = joint_qpos_map(model)
        self.model.opt.timestep = PHYS_DT
        # 关节名 → actuator 下标（actuator 命名规则见 scene._actuators）
        self._amap: dict[str, int] = {}
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            for side in ("left", "right"):
                for j in range(1, 8):
                    if name == f"{side}_joint{j}_act":
                        self._amap[f"openarm_{side}_joint{j}"] = i
                if name == f"{side}_grip_act":
                    self._amap[f"openarm_{side}_finger_joint1"] = i
        self._targets: dict[str, float] = {n: 0.0 for n in ALL_ACTUATED}
        self.reset()

    def sync_targets_to_current(self) -> None:
        """把伺服目标对齐到当前 qpos —— 切进仿真时调用。

        目标留在别处（比如孪生最后一帧姿态）会让刚度 230 的伺服一瞬间
        把臂拽过去，看起来就是"一点仿真就乱动"。
        """
        for name, idx in self._qmap.items():
            if name in self._targets:
                self._targets[name] = float(self.data.qpos[idx])

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.sync_targets_to_current()
        for _ in range(250):
            self._apply_ctrl()
            mujoco.mj_step(self.model, self.data)

    def _apply_ctrl(self) -> None:
        for name, idx in self._amap.items():
            self.data.ctrl[idx] = self._targets.get(name, 0.0)

    def set_targets(self, targets: dict[str, float]) -> None:
        for name, val in targets.items():
            if name in self._amap:
                self._targets[name] = float(val)

    def nudge(self, deltas: dict[str, float]) -> dict[str, float]:
        """在当前位置上叠加增量（关节滑块用，避免一次跳变）。"""
        for name, delta in deltas.items():
            idx = self._qmap.get(name)
            if idx is not None:
                self._targets[name] = self._targets.get(name, 0.0) + float(delta)
        return dict(self._targets)

    def step(self) -> None:
        self._apply_ctrl()
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

    def settle(self, steps: int = 200) -> None:
        """空跑若干物理步让瞬态衰减（不渲染），用于模式切换/启动预热。"""
        for _ in range(max(1, int(steps))):
            self._apply_ctrl()
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def positions(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, idx in self._qmap.items():
            out[name] = float(self.data.qpos[idx])
        return out

    def has_actuators(self) -> bool:
        return len(self._amap) == len(ALL_ACTUATED)
