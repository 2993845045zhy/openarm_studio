"""末端逆运动学：复用官方 openarm_control（mink + daqp QP）。

模型用官方 `enactic/openarm_mujoco` 的 v1 双臂 MJCF（本机为 v1.0 硬件）。
实测：该模型与 Studio 自建模型的末端位置差 <1mm，运动学一致；
动力学（质量/惯量）差异见 backend/gravity.py 的说明。

用法：
    ik = ArmIK.shared()
    pose = ik.fk("right", q8)          # 7 维位姿 [x,y,z,qw,qx,qy,qz]
    q8    = ik.ik("right", pose, q8_other)   # 需给另一臂当前角作为副目标
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from ..paths import mujoco_v1_xml

V1_XML = mujoco_v1_xml()


class ArmIK:
    """双臂 IK（线程安全，进程内单例）。"""

    _instance: "ArmIK | None" = None
    _lock = threading.Lock()

    @classmethod
    def available(cls) -> bool:
        return V1_XML.exists()

    @classmethod
    def shared(cls) -> "ArmIK":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self, xml: Path | None = None):
        from openarm_control import ArmSetup, IKParams, Kinematics

        path = Path(xml) if xml else V1_XML
        if not path.exists():
            raise FileNotFoundError(f"官方 v1 模型不存在：{path}")
        setup = ArmSetup.from_args(
            xml=str(path), mode="bimanual",
            # 用 hand body 作为控制点：ee_control_point site 在腕部靠内
            # （差 12cm），球会显示在手腕上而不是夹爪上。
            frame_right="openarm_right_hand", frame_type_right="body",
            frame_left="openarm_left_hand", frame_type_left="body",
            keyframe="home",
            # 与 Studio 的 3D 视图统一用 world 系（视图拖拽目标即 world 坐标）
            origin_frame="world", origin_frame_type="body",
        )
        # 姿态权重从默认 1.5 提到 6.0：默认太轻时 J6/J7 的姿态几乎不约束，
        # 表现为"最后一个关节跟着乱转、控制不了"。
        self._kin = Kinematics(setup, IKParams(orientation_cost=6.0))
        self._io = threading.Lock()

    def fk(self, side: str, q8) -> list[float]:
        """关节角（8 维：7 关节 + 夹爪）→ 末端位姿 [x,y,z,qw,qx,qy,qz]。"""
        with self._io:
            return [float(v) for v in np.asarray(
                self._kin.fk(side, np.asarray(q8, dtype=np.float32)))]

    def ik(self, side: str, pose7, other_q8=None,
           iters: int = 6) -> np.ndarray | None:
        """末端位姿 → 8 维关节角。other_q8 为另一臂当前角（副目标保持）。

        mink 的 QP 是增量式，单次 solve 只走一步；这里迭代 iters 次让
        残差收敛（实测抬高 5cm 时 z 误差 <1mm）。
        """
        target = np.asarray(pose7, dtype=np.float32)
        other = None if other_q8 is None else np.asarray(other_q8, dtype=np.float32)
        other_side = "left" if side == "right" else "right"
        with self._io:
            try:
                for _ in range(max(1, iters)):
                    self._kin.set_target(side, target)
                    if other is not None:
                        self._kin.set_target(
                            other_side, np.asarray(self._kin.fk(other_side, other)))
                    sol = self._kin.solve()
                    if sol is None:
                        return None
                return np.asarray(sol[:8] if side == "right" else sol[8:],
                                  dtype=float)
            except Exception:  # noqa: BLE001
                return None
