"""重力矩计算：优先用官方 v1 MuJoCo 模型，回退 KDL。

两条后端：

1. **MuJoCo（首选）**：官方 `enactic/openarm_mujoco` 仓库的 `v1/openarm_bimanual.xml`
   （真机为 v1.0；PyPI 的 openarm-mujoco 只打包了 v2）。用 `mj_forward` 后
   的 `qfrc_bias`（重力+科氏偏置力）取前 7 个臂关节。
   实测：随附的双臂 URDF 与官方 v1 模型
   在连杆质量和关节轴上都不同（差异最高 243%），校准后的官方模型才可信。

2. **KDL（回退）**：enactic/openarm_teleop 的算法（URDF→KDL→ChainDynParam），
   依赖 openarm_description 的 URDF；模型不含 v1 精确惯性，仅作兜底。
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from ..paths import mujoco_v1_xml

GRAVITY = (0.0, 0.0, -9.81)
_BASE_LINK = "openarm_body_link0"
_JOINTS_PER_ARM = 7

# 关节摩擦模型（官方 openarm_teleop，逐关节标定，左右臂同一套）：
#     τ_fric(ω) = Fo + Fv·ω + Fc·tanh(coef·k·ω)
# 参数取自 config/{leader,follower}.yaml 的前 7 项（第 8 项是夹爪，
# 夹爪走 POS_FORCE 的电流限幅，不走这条）。
# 注意：官方 tanh 里还有 coef_tmp=0.1（见 Control::ComputeFriction）。
_FRIC_COEF = 0.1
_FRIC_FC = np.array([0.306, 0.306, 0.40, 0.166, 0.050, 0.083, 0.172])
_FRIC_K = np.array([28.417, 28.417, 29.065, 130.038, 151.771, 242.287, 7.888])
_FRIC_FV = np.array([0.063, 0.063, 0.604, 0.813, 0.029, 0.072, 0.084])
# Fo 是恒定偏置（线缆/装配应力），不是速度相关项；拖动时加上会让臂
# 缓慢漂移，故默认不下发（include_bias=True 才加）。
_FRIC_FO = np.array([0.088, 0.088, 0.008, -0.058, 0.005, 0.009, -0.059])


def _build_chain(robot, root: str, tip: str, kdl):
    """URDF 树 → KDL Chain（等价 kdl_parser 的 getChain）。"""
    by_child = {j.child: j for j in robot.joints}
    by_name = {ln.name: ln for ln in robot.links}
    path, cur = [], tip
    while cur != root:
        j = by_child.get(cur)
        if j is None:
            raise RuntimeError(f"URDF 中找不到 {root} → {tip} 的链（断在 {cur}）")
        path.append(j)
        cur = j.parent

    chain = kdl.Chain()
    names: list[str] = []
    for j in reversed(path):
        rpy, xyz = j.origin.rpy, j.origin.xyz
        if j.type in ("revolute", "continuous"):
            axis = j.axis if j.axis is not None else [0.0, 0.0, 1.0]
            jnt = kdl.Joint(j.name, kdl.Vector(0, 0, 0),
                              kdl.Vector(*axis), kdl.Joint.RotAxis)
            names.append(j.name)
        else:
            jnt = kdl.Joint(j.name, kdl.Joint.Fixed)
        link = by_name.get(j.child)
        inr = link.inertial if link is not None else None
        if inr is None:
            rb = kdl.RigidBodyInertia(0.0)
        else:
            i = inr.inertia
            rb = kdl.RigidBodyInertia(
                float(inr.mass),
                kdl.Vector(*inr.origin.xyz),
                kdl.RotationalInertia(i.ixx, i.iyy, i.izz,
                                        i.ixy, i.ixz, i.iyz),
            )
        chain.addSegment(kdl.Segment(
            j.child, jnt,
            kdl.Frame(kdl.Rotation.RPY(*rpy), kdl.Vector(*xyz)),
            rb,
        ))
    return chain, names


class GravityModel:
    """双臂重力矩模型（线程安全，双臂共享一份）。

    后端优先级：官方 v1 MuJoCo 模型 → KDL（URDF 回退）。
    """

    _instance: "GravityModel | None" = None
    _lock = threading.Lock()

    # 官方 v1 MuJoCo 模型位置（enactic/openarm_mujoco 仓库）
    MUJOCO_CANDIDATES = (mujoco_v1_xml(),)

    @classmethod
    def shared(cls) -> "GravityModel":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.backend = ""
        self._mj = None
        self._mj_data = None
        self._mj_qadr: dict[str, int] = {}
        self._mj_dofadr: dict[str, int] = {}
        self._mj_body: dict[str, int] = {}
        self._order: dict[str, list[str]] = {}
        self._chains = {}
        self._solvers = {}
        self._jac_solvers = {}
        self._fk_solvers = {}

        for cand in self.MUJOCO_CANDIDATES:
            if cand.exists():
                try:
                    self._init_mujoco(cand)
                    self.backend = f"mujoco-v1 ({cand.parent.name})"
                    return
                except Exception:  # noqa: BLE001
                    pass
        self._init_kdl()
        self.backend = "kdl-urdf"

    # ── MuJoCo 后端（官方 v1 模型）──────────────────────
    def _init_mujoco(self, path: Path) -> None:
        import mujoco

        m = mujoco.MjModel.from_xml_path(str(path))
        self._mj = m
        self._mj_data = mujoco.MjData(m)
        for i in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self._mj_qadr[name] = m.jnt_qposadr[i]
                self._mj_dofadr[name] = m.jnt_dofadr[i]
        for side in ("left", "right"):
            names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
            missing = [n for n in names if n not in self._mj_qadr]
            if missing:
                raise RuntimeError(f"官方 v1 模型缺少关节：{missing}")
            self._order[side] = names
            body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                     f"openarm_{side}_hand")
            if body < 0:
                raise RuntimeError(f"官方 v1 模型缺少 body：openarm_{side}_hand")
            self._mj_body[side] = body

    # ── KDL 后端（URDF 回退）───────────────────────────
    def _init_kdl(self) -> None:
        try:
            import PyKDL
        except ImportError:
            import sys

            sys.path.append("/usr/lib/python3/dist-packages")
            import PyKDL
        from urdf_parser_py.urdf import URDF

        from ..model.urdf2mjcf import BIMANUAL_URDF

        self._kdl = PyKDL
        robot = URDF.from_xml_string(BIMANUAL_URDF.read_text())
        for side in ("left", "right"):
            chain, names = _build_chain(robot, _BASE_LINK, f"openarm_{side}_hand", PyKDL)
            if len(names) != _JOINTS_PER_ARM:
                raise RuntimeError(f"{side} 臂关节数异常：{len(names)}")
            self._order[side] = names
            # Chain 必须长期持有：PyKDL.ChainDynParam 内部保存对 Chain
            # 的引用，若 Chain 被 GC，JntToGravity 会返回全 0。
            self._chains[side] = chain
            self._solvers[side] = PyKDL.ChainDynParam(
                chain, PyKDL.Vector(*GRAVITY))
            self._jac_solvers[side] = PyKDL.ChainJntToJacSolver(chain)
            self._fk_solvers[side] = PyKDL.ChainFkSolverPos_recursive(chain)

    def gravity(self, side: str, q: list[float] | np.ndarray) -> np.ndarray:
        """给定 7 维关节角（rad），返回各关节重力矩（Nm）。"""
        n = len(self._order[side])
        if self._mj is not None:
            import mujoco

            with self._lock:
                d = self._mj_data
                for i, name in enumerate(self._order[side]):
                    d.qpos[self._mj_qadr[name]] = float(q[i])
                d.qvel[:] = 0.0
                mujoco.mj_forward(self._mj, d)
                return np.array([
                    d.qfrc_bias[self._mj_dofadr[self._order[side][i]]]
                    for i in range(n)], dtype=float)
        solver = self._solvers[side]
        jq = self._kdl.JntArray(n)
        for i in range(n):
            jq[i] = float(q[i])
        out = self._kdl.JntArray(n)
        solver.JntToGravity(jq, out)
        return np.array([out[i] for i in range(n)], dtype=float)

    def basis(self, side: str) -> list[str]:
        """链上关节名（顺序与 gravity() 返回一致）。"""
        return list(self._order[side])

    # ── 摩擦（官方 tanh 模型）──────────────────────────
    def friction(self, side: str, qvel, include_bias: bool = False) -> np.ndarray:
        """给定 7 维关节角速度（rad/s），返回需**施加**的摩擦补偿力矩（Nm）。

        模型与官方 `Control::ComputeFriction` 一致：
        `Fc·tanh(0.1·k·ω) + Fv·ω (+ Fo)`。ω>0 时结果为正（驱动需同向出力）。

        `include_bias` 默认 False：Fo 是恒定偏置，拖动模式下会让臂缓慢
        漂移；官方把它放在 leader 臂上是为了双边力反馈的零点一致。
        """
        v = np.asarray(qvel, dtype=float).reshape(-1)[:7]
        if v.size != 7:
            return np.zeros(7)
        f = _FRIC_FC * np.tanh(_FRIC_COEF * _FRIC_K * v) + _FRIC_FV * v
        if include_bias:
            f = f + _FRIC_FO
        return f

    # ── 运动学（导纳控制用）────────────────────────────
    def _set_pose(self, side: str, q) -> None:
        """把 7 维关节角写进模型并 forward（调用方需持 self._lock）。"""
        import mujoco

        d = self._mj_data
        for i, name in enumerate(self._order[side]):
            d.qpos[self._mj_qadr[name]] = float(q[i])
        d.qvel[:] = 0.0
        mujoco.mj_forward(self._mj, d)

    def jacobian(self, side: str, q) -> np.ndarray:
        """末端（hand body）平动雅可比，3×7，世界系。"""
        if self._mj is not None:
            import mujoco

            with self._lock:
                self._set_pose(side, q)
                d = self._mj_data
                bid = self._mj_body[side]
                jacp = np.zeros((3, self._mj.nv))
                jacr = np.zeros((3, self._mj.nv))
                mujoco.mj_jac(self._mj, d, jacp, jacr, d.xpos[bid], bid)
                cols = [self._mj_dofadr[n] for n in self._order[side]]
                return np.array(jacp[:, cols], dtype=float)
        solver = self._jac_solvers[side]
        jq = self._kdl.JntArray(7)
        for i in range(7):
            jq[i] = float(q[i])
        jac = self._kdl.Jacobian(7)
        solver.JntToJac(jq, jac)
        return np.array([[jac[row, col] for col in range(7)]
                         for row in range(3)], dtype=float)

    def ee_position(self, side: str, q) -> np.ndarray:
        """末端（hand body）位置，世界系 3 维。"""
        if self._mj is not None:
            with self._lock:
                self._set_pose(side, q)
                return np.array(self._mj_data.xpos[self._mj_body[side]],
                                dtype=float).copy()
        solver = self._fk_solvers[side]
        jq = self._kdl.JntArray(7)
        for i in range(7):
            jq[i] = float(q[i])
        frame = self._kdl.Frame()
        solver.JntToCart(jq, frame)
        return np.array([frame.p.x(), frame.p.y(), frame.p.z()], dtype=float)
