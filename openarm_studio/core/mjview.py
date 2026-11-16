"""内嵌 MuJoCo 渲染后端（无 Qt 依赖，只产出 RGB 数组）。

照 pbvs_workbench/core/mjview.py 的做法：
- 用 `mujoco.Renderer`（内部持有 MjrContext），GL 后端由 MUJOCO_GL 决定
- EGL 上下文**线程绑定**：谁创建谁销毁，`close()` 必须在创建线程调用
- `Renderer.render()` 复用内部缓冲，返回前必须 `.copy()`

本模块不持有 model/data 的所有权，只负责“给定 model/data/相机 → 一帧 RGB”。
"""

from __future__ import annotations

import atexit

import mujoco
import numpy as np

# 默认取景（臂展约 0.6m，机身立在原点附近）
DEFAULT_VIEW = {
    "lookat": (0.0, 0.0, 0.32),
    "distance": 1.75,
    "azimuth": 135.0,
    "elevation": -18.0,
}

_RES_H_QUANT = 64
_RES_H_MIN, _RES_H_MAX = 384, 1200
_OFF_W_MAX, _OFF_H_MAX = 1600, 1200
_ASPECT_MIN, _ASPECT_MAX = 0.5, 3.0

# EGL/GLFW 的 atexit 顺序：mujoco.egl 第一次建上下文时注册 eglTerminate；
# 我们必须在它之前关闭自己的上下文，所以延后注册（LIFO 先跑）。
_GL_OWNERS: list = []


def _register_gl(obj) -> None:
    if obj not in _GL_OWNERS:
        _GL_OWNERS.append(obj)


def _close_all_gl() -> None:
    for obj in list(_GL_OWNERS):
        try:
            obj.close()
        except Exception:  # noqa: BLE001
            pass
    _GL_OWNERS.clear()


def install_gl_atexit() -> None:
    """在第一次 render 之后调用一次即可。"""
    atexit.register(_close_all_gl)


def apply_cam_state(cam: mujoco.MjvCamera, state: dict) -> None:
    lk = state.get("lookat", DEFAULT_VIEW["lookat"])
    cam.lookat[:] = lk
    cam.distance = float(state.get("distance", DEFAULT_VIEW["distance"]))
    cam.azimuth = float(state.get("azimuth", DEFAULT_VIEW["azimuth"]))
    cam.elevation = float(state.get("elevation", DEFAULT_VIEW["elevation"]))


class MjEmbedRenderer:
    def __init__(self, height: int = 720):
        self._h = int(height)
        self._model_ref: mujoco.MjModel | None = None
        self._renderer: mujoco.Renderer | None = None
        self._opt: mujoco.MjvOption | None = None
        self._cam: mujoco.MjvCamera | None = None
        self._registered = False

    @property
    def alive(self) -> bool:
        return self._renderer is not None

    def _quantize(self, h: float) -> int:
        q = int(round(h / _RES_H_QUANT)) * _RES_H_QUANT
        return max(_RES_H_MIN, min(_RES_H_MAX, q))

    def _ensure(self, model: mujoco.MjModel, aspect: float) -> None:
        aspect = max(_ASPECT_MIN, min(_ASPECT_MAX, aspect))
        h = self._quantize(self._h)
        w = self._quantize(h * aspect)
        w = min(w, _OFF_W_MAX)
        h = min(h, _OFF_H_MAX)
        if (self._renderer is not None and self._model_ref is model
                and self._renderer.height == h and self._renderer.width == w):
            return
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._renderer = mujoco.Renderer(model, height=h, width=w)
        self._model_ref = model
        if self._opt is None:
            self._opt = mujoco.MjvOption()
            # 默认只开 0-2 组；障碍/线框在 3-5 组
            self._opt.geomgroup[3] = 1
            self._opt.geomgroup[4] = 1
            self._opt.geomgroup[5] = 1
        if self._cam is None:
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            apply_cam_state(self._cam, DEFAULT_VIEW)

    def render(self, model: mujoco.MjModel, data: mujoco.MjData,
               cam_state: dict) -> np.ndarray:
        self._ensure(model, float(cam_state.get("aspect", 16 / 9)))
        assert self._renderer is not None and self._opt is not None and self._cam is not None
        apply_cam_state(self._cam, cam_state)
        self._opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = bool(
            cam_state.get("transparent", False))
        self._renderer.update_scene(data, camera=self._cam, scene_option=self._opt)
        scn = self._renderer.scene
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 1
        scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = bool(
            cam_state.get("wireframe", False))
        img = self._renderer.render()
        if not self._registered:
            _register_gl(self)
            install_gl_atexit()
            self._registered = True
        return np.array(img, copy=True)

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            finally:
                self._renderer = None
        self._model_ref = None
