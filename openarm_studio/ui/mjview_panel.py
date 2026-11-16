"""内嵌 3D 面板：显示、阻尼相机交互、世界轴指示器、开场运镜。

pbvs_workbench 同款标准：
- 相机交互：左拖旋转（灵敏度随宽度归一）/ 右拖平移 / 滚轮缩放 /
  双击复位（复用开场运镜的时长与缓动），指数阻尼跟随
- 世界轴指示器（右上角悬浮 gizmo）：三色轴 + 深度排序 + 超采样抗锯齿，
  投影公式与 MuJoCo 自由相机逐分量吻合（_cam_basis）
- 开场运镜：远机位环绕推近（1500ms OutCubic），全程线框，到位后
  线框→实体交叉淡入；用户一上手立即中止

线程契约：渲染线程 push_frame(rgb) / 读 cam_state()；GUI 线程 16ms 轮询。
"""

from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QEasingCurve, Qt, QTimer, QVariantAnimation
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget

from ..core.mjview import DEFAULT_VIEW

_POLL_MS = 16
_SMOOTH_MS = 8
_CAM_TAU_S = 0.10
_CAM_TAU_ZOOM_S = 0.15
_SMOOTH_DT_MAX_S = 0.10

_ORBIT_TURN_DEG = 300.0
_ORBIT_DEG_PX_RANGE = (0.15, 0.55)
_ELEV_CLAMP = (-85.0, 85.0)
_DIST_CLAMP = (0.35, 8.0)
_PAN_SENS = 0.0018

VIEW_PRESETS = {
    "home": dict(DEFAULT_VIEW),
    "front": {"lookat": (0.0, 0.0, 0.32), "distance": 1.75, "azimuth": 180.0, "elevation": -8.0},
    "side": {"lookat": (0.0, 0.0, 0.32), "distance": 1.75, "azimuth": 90.0, "elevation": -8.0},
    "top": {"lookat": (0.0, 0.0, 0.25), "distance": 1.9, "azimuth": 135.0, "elevation": -70.0},
}

# ── 世界轴指示器（pbvs 同款三色 + 数学）────────────────────
_AXIS_DEFS = ((np.array([1.0, 0.0, 0.0]), (255, 92, 92), "X"),
              (np.array([0.0, 1.0, 0.0]), (70, 191, 114), "Y"),
              (np.array([0.0, 0.0, 1.0]), (64, 153, 255), "Z"))
_GIZMO_DEG_EPS = 0.05

# ── 开场运镜 ────────────────────────────────────────────────
_INTRO_MS = 1500
_INTRO_FAR = {"distance": 3.2, "azimuth": 177.0, "elevation": -10.0}
_INTRO_XFADE_S = 0.85
_INTRO_FADE_S = 0.60


def _hex_rgb(color) -> tuple:
    if isinstance(color, (tuple, list)):
        return tuple(int(c) for c in color[:3])
    c = str(color).lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def _ang_diff(a: float, b: float) -> float:
    return ((b - a + 180.0) % 360.0) - 180.0


def _cam_basis(az: float, el: float):
    """MuJoCo 自由相机基向量（前/右/上），与 scene.camera 逐分量吻合。"""
    azr, elr = math.radians(az), math.radians(el)
    fwd = np.array([math.cos(elr) * math.cos(azr),
                    math.cos(elr) * math.sin(azr),
                    math.sin(elr)])
    right = np.array([math.sin(azr), -math.cos(azr), 0.0])
    up = np.cross(right, fwd)
    return fwd, right, up


def _gizmo_items(az: float, el: float):
    fwd, right, up = _cam_basis(az, el)
    items = []
    for axis, color, _label in _AXIS_DEFS:
        z = float(axis @ fwd)
        sx, sy = float(axis @ right), float(-(axis @ up))
        ln = math.hypot(sx, sy)
        deg = ln < _GIZMO_DEG_EPS
        items.append((z, sx / ln if not deg else 0.0,
                      sy / ln if not deg else 0.0, deg, color))
    items.sort(key=lambda t: -t[0])   # 远 → 近
    return items


class MujocoViewPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mjPanel")
        self.setMinimumSize(320, 220)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.NoContextMenu)

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._wire_frame: np.ndarray | None = None   # 运镜线框源帧
        self._frame_count = 0
        self._pal: dict = {}

        self.lbl = QLabel("3D 场景准备中…", self)
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setMinimumSize(1, 1)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self.lbl, 0, 0)

        self._cam = dict(VIEW_PRESETS["home"])
        self._cam_s = dict(VIEW_PRESETS["home"])
        self._wireframe = False
        self._transparent = False
        self._collision = False
        self._press = None
        self._last_t = time.monotonic()
        # 末端拖拽目标（世界坐标）+ 当前末端位姿
        self._ee_world: tuple | None = None
        self._ee_quat: tuple | None = None
        self._target_world: tuple | None = None
        self._drag_marker = False
        self._drag_base = None
        self.on_target_drag = None      # callback(world_xyz, done: bool)
        self._fovy = 45.0

        # 开场运镜状态
        self._intro_t0 = time.monotonic()
        self._intro_done = False
        self._xfade_t0: float | None = None
        self._fade_t0 = time.monotonic()
        self._intro = QVariantAnimation(self)
        self._intro.setDuration(_INTRO_MS)
        self._intro.setStartValue(0.0)
        self._intro.setEndValue(1.0)
        self._intro.setEasingCurve(QEasingCurve.OutCubic)
        self._intro.valueChanged.connect(self._on_intro_value)
        self._intro.finished.connect(self._on_intro_end)

        self._poll = QTimer(self)
        self._poll.setInterval(_POLL_MS)
        self._poll.timeout.connect(self._on_poll)
        self._poll.start()

        self._smooth = QTimer(self)
        self._smooth.setInterval(_SMOOTH_MS)
        self._smooth.timeout.connect(self._on_smooth)
        self._smooth.start()

    # ── 渲染线程接口 ───────────────────────────────────
    def push_frame(self, rgb: np.ndarray, wire: bool | None = None) -> None:
        with self._lock:
            self._frame = rgb
            self._frame_count += 1
            if wire is not None and wire:
                self._wire_frame = rgb

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def cam_state(self) -> dict:
        with self._lock:
            st = dict(self._cam_s)
        st["aspect"] = max(0.4, self.width() / max(1, self.height()))
        st["wireframe"] = (not self._intro_done) or self._wireframe
        st["transparent"] = self._transparent
        st["collision"] = self._collision
        return st

    # ── 末端目标标记（拖拽规划）──────────────────────
    def set_ee_pose(self, pos: tuple | None, quat: tuple | None = None) -> None:
        self._ee_world = pos
        self._ee_quat = quat
        if self._target_world is None and pos is not None:
            self._target_world = pos

    def ee_quat(self) -> tuple | None:
        return self._ee_quat

    def clear_target(self) -> None:
        self._target_world = None

    def _project(self, p_world) -> tuple[float, float, float] | None:
        """世界点 → 面板像素坐标 + 深度（相机基向量投影）。"""
        with self._lock:
            cam = dict(self._cam_s)
        look = np.array(cam["lookat"], dtype=float)
        fwd, right, up = _cam_basis(cam["azimuth"], cam["elevation"])
        rel = np.array(p_world, dtype=float) - look
        depth = float(rel @ fwd)
        if depth <= 0.05:
            return None
        half_h = math.tan(math.radians(self._fovy) / 2.0) * depth
        half_w = half_h * (self.width() / max(1, self.height()))
        sx = float(rel @ right) / half_w
        sy = float(rel @ up) / half_h
        if abs(sx) > 1.2 or abs(sy) > 1.2:
            return None
        return ((sx + 1.0) * 0.5 * self.width(),
                (1.0 - sy) * 0.5 * self.height(), depth)

    def _draw_target_marker(self, rgb: np.ndarray) -> None:
        if self._target_world is None:
            return
        pr = self._project(self._target_world)
        if pr is None:
            return
        x, y, _d = pr
        pal = self._pal or {}
        col = _hex_rgb(pal.get("accent", "#4099ff"))
        bgr = (col[2], col[1], col[0])
        h, w = rgb.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        cv2.circle(rgb, (int(x), int(y)), 11, bgr, 2, cv2.LINE_AA)
        cv2.circle(rgb, (int(x), int(y)), 2, bgr, -1, cv2.LINE_AA)
        for dx, dy in ((0, -17), (0, 17), (-17, 0), (17, 0)):
            cv2.line(rgb, (int(x), int(y)),
                     (int(x + dx * 0.6), int(y + dy * 0.6)), bgr, 1, cv2.LINE_AA)

    def start_intro(self) -> None:
        self._intro_t0 = time.monotonic()
        self._intro_done = False
        self._xfade_t0 = None
        with self._lock:
            self._cam = dict(VIEW_PRESETS["home"], **_INTRO_FAR)
            self._cam_s = dict(self._cam)
        self._intro.start()

    def set_view_preset(self, key: str) -> None:
        preset = VIEW_PRESETS.get(key)
        if preset:
            with self._lock:
                self._cam = dict(preset)

    def set_overlay(self, key: str, on: bool) -> None:
        if key == "wireframe":
            self._wireframe = bool(on)
        elif key == "transparent":
            self._transparent = bool(on)
        elif key == "collision":
            self._collision = bool(on)

    def apply_theme(self, pal: dict) -> None:
        self._pal = pal
        self.lbl.setStyleSheet(
            "QLabel { color: %s; background: %s; font-size: 12px; }"
            % (pal["muted"], pal["bg"]))

    # ── 开场运镜 ──────────────────────────────────────
    def _on_intro_value(self, v: float) -> None:
        t = float(v)
        home = VIEW_PRESETS["home"]
        with self._lock:
            for k in ("distance", "azimuth", "elevation"):
                far = _INTRO_FAR[k]
                self._cam[k] = far + (home[k] - far) * t
            self._cam_s = dict(self._cam)   # 运镜期间硬跟，不吃阻尼

    def _on_intro_end(self) -> None:
        if not self._intro_done:
            self._intro_done = True
            self._xfade_t0 = time.monotonic()

    def _cancel_intro(self) -> None:
        if self._intro.state() == QVariantAnimation.Running:
            self._intro.stop()
        if not self._intro_done:
            self._intro_done = True
            self._xfade_t0 = time.monotonic()
        with self._lock:
            self._cam = dict(VIEW_PRESETS["home"])

    # ── 绘制 ──────────────────────────────────────────
    def _on_poll(self) -> None:
        with self._lock:
            frame = self._frame
            wire_frame = self._wire_frame
        if frame is None:
            return
        h, w = frame.shape[:2]
        arr = np.ascontiguousarray(frame)

        # 线框 → 实体交叉淡入（运镜结束后）
        if self._xfade_t0 is not None and wire_frame is not None \
                and wire_frame.shape == arr.shape:
            a = (time.monotonic() - self._xfade_t0) / _INTRO_XFADE_S
            if a >= 1.0:
                self._xfade_t0 = None
                self._wire_frame = None
            else:
                a = a * a * (3.0 - 2.0 * a)
                arr = (wire_frame.astype(np.float32) * (1.0 - a)
                       + arr.astype(np.float32) * a).astype(np.uint8)

        # 开机渐显（从背板色化出）
        since = time.monotonic() - self._fade_t0
        if since < _INTRO_FADE_S:
            a = since / _INTRO_FADE_S
            a = a * a * (3.0 - 2.0 * a)
            pal = self._pal or {}
            bg = np.array(_hex_rgb(pal.get("intro_fade", "#161616")),
                          dtype=np.float32)
            arr = (bg[None, None, :] * (1.0 - a)
                   + arr.astype(np.float32) * a).astype(np.uint8)

        self._draw_target_marker(arr)
        self._draw_axis_gizmo(arr)

        qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
        qimg._ref = arr
        pm = QPixmap.fromImage(qimg)
        target = self.lbl.size()
        if target.width() > 4 and target.height() > 4:
            pm = pm.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl.setPixmap(pm)

    def _draw_axis_gizmo(self, rgb: np.ndarray) -> None:
        """右上角悬浮世界轴（pbvs 同款：3x 超采样、深度排序、预乘合成）。"""
        h, w = rgb.shape[:2]
        s = min(104, w // 5, h // 5)
        if s < 48:
            return
        scale = 3
        pad = 12
        x0, y0 = w - s - pad, pad
        roi = rgb[y0:y0 + s, x0:x0 + s]
        ink = np.zeros((s * scale, s * scale, 3), dtype=np.uint8)
        mask = np.zeros(ink.shape[:2], dtype=np.uint8)
        cx = cy = s / 2.0
        k = s / 104.0
        pal = self._pal or {}
        bg = np.array(_hex_rgb(pal.get("bg", "#161616")))
        light = float(bg.mean()) > 140.0
        edge = (247, 248, 250) if light else (20, 22, 26)

        def point(x, y):
            return int(round(x * scale)), int(round(y * scale))

        def mix(a, b, t):
            return tuple(int(round(x * (1.0 - t) + y * t))
                         for x, y in zip(a, b))

        def line(a, b, color, width):
            width = max(1, int(round(width * scale)))
            cv2.line(ink, point(*a), point(*b), color, width, cv2.LINE_8)
            cv2.line(mask, point(*a), point(*b), 255, width, cv2.LINE_8)

        def circle(x, y, radius, color):
            radius = max(1, int(round(radius * scale)))
            cv2.circle(ink, point(x, y), radius, color, -1, cv2.LINE_8)
            cv2.circle(mask, point(x, y), radius, 255, -1, cv2.LINE_8)

        def bead(x, y, radius, color):
            circle(x, y, radius + 0.85 * k, edge)
            circle(x, y, radius, mix(color, (12, 16, 24), 0.34))
            for i in range(1, 9):
                t = i / 9.0
                circle(x - radius * 0.27 * t, y - radius * 0.30 * t,
                       radius * (1.0 - t * 0.82),
                       mix(color, (255, 255, 255), t * 0.43))

        with self._lock:
            az = self._cam_s["azimuth"]
            el = self._cam_s["elevation"]
        limbs = []
        for z, dx, dy, deg, color in _gizmo_items(az, el):
            projection = math.sqrt(max(0.0, 1.0 - z * z))
            for sign in (-1, 1):
                limbs.append((z * sign, sign, dx * sign, dy * sign,
                              projection, deg, color))
        limbs.sort(key=lambda item: -item[0])

        def draw_limb(item):
            depth, sign, dx, dy, projection, deg, color = item
            positive = sign > 0
            near = depth < 0.0
            tint = mix(edge, color, (0.92 if near else 0.63)
                       if positive else (0.48 if near else 0.30))
            length = s * (0.33 if positive else 0.24) * projection
            ex, ey = cx + dx * length, cy + dy * length
            width = (3.0 if positive else 1.45) * k
            if not deg:
                line((cx, cy), (ex, ey), edge, width + 1.8 * k)
                line((cx, cy), (ex, ey), tint, width)
                if positive:
                    line((cx - 0.45 * k, cy - 0.55 * k),
                         (ex - 0.45 * k, ey - 0.55 * k),
                         mix(tint, (255, 255, 255), 0.27), 0.7 * k)
            bead(ex, ey, (5.2 if positive else 2.8) * k, tint)

        for item in limbs:
            if item[0] >= 0.0:
                draw_limb(item)
        bead(cx, cy, 3.6 * k, (182, 190, 202) if light else (207, 215, 226))
        for item in limbs:
            if item[0] < 0.0:
                draw_limb(item)

        alpha_hi = mask.astype(np.float32) / 255.0
        premul = ink.astype(np.float32) * alpha_hi[:, :, None]
        alpha = cv2.resize(alpha_hi, (s, s), interpolation=cv2.INTER_AREA)
        col = cv2.resize(premul, (s, s), interpolation=cv2.INTER_AREA)
        roi[:] = np.clip(np.rint(col + roi * (1.0 - alpha[:, :, None])),
                         0, 255).astype(np.uint8)

    # ── 阻尼相机 ──────────────────────────────────────
    def _on_smooth(self) -> None:
        now = time.monotonic()
        dt = min(_SMOOTH_DT_MAX_S, now - self._last_t)
        self._last_t = now
        if self._intro.state() == QVariantAnimation.Running:
            return    # 运镜期间硬跟
        with self._lock:
            tgt = dict(self._cam)
            cur = dict(self._cam_s)
        a_rot = 1.0 - math.exp(-dt / _CAM_TAU_S)
        a_zoom = 1.0 - math.exp(-dt / _CAM_TAU_ZOOM_S)
        cur["azimuth"] = cur["azimuth"] + _ang_diff(cur["azimuth"], tgt["azimuth"]) * a_rot
        cur["elevation"] += (tgt["elevation"] - cur["elevation"]) * a_rot
        cur["distance"] += (tgt["distance"] - cur["distance"]) * a_zoom
        lk_t, lk_c = np.array(tgt["lookat"]), np.array(cur["lookat"])
        cur["lookat"] = tuple(lk_c + (lk_t - lk_c) * a_rot)
        with self._lock:
            self._cam_s = cur

    # ── 鼠标交互 ──────────────────────────────────────
    def mousePressEvent(self, ev) -> None:  # noqa: N802
        self._cancel_intro()
        if ev.button() == Qt.LeftButton and self._target_world is not None:
            pr = self._project(self._target_world)
            if pr is not None:
                dx = ev.position().x() - pr[0]
                dy = ev.position().y() - pr[1]
                if (dx * dx + dy * dy) ** 0.5 < 20:
                    self._drag_marker = True
                    self._drag_base = (ev.position().x(), ev.position().y(),
                                       pr[2], tuple(self._target_world))
                    self.setCursor(Qt.ClosedHandCursor)
                    return
        self._press = (ev.button(), ev.position().x(), ev.position().y())

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        if self._drag_marker:
            self._drag_marker = False
            self.setCursor(Qt.ArrowCursor)
            if self.on_target_drag is not None and self._target_world is not None:
                self.on_target_drag(self._target_world, True)
            self._drag_base = None
            return
        self._press = None

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._drag_marker and self._drag_base is not None:
            x0, y0, depth, base = self._drag_base
            x, y = ev.position().x(), ev.position().y()
            with self._lock:
                cam = dict(self._cam_s)
            fwd, right, up = _cam_basis(cam["azimuth"], cam["elevation"])
            half_h = math.tan(math.radians(self._fovy) / 2.0) * depth
            half_w = half_h * (self.width() / max(1, self.height()))
            wx = (x - x0) / max(1, self.width()) * 2.0 * half_w
            wy = -(y - y0) / max(1, self.height()) * 2.0 * half_h
            newp = tuple(np.array(base) + right * wx + up * wy)
            self._target_world = newp
            if self.on_target_drag is not None:
                self.on_target_drag(newp, False)
            return
        if self._press is None:
            return
        btn, px, py = self._press
        x, y = ev.position().x(), ev.position().y()
        dx, dy = x - px, y - py
        self._press = (btn, x, y)
        with self._lock:
            cam = dict(self._cam)
        if btn == Qt.LeftButton:
            scale = max(_ORBIT_DEG_PX_RANGE[0],
                        min(_ORBIT_DEG_PX_RANGE[1],
                            _ORBIT_TURN_DEG / max(1, self.width())))
            cam["azimuth"] -= dx * scale
            cam["elevation"] -= dy * scale
            cam["elevation"] = max(_ELEV_CLAMP[0], min(_ELEV_CLAMP[1], cam["elevation"]))
        elif btn == Qt.RightButton:
            lk = np.array(cam["lookat"], dtype=float)
            az = math.radians(cam["azimuth"])
            right = np.array([math.sin(az), -math.cos(az), 0.0])
            up = np.array([0.0, 0.0, 1.0])
            k = cam["distance"] * _PAN_SENS
            lk = lk + (-right * dx + up * dy) * k
            cam["lookat"] = tuple(lk)
        with self._lock:
            self._cam = cam

    def wheelEvent(self, ev) -> None:  # noqa: N802
        with self._lock:
            cam = dict(self._cam)
        cam["distance"] = max(_DIST_CLAMP[0], min(
            _DIST_CLAMP[1], cam["distance"] * math.exp(-ev.angleDelta().y() * 0.0012)))
        with self._lock:
            self._cam = cam

    def mouseDoubleClickEvent(self, ev) -> None:  # noqa: N802
        self.set_view_preset("home")
