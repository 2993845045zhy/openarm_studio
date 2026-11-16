"""内嵌 3D 面板：只负责显示与相机交互，渲染在外部线程完成。

线程契约（照 pbvs_workbench）：
- 渲染线程调 `push_frame(rgb)` 送帧（加锁换指针）
- 渲染线程每帧读 `cam_state()` 拿相机（已阻尼）
- GUI 线程用 16ms 定时器取最新帧画到 QLabel
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget

from ..core.mjview import DEFAULT_VIEW

_POLL_MS = 16
_SMOOTH_MS = 8
_CAM_TAU_S = 0.10
_CAM_TAU_ZOOM_S = 0.15

VIEW_PRESETS = {
    "home": dict(DEFAULT_VIEW),
    "front": {"lookat": (0.0, 0.0, 0.32), "distance": 1.75, "azimuth": 180.0, "elevation": -8.0},
    "side": {"lookat": (0.0, 0.0, 0.32), "distance": 1.75, "azimuth": 90.0, "elevation": -8.0},
    "top": {"lookat": (0.0, 0.0, 0.25), "distance": 1.9, "azimuth": 135.0, "elevation": -70.0},
}


class MujocoViewPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mjPanel")
        self.setMinimumSize(320, 220)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.NoContextMenu)

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_count = 0
        self._last_draw = 0.0

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
        self._press: tuple[Qt.MouseButton, float, float] | None = None

        self._poll = QTimer(self)
        self._poll.setInterval(_POLL_MS)
        self._poll.timeout.connect(self._on_poll)
        self._poll.start()

        self._smooth = QTimer(self)
        self._smooth.setInterval(_SMOOTH_MS)
        self._smooth.timeout.connect(self._on_smooth)
        self._smooth.start()

        self._theme_bg = "#161616"

    # ── 渲染线程接口 ───────────────────────────────────
    def push_frame(self, rgb: np.ndarray) -> None:
        with self._lock:
            self._frame = rgb
            self._frame_count += 1

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def cam_state(self) -> dict:
        with self._lock:
            st = dict(self._cam_s)
        st["aspect"] = max(0.4, self.width() / max(1, self.height()))
        st["wireframe"] = self._wireframe
        st["transparent"] = self._transparent
        st["collision"] = self._collision
        return st

    # ── 对外操作 ──────────────────────────────────────
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
        # 底板颜色取卡片底色（与天空盒同色），letterbox 才不会露出深色边
        self._theme_bg = pal["bg"]
        self.lbl.setStyleSheet(
            "QLabel { color: %s; background: %s; font-size: %spx; }"
            % (pal["muted"], self._theme_bg, 12))

    # ── 绘制 ──────────────────────────────────────────
    def _on_poll(self) -> None:
        with self._lock:
            frame = self._frame
        if frame is None:
            return
        h, w = frame.shape[:2]
        arr = np.ascontiguousarray(frame)
        qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
        qimg._ref = arr  # 保住 numpy 缓冲
        pm = QPixmap.fromImage(qimg)
        target = self.lbl.size()
        if target.width() > 4 and target.height() > 4:
            pm = pm.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl.setPixmap(pm)

    def _on_smooth(self) -> None:
        with self._lock:
            tgt = dict(self._cam)
            cur = dict(self._cam_s)
        alpha = 1.0 - math.exp(-_SMOOTH_MS / 1000.0 / _CAM_TAU_S)
        zoom_alpha = 1.0 - math.exp(-_SMOOTH_MS / 1000.0 / _CAM_TAU_ZOOM_S)
        for k in ("azimuth", "elevation"):
            cur[k] += (tgt[k] - cur[k]) * alpha
        cur["distance"] += (tgt["distance"] - cur["distance"]) * zoom_alpha
        lk_t, lk_c = np.array(tgt["lookat"]), np.array(cur["lookat"])
        cur["lookat"] = tuple(lk_c + (lk_t - lk_c) * alpha)
        with self._lock:
            self._cam_s = cur

    # ── 鼠标交互（方向与官方 mjv_moveCamera 同号）────────
    def mousePressEvent(self, ev) -> None:  # noqa: N802
        self._press = (ev.button(), ev.position().x(), ev.position().y())

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        self._press = None

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._press is None:
            return
        btn, px, py = self._press
        x, y = ev.position().x(), ev.position().y()
        dx, dy = x - px, y - py
        self._press = (btn, x, y)
        with self._lock:
            cam = dict(self._cam)
        if btn == Qt.LeftButton:
            scale = max(0.15, min(0.55, 300.0 / max(1, self.width())))
            cam["azimuth"] -= dx * scale
            cam["elevation"] -= dy * scale
            cam["elevation"] = max(-89.0, min(89.0, cam["elevation"]))
        elif btn == Qt.RightButton:
            lk = np.array(cam["lookat"], dtype=float)
            az = math.radians(cam["azimuth"])
            right = np.array([math.cos(az), math.sin(az), 0.0])
            up = np.array([0.0, 0.0, 1.0])
            k = cam["distance"] * 0.0018
            lk = lk + (-right * dx + up * dy) * k
            cam["lookat"] = tuple(lk)
        with self._lock:
            self._cam = cam

    def wheelEvent(self, ev) -> None:  # noqa: N802
        with self._lock:
            cam = dict(self._cam)
        cam["distance"] = max(0.35, min(8.0, cam["distance"] * math.exp(-ev.angleDelta().y() * 0.0012)))
        with self._lock:
            self._cam = cam

    def mouseDoubleClickEvent(self, ev) -> None:  # noqa: N802
        self.set_view_preset("home")
