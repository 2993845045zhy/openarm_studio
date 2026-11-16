"""无边框窗口chrome：拖拽、最小化/最大化/关闭、边缘缩放热区。"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from .icons import ICONS

_EDGE = 6


class WindowButtons(QWidget):
    """右上角窗控（zcode 同款：26x26 圆角方钮，关闭钮悬停转红）。"""

    def __init__(self, win: QWidget, pal: dict, parent=None):
        super().__init__(parent)
        self._win = win
        self._pal = pal
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        self.btn_min = self._mk("min")
        self.btn_max = self._mk("max")
        self.btn_close = self._mk("close")
        for b in (self.btn_min, self.btn_max, self.btn_close):
            lay.addWidget(b)

        self.btn_min.clicked.connect(self._win.showMinimized)
        self.btn_max.clicked.connect(self._toggle_max)
        self.btn_close.clicked.connect(self._win.close)
        self.apply_palette(pal)

    def _mk(self, kind: str) -> QPushButton:
        b = QPushButton(self)
        b.setObjectName("tool")
        b.setFixedSize(26, 26)
        b.setProperty("kind", kind)
        b.setCursor(Qt.PointingHandCursor)
        return b

    def _toggle_max(self) -> None:
        if self._win.isMaximized():
            self._win.showNormal()
        else:
            self._win.showMaximized()

    def apply_palette(self, pal: dict) -> None:
        self._pal = pal
        ICONS.clear()
        self.btn_min.setIcon(ICONS.get("chevron-down", pal["ink2"], 14))
        self.btn_max.setIcon(ICONS.get("maximize", pal["ink2"], 13))
        self.btn_close.setIcon(ICONS.get("close", pal["ink2"], 14))
        self.setStyleSheet(
            "QPushButton#tool:hover { background: %s; border-radius: 6px; }" % pal["hover_bg"]
        )


class DragHandle(QWidget):
    """标题区拖动：优先系统 move，失败则手动 move。"""

    def __init__(self, win: QWidget, parent=None):
        super().__init__(parent)
        self._win = win
        self._press: QPoint | None = None
        self.setAttribute(Qt.WA_StyledBackground, False)

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.LeftButton:
            self._press = ev.globalPosition().toPoint()

    def mouseDoubleClickEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.LeftButton:
            wh = self._win.windowHandle()
            if wh is not None:
                wh.startSystemMove()
                return
            if self._win.isMaximized():
                self._win.showNormal()
            else:
                self._win.showMaximized()

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._press is None or not (ev.buttons() & Qt.LeftButton):
            return
        if self._win.isMaximized():
            return
        cur = ev.globalPosition().toPoint()
        delta = cur - self._press
        self._win.move(self._win.pos() + delta)
        self._press = cur

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        self._press = None


class ResizeHandles(QWidget):
    """窗口边缘/角落缩放热区（透明，压在子控件之上）。"""

    def __init__(self, win: QWidget):
        super().__init__(win)
        self._win = win
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)
        self._mode = None
        self._press = QPoint()
        self._geo = None
        win.installEventFilter(self)

    def eventFilter(self, obj, ev) -> bool:  # noqa: N802
        if obj is self._win and ev.type() == QEvent.Resize:
            self.setGeometry(0, 0, self._win.width(), self._win.height())
            self.raise_()
        return False

    def _hit(self, pos: QPoint):
        w, h = self._win.width(), self._win.height()
        x, y = pos.x(), pos.y()
        left, right = x <= _EDGE, x >= w - _EDGE
        top, bottom = y <= _EDGE, y >= h - _EDGE
        if top and left:
            return "tl"
        if top and right:
            return "tr"
        if bottom and left:
            return "bl"
        if bottom and right:
            return "br"
        if left:
            return "l"
        if right:
            return "r"
        if top:
            return "t"
        if bottom:
            return "b"
        return None

    _CURSORS = {
        "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
        "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
        "l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
        "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
    }

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._mode and self._geo is not None and (ev.buttons() & Qt.LeftButton):
            g = self._win.geometry()
            cur = ev.globalPosition().toPoint()
            delta = cur - self._press
            x, y, w, h = g.x(), g.y(), g.width(), g.height()
            min_w, min_h = 1400, 800
            if "l" in self._mode:
                x += delta.x()
                w -= delta.x()
            if "r" in self._mode:
                w += delta.x()
            if "t" in self._mode:
                y += delta.y()
                h -= delta.y()
            if "b" in self._mode:
                h += delta.y()
            if w >= min_w and h >= min_h:
                self._win.setGeometry(x, y, w, h)
            self._press = cur
            return
        hit = self._hit(ev.position().toPoint())
        self.setCursor(self._CURSORS.get(hit, Qt.ArrowCursor))

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() != Qt.LeftButton:
            return
        self._mode = self._hit(ev.position().toPoint())
        self._press = ev.globalPosition().toPoint()

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        self._mode = None
