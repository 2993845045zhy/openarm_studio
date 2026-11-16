"""无边框窗口chrome：拖拽、最小化/最大化/关闭、边缘缩放。

缩放热区是 8 条贴边的窄条（每条 _EDGE 像素宽），不是覆盖层——
中心区域没有任何覆盖物，鼠标事件直达下层控件。
"""

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
        b.setObjectName("winClose" if kind == "close" else "winBtn")
        b.setFixedSize(26, 26)
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


class DragHandle(QWidget):
    """标题区拖动：优先系统 move，失败则手动 move。"""

    def __init__(self, win: QWidget, parent=None):
        super().__init__(parent)
        self._win = win
        self._press: QPoint | None = None

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.LeftButton:
            wh = self._win.windowHandle()
            if wh is not None:
                wh.startSystemMove()
                self._press = None
                return
            self._press = ev.globalPosition().toPoint()

    def mouseDoubleClickEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.LeftButton:
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


class _EdgeBar(QWidget):
    """单条缩放热区。press 后 grabMouse，全程自己消化事件。"""

    def __init__(self, win: QWidget, mode: str):
        super().__init__(win)
        self._win = win
        self._mode = mode
        self._press = QPoint()
        self.setCursor({
            "l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
            "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
            "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
            "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
        }[mode])
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.LeftButton:
            self._press = ev.globalPosition().toPoint()
            self.grabMouse()

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        self.releaseMouse()
        self._press = QPoint()

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if not (ev.buttons() & Qt.LeftButton) or self._press.isNull():
            return
        cur = ev.globalPosition().toPoint()
        delta = cur - self._press
        self._press = cur
        g = self._win.geometry()
        x, y, w, h = g.x(), g.y(), g.width(), g.height()
        min_w, min_h = 1400, 800
        if "l" in self._mode:
            x += delta.x(); w -= delta.x()
        if "r" in self._mode:
            w += delta.x()
        if "t" in self._mode:
            y += delta.y(); h -= delta.y()
        if "b" in self._mode:
            h += delta.y()
        if w >= min_w and h >= min_h:
            self._win.setGeometry(x, y, w, h)
        else:
            # 保持合法尺寸：夹住变小的方向
            if w < min_w:
                if "l" in self._mode:
                    x -= min_w - w
                w = min_w
            if h < min_h:
                if "t" in self._mode:
                    y -= min_h - h
                h = min_h
            self._win.setGeometry(x, y, w, h)


class ResizeHandles(QWidget):
    """8 条贴边缩放热区的容器（透明、不覆盖中心）。"""

    def __init__(self, win: QWidget):
        super().__init__(win)
        self._win = win
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._bars: list[_EdgeBar] = []
        for mode in ("l", "r", "t", "b", "tl", "tr", "bl", "br"):
            bar = _EdgeBar(win, mode)
            self._bars.append(bar)
        win.installEventFilter(self)
        self._layout()

    def eventFilter(self, obj, ev) -> bool:  # noqa: N802
        if obj is self._win and ev.type() in (QEvent.Resize, QEvent.Show):
            self._layout()
        return False

    def _layout(self) -> None:
        w, h = self._win.width(), self._win.height()
        e = _EDGE
        geo = {
            "l": (0, e, e, h - 2 * e),
            "r": (w - e, e, e, h - 2 * e),
            "t": (e, 0, w - 2 * e, e),
            "b": (e, h - e, w - 2 * e, e),
            "tl": (0, 0, e, e),
            "tr": (w - e, 0, e, e),
            "bl": (0, h - e, e, e),
            "br": (w - e, h - e, e, e),
        }
        for bar in self._bars:
            bar.setGeometry(*geo[bar._mode])
            bar.raise_()
