"""zcode 组件库（结构与规格逐项移植自 pbvs_workbench/ui）。

包含：导航行、可折叠分组头（chevron 150ms 旋转）、命令面板行、
最近运行行、zcode Switch（32×18 胶囊）、键帽 Kbd、状态点、
滑动下划线页签、logo 折叠按钮、工具条幽灵按钮。
"""

from __future__ import annotations

import math

from PySide6.QtCore import (QEasingCurve, QPointF, QRectF, QSize, Qt,
                            QVariantAnimation, Signal)
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QTabBar, QVBoxLayout, QWidget)

from .icons import ICONS, pixmap

NAV_H = 32


# ── 导航 ────────────────────────────────────────────────────
class NavItem(QPushButton):
    """导航行：图标 + 标签。选中 = nav_selected 底。"""

    def __init__(self, label: str, icon_name: str, page_index: int, parent=None):
        super().__init__(parent)
        self.page_index = page_index
        self._label = label
        self._icon_name = icon_name
        self.setObjectName("navItem")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(NAV_H)
        self.setText(f"  {label}")
        self.setIconSize(QSize(16, 16))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_icon_color(self, color: str) -> None:
        self.setIcon(ICONS.get(self._icon_name, color, 16))


class _Chevron(QWidget):
    """分组头右端箭头：展开 90° / 折叠 0°，随 hover 提亮。"""

    def __init__(self, pal: dict, parent=None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._angle = 90.0
        self._base = pal.get("muted", "#9a9a9a")
        self._hot = pal.get("ink2", "#666666")
        self._hot_on = False
        self._pix = pixmap("chevron-right", self._base, 12)

    def set_palette_colors(self, pal: dict) -> None:
        self._base = pal.get("muted", "#9a9a9a")
        self._hot = pal.get("ink2", "#666666")
        self._pix = pixmap("chevron-right", self.current_color(), 12)
        self.update()

    def current_color(self) -> str:
        return self._hot if self._hot_on else self._base

    def set_hot(self, on: bool) -> None:
        if on != self._hot_on:
            self._hot_on = on
            self._pix = pixmap("chevron-right", self.current_color(), 12)
            self.update()

    def set_angle(self, deg: float) -> None:
        self._angle = float(deg)
        self.update()

    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.translate(self.width() / 2.0, self.height() / 2.0)
        p.rotate(self._angle)
        p.drawPixmap(-6, -6, self._pix)


class NavGroupHeader(QPushButton):
    """节标题语义的分组头：小字、无底色、整行可点折叠、箭头旋转。"""

    def __init__(self, label: str, pal: dict, parent=None):
        super().__init__(label, parent)
        self.setObjectName("navGroupHeader")
        self.setCheckable(True)
        self.setChecked(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(26)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._chev = _Chevron(pal, self)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._chev.set_angle)
        self._sync(False)

    def _sync(self, animate: bool = True) -> None:
        target = 90.0 if self.isChecked() else 0.0
        if not animate:
            self._anim.stop()
            self._chev.set_angle(target)
            return
        self._anim.stop()
        self._anim.setStartValue(self._chev._angle)  # noqa: SLF001
        self._anim.setEndValue(target)
        self._anim.start()

    def resizeEvent(self, ev) -> None:  # noqa: N802
        super().resizeEvent(ev)
        self._chev.move(self.width() - 6 - 12, (self.height() - 12) // 2)

    def set_palette_colors(self, pal: dict) -> None:
        self._chev.set_palette_colors(pal)

    def enterEvent(self, ev) -> None:  # noqa: N802
        self._chev.set_hot(True)
        super().enterEvent(ev)

    def leaveEvent(self, ev) -> None:  # noqa: N802
        self._chev.set_hot(False)
        super().leaveEvent(ev)

    def nextCheckState(self) -> None:  # noqa: N802
        super().nextCheckState()
        self._sync(True)


class Kbd(QLabel):
    """键帽 chip。"""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("kbd")
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(20)


class NavCommandRow(QPushButton):
    """底部「命令面板」行：左图标 + 文字，右端 ⌘K 键帽。"""

    def __init__(self, pal: dict, parent=None):
        super().__init__("命令面板", parent)
        self.setObjectName("navCommand")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(30)
        self.setIconSize(QSize(15, 15))
        self.setIcon(ICONS.get("target", pal.get("ink2", "#8a8a8a"), 15))
        self.kbd = Kbd("⌘K", self)

    def resizeEvent(self, ev) -> None:  # noqa: N802
        super().resizeEvent(ev)
        w = max(20, self.kbd.sizeHint().width())
        self.kbd.setFixedWidth(w)
        self.kbd.move(self.width() - 8 - w, (self.height() - 20) // 2)

    def set_palette_colors(self, pal: dict) -> None:
        self.setIcon(ICONS.get("target", pal.get("ink2", "#8a8a8a"), 15))
        self.update()


class HistoryRow(QWidget):
    """最近运行行：标题（elide）+ 右端相对时间。"""

    clicked = Signal()

    def __init__(self, title: str, rel: str, parent=None):
        super().__init__(parent)
        self.setObjectName("historyRow")
        self.setFixedHeight(28)
        self.setCursor(Qt.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 6, 0)
        lay.setSpacing(8)
        self.title = QLabel(title)
        self.title.setObjectName("historyTitle")
        self.title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(self.title, 1)
        self.time = QLabel(rel)
        self.time.setObjectName("historyTime")
        self.time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.time)

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.LeftButton:
            self.clicked.emit()


class LogoButton(QWidget):
    """左上角 logo：点击折叠导航轨。"""

    clicked = Signal()

    def __init__(self, pal: dict, parent=None):
        super().__init__(parent)
        self.setFixedSize(28, 28)
        self.setCursor(Qt.PointingHandCursor)
        self._pal = pal

    def set_palette_colors(self, pal: dict) -> None:
        self._pal = pal
        self.update()

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.LeftButton:
            self.clicked.emit()

    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pal = self._pal
        rect = QRectF(1.5, 1.5, self.width() - 3, self.height() - 3)
        # 断缝环（Logo 语义：闭环唯一不肯闭合的那道缝）
        pen = QPen(QColor(pal.get("ink", "#d4d4d4")))
        pen.setWidthF(2.4)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, int(72 * 16), int((360 - 72) * 16))
        p.setBrush(QColor(pal.get("accent", "#4099ff")))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(rect.center().x(), rect.top() + 1.8), 1.9, 1.9)


# ── zcode Switch（32×18 胶囊 + 140ms 滑块）──────────────────
class Switch(QWidget):
    toggled = Signal(bool)

    def __init__(self, pal: dict, checked: bool = False, parent=None):
        super().__init__(parent)
        self._pal = pal
        self._on = checked
        self._pos = 1.0 if checked else 0.0
        self.setFixedSize(32, 18)
        self.setCursor(Qt.PointingHandCursor)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._set_pos)

    def isChecked(self) -> bool:
        return self._on

    def setChecked(self, on: bool) -> None:
        if on == self._on:
            return
        self._on = bool(on)
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()
        self.toggled.emit(self._on)

    def _set_pos(self, v) -> None:
        self._pos = float(v)
        self.update()

    def set_palette_colors(self, pal: dict) -> None:
        self._pal = pal
        self.update()

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.LeftButton and self.isEnabled():
            self.setChecked(not self._on)

    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pal = self._pal
        on = self._pos
        off_c = QColor(pal.get("panel_alt", "#202020"))
        on_c = QColor(pal.get("primary", "#ffffff"))
        track = QColor(
            int(off_c.red() + (on_c.red() - off_c.red()) * on),
            int(off_c.green() + (on_c.green() - off_c.green()) * on),
            int(off_c.blue() + (on_c.blue() - off_c.blue()) * on),
        )
        p.setPen(QPen(QColor(pal.get("hairline_strong", "#33FFFFFF")), 1))
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0.5, 0.5, 31, 17), 9, 9)
        knob_x = 2.0 + on * 14.0
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(pal.get("panel", "#2b2b2b")) if on < 0.5
                   else QColor(pal.get("primary_fg", "#0d0d0d")))
        p.drawEllipse(QPointF(knob_x + 7.0, 9.0), 6.0, 6.0)


# ── 状态点（自绘圆点 + 呼吸）────────────────────────────────
class StatusDot(QWidget):
    def __init__(self, pal: dict, text: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("statusDot")
        self._pal = pal
        self._text = text
        self._color = pal.get("muted", "#606060")
        self.setFixedHeight(20)
        self._t = 0.0
        self._timer = None

    def set_state(self, color: str, text: str, breathe: bool = False) -> None:
        from PySide6.QtCore import QTimer

        self._color = color
        self._text = text
        if breathe and self._timer is None:
            self._timer = QTimer(self)
            self._timer.setInterval(50)
            self._timer.timeout.connect(self._tick)
            self._timer.start()
        elif not breathe and self._timer is not None:
            self._timer.stop()
            self._timer = None
            self._t = 0.0
        self.update()

    def _tick(self) -> None:
        self._t = (self._t + 0.05) % 2.0
        self.update()

    def set_palette_colors(self, pal: dict) -> None:
        self._pal = pal
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(12 + 5 + len(self._text) * 9 + 6, 20)

    def paintEvent(self, _ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        a = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(self._t * math.pi)) if self._timer else 1.0
        c = QColor(self._color)
        c.setAlphaF(a)
        p.setBrush(c)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(4.5, 10.0), 3.0, 3.0)
        p.setPen(QColor(self._pal.get("ink2", "#8a8a8a")))
        f = p.font()
        f.setPixelSize(11)
        p.setFont(f)
        p.drawText(QRectF(12, 0, self.width() - 12, 20),
                   Qt.AlignLeft | Qt.AlignVCenter, self._text)


# ── 滑动下划线页签 ─────────────────────────────────────────
class AnimatedTabBar(QTabBar):
    """选中指示：2px 滑动下划线（180ms OutCubic）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._x = 0.0
        self._w = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self.currentChanged.connect(self._glide)
        self._pal: dict = {}

    def set_palette_colors(self, pal: dict) -> None:
        self._pal = pal
        self.update()

    def _on_anim(self, v) -> None:
        self._x, self._w = v[0], v[1]
        self.update()

    def _target(self) -> tuple[float, float]:
        i = self.currentIndex()
        if i < 0:
            return (0.0, 0.0)
        r = self.tabRect(i)
        return (float(r.x()), float(r.width()))

    def _glide(self, _i: int) -> None:
        self._anim.stop()
        self._anim.setStartValue((self._x, self._w))
        self._anim.setEndValue(self._target())
        self._anim.start()

    def showEvent(self, ev) -> None:  # noqa: N802
        super().showEvent(ev)
        self._x, self._w = self._target()
        self.update()

    def paintEvent(self, ev) -> None:  # noqa: N802
        super().paintEvent(ev)
        if self._w <= 0:
            return
        p = QPainter(self)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self._pal.get("ink", "#d4d4d4")))
        y = self.height() - 2
        p.drawRoundedRect(QRectF(self._x + 8, y, max(6.0, self._w - 16), 2), 1, 1)


class ToolChip(QPushButton):
    """3D 工具条的幽灵小胶囊。"""

    def __init__(self, text: str, parent=None, checkable: bool = False):
        super().__init__(text, parent)
        self.setObjectName("chip")
        self.setCheckable(checkable)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(22)


class GroupRule(QFrame):
    """设置行之间的 1px 发丝线。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rowRule")
        self.setFixedHeight(1)
        self.setAttribute(Qt.WA_StyledBackground, True)


class SettingsRow(QWidget):
    """zcode SettingsRow：左标题(+说明) 右控件，行高自适应。"""

    def __init__(self, title: str, widget: QWidget | None = None,
                 desc: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("settingRowHost")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 6)
        lay.setSpacing(10)
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(1)
        self.lbl_title = QLabel(title, self)
        self.lbl_title.setObjectName("settingTitle")
        col.addWidget(self.lbl_title)
        self.lbl_desc = QLabel(desc, self)
        self.lbl_desc.setObjectName("settingDesc")
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setVisible(bool(desc))
        col.addWidget(self.lbl_desc)
        lay.addLayout(col, 1)
        self.control = widget
        if widget is not None:
            lay.addWidget(widget, 0, Qt.AlignVCenter)
