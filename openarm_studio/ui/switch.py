"""分段控件（zcode ModeSwitch 同款）：200ms OutCubic 滑块平移。

API 对齐 QComboBox 子集：setCurrentIndex / currentIndex /
currentIndexChanged / setEnabled / button(i)。
"""

from __future__ import annotations

from PySide6.QtCore import (QEasingCurve, QPropertyAnimation, QRect, Qt,
                            Signal)
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton

_THUMB_MS = 200


class ModeSwitch(QFrame):
    currentIndexChanged = Signal(int)

    def __init__(self, labels: list[str], parent=None, height: int = 30):
        super().__init__(parent)
        self.setObjectName("segmented")
        self.setFixedHeight(height)
        self._labels = list(labels)
        self._index = 0
        self._btns: list[QPushButton] = []

        self._thumb = QFrame(self)
        self._thumb.setObjectName("segThumb")
        self._thumb.setGeometry(0, 0, 0, 0)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(0)
        for i, text in enumerate(self._labels):
            b = QPushButton(text, self)
            b.setObjectName("segBtn")
            b.setCheckable(True)
            b.setChecked(i == 0)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, idx=i: self.setCurrentIndex(idx))
            lay.addWidget(b, 1)
            self._btns.append(b)

        self._anim = QPropertyAnimation(self._thumb, b"geometry", self)
        self._anim.setDuration(_THUMB_MS)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._thumb.lower()

    # ── API ────────────────────────────────────────────
    def currentIndex(self) -> int:
        return self._index

    def setCurrentIndex(self, idx: int, animate: bool = True) -> None:
        if not 0 <= idx < len(self._btns):
            return
        changed = idx != self._index
        self._index = idx
        for i, b in enumerate(self._btns):
            b.setChecked(i == idx)
        self._move_thumb(animate)
        if changed:
            self.currentIndexChanged.emit(idx)

    def button(self, i: int) -> QPushButton:
        return self._btns[i]

    def labels(self) -> list[str]:
        return list(self._labels)

    def setEnabled(self, on: bool) -> None:  # noqa: N802 (Qt naming)
        super().setEnabled(on)
        for b in self._btns:
            b.setEnabled(on)

    # ── layout ─────────────────────────────────────────
    def resizeEvent(self, ev) -> None:  # noqa: N802
        super().resizeEvent(ev)
        self._move_thumb(False)

    def showEvent(self, ev) -> None:  # noqa: N802
        super().showEvent(ev)
        self._move_thumb(False)

    def _thumb_rect(self) -> QRect:
        n = len(self._btns)
        if n == 0:
            return QRect(0, 0, 0, 0)
        w = self.width() - 4
        cw = w // n
        x = 2 + cw * self._index
        return QRect(x, 2, cw, self.height() - 4)

    def _move_thumb(self, animate: bool) -> None:
        target = self._thumb_rect()
        if not animate:
            self._anim.stop()
            self._thumb.setGeometry(target)
            return
        if self._thumb.geometry() == target:
            return
        self._anim.stop()
        self._anim.setStartValue(self._thumb.geometry())
        self._anim.setEndValue(target)
        self._anim.start()
