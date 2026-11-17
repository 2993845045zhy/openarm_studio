"""分段控件（pbvs_workbench ModeSwitch 同款）：200ms OutCubic 滑块平移。

规格对齐 pbvs 原版（2026-11-16 修复：不再 setFixedHeight——固定高度
会把 26px 的按钮塞进 24px 槽里，文字偏上、滑块错位；改为自然高度，
滑块直接贴按钮 geometry，文字由 QPushButton 自身居中）。

API 对齐 QComboBox 子集：setCurrentIndex / currentIndex /
currentIndexChanged / setEnabled / button(i)。
"""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton

_THUMB_MS = 200


class ModeSwitch(QFrame):
    currentIndexChanged = Signal(int)

    def __init__(self, labels: list[str], parent=None):
        super().__init__(parent)
        self.setObjectName("segmented")
        self._labels = list(labels)
        self._index = 0
        self._btns: list[QPushButton] = []

        # 滑块先建、藏起来；首次布局完成后贴到按钮上再显示（z 序在下）
        self._thumb = QFrame(self)
        self._thumb.setObjectName("segThumb")
        self._thumb.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._thumb.hide()

        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(1)
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
        self._place_thumb(animate)
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
        self._place_thumb(False)

    def showEvent(self, ev) -> None:  # noqa: N802
        super().showEvent(ev)
        self._place_thumb(False)

    def _place_thumb(self, animate: bool) -> None:
        """滑块贴当前按钮的真实 geometry（未布局完时先不动）。"""
        if not self._btns:
            return
        target = self._btns[self._index].geometry()
        if target.width() <= 0 or target.height() <= 0:
            return
        if not self._thumb.isVisible():
            self._anim.stop()
            self._thumb.setGeometry(target)
            self._thumb.show()
            return
        if not animate or self._thumb.geometry() == target:
            self._anim.stop()
            self._thumb.setGeometry(target)
            return
        self._anim.stop()
        self._anim.setStartValue(self._thumb.geometry())
        self._anim.setEndValue(target)
        self._anim.start()
