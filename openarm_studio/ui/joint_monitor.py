"""3D 视口左上角的「关节」胶囊：点开原地展开成角度条。

数据来自外部 provider（返回 关节名→弧度 的字典），30Hz 刷新。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)

MAX_RAD = 3.5  # 条满格对应的角度（±3.5rad 足够覆盖所有关节量程）


class _JointBar(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("jointRow")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.lbl = QLabel(label, self)
        self.lbl.setObjectName("jointName")
        self.lbl.setFixedWidth(46)
        self.val = QLabel("--", self)
        self.val.setObjectName("jointVal")
        self.val.setFixedWidth(62)
        self.val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.bar = QFrame(self)
        self.bar.setFixedHeight(3)
        self._fill = QFrame(self.bar)
        self._fill.setFixedHeight(3)
        lay.addWidget(self.lbl)
        lay.addWidget(self.bar, 1)
        lay.addWidget(self.val)

    def set_value(self, rad: float | None, pal: dict) -> None:
        if rad is None:
            self.val.setText("--")
            self._fill.setFixedWidth(0)
            return
        self.val.setText(f"{rad:+.3f}")
        frac = min(1.0, abs(rad) / MAX_RAD)
        self._fill.setFixedWidth(int(self.bar.width() * frac))
        self._fill.setStyleSheet(
            "background: %s; border-radius: 1px;" % (pal["ink"] if abs(rad) > 1e-6 else pal["faint"]))

    def resizeEvent(self, ev) -> None:  # noqa: N802
        super().resizeEvent(ev)
        self._fill.setMove(0, 0)
        self.bar.setStyleSheet("background: %s; border-radius: 1px;"
                               % getattr(self, "_track", "#333333"))


class JointOverlay(QFrame):
    """胶囊 + 展开面板。"""

    def __init__(self, provider, pal: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("jointCapsule")
        self._provider = provider
        self._pal = pal
        self._open = False
        self._bars: dict[str, _JointBar] = {}

        self.toggle = QPushButton("关节", self)
        self.toggle.setObjectName("ghost")
        self.toggle.setFixedHeight(26)
        self.toggle.setCursor(Qt.PointingHandCursor)
        self.toggle.clicked.connect(self._on_toggle)

        self.body = QWidget(self)
        self.body.setVisible(False)
        grid = QGridLayout(self.body)
        grid.setContentsMargins(10, 6, 10, 10)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(3)

        for side, prefix, col in (("左臂", "openarm_left", 0), ("右臂", "openarm_right", 1)):
            head = QLabel(side, self.body)
            head.setObjectName("jointName")
            grid.addWidget(head, 0, col)
            for i in range(1, 8):
                name = f"{prefix}_joint{i}"
                bar = _JointBar(f"J{i}", self.body)
                bar._track = pal["panel_alt"]
                bar.bar.setStyleSheet("background: %s; border-radius: 1px;" % pal["panel_alt"])
                grid.addWidget(bar, i, col)
                self._bars[name] = bar
            gname = f"{prefix}_finger_joint1"
            gbar = _JointBar("夹爪", self.body)
            gbar._track = pal["panel_alt"]
            gbar.bar.setStyleSheet("background: %s; border-radius: 1px;" % pal["panel_alt"])
            grid.addWidget(gbar, 8, col)
            self._bars[gname] = gbar

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 5, 6, 6)
        outer.setSpacing(2)
        head_row = QHBoxLayout()
        head_row.setContentsMargins(0, 0, 0, 0)
        head_row.addWidget(self.toggle)
        head_row.addStretch(1)
        outer.addLayout(head_row)
        outer.addWidget(self.body)

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def apply_palette(self, pal: dict) -> None:
        self._pal = pal
        for bar in self._bars.values():
            bar._track = pal["panel_alt"]
            bar.bar.setStyleSheet("background: %s; border-radius: 1px;" % pal["panel_alt"])

    def _on_toggle(self) -> None:
        self._open = not self._open
        self.body.setVisible(self._open)
        self.toggle.setText("关节 ▾" if self._open else "关节")
        self.adjustSize()
        self.resize(self.sizeHint())

    def _refresh(self) -> None:
        if not self._open:
            return
        try:
            data = self._provider() or {}
        except Exception:  # noqa: BLE001
            return
        for name, bar in self._bars.items():
            bar.set_value(data.get(name), self._pal)
