"""三面板通用小件：分组卡片、行、统计瓦片、细分隔线。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy,
                               QVBoxLayout, QWidget)


class Hairline(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("hairline")
        self.setFixedHeight(1)
        # QFrame 默认不画 QSS 背景，必须显式打开，否则整条线是透明的
        self.setAttribute(Qt.WA_StyledBackground, True)


class GroupCard(QFrame):
    """Inspector 里的分组卡片：小标题 + 内容，行间无分隔线。"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("inspCard")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(8)

        self.lbl_title = QLabel(title, self)
        self.lbl_title.setObjectName("groupLabel")
        outer.addWidget(self.lbl_title)

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(7)
        outer.addLayout(self.body)

    def add(self, widget: QWidget) -> None:
        self.body.addWidget(widget)


class Row(QWidget):
    """一行：左标签 + 右侧控件。"""

    def __init__(self, label: str, widget: QWidget | None = None,
                 parent=None, value: bool = False):
        super().__init__(parent)
        self.setObjectName("inspRow")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.lbl = QLabel(label, self)
        self.lbl.setObjectName("rowValue" if value else "rowLabel")
        lay.addWidget(self.lbl)
        lay.addStretch(1)
        if widget is not None:
            lay.addWidget(widget)

    def set_value(self, text: str) -> None:
        self.lbl.setText(text)


class StatTile(QFrame):
    """圆角统计瓦片：小标题 + 数值。"""

    def __init__(self, title: str, value: str = "--", parent=None):
        super().__init__(parent)
        self.setObjectName("statTile")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(9, 6, 9, 7)
        lay.setSpacing(1)
        self.lbl_title = QLabel(title, self)
        self.lbl_title.setObjectName("jointName")
        self.lbl_value = QLabel(value, self)
        self.lbl_value.setObjectName("statValue")
        self.lbl_value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_value.setTextInteractionFlags(Qt.NoTextInteraction)
        lay.addWidget(self.lbl_title)
        lay.addWidget(self.lbl_value)

    def set_value(self, text: str) -> None:
        self.lbl_value.setText(text)


class SectionTitle(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("cardTitle")
