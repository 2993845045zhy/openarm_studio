"""三面板通用小件：分组卡片、行、统计瓦片、细分隔线。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy,
                               QVBoxLayout, QWidget)


class Hairline(QFrame):
    """zcode SettingsRow 的行间 1px 发丝线。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rowRule")
        self.setFixedHeight(1)
        self.setAttribute(Qt.WA_StyledBackground, True)


class GroupCard(QFrame):
    """Inspector 里的分组卡片（zcode groupCard：面板色 + 发丝边 + 10px 圆角）。"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("groupCard")
        self.setAttribute(Qt.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(8)

        self.lbl_title = QLabel(title, self)
        self.lbl_title.setObjectName("groupTitle")
        outer.addWidget(self.lbl_title)

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        # 卡内行距与卡片外边距（outer.spacing=8）保持同一节奏
        self.body.setSpacing(8)
        outer.addLayout(self.body)

    def add(self, widget: QWidget) -> None:
        self.body.addWidget(widget)


class Row(QWidget):
    """zcode SettingsRow 的单行键值：左键（ink2/500）+ 右值（等宽/ink）。"""

    def __init__(self, label: str, widget: QWidget | None = None,
                 parent=None, value: bool = False):
        super().__init__(parent)
        self.setObjectName("settingRowHost")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 5, 2, 5)
        lay.setSpacing(8)
        self.lbl = QLabel(label, self)
        self.lbl.setObjectName("settingTitle")
        lay.addWidget(self.lbl)
        lay.addStretch(1)
        if widget is not None:
            lay.addWidget(widget, 0, Qt.AlignVCenter)
            self.val: QLabel | None = None
        else:
            self.val = QLabel("—", self)
            self.val.setObjectName("fieldValue")
            self.val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lay.addWidget(self.val)

    def set_value(self, text: str) -> None:
        if getattr(self, "val", None) is not None:
            self.val.setText(text)
        else:
            self.lbl.setText(text)


class StatTile(QFrame):
    """KPI 瓦片（zcode：透明 + 排版分组，kpiTitle/kpiValue）。"""

    def __init__(self, title: str, value: str = "--", parent=None):
        super().__init__(parent)
        self.setObjectName("statGroup")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        self.lbl_title = QLabel(title, self)
        self.lbl_title.setObjectName("kpiTitle")
        self.lbl_value = QLabel(value, self)
        self.lbl_value.setObjectName("kpiValue")
        self.lbl_value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        lay.addWidget(self.lbl_title)
        lay.addWidget(self.lbl_value)

    def set_value(self, text: str) -> None:
        self.lbl_value.setText(text)


class SectionTitle(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("sectionHead")
