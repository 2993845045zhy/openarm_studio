"""内嵌 Lucide 图标（1.5 描边），与 pbvs_workbench 同款画法。

颜色在生成时烘焙进像素，所以主题切换后需要重新生成（调用方缓存需失效）。
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

# 24x24 viewBox，只收本应用用到的形状（路径取自 Lucide 1.17）
_PATHS: dict[str, str] = {
    "play": '<path d="m5 3 14 9-14 9V3z"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="1"/>',
    "estop": '<path d="M7.86 2h8.28L22 7.86v8.28L16.14 22H7.86L2 16.14V7.86L7.86 2Z"/>'
             '<path d="M12 8v5"/><path d="M12 16h.01"/>',
    "panel-left": '<rect x="3" y="3" width="18" height="18" rx="2"/>'
                  '<path d="M9 3v18"/>',
    "sun": '<circle cx="12" cy="12" r="4"/>'
           '<path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4'
           'M2 12h2M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4"/>',
    "moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    "chevron-down": '<path d="m6 9 6 6 6-6"/>',
    "chevron-right": '<path d="m9 18 6-6-6-6"/>',
    "reset": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
    "home": '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
            '<path d="M9 22V12h6v10"/>',
    "maximize": '<path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M16 3h3a2 2 0 0 1 2 2v3"/>'
                '<path d="M8 21H5a2 2 0 0 1-2-2v-3"/>'
                '<path d="M16 21h3a2 2 0 0 0 2-2v-3"/>',
    "restore": '<rect x="8" y="8" width="13" height="13" rx="2"/>'
               '<path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
    "close": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "robot": '<rect x="4" y="8" width="16" height="12" rx="2"/>'
             '<path d="M12 4v4"/><circle cx="12" cy="3" r="1"/>'
             '<path d="M9 13v2M15 13v2"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/>'
              '<path d="M12 3v3M12 18v3M3 12h3M18 12h3"/>',
    "gauge": '<path d="M12 14 8 9"/><path d="M3.5 17a9 9 0 1 1 17 0"/>',
    "terminal": '<path d="m4 17 6-6-6-6"/><path d="M12 19h8"/>',
    "list": '<path d="M8 6h13M8 12h13M8 18h13"/><path d="M3 6h.01M3 12h.01M3 18h.01"/>',
    "box": '<path d="m12 2 9 5v10l-9 5-9-5V7z"/><path d="m3 7 9 5 9-5"/>'
           '<path d="M12 12v10"/>',
    "refresh": '<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/>',
    "check": '<path d="m4 12 5 5L20 6"/>',
    "alert": '<path d="M12 3 2 21h20L12 3z"/><path d="M12 9v5"/><path d="M12 18h.01"/>',
    "layers": '<path d="m12 2 9 5-9 5-9-5z"/><path d="m3 12 9 5 9-5"/>'
              '<path d="m3 17 9 5 9-5"/>',
    "sliders": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/>'
               '<circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/>'
               '<circle cx="18" cy="18" r="2"/>',
}

_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
    'stroke-linejoin="round">{body}</svg>'
)


def svg_bytes(name: str, color: str = "#262626", width: float = 1.5) -> QByteArray:
    body = _PATHS.get(name, _PATHS["box"])
    return QByteArray(_SVG.format(color=color, width=width, body=body).encode("utf-8"))


def pixmap(name: str, color: str = "#262626", size: int = 16,
           ratio: float = 2.0, width: float = 1.5) -> QPixmap:
    px = int(size * ratio)
    ren = QSvgRenderer(svg_bytes(name, color, width))
    pm = QPixmap(px, px)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    ren.render(painter)
    painter.end()
    pm.setDevicePixelRatio(ratio)
    return pm


def icon(name: str, color: str = "#262626", size: int = 16,
         width: float = 1.5) -> QIcon:
    ic = QIcon()
    for ratio in (1.0, 2.0, 3.0):
        ic.addPixmap(pixmap(name, color, size, ratio, width))
    return ic


class IconCache:
    """按 (name, color, size) 缓存，主题切换后 clear()。"""

    def __init__(self) -> None:
        self._c: dict[tuple[str, str, int], QIcon] = {}

    def get(self, name: str, color: str, size: int = 16,
            width: float = 1.5) -> QIcon:
        key = (name, color, size)
        ic = self._c.get(key)
        if ic is None:
            ic = icon(name, color, size, width)
            self._c[key] = ic
        return ic

    def clear(self) -> None:
        self._c.clear()


ICONS = IconCache()
