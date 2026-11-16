#!/usr/bin/env python3
"""把 SVG 渲染成多尺寸 PNG（供 dock / 任务栏图标使用）。"""
import sys
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "openarm_studio/ui/assets/openarm_studio.svg"
OUT = Path.home() / ".local/share/icons/hicolor"


def render(size: int, out: Path) -> None:
    ren = QSvgRenderer(str(SVG))
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)
    ren.render(p, QRectF(0, 0, size, size))
    p.end()
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out), "PNG")


def main() -> int:
    QApplication(sys.argv)
    for size in (16, 24, 32, 48, 64, 128, 256, 512):
        out = OUT / f"{size}x{size}/apps/openarm-studio.png"
        render(size, out)
        print("wrote", out)
    # scalable
    scal = OUT / "scalable/apps/openarm-studio.svg"
    scal.parent.mkdir(parents=True, exist_ok=True)
    scal.write_text(SVG.read_text())
    print("wrote", scal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
