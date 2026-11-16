"""OpenArm Studio 入口。

    python -m openarm_studio.app [--theme dark|light] [--mode real|sim]
                                [--screenshot out.png]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtGui import QIcon  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from .ui.shell import StudioWindow  # noqa: E402

_ICON = Path(__file__).resolve().parent / "ui/assets/openarm_studio.svg"


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenArm Studio 上位机")
    parser.add_argument("--theme", "-t", default="dark", choices=["dark", "light"])
    parser.add_argument("--mode", "-m", default="real", choices=["real", "sim"])
    parser.add_argument("--screenshot", "-s", default=None,
                        help="离屏自检：渲染若干帧后截图退出")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("openarm_studio")
    app.setDesktopFileName("openarm-studio")
    if _ICON.exists():
        app.setWindowIcon(QIcon(str(_ICON)))

    win = StudioWindow(theme_name=args.theme)
    if args.mode == "sim":
        win.sw_mode.setCurrentIndex(1, animate=False)
    win.show()

    if args.screenshot:
        out = args.screenshot

        def shoot() -> None:
            # 这台机器的 Qt 6.11 + X11 上 QWidget.grab() 会段错误，
            # 改用屏幕区域抓取（窗口已 show，渲染结果就是屏幕上的样子）
            pm = None
            scr = app.primaryScreen()
            if scr is not None:
                g = win.frameGeometry()
                pm = scr.grabWindow(0, g.x(), g.y(), g.width(), g.height())
            if pm is None or pm.isNull():
                pm = win.grab()
            pm.save(out)
            print(f"screenshot saved: {out}")
            win.close()
            app.quit()

        def wait_ready() -> None:
            if win._worker is not None and win._worker.wait_ready(25.0):
                QTimer.singleShot(1200, shoot)
            else:
                print("WARN: 3D 模型未在超时内就绪，仍然截图")
                QTimer.singleShot(200, shoot)

        QTimer.singleShot(50, wait_ready)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
