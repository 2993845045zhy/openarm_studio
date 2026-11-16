"""OpenArm Studio 主题系统 —— Zai/zcode 双调色板 + QSS 生成。

与 pbvs_workbench 同源的设计语言（token 值逐一取自
~/pbvs_workbench/pbvs_workbench/ui/theme.py），本模块只覆盖 OpenArm
Studio 用到的组件，因此比原版精简，但调色板与尺寸标尺完全一致。

尺寸标尺（几何不随字号缩放）：
    S1 16 / S2 12 / S3 10 / S4 8 / S5 6 / S6 4
    R1 18 壳层 / R2 12 卡片 / R3 10 凹槽 / R4 8 控件 / R5 6 小件 / R6 4
"""

from __future__ import annotations

import re

LIGHT = {
    "name": "light",
    "label": "浅色",
    "bg": "#f8f8f8",
    "gutter": "#ececee",
    "nav_bg": "transparent",
    "nav_card": "#ffffff",
    "shell_panel": "#f8f8f8",
    "selected_bg": "#e9e9e9",
    "surface": "#f3f3f3",
    "border_hover_soft": "#d4d4d4",
    "sidebar": "#f0f0f0",
    "panel": "#ffffff",
    "panel_alt": "#f0f0f0",
    "hairline": "#1A0D0D0D",
    "hairline_strong": "#290D0D0D",
    "ink": "#262626",
    "ink2": "#6b6b6b",
    "muted": "#9a9a9a",
    "faint": "#c2c2c2",
    "primary": "#000000",
    "primary_fg": "#ffffff",
    "primary_hover": "#2e2e2e",
    "primary_press": "#454545",
    "accent": "#0b7fff",
    "accent_hover": "#0a70e0",
    "accent_press": "#0960c0",
    "accent_ink": "#1a70b8",
    "accent_soft": "#ebf4ff",
    "accent_soft_2": "#dbeafe",
    "accent_border": "#bcd7f7",
    "danger": "#e03131",
    "danger_hover": "#cb2b2b",
    "danger_press": "#b32525",
    "danger_soft": "#fdeeee",
    "danger_border": "#f1baba",
    "danger_ink": "#c22626",
    "success": "#1e8a3e",
    "warn": "#e07b00",
    "warn_soft": "#fdf3e5",
    "warn_border": "#f2c999",
    "success_soft": "#e9f4ec",
    "success_border": "#a8d0b4",
    "viewport": "#ececee",
    "intro_fade": "#b8b8b8",
    "scroll": "#d4d4d4",
    "scroll_hover": "#b8b8b8",
    "tooltip_bg": "#f0f0f0",
    "tooltip_fg": "#171717",
    "disabled_fg": "#a8a8a8",
    "disabled_bg": "#f0f0f0",
    "disabled_border": "#e3e3e3",
    "primary_off_bg": "#c9c9c9",
    "primary_off_fg": "#f5f5f5",
    "hover_bg": "#ededed",
    "nav_hover": "#e1e1e1",
    "nav_selected": "#d6d6d6",
    "scene_sky_top": "0.973 0.973 0.973",
    "scene_sky_horizon": "0.973 0.973 0.973",
    "scene_floor": "0.973 0.973 0.973",
    "scene_grid_line": 0.86,
    "scene_light_scale": 1.55,
    "scene_floor_reflect": 0.12,
    "scene_haze": 0.35,
    "scene_shadow": 0,
    "curve_pos": "#1a70b8",
    "curve_ang": "#7453b0",
    "curve_v": "#14807a",
    "curve_w": "#b87a1a",
    "curve_grid": "#e8e8e8",
    "curve_text": "#8f8f8f",
    "curve_bg": "#f8f8f8",
}

DARK = {
    "name": "dark",
    "label": "深色",
    "bg": "#161616",
    "gutter": "#2b2b2b",
    "nav_bg": "transparent",
    "shell_panel": "#161616",
    "selected_bg": "#333333",
    "surface": "#202020",
    "border_hover_soft": "#4a4a4a",
    "sidebar": "#161616",
    "panel": "#2b2b2b",
    "panel_alt": "#202020",
    "hairline": "#1AFFFFFF",
    "hairline_strong": "#29FFFFFF",
    "ink": "#d4d4d4",
    "ink2": "#8a8a8a",
    "muted": "#606060",
    "faint": "#454545",
    "primary": "#ffffff",
    "primary_fg": "#0d0d0d",
    "primary_hover": "#d6d6d6",
    "primary_press": "#b8b8b8",
    "accent": "#4099ff",
    "accent_hover": "#5ca6ff",
    "accent_press": "#3585e0",
    "accent_ink": "#8fc5ef",
    "accent_soft": "#0d2440",
    "accent_soft_2": "#14305c",
    "accent_border": "#1d4a8c",
    "danger": "#ff5c5c",
    "danger_hover": "#ff7070",
    "danger_press": "#e64f4f",
    "danger_soft": "#403030",
    "danger_border": "#7a3d3d",
    "danger_ink": "#ff8c8c",
    "success": "#46bf72",
    "warn": "#ff8a30",
    "warn_soft": "#41322a",
    "warn_border": "#7a5630",
    "success_soft": "#27392d",
    "success_border": "#356647",
    "viewport": "#101010",
    "intro_fade": "#0a0a0a",
    "scroll": "#2e2e2e",
    "scroll_hover": "#3f3f3f",
    "tooltip_bg": "#2b2b2b",
    "tooltip_fg": "#f8f8f8",
    "disabled_fg": "#666666",
    "disabled_bg": "#242424",
    "disabled_border": "#333333",
    "primary_off_bg": "#5a5a5a",
    "primary_off_fg": "#161616",
    "hover_bg": "#383838",
    "nav_hover": "#3f3f3f",
    "nav_selected": "#4a4a4a",
    "nav_card": "#161616",
    "scene_sky_top": "0.0865 0.0865 0.0865",
    "scene_sky_horizon": "0.0865 0.0865 0.0865",
    "scene_floor": "0.0865 0.0865 0.0865",
    "scene_grid_line": 1.45,
    "scene_light_scale": 1.0,
    "scene_floor_reflect": 0.16,
    "scene_haze": 0.35,
    "scene_shadow": 0,
    "curve_pos": "#8fc5ef",
    "curve_ang": "#bda5e6",
    "curve_v": "#93d8d2",
    "curve_w": "#efc58f",
    "curve_grid": "#363636",
    "curve_text": "#707070",
    "curve_bg": "#161616",
}

THEMES = {"light": LIGHT, "dark": DARK}

FS_DEFAULT = 15


def font_tokens(base: float = FS_DEFAULT) -> dict:
    return {
        "fs_2xl": "24px",
        "fs_xl": f"{base + 4:g}px",
        "fs_lg": f"{base + 2:g}px",
        "fs_base": f"{base:g}px",
        "fs_caption": f"{base - 1:g}px",
        "fs_sm": f"{base - 2:g}px",
        "fs_xs": f"{base - 4:g}px",
        "fs_stat": f"{base + 1:g}px",
    }


def palette(name: str, base: float = FS_DEFAULT) -> dict:
    pal = dict(THEMES.get(name, LIGHT))
    pal.update(font_tokens(base))
    return pal


def scene_colors(p: dict) -> dict:
    return {
        "sky_top": p["scene_sky_top"],
        "sky_horizon": p["scene_sky_horizon"],
        "floor_rgba": _rgba(p["scene_floor"]),
        "grid_line": float(p.get("scene_grid_line", 0.9)),
        "haze": float(p.get("scene_haze", 0.5)),
        "haze_rgba": _rgba(p["scene_sky_horizon"]),
        "shadow": int(p.get("scene_shadow", 0)),
        "light_scale": float(p.get("scene_light_scale", 1.0)),
        "floor_reflect": float(p.get("scene_floor_reflect", 0.10)),
        "name": p["name"],
    }


def _rgba(space_sep: str, alpha: float = 1.0) -> list:
    vals = [float(v) for v in space_sep.split()]
    return vals + [alpha]


_QSS = """
QWidget {
    color: $ink;
    font-size: $fs_base;
    font-family: "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC",
                 "Microsoft YaHei", sans-serif;
}
QWidget#windowRoot {
    background: $gutter;
    border-radius: $r1;
}
QWidget#windowRoot[maximized="true"] { border-radius: 0px; }
QWidget#shellBody, QWidget#workspaceBody { background: transparent; }

/* ── 导航轨 ─────────────────────────────────────────── */
QFrame#navRail {
    background: $nav_bg;
    border: none;
}
QLabel#navTitle { font-size: $fs_lg; font-weight: 600; color: $ink; }
QLabel#navSubtitle { font-size: $fs_xs; color: $muted; }
QFrame#navGroupHeader {
    background: transparent;
    border: none;
}
QLabel#navGroupLabel { font-size: $fs_xs; color: $muted; font-weight: 600; letter-spacing: 0.4px; }
QPushButton#navItem {
    background: transparent;
    border: none;
    border-radius: $r4;
    text-align: left;
    padding: 0 10px;
    color: $ink2;
    font-size: $fs_caption;
}
QPushButton#navItem:hover { background: $nav_hover; color: $ink; }
QPushButton#navItem:checked { background: $nav_selected; color: $ink; font-weight: 600; }
QFrame#navSummary {
    background: $nav_card;
    border: 1px solid $hairline;
    border-radius: $r2;
}

/* ── Inspector ──────────────────────────────────────── */
QFrame#inspector {
    background: $shell_panel;
    border: none;
}
QLabel#pageTitle { font-size: $fs_xl; font-weight: 600; color: $ink; }
QLabel#pageDesc { font-size: $fs_caption; color: $ink2; }
QLabel#groupLabel {
    font-size: $fs_xs; color: $muted; font-weight: 600; letter-spacing: 0.3px;
}
QFrame#inspCard {
    background: $panel;
    border: 1px solid $hairline;
    border-radius: $r2;
}
QFrame#inspRow { background: transparent; border: none; }
QLabel#rowLabel { font-size: $fs_caption; color: $ink2; }
QLabel#rowValue { font-size: $fs_caption; color: $ink; font-weight: 600; }
QLabel#hint { font-size: $fs_xs; color: $muted; }

/* ── 主卡 ───────────────────────────────────────────── */
QFrame#workspaceCard {
    background: $shell_panel;
    border: 1px solid $hairline;
    border-radius: $r2;
}
QFrame#view3dCard {
    background: $panel;
    border: 1px solid $hairline;
    border-radius: $r2;
}
QFrame#statusCard {
    background: $panel;
    border: 1px solid $hairline;
    border-radius: $r2;
}
QFrame#cardHeader { background: transparent; border: none; }
QLabel#cardTitle { font-size: $fs_lg; font-weight: 600; color: $ink; }
QLabel#chip {
    font-size: $fs_xs;
    color: $ink2;
    background: $surface;
    border: 1px solid $hairline;
    border-radius: $r5;
    padding: 2px 7px;
}
QFrame#statStrip {
    background: $surface;
    border: none;
    border-radius: $r3;
}

/* ── 按钮 ───────────────────────────────────────────── */
QPushButton {
    background: $panel;
    border: 1px solid $hairline_strong;
    border-radius: $r4;
    padding: 0 12px;
    min-height: 30px;
    color: $ink;
    font-size: $fs_caption;
}
QPushButton:hover { background: $hover_bg; }
QPushButton:pressed { background: $selected_bg; }
QPushButton:disabled { color: $disabled_fg; background: $disabled_bg; border-color: $disabled_border; }
QPushButton#primary {
    background: $primary;
    color: $primary_fg;
    border: 1px solid $primary;
    font-weight: 600;
}
QPushButton#primary:hover { background: $primary_hover; }
QPushButton#primary:disabled { background: $primary_off_bg; color: $primary_off_fg; border-color: $primary_off_bg; }
QPushButton#danger {
    background: $danger;
    color: #ffffff;
    border: 1px solid $danger;
    font-weight: 600;
}
QPushButton#danger:hover { background: $danger_hover; }
QPushButton#danger:pressed { background: $danger_press; }
QPushButton#ghost {
    background: transparent;
    border: 1px solid transparent;
    color: $ink2;
}
QPushButton#ghost:hover { background: $hover_bg; color: $ink; }
QPushButton#tool {
    background: transparent;
    border: 1px solid transparent;
    min-width: 28px;
    min-height: 26px;
    padding: 0;
}
QPushButton#tool:hover { background: $hover_bg; }
QPushButton#tool:checked { background: $selected_bg; }

/* ── 输入 ───────────────────────────────────────────── */
QLineEdit, QDoubleSpinBox, QSpinBox {
    background: $panel_alt;
    border: 1px solid $hairline_strong;
    border-radius: $r4;
    padding: 0 8px;
    min-height: 28px;
    selection-background-color: $accent;
    selection-color: #ffffff;
}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus { border-color: $accent_border; }
QComboBox {
    background: $panel_alt;
    border: 1px solid $hairline_strong;
    border-radius: $r4;
    padding: 0 8px;
    min-height: 28px;
}
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background: $panel;
    border: 1px solid $hairline_strong;
    border-radius: $r4;
    selection-background-color: $selected_bg;
    selection-color: $ink;
    outline: none;
}

/* ── 滑块 ───────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 3px;
    background: $panel_alt;
    border-radius: 2px;
}
QSlider::sub-page:horizontal { background: $ink2; border-radius: 2px; }
QSlider::handle:horizontal {
    background: $panel;
    border: 1px solid $border_hover_soft;
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}
QSlider::handle:horizontal:hover { border-color: $ink2; }
QSlider:disabled::sub-page:horizontal { background: $disabled_border; }

/* ── 页签 ───────────────────────────────────────────── */
QTabWidget::pane { border: none; background: transparent; }
QTabBar::tab {
    background: transparent;
    border: none;
    color: $muted;
    padding: 5px 10px;
    font-size: $fs_caption;
}
QTabBar::tab:selected { color: $ink; font-weight: 600; }
QTabBar::tab:hover { color: $ink2; }

/* ── 分段控件 ───────────────────────────────────────── */
QFrame#segTrack {
    background: $panel_alt;
    border: none;
    border-radius: $r5;
}
QFrame#segThumb {
    background: $panel;
    border: 1px solid $hairline_strong;
    border-radius: $r5;
}
QPushButton#segBtn {
    background: transparent;
    border: none;
    border-radius: $r5;
    padding: 0;
    min-height: 24px;
    color: $muted;
    font-size: $fs_xs;
    font-weight: 600;
}
QPushButton#segBtn:hover { color: $ink2; }
QPushButton#segBtn:checked { color: $ink; }
QPushButton#segBtn:disabled { color: $disabled_fg; }

/* ── 日志/文本 ──────────────────────────────────────── */
QPlainTextEdit, QTextEdit {
    background: $panel_alt;
    border: 1px solid $hairline;
    border-radius: $r3;
    padding: 6px;
    font-family: "JetBrains Mono", "Noto Sans Mono", monospace;
    font-size: $fs_xs;
    color: $ink2;
}

/* ── 滚动条 ─────────────────────────────────────────── */
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical {
    background: $scroll;
    border-radius: 5px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: $scroll_hover; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { background: transparent; height: 10px; }
QScrollBar::handle:horizontal { background: $scroll; border-radius: 5px; min-width: 28px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

QSplitter::handle { background: transparent; }

QToolTip {
    background: $tooltip_bg;
    color: $tooltip_fg;
    border: 1px solid $hairline_strong;
    border-radius: $r5;
    padding: 4px 8px;
    font-size: $fs_xs;
}

/* ── 状态点 ─────────────────────────────────────────── */
QLabel#statusText { font-size: $fs_xs; color: $ink2; }
QFrame#hairline { background: $hairline; border: none; }

/* ── 关节条 ─────────────────────────────────────────── */
QFrame#jointRow { background: transparent; border: none; }
QLabel#jointName { font-size: $fs_xs; color: $muted; }
QLabel#jointVal {
    font-size: $fs_xs;
    color: $ink;
    font-family: "JetBrains Mono", "Noto Sans Mono", monospace;
}

/* ── 统计瓦片 ───────────────────────────────────────── */
QFrame#statTile {
    background: $panel_alt;
    border: 1px solid $hairline;
    border-radius: $r3;
}
QLabel#statValue {
    font-size: $fs_stat;
    font-weight: 600;
    color: $ink;
    font-family: "JetBrains Mono", "Noto Sans Mono", monospace;
}

/* ── 3D 视图底 ──────────────────────────────────────── */
QWidget#mjPanel { background: $bg; }
QLabel#mjHint { font-size: $fs_xs; color: $muted; }
QFrame#jointCapsule {
    background: $panel;
    border: 1px solid $hairline_strong;
    border-radius: $r3;
}
QLabel#breadcrumb { font-size: $fs_caption; color: $ink2; }
QLabel#modeBadge {
    font-size: $fs_xs;
    font-weight: 600;
    color: $accent_ink;
    background: $accent_soft;
    border: 1px solid $accent_border;
    border-radius: $r5;
    padding: 2px 8px;
}
"""

# 尺寸标尺（几何不随字号缩放）
_RADII = {"r1": 18, "r2": 12, "r3": 10, "r4": 8, "r5": 6, "r6": 4}


def build_qss(theme: dict) -> str:
    """把 token 代入 QSS 模板。

    必须按标识符边界替换：朴素 str.replace 会让 `$ink` 吃掉 `$ink2`
    的前缀（pbvs 踩过的坑，QSS 会整体失效且被 Qt 静默丢弃）。
    圆角 token 的值自带 `px`，模板里只写 `$r4`（写 `$r4` 会因边界保护
    匹配不到，QSS 直接非法）。
    """
    sub = dict(theme)
    for key, val in _RADII.items():
        sub[key] = f"{val}px"
    qss = _QSS
    for key, val in sub.items():
        qss = re.sub(r"\$" + re.escape(key) + r"(?![A-Za-z0-9_])", str(val), qss)
    return qss
