"""OpenArm Studio 主题系统 —— Zai/zcode 双调色板 + QSS 生成。

调色板 token 与 QSS 模板逐值取自 ~/pbvs_workbench/pbvs_workbench/ui/theme.py
（2026-11-16 用户要求与 pbvs 工作台完全同规格：字体栈、字号档、圆角、
行高、字重全部保留原值）。未用到的选择器保留（Qt 忽略不存在的
objectName，缺了才是问题）。

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
    # 滑块已走过的轨道：深浅模式都用中性色（深白/浅黑），不用强调色。
    # 关节与夹爪滑块是常驻控件，蓝色在大面积使用时会抢注意力。
    "slider_fill": "#000000",
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
    # 深色模式下滑块轨道用白（对应浅色模式的黑）
    "slider_fill": "#ffffff",
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


# ──────────────────────── QSS 模板（pbvs_workbench 原版） ────────────────────

_QSS = ("""
/* OpenArm Studio 主题（由 ui/theme.py 生成，勿手改）——zcode/Zai 风
   （模板逐值取自 pbvs_workbench，未用选择器保留） */

/* 壳层：根容器画圆角窗口（无边框 + 半透明底），内容在其内 6px。
   外壳半径 = 面板半径 12 + 缝隙 6 = 18px */
QMainWindow { background: transparent; }
QWidget#windowRoot {
    background-color: $gutter;
    border: 1px solid $hairline_strong;
    border-radius: 18px;
}
QWidget#windowRoot[maximized="true"] {
    border-radius: 0;
    border: none;
}
QWidget#shellBody, QScrollArea { background: transparent; }
/* QScrollArea 的 viewport 默认按 QPalette.Base 填充（深色主题下仍是白），
   只给 QScrollArea 设透明不够，必须点名 viewport 和里面的内容 widget。 */
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollArea#statusScroll { border: none; background: transparent; }
QWidget#pageRoot { background: transparent; }

QWidget {
    color: $ink;
    font-family: "Inter Variable", "Inter", "Noto Sans SC",
                 "Noto Sans CJK SC", "Noto Sans", "WenQuanYi Micro Hei",
                 "Segoe UI", "Microsoft YaHei UI", "DejaVu Sans", sans-serif;
    font-size: $fs_base;
    font-weight: 400;
}

/* ── 分段控件（模式 / 主题）——zcode 分段：槽内凹，激活块滑动 ── */
QFrame#segmented {
    background-color: $panel_alt;
    border: 1px solid $hairline;
    border-radius: 8px;
    padding: 2px;
}
QFrame#segThumb {
    background-color: $panel;
    border: 1px solid $hairline;
    border-radius: 6px;
}
QPushButton#segBtn {
    background-color: transparent;
    border: none;
    border-radius: 6px;
    padding: 4px 14px;
    font-size: $fs_base;
    font-weight: 500;
    color: $ink2;
    min-height: 18px;
}
QPushButton#segBtn:hover { color: $ink; }
QPushButton#segBtn:checked {
    background-color: transparent;
    color: $ink;
    font-weight: 600;
}
QPushButton#segBtn:disabled { color: $faint; }

/* ── zcode 壳层：外沿底色 + 三个独立面板（4~6px 缝隙露外沿） ── */
QMainWindow { background-color: transparent; }
QWidget#shellBody { background-color: transparent; }
QFrame#inspectorPanel, QFrame#workspaceCard {
    background-color: $shell_panel;
    border: 1px solid $hairline_strong;
    border-radius: 12px;
}
QWidget#workspaceBody { background: transparent; }
QFrame#navPanel {
    background-color: $nav_bg;
    border: none;
    border-radius: 0;
}
QScrollArea#navScroll, QWidget#navHost, QScrollArea#inspScroll {
    background: transparent;
    border: none;
}
QPushButton#navItem {
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 0 10px;
    text-align: left;
    color: $ink2;
    font-size: $fs_base;
    font-weight: 400;
    min-height: 32px;
    max-height: 32px;
}
QPushButton#navItem:hover { background-color: $nav_hover; color: $ink; }
QPushButton#navItem[collapsed="true"] { padding: 0; }
QPushButton#navItem:checked {
    background-color: $nav_selected;
    color: $ink;
    font-weight: 600;
}
QPushButton#navGroupHeader {
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 0 24px 0 10px;
    text-align: left;
    color: $ink;
    font-size: $fs_base;
    font-weight: 600;
    min-height: 26px;
    max-height: 26px;
    letter-spacing: 0.01em;
}
QPushButton#navGroupHeader:hover {
    background: transparent;
    color: $ink;
}
QWidget#navGroupBody { background: transparent; }
QWidget#historyRow { background: transparent; }
QLabel#historyTitle { color: $ink; font-size: $fs_base; background: transparent; }
QLabel#historyTime { color: $ink2; font-size: $fs_sm; background: transparent; }
QFrame#navSummary {
    background-color: $nav_card;
    border: 1px solid $hairline;
    border-radius: 10px;
}
QLabel#navSummaryTitle { color: $ink; font-size: $fs_base; font-weight: 600; }
QLabel#navSummaryKey { color: $ink2; font-size: $fs_base; font-weight: 500; }
QLabel#navSummaryVal {
    color: $ink;
    font-size: $fs_caption;
    font-weight: 500;
    font-family: "JetBrains Mono", "Fira Code", "Consolas",
                 "DejaVu Sans Mono", monospace;
}
QFrame#navSeparator {
    background-color: $hairline;
    border: none;
    max-height: 1px;
    min-height: 1px;
}
QLabel#navFooter {
    color: $muted;
    font-size: $fs_caption;
    padding: 6px 10px 2px 10px;
    font-weight: 400;
}
QPushButton#navAction {
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 0 10px;
    text-align: left;
    color: $ink;
    font-size: $fs_base;
    font-weight: 400;
    min-height: 30px;
    max-height: 30px;
}
QPushButton#navAction:hover { background-color: $nav_hover; }
QPushButton#navAction:checked {
    background-color: $nav_selected; font-weight: 600;
}

/* 工作区顶栏（zcode workspace header）：标题/窗控/可拖动，仅一条底线 */
QFrame#shellHeader { background: transparent; border: none; }
QLabel#crumb {
    color: $ink2;
    font-size: $fs_caption;
    font-weight: 400;
}

/* 3D 视口左上角「关节」胶囊 + 展开卡片（zcode Popover 形态） */
QWidget#jointOverlay { background: transparent; }
QWidget#jointContent { background: transparent; }
QLabel#jointTitle { color: $ink; font-size: $fs_base; font-weight: 500; }
QLabel#jointSection {
    color: $ink2;
    font-size: $fs_sm;
    padding-top: 2px;
    font-weight: 500;
}
QLabel#jointName {
    color: $ink2;
    font-size: $fs_sm;
    font-family: "JetBrains Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-weight: 400;
}
QLabel#jointVal {
    color: $ink;
    font-size: $fs_caption;
    font-family: "JetBrains Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-weight: 500;
}
/* 关节力矩读数：超过额定 60% 转警示色（不是故障，是"这轴在使劲"） */
QLabel#jointTorque {
    color: $ink2;
    font-size: $fs_caption;
    font-family: "JetBrains Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-weight: 400;
}
QLabel#jointTorque[warn="true"] { color: $warn; }
QLabel#jointHint {
    color: $ink2;
    font-size: $fs_sm;
    padding-top: 2px;
    font-weight: 400;
}

QMenu {
    background-color: $panel;
    border: 1px solid $hairline_strong;
    border-radius: 8px;
    padding: 4px;
    color: $ink;
}
QMenu::item {
    padding: 5px 22px 5px 26px;
    border-radius: 4px;
    font-size: $fs_base;
    font-weight: 400;
}
QMenu::item:selected { background-color: $selected_bg; }
QMenu::indicator { width: 12px; height: 12px; left: 8px; }

QPushButton#navIconBtn {
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 0;
    min-width: 26px; max-width: 26px;
    min-height: 26px; max-height: 26px;
}
QPushButton#navIconBtn:hover { background-color: $nav_hover; }

QPushButton#winBtn, QPushButton#winClose {
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 0;
    min-width: 28px; max-width: 28px;
    min-height: 28px; max-height: 28px;
}
QPushButton#winBtn:hover { background-color: $hover_bg; }
QPushButton#winClose:hover { background-color: $danger; }

/* 导航轨按钮的禁用态：收敛成安静的内凹块 */
QFrame#navPanel QPushButton#btnPrimary:disabled {
    background-color: transparent;
    border: 1px solid $hairline_strong;
    color: $muted;
}
QFrame#navPanel QPushButton#btnDanger:disabled {
    background-color: transparent;
    border: 1px solid $disabled_border;
    color: $disabled_fg;
}
QFrame#navPanel QPushButton#btnEstop:disabled {
    background-color: transparent;
    border: 1px solid $danger_border;
    color: $danger_ink;
}
QFrame#navPanel QPushButton#btnPrimary {
    border-radius: 8px;
    font-size: $fs_caption;
    font-weight: 500;
    padding: 5px 8px;
    min-height: 16px;
    max-height: 16px;
}
QFrame#navPanel QPushButton#btnDanger,
QFrame#navPanel QPushButton#btnEstop {
    border-radius: 8px;
    font-size: $fs_caption;
    font-weight: 600;
    padding: 5px 8px;
    min-height: 16px;
    max-height: 16px;
}

/* Inspector 内容列：大标题 + 副标题（zcode 设置页 text-2xl/subtle） */
QLabel#inspTitle {
    color: $ink;
    font-size: $fs_xl;
    font-weight: 600;
    letter-spacing: 0.2px;
    padding: 4px 2px 0 2px;
}
QLabel#inspSub {
    color: $ink2;
    font-size: $fs_caption;
    line-height: 18px;
    padding: 5px 2px 0 2px;
    font-weight: 400;
}
QStackedWidget#inspectorStack { background: transparent; }

QFrame#sceneFrame {
    /* pbvs stage-shell 标准：1px 发丝线框把 MuJoCo 画面裱起来 */
    background: transparent;
    border: 1px solid $hairline_strong;
    border-radius: 10px;
}

/* 视口悬浮视角胶囊已移除（改走 Inspector「场景 → 视图」） */

/* ── 滑块（关节/夹爪直控）：4px 凹槽 + 12px 圆头 ── */
QSlider::groove:horizontal {
    height: 4px;
    border-radius: 2px;
    background-color: $panel_alt;
    border: 1px solid $hairline;
}
QSlider::handle:horizontal {
    width: 12px;
    height: 12px;
    margin: -6px 0;
    border-radius: 6px;
    background-color: $panel;
    border: 1px solid $hairline_strong;
}
QSlider::handle:horizontal:hover { background-color: $hover_bg; }
QSlider::handle:horizontal:pressed { background-color: $selected_bg; }
QSlider::sub-page:horizontal {
    background-color: $slider_fill;
    border-radius: 2px;
    margin: 0 0 0 2px;
}
QSlider::sub-page:horizontal:disabled { background-color: $panel_alt; }
QSlider:disabled { border: none; }

/* 工具条幽灵按钮（zcode：无边框、hover 变底、选中变底+正常前景） */
QPushButton#chip {
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 3px 8px;
    color: $ink2;
    font-size: $fs_sm;
    font-weight: 500;
    min-height: 16px;
    max-height: 16px;
}
QPushButton#chipIcon {
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 0;
}
QPushButton#chipIcon:hover { background-color: $hover_bg; }
QPushButton#chip:hover { background-color: $hover_bg; color: $ink; }
QPushButton#chip:checked {
    background-color: $selected_bg;
    color: $ink;
    font-weight: 600;
}

/* ── 卡片 / 分组设置（zcode：面板色卡片 + 发丝边） ── */
QFrame#groupCard {
    background-color: $panel;
    border: 1px solid $hairline;
    border-radius: 10px;
}
QLabel#groupTitle {
    font-size: $fs_caption;
    font-weight: 500;
    color: $ink2;
    padding-left: 2px;
}
QFrame#rowRule {
    background-color: $hairline;
    border: none;
    max-height: 1px;
}
QWidget#settingRowHost { background: transparent; }
QLabel#settingTitle { font-size: $fs_base; font-weight: 500; color: $ink; }
QLabel#settingDesc { color: $ink2; font-size: $fs_base; line-height: 19px; font-weight: 400; }
QLabel#groupDesc { color: $ink2; font-size: $fs_caption; padding: 1px 2px 6px 2px; font-weight: 400; }

QFrame#view3dCard {
    background: transparent;
    border: none;
    border-radius: 0;
}
QLabel#cardTitle { font-size: $fs_base; font-weight: 500; color: $ink; }
QLabel#sectionHead { font-size: $fs_base; font-weight: 500; color: $ink; }
QLabel#statusChip {
    font-size: $fs_sm;
    color: $ink2;
    background-color: $hover_bg;
    border-radius: 8px;
    padding: 2px 8px;
    font-weight: 500;
}
QLabel#cardHint { color: $ink2; font-size: $fs_sm; font-weight: 400; }
QLabel#sectionLabel {
    font-size: $fs_sm;
    font-weight: 500;
    color: $ink2;
    padding: 0 0 2px 0;
}
QLabel#fieldLabel { color: $ink2; font-size: $fs_sm; font-weight: 400; }
QLabel#hintText { color: $ink2; font-size: $fs_sm; font-weight: 400; }
QLabel#fieldValue {
    font-family: "JetBrains Mono", "Fira Code", "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: $fs_base; font-weight: 500; color: $ink;
}

QGroupBox {
    background-color: $panel;
    border: 1px solid $hairline;
    border-radius: 12px;
    padding: 32px 12px 12px 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: padding;
    left: 12px;
    top: 8px;
    padding: 0 2px;
    background-color: transparent;
    color: $ink2;
    font-size: $fs_sm;
    font-weight: 500;
}

/* ── 按钮（zcode：圆角矩形 6px、hover 只变底色） ── */
QPushButton {
    background-color: $panel;
    border: 1px solid $hairline_strong;
    border-radius: 6px;
    padding: 4px 12px;
    color: $ink;
    font-weight: 600;
    font-size: $fs_caption;
    min-height: 18px;
}
QPushButton:hover { background-color: $hover_bg; }
QPushButton:pressed { background-color: $panel_alt; }
QPushButton:focus { border-color: $ink2; outline: none; }
QPushButton:disabled {
    background-color: $disabled_bg;
    color: $disabled_fg;
    border-color: $disabled_border;
}

QPushButton#btnPrimary {
    background-color: $primary;
    border: 1px solid $primary;
    color: $primary_fg;
    font-weight: 600;
}
QPushButton#btnPrimary:hover { background-color: $primary_hover; border-color: $primary_hover; }
QPushButton#btnPrimary:pressed { background-color: $primary_press; }
QPushButton#btnPrimary:disabled {
    background-color: $primary_off_bg;
    border-color: $primary_off_bg;
    color: $primary_off_fg;
}

QPushButton#btnDanger {
    background-color: $panel;
    border: 1px solid $danger_border;
    color: $danger_ink;
    font-weight: 600;
}
QPushButton#btnDanger:hover { background-color: $danger_soft; border-color: $danger; }
QPushButton#btnDanger:pressed { background-color: $danger_soft; }
QPushButton#btnDanger:disabled {
    background-color: $disabled_bg; color: $disabled_fg; border-color: $disabled_border;
}

QPushButton#btnEstop {
    background-color: $danger;
    border: 1px solid $danger;
    color: #FFFFFF;
    font-weight: 700;
    font-size: $fs_caption;
    min-height: 22px;
    padding: 6px 16px;
    border-radius: 8px;
}
QPushButton#btnEstop:hover { background-color: $danger_hover; }
QPushButton#btnEstop:pressed { background-color: $danger_press; }
QPushButton#btnEstop:disabled {
    background-color: $disabled_bg; color: $disabled_fg; border-color: $disabled_border;
}

QPushButton#btnSecondary {
    background-color: transparent;
    border: 1px solid $hairline_strong;
    color: $ink2;
}
QPushButton#btnSecondary:hover { background-color: $hover_bg; color: $ink; }
QPushButton#btnSecondary:checked {
    background-color: $selected_bg;
    border-color: $hairline_strong;
    color: $ink;
    font-weight: 600;
}
QPushButton#btnSecondary:disabled {
    background-color: transparent; color: $disabled_fg; border-color: $disabled_border;
}

/* ── 输入控件（zcode input：同按钮刻度，focus 边框 mono、无光晕） ── */
QLineEdit, QDoubleSpinBox, QSpinBox {
    background-color: $panel;
    border: 1px solid $hairline_strong;
    border-radius: 6px;
    padding: 4px 8px;
    color: $ink;
    font-size: $fs_base;
    min-height: 18px;
    selection-background-color: $accent_soft_2;
    selection-color: $ink;
    font-weight: 400;
}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus { border-color: $ink2; }
QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {
    color: $disabled_fg; border-color: $disabled_border; background-color: $disabled_bg;
}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    background-color: transparent;
    border: none;
    width: 16px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-position: top right; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-position: bottom right; }

QComboBox {
    background-color: $panel;
    border: 1px solid $hairline_strong;
    border-radius: 6px;
    padding: 3px 8px;
    color: $ink;
    font-size: $fs_caption;
    min-height: 18px;
}
QComboBox:focus { border-color: $ink2; }
QComboBox:disabled {
    color: $disabled_fg; border-color: $disabled_border; background-color: $disabled_bg;
}
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background-color: $panel;
    border: 1px solid $hairline_strong;
    border-radius: 8px;
    padding: 4px;
    color: $ink;
    outline: none;
    selection-background-color: $selected_bg;
    selection-color: $ink;
}
QComboBox QAbstractItemView::item { padding: 5px 8px; border-radius: 4px; min-height: 18px; }
QComboBox QAbstractItemView::item:hover { background-color: $hover_bg; }

/* 统计条（zcode bg-surface 圆角条：无边框、无阴影） */
QFrame#statBar {
    background-color: $surface;
    border: none;
    border-radius: 10px;
}

QLabel#statLabel { color: $ink2; font-size: $fs_sm; font-weight: 500; }
QLabel#statValue {
    font-family: "JetBrains Mono", "Fira Code", "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: $fs_caption;
    font-weight: 500;
    color: $ink;
}
QLabel#kpiTitle { color: $ink2; font-size: $fs_caption; font-weight: 500; }
QLabel#kpiValue {
    font-family: "JetBrains Mono", "Fira Code", "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: $fs_stat;
    font-weight: 600;
    color: $ink;
}
QFrame#cardSep {
    background-color: $hairline;
    max-height: 1px;
    border: none;
}
QFrame#dashCard {
    background-color: $panel;
    border: 1px solid $hairline;
    border-radius: 8px;
}

/* ── 标签页 ── */
QTabWidget {
    background: transparent;
    border: none;
    border-radius: 0;
}
QTabWidget::pane {
    background-color: transparent;
    border: none;
    padding: 10px 16px 10px 16px;
}
QTabWidget::tab-bar { left: 16px; }
QTabBar { background-color: transparent; }
QTabBar::tab {
    background-color: transparent;
    border: none;
    padding: 6px 12px;
    margin-right: 2px;
    color: $muted;
    font-weight: 400;
    font-size: $fs_base;
}
QTabBar::tab:selected { color: $ink; font-weight: 600; }
QTabBar::tab:hover:!selected { color: $ink2; }

/* ── 文本 / 日志 ── */
QPlainTextEdit, QTextEdit {
    background-color: transparent;
    border: none;
    border-radius: 0;
    padding: 10px 12px;
    font-family: "JetBrains Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: $fs_caption;
    color: $ink2;
    selection-background-color: $accent_soft_2;
    selection-color: $ink;
    font-weight: 400;
}
QPlainTextEdit:focus, QTextEdit:focus { border-color: $hairline_strong; }
QPlainTextEdit#consoleLog {
    background-color: transparent; color: $ink2; border: none;
}

/* ── 复选框 ── */
QCheckBox { spacing: 6px; color: $ink; font-size: $fs_base; font-weight: 400; }
QCheckBox::indicator {
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid $hairline_strong; background-color: $panel;
}
QCheckBox::indicator:hover { border-color: $ink2; }
QCheckBox::indicator:checked { background-color: $primary; border-color: $primary; }
QCheckBox::indicator:disabled { border-color: $disabled_border; background-color: $disabled_bg; }

/* ── 滚动条（悬浮圆条）/ tooltip / 对话框 ── */
QDialog {
    background-color: $panel;
    border: 1px solid $hairline;
}
QMessageBox QLabel { background: transparent; color: $ink; }
QMessageBox QPushButton { min-width: 64px; }
QScrollBar:vertical { background: transparent; width: 14px; margin: 2px 0; }
QScrollBar::handle:vertical {
    background: $scroll; border-radius: 4px; min-height: 30px;
    margin: 2px 3px;
}
QScrollBar::handle:vertical:hover { background: $scroll_hover; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { background: transparent; height: 14px; margin: 0 2px; }
QScrollBar::handle:horizontal {
    background: $scroll; border-radius: 4px; min-width: 30px;
    margin: 3px 2px;
}
QScrollBar::handle:horizontal:hover { background: $scroll_hover; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
QToolTip {
    background-color: $tooltip_bg; color: $tooltip_fg;
    border: 1px solid $hairline_strong; border-radius: 8px;
    padding: 5px 10px; font-size: $fs_sm;
    font-weight: 400;
}

/* 空态 */
QLabel#emptyTitle { color: $ink; font-size: $fs_base; font-weight: 500; background: transparent; }
QLabel#emptyHint { color: $ink2; font-size: $fs_caption; background: transparent; }

/* 状态点文字 */
QWidget#statusDot { color: $muted; font-size: $fs_caption; font-weight: 600; }

/* 键帽 chip */
QLabel#kbd {
    background-color: $panel_alt;
    border: 1px solid $hairline;
    border-radius: 4px;
    color: $ink2;
    font-size: $fs_xs;
    font-weight: 500;
    padding: 0 5px;
}

QSplitter::handle { background-color: transparent; }
QSplitter::handle:hover { background-color: transparent; }

/* ── 对比度收口：说明/标签类统一 ink2，$muted 只留给禁用态与装饰 ── */
QLabel#navSummaryKey, QLabel#navFooter,
QLabel#jointSection, QLabel#jointHint,
QLabel#inspSub, QLabel#toolLabel,
QLabel#groupDesc, QLabel#cardHint,
QLabel#fieldLabel, QLabel#hintText,
QLabel#statusChip, QTabBar::tab {
    color: $ink2;
}
""")


def build_qss(theme: dict) -> str:
    """把 token 代入 QSS 模板。

    必须按标识符边界替换：朴素 str.replace 会让 `$ink` 吃掉 `$ink2`
    的前缀（pbvs 踩过的坑，QSS 会整体失效且被 Qt 静默丢弃）。
    """
    qss = _QSS
    for key, val in theme.items():
        qss = re.sub(r"\$" + re.escape(key) + r"(?![A-Za-z0-9_])", str(val), qss)
    return qss
