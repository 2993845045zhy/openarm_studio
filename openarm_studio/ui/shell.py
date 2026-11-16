"""OpenArm Studio 主窗口 —— 三面板壳层（导航轨 / Inspector / 主卡）。

布局照 pbvs_workbench：左导航轨 236px → Inspector 300px → 右侧主卡，
面板间留 6px 缝隙露出外沿底色；无边框 + 圆角 18px。

3D 渲染在独立线程里跑（EGL 上下文线程绑定），GUI 只在定时器里取帧。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import mujoco
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout,
                               QLabel, QMainWindow, QPlainTextEdit, QPushButton,
                               QSlider, QSplitter, QTabWidget, QVBoxLayout,
                               QWidget)

from ..core.mjview import MjEmbedRenderer
from ..core.sim import SimBackend
from ..model import scene as model_scene
from ..ros.bridge import JointStatePublisher, RosBridge
from .chrome import DragHandle, ResizeHandles, WindowButtons
from .icons import ICONS
from .joint_monitor import JointOverlay
from .mjview_panel import MujocoViewPanel
from .switch import ModeSwitch
from .theme import build_qss, palette, scene_colors
from .widgets import GroupCard, Row, StatTile
from .zcode import (AnimatedTabBar, GroupRule, HistoryRow, LogoButton, NavCommandRow,
                    NavGroupHeader, NavItem, SettingsRow, StatusDot, Switch, ToolChip)

SIDE_W = 236
SIDE_COLLAPSED_W = 56
INSP_W = 300


# ────────────────────────────────────────────────────────────
# 渲染线程
# ────────────────────────────────────────────────────────────
class RenderWorker(threading.Thread):
    """独占 MuJoCo Renderer 的线程：谁创建谁销毁。"""

    def __init__(self, panel: MujocoViewPanel, cache_dir: Path,
                 theme_name: str, on_log=None):
        super().__init__(daemon=True, name="openarm-render")
        self._panel = panel
        self._cache = cache_dir
        self._theme = theme_name
        self._log = on_log or (lambda *_: None)
        self._lock = threading.Lock()
        self._pose: dict[str, float] = {}
        self._latest: dict[str, float] = {}
        self._mode = "twin"
        self._pending_sync = False
        self._rebuild = True
        self._running = True
        self._ready = threading.Event()
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.sim: SimBackend | None = None
        self._qmap: dict[str, int] = {}

    # ── 外部接口（线程安全）────────────────────────────
    def set_pose(self, positions: dict[str, float]) -> None:
        with self._lock:
            self._pose = dict(positions)

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self._mode = mode
            self._pending_sync = (mode == "sim")

    def positions(self) -> dict[str, float]:
        with self._lock:
            return dict(self._latest)

    def set_theme(self, theme_name: str) -> None:
        with self._lock:
            self._theme = theme_name
            self._rebuild = True

    def wait_ready(self, timeout: float = 20.0) -> bool:
        return self._ready.wait(timeout)

    def stop(self) -> None:
        self._running = False

    def joint_positions(self) -> dict[str, float]:
        with self._lock:
            return dict(self._pose)

    # ── 主循环 ────────────────────────────────────────
    def run(self) -> None:
        renderer = MjEmbedRenderer(height=720)
        try:
            while self._running:
                with self._lock:
                    theme = self._theme
                    rebuild = self._rebuild
                if rebuild or self.model is None:
                    sc = scene_colors(palette(theme))
                    self.model = model_scene.build_model(self._cache, sc)
                    self.data = mujoco.MjData(self.model)
                    self._qmap = model_scene.joint_qpos_map(self.model)
                    self.sim = SimBackend(self.model)
                    with self._lock:
                        self._rebuild = False
                    self._ready.set()
                    self._log(f"3D 模型就绪（{theme}）：nq={self.model.nq} "
                              f"ngeom={self.model.ngeom} nu={self.model.nu}")

                with self._lock:
                    pose = dict(self._pose)
                    mode = self._mode
                    pending_sync = self._pending_sync
                    self._pending_sync = False
                assert self.model is not None and self.data is not None

                if mode == "sim" and self.sim is not None:
                    if pending_sync:
                        # 切进仿真：伺服目标钉在当前姿态再预热，
                        # 否则刚度 230 的伺服会一把把臂拽到旧目标
                        self.sim.set_targets(self.sim.positions())
                        self.sim.sync_targets_to_current()
                        self.sim.settle(steps=200)
                        pose = self.sim.positions()
                        with self._lock:
                            self._pose = dict(pose)
                    self.sim.set_targets(pose)
                    self.sim.step()
                    render_data = self.sim.data
                    latest = self.sim.positions()
                else:
                    # 孪生：外部姿态直接写 qpos（运动学镜像，不跑物理）
                    render_data = self.data
                    for name, val in pose.items():
                        idx = self._qmap.get(name)
                        if idx is not None:
                            render_data.qpos[idx] = float(val)
                    mujoco.mj_forward(self.model, render_data)
                    latest = dict(pose)
                with self._lock:
                    self._latest = latest

                state = self._panel.cam_state()
                rgb = renderer.render(self.model, render_data, state)
                self._panel.push_frame(rgb, wire=state.get("wireframe"))
                time.sleep(1 / 30)
        except Exception as exc:  # noqa: BLE001
            self._log(f"[3D] 渲染线程异常: {type(exc).__name__}: {exc}")
        finally:
            renderer.close()


# ────────────────────────────────────────────────────────────
# 主窗口
# ────────────────────────────────────────────────────────────
class StudioWindow(QMainWindow):
    log_sig = Signal(str)

    def __init__(self, theme_name: str = "dark"):
        super().__init__()
        self._theme_name = theme_name
        self._reduced = False
        self._worker: RenderWorker | None = None
        self.bridge: RosBridge | None = None
        self.js_pub: JointStatePublisher | None = None
        self._local_pose: dict[str, float] = {}
        self.nav_items: list[NavItem] = []
        self._switches: list[Switch] = []
        self._tabbars: list[AnimatedTabBar] = []
        self._pal_now = palette(theme_name)

        self.setWindowTitle("OpenArm Studio")
        self.resize(1560, 960)
        self.setMinimumSize(1400, 800)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

        self._build_ui()
        self._apply_theme(theme_name, animate=False)
        self._goto_page(0)
        self._start_render()
        self._start_bridge()

        self.log_sig.connect(self._append_log)
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(200)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()
        # 仿真模式要按 50Hz 发 /joint_states（MoveIt/RViz 依赖这个频率）
        self._pub_timer = QTimer(self)
        self._pub_timer.setInterval(20)
        self._pub_timer.timeout.connect(self._publish_sim_state)
        self._pub_timer.start()

    # ── 构建 ──────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("windowRoot")
        root.setAttribute(Qt.WA_StyledBackground, True)
        self.setCentralWidget(root)

        body = QWidget(root)
        body.setObjectName("shellBody")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(body)

        bl = QHBoxLayout(body)
        bl.setContentsMargins(8, 8, 8, 8)
        bl.setSpacing(6)

        self.nav = self._build_nav()
        bl.addWidget(self.nav)

        right = QWidget(body)
        right.setObjectName("workspaceBody")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)

        # 标题栏本身就是拖动区：面包屑标签不消费鼠标，事件冒泡到这里
        header = DragHandle(self, right)
        header.setObjectName("shellHeader")
        header.setFixedHeight(34)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(2, 0, 0, 0)
        hl.setSpacing(8)
        self.crumb = QLabel("真机 · 概览", header)
        self.crumb.setObjectName("crumb")
        hl.addWidget(self.crumb)
        hl.addStretch(1)
        self.badge_mode = QLabel("数字孪生", header)
        self.badge_mode.setObjectName("modeBadge")
        hl.addWidget(self.badge_mode)
        self.win_btns = WindowButtons(self, palette(self._theme_name), header)
        hl.addWidget(self.win_btns)
        rl.addWidget(header)

        cards = QHBoxLayout()
        cards.setContentsMargins(0, 0, 0, 0)
        cards.setSpacing(6)
        self.inspector = self._build_inspector()
        cards.addWidget(self.inspector)
        cards.addWidget(self._build_workspace(), 1)
        rl.addLayout(cards, 1)

        bl.addWidget(right, 1)

        self._resize_handles = ResizeHandles(self)

    # ── 导航轨 ────────────────────────────────────────
    def _build_nav(self) -> QFrame:
        nav = QFrame(self)
        nav.setObjectName("navPanel")
        nav.setMinimumWidth(SIDE_W)
        nav.setMaximumWidth(SIDE_W)
        lay = QVBoxLayout(nav)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        # 品牌区：logo（点击折叠）+ 标题/副标题
        brand = QWidget(nav)
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(2, 0, 0, 0)
        bl.setSpacing(9)
        self.logo = LogoButton(palette(self._theme_name), brand)
        self.logo.clicked.connect(self._toggle_nav)
        bl.addWidget(self.logo, 0, Qt.AlignTop)
        col = QVBoxLayout()
        col.setContentsMargins(0, 1, 0, 0)
        col.setSpacing(0)
        title = QLabel("OpenArm Studio", brand)
        title.setObjectName("navSummaryTitle")
        sub = QLabel("双臂 · 数字孪生工作台", brand)
        sub.setObjectName("navFooter")
        col.addWidget(title)
        col.addWidget(sub)
        bl.addLayout(col, 1)
        lay.addWidget(brand)

        # 模式分段
        self.sw_mode = ModeSwitch(["真机", "仿真"], nav)
        self.sw_mode.currentIndexChanged.connect(self._on_mode_change)
        lay.addWidget(self.sw_mode)

        # 主操作（整宽）+ 重置（ghost）
        self.btn_start = QPushButton("启动", nav)
        self.btn_start.setObjectName("btnPrimary")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_start)

        self.btn_reset = QPushButton("重置", nav)
        self.btn_reset.setObjectName("navAction")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setIcon(ICONS.get("reset", palette(self._theme_name)["ink2"], 15))
        self.btn_reset.clicked.connect(self._zero_pose)
        lay.addWidget(self.btn_reset)

        # 服务控制（停止 / 急停：不再挤在主按钮行）
        ctl = QHBoxLayout()
        ctl.setSpacing(4)
        self.btn_stop = QPushButton("停止", nav)
        self.btn_stop.setObjectName("btnDanger")
        self.btn_estop = QPushButton("急停", nav)
        self.btn_estop.setObjectName("btnEstop")
        for b in (self.btn_stop, self.btn_estop):
            b.setCursor(Qt.PointingHandCursor)
            ctl.addWidget(b, 1)
        lay.addLayout(ctl)

        # 可折叠分组 + 图标导航行
        self.nav_groups: list[tuple[NavGroupHeader, list[NavItem]]] = []
        lay.addWidget(self._nav_group("设备", [
            ("概览", "robot", 0), ("机器人", "target", 1), ("场景", "layers", 2)]))
        lay.addWidget(self._nav_group("控制", [
            ("关节", "sliders", 3), ("规划", "crosshair", 4)]))
        lay.addWidget(self._nav_group("诊断", [
            ("日志", "terminal", 5)]))
        lay.addStretch(1)

        # 最近运行
        hist_box = QWidget(nav)
        hl = QVBoxLayout(hist_box)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)
        self.hist_header = NavGroupHeader("最近运行", palette(self._theme_name), hist_box)
        hl.addWidget(self.hist_header)
        self.hist_body = QWidget(hist_box)
        hbl = QVBoxLayout(self.hist_body)
        hbl.setContentsMargins(0, 0, 0, 0)
        hbl.setSpacing(0)
        self.hist_rows: list[HistoryRow] = []
        for title, rel in (("连接真机", "刚刚"), ("仿真运行", "3分"),
                           ("规划 home", "1小时"), ("启动检查", "1天")):
            row = HistoryRow(title, rel, self.hist_body)
            hbl.addWidget(row)
            self.hist_rows.append(row)
        hl.addWidget(self.hist_body)
        self.hist_header.toggled.connect(self.hist_body.setVisible)
        lay.addWidget(hist_box)

        # 运行摘要卡（key/value 行）
        recent = QFrame(nav)
        recent.setObjectName("navSummary")
        rl = QVBoxLayout(recent)
        rl.setContentsMargins(11, 9, 11, 10)
        rl.setSpacing(4)
        sum_title = QLabel("运行摘要", recent)
        sum_title.setObjectName("navSummaryTitle")
        rl.addWidget(sum_title)
        self.sum_rows: dict[str, QLabel] = {}
        for key, init in (("状态", "就绪"), ("模式", "真机"), ("延迟", "—"),
                          ("上次", "—")):
            row = QWidget(recent)
            rwl = QHBoxLayout(row)
            rwl.setContentsMargins(0, 0, 0, 0)
            rwl.setSpacing(8)
            k = QLabel(key, row)
            k.setObjectName("navSummaryKey")
            v = QLabel(init, row)
            v.setObjectName("navSummaryVal")
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            rwl.addWidget(k)
            rwl.addStretch(1)
            rwl.addWidget(v)
            rl.addWidget(row)
            self.sum_rows[key] = v
        lay.addWidget(recent)

        # 命令面板行
        self.cmd_row = NavCommandRow(palette(self._theme_name), nav)
        lay.addWidget(self.cmd_row)

        # 页脚：主题分段 + 状态点 + 齿轮
        foot = QHBoxLayout()
        foot.setSpacing(6)
        self.sw_theme = ModeSwitch(["浅色", "深色"], nav, height=28)
        self.sw_theme.setCurrentIndex(1 if self._theme_name == "dark" else 0)
        self.sw_theme.currentIndexChanged.connect(
            lambda i: self._apply_theme("dark" if i == 1 else "light"))
        foot.addWidget(self.sw_theme, 1)
        lay.addLayout(foot)

        foot2 = QHBoxLayout()
        foot2.setSpacing(6)
        self.status_dot = StatusDot(palette(self._theme_name), "就绪", nav)
        foot2.addWidget(self.status_dot, 1)
        self.btn_gear = QPushButton(nav)
        self.btn_gear.setObjectName("navIconBtn")
        self.btn_gear.setIcon(ICONS.get("sliders", palette(self._theme_name)["ink2"], 15))
        self.btn_gear.setCursor(Qt.PointingHandCursor)
        foot2.addWidget(self.btn_gear)
        lay.addLayout(foot2)
        return nav

    def _nav_group(self, title: str, items: list[tuple[str, str, int]]) -> QWidget:
        pal = palette(self._theme_name)
        box = QWidget(self)
        vl = QVBoxLayout(box)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(1)
        header = NavGroupHeader(title, pal, box)
        vl.addWidget(header)
        body = QWidget(box)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 2)
        bl.setSpacing(1)
        rows: list[NavItem] = []
        for text, icon_name, page in items:
            item = NavItem(text, icon_name, page, body)
            item.set_icon_color(pal["ink2"])
            item.clicked.connect(lambda _=False, p=page: self._goto_page(p))
            bl.addWidget(item)
            rows.append(item)
            self.nav_items.append(item)
        vl.addWidget(body)
        header.toggled.connect(body.setVisible)
        self.nav_groups.append((header, rows))
        return box

    def _toggle_nav(self) -> None:
        collapsed = self.nav.width() > SIDE_COLLAPSED_W + 8
        w = SIDE_COLLAPSED_W if collapsed else SIDE_W
        self.nav.setMinimumWidth(w)
        self.nav.setMaximumWidth(w)
        for item in self.nav_items:
            item.setText("" if collapsed else f"  {item._label}")  # noqa: SLF001
            item.setToolTip(item._label if collapsed else "")      # noqa: SLF001

    # ── Inspector ─────────────────────────────────────
    def _build_inspector(self) -> QFrame:
        insp = QFrame(self)
        insp.setObjectName("inspectorPanel")
        insp.setMinimumWidth(INSP_W)
        insp.setMaximumWidth(INSP_W)
        lay = QVBoxLayout(insp)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)

        self.insp_title = QLabel("概览", insp)
        self.insp_title.setObjectName("inspTitle")
        self.insp_desc = QLabel("连接状态与双臂实时关节角", insp)
        self.insp_desc.setObjectName("inspSub")
        self.insp_desc.setWordWrap(True)
        lay.addWidget(self.insp_title)
        lay.addWidget(self.insp_desc)

        self.pages = QTabWidget(insp)
        self.pages.setObjectName("inspectorStack")
        self.pages.tabBar().hide()
        self.pages.setDocumentMode(True)
        self.pages.addTab(self._page_overview(), "概览")
        self.pages.addTab(self._page_robot(), "机器人")
        self.pages.addTab(self._page_scene(), "场景")
        self.pages.addTab(self._page_joints(), "关节")
        self.pages.addTab(self._page_plan(), "规划")
        self.pages.addTab(self._page_log(), "日志")
        lay.addWidget(self.pages, 1)
        return insp

    def _wrap_page(self, title: str, desc: str, cards: list[QWidget]) -> QWidget:
        page = QWidget(self)
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(8)
        for c in cards:
            vl.addWidget(c)
        vl.addStretch(1)
        c = QWidget()
        return page

    def _page_overview(self) -> QWidget:
        page = QWidget(self)
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(8)
        pal = self._pal_now

        card = GroupCard("运行状态", page)
        self.row_mode = Row("模式", value=True)
        self.row_link = Row("连接", value=False)
        self.row_ctrl = Row("控制器", value=False)
        self.row_rate = Row("状态频率", value=False)
        card.add(self.row_mode)
        card.add(GroupRule(card))
        card.add(self.row_link)
        card.add(GroupRule(card))
        card.add(self.row_ctrl)
        card.add(GroupRule(card))
        card.add(self.row_rate)
        vl.addWidget(card)

        card2 = GroupCard("安全", page)
        self.btn_check = QPushButton("启动前检查", card2)
        self.btn_check.setCursor(Qt.PointingHandCursor)
        self.btn_check.clicked.connect(self._on_precheck)
        card2.add(self.btn_check)
        hint = QLabel("启动会让硬件执行回零：所有关节被驱动到 0。"
                      "任一关节偏离零位超过 0.1 rad 时先手动摆正。", card2)
        hint.setObjectName("hintText")
        hint.setWordWrap(True)
        card2.add(hint)
        vl.addWidget(card2)
        vl.addStretch(1)
        return page

    def _page_robot(self) -> QWidget:
        page = QWidget(self)
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(8)

        card = GroupCard("接口", page)
        self.row_topic = Row("关节话题", value=False)
        self.row_ctrl_ns = Row("控制器", value=False)
        self.row_moveit = Row("MoveIt", value=False)
        card.add(self.row_topic)
        card.add(self.row_ctrl_ns)
        card.add(self.row_moveit)
        vl.addWidget(card)

        card2 = GroupCard("机器人", page)
        card2.add(Row("型号  OpenArm v10", value=True))
        card2.add(Row("自由度  7 × 2", value=True))
        card2.add(Row("夹爪  0.000 – 0.044 m", value=True))
        vl.addWidget(card2)
        vl.addStretch(1)
        return page

    def _page_scene(self) -> QWidget:
        page = QWidget(self)
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(8)

        card = GroupCard("视图", page)
        row = QHBoxLayout()
        row.setSpacing(4)
        for key, text in (("home", "复位"), ("front", "前"), ("side", "侧"), ("top", "顶")):
            b = ToolChip(text, card)
            b.clicked.connect(lambda _=False, k=key: self.view.set_view_preset(k))
            row.addWidget(b, 1)
        holder = QWidget(card)
        holder.setLayout(row)
        card.add(holder)
        vl.addWidget(card)

        card2 = GroupCard("显示", page)
        for label, desc, key, on in (
            ("线框", "只看网格骨架", "wireframe", False),
            ("碰撞体", "显示碰撞包围几何", "collision", False),
            ("半透明", "看穿本体内部结构", "transparent", False),
        ):
            sw = Switch(self._pal_now, on, card2)
            sw.toggled.connect(lambda v, k=key: self.view.set_overlay(k, v))
            self._switches.append(sw)
            card2.add(SettingsRow(label, sw, desc, card2))
            card2.add(GroupRule(card2))
        vl.addWidget(card2)
        vl.addStretch(1)
        return page

    def _page_joints(self) -> QWidget:
        page = QWidget(self)
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(8)

        card = GroupCard("目标关节角", page)
        self.combo_arm = QComboBox(card)
        self.combo_arm.addItems(["左臂 openarm_left", "右臂 openarm_right"])
        self.combo_arm.currentIndexChanged.connect(self._rebuild_sliders)
        card.add(self.combo_arm)

        self._slider_host = QWidget(card)
        self._slider_layout = QVBoxLayout(self._slider_host)
        self._slider_layout.setContentsMargins(0, 0, 0, 0)
        self._slider_layout.setSpacing(4)
        card.add(self._slider_host)
        vl.addWidget(card)

        card2 = GroupCard("操作", page)
        self.btn_zero = QPushButton("全部回零", card2)
        self.btn_zero.setCursor(Qt.PointingHandCursor)
        self.btn_zero.clicked.connect(self._zero_pose)
        note = QLabel("当前仅驱动 3D 孪生模型；真机下发在「启动」后开放。", card2)
        note.setObjectName("hintText")
        note.setWordWrap(True)
        card2.add(self.btn_zero)
        card2.add(note)
        vl.addWidget(card2)

        self._sliders: dict[str, tuple[QSlider, QLabel]] = {}
        self._rebuild_sliders()
        vl.addStretch(1)
        return page

    def _page_plan(self) -> QWidget:
        page = QWidget(self)
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(8)

        card = GroupCard("MoveIt 规划", page)
        self.combo_group = QComboBox(card)
        self.combo_group.addItems(["left_arm", "right_arm", "left_gripper", "right_gripper"])
        card.add(self.combo_group)
        self.combo_named = QComboBox(card)
        self.combo_named.addItems(["（选择命名状态）", "home", "hands_up", "open", "closed"])
        card.add(self.combo_named)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.btn_plan = QPushButton("规划", card)
        self.btn_plan.setCursor(Qt.PointingHandCursor)
        self.btn_exec = QPushButton("执行", card)
        self.btn_exec.setObjectName("primary")
        self.btn_exec.setCursor(Qt.PointingHandCursor)
        holder = QWidget(card)
        holder.setLayout(row)
        row.addWidget(self.btn_plan, 1)
        row.addWidget(self.btn_exec, 1)
        card.add(holder)

        self.lbl_plan = QLabel("MoveIt 未就绪（P4 阶段接入 /move_action）", card)
        self.lbl_plan.setObjectName("hintText")
        self.lbl_plan.setWordWrap(True)
        card.add(self.lbl_plan)
        vl.addWidget(card)
        vl.addStretch(1)
        return page

    def _page_log(self) -> QWidget:
        page = QWidget(self)
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(6)
        self.log_view = QPlainTextEdit(page)
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        vl.addWidget(self.log_view, 1)
        return page

    # ── 主工作区 ──────────────────────────────────────
    def _build_workspace(self) -> QWidget:
        card = QFrame(self)
        card.setObjectName("workspaceCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        split = QSplitter(Qt.Horizontal, card)
        split.setHandleWidth(8)
        split.setChildrenCollapsible(False)

        split.addWidget(self._build_status_card(split))
        split.addWidget(self._build_3d_card(split))
        split.setSizes([430, 700])
        lay.addWidget(split, 1)
        return card

    def _build_status_card(self, parent) -> QWidget:
        # pbvs 风格：工作区内区域透明，靠发丝线分区（不再叠卡片）
        host = QWidget(parent)
        host.setMinimumWidth(360)
        lay = QVBoxLayout(host)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(6)
        title = QLabel("状态", host)
        title.setObjectName("sectionHead")
        self.chip_src = QLabel("未连接", host)
        self.chip_src.setObjectName("statusChip")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.chip_src)
        lay.addLayout(head)

        strip = QFrame(host)
        strip.setObjectName("statBar")
        sl = QHBoxLayout(strip)
        sl.setContentsMargins(10, 8, 10, 9)
        sl.setSpacing(6)
        self.tile_left = StatTile("左臂", "--")
        self.tile_right = StatTile("右臂", "--")
        self.tile_grip = StatTile("夹爪", "--")
        for t in (self.tile_left, self.tile_right, self.tile_grip):
            sl.addWidget(t, 1)
        lay.addWidget(strip)

        self.tabs = QTabWidget(host)
        self.tabs.setDocumentMode(True)
        bar = AnimatedTabBar(self.tabs)
        self._tabbars.append(bar)
        self.tabs.setTabBar(bar)
        self.tabs.addTab(self._build_joint_table(host), "关节")
        self.tabs.addTab(self._build_log_tab(host), "日志")
        lay.addWidget(self.tabs, 1)
        return host

    def _build_joint_table(self, parent) -> QWidget:
        page = QWidget(parent)
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 6, 0, 0)
        vl.setSpacing(2)
        self.joint_rows: dict[str, QLabel] = {}
        for side, prefix in (("左臂", "openarm_left"), ("右臂", "openarm_right")):
            head = QLabel(side, page)
            head.setObjectName("groupLabel")
            vl.addWidget(head)
            for i in range(1, 8):
                name = f"{prefix}_joint{i}"
                row = QWidget(page)
                row.setFixedHeight(20)
                rl = QHBoxLayout(row)
                rl.setContentsMargins(2, 0, 2, 0)
                rl.setSpacing(6)
                lbl = QLabel(f"J{i}", row)
                lbl.setObjectName("jointName")
                lbl.setFixedWidth(24)
                val = QLabel("--", row)
                val.setObjectName("jointVal")
                val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                rl.addWidget(lbl)
                rl.addStretch(1)
                rl.addWidget(val)
                vl.addWidget(row)
                self.joint_rows[name] = val
        vl.addStretch(1)
        return page

    def _build_log_tab(self, parent) -> QWidget:
        page = QWidget(parent)
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 6, 0, 0)
        self.log_inline = QPlainTextEdit(page)
        self.log_inline.setReadOnly(True)
        self.log_inline.setMaximumBlockCount(1000)
        vl.addWidget(self.log_inline, 1)
        return page

    def _build_3d_card(self, parent) -> QWidget:
        # pbvs 风格：3D 无框融合——sceneFrame 一圈发丝边把画面圈住
        card = QFrame(parent)
        card.setObjectName("view3dCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        head = QHBoxLayout()
        head.setContentsMargins(6, 0, 6, 0)
        head.setSpacing(4)
        title = QLabel("3D 场景", card)
        title.setObjectName("sectionHead")
        self.chip_scene = QLabel("数字孪生", card)
        self.chip_scene.setObjectName("statusChip")
        head.addWidget(title)
        head.addWidget(self.chip_scene)
        head.addStretch(1)
        for key, text in (("home", "复位"), ("front", "前"), ("side", "侧"), ("top", "顶")):
            b = QPushButton(text, card)
            b.setObjectName("chip")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: self.view.set_view_preset(k))
            head.addWidget(b)
        lay.addLayout(head)

        frame = QFrame(card)
        frame.setObjectName("sceneFrame")
        flay = QVBoxLayout(frame)
        flay.setContentsMargins(0, 0, 0, 0)
        self.view = MujocoViewPanel(frame)
        flay.addWidget(self.view)
        lay.addWidget(frame, 1)

        # 底部工具条（pbvs 同款：视角 | 显示 | 叠加）
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 0, 6, 0)
        bar.setSpacing(3)
        lbl = QLabel("视角", card)
        lbl.setObjectName("toolLabel")
        bar.addWidget(lbl)
        for key, text in (("home", "复位"), ("front", "前"), ("side", "侧"), ("top", "顶")):
            b = ToolChip(text, card)
            b.clicked.connect(lambda _=False, k=key: self.view.set_view_preset(k))
            bar.addWidget(b)
        bar.addStretch(1)
        self.btn_wire_q = ToolChip("线框", card, checkable=True)
        self.btn_wire_q.toggled.connect(lambda on: self.view.set_overlay("wireframe", on))
        self.btn_col_q = ToolChip("碰撞体", card, checkable=True)
        self.btn_col_q.toggled.connect(lambda on: self.view.set_overlay("collision", on))
        lbl2 = QLabel("显示", card)
        lbl2.setObjectName("toolLabel")
        bar.addWidget(lbl2)
        bar.addWidget(self.btn_wire_q)
        bar.addWidget(self.btn_col_q)
        lay.addLayout(bar)

        self.joint_overlay = JointOverlay(self._joint_provider, palette(self._theme_name), self.view)
        self.joint_overlay.move(10, 10)
        self.joint_overlay.adjustSize()
        return card

    # ── 关节滑块 ──────────────────────────────────────
    def _rebuild_sliders(self) -> None:
        while self._slider_layout.count():
            item = self._slider_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._sliders.clear()
        prefix = "openarm_left" if self.combo_arm.currentIndex() == 0 else "openarm_right"
        names = [f"{prefix}_joint{i}" for i in range(1, 8)]
        names.append(f"{prefix}_finger_joint1")
        for i, name in enumerate(names):
            row = QWidget(self._slider_host)
            rl = QVBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(1)
            top = QHBoxLayout()
            top.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(f"J{i + 1}" if i < 7 else "夹爪", row)
            lbl.setObjectName("jointName")
            val = QLabel("+0.000", row)
            val.setObjectName("jointVal")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            top.addWidget(lbl)
            top.addStretch(1)
            top.addWidget(val)
            s = QSlider(Qt.Horizontal, row)
            if i < 7:
                s.setRange(-314, 314)   # ±3.14 rad
            else:
                s.setRange(0, 44)       # 0–0.044 m
            s.setValue(0)
            s.valueChanged.connect(lambda v, n=name: self._on_slider(n, v))
            rl.addLayout(top)
            rl.addWidget(s)
            self._slider_layout.addWidget(row)
            self._sliders[name] = (s, val)

    def _on_slider(self, name: str, raw: int) -> None:
        is_grip = "finger" in name
        val = raw / 1000.0 if is_grip else raw / 100.0
        pair = self._sliders.get(name)
        if pair:
            pair[1].setText(f"{val:+.3f}" if not is_grip else f"{val:.3f}")
        self._local_pose[name] = val
        if self._worker and not self._following_real():
            self._worker.set_pose(self._local_pose)

    def _zero_pose(self) -> None:
        for s, val in self._sliders.values():
            s.blockSignals(True)
            s.setValue(0)
            s.blockSignals(False)
            val.setText("+0.000")
        # 只清滑块涉及的关节，别把另一条臂的姿态也抹掉
        prefix = "openarm_left" if self.combo_arm.currentIndex() == 0 else "openarm_right"
        for name in list(self._local_pose):
            if name.startswith(prefix):
                self._local_pose.pop(name, None)
        if self._worker and not self._following_real():
            self._worker.set_pose(self._local_pose)

    def _following_real(self) -> bool:
        # 仿真模式下状态源是 MuJoCo 自己，不跟真机
        if self._is_sim:
            return False
        return self.bridge is not None and self.bridge.connected

    def _effective_pose(self) -> dict[str, float]:
        """仿真 → MuJoCo 物理状态；真机在线 → /joint_states；否则本地滑块姿态。"""
        if self._is_sim and self._worker is not None:
            pos = self._worker.positions()
            if pos:
                return pos
        if self._following_real():
            assert self.bridge is not None
            return self.bridge.positions()
        return dict(self._local_pose)

    def _joint_provider(self) -> dict[str, float]:
        return self._effective_pose()

    # ── 页面切换 ──────────────────────────────────────
    _PAGE_META = [
        ("概览", "连接状态与双臂实时关节角"),
        ("机器人", "ROS 2 接口与本体参数"),
        ("场景", "3D 视图取景与叠加"),
        ("关节", "拖动滑块驱动孪生模型"),
        ("规划", "MoveIt 2 轨迹规划"),
        ("日志", "运行日志"),
    ]

    def _goto_page(self, page: int) -> None:
        self.pages.setCurrentIndex(page)
        for b in self.nav_items:
            b.setChecked(b.page_index == page)
        title, desc = self._PAGE_META[page]
        self.insp_title.setText(title)
        self.insp_desc.setText(desc)
        mode = "仿真" if self.sw_mode.currentIndex() == 1 else "真机"
        self.crumb.setText(f"{mode} · {title}")

    # ── 主题 ──────────────────────────────────────────
    def _apply_theme(self, name: str, animate: bool = True) -> None:
        self._theme_name = name
        pal = palette(name)
        self._pal_now = pal
        self.setStyleSheet(build_qss(pal))
        ICONS.clear()
        self.win_btns.apply_palette(pal)
        self.view.apply_theme(pal)
        self.joint_overlay.apply_palette(pal)
        # 自绘组件要单独喂配色（QSS 管不到 QPainter）
        self.logo.set_palette_colors(pal)
        self.status_dot.set_palette_colors(pal)
        self.cmd_row.set_palette_colors(pal)
        self.btn_gear.setIcon(ICONS.get("sliders", pal["ink2"], 15))
        self.btn_reset.setIcon(ICONS.get("reset", pal["ink2"], 15))
        for header, rows in self.nav_groups:
            header.set_palette_colors(pal)
            for item in rows:
                item.set_icon_color(pal["ink2"])
        for sw in getattr(self, "_switches", []):
            sw.set_palette_colors(pal)
        for tabbar in getattr(self, "_tabbars", []):
            tabbar.set_palette_colors(pal)
        self.setProperty("theme", name)
        if self._worker:
            self._worker.set_theme(name)
        self._log(f"主题切换：{pal['label']}")

    # ── 运行/渲染 ─────────────────────────────────────
    def _start_render(self) -> None:
        cache = Path.home() / ".cache/openarm_studio"
        self._worker = RenderWorker(self.view, cache, self._theme_name,
                                    on_log=lambda m: self.log_sig.emit(m))
        self._worker.start()
        self._worker.set_pose({})

    def _start_bridge(self) -> None:
        """只读订阅 /joint_states；失败不影响 3D（此时用本地姿态）。"""
        self.bridge = RosBridge(on_log=lambda m: self.log_sig.emit(m))
        if self.bridge.start() and self.bridge._node is not None:  # noqa: SLF001
            self.js_pub = JointStatePublisher(self.bridge._node)  # noqa: SLF001
            self._log("仿真状态发布器就绪（仅仿真模式使用）")
    @property
    def _is_sim(self) -> bool:
        return self.sw_mode.currentIndex() == 1

    def _on_mode_change(self, idx: int) -> None:
        sim = idx == 1
        if self._worker is not None:
            self._worker.set_mode("sim" if sim else "twin")
            if sim:
                # 切到仿真时把目标对齐到当前姿态，避免一进去就甩过去
                self._worker.set_pose(self._worker.positions())
        self.chip_scene.setText("物理仿真" if sim else "数字孪生")
        self.badge_mode.setText("仿真" if sim else "数字孪生")
        self._log(f"模式：{'物理仿真（MuJoCo 重力和位置伺服在工作）' if sim else '数字孪生（跟随 /joint_states）'}")
        self._goto_page(self.pages.currentIndex())

    def showEvent(self, ev) -> None:  # noqa: N802
        super().showEvent(ev)
        if not getattr(self, "_intro_started", False):
            self._intro_started = True
            self.view.start_intro()

    def _on_precheck(self) -> None:
        pose = self._joint_provider()
        worst = max((abs(v) for k, v in pose.items() if "finger" not in k), default=0.0)
        if worst > 0.1:
            self._log(f"启动前检查：最大关节偏差 {worst:.3f} rad —— 超过 0.1，"
                      "启动会把关节驱动到零位，请先手动摆正")
        else:
            self._log(f"启动前检查：最大关节偏差 {worst:.3f} rad —— 可以启动")

    def _publish_sim_state(self) -> None:
        if not self._is_sim or self.js_pub is None or self._worker is None:
            return
        pos = self._worker.positions()
        if pos:
            self.js_pub.publish(pos)

    def _refresh_status(self) -> None:
        pose = self._effective_pose()
        following = self._following_real()
        sim = self._is_sim
        pub = getattr(self, "js_pub", None)

        if sim:
            # 仿真模式由上位机冒充 joint_state_broadcaster
            if pub is not None and pose:
                pub.publish(pose)
        elif self._worker is not None and following:
            # 跟真机时把姿态推给渲染线程（滑块姿态暂停生效）
            self._worker.set_pose(pose)

        left = [pose.get(f"openarm_left_joint{i}", 0.0) for i in range(1, 8)]
        right = [pose.get(f"openarm_right_joint{i}", 0.0) for i in range(1, 8)]
        gl = pose.get("openarm_left_finger_joint1", 0.0)
        gr = pose.get("openarm_right_finger_joint1", 0.0)
        self.tile_left.set_value(f"{np.mean(np.abs(left)):.3f} rad")
        self.tile_right.set_value(f"{np.mean(np.abs(right)):.3f} rad")
        self.tile_grip.set_value(f"{(gl + gr) / 2:.3f} m")
        for name, lbl in self.joint_rows.items():
            lbl.setText(f"{pose.get(name, 0.0):+.3f}")

        if sim:
            self.status_dot.set_state(self._pal_now["accent"], "仿真运行中", breathe=True)
            self.chip_src.setText("仿真")
            self.row_link.set_value("MuJoCo 物理")
            self.row_rate.set_value("1 kHz 物理")
        elif following:
            rate = self.bridge.rate if self.bridge else 0.0
            self.status_dot.set_state(self._pal_now["success"], "真机在线", breathe=False)
            self.chip_src.setText("真机")
            self.row_link.set_value("已连接")
            self.row_rate.set_value(f"{rate:.0f} Hz")
        else:
            self.status_dot.set_state(self._pal_now["warn"], "3D 运行中", breathe=False)
            self.chip_src.setText("未连接")
            self.row_link.set_value("未连接")
            self.row_rate.set_value("--")
        self.row_mode.set_value("仿真" if sim else "真机")
        self.sum_rows["状态"].setText(
            "仿真" if sim else ("跟随真机" if following else "本地姿态"))
        self.sum_rows["模式"].setText("仿真" if sim else "真机")
        self.sum_rows["延迟"].setText(
            f"{self.bridge.rate:.0f}Hz" if following and self.bridge else "—")
        self.sum_rows["上次"].setText(f"{self.view.frame_count}帧")

    # ── 日志 ─────────────────────────────────────────
    def _append_log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {text}"
        self.log_inline.appendPlainText(line)
        self.log_view.appendPlainText(line)

    def _log(self, text: str) -> None:
        self.log_sig.emit(text)

    # ── 关闭 ─────────────────────────────────────────
    def closeEvent(self, ev) -> None:  # noqa: N802
        self._status_timer.stop()
        if self.bridge is not None:
            self.bridge.stop()
            self.bridge = None
        if self._worker is not None:
            self._worker.stop()
            self._worker.join(timeout=3.0)
            self._worker = None
        super().closeEvent(ev)
