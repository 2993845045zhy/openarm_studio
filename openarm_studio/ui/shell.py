"""OpenArm Studio 主窗口 —— 三面板壳层（导航轨 / Inspector / 主卡）。

布局照 pbvs_workbench：左导航轨 236px → Inspector 300px → 右侧主卡，
面板间留 6px 缝隙露出外沿底色；无边框 + 圆角 18px。
主卡左列「状态」：统计瓦片 + 关节控制滑块 + 夹爪滑块（常住控制台）。

3D 渲染在独立线程里跑（EGL 上下文线程绑定），GUI 只在定时器里取帧。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import mujoco
import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel,
                               QMainWindow, QPlainTextEdit, QPushButton,
                               QScrollArea, QSlider, QSpinBox, QSplitter, QTabWidget,
                               QVBoxLayout, QWidget)

from ..core.mjview import MjEmbedRenderer
from ..core.sim import SimBackend
from ..model import scene as model_scene
from ..backend.driver_arm import (CONFIGS, GRIP_OPEN_M as GRIP_OPEN, DriverArm,
                                  MANUAL_JOINT_SPEED_RAD_S, SLIDER_JOINT_SPEED_LIMITS_RAD_S,
                                  can_ifstate, probe_bus)
from ..backend.teleop import BimanualTeleop, JointFeedback
from .chrome import DragHandle, ResizeHandles, WindowButtons
from .icons import ICONS
from .mjview_panel import MujocoViewPanel
from .switch import ModeSwitch
from .theme import build_qss, palette, scene_colors
from .widgets import GroupCard, Row, StatTile
from .zcode import (GroupRule, LogoButton, NavGroupHeader, NavItem,
                    SettingsRow, StatusDot, Switch, ToolChip)

SIDE_W = 236
SIDE_COLLAPSED_W = 56
INSP_W = 300

# 同一列里卡片之间的间距：全页面只认这一个值（状态列 / 概览 / 机器人 / 场景 / 日志）。
# 之前左列改成 12、页面还是 8，两列并排就能看出不齐。
CARD_GAP = 12


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
        self._metallic = True
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

    def ee_pose(self, side: str = "left"):
        """末端（hand）世界位姿：((x,y,z), (w,x,y,z))。"""
        with self._lock:
            model, data = self.model, self.data
        if model is None or data is None:
            return None, None
        name = f"openarm_{side}_hand"
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            return None, None
        pos = tuple(float(v) for v in data.xpos[bid])
        quat = tuple(float(v) for v in data.xquat[bid])
        return pos, quat

    def set_theme(self, theme_name: str) -> None:
        with self._lock:
            self._theme = theme_name
            self._rebuild = True

    def set_metallic(self, on: bool) -> None:
        """金属质感开关：改材质/灯光，需要重建模型（~0.2s）。"""
        with self._lock:
            self._metallic = bool(on)
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
        renderer = MjEmbedRenderer(height=640)
        try:
            while self._running:
                with self._lock:
                    theme = self._theme
                    rebuild = self._rebuild
                    metallic = self._metallic
                if rebuild or self.model is None:
                    sc = scene_colors(palette(theme))
                    self.model = model_scene.build_model(
                        self._cache, sc, metallic=metallic)
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
                # 1.5× 超采样：MuJoCo 默认无 MSAA，直接按面板尺寸渲染边缘
                # 锯齿明显（看着"分辨率低"）。放大渲染再平滑缩回等于抗锯齿。
                renderer.set_height(int(state.get("view_h", 720) * 1.5))
                rgb = renderer.render(self.model, render_data, state)
                self._panel.push_frame(rgb, wire=state.get("wireframe"))
                time.sleep(1 / 60)
        except Exception as exc:  # noqa: BLE001
            self._log(f"[3D] 渲染线程异常: {type(exc).__name__}: {exc}")
        finally:
            renderer.close()


# ────────────────────────────────────────────────────────────
# 主窗口
# ────────────────────────────────────────────────────────────
class StudioWindow(QMainWindow):
    log_sig = Signal(str)
    conn_sig = Signal(dict, str)
    reset_sig = Signal(bool)
    probe_sig = Signal(dict)
    travel_sig = Signal(int, str, bool)  # 走位代次、臂、结果

    def __init__(self, theme_name: str = "dark"):
        super().__init__()
        self._theme_name = theme_name
        self._reduced = False
        self._worker: RenderWorker | None = None
        # SDK 直连双臂：left→can1, right→can0（与 bringup 约定一致）
        self.arms: dict[str, DriverArm] = {}
        self._local_pose: dict[str, float] = {}
        self.nav_items: list[NavItem] = []
        self._switches: list[Switch] = []
        self._pal_now = palette(theme_name)
        self._conn_state = "disconnected"
        self._can_resetting = False
        self._cmd_pose: dict[str, float] = {}
        self._slider_target_sides: set[str] = set()

        self.setWindowTitle("OpenArm Studio")
        # 默认高度 1060：状态列（关节控制 + 力控含两个参数滑块 + 夹爪）
        # 自然高约 975，加上标题栏/边距正好一屏放下。
        self.resize(1560, 1060)
        self.setMinimumSize(1400, 800)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

        # 日志/连接信号先接好：构建与渲染线程的启动期消息也能进日志页
        self.log_sig.connect(self._append_log)
        self.conn_sig.connect(self._on_conn_result)
        self.reset_sig.connect(self._can_reset_done)
        self.probe_sig.connect(self._on_bus_result)
        self.travel_sig.connect(self._on_arm_travel_done)
        self._probing = False
        self._traveling = False
        self._disabling = False
        self._enabling = False
        self._travel_id = 0
        self._travel_arms: set[str] = set()
        self._travel_ok = True
        self._teleop_session: BimanualTeleop | None = None
        self._teleop_stop_reason: str | None = None
        self._grip_options: dict[str, tuple[QSpinBox, QPushButton, QPushButton, QLabel]] = {}
        self._ik = None
        self._ee_drag_on = False
        self._pending_sliders: set[str] = set()   # 已下发目标、等待实测追上
        # 示教录制状态（由「示教」页的开始/停止按钮驱动）
        self._recording = False
        self._rec_traj: list[list[float]] = []
        self._rec_t0 = 0.0
        # 示教回放状态
        self._play_rows: list[list[float]] = []
        self._play_live_sides: set[str] = set()
        self._after_disable: str | None = None
        self._estopping = False
        self._play_i = 0
        self._play_t0 = 0.0
        self._play_last_tick = 0.0
        self._play_dur = 0.0
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(20)
        self._play_timer.timeout.connect(self._play_tick)

        self._build_ui()
        self._apply_theme(theme_name, animate=False)
        self._goto_page(0)
        self._start_render()

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(200)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()
        # 3D 姿态推送 50Hz：状态刷新只有 5Hz，用它推渲染线程会显得"卡住"
        self._pose_timer = QTimer(self)
        self._pose_timer.setInterval(20)
        self._pose_timer.timeout.connect(self._push_pose)
        self._pose_timer.start()
        # 每 20ms 提交一次最新的真机目标。
        self._target_timer = QTimer(self)
        self._target_timer.setInterval(20)
        self._target_timer.timeout.connect(self._send_live_targets)
        self._target_timer.start()
        # 定期刷新双臂诊断状态。
        self._ctrl_timer = QTimer(self)
        self._ctrl_timer.setInterval(2000)
        self._ctrl_timer.timeout.connect(self._poll_controllers)
        self._ctrl_timer.start()

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

        # 连接（整宽主按钮，连接/断开切换）+ 急停
        self.btn_connect = QPushButton("连接", nav)
        self.btn_connect.setObjectName("btnPrimary")
        self.btn_connect.setCursor(Qt.PointingHandCursor)
        self.btn_connect.clicked.connect(self._on_connect_toggle)
        lay.addWidget(self.btn_connect)

        self.btn_estop = QPushButton("急停", nav)
        self.btn_estop.setObjectName("btnEstop")
        self.btn_estop.setCursor(Qt.PointingHandCursor)
        self.btn_estop.setEnabled(False)
        self.btn_estop.clicked.connect(self._on_estop)
        lay.addWidget(self.btn_estop)

        # 使能/失能：连接成功后才出现（真机专用；使能会上力矩）
        self.btn_enable = QPushButton("使能", nav)
        self.btn_enable.setObjectName("btnSecondary")
        self.btn_enable.setCursor(Qt.PointingHandCursor)
        self.btn_enable.setVisible(False)
        self.btn_enable.clicked.connect(self._on_enable_toggle)
        lay.addWidget(self.btn_enable)

        self.btn_retry_enable = QPushButton("重试未使能臂", nav)
        self.btn_retry_enable.setObjectName("btnSecondary")
        self.btn_retry_enable.setCursor(Qt.PointingHandCursor)
        self.btn_retry_enable.setToolTip("只对未使能的机械臂发起启动走位")
        self.btn_retry_enable.hide()
        self.btn_retry_enable.clicked.connect(self._on_retry_enable)
        lay.addWidget(self.btn_retry_enable)

        # 可折叠分组 + 图标导航行
        self.nav_groups: list[tuple[NavGroupHeader, list[NavItem]]] = []
        lay.addWidget(self._nav_group("设备", [
            ("概览", "robot", 0), ("机器人", "target", 1), ("场景", "layers", 2),
            ("示教", "play", 4)]))
        lay.addWidget(self._nav_group("诊断", [
            ("日志", "terminal", 3)]))
        lay.addStretch(1)

        # 双臂状态（真实数据：CAN 在线/轮询频率/使能态）
        hist_box = QWidget(nav)
        hl = QVBoxLayout(hist_box)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)
        self.hist_header = NavGroupHeader("双臂", palette(self._theme_name), hist_box)
        hl.addWidget(self.hist_header)
        self.hist_body = QWidget(hist_box)
        hbl = QVBoxLayout(self.hist_body)
        hbl.setContentsMargins(0, 0, 0, 0)
        hbl.setSpacing(0)
        self.ctrl_rows: dict[str, QLabel] = {}
        for side, short in (("left", "左臂"), ("right", "右臂")):
            row = QWidget(self.hist_body)
            rl = QHBoxLayout(row)
            rl.setContentsMargins(10, 0, 6, 0)
            rl.setSpacing(6)
            k = QLabel(short, row)
            k.setObjectName("historyTitle")
            v = QLabel("离线", row)
            v.setObjectName("historyTime")
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            rl.addWidget(k)
            rl.addStretch(1)
            rl.addWidget(v)
            row.setFixedHeight(26)
            hbl.addWidget(row)
            self.ctrl_rows[side] = v
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

        # 页脚：主题分段 + 状态点
        foot = QHBoxLayout()
        foot.setSpacing(6)
        self.sw_theme = ModeSwitch(["浅色", "深色"], nav)
        self.sw_theme.setCurrentIndex(1 if self._theme_name == "dark" else 0)
        self.sw_theme.currentIndexChanged.connect(
            lambda i: self._apply_theme("dark" if i == 1 else "light"))
        foot.addWidget(self.sw_theme, 1)
        lay.addLayout(foot)

        self.status_dot = StatusDot(palette(self._theme_name), "未连接", nav)
        lay.addWidget(self.status_dot)
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
        self.pages.addTab(self._page_log(), "日志")
        self.pages.addTab(self._page_teach(), "示教")
        lay.addWidget(self.pages, 1)
        return insp

    def _wrap_page(self, title: str, desc: str, cards: list[QWidget]) -> QWidget:
        page = QWidget(self)
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(CARD_GAP)
        for c in cards:
            vl.addWidget(c)
        vl.addStretch(1)
        c = QWidget()
        return page

    def _page_overview(self) -> QWidget:
        page = QWidget(self)
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(CARD_GAP)
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

        card_cfg = GroupCard("驱动配置", page)
        cfg_row = QHBoxLayout()
        cfg_row.setContentsMargins(0, 0, 0, 0)
        cfg_row.setSpacing(8)
        lbl_cfg = QLabel("openarm_driver", card_cfg)
        lbl_cfg.setObjectName("fieldLabel")
        self.combo_config = QComboBox(card_cfg)
        self.combo_config.addItems(list(CONFIGS))
        cfg_row.addWidget(lbl_cfg)
        cfg_row.addWidget(self.combo_config, 1)
        cfg_holder = QWidget(card_cfg)
        cfg_holder.setLayout(cfg_row)
        card_cfg.add(cfg_holder)
        vl.addWidget(card_cfg)

        # 总线体检：不动硬件的只读探测，连接前后都能用
        card_bus = GroupCard("总线体检", page)
        bus_row = QHBoxLayout()
        bus_row.setContentsMargins(0, 0, 0, 0)
        bus_row.setSpacing(6)
        self.btn_bus_check = QPushButton("体检", card_bus)
        self.btn_bus_check.setObjectName("btnSecondary")
        self.btn_bus_check.setCursor(Qt.PointingHandCursor)
        self.btn_bus_check.clicked.connect(self._on_bus_check)
        self.btn_can_reset = QPushButton("CAN 重连", card_bus)
        self.btn_can_reset.setObjectName("btnSecondary")
        self.btn_can_reset.setCursor(Qt.PointingHandCursor)
        self.btn_can_reset.clicked.connect(self._on_can_reset)
        bus_row.addWidget(self.btn_bus_check)
        bus_row.addWidget(self.btn_can_reset, 1)
        bus_holder = QWidget(card_bus)
        bus_holder.setLayout(bus_row)
        card_bus.add(bus_holder)
        self.row_bus_r = Row("右臂 can0", value=False)
        self.row_bus_r.set_value("—")
        self.row_bus_l = Row("左臂 can1", value=False)
        self.row_bus_l.set_value("—")
        card_bus.add(self.row_bus_r)
        card_bus.add(GroupRule(card_bus))
        card_bus.add(self.row_bus_l)
        vl.addWidget(card_bus)

        card2 = GroupCard("安全", page)
        self.btn_check = QPushButton("启动前检查", card2)
        self.btn_check.setCursor(Qt.PointingHandCursor)
        self.btn_check.clicked.connect(self._on_precheck)
        card2.add(self.btn_check)
        vl.addWidget(card2)
        vl.addStretch(1)
        return page

    def _page_robot(self) -> QWidget:
        page = QWidget(self)
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(CARD_GAP)

        card = GroupCard("CAN 总线", page)
        self.row_can_r = Row("右臂", value=False)
        self.row_can_r.set_value("can0")
        self.row_can_l = Row("左臂", value=False)
        self.row_can_l.set_value("can1")
        self.row_bus = Row("总线", value=False)
        self.row_bus.set_value("CAN-FD")
        card.add(self.row_can_r)
        card.add(GroupRule(card))
        card.add(self.row_can_l)
        card.add(GroupRule(card))
        card.add(self.row_bus)
        vl.addWidget(card)

        card2 = GroupCard("电机温度", page)
        self.row_temp_l = Row("左臂 MOS/转子", value=False)
        self.row_temp_l.set_value("—")
        self.row_temp_r = Row("右臂 MOS/转子", value=False)
        self.row_temp_r.set_value("—")
        card2.add(self.row_temp_r)
        card2.add(GroupRule(card2))
        card2.add(self.row_temp_l)
        vl.addWidget(card2)

        card3 = GroupCard("总线健康", page)
        self.row_health_r = Row("右臂", value=False)
        self.row_health_r.set_value("—")
        self.row_health_l = Row("左臂", value=False)
        self.row_health_l.set_value("—")
        self.row_safety = Row("安全锁存", value=False)
        self.row_safety.set_value("—")
        card3.add(self.row_health_r)
        card3.add(GroupRule(card3))
        card3.add(self.row_health_l)
        card3.add(GroupRule(card3))
        card3.add(self.row_safety)
        vl.addWidget(card3)

        card4 = GroupCard("夹爪", page)
        self.row_grip_travel = Row("行程", value=False)
        self.row_grip_travel.set_value("0–44 mm")
        card4.add(self.row_grip_travel)
        vl.addWidget(card4)
        vl.addStretch(1)
        return page

    def _page_scene(self) -> QWidget:
        page = QWidget(self)
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(CARD_GAP)

        card = GroupCard("视图", page)
        row = QHBoxLayout()
        row.setSpacing(4)
        for key, text in (("home", "复位"), ("front", "前"), ("side", "侧"), ("top", "顶")):
            b = ToolChip(text, card)
            b.clicked.connect(lambda _=False, k=key: self._set_view_preset(k))
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

        # 金属质感：改材质反射率 + 灯光高光，需要重建模型（~0.2s）
        sw_metal = Switch(self._pal_now, True, card2)
        sw_metal.toggled.connect(self._on_metallic_toggle)
        self._switches.append(sw_metal)
        card2.add(SettingsRow("金属质感", sw_metal,
                              "银色件顶格高光 + 影棚打光", card2))
        vl.addWidget(card2)
        vl.addStretch(1)
        return page

    def _on_metallic_toggle(self, on: bool) -> None:
        if self._worker is not None:
            self._worker.set_metallic(on)
        self._log("3D 材质：金属质感" + ("开启（重建模型）" if on else "关闭"))

    def _set_view_preset(self, key: str) -> None:
        """视角预设（来自 Inspector「场景 → 视图」按钮）。"""
        self.view._cancel_intro()
        self.view.set_view_preset(key)

    # ── 页面切换 ──────────────────────────────────────
    _PAGE_META = [
        ("概览", "连接状态与驱动配置"),
        ("机器人", "CAN 总线与本体参数"),
        ("场景", "3D 视图取景与叠加"),
        ("日志", "运行日志"),
        ("示教", "录制与回放关节轨迹"),
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
        # 自绘组件要单独喂配色（QSS 管不到 QPainter）
        self.logo.set_palette_colors(pal)
        self.status_dot.set_palette_colors(pal)
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

    @property
    def _connected(self) -> bool:
        return self._conn_state == "connected"

    def _set_conn_state(self, state: str, text: str) -> None:
        self._conn_state = state
        self.btn_connect.setText({
            "disconnected": "连接", "connecting": "连接中…", "connected": "断开",
        }.get(state, "连接"))
        self.btn_connect.setEnabled(state != "connecting" and not self._power_motion_pending)
        self.btn_estop.setEnabled(state == "connected" and not self._is_sim)
        self.sum_rows["状态"].setText(text)
        color = {"disconnected": self._pal_now["muted"],
                 "connecting": self._pal_now["warn"],
                 "connected": self._pal_now["success"]}[state]
        self.status_dot.set_state(color, text, breathe=(state == "connecting"))

    def _on_connect_toggle(self) -> None:
        if self._conn_state == "connected":
            self._disconnect()
        else:
            self._connect()

    def _connect(self) -> None:
        """连接：仿真=本地 MuJoCo（不碰 CAN）；真机=SDK 直连 can0/can1。"""
        if self._conn_state != "disconnected":
            return
        if self._probing and not self._is_sim:
            self._log("请等待总线体检结束后再连接")
            return
        if self._is_sim:
            self._set_conn_state("connected", "仿真已就绪")
            self._log("仿真模式：MuJoCo 本地物理，无需连接硬件")
            return
        self._set_conn_state("connecting", "连接中…")
        config_name = self.combo_config.currentText()
        threading.Thread(target=self._connect_worker, args=(config_name,), daemon=True,
                         name="sdk-connect").start()

    def _connect_worker(self, config_name: str) -> None:
        """后台连两条臂（打开 CAN + 轮询线程），成功后**不自动使能**。

        0 应答的总线在 DriverArm.connect 里直接判失败（不再出现
        "已连接"假象），这里按臂汇总结果。
        """
        names = {"right": "右臂", "left": "左臂"}
        connected, failed = {}, []
        for side, can_if in (("right", "can0"), ("left", "can1")):
            arm = DriverArm(side, can_if, config_name=config_name, on_log=self.log_sig.emit)
            if arm.connect():
                connected[side] = arm
            else:
                failed.append(side)
        if connected and not failed:
            text = "已连接（双臂）"
        elif connected:
            text = f"{'、'.join(names[side] for side in connected)}已连接；另一臂连接失败"
        else:
            text = "连接失败（总线无应答，可用「总线体检」定位）"
        self.conn_sig.emit(connected, text)

    def _on_conn_result(self, arms: dict[str, DriverArm], text: str) -> None:
        self.arms.update(arms)
        if arms:
            self._set_conn_state("connected", "已连接")
            for side, arm in arms.items():
                arm.set_friction_comp(self.sw_friction.isChecked())
                arm.set_gripper_force(self._grip_options[side][0].value())
                self._send_force_mode(arm, self._selected_force_mode())
            self._log(f"{text}：点「使能」后带重力前馈走到起始位")
            self.btn_enable.setText("使能")
            self.btn_enable.setVisible(not self._is_sim)
        else:
            self._set_conn_state("disconnected", "未连接")
            self._log(text)

    def _has_enabled_arms(self) -> bool:
        return any(arm.connected and arm.enabled for arm in self.arms.values())

    def _disconnect(self) -> None:
        if self._power_motion_pending or self._estopping:
            return
        if self._has_enabled_arms():
            self._after_disable = "disconnect"
            self._on_enable_toggle()
            return
        self._stop_teleop("断开连接")
        self._set_conn_state("connecting", "断开中…")
        self._stop_teach_play("断开连接")
        if self._recording:
            self._stop_record()             # 断开前把已录的帧落盘
        for side, arm in list(self.arms.items()):
            arm.disconnect()
            if not arm.connected:
                del self.arms[side]
        self._after_travel(False)
        self._clear_motion_targets()
        self._set_force_switches("position")
        if self.arms:
            self._set_conn_state("connected", "驱动正在退出")
            self._log("控制线程尚未退出，请稍后重试断开")
        else:
            self._set_conn_state("disconnected", "未连接")
            self.btn_enable.setVisible(False)
            self._log("已断开")
        self._update_power_controls()

    def _on_estop(self) -> None:
        """急停：立即 disable_all，无回程轨迹。真机注意手臂会失去力矩下垂。"""
        if not self.arms:
            self._log("急停：未连接")
            return
        for arm in self.arms.values():
            arm.estop()
        self._stop_teleop("急停")
        self._stop_teach_play("急停", hold=False)
        self._clear_motion_targets()
        self._travel_id += 1
        self._travel_arms.clear()
        self._enabling = self._disabling = self._traveling = False
        self._after_disable = None
        self._estopping = True
        self._set_force_switches("position")
        if self._recording:
            self._stop_record()
        self._set_conn_state("connected", "急停")
        self._update_power_controls()
        self._log("已请求急停")

    @property
    def _power_motion_pending(self) -> bool:
        return self._enabling or self._disabling

    def _on_enable_toggle(self) -> None:
        if self._power_motion_pending or self._estopping or self._is_sim:
            return
        live = {side: arm for side, arm in self.arms.items() if arm.connected}
        if not live:
            return
        if not any(arm.enabled for arm in live.values()):
            self._begin_enable(live)
            return
        self.sw_ee_drag.setChecked(False)
        self._clear_motion_targets(hold=True)
        self._stop_teleop("失能")
        self._stop_teach_play("失能")
        if self._recording:
            self._stop_record()
        self._disabling = True
        self._update_power_controls()
        self._log("开始失能回程，已开启的重力前馈继续工作")
        self._start_arm_travel(live, "disable")

    def _arms_to_retry(self) -> dict[str, DriverArm]:
        if (not self._connected or self._is_sim or self._power_motion_pending
                or self._traveling or self._estopping):
            return {}
        live = {side: arm for side, arm in self.arms.items() if arm.connected}
        waiting = {side: arm for side, arm in live.items() if not arm.enabled}
        return waiting if len(live) == 2 and len(waiting) == 1 else {}

    def _on_retry_enable(self) -> None:
        waiting = self._arms_to_retry()
        if waiting:
            self._begin_enable(waiting, retry=True)

    def _begin_enable(self, arms: dict[str, DriverArm], *, retry: bool = False) -> None:
        if self._probing:
            self._log("请等待总线体检结束后再使能")
            return
        self._stop_teleop("使能")
        self._stop_teach_play("使能")
        self.sw_ee_drag.setChecked(False)
        self._clear_motion_targets(hold=True)
        mode = self._selected_force_mode()
        was_position = mode == "position"
        if was_position:
            mode = "gravity"
            self._set_force_switches(mode)
        if retry:
            # 启动失败会清掉该臂的驱动模式，重试前重新提交当前力控设置。
            # 若原先是位置保持，双臂都对齐到使能默认的重力补偿。
            for source in (self.arms if was_position else arms).values():
                if source.connected:
                    self._send_force_mode(source, mode)
        self._enabling = True
        self._update_power_controls()
        labels = "、".join("左臂" if side == "left" else "右臂" for side in arms)
        self._log(f"{labels}开始使能，根据实测姿态建立重力补偿，再走向起始位")
        self._start_arm_travel(arms, "enable")

    def _update_power_controls(self) -> None:
        busy = self._power_motion_pending
        self.btn_enable.setEnabled(not busy and not self._estopping)
        self.btn_retry_enable.setVisible(bool(self._arms_to_retry()))
        self.btn_connect.setEnabled(not busy and not self._estopping
                                    and self._conn_state != "connecting")
        self.sw_mode.setEnabled(not busy and not self._estopping)
        self._refresh_teleop_controls()
        if self._estopping:
            self.btn_enable.setText("急停中…")
        elif self._enabling:
            self.btn_enable.setText("使能走位中…")
        elif self._disabling:
            self.btn_enable.setText("失能回程中…")
        else:
            self.btn_enable.setText("失能" if any(a.enabled for a in self.arms.values()) else "使能")
        self._refresh_impedance_status()

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
        split.setSizes([420, 720])
        lay.addWidget(split, 1)
        return card

    def _build_status_card(self, parent) -> QWidget:
        host = QWidget(parent)
        host.setMinimumWidth(340)
        # 滑块与力矩读数都按关节名登记；_rebuild_sliders 会重建前者
        self._sliders: dict[str, tuple[QSlider, QLabel]] = {}
        self._torque_labels: dict[str, QLabel] = {}
        lay = QVBoxLayout(host)
        lay.setContentsMargins(8, 6, 8, 8)
        # 卡片之间的呼吸感：与页面其它列共用 CARD_GAP，避免两列并排时疏密不一
        lay.setSpacing(CARD_GAP)
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

        # 关节控制（常住）：选臂 + 7 关节滑块 + 全部回零
        lay.addWidget(self._build_teleop_card(host))

        self._joint_controls = jcard = GroupCard("关节控制", host)
        self.combo_arm = QComboBox(jcard)
        self.combo_arm.addItems(["左臂 openarm_left", "右臂 openarm_right"])
        self.combo_arm.currentIndexChanged.connect(self._rebuild_sliders)
        jcard.add(self.combo_arm)
        self._slider_host = QWidget(jcard)
        self._slider_layout = QVBoxLayout(self._slider_host)
        self._slider_layout.setContentsMargins(0, 0, 0, 0)
        self._slider_layout.setSpacing(4)
        jcard.add(self._slider_host)

        pose_row = QHBoxLayout()
        pose_row.setContentsMargins(0, 0, 0, 0)
        pose_row.setSpacing(6)
        self.btn_zero = QPushButton("全部回零", jcard)
        self.btn_zero.setObjectName("btnSecondary")
        self.btn_zero.setCursor(Qt.PointingHandCursor)
        self.btn_zero.clicked.connect(self._zero_pose)
        self.btn_goto_start = QPushButton("回起始位", jcard)
        self.btn_goto_start.setObjectName("btnSecondary")
        self.btn_goto_start.setCursor(Qt.PointingHandCursor)
        self.btn_goto_start.clicked.connect(lambda: self._on_move_pose("goto_start"))
        pose_row.addWidget(self.btn_zero, 1)
        pose_row.addWidget(self.btn_goto_start, 1)
        pose_actions = QWidget(jcard)
        pose_actions.setLayout(pose_row)
        jcard.add(pose_actions)

        # 末端拖拽（官方 openarm_control IK，基于官方 v1 MuJoCo 模型）
        self.sw_ee_drag = Switch(self._pal_now, False, jcard)
        self.sw_ee_drag.toggled.connect(self._on_ee_drag_toggle)
        self._switches.append(self.sw_ee_drag)
        self.sw_ee_drag.setToolTip("在 3D 里拖动末端标记，松手后逆解到关节")
        srow = QHBoxLayout(); srow.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel("末端拖拽", jcard); lbl.setObjectName("settingTitle")
        srow.addWidget(lbl); srow.addStretch(1); srow.addWidget(self.sw_ee_drag)
        sh = QWidget(jcard); sh.setLayout(srow); jcard.add(sh)
        lay.addWidget(jcard)

        # 力控：重力/摩擦/示教/导纳集中一张卡（放关节卡里会把左列挤爆）
        fcard = GroupCard("力控", host)
        self.sw_gravity = self._switch_row(
            fcard, "重力补偿", "前馈重力矩（官方 v1 MuJoCo 模型）")
        self.sw_gravity.toggled.connect(self._on_gravity_toggle)

        # 关节阻抗（整条臂）：7 个关节同时按"刚度比例 + 阻尼比"给 PD，
        # 前馈仍是 G+fric。刚度和阻尼比在下面两行滑块里调。
        self.sw_jimp = self._switch_row(
            fcard, "关节阻抗",
            "整臂关节空间阻抗：刚度低＝推着软、阻尼比小＝回弹过冲大")
        self.sw_jimp.toggled.connect(self._on_jimp_toggle)

        # 关节阻抗的两个连续参数：刚度（Kp 比例）与阻尼比。
        # 用滑块而不是档位——"阻抗感"很主观，四档根本调不到位。
        self.sld_stiff = QSlider(Qt.Horizontal, fcard)
        self.sld_stiff.setRange(1, 100)
        self.sld_stiff.setValue(35)
        self.sld_stiff.valueChanged.connect(self._on_imp_param)
        self.sld_stiff.setEnabled(False)
        self.lbl_stiff = QLabel("35%", fcard)
        self.lbl_stiff.setObjectName("jointVal")
        self.lbl_stiff.setFixedWidth(42)
        self.lbl_stiff.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        fcard.add(self._param_row(fcard, "刚度", self.sld_stiff, self.lbl_stiff))

        self.sld_zeta = QSlider(Qt.Horizontal, fcard)
        self.sld_zeta.setRange(20, 200)          # 0.20 ~ 2.00
        self.sld_zeta.setValue(100)
        self.sld_zeta.valueChanged.connect(self._on_imp_param)
        self.sld_zeta.setEnabled(False)
        self.lbl_zeta = QLabel("1.00", fcard)
        self.lbl_zeta.setObjectName("jointVal")
        self.lbl_zeta.setFixedWidth(42)
        self.lbl_zeta.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        fcard.add(self._param_row(fcard, "阻尼", self.sld_zeta, self.lbl_zeta))

        self.lbl_gain = QLabel("关节阻抗未开启", fcard)
        self.lbl_gain.setWordWrap(True)
        self.lbl_gain.setToolTip("J1–J7 最近提交给 CAN 驱动的增益。可在上方切换左右臂。")
        self.lbl_gain.setObjectName("jointHint")
        fcard.add(self.lbl_gain)
        self._on_imp_param()          # 初始把数值标签填上

        self.sw_friction = self._switch_row(
            fcard, "摩擦补偿", "根据各关节的实测速度补偿摩擦，"
                              "静止时抑制噪声，反向时撤掉旧方向的助力")
        self.sw_friction.setChecked(True)
        self.sw_friction.toggled.connect(self._on_friction_toggle)

        self.sw_teach = self._switch_row(
            fcard, "示教（零重力）", "零重力手推机械臂；录制请到「示教」页")
        self.sw_teach.toggled.connect(self._on_teach_toggle)
        lay.addWidget(fcard)

        # 夹爪控制（常住）：左右滑块（0–44 mm）
        self._gripper_controls = gcard = GroupCard("夹爪控制", host)
        gcard.add(self._build_grip_row(gcard, "left", "左夹爪"))
        gcard.add(GroupRule(gcard))
        gcard.add(self._build_grip_row(gcard, "right", "右夹爪"))
        lay.addWidget(gcard)

        self._rebuild_sliders()
        self._refresh_teleop_controls()
        lay.addStretch(1)
        # 左列内容会随窗口高度变化（1366×768 这类屏放不下全部卡片），
        # 包一层滚动区：宁可滚动，也不要把卡片裁掉半截。
        scroll = QScrollArea(parent)
        scroll.setObjectName("statusScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # viewport 默认按调色板 Base 自行填底（深色主题下也是白），QSS 光给
        # QScrollArea 设透明管不到它，这里再显式关掉自动填充。
        scroll.viewport().setAutoFillBackground(False)
        # 宽度下限放在滚动区上：放进 widget 里的话滚动区可以把自己压窄，
        # 内容会被横向裁掉（1400 宽窗口实测只给到 286px）。
        scroll.setMinimumWidth(354)
        scroll.setWidget(host)
        return scroll

    def _build_teleop_card(self, parent) -> GroupCard:
        card = GroupCard("双臂协同", parent)
        self.combo_teleop = QComboBox(card)
        self.combo_teleop.addItem("左臂手推 → 右臂跟随", "left")
        self.combo_teleop.addItem("右臂手推 → 左臂跟随", "right")
        self.btn_teleop = QPushButton("开始跟随", card)
        self.btn_teleop.setObjectName("btnSecondary")
        self.btn_teleop.clicked.connect(self._toggle_teleop)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.combo_teleop, 1)
        row.addWidget(self.btn_teleop)
        holder = QWidget(card)
        holder.setLayout(row)
        card.add(holder)
        self.sw_teleop_mirror = self._switch_row(
            card, "镜像跟随", "开启时左右镜像。关闭时按底座坐标系的零位轴方向映射关节增量。")
        self.sw_teleop_mirror.setChecked(True)
        self.lbl_teleop = QLabel(card)
        self.lbl_teleop.setObjectName("jointHint")
        self.lbl_teleop.setWordWrap(True)
        card.add(self.lbl_teleop)
        return card

    def _switch_row(self, card: GroupCard, label: str, tip: str) -> Switch:
        """在分组卡里加一行「标签 + 开关」，返回开关。"""
        sw = Switch(self._pal_now, False, card)
        self._switches.append(sw)
        sw.setToolTip(tip)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label, card)
        lbl.setObjectName("settingTitle")
        row.addWidget(lbl)
        row.addStretch(1)
        row.addWidget(sw)
        holder = QWidget(card)
        holder.setLayout(row)
        card.add(holder)
        return sw

    def _param_row(self, card: GroupCard, label: str, sld: QSlider,
                   value: QLabel) -> QWidget:
        """一行「标签 + 滑块 + 数值」，给力控参数用（省纵向空间）。"""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        lbl = QLabel(label, card)
        lbl.setObjectName("jointName")
        lbl.setFixedWidth(30)
        row.addWidget(lbl)
        row.addWidget(sld, 1)
        row.addWidget(value)
        holder = QWidget(card)
        holder.setLayout(row)
        return holder

    def _build_grip_row(self, parent, side: str, label: str) -> QWidget:
        row = QWidget(parent)
        rl = QVBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(2)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label, row)
        lbl.setObjectName("jointName")
        val = QLabel("0 mm", row)
        val.setObjectName("jointVal")
        val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        tq = QLabel("--", row)
        tq.setObjectName("jointTorque")
        tq.setFixedWidth(46)
        tq.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        tq.setToolTip("电机回报力矩（Nm）")
        tq.setProperty("warn", False)
        top.addWidget(lbl)
        top.addStretch(1)
        top.addWidget(val)
        top.addSpacing(8)
        top.addWidget(tq)
        sld = QSlider(Qt.Horizontal, row)
        sld.setRange(0, int(GRIP_OPEN * 1000))
        name = f"openarm_{side}_finger_joint1"
        sld.valueChanged.connect(lambda v, n=name: self._on_slider(n, v))
        rl.addLayout(top)
        rl.addWidget(sld)
        self._sliders[name] = (sld, val)
        self._torque_labels[name] = tq

        controls = QHBoxLayout()
        limit = QSpinBox(row)
        limit.setRange(10, 100)
        limit.setValue(60)
        limit.setSuffix(" %")
        limit.setToolTip(
            "夹持电流上限，100% = 设备配置上限：实测堵转约 2.1 N·m（指尖约 40–50 N），"
            "60% ≈ 1.3 N·m（约 20–30 N）。机构摩擦实测 0.24–0.53 N·m，低于约 40% 时"
            "夹爪可能半途卡住；智能抓取最低按 40% 执行。")
        limit.valueChanged.connect(lambda value, side=side: self._on_gripper_force(side, value))
        grasp = QPushButton("智能抓取", row)
        grasp.setObjectName("btnSecondary")
        grasp.setToolTip(
            "夹爪以约 21 mm/s 闭合到物体上，夹住后按上方电流上限持续夹持，"
            "拖开口滑块或点「张开」才松开。夹爪全闭仍未碰到物体时提示未夹到。")
        grasp.clicked.connect(lambda _checked=False, side=side: self._on_grasp(side))
        release = QPushButton("张开", row)
        release.setObjectName("btnSecondary")
        release.clicked.connect(lambda _checked=False, n=name: self._on_slider(n, int(GRIP_OPEN * 1000)))
        status = QLabel(row)
        status.setObjectName("jointHint")
        status.setWordWrap(True)
        controls.addWidget(QLabel("电流上限", row))
        controls.addWidget(limit)
        controls.addStretch(1)
        controls.addWidget(grasp)
        controls.addWidget(release)
        rl.addLayout(controls)
        rl.addWidget(status)
        self._grip_options[side] = limit, grasp, release, status
        return row

    def _build_3d_card(self, parent) -> QWidget:
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
        lay.addLayout(head)

        frame = QFrame(card)
        frame.setObjectName("sceneFrame")
        flay = QVBoxLayout(frame)
        flay.setContentsMargins(0, 0, 0, 0)
        self.view = MujocoViewPanel(frame)
        flay.addWidget(self.view)
        # 视口左下角不再放视角胶囊（用户要求删掉）；切换视角走
        # Inspector「场景 → 视图」里那排按钮，或直接鼠标拖相机。
        lay.addWidget(frame, 1)
        return card

    def _page_log(self) -> QWidget:
        page = QWidget(self)
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(CARD_GAP)
        self.log_view = QPlainTextEdit(page)
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        vl.addWidget(self.log_view, 1)
        return page

    def _page_teach(self) -> QWidget:
        """示教页：录制（显式开始/停止）+ 回放（选文件播放）。"""
        page = QWidget(self)
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(CARD_GAP)

        # ── 录制 ──
        card = GroupCard("示教录制", page)
        self.lbl_rec = QLabel("空闲 · 未录制", card)
        self.lbl_rec.setObjectName("settingTitle")
        self.lbl_rec.setWordWrap(True)
        card.add(self.lbl_rec)

        self.btn_record = QPushButton("开始录制", card)
        self.btn_record.setObjectName("btnPrimary")
        self.btn_record.setCursor(Qt.PointingHandCursor)
        self.btn_record.setMinimumHeight(32)
        self.btn_record.clicked.connect(self._on_record_toggle)
        card.add(self.btn_record)

        hint = QLabel("50Hz 记录双臂关节与夹爪。先在「力控」里开「示教（零重力）」，"
                      "再点这里开始，推完点停止即存盘。", card)
        hint.setObjectName("jointHint")
        hint.setWordWrap(True)
        card.add(hint)
        vl.addWidget(card)

        # ── 回放 ──
        card2 = GroupCard("示教回放", page)
        self.combo_teach = QComboBox(card2)
        self.combo_teach.setToolTip("records/teach_*.csv")
        card2.add(self.combo_teach)

        prow = QHBoxLayout()
        prow.setContentsMargins(0, 0, 0, 0)
        prow.setSpacing(6)
        self.btn_teach_play = QPushButton("播放", card2)
        self.btn_teach_play.setObjectName("btnSecondary")
        self.btn_teach_play.setCursor(Qt.PointingHandCursor)
        self.btn_teach_play.clicked.connect(self._on_teach_play)
        self.btn_teach_refresh = QPushButton("刷新列表", card2)
        self.btn_teach_refresh.setObjectName("btnSecondary")
        self.btn_teach_refresh.setCursor(Qt.PointingHandCursor)
        self.btn_teach_refresh.clicked.connect(self._refresh_teach_files)
        prow.addWidget(self.btn_teach_play, 1)
        prow.addWidget(self.btn_teach_refresh, 1)
        ph = QWidget(card2)
        ph.setLayout(prow)
        card2.add(ph)

        self.lbl_teach = QLabel("回放 --", card2)
        self.lbl_teach.setObjectName("jointHint")
        self.lbl_teach.setWordWrap(True)
        card2.add(self.lbl_teach)
        rhint = QLabel("使能时驱动真机；未使能只推 3D 孪生。播完停在末帧保持。", card2)
        rhint.setObjectName("jointHint")
        rhint.setWordWrap(True)
        card2.add(rhint)
        vl.addWidget(card2)

        vl.addStretch(1)
        self._refresh_teach_files()
        return page

    # ── 关节直控（滑块）──────────────────────────────
    def _following_real(self) -> bool:
        if self._is_sim:
            return False
        return any(a.connected and a.positions() for a in self.arms.values())

    def _effective_pose(self) -> dict[str, float]:
        if self._following_real():
            out = {}
            for a in self.arms.values():
                out.update(a.positions())
            return out
        return dict(self._local_pose)

    def _joint_provider(self) -> dict[str, float]:
        return self._effective_pose()

    def _rebuild_sliders(self) -> None:
        while self._slider_layout.count():
            item = self._slider_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for name in [n for n in self._sliders if "finger" not in n]:
            del self._sliders[name]
        for name in [n for n in self._torque_labels if "finger" not in n]:
            del self._torque_labels[name]
        prefix = "openarm_left" if self.combo_arm.currentIndex() == 0 else "openarm_right"
        for i in range(1, 8):
            name = f"{prefix}_joint{i}"
            row = QWidget(self._slider_host)
            rl = QVBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(2)
            top = QHBoxLayout()
            top.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(f"J{i}", row)
            lbl.setObjectName("jointName")
            val = QLabel("+0.000", row)
            val.setObjectName("jointVal")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            tq = QLabel("--", row)
            tq.setObjectName("jointTorque")
            tq.setFixedWidth(46)
            tq.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            tq.setToolTip("关节力矩（Nm，正=沿关节正方向；编码器+电流估计值）")
            tq.setProperty("warn", False)
            top.addWidget(lbl)
            top.addStretch(1)
            top.addWidget(val)
            top.addSpacing(8)
            top.addWidget(tq)
            sld = QSlider(Qt.Horizontal, row)
            sld.setRange(-314, 314)
            peak = SLIDER_JOINT_SPEED_LIMITS_RAD_S[i - 1]
            sld.setToolTip(f"J{i} 真机滑块目标峰值 {peak:.2f} rad/s，受设备配置限速约束")
            sld.valueChanged.connect(lambda v, n=name: self._on_slider(n, v))
            rl.addLayout(top)
            rl.addWidget(sld)
            # 切臂时回显该臂已有目标（真机跟随时回显实测角）
            cur = self._effective_pose().get(name, 0.0)
            sld.blockSignals(True)
            sld.setValue(int(max(-314, min(314, round(cur * 100)))))
            sld.blockSignals(False)
            val.setText(f"{cur:+.3f}")
            self._slider_layout.addWidget(row)
            self._sliders[name] = (sld, val)
            self._torque_labels[name] = tq

    def _on_slider(self, name: str, raw: int) -> None:
        if self._teleop_session is not None or self._traveling or self._estopping:
            return
        is_grip = "finger" in name
        if not self._is_sim and self.sw_teach.isChecked() and not is_grip:
            return
        self._stop_teach_play("手动调节")
        if not is_grip and self.sw_ee_drag.isChecked():
            self.sw_ee_drag.setChecked(False)
            self._clear_motion_targets(hold=True)
        val = raw / 1000.0 if is_grip else raw / 100.0
        pair = self._sliders.get(name)
        if pair:
            pair[1].setText(f"{val * 1000:.0f} mm" if is_grip else f"{val:+.3f}")
        # 进入 pending：在实测追上该目标前，滑块保持显示用户设定的值，
        # 避免松手瞬间被实测回写造成"回弹/突变"。
        if hasattr(self, "_pending_sliders"):
            self._pending_sliders.add(name)
        self._local_pose[name] = val
        if self._worker and not self._following_real():
            self._worker.set_pose(self._local_pose)
        # 关节目标立即交给驱动，实际 CAN 下发仍由控制线程节流。
        side = name.split("_", 2)[1]
        arm = self.arms.get(side)
        if self._connected and not self._is_sim and not self._traveling and arm and arm.enabled:
            self._cmd_pose[name] = val
            if not is_grip:
                self._slider_target_sides.add(side)
            self._send_live_targets(only_side=side)

    # ── 末端拖拽（官方 openarm_control IK）─────────────────
    @property
    def _active_side(self) -> str:
        return "left" if self.combo_arm.currentIndex() == 0 else "right"

    def _q8(self, side: str) -> list[float]:
        pose = self._effective_pose()
        q = [float(pose.get(f"openarm_{side}_joint{i}", 0.0)) for i in range(1, 8)]
        q.append(float(pose.get(f"openarm_{side}_finger_joint1", 0.0)))
        return q

    def _ensure_ik(self):
        if getattr(self, "_ik", None) is None:
            try:
                from ..backend.ik import V1_XML, ArmIK
                if not ArmIK.available():
                    self._log(f"IK 不可用：缺少官方 v1 MuJoCo 模型（{V1_XML}）")
                    return None
                self._ik = ArmIK.shared()
                self._log("IK 就绪（官方 openarm_control + v1 模型）")
            except Exception as exc:  # noqa: BLE001
                self._log(f"IK 初始化失败：{type(exc).__name__}: {exc}")
                self._ik = None
        return self._ik

    def _on_teach_toggle(self, on: bool) -> None:
        self._select_force_mode("teach" if on else "position")

    def _selected_force_mode(self) -> str:
        if self.sw_teach.isChecked():
            return "teach"
        if self.sw_jimp.isChecked():
            return "impedance"
        return "gravity" if self.sw_gravity.isChecked() else "position"

    def _set_force_switches(self, mode: str) -> None:
        for name, switch in (("gravity", self.sw_gravity), ("teach", self.sw_teach),
                             ("impedance", self.sw_jimp)):
            with QSignalBlocker(switch):
                switch.setChecked(name == mode)
        self.setProperty("teach", "on" if mode == "teach" else "off")

    def _send_force_mode(self, arm: DriverArm, mode: str) -> None:
        arm.set_control_mode(mode, stiffness=self._imp_stiffness(), zeta=self._imp_zeta())

    def _select_force_mode(self, mode: str) -> None:
        if self._traveling or self._teleop_session is not None or self._estopping:
            return
        connected = [arm for arm in self.arms.values() if arm.connected]
        if self._is_sim or (mode != "position" and not connected):
            self._set_force_switches("position")
            self._log("力控模式需要连接真机")
            return
        previously_teaching = self.property("teach") == "on"
        self._stop_teach_play("切换力控")
        self._clear_motion_targets(hold=True)
        if mode in ("teach", "impedance"):
            self.sw_ee_drag.setChecked(False)
        if previously_teaching and mode != "teach" and self._recording:
            self._stop_record()
        self._set_force_switches(mode)
        for arm in connected:
            self._send_force_mode(arm, mode)
        self._refresh_impedance_status()
        self._refresh_teleop_controls()

    def _clear_motion_targets(self, *, hold: bool = False) -> None:
        self._cmd_pose.clear()
        self._slider_target_sides.clear()
        self._pending_sliders.clear()
        if hold and not self._is_sim:
            for arm in self.arms.values():
                if arm.connected:
                    arm.set_targets(None)

    def _on_gripper_force(self, side: str, percent: int) -> None:
        arm = self.arms.get(side)
        if arm is not None and arm.connected:
            arm.set_gripper_force(percent)

    def _on_grasp(self, side: str) -> None:
        arm = self.arms.get(side)
        if (self._is_sim or self._traveling or self._estopping or self._teleop_session is not None
                or arm is None or not arm.connected or not arm.enabled):
            return
        self._stop_teach_play("智能抓取")
        if arm.request_grasp():
            name = f"openarm_{side}_finger_joint1"
            self._cmd_pose.pop(name, None)
            self._pending_sliders.discard(name)
            self._refresh_teleop_controls()

    def _refresh_gripper_controls(self, available: bool) -> None:
        self._gripper_controls.setEnabled(available)
        texts = {
            "pending": "等待夹爪反馈", "closing": "闭合中 · 夹住后按电流上限保持",
            "holding": "已夹持 · 保持中", "empty": "未夹到物体 · 夹爪已全闭",
            "timeout": "未在限时内停住 · 仍在闭合", "lost": "物体已脱出 · 夹爪已全闭",
            "feedback_lost": "夹爪反馈异常 · 抓取未确认", "cancelled": "抓取已停止",
        }
        for side, (limit, grasp, release, status) in self._grip_options.items():
            arm = self.arms.get(side)
            supported = arm is not None and getattr(arm, "supports_gripper_control", False)
            enabled = supported and arm.enabled and not self._is_sim
            phase = getattr(arm, "grasp_status", "idle")
            busy = phase in ("pending", "closing", "holding")
            limit.setEnabled(available and not self._is_sim and not busy)
            grasp.setEnabled(available and enabled and not busy)
            release.setEnabled(available and (self._is_sim or enabled))
            text = texts.get(phase, "")
            torque_limit = getattr(arm, "grasp_torque_limit", None)
            if torque_limit is not None and phase not in ("idle", "pending"):
                text += f"（电流上限 {torque_limit:.2f} N·m）"
            status.setText(text)
            status.setVisible(enabled and phase != "idle")

    # ── 录制（显式开始/停止）───────────────────────────
    def _on_record_toggle(self) -> None:
        if self._recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self) -> None:
        if self._is_sim or self._traveling or self._estopping:
            self._log("录制需要真机，当前走位结束后可开始录制")
            return
        if self._play_timer.isActive():
            self._stop_teach_play("开始录制")
        if self._read_record_pose() is None:
            self._log("录制：需要先连接双臂并取得完整的实测反馈")
            return
        self._rec_traj = []
        self._rec_t0 = time.monotonic()
        self._recording = True
        self._style_record_btn(True)
        self._log("录制开始：50Hz 采样双臂关节 + 夹爪"
                  + ("" if self.sw_teach.isChecked() or self._teleop_session is not None
                     else "（提示：当前不是零重力，手推不动；可直接用滑块）"))

    def _stop_record(self) -> None:
        self._recording = False
        self._style_record_btn(False)
        self._save_teach_traj()

    def _style_record_btn(self, recording: bool) -> None:
        self.btn_record.setText("停止并保存" if recording else "开始录制")
        self.btn_record.setObjectName("btnEstop" if recording else "btnPrimary")
        # 换 objectName 后必须重新 polish，否则 QSS 不重新匹配
        self.btn_record.style().unpolish(self.btn_record)
        self.btn_record.style().polish(self.btn_record)
        self.lbl_rec.setText("● 录制中 …" if recording else "空闲 · 未录制")

    def _read_record_pose(self) -> dict[str, float] | None:
        pose = {}
        for side in ("left", "right"):
            arm = self.arms.get(side)
            if arm is None or not arm.connected or arm.joint_feedback(require_enabled=False) is None:
                return None
            positions = arm.positions()
            prefix = f"openarm_{side}_"
            names = [f"{prefix}joint{i}" for i in range(1, 8)] + [f"{prefix}finger_joint1"]
            if any(name not in positions or not np.isfinite(positions[name]) for name in names):
                return None
            pose.update({name: positions[name] for name in names})
        return pose

    def _record_teach(self, pose: dict[str, float]) -> None:
        """50Hz 记录当前关节角（录制中由 _push_pose 调用）。

        列：t + 左右各 7 个关节角 + 左右夹爪开口（米）。
        早期 CSV 只有前 15 列，回放时会自动兼容（没有夹爪就只放关节）。
        """
        row = [round(pose[f"openarm_{s}_joint{i}"], 4)
               for s in ("left", "right") for i in range(1, 8)]
        grip = [round(pose[f"openarm_{s}_finger_joint1"], 5)
                for s in ("left", "right")]
        self._rec_traj.append([round(time.monotonic() - self._rec_t0, 4),
                               *row, *grip])

    def _update_rec_label(self) -> None:
        if not self._recording:
            return
        n = len(self._rec_traj)
        self.lbl_rec.setText(
            f"● 录制中 · {time.monotonic() - self._rec_t0:.1f}s · {n} 帧")

    def _save_teach_traj(self) -> None:
        rows = list(self._rec_traj)
        if not rows:
            self._log("录制结束：没有采到帧（臂未连接？）")
            return
        out_dir = self._teach_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / time.strftime("teach_%Y%m%d_%H%M%S.csv")
        cols = ["t"] + [f"{s}_joint{i}" for s in ("left", "right")
                        for i in range(1, 8)] + ["left_finger", "right_finger"]
        try:
            import csv
            with path.open("w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(cols)
                wr.writerows(rows)
            self._log(f"示教数据已保存：{path}（{len(rows)} 帧）")
            self._refresh_teach_files()
        except Exception as exc:  # noqa: BLE001
            self._log(f"示教数据保存失败：{exc}")
        self._rec_traj = []

    # ── 示教回放 ──────────────────────────────────────
    def _teach_dir(self) -> Path:
        return Path.home() / "openarm_studio/records"

    def _refresh_teach_files(self) -> None:
        """把 records/ 里的 teach_*.csv 填进下拉（新的在前）。"""
        d = self._teach_dir()
        files = sorted(d.glob("teach_*.csv"), reverse=True) if d.is_dir() else []
        cur = self.combo_teach.currentText()
        self.combo_teach.blockSignals(True)
        self.combo_teach.clear()
        self.combo_teach.addItems([f.name for f in files])
        names = [f.name for f in files]
        if cur in names:
            self.combo_teach.setCurrentText(cur)
        self.combo_teach.blockSignals(False)
        self.btn_teach_play.setEnabled(bool(files))
        if not files:
            self.combo_teach.addItem("（无录制文件）")

    def _load_teach_csv(self, path: Path) -> list[list[float]]:
        """读回 CSV → [[t, 14×关节角], ...]，跳过坏行。"""
        import csv
        rows: list[list[float]] = []
        with path.open(newline="") as f:
            rd = csv.reader(f)
            next(rd, None)                        # 表头
            for row in rd:
                if len(row) not in (15, 17):
                    continue
                try:
                    values = [float(v) for v in row]
                    if np.isfinite(values).all() and (not rows or values[0] > rows[-1][0]):
                        rows.append(values)
                except ValueError:
                    continue
        return rows

    def _on_teach_play(self) -> None:
        if self._traveling or self._estopping or self._teleop_session is not None:
            return
        if self._play_timer.isActive():
            self._stop_teach_play("手动停止")
            return
        name = self.combo_teach.currentText()
        if not name:
            return
        try:
            rows = self._load_teach_csv(self._teach_dir() / name)
        except (OSError, UnicodeError) as exc:
            self._log(f"回放：无法读取 {name}（{exc}）")
            return
        if len(rows) < 2:
            self._log(f"回放：{name} 没有足够的有效帧")
            return
        live = {side for side, arm in self.arms.items() if arm.connected and arm.enabled}
        if self._is_sim:
            live.clear()
        if live and not all(self.arms[side].joint_feedback() is not None for side in live):
            self._log("回放：参与机械臂缺少实时关节反馈")
            return
        if live and self.sw_teach.isChecked():
            self._select_force_mode("gravity")
        if self._recording:
            self._stop_record()
        self._clear_motion_targets(hold=bool(live))
        self.sw_ee_drag.setChecked(False)
        self._play_live_sides = live
        if not live:
            self._log("回放：仅预览 3D 姿态")
        self._play_rows = rows
        self._play_i = 0
        self._play_t0 = self._play_last_tick = time.monotonic()
        self._play_dur = max(rows[-1][0] - rows[0][0], 0.1)
        self._play_timer.start()
        self.btn_teach_play.setText("停止")
        self._log(f"回放开始：{name}（{len(rows)} 帧 / {self._play_dur:.1f}s）")
        self._play_tick()

    def _playback_feedback_ok(self) -> bool:
        return all(
            (arm := self.arms.get(side)) is not None and arm.connected and arm.enabled
            and arm.joint_feedback() is not None for side in self._play_live_sides
        )

    def _play_tick(self) -> None:
        now = time.monotonic()
        if self._play_live_sides and now - self._play_last_tick > 0.15:
            self._stop_teach_play("回放更新中断")
            return
        self._play_last_tick = now
        if not self._playback_feedback_ok():
            self._stop_teach_play("参与回放的机械臂失去反馈或失能")
            return
        rows = self._play_rows
        if len(rows) < 2:
            self._stop_teach_play("无数据")
            return
        t = now - self._play_t0 + rows[0][0]
        while self._play_i + 1 < len(rows) and rows[self._play_i + 1][0] <= t:
            self._play_i += 1
        if self._play_i + 1 >= len(rows):
            self._apply_teach_frame(rows[-1])
            self._send_live_targets(final_frame=True)
            self._stop_teach_play("播放完成", hold=False)
            return
        a, b = rows[self._play_i], rows[self._play_i + 1]
        span = max(b[0] - a[0], 1e-6)
        k = max(0.0, min(1.0, (t - a[0]) / span))
        n = min(len(a), len(b))
        self._apply_teach_frame([a[j] + (b[j] - a[j]) * k for j in range(n)])
        self.lbl_teach.setText(
            f"回放 {min(max(t - rows[0][0], 0.0), self._play_dur):.1f}"
            f" / {self._play_dur:.1f} s")

    def _apply_teach_frame(self, frame: list[float]) -> None:
        """一帧（关节角 + 可选夹爪）→ 目标。使能时进队列，否则只推孪生。"""
        vals = {}
        for i, side in enumerate(("left", "right")):
            for j in range(7):
                vals[f"openarm_{side}_joint{j + 1}"] = float(frame[1 + i * 7 + j])
        # 新版 CSV 带夹爪（第 15、16 列，单位米）；旧文件没有就保持原样
        if len(frame) >= 17:
            for i, side in enumerate(("left", "right")):
                vals[f"openarm_{side}_finger_joint1"] = float(frame[15 + i])
        self._local_pose.update(vals)
        if self._play_live_sides:
            self._cmd_pose = {name: value for name, value in vals.items()
                              if name.split("_", 2)[1] in self._play_live_sides}
        elif self._worker is not None:
            self._worker.set_pose(self._local_pose)

    def _stop_teach_play(self, why: str, *, hold: bool = True) -> None:
        was = self._play_timer.isActive()
        if not was and not self._play_rows:
            return
        live = bool(self._play_live_sides)
        self._play_timer.stop()
        self._play_rows = []
        self._play_live_sides.clear()
        self.btn_teach_play.setText("播放")
        self.lbl_teach.setText("回放 --")
        if was:
            self._clear_motion_targets(hold=hold and live)
            self._log(f"回放结束：{why}")

    def _on_gravity_toggle(self, on: bool) -> None:
        self._select_force_mode("gravity" if on else "position")

    # ── 关节阻抗参数 ──────────────────────────────────
    def _imp_stiffness(self) -> float:
        return self.sld_stiff.value() / 100.0

    def _imp_zeta(self) -> float:
        return self.sld_zeta.value() / 100.0

    def _on_jimp_toggle(self, on: bool) -> None:
        self._select_force_mode("impedance" if on else "position")

    def _on_imp_param(self) -> None:
        """刚度/阻尼比滑块变化：立即生效（增益按速率平滑过渡）。"""
        s, z = self._imp_stiffness(), self._imp_zeta()
        self.lbl_stiff.setText(f"{s:.0%}")
        self.lbl_zeta.setText(f"{z:.2f}")
        if (self._is_sim or self._traveling or self._teleop_session is not None
                or not self.sw_jimp.isChecked()):
            return
        for arm in self.arms.values():
            if arm.connected:
                self._send_force_mode(arm, "impedance")

    def _refresh_impedance_status(self) -> None:
        sim = self._is_sim
        on = self.sw_jimp.isChecked()
        editable = not sim and not self._traveling and not self._estopping and self._teleop_session is None
        self.sw_jimp.setEnabled(editable)
        self.sld_stiff.setEnabled(on and editable)
        self.sld_zeta.setEnabled(on and editable)
        if sim:
            text = "仿真使用固定伺服增益"
        elif self._teleop_session is not None:
            text = "协同模式使用重力前馈"
        elif not on:
            text = "关节阻抗未开启"
        else:
            side = self._active_side
            name = "左臂" if side == "left" else "右臂"
            arm = self.arms.get(side)
            if arm is None or not arm.connected:
                text = f"{name}未连接"
            elif not arm.enabled:
                text = f"{name}待使能"
            elif (gains := arm.impedance_gains) is None:
                text = f"{name}尚无近期阻抗帧"
            else:
                kp, kd = gains
                kp_values = " / ".join(f"{v:.1f}" for v in kp)
                kd_values = " / ".join(f"{v:.2f}" for v in kd)
                text = f"{name} Kp {kp_values}\nKd {kd_values}"
        self.lbl_gain.setText(text)

    def _on_friction_toggle(self, on: bool) -> None:
        """设置重力补偿、零重力示教和关节阻抗中的摩擦助力。"""
        for arm in self.arms.values():
            if arm.connected:
                arm.set_friction_comp(on)
        self._log("摩擦补偿已开启" if on else "摩擦补偿已关闭")

    def _on_ee_drag_toggle(self, on: bool) -> None:
        if on and (self._traveling or self._estopping or self._teleop_session is not None
                   or (not self._is_sim and (self.sw_teach.isChecked() or self.sw_jimp.isChecked()))):
            self.sw_ee_drag.setChecked(False)
            return
        self._ee_drag_on = bool(on)
        if on:
            self._stop_teach_play("开始末端拖拽")
            self._clear_motion_targets(hold=True)
            ik = self._ensure_ik()
            if ik is None:
                self.sw_ee_drag.setChecked(False)
                return
            self.view.on_target_drag = self._on_ee_drag
            self._update_ee_marker()
            self._log(f"末端拖拽开启（{self._active_side} 臂）：拖动 3D 里的十字标记")
        else:
            self.view.on_target_drag = None
            self.view.clear_target()
            self._log("末端拖拽关闭")

    def _update_ee_marker(self) -> None:
        if not getattr(self, "_ee_drag_on", False):
            return
        ik = getattr(self, "_ik", None)
        if ik is None:
            return
        side = self._active_side
        try:
            pose = ik.fk(side, self._q8(side))
        except Exception:  # noqa: BLE001
            return
        self.view.set_ee_pose(tuple(pose[:3]), tuple(pose[3:]))

    def _on_ee_drag(self, world, done: bool) -> None:
        """拖动中实时跟随，松手后精确收敛。"""
        if self._traveling or self._teleop_session is not None or self._estopping:
            return
        if not self._is_sim and (self.sw_teach.isChecked() or self.sw_jimp.isChecked()):
            return
        if not self._ee_drag_on:
            return
        self._stop_teach_play("末端拖拽")
        ik = getattr(self, "_ik", None)
        if ik is None:
            return
        now = time.monotonic()
        # 拖动中节流 ~30Hz、只迭代 1 步（mink 是增量式，单步即可跟上手掌）；
        # 松手时迭代 8 步做精确收敛。
        if not done and now - getattr(self, "_ik_t", 0.0) < 0.033:
            return
        self._ik_t = now
        side = self._active_side
        other = "left" if side == "right" else "right"
        try:
            cur = ik.fk(side, self._q8(side))
            target = [float(world[0]), float(world[1]), float(world[2]),
                      *cur[3:]]          # 位置跟手，姿态保持
            sol = ik.ik(side, target, self._q8(other), iters=8 if done else 1)
        except Exception as exc:  # noqa: BLE001
            if done:
                self._log(f"IK 失败：{type(exc).__name__}: {exc}")
            return
        if sol is None:
            if done:
                self._log("IK 无解：目标点可能超出工作空间")
            return
        for i in range(1, 8):
            name = f"openarm_{side}_joint{i}"
            self._local_pose[name] = float(sol[i - 1])
            if self._connected and not self._is_sim and not self._traveling:
                self._cmd_pose[name] = float(sol[i - 1])
        if self._worker and not self._following_real():
            self._worker.set_pose(self._local_pose)
        if done:
            self._log("IK 解已下发：" + " ".join(f"{v:+.2f}" for v in sol[:7]))
        self._update_ee_marker()

    def _on_move_pose(self, kind: str) -> None:
        """按手动限速移动到配置位姿，保留当前的补偿和阻抗设置。"""
        if self._traveling or self._is_sim or self._estopping or self._teleop_session is not None:
            return
        live = {side: arm for side, arm in self.arms.items()
                if arm.connected and arm.enabled}
        if not live:
            self._log("平滑运动：臂未连接或未使能")
            return
        if self.sw_teach.isChecked():
            self._log("平滑运动：请先结束手动示教")
            return
        self._stop_teach_play("平滑运动")
        self.sw_ee_drag.setChecked(False)
        self._clear_motion_targets(hold=True)
        self._log(f"平滑运动：{'起始位' if kind == 'goto_start' else '收臂位'}，"
                  f"限制关节目标速度为 {MANUAL_JOINT_SPEED_RAD_S:.2f} rad/s")
        self._start_arm_travel(live, "move_start" if kind == "goto_start" else "move_park")

    def _start_arm_travel(self, arms: dict[str, DriverArm], command: str) -> None:
        self._traveling = True
        self._travel_id += 1
        travel_id = self._travel_id
        self._travel_arms = set(arms)
        self._travel_ok = True
        self._refresh_teleop_controls()
        for side, arm in arms.items():
            move = getattr(arm, command)
            if not move(done=lambda ok, side=side: self.travel_sig.emit(travel_id, side, ok)):
                self._on_arm_travel_done(travel_id, side, False)

    def _on_arm_travel_done(self, travel_id: int, side: str, ok: bool) -> None:
        if travel_id != self._travel_id or side not in self._travel_arms:
            return
        self._travel_arms.remove(side)
        self._travel_ok = self._travel_ok and ok
        if self._enabling and not ok:
            label = "左臂" if side == "left" else "右臂"
            arm = self.arms.get(side)
            status = "保持使能，起始位未确认" if arm is not None and arm.enabled else "未使能"
            self._log(f"{label}{status}，请查看该臂日志")
        if not self._travel_arms:
            self._after_travel(self._travel_ok)

    def _after_travel(self, ok: bool = True) -> None:
        if not self._traveling:
            return
        self._travel_arms.clear()
        pose = self._effective_pose()
        for name, (sld, val) in self._sliders.items():
            cur = pose.get(name)
            if cur is None:
                continue
            sld.blockSignals(True)
            if "finger" in name:
                sld.setValue(int(max(0, min(44, round(cur * 1000)))))
                val.setText(f"{cur * 1000:.0f} mm")
            else:
                sld.setValue(int(max(-314, min(314, round(cur * 100)))))
                val.setText(f"{cur:+.3f}")
            sld.blockSignals(False)
        self._traveling = False
        if self._power_motion_pending:
            enabling = self._enabling
            self._enabling = self._disabling = False
            self._clear_motion_targets()
            if not any(arm.enabled for arm in self.arms.values()):
                for switch in (self.sw_gravity, self.sw_teach, self.sw_jimp):
                    with QSignalBlocker(switch):
                        switch.setChecked(False)
                self.setProperty("teach", "off")
            self._update_power_controls()
            if enabling:
                if ok:
                    self._log("使能走位完成，重力前馈继续工作")
                elif self._arms_to_retry():
                    self._log("仅一臂已使能，可点击「重试未使能臂」")
                else:
                    self._log("使能走位未完成，请查看各臂状态")
            else:
                self._log("失能回程完成" if ok else "失能回程中止，请查看各臂状态")
            action, self._after_disable = self._after_disable, None
            if action and ok and not self._has_enabled_arms():
                QTimer.singleShot(0, self.close if action == "close" else self._disconnect)
        else:
            self._log(f"轨迹{'完成' if ok else '中止'}：滑块已同步到当前姿态")
        self._refresh_teleop_controls()
        self._refresh_impedance_status()

    def _zero_pose(self) -> None:
        """全部回零：滑块/孪生清零；真机在已使能的臂上同步下发零目标。"""
        if self._traveling or self._estopping or self._teleop_session is not None:
            return
        if not self._is_sim and self.sw_teach.isChecked():
            self._log("全部回零：请先结束手动示教")
            return
        self._stop_teach_play("全部回零")
        self.sw_ee_drag.setChecked(False)
        self._clear_motion_targets(hold=True)
        for name, (sld, val) in self._sliders.items():
            sld.blockSignals(True)
            sld.setValue(0)
            sld.blockSignals(False)
            val.setText("0 mm" if "finger" in name else "+0.000")
        # 关键：写 0 而不是从字典里删——渲染线程只更新字典里存在的关节，
        # 删掉的名字会停留在旧 qpos（"全部回零没反应"的根因）
        for side in ("left", "right"):
            for i in range(1, 8):
                self._local_pose[f"openarm_{side}_joint{i}"] = 0.0
            self._local_pose[f"openarm_{side}_finger_joint1"] = 0.0
        if self._worker and not self._following_real():
            self._worker.set_pose(self._local_pose)
        armed = [a for a in self.arms.values() if a.connected and a.enabled]
        if self._connected and not self._is_sim and armed:
            self._cmd_pose.update(self._local_pose)
            self._log(f"全部回零：孪生清零，真机目标清零下发（{len(armed)} 臂已使能）")
        elif self._connected and not self._is_sim:
            self._log("全部回零：孪生清零；真机未使能，未下发")
        else:
            self._log("全部回零：目标姿态清零")

    def _poll_controllers(self) -> None:
        """2s 刷新：导航轨双臂行 + 机器人页温度/健康/安全锁存。"""
        for side in ("left", "right"):
            lbl = self.ctrl_rows.get(side)
            if lbl is None:
                continue
            arm = self.arms.get(side)
            if arm is None or not arm.connected:
                lbl.setText("离线")
            elif arm.safety_stop_reason:
                lbl.setText("安全停止")
            else:
                lbl.setText(f"{arm.rate:.0f}Hz·{'使能' if arm.enabled else '未使能'}"
                            f"·{arm.max_mos_temp()}°")
            self._refresh_diag_rows(side, arm)

    _TEMP_WARN, _TEMP_DANGER = 60, 80

    def _refresh_diag_rows(self, side: str, arm) -> None:
        zh = "左臂" if side == "left" else "右臂"
        temp_row = self.row_temp_l if side == "left" else self.row_temp_r
        health_row = self.row_health_l if side == "left" else self.row_health_r
        if arm is None or not arm.connected:
            temp_row.set_value("—")
            temp_row.val.setStyleSheet("")
            health_row.set_value("—")
            return
        # 温度：报 MOS/转子最大值及所在关节，超温变色
        tmos, trot = arm.temps(), arm.rotor_temps()
        if tmos:
            jm = max(tmos, key=tmos.get)
            jr = max(trot, key=trot.get) if trot else jm
            short = lambda n: n.split("joint")[-1].replace("finger", "爪")  # noqa: E731
            temp_row.set_value(f"{tmos[jm]}°({short(jm)}) / {trot.get(jr, 0)}°({short(jr)})")
            hot = max(tmos.values())
            if hot >= self._TEMP_DANGER:
                temp_row.val.setStyleSheet("color: #e03131; font-weight: 600;")
            elif hot >= self._TEMP_WARN:
                temp_row.val.setStyleSheet("color: #e07b00; font-weight: 600;")
            else:
                temp_row.val.setStyleSheet("")
        # 健康：各轴状态 + 总线错误计数
        h = arm.health
        if h is not None:
            statuses, bus = h
            silent = sum(1 for s in statuses if s == "SILENT")
            faults = sum(v for v in bus.values()
                         if isinstance(v, (int, float)) and not isinstance(v, bool))
            carrier = bus.get("carrier", True)
            queue_full = bus.get("write_no_buffer", 0)
            if silent == 0 and faults == 0 and carrier:
                health_row.set_value("正常")
                health_row.val.setStyleSheet("")
            else:
                bits = []
                if silent:
                    bits.append(f"{silent} 轴静默")
                if queue_full:
                    bits.append(f"发送队列满 {queue_full} 次")
                if faults > queue_full:
                    bits.append(f"总线 {faults - queue_full}")
                if not carrier:
                    bits.append("载波丢失")
                health_row.set_value("·".join(bits))
                health_row.val.setStyleSheet("color: #e07b00;")
        else:
            health_row.set_value("采样中…")
        # 安全锁存（两臂任一）
        reasons = [a.safety_stop_reason for a in self.arms.values()
                   if a.safety_stop_reason]
        if reasons:
            self.row_safety.set_value(reasons[0])
            self.row_safety.val.setStyleSheet("color: #e03131; font-weight: 600;")
        else:
            self.row_safety.set_value("—")
            self.row_safety.val.setStyleSheet("")

    @property
    def _is_sim(self) -> bool:
        return self.sw_mode.currentIndex() == 1

    def _on_mode_change(self, idx: int) -> None:
        sim = idx == 1
        if self.arms or self._conn_state == "connecting" or self._can_resetting:
            with QSignalBlocker(self.sw_mode):
                self.sw_mode.setCurrentIndex(1 - idx)
            self._log("请先断开真机或等待连接操作结束，再切换模式")
            return
        self._stop_teach_play("切换模式")
        self._clear_motion_targets()
        self._set_force_switches("position")
        self._set_conn_state("disconnected", "未连接")
        if sim:
            self._stop_teleop("切换仿真")
        if self._worker is not None:
            self._worker.set_mode("sim" if sim else "twin")
            if sim:
                # 切到仿真时把目标对齐到当前姿态，避免一进去就甩过去
                self._local_pose.update(self._worker.positions())
                self._worker.set_pose(self._local_pose)
        self.chip_scene.setText("物理仿真" if sim else "数字孪生")
        self.badge_mode.setText("仿真" if sim else "数字孪生")
        self.btn_enable.setVisible(not sim and self._connected)
        self._refresh_impedance_status()
        self._refresh_teleop_controls()
        self._log(f"模式：{'物理仿真（MuJoCo 重力和位置伺服在工作）' if sim else '数字孪生（跟随真机反馈）'}")
        self._goto_page(self.pages.currentIndex())

    def showEvent(self, ev) -> None:  # noqa: N802
        super().showEvent(ev)
        if not getattr(self, "_intro_started", False):
            self._intro_started = True
            self.view.start_intro()

    def _on_precheck(self) -> None:
        if self._is_sim or not self.arms:
            self._log("启动前检查：请先连接真机")
            return
        samples = [arm.joint_feedback(require_enabled=False) for arm in self.arms.values()]
        if any(sample is None for sample in samples):
            self._log("启动前检查：尚无完整的关节反馈")
        elif all(sample.stationary() for sample in samples):
            self._log("启动前检查：机械臂反馈完整且静止")
        else:
            self._log("启动前检查：机械臂仍在运动")

    # ── 总线体检（只读探测，后台线程跑，不卡 GUI）──────
    _BUS_LABELS = (("can0", "右臂", "row_bus_r"), ("can1", "左臂", "row_bus_l"))

    def _on_bus_check(self) -> None:
        if self._has_enabled_arms() or self._traveling or self._estopping:
            self._log("总线体检请在失能后执行，运行时可查看「机器人」页")
            return
        if self._probing or self._conn_state == "connecting":
            return
        self._probing = True
        self.btn_bus_check.setEnabled(False)
        self.btn_bus_check.setText("体检中…")
        self.row_bus_r.set_value("…")
        self.row_bus_l.set_value("…")

        def work() -> None:
            out = {}
            for iface, label, _ in self._BUS_LABELS:
                try:
                    out[iface] = probe_bus(iface)
                except Exception as exc:  # noqa: BLE001
                    out[iface] = {"error": f"{type(exc).__name__}: {exc}"}
            self.probe_sig.emit(out)

        threading.Thread(target=work, daemon=True, name="bus-probe").start()

    # CAN 重置脚本路径（接口名硬编码在脚本内部，这里只传固定路径）
    CAN_RESET_SCRIPT = str(Path(__file__).resolve().parents[2] / "tools" / "can_reset.sh")

    def _on_can_reset(self) -> None:
        """CAN 重连：断开 → 重置接口（等效拔插一次 USB）→ 重新连接双臂。"""
        if self._power_motion_pending:
            return
        if self._can_resetting or self._conn_state == "connecting" or self._estopping or self._probing:
            return
        if self._is_sim or self._has_enabled_arms():
            self._log("CAN 重连前请切到真机模式并先失能")
            return
        self._disconnect()
        if self.arms:
            return
        self._can_resetting = True
        self._set_conn_state("connecting", "CAN 重连中…")
        self.btn_can_reset.setEnabled(False)
        self.btn_can_reset.setText("重置中…")
        threading.Thread(target=self._can_reset_worker, daemon=True,
                         name="can-reset").start()

    def _can_reset_worker(self) -> None:
        import subprocess

        script = self.CAN_RESET_SCRIPT
        if not Path(script).exists():
            self._log(f"CAN 重连：缺少脚本 {script}")
            self.reset_sig.emit(False)
            return
        # argv 全部是常量（脚本路径固定、接口名在脚本内），不做任何拼接；
        # sudo -n（免密）优先，退回到 pkexec（图形授权框）
        rc = None
        for argv in (["sudo", "-n", "bash", script],
                     ["pkexec", "bash", script]):
            try:
                proc = subprocess.run(argv, capture_output=True, timeout=40,
                                      shell=False, check=False)
            except Exception as exc:  # noqa: BLE001
                self._log(f"CAN 重连：调用失败 {type(exc).__name__}: {exc}")
                continue
            text = ((proc.stdout or b"").decode(errors="ignore") + "\n"
                    + (proc.stderr or b"").decode(errors="ignore"))
            for line in text.splitlines():
                if line.strip():
                    self._log("CAN 重连：" + line.strip())
            rc = proc.returncode
            if rc == 0:
                break
        for iface in ("can0", "can1"):
            st = can_ifstate(iface)
            self._log(f"CAN 重连：{iface} UP={st['up']} FD={st['fd']}")
        self.reset_sig.emit(rc == 0)

    def _can_reset_done(self, ok: bool) -> None:
        self._can_resetting = False
        self._set_conn_state("disconnected", "未连接")
        self.btn_can_reset.setEnabled(True)
        self.btn_can_reset.setText("CAN 重连")
        if ok:
            self._log("CAN 重连：接口已重置，开始重新连接…")
            QTimer.singleShot(300, self._connect)
        else:
            self._log(f"CAN 重连：重置未成功（可手动执行 "
                      f"sudo bash {self.CAN_RESET_SCRIPT}）")

    def _on_bus_result(self, res: dict) -> None:
        self._probing = False
        self.btn_bus_check.setEnabled(True)
        self.btn_bus_check.setText("体检 can0 / can1")
        for iface, label, row_attr in self._BUS_LABELS:
            row = getattr(self, row_attr)
            r = res.get(iface, {})
            if "error" in r:
                row.set_value(f"打不开：{r['error']}")
                self._log(f"总线体检 {label}({iface})：接口打不开 —— {r['error']}")
                continue
            alive = r.get("alive", 0)
            queue_full = r.get("tx_queue_full", 0)
            if queue_full:
                row.set_value(f"发送队列满 · {alive}/7 回包")
                self._log(f"总线体检 {label}({iface})：发送队列满 {queue_full} 次，"
                          f"{alive}/7 轴有状态回包")
                continue
            if alive == 7:
                row.set_value(f"在线 {alive}/7")
                self._log(f"总线体检 {label}({iface})：✅ 7/7 轴在线，"
                          f"位置 {r['positions']}")
            elif alive == 0:
                row.set_value("❌ 0/7 无应答")
                self._log(f"总线体检 {label}({iface})：0/7 轴有状态回包，"
                          "请检查该臂供电、急停、CAN 线和电机 ID")
            else:
                row.set_value(f"⚠️ {alive}/7 部分在线")
                dead = [f"J{i+1}" for i, (resp, _) in enumerate(r["axes"])
                        if resp == 0]
                self._log(f"总线体检 {label}({iface})：⚠️ {alive}/7 在线，"
                          f"失联轴 {dead}")

    def _read_teleop_feedback(self) -> tuple[dict[str, JointFeedback], str | None]:
        if not self._connected or self._is_sim:
            return {}, "需要连接真机双臂"
        if self._estopping:
            return {}, "急停"
        if self._traveling:
            return {}, "其他走位正在执行"
        samples = {}
        for side in ("left", "right"):
            arm = self.arms.get(side)
            label = "左臂" if side == "left" else "右臂"
            if arm is None or not arm.connected:
                return {}, f"{label}已断开"
            if arm.safety_stop_reason:
                return {}, f"{label}安全停止（{arm.safety_stop_reason}）"
            if not arm.enabled:
                return {}, f"{label}未使能"
            sample = arm.joint_feedback()
            if sample is None:
                return {}, f"{label}关节反馈超时或不完整"
            samples[side] = sample
        return samples, None

    def _toggle_teleop(self) -> None:
        if self._teleop_session is not None:
            self._stop_teleop("手动停止")
            return
        samples, reason = self._read_teleop_feedback()
        if reason:
            self._log(f"双臂协同：{reason}")
            return
        if not all(sample.stationary() for sample in samples.values()):
            self._log("双臂协同：等待两臂静止")
            return
        self._stop_teach_play("开始双臂协同")
        self._clear_motion_targets(hold=True)
        self.sw_ee_drag.setChecked(False)
        self._set_force_switches("position")
        side = self.combo_teleop.currentData()
        other = "right" if side == "left" else "left"
        session = BimanualTeleop(
            side, samples[side].positions, samples[other].positions,
            mirror=self.sw_teleop_mirror.isChecked(),
        )
        self._teleop_stop_reason = None
        self._teleop_session = session
        self._send_teleop_commands(session, samples)
        self._refresh_teleop_controls()
        self._refresh_impedance_status()
        self._log(f"双臂协同开启，{self.combo_teleop.currentText()}")

    def _send_teleop_commands(self, session: BimanualTeleop, samples: dict[str, JointFeedback]) -> None:
        leader, follower = session.leader_side, session.follower_side
        leader_cmd, follower_cmd = session.commands(samples[leader], samples[follower])
        self.arms[leader].set_teleop(leader_cmd)
        self.arms[follower].set_teleop(follower_cmd)

    def _stop_teleop(self, reason: str) -> None:
        session = self._teleop_session
        if session is None:
            return
        session.watchdog.stop(reason)
        self._teleop_session = None
        self._teleop_stop_reason = reason
        self._clear_motion_targets()
        for arm in self.arms.values():
            arm.set_teleop(None)
        if reason != "急停":
            self._set_force_switches("gravity")
        self._refresh_teleop_controls()
        self._refresh_impedance_status()
        self._log(f"双臂协同停止，{reason}")

    def _refresh_teleop_controls(self) -> None:
        session = self._teleop_session
        active = session is not None
        available = not active and not self._traveling and not self._estopping
        teaching = not self._is_sim and self.sw_teach.isChecked()
        self._joint_controls.setEnabled(available)
        for control in (self._slider_host, self.btn_zero, self.btn_goto_start):
            control.setEnabled(not teaching)
        self.sw_ee_drag.setEnabled(not teaching and not self.sw_jimp.isChecked())
        self._refresh_gripper_controls(available)
        for switch in (self.sw_gravity, self.sw_teach, self.sw_friction):
            switch.setEnabled(not active and not self._traveling and not self._estopping
                              and not self._is_sim)
        self.combo_teleop.setEnabled(not active and not self._traveling)
        self.sw_teleop_mirror.setEnabled(not active and not self._traveling and not self._is_sim)
        self.btn_teleop.setText("停止跟随" if active else "开始跟随")
        self.btn_teleop.setEnabled(active or (not self._is_sim and available))
        if session is not None:
            mapping = "镜像" if session.mirror else "同向"
            text = f"{mapping}跟随中 · 停止后两臂保持姿态"
        elif self._is_sim:
            text = "需要真机的关节反馈"
        elif self._traveling:
            text = "等待当前轨迹结束"
        elif self._teleop_stop_reason:
            text = f"已停止 · {self._teleop_stop_reason}"
        else:
            text = ""
        self.lbl_teleop.setText(text)
        self.lbl_teleop.setVisible(bool(text))

    def _send_live_targets(self, *, final_frame: bool = False, only_side: str | None = None) -> None:
        session = self._teleop_session
        if session is not None:
            samples, reason = self._read_teleop_feedback()
            reason = reason or session.watchdog.reason
            if reason:
                self._stop_teleop(reason)
            else:
                self._send_teleop_commands(session, samples)
            return
        if self._is_sim or not self._cmd_pose or self._traveling or self._estopping:
            return
        if self._play_timer.isActive() and not self._playback_feedback_ok():
            self._stop_teach_play("参与回放的机械臂失去反馈或失能")
            return
        manual = final_frame or not self._play_timer.isActive()
        for side, arm in self.arms.items():
            if (only_side is None or side == only_side) and arm.connected and arm.enabled:
                targets = {name: value for name, value in self._cmd_pose.items()
                           if name.startswith(f"openarm_{side}_")}
                if targets:
                    arm.set_targets(targets, manual=manual,
                                    slider=manual and not final_frame and side in self._slider_target_sides)

    def _push_pose(self) -> None:
        """50Hz：把当前姿态推给 3D 渲染线程。

        直接用「有没有实测数据」判断，而不是 `_following_real()`——后者
        还要求 `connected`，失能瞬间会短暂为 False，于是退回 `_local_pose`
        （滑块旧目标），3D 就卡在旧值上，之后才"跳"到实测位置。
        """
        if self._worker is None:
            return
        pose: dict[str, float] = {}
        # 回放中且没有臂使能：只看录制轨迹（否则会被实测姿态覆盖掉）
        if self._is_sim or (self._play_timer.isActive() and not self._play_live_sides):
            pose = dict(self._local_pose)
        else:
            for a in self.arms.values():
                p = a.positions()
                if p:
                    pose.update(p)
            if not pose:
                pose = dict(self._local_pose)
        self._worker.set_pose(pose)
        if self._recording:
            recorded_pose = self._read_record_pose()
            if recorded_pose is None:
                self._log("录制停止：双臂反馈不完整")
                self._stop_record()
            else:
                self._record_teach(recorded_pose)

    def _refresh_status(self) -> None:
        if self._estopping and not self._has_enabled_arms():
            self._estopping = False
        pose = self._effective_pose()
        following = self._following_real()
        sim = self._is_sim
        self._last_pose = pose        # 供「同步实测」用
        self._update_ee_marker()      # 末端标记跟随实际末端
        self._update_rec_label()      # 录制计时/帧数（5Hz 够用）

        # 姿态推给渲染线程由 50Hz 的 _pose_timer 负责（这里 200ms 一次太慢，
        # 3D 模型看着像卡住）。此处只做界面数值刷新。

        left = [pose.get(f"openarm_left_joint{i}", 0.0) for i in range(1, 8)]
        right = [pose.get(f"openarm_right_joint{i}", 0.0) for i in range(1, 8)]
        gl = pose.get("openarm_left_finger_joint1", 0.0)
        gr = pose.get("openarm_right_finger_joint1", 0.0)
        self.tile_left.set_value(f"{np.mean(np.abs(left)):.3f} rad")
        self.tile_right.set_value(f"{np.mean(np.abs(right)):.3f} rad")
        self.tile_grip.set_value(f"{(gl + gr) / 2 * 1000:.0f} mm")

        rate = 0.0
        if sim:
            self.status_dot.set_state(self._pal_now["accent"],
                                      "仿真运行中", breathe=False)
            self.chip_src.setText("仿真")
            self.row_link.set_value("MuJoCo 本地物理")
            self.row_rate.set_value("60 Hz 渲染")
        elif following:
            rates = [a.rate for a in self.arms.values() if a.connected]
            rate = sum(rates) / len(rates) if rates else 0.0
            self.status_dot.set_state(self._pal_now["success"], "真机在线", breathe=False)
            self.chip_src.setText("真机")
            self.row_link.set_value("已连接")
            self.row_rate.set_value(f"{rate:.0f} Hz")
        else:
            self.status_dot.set_state(self._pal_now["warn"], "3D 运行中", breathe=False)
            self.chip_src.setText("未连接")
            self.row_link.set_value("未连接")
            self.row_rate.set_value("--")
        live = sum(1 for a in self.arms.values() if a.connected)
        self.row_ctrl.set_value(f"{live}/2 臂在线")
        self.row_mode.set_value("仿真" if sim else "真机")
        status = "仿真" if sim else ("跟随真机" if following else "本地姿态")
        if self._enabling:
            status = "使能走位中"
        elif self._disabling:
            status = "失能回程中"
        self.sum_rows["状态"].setText(status)
        self.sum_rows["模式"].setText("仿真" if sim else "真机")
        self.sum_rows["延迟"].setText(
            f"{rate:.0f}Hz" if following else "—")
        self.sum_rows["上次"].setText(f"{self.view.frame_count}帧")
        self._update_slider_labels(pose)
        self._update_torque_labels()
        self._update_power_controls()

    # 各轴额定力矩（DM8009 54 / DM4340 28 / DM4310 10 Nm），
    # 用来把读数归一化，判断这一轴是不是已经接近能力边界。
    _TAU_MAX = (54.0, 54.0, 28.0, 28.0, 10.0, 10.0, 10.0)

    def _update_torque_labels(self) -> None:
        """把实测关节力矩填到滑块行右侧（5Hz，够读数）。"""
        tau: dict[str, float] = {}
        for arm in self.arms.values():
            if arm.connected:
                tau.update(arm.torques())
        for name, lbl in self._torque_labels.items():
            v = tau.get(name)
            if v is None:
                lbl.setText("--")
                continue
            lbl.setText(f"{v:+.1f}")
            if "finger" in name:
                tmax = 1.0
            else:
                try:
                    idx = int(name.rsplit("joint", 1)[1]) - 1
                except (ValueError, IndexError):
                    continue
                tmax = self._TAU_MAX[min(max(idx, 0), 6)]
            warn = abs(v) > 0.6 * tmax
            if bool(lbl.property("warn")) != warn:
                lbl.setProperty("warn", warn)
                lbl.style().unpolish(lbl)
                lbl.style().polish(lbl)

    def _update_slider_labels(self, pose: dict) -> None:
        """滑块默认显示**实时关节角**；只有用户按住拖动时才是目标。

        为避免"松手瞬间回弹"：松手后该关节进入 pending，滑块继续显示
        目标直到实测追上（|误差|<0.02 rad）才恢复跟随实测。
        """
        for name, (sld, val) in self._sliders.items():
            cur = pose.get(name)
            if cur is None or sld.isSliderDown():
                continue
            if name in self._pending_sliders:
                tgt = sld.value() / (1000.0 if "finger" in name else 100.0)
                tol = 0.001 if "finger" in name else 0.02
                if abs(tgt - cur) > tol:
                    continue            # 还在跟踪，保持目标显示
                self._pending_sliders.discard(name)
            if "finger" in name:
                v = int(max(0, min(44, round(cur * 1000))))
            else:
                v = int(max(-314, min(314, round(cur * 100))))
            if sld.value() != v:
                sld.blockSignals(True)
                sld.setValue(v)
                sld.blockSignals(False)
            if "finger" in name:
                val.setText(f"{cur * 1000:.0f} mm")
            else:
                val.setText(f"{cur:+.3f}")

    # ── 日志 ─────────────────────────────────────────
    def _append_log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {text}"
        if hasattr(self, "log_view"):
            self.log_view.appendPlainText(line)

    def _log(self, text: str) -> None:
        self.log_sig.emit(text)

    # ── 关闭 ─────────────────────────────────────────
    def closeEvent(self, ev) -> None:  # noqa: N802
        if self._conn_state == "connecting" or self._power_motion_pending or self._estopping:
            ev.ignore()
            return
        if self._has_enabled_arms():
            ev.ignore()
            self._after_disable = "close"
            self._on_enable_toggle()
            return
        if self.arms:
            self._disconnect()
            if self.arms:
                ev.ignore()
                return
        self._status_timer.stop()
        self._play_timer.stop()
        if self._recording:
            self._stop_record()             # 关窗口前把录制的帧落盘
        if self._worker is not None:
            self._worker.stop()
            self._worker.join(timeout=3.0)
            self._worker = None
        super().closeEvent(ev)
