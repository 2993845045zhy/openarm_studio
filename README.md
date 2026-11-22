# OpenArm Studio

<div align="center">

**OpenArm v1.0 双臂机器人的桌面上位机** —— 3D 数字孪生 / MuJoCo 仿真 / CAN 直连真机控制三合一，不依赖 ROS。

A PySide6 desktop workbench for the OpenArm v1.0 bimanual arm: an embedded MuJoCo digital twin
and simulation, plus direct SocketCAN control of the real arms (gravity compensation, joint
impedance, teaching and playback, drag IK, smart grasping). No ROS 2 required.
[English](#english)

</div>

> **非官方项目。** 与 Enactic / OpenArm 官方团队没有隶属关系。本仓库是站在官方开源组件
> （`openarm_driver`、`openarm_can`、`openarm_control`、`openarm_description`、`openarm_mujoco`）
> 之上的上位机实现，"OpenArm" 名称归其权利人所有。

## 安全须知

本软件会直接向真实电机下发位置和力矩指令，中间没有任何硬件级的安全层。上真机前请读完：

1. **先仿真，后真机。** `./run_studio.sh --mode sim` 只跑内嵌 MuJoCo，不碰 CAN。
2. **手边留急停。** 本机实测 16 个电机的通信超时寄存器（`TIMEOUT`）都是 0。也就是说上位机崩溃、
   USB 松动或总线掉线之后，**已使能的电机不会自己失能，而是继续执行最后一条指令**，
   软件后来发出的失能指令也到不了它。这时只能切断电机动力。
3. **本机有一个没查清的左臂掉线问题**，见「已知问题」。左臂使能状态下工作时必须有人守着急停。
4. **夹爪没有力传感器。** 智能抓取靠电流上限约束夹持力，夹持力数值是估算值。不要夹手指和易碎品。
5. **所有参数都是在一台机器上调的。** 重力前馈来自官方 v1 模型，摩擦参数是官方在另一台机器上的
   标定值，速度、刚度和限位只在这一台 v1.0 上验证过。换一台机器请从低增益、小幅度开始。
6. **不要和 ROS 2 的 bringup 同时开。** 官方 ros2_control 和 Studio 会抢同一个 CAN 接口。

## 功能

- **3D 数字孪生**：内嵌 MuJoCo 的 3D 视图，独立渲染线程，实时跟随真机关节角。
- **仿真**：在内嵌 MuJoCo 里跑物理和位置伺服，不需要硬件。
- **真机直连**：经官方 `openarm_driver` / `openarm_can` 走 SocketCAN，双臂各 7 个关节加一个夹爪。
- **力控**：重力补偿、摩擦补偿、整臂关节阻抗（刚度和阻尼比连续可调）、零重力示教。
- **示教**：录制双臂关节角和夹爪开口为 CSV，回放时可以驱动真机，也可以只推 3D 视图。
- **末端拖拽 IK**：基于官方 `openarm_control`（mink + QP）。
- **双臂协同遥操**：手推主臂，从臂按关节增量跟随，可镜像。
- **夹爪**：位置加电流上限的力位控制；「智能抓取」闭合到物体上后持续保持，并能报出空夹和物体脱出。
- **运维**：只读的总线体检、CAN 重连、关节力矩显示、急停和失能回程。

界面目前只有中文。

## 已验证的环境

| 项 | 配置 |
| --- | --- |
| 机器人 | OpenArm v1.0（v10）：每臂 7 个关节加 1 个夹爪电机，达妙 DM8009×2、DM4340×2、DM4310×3，夹爪 DM4310 |
| CAN | PEAK PCAN-USB Pro FD（单适配器双通道），CAN-FD 仲裁 1 Mbit/s、数据 5 Mbit/s，SocketCAN |
| 系统 | Ubuntu 24.04 |
| Python | 3.12 |
| 显卡 | NVIDIA，EGL 离屏渲染（`MUJOCO_GL=egl`） |

没有在 OpenArm v2 或其他 CAN 适配器上测试过。

## 安装

```bash
# 1) 官方资源：3D 网格（openarm_description）和 v1 动力学模型（openarm_mujoco）
mkdir -p ~/openarm_ws/src && cd ~/openarm_ws
git clone https://github.com/enactic/openarm_description.git src/openarm_description   # 约 180 MB
git clone https://github.com/enactic/openarm_mujoco.git official_openarm_mujoco

# 2) Studio 和 Python 依赖（任意 Python 3.12 环境；用系统 Python 编译扩展需要 python3.12-dev）
git clone https://github.com/2993845045zhy/openarm_studio.git && cd openarm_studio
sudo apt install build-essential cmake libcli11-dev python3-venv python3.12-dev
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 3) CAN 接口：装一条 udev 规则，适配器插入时自动把 can0/can1 配成 CAN-FD 1M/5M
sudo bash tools/install_can_udev.sh
ip -br link show type can        # can0、can1 都应为 UP
```

说明：

- `openarm-can` 在 PyPI 上只有源码包，安装时要编译，所以需要 C++ 编译器、CMake 和 CLI11 头文件。
  没有 sudo 时把 CLI11 装到用户目录，再告诉构建过程去哪里找：

  ```bash
  git clone --depth 1 --branch v2.5.0 https://github.com/CLIUtils/CLI11.git
  cmake -S CLI11 -B CLI11/build && cmake --install CLI11/build --prefix ~/.local/opt/cli11-prefix
  pip install -r requirements.txt \
    --config-settings="cmake.args=-DCLI11_DIR=$HOME/.local/opt/cli11-prefix/share/cmake/CLI11"
  ```

- 两个官方仓库默认放在 `~/openarm_ws` 下。放在别处时用环境变量指定，见「配置」。
- `openarm_description` 的两代目录布局（`meshes/` 和 `assets/`）都能识别。

## 快速开始

```bash
./run_studio.sh --mode sim      # 仿真，不碰 CAN
./run_studio.sh --theme light   # 浅色主题
./run_studio.sh                 # 真机模式
bash tools/install_desktop_entry.sh   # 可选：在应用列表里加一个启动项
```

真机的顺序是：点「连接」，确认两臂关节角和实际姿态一致；点「总线体检」，确认两臂 7 个轴都有应答；
清空工作空间、手放急停，再点「使能」。

## 使用说明

### 双臂控制

- Studio 通过 SocketCAN 直连两条 CAN-FD 总线，默认右臂是 `can0`，左臂是 `can1`。连接和总线体检会区分发送队列满与未收到状态回包。失能时以 25 Hz 查询状态，发送队列满时暂降到 4 Hz。
- 真机模式支持双臂关节与夹爪控制、重力补偿、关节阻抗、末端拖拽和示教。仿真模式在内嵌 MuJoCo 中运行。
- 真机关节滑块的长行程峰值为 J1–J4 各 1.10 rad/s，J5、J6、J7 分别为 1.18、1.20、1.25 rad/s，且受设备配置限速约束。J1–J4 的加速和制动分别为 3.6、4.2 rad/s²，J5–J7 分别提高到 5.0–5.8 和 6.0–6.8 rad/s²，让腕部短行程更快起步并停稳。末端拖拽、回起始位和全部回零的目标速度每轴最高 0.60 rad/s。预设走位与回零会继续使用已经开启的重力补偿或关节阻抗。
- 开启重力补偿、关节阻抗或零重力示教时，失能回程继续发送重力前馈，关节目标限速 0.25 rad/s，并在收臂位反馈稳定后断力矩。反馈未稳定时保留使能。
- 使能时先以实测姿态建立重力前馈，再以每轴 0.40 rad/s 的目标速度上限走到起始位。补偿随后持续工作，直到手动关闭或失能。使能前已选中的关节阻抗或零重力示教也会继续使用各自的重力前馈。
- 使能后需要收到本次七轴的反馈并确认已上力矩。电机未确认使能时，日志会列出对应臂、CAN 接口和关节。如果双臂在线但仅一臂使能，可点「重试未使能臂」，已使能臂继续当前控制。
- 普通位置模式的失能回程和示教回放沿用原有时序。
- 摩擦补偿使用实测关节速度和官方标定模型，在重力补偿、零重力示教和关节阻抗中生效。接近静止时平滑减弱，反向或该轴反馈超时时撤掉旧助力，开关力矩渐变。
- 零重力示教时，J5–J7 采用较低的阻尼。低速转动时，驱动用连续同向的编码器位移估计低速转速并给出有限助力。慢速助力平滑过渡到速度反馈补偿，转向、静止、反馈过期或到达关节限位时撤回。

### 双臂协同遥操

在真机模式连接并使能双臂，等待两臂静止。在「双臂协同」选择手推哪只臂，再点「开始跟随」。主臂进入零重力拖动模式，从臂以各自开始时的姿态为基准跟随主臂的七个关节增量。默认开启「镜像跟随」。关闭后，关节增量根据双臂模型在零位的底座坐标系轴向映射。J1 和 J7 的左右轴方向相反，跟随时对编码器增量取反。其余五轴保持符号。停止后可切换方向或镜像设置，再重新开始。

协同从臂的目标速度上限为 J1–J4 各 1.05 rad/s、J5–J7 各 1.25 rad/s，并受设备配置限速和关节限位约束。每轴跟随目标限制在该轴实测位置的 ±0.08 rad 内，接近目标时减速。点击「停止跟随」、任一臂反馈超时或协同指令中断时，两臂退出跟随并保持姿态，重力补偿持续工作。协同卡片和日志显示退出原因。信号恢复后，点击「开始跟随」可重新对齐两臂。可在跟随期间录制双臂动作。

### 夹爪力位控制与智能抓取

真机使能后，在左、右夹爪控制行分别调节目标开口（0–44 mm）和电流上限。电流百分比以设备配置上限为基准：100% 实测堵转约 2.1 N·m，60% 约 1.3 N·m。夹指端的夹持力没有用力传感器标定，按机构行程比估算 60% 约为 20–30 N。

先张开夹爪，对准物体，再点击对应侧的「智能抓取」。夹爪以约 21 mm/s（0.5 rad/s）直接闭合到物体上，全程受电流上限约束；夹住后电机按这个上限持续输出，上位机不回退目标，保持力在提起和搬运期间不变。夹爪停在全闭位置（开口约 3 mm 以内）时显示「未夹到物体」；保持期间物体滑出、夹爪走到全闭时显示「物体已脱出」。点「张开」或拖动该侧夹爪滑块才松开。后续任务可通过 `DriverArm.grasp_status` 读取状态：`closing` 闭合中，`holding` 已夹持，`empty` 未夹到，`lost` 物体脱出。

机构摩擦实测 0.24–0.53 N·m，两爪各有一处紧点需要超过 0.43 N·m。面板低于 40% 时智能抓取按 40% 执行，否则夹爪会半途卡住；用滑块直接闭合时，面板过低同样可能卡住。

真机数据、阈值依据和复现实验的脚本见 `tests/fixtures/gripper_posforce/README.md` 与 `tools/gripper_posforce_probe.py`。

### 控制切换

切换力控或停止回放时，Studio 会取消旧目标并保持当前姿态。末端拖拽、示教回放、预设走位和双臂协同分别接管运动，手动拖关节滑块会结束末端拖拽或回放。回放到末帧会保持末帧姿态。实时回放中断后，重新播放需要主动点击播放。双臂录制要求两臂实测反馈完整，反馈中断时只保存已经录到的有效帧。

切换真机和仿真前先断开真机。已使能时关闭窗口或断开连接会先请求正常失能回程，回程中止时保留界面。急停请求优先于普通走位和旧指令。

## 配置

| 环境变量 | 作用 | 默认值 |
| --- | --- | --- |
| `OPENARM_STUDIO_PYTHON` | 启动脚本用的解释器 | conda 环境 `ros2ws`，找不到则用 PATH 上的 `python3` |
| `OPENARM_WS` | 两个官方仓库的存放目录 | `~/openarm_ws` |
| `OPENARM_MESH_DIR` | `openarm_description` 的位置：仓库根目录，或其中的 `meshes/`、`assets/` | `$OPENARM_WS/src/openarm_description` |
| `OPENARM_MUJOCO_V1_XML` | 官方 v1 双臂 MuJoCo 模型 | `$OPENARM_WS/official_openarm_mujoco/v1/openarm_bimanual.xml` |
| `MUJOCO_GL` | MuJoCo 的渲染后端 | `egl` |

缺少网格或 v1 模型时 3D 视图和末端拖拽不可用，日志页会写明找不到的路径；重力补偿会退到 KDL 后端。

驱动配置在界面里选，可选值见 `backend/driver_arm.py` 的 `CONFIGS`。本机的电机零位等于 URDF 零位，所以默认用不带偏移的 `openarm_pedestal`。如果你的机器零位约定不同，姿态显示和关节限位都会错，需要先换配置。

「CAN 重连」按钮通过 `sudo` 重置接口。想免密码运行，在 `sudo visudo` 里加一行（换成你的仓库绝对路径）：

```
你的用户名 ALL=(root) NOPASSWD: /usr/bin/bash /绝对路径/openarm_studio/tools/can_reset.sh
```

这条规则等于允许任何能改这个脚本的人以 root 执行命令，请自己权衡。

## 目录结构

```
openarm_studio/
  app.py                 入口（--theme / --mode / --screenshot）
  paths.py               外部资源的位置（环境变量）
  backend/
    driver_arm.py        每臂一个 DriverArm：独占控制线程，封装 openarm_driver
    gripper.py           智能抓取的状态机
    gravity.py           重力矩与摩擦：官方 v1 MuJoCo 模型，KDL 回退
    ik.py                末端逆运动学（openarm_control）
    teleop.py            双臂协同跟随
  core/
    mjview.py            MuJoCo 离屏渲染后端
    sim.py               仿真后端
  model/
    assets/              随附的双臂 URDF（衍生自 openarm_description）
    urdf2mjcf.py         URDF → MuJoCo 预处理：网格定位、按颜色拆分、镜像
    scene.py             装配地板、灯光和天空盒
  ui/
    shell.py             三面板主窗口、渲染线程、全部交互逻辑
    mjview_panel.py      3D 面板
    theme.py             浅/深两套调色板与 QSS
    icons.py             内嵌的 Lucide 图标
    zcode.py, switch.py, chrome.py, widgets.py, joint_monitor.py   其余组件
tools/                   CAN 配置与体检脚本、桌面入口安装、夹爪标定与自检
tests/                   单元与离屏界面测试；fixtures/gripper_posforce 是夹爪真机数据
docs/                    接口清单、环境审计、左臂掉线排查记录
```

## 测试

```bash
pip install pytest
MUJOCO_GL=egl QT_QPA_PLATFORM=offscreen PYTHONPATH=$PWD python -m pytest tests/ -q
```

测试不碰硬件。缺少可选依赖（PyKDL、`openarm_control`）或外部模型时，相关的测试会跳过并说明原因。
夹爪的判定逻辑用真机录下的数据回放来验证，数据在 `tests/fixtures/gripper_posforce/`。

## 已知问题

**左臂间歇掉线（未解决）。** 本机的左臂（`can1`）会间歇性整臂掉线：8 个轴同一瞬间不应答，之后适配器像
「总线上一个节点都没有」，只能重置接口恢复，等十几分钟也不会自愈。已经排除总线过载（掉线时负载最低
只有 8 帧/秒）、Studio 的代码、单个电机和两条通道互相干扰。故障定位在「适配器第 2 通道 ↔ 左臂」这段
通路上，是适配器还是线缆还没有区分开。排查记录见 [`docs/左臂掉线排查_2026-11-22.md`](docs/左臂掉线排查_2026-11-22.md)。
不确定是不是本机独有的问题；如果你也遇到，欢迎开 issue，并附上 `sudo dmesg | grep -i -E "pcan|peak"` 的输出。

掉线时：左臂处于使能状态就只能拍急停。之后在 Studio 里点「CAN 重连」，或执行
`sudo bash tools/can_reset.sh`。

其他限制：

- **摩擦参数不是本机标定的。** 装配和润滑不同就会有差别，出现「推不动、一动就窜」时先关掉摩擦补偿对比。
- **KDL 回退后端只有兜底精度。** 随附的 URDF 不含 v1 的精确惯性参数，重力前馈应尽量用官方 v1 MuJoCo 模型。
- **关节阻抗的刚度刻度是相对值。** 100% 等于官方增益，同样的刚度在不同姿态下手感会变。
- **截图依赖屏幕抓取。** 本机 Qt 6.11 + X11 下 `QWidget.grab()` 会段错误，所以 `--screenshot` 改用
  `QScreen.grabWindow()`，窗口如果被别的窗口挡住，截到的就是挡住它的内容。正常显示不受影响。

## 模型管线（踩过的坑，都已在代码里处理）

1. **`.dae` 不支持** —— 视觉网格换成同名 `.stl`；没有则退到 `collision/*.stl`
   （手部 `hand.dae` 就是这种情况）。能转换的 `.dae` 按材质颜色拆成多片 STL，再各自绑定颜色。
2. **网格路径** —— 从网格根目录解析为绝对文件路径，兼容 `openarm_description` 的两代目录布局。
3. **手指连杆惯量不合法** —— `balanceinertia="true"`。
4. **URDF 的 `<mujoco>` 块只吃 compiler 选项** —— 地板/灯光/资产会被静默忽略，
   所以场景元素是在导出的 MJCF 文本上注入的。
5. **MuJoCo 默认 headlight 很亮** —— 会把地板打成白片、网格线冲掉；压到 0.30。
6. **skybox 用内置 `builtin="gradient"`（两端同色）** —— 纯色小图会渲染成黑。
7. **EGL 上下文线程绑定** —— 渲染器在渲染线程创建、在该线程 `close()`。
8. **`Renderer.render()` 复用缓冲** —— 返回前 `.copy()`。
9. **`vis.global_.ipd` 必须为 0** —— 否则 lookat 不在画面中心，取景/拖拽全偏。

## 许可与第三方组件

本仓库的代码以 [MIT 许可](LICENSE)发布。随仓库分发的第三方衍生内容（Lucide 图标、衍生自
`openarm_description` 的 URDF、取自 `openarm_teleop` 的摩擦模型和参数）保留各自的许可，
清单见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## English

**OpenArm Studio** is an unofficial desktop control app for the OpenArm v1.0 bimanual arm.
It embeds a MuJoCo digital twin and simulation in a PySide6 window and talks to the real motors
directly over SocketCAN through the official `openarm_driver` / `openarm_can` packages, so no ROS 2
installation is needed. It provides gravity and friction compensation, continuously adjustable joint
impedance, zero-gravity teaching with CSV record and playback, drag-to-move IK, leader–follower
bimanual teleoperation, and a current-limited "smart grasp" for the gripper.

Status: developed and tested by one developer on a single OpenArm v1.0 with a PEAK PCAN-USB Pro FD
adapter (Ubuntu 24.04, Python 3.12). The UI is Chinese only. The motors on this robot do not
disable themselves when communication is lost, and the left arm drops off the bus intermittently
for reasons not yet found, so **keep a hand on the e-stop whenever the real arms are enabled**.

Install: clone [`openarm_description`](https://github.com/enactic/openarm_description) and
[`openarm_mujoco`](https://github.com/enactic/openarm_mujoco) into `~/openarm_ws`, then
`pip install -r requirements.txt` and run `./run_studio.sh --mode sim` for a hardware-free tour.
See the Chinese sections above for details; issues and pull requests are welcome.

License: MIT for this repository; third-party notices are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
