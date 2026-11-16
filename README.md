# OpenArm Studio

<div align="center">

**OpenArm 双臂机器人的上位机** —— 数字孪生 / MuJoCo 仿真 / 真机控制三合一。

设计语言沿用 [pbvs_workbench](../pbvs_workbench)（zcode / Zai：三面板壳层、
浅深双主题、Zai 调色板与尺寸标尺）；3D 场景是**内嵌 MuJoCo 渲染**，不弹官方 viewer。

</div>

## 现在能做什么

| 能力 | 状态 |
|---|---|
| 三面板壳层（导航轨 236 / Inspector 300 / 主卡） | ✅ |
| 内嵌 3D 双臂孪生（MuJoCo，EGL 离屏渲染进 Qt） | ✅ |
| 深浅主题切换 + 3D 场景配色同步 | ✅ |
| 鼠标取景（左拖旋转 / 右拖平移 / 滚轮缩放 / 双击复位）+ 视图预设 | ✅ |
| 关节滑块驱动孪生（左右臂各 7 关节 + 夹爪） | ✅ |
| 线框 / 半透明叠加、关节胶囊（实时角度条） | ✅ |
| ROS 2 桥：`/joint_states` → 孪生 | ⏳ P2 |
| 仿真模式（MuJoCo 物理 + 冒充真机的 ROS 接口） | ⏳ P3 |
| MoveIt 2 规划面板（Plan / Execute / 命名状态 / 预览） | ⏳ P4 |
| 真机直接控制 + 急停 + 启动前检查 | ⏳ P5 |

## 快速开始

```bash
./run_studio.sh                  # 真机模式（数字孪生）
./run_studio.sh --mode sim       # 仿真模式
./run_studio.sh --theme light    # 浅色主题
./run_studio.sh --screenshot out.png   # 离屏自检截图
```

环境：conda `ros2ws`（PySide6 + mujoco + rclpy + moveit_py **同进程可用**），
`MUJOCO_GL=egl` 由 `run_studio.sh` 设置。

## 结构

```
openarm_studio/
  app.py                 入口（--theme/--mode/--screenshot）
  model/urdf2mjcf.py     xacro→URDF→MuJoCo 预处理（见下）
  model/scene.py         场景装配：地板网格、三点光、纯色天空盒
  core/mjview.py         内嵌渲染后端（mujoco.Renderer，线程绑定 GL）
  ui/theme.py            Zai 双调色板 + QSS（token 按标识符边界替换）
  ui/shell.py            三面板主窗口 + 渲染线程
  ui/mjview_panel.py     3D 面板（显示 + 阻尼相机交互）
  ui/joint_monitor.py    关节胶囊
  ui/{switch,chrome,widgets,icons}.py   组件
tests/test_pose_render.py  姿态→渲染 冒烟测试
```

## 模型管线（踩过的坑，全部已在代码里处理）

1. **`.dae` 不支持** —— 视觉网格换成同名 `.stl`；没有则退到 `collision/*.stl`
   （手部 `hand.dae` 就是这种）。
2. **`package://` 不认** —— 换成绝对路径。
3. **手指连杆惯量不合法** —— `balanceinertia="true"`。
4. **URDF 的 `<mujoco>` 块只吃 compiler 选项** —— 地板/灯光/资产会被静默忽略，
   所以场景元素是在导出的 MJCF 文本上注入的。
5. **MuJoCo 默认 headlight 很亮** —— 会把地板打成白片、网格线冲掉；压到 0.30。
6. **skybox 用内置 `builtin="gradient"`（两端同色）** —— 纯色小图会渲染成黑。
7. **EGL 上下文线程绑定** —— 渲染器在渲染线程创建、在该线程 `close()`。
8. **`Renderer.render()` 复用缓冲** —— 返回前 `.copy()`。
9. **`vis.global_.ipd` 必须为 0** —— 否则 lookat 不在画面中心，取景/拖拽全偏。

## 已知环境怪癖

- 本机 Qt 6.11 + X11 下 `QWidget.grab()` 会段错误 → `--screenshot` 改用
  `QScreen.grabWindow()`。真机上界面显示不受影响。
