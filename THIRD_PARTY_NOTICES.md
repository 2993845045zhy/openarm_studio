# 第三方组件声明

本仓库自身的代码以 MIT 许可发布（见 [LICENSE](LICENSE)）。仓库里随附的少量第三方衍生内容
保留各自的许可；运行时依赖不随本仓库分发，由使用者自行安装。

## 1. Lucide 图标 —— `openarm_studio/ui/icons.py`

界面图标内嵌为 SVG 路径数据，取自 [Lucide](https://lucide.dev)（`lucide-icons/lucide`）。
Lucide 以 ISC 许可发布；其中源自 Feather 项目的图标同时适用 MIT 许可。本仓库用到的
Feather 衍生图标：`check`、`chevron-down`、`chevron-right`、`maximize`、`moon`、`target`、
`terminal`，以及改过名的 `close`（Lucide `x`）和 `alert`（Lucide `triangle-alert`，原名
`alert-triangle`）。

以下为 Lucide 仓库 `LICENSE` 的原文：

```
ISC License

Copyright (c) 2026 Lucide Icons and Contributors

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

---

The following Lucide icons are derived from the Feather project:

airplay, alert-circle, alert-octagon, alert-triangle, aperture, arrow-down-circle, arrow-down-left, arrow-down-right, arrow-down, arrow-left-circle, arrow-left, arrow-right-circle, arrow-right, arrow-up-circle, arrow-up-left, arrow-up-right, arrow-up, at-sign, calendar, cast, check, chevron-down, chevron-left, chevron-right, chevron-up, chevrons-down, chevrons-left, chevrons-right, chevrons-up, circle, clipboard, clock, code, columns, command, compass, corner-down-left, corner-down-right, corner-left-down, corner-left-up, corner-right-down, corner-right-up, corner-up-left, corner-up-right, crosshair, database, divide-circle, divide-square, dollar-sign, download, external-link, feather, frown, hash, headphones, help-circle, info, italic, key, layout, life-buoy, link-2, link, loader, lock, log-in, log-out, maximize, meh, minimize, minimize-2, minus-circle, minus-square, minus, monitor, moon, more-horizontal, more-vertical, move, music, navigation-2, navigation, octagon, pause-circle, percent, plus-circle, plus-square, plus, power, radio, rss, search, server, share, shopping-bag, sidebar, smartphone, smile, square, table-2, tablet, target, terminal, trash-2, trash, triangle, tv, type, upload, x-circle, x-octagon, x-square, x, zoom-in, zoom-out

The MIT License (MIT) (for the icons listed above)

Copyright (c) 2013-present Cole Bemis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. 双臂 URDF —— `openarm_studio/model/assets/openarm_v10_bimanual.urdf`

由 [enactic/openarm_description](https://github.com/enactic/openarm_description)（v10 版本）
的描述文件整理而来：把左右臂合成为一个静态 URDF，去掉了 ROS 2 相关标签，并在文件头注明来源。
上游版权 Copyright 2025 Enactic, Inc.，以 Apache License 2.0 发布，全文见同目录的
[`LICENSE.txt`](openarm_studio/model/assets/LICENSE.txt)。本文件继续按 Apache-2.0 分发。

## 3. 关节摩擦模型与参数 —— `openarm_studio/backend/gravity.py`

摩擦模型 `τ = Fo + Fv·ω + Fc·tanh(0.1·k·ω)` 及逐关节参数取自
[enactic/openarm_teleop](https://github.com/enactic/openarm_teleop)（`src/controller/control.cpp`
里的 `ComputeFriction`，以及 `config/leader.yaml`、`config/follower.yaml` 的前 7 项）。
上游版权 Copyright 2025 Enactic, Inc.，Apache License 2.0。这里的实现是按该模型重写的 Python 版本。

## 4. 运行时依赖（不随本仓库分发）

| 依赖 | 用途 | 许可 |
| --- | --- | --- |
| [openarm_driver](https://github.com/enactic/openarm_driver) / [openarm_can](https://github.com/enactic/openarm_can) / [openarm_control](https://github.com/enactic/openarm_control) | 电机驱动封装、CAN 绑定、IK | Apache-2.0 |
| openarm_description、openarm_mujoco（运行时读取，不打包） | 渲染网格、v1 动力学模型 | Apache-2.0 |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | 仿真与离屏渲染 | Apache-2.0 |
| [Qt for Python / PySide6](https://doc.qt.io/qtforpython-6/) | 界面 | LGPL-3.0（及商业许可） |
| [NumPy](https://numpy.org) | 数值计算 | BSD-3-Clause |
| [trimesh](https://github.com/mikedh/trimesh) | 网格处理 | MIT |
| pycollada（经 trimesh 读取 COLLADA） | 网格处理 | BSD |
| [OpenCV](https://opencv.org) | 图像缩放 | Apache-2.0 |
| mink（经 openarm_control 引入） | 微分 IK | Apache-2.0 |
| daqp（经 openarm_control 引入） | 微分 IK 的 QP 求解 | MIT |
| PyKDL、urdf_parser_py（可选，KDL 回退） | 重力矩的备用后端 | LGPL-2.1、BSD-3-Clause |
