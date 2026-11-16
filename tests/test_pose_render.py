#!/usr/bin/env python3
"""Smoke test: 姿态写入是否真的改变了渲染结果。

不依赖 GUI，直接跑渲染线程的核心路径。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openarm_studio.core.mjview import DEFAULT_VIEW, MjEmbedRenderer
from openarm_studio.model.scene import build_model, joint_qpos_map
from openarm_studio.ui.theme import palette, scene_colors


def render(model, data, renderer):
    cam = dict(DEFAULT_VIEW)
    cam["aspect"] = 1.4
    return renderer.render(model, data, cam)


def main() -> int:
    cache = Path.home() / ".cache/openarm_studio"
    model = build_model(cache, scene_colors(palette("dark")))
    qmap = joint_qpos_map(model)
    data = mujoco.MjData(model)
    renderer = MjEmbedRenderer(640)

    mujoco.mj_forward(model, data)
    home = render(model, data, renderer).astype(np.int16)

    for side, sign in (("left", 1.0), ("right", -1.0)):
        data.qpos[qmap[f"openarm_{side}_joint1"]] = 0.6 * sign
        data.qpos[qmap[f"openarm_{side}_joint4"]] = 1.4
        data.qpos[qmap[f"openarm_{side}_joint6"]] = 0.4
    mujoco.mj_forward(model, data)
    posed = render(model, data, renderer).astype(np.int16)

    diff = np.abs(home - posed)
    changed = float((diff > 8).mean())
    print(f"frames: {home.shape}, mean abs diff={diff.mean():.2f}, "
          f"changed pixels={changed * 100:.1f}%")

    # 夹爪开合也必须可见
    data.qpos[qmap["openarm_left_finger_joint1"]] = 0.044
    data.qpos[qmap["openarm_right_finger_joint1"]] = 0.044
    mujoco.mj_forward(model, data)
    gripper = render(model, data, renderer).astype(np.int16)
    gdiff = float((np.abs(posed - gripper) > 8).mean())
    print(f"gripper open changed pixels={gdiff * 100:.2f}%")

    renderer.close()
    ok = changed > 0.02 and gdiff > 0.0005
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
