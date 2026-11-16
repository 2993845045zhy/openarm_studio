#!/usr/bin/env python3
"""渲染夹爪特写，诊断装配问题（两根手指是否联动/对齐）。"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openarm_studio.core.mjview import MjEmbedRenderer  # noqa: E402
from openarm_studio.model.scene import build_model, joint_qpos_map  # noqa: E402
from openarm_studio.ui.theme import palette, scene_colors  # noqa: E402

m = build_model(Path.home() / ".cache/openarm_studio", scene_colors(palette("dark")))
qm = joint_qpos_map(m)
print("joints:", {k: v for k, v in qm.items() if "finger" in k})
print("neq (equalities):", m.neq)
for i in range(m.neq):
    print("  eq", i, "type", m.eq_type[i], "obj1", m.eq_obj1id[i], "obj2", m.eq_obj2id[i])

d = mujoco.MjData(m)

# 左臂抬到面前便于看夹爪
d.qpos[qm["openarm_left_joint1"]] = 0.5
d.qpos[qm["openarm_left_joint4"]] = 1.9
d.qpos[qm["openarm_left_joint6"]] = 0.5
mujoco.mj_forward(m, d)

r = MjEmbedRenderer(760)
cam = {"lookat": (0.0, 0.25, 0.45), "distance": 0.55, "azimuth": 120.0,
       "elevation": -15.0, "aspect": 1.25}
img = r.render(m, d, cam)
cv2.imwrite("/tmp/grip_closed.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# 张开 finger1，看 finger2 动不动
d.qpos[qm["openarm_left_finger_joint1"]] = 0.044
mujoco.mj_forward(m, d)
img2 = r.render(m, d, cam)
cv2.imwrite("/tmp/grip_open.png", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

print("saved /tmp/grip_closed.png /tmp/grip_open.png")
print("finger2 qpos after set finger1=0.044 + forward:",
      d.qpos[qm["openarm_left_finger_joint2"]])
r.close()
