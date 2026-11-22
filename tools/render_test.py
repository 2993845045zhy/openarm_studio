#!/usr/bin/env python3
"""P1 自检：构建双臂模型并离屏渲染一张图。

用法（Python 需装好 requirements.txt，可改用 OPENARM_STUDIO_PYTHON 指定）:
    MUJOCO_GL=egl python tools/render_test.py [theme]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openarm_studio.core.mjview import DEFAULT_VIEW, MjEmbedRenderer  # noqa: E402
from openarm_studio.model.scene import build_model, joint_qpos_map  # noqa: E402
from openarm_studio.ui.theme import palette, scene_colors  # noqa: E402


def main() -> int:
    theme_name = sys.argv[1] if len(sys.argv) > 1 else "dark"
    pal = palette(theme_name)
    sc = scene_colors(pal)

    cache = Path.home() / ".cache/openarm_studio"
    model = build_model(cache, sc)
    print(f"model: nq={model.nq} njnt={model.njnt} nbody={model.nbody} "
          f"ngeom={model.ngeom} nlight={model.nlight}")

    qmap = joint_qpos_map(model)
    wanted = [f"openarm_{side}_joint{i}" for side in ("left", "right") for i in range(1, 8)]
    missing = [n for n in wanted if n not in qmap]
    print(f"joint map: {len(qmap)} entries, missing={missing}")

    data = mujoco.MjData(model)
    # 摆一个可辨认的姿势：两臂 J4 抬肘、J1 外展
    for side, sign in (("left", 1.0), ("right", -1.0)):
        data.qpos[qmap[f"openarm_{side}_joint1"]] = 0.35 * sign
        data.qpos[qmap[f"openarm_{side}_joint4"]] = 0.9
        data.qpos[qmap[f"openarm_{side}_joint6"]] = 0.25
    mujoco.mj_forward(model, data)

    renderer = MjEmbedRenderer(height=720)
    cam = dict(DEFAULT_VIEW)
    cam["aspect"] = 16 / 9
    rgb = renderer.render(model, data, cam)
    print(f"frame: {rgb.shape} mean={rgb.mean():.1f}")

    out = ROOT / f"docs/render_{theme_name}.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"saved: {out}")
    renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
