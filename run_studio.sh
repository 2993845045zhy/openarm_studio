#!/usr/bin/env bash
# OpenArm Studio —— 双臂直连上位机（openarm_driver / CAN）
#
# 环境：任意装好 requirements.txt 的 Python。默认先找 conda ros2ws，其次 PATH 上的
#       python3；也可用 OPENARM_STUDIO_PYTHON 指定。外部模型路径见 README「安装」。
# GL：   MUJOCO_GL 默认 egl（离屏渲染，不弹官方 viewer）。

cd "$(dirname "$0")" || exit 1

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# shellcheck source=tools/python_env.sh
source tools/python_env.sh
if ! PY="$(studio_python)"; then
  echo "找不到 Python：请安装 python3 与 requirements.txt 里的依赖，或用 OPENARM_STUDIO_PYTHON 指定解释器" >&2
  exit 1
fi

exec "$PY" -m openarm_studio.app "$@"
