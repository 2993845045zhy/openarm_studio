#!/usr/bin/env bash
# OpenArm Studio —— 双臂直连上位机（openarm_driver / CAN，无 ROS）
#
# 环境：conda ros2ws（PySide6 + mujoco + openarm_driver + openarm_can>=1.4）
# GL：   MUJOCO_GL=egl（离屏渲染，不弹官方 viewer）

cd "$(dirname "$0")" || exit 1

export MUJOCO_GL=egl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

PY="$HOME/anaconda3/envs/ros2ws/bin/python"
if [ ! -x "$PY" ]; then
  echo "找不到 conda 环境 ~/anaconda3/envs/ros2ws/bin/python" >&2
  exit 1
fi

exec "$PY" -m openarm_studio.app "$@"
