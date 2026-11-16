#!/usr/bin/env bash
# OpenArm Studio —— 数字孪生 / 仿真 / 真机三合一上位机
#
# 环境：conda ros2ws（PySide6 + mujoco + rclpy + moveit_py 同进程可用）
# GL：   MUJOCO_GL=egl（离屏渲染，不弹官方 viewer）

cd "$(dirname "$0")" || exit 1

export MUJOCO_GL=egl
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# ROS 2 环境（真机/仿真模式需要；纯孪生也靠它订阅 /joint_states）
if [ -f /opt/ros/jazzy/setup.bash ]; then
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
fi
if [ -f "$HOME/openarm_ws/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "$HOME/openarm_ws/install/setup.bash"
fi

PY="$HOME/anaconda3/envs/ros2ws/bin/python"
if [ ! -x "$PY" ]; then
  echo "找不到 conda 环境 ~/anaconda3/envs/ros2ws/bin/python" >&2
  exit 1
fi

exec "$PY" -m openarm_studio.app "$@"
