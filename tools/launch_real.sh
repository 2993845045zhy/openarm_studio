#!/usr/bin/env bash
# 启动 OpenArm 规划栈（bringup + move_group），**不弹 RViz**。
#
#   bash launch_real.sh          # 假硬件（默认，不碰电机）
#   bash launch_real.sh --real   # 真硬件（驱动电机，注意安全）
set -e
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "$HOME/openarm_ws/install/setup.bash"
cd "$HOME/openarm_studio"

export PYTHONPATH="$PWD:$PYTHONPATH"
export OPENARM_FAKE=1
for arg in "$@"; do
  [ "$arg" = "--real" ] && OPENARM_FAKE=0
done
export OPENARM_RIGHT_CAN=can0
export OPENARM_LEFT_CAN=can1

exec ros2 launch openarm_studio/launch/studio_stack.launch.py
