#!/usr/bin/env bash
# 起 OpenArm 规划栈 —— 全部用官方 launch。
#
# move_group 与 bringup **并行**启动：move_group 只依赖 /robot_description
# 与 TF，不需要等控制器 spawn 完（之前的固定 sleep 8 纯属浪费）。
#
#   bash launch_real.sh          # 假硬件（默认，不碰电机）
#   bash launch_real.sh --real   # 真硬件
set -e
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "$HOME/openarm_ws/install/setup.bash"

FAKE=true
for arg in "$@"; do
  [ "$arg" = "--real" ] && FAKE=false
done

ros2 launch openarm_bimanual_moveit_config move_group.launch.py &
MOVEIT=$!

ros2 launch openarm_bringup openarm.bimanual.launch.py \
  arm_type:=v10 \
  use_fake_hardware:="$FAKE" \
  robot_controller:=joint_trajectory_controller \
  launch_rviz:=false &
BRINGUP=$!

trap 'kill -INT $MOVEIT $BRINGUP 2>/dev/null || true' INT TERM
wait
