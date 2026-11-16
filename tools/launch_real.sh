#!/usr/bin/env bash
# 起 OpenArm 规划栈 —— 全部用官方 launch，不自造：
#   1) openarm_bringup/openarm.bimanual.launch.py   （robot + controllers，launch_rviz 可关）
#   2) openarm_bimanual_moveit_config/move_group.launch.py（只起 move_group）
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

# 1) 官方 bringup（不弹 RViz）
ros2 launch openarm_bringup openarm.bimanual.launch.py \
  arm_type:=v10 \
  use_fake_hardware:="$FAKE" \
  robot_controller:=joint_trajectory_controller \
  launch_rviz:=false &
BRINGUP=$!

# 2) 官方 move_group（等控制器加载完再起，避免抢不到 action）
sleep 8
ros2 launch openarm_bimanual_moveit_config move_group.launch.py &
MOVEIT=$!

trap 'kill -INT $MOVEIT $BRINGUP 2>/dev/null || true' INT TERM
wait
