#!/usr/bin/env bash
# 启动 OpenArm 真机栈（bringup + move_group）。
#
#   bash launch_real.sh          # 真硬件
#   bash launch_real.sh --fake   # 假硬件（无电机，可离线验证 UI/MoveIt 链路）
#
# 控制器与 MoveIt config 都沿用 openarm_bimanual_moveit_config 的 demo launch：
# 它同时拉起 robot_state_publisher + ros2_control_node + move_group，
# 是"真机 + MoveIt"的完整链路（不要再叠加 openarm.bimanual.launch.py，
# 两个 controller_manager 会抢同一个服务）。
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "$HOME/openarm_ws/install/setup.bash"

FAKE=false
for arg in "$@"; do
  [ "$arg" = "--fake" ] && FAKE=true
done

if [ "$FAKE" = true ]; then
  exec ros2 launch openarm_bimanual_moveit_config demo.launch.py \
    arm_type:=v10 use_fake_hardware:=true
else
  exec ros2 launch openarm_bimanual_moveit_config demo.launch.py \
    arm_type:=v10
fi
