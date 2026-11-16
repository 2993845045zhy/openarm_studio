#!/usr/bin/env bash
# 测连接耗时
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
pkill -INT -f "openarm.bimanual.launch.py" 2>/dev/null || true
pkill -INT -f "move_group.launch.py" 2>/dev/null || true
pkill -f ros2_control_node 2>/dev/null || true
pkill -f "lib/moveit_ros_move_group/move_group" 2>/dev/null || true
sleep 3
source /opt/ros/jazzy/setup.bash
source "$HOME/openarm_ws/install/setup.bash"
cd "$HOME/openarm_studio"
export PYTHONPATH="$PWD:$PYTHONPATH"
exec "$HOME/anaconda3/envs/ros2ws/bin/python" tools/time_connect.py
