#!/usr/bin/env bash
# 只验证 MoveItClient 规划（假硬件栈需已启动；不执行、不碰电机）
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /opt/ros/jazzy/setup.bash
source "$HOME/openarm_ws/install/setup.bash"
cd "$HOME/openarm_studio"
export PYTHONPATH="$PWD:$PYTHONPATH"
exec "$HOME/anaconda3/envs/ros2ws/bin/python" tools/plan_check.py
