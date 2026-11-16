#!/usr/bin/env bash
# 端到端验证：假硬件栈 + 控制器 + MoveIt 规划（不执行、不碰真电机）
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /opt/ros/jazzy/setup.bash
source "$HOME/openarm_ws/install/setup.bash"
cd "$HOME/openarm_studio"
export PYTHONPATH="$PWD:$PYTHONPATH"
exec "$HOME/anaconda3/envs/ros2ws/bin/python" tools/e2e_check.py
