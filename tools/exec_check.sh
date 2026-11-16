#!/usr/bin/env bash
# e2e：规划 + 执行 + 关节到位校验（需先起栈；假硬件即可）
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /opt/ros/jazzy/setup.bash
source "$HOME/openarm_ws/install/setup.bash"
cd "$HOME/openarm_studio"
export PYTHONPATH="$PWD:$PYTHONPATH"
exec "$HOME/anaconda3/envs/ros2ws/bin/python" tools/exec_check.py
