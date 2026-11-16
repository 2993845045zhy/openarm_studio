#!/usr/bin/env bash
# 连接诊断：查我们的客户端能否看到 controller_manager / move_action
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /opt/ros/jazzy/setup.bash
source "$HOME/openarm_ws/install/setup.bash"
cd "$HOME/openarm_studio"
export PYTHONPATH="$PWD:$PYTHONPATH"
exec "$HOME/anaconda3/envs/ros2ws/bin/python" tools/conn_diag.py
