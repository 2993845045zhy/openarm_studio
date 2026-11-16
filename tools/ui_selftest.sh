#!/usr/bin/env bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /opt/ros/jazzy/setup.bash
source "$HOME/openarm_ws/install/setup.bash"
cd "$HOME/openarm_studio"
export PYTHONPATH="$PWD:$PYTHONPATH"
export MUJOCO_GL=egl
export DISPLAY="${DISPLAY:-:1}"
# -u：无缓冲，重定向到文件时也能看到实时进度
exec "$HOME/anaconda3/envs/ros2ws/bin/python" -u tools/ui_selftest.py
