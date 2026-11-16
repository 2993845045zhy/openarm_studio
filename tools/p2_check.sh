#!/usr/bin/env bash
# P2 验证：起一个假 /joint_states 发布器，再让上位机截图，
# 看孪生有没有跟着动（不碰硬件、不发指令）。
set -e
cd "$(dirname "$0")/.."

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
[ -f "$HOME/openarm_ws/install/setup.bash" ] && source "$HOME/openarm_ws/install/setup.bash"

PY="$HOME/anaconda3/envs/ros2ws/bin/python"
OUT="${1:-/tmp/p2_twin.png}"

"$PY" tools/fake_joint_states.py > /tmp/fake_js.log 2>&1 &
PUB=$!
trap 'kill $PUB 2>/dev/null || true' EXIT

sleep 2
MUJOCO_GL=egl DISPLAY="${DISPLAY:-:1}" PYTHONPATH="$PWD:$PYTHONPATH" \
  "$PY" -m openarm_studio.app --screenshot "$OUT"

kill $PUB 2>/dev/null || true
echo "done: $OUT"
