# 解析 Studio 用的 Python 解释器（run_studio.sh 与 tools/ 下的脚本共用）。
# 顺序：OPENARM_STUDIO_PYTHON → conda ros2ws → PATH 上的 python3。
# 显式指定的解释器不可用时直接报错，不悄悄回退到别的环境。
# 用法：source <仓库>/tools/python_env.sh && PY="$(studio_python)"

studio_python() {
  local cand
  if [ -n "${OPENARM_STUDIO_PYTHON:-}" ]; then
    if command -v "$OPENARM_STUDIO_PYTHON" >/dev/null 2>&1; then
      command -v "$OPENARM_STUDIO_PYTHON"
      return 0
    fi
    echo "OPENARM_STUDIO_PYTHON 指定的解释器不可用：$OPENARM_STUDIO_PYTHON" >&2
    return 1
  fi
  for cand in "$HOME/anaconda3/envs/ros2ws/bin/python" python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      command -v "$cand"
      return 0
    fi
  done
  return 1
}
