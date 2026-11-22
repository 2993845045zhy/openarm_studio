#!/usr/bin/env bash
# 安装桌面入口：之后在应用列表里搜索 "OpenArm Studio" 即可启动。
# 写入 ~/.local/share/applications/openarm-studio.desktop，并把图标装进用户图标目录。
# 可重复运行，会覆盖旧的入口。不需要 root。
#     bash tools/install_desktop_entry.sh
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/run_studio.sh" ] && [ -x "$ROOT/run_studio.sh" ] || {
  echo "run_studio.sh 不可执行：chmod +x $ROOT/run_studio.sh" >&2
  exit 1
}
case "$ROOT" in
  *[[:space:]]*|*%*)
    echo "仓库路径含空格或 %，桌面入口无法可靠启动，请把仓库放到纯英文无空格的路径下：$ROOT" >&2
    exit 1
    ;;
esac

DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
mkdir -p "$DATA/applications"
cat > "$DATA/applications/openarm-studio.desktop" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=OpenArm Studio
Name[zh_CN]=OpenArm 上位机
GenericName=Robot Workbench
Comment=Dual-arm digital twin, MuJoCo simulation and direct CAN control
Comment[zh_CN]=OpenArm 双臂数字孪生 / 仿真 / CAN 直连控制
Exec=$ROOT/run_studio.sh
Icon=openarm-studio
Terminal=false
Categories=Development;Science;
Keywords=robot;can;mujoco;openarm;
StartupWMClass=openarm_studio
StartupNotify=true
EOF
echo "已写入 $DATA/applications/openarm-studio.desktop"

# shellcheck source=python_env.sh
source "$ROOT/tools/python_env.sh"
if PY="$(studio_python)" && "$PY" -c "import PySide6.QtSvg" >/dev/null 2>&1; then
  QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}" "$PY" "$ROOT/tools/install_icon.py" >/dev/null
  echo "图标已安装到 $DATA/icons/hicolor"
else
  echo "未装图标：当前 Python 没有 PySide6（装好依赖后重新运行本脚本即可）" >&2
fi

command -v update-desktop-database >/dev/null 2>&1 \
  && update-desktop-database "$DATA/applications" >/dev/null 2>&1 || true
echo "完成。"
