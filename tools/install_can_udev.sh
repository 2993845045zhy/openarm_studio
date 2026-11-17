#!/usr/bin/env bash
# 一次性安装：PCAN 适配器插入时自动把 can0/can1 配成 CAN-FD 1M/5M 并启用。
# USB 重插后接口会被内核按默认重建（DOWN + 经典 CAN），此规则免去每次手动
# sudo ip link set ...。只需运行一次：
#     sudo bash ~/openarm_studio/tools/install_can_udev.sh
set -eu

[ "$(id -u)" -eq 0 ] || { echo "请用 sudo 运行"; exit 1; }

# 1) 网络配置脚本（udev 触发时以 root 执行）
cat > /usr/local/sbin/openarm-can-up.sh <<'SCRIPT'
#!/bin/bash
# udev 调用：%k = 接口名（can0/can1）。等待驱动完成初始化再配置。
iface="$1"
for _ in $(seq 1 20); do
    [ -d "/sys/class/net/$iface" ] && break
    sleep 0.1
done
[ -d "/sys/class/net/$iface" ] || exit 0
ip link set dev "$iface" down 2>/dev/null || true
ip link set dev "$iface" type can bitrate 1000000 sample-point 0.75 \
    dbitrate 5000000 dsample-point 0.75 fd on
ip link set dev "$iface" up
SCRIPT
chmod 755 /usr/local/sbin/openarm-can-up.sh

# 2) udev 规则：peak_usb 驱动注册的网络接口出现即触发
cat > /etc/udev/rules.d/99-openarm-can.rules <<'RULE'
ACTION=="add", SUBSYSTEM=="net", KERNEL=="can[0-9]*", DRIVERS=="peak_usb", \
    RUN+="/usr/local/sbin/openarm-can-up.sh %k"
RULE

udevadm control --reload-rules

# 3) 顺手把当前已存在的接口立即配好（不用重插）
for ifc in can0 can1; do
    [ -d "/sys/class/net/$ifc" ] && /usr/local/sbin/openarm-can-up.sh "$ifc" || true
done

echo "已安装 udev 规则并配置当前接口："
ip -br link show type can || true
