#!/bin/bash
# CAN 重连：把 can0/can1 重置为 CAN-FD 1M/5M 并启用（等效拔插一次 USB 线）。
# 只由 openarm_studio 的「CAN 重连」按钮以 root 调用，接口名固定在本脚本内。
set -u

CONFIGURED=0
for iface in can0 can1; do
    if [ ! -d "/sys/class/net/$iface" ]; then
        echo "[can-reset] $iface 不存在（适配器未插入？）" >&2
        continue
    fi
    ip link set dev "$iface" down 2>/dev/null || true
    if ip link set dev "$iface" type can bitrate 1000000 sample-point 0.75 \
            dbitrate 5000000 dsample-point 0.75 fd on 2>/dev/null; then
        ip link set dev "$iface" up 2>/dev/null || true
        echo "[can-reset] $iface 已重置为 CAN-FD 1M/5M 并启用"
        CONFIGURED=$((CONFIGURED + 1))
    else
        echo "[can-reset] $iface 配置失败（适配器不支持 FD？）" >&2
    fi
done

[ "$CONFIGURED" -gt 0 ] || exit 1
exit 0
