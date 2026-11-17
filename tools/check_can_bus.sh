#!/usr/bin/env bash
# 双臂 CAN 总线体检：对每条总线的 7 个电机轴发 5 轮查询，报应答数。
# 用法：bash check_can_bus.sh            （can0/can1 都测）
#       bash check_can_bus.sh can1       （只测左臂）
# 判读：❌ 无应答 = 总线上没有任何设备 ACK（断电/急停/线断），软件层已无嫌疑。
PY="$HOME/anaconda3/envs/ros2ws/bin/python"
[ -x "$PY" ] || { echo "找不到 $PY"; exit 1; }
exec "$PY" - "$@" <<'EOF'
import sys
import time

import openarm_can as oa

IFS = sys.argv[1:] or ["can0", "can1"]
LABEL = {"can0": "右臂", "can1": "左臂"}
TYPES = [oa.MotorType.DM8009] * 2 + [oa.MotorType.DM4340] * 2 + [oa.MotorType.DM4310] * 3
SEND = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
RECV = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]

for iface in IFS:
    label = LABEL.get(iface, "?")
    print(f"=== {iface}（{label}）===")
    try:
        o = oa.OpenArm(iface, True)
        o.init_arm_motors(TYPES, SEND, RECV)
        o.set_callback_mode_all(oa.CallbackMode.STATE)
        for _ in range(5):
            o.refresh_all()
            o.recv_all(first_timeout_us=100_000)
            time.sleep(0.1)
        per = []
        alive = 0
        for i in range(7):
            ls = o.get_arm().get_link_stats(i)
            per.append(f"J{i+1} {ls.responses}/{ls.commands_sent}")
            alive += ls.responses > 0
        print("  各轴 应答/发送:", "  ".join(per))
        if alive >= 5:
            pos = [round(m.get_position(), 3) for m in o.get_arm().get_motors()]
            print(f"  结论: 在线 ✅  位置(rad): {pos}")
        elif alive == 0:
            print("  结论: ❌ 0/7 应答 —— 总线上没有设备 ACK（查供电/急停/CAN 线）")
        else:
            print(f"  结论: ⚠️ 部分在线 {alive}/7 —— 有电机掉链或损坏")
        del o
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 打不开接口: {type(exc).__name__}: {exc}")
    time.sleep(0.3)

print("""
排查顺序（左臂 0 应答时）：
 1 左臂电源：关节电机有没有 LED/上电声（和右臂对比）
 2 急停：串在左臂供电链上的急停是否被拍下
 3 CAN 线：PCAN 适配器第 2 通道 → 左臂底座，两端插头重新插紧
 4 换线对照：左右 CAN 线在适配器上对调再跑本脚本——
   问题跟着线走 = 左臂侧问题；问题留在 can1 = 适配器通道问题
""")
EOF
