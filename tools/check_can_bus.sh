#!/usr/bin/env bash
# 双臂 CAN 总线体检：对每条总线的 7 个电机轴发 5 轮查询，报应答数。
# 用法：bash check_can_bus.sh            （can0/can1 都测）
#       bash check_can_bus.sh can1       （只测左臂）
# 判读：无应答表示未收到匹配的电机状态帧。先区分发送队列满和成功发送后无回包。
# shellcheck source=python_env.sh
source "$(dirname "$(readlink -f "$0")")/python_env.sh"
PY="$(studio_python)" || { echo "找不到 Python（可用 OPENARM_STUDIO_PYTHON 指定）"; exit 1; }
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
        o.recv_all(first_timeout_us=10_000)
        per = []
        alive = 0
        for i in range(7):
            ls = o.get_arm().get_link_stats(i)
            per.append(f"J{i+1} {ls.responses}/{ls.commands_sent}")
            alive += ls.responses > 0
        print("  各轴 应答/发送:", "  ".join(per))
        queue_full = o.get_bus_status().write_no_buffer.count
        if queue_full:
            print(f"  结论: 发送队列满 {queue_full} 次，{alive}/7 轴有状态回包")
        elif alive == 7:
            pos = [round(m.get_position(), 3) for m in o.get_arm().get_motors()]
            print(f"  结论: 在线 ✅  位置(rad): {pos}")
        elif alive == 0:
            print("  结论: ❌ 0/7 轴有状态回包（查供电、急停、CAN 线和电机 ID）")
        else:
            print(f"  结论: ⚠️ 部分在线 {alive}/7")
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
