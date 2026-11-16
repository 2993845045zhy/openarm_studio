"""OpenArm Studio 自带 launch：bringup + move_group，**不启动 RViz**。

我们自己的 Qt 上位机就是可视化端（内嵌 MuJoCo + 规划面板），不需要
再弹 RViz 抢焦点。结构照 openarm_bimanual_moveit_config 的 demo.launch.py，
只去掉 rviz2。

假硬件/真机由环境变量 OPENARM_FAKE 决定（launch_real.sh 里设置）——
URDF 必须在 generate 阶段就定下 use_fake_hardware，不能用 LaunchConfiguration
延迟求值，所以走环境变量而不是 launch 参数。
"""
import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from pathlib import Path


def generate_launch_description() -> LaunchDescription:
    fake = os.environ.get("OPENARM_FAKE", "1") == "1"

    desc_dir = Path(get_package_share_directory("openarm_description"))
    robot_description = {
        "robot_description": xacro.process_file(
            str(desc_dir / "urdf/robot/v10.urdf.xacro"),
            mappings={
                "arm_type": "v10",
                "bimanual": "true",
                "hand": "true",
                "use_fake_hardware": "true" if fake else "false",
                "ros2_control": "true",
                "right_can_interface": os.environ.get("OPENARM_RIGHT_CAN", "can0"),
                "left_can_interface": os.environ.get("OPENARM_LEFT_CAN", "can1"),
            },
        ).toxml()
    }

    controllers_yaml = str(
        Path(get_package_share_directory("openarm_bringup"))
        / "config/v10_controllers/openarm_v10_bimanual_controllers.yaml")

    moveit_config = (
        MoveItConfigsBuilder("openarm", package_name="openarm_bimanual_moveit_config")
        .to_moveit_configs()
    )

    def spawner(names: str) -> Node:
        return Node(package="controller_manager", executable="spawner",
                    output="screen", arguments=names.split())

    actions = [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             output="screen", parameters=[robot_description]),
        Node(package="controller_manager", executable="ros2_control_node",
             output="screen",
             # YAML 作为"参数文件路径"传入（字符串），不能包成 dict
             parameters=[robot_description, controllers_yaml]),
        Node(package="moveit_ros_move_group", executable="move_group",
             output="screen", parameters=[moveit_config.to_dict()]),
        TimerAction(period=2.0, actions=[spawner("joint_state_broadcaster")]),
        TimerAction(period=3.0, actions=[spawner(
            "left_joint_trajectory_controller right_joint_trajectory_controller")]),
        TimerAction(period=4.0, actions=[spawner(
            "left_gripper_controller right_gripper_controller")]),
    ]
    return LaunchDescription(actions)
