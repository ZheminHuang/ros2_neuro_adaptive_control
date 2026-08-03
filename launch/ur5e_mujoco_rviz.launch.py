# Copyright 2026 Zhemin Huang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch one MuJoCo UR5e/2F-85 plant with display-only RViz."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import FindExecutable


def generate_launch_description() -> LaunchDescription:
    """Create the full-dynamics trajectory and RViz process graph."""
    share = FindPackageShare("neuro_adaptive_control")
    config = PathJoinSubstitution([share, "config", "mujoco_ur5e_robotiq.yaml"])
    xacro_file = PathJoinSubstitution(
        [share, "urdf", "ur5e_robotiq_2f85.urdf.xacro"]
    )
    rviz_config = PathJoinSubstitution(
        [share, "rviz", "ur5e_robotiq_mujoco.rviz"]
    )
    robot_description = ParameterValue(
        Command([FindExecutable(name="xacro"), " ", xacro_file]),
        value_type=str,
    )
    start_rviz = LaunchConfiguration("start_rviz")
    start_viewer = LaunchConfiguration("start_mujoco_viewer")
    scenario = LaunchConfiguration("scenario")
    trajectory = LaunchConfiguration("trajectory")
    duration = LaunchConfiguration("duration_sec")
    adaptation = LaunchConfiguration("adaptation_enabled")
    wrench_mode = LaunchConfiguration("external_wrench_mode")
    metrics_path = LaunchConfiguration("metrics_path")
    arguments = [
        DeclareLaunchArgument("start_rviz", default_value="true"),
        DeclareLaunchArgument("start_mujoco_viewer", default_value="false"),
        DeclareLaunchArgument("scenario", default_value="trajectory"),
        DeclareLaunchArgument("trajectory", default_value="circle"),
        DeclareLaunchArgument("duration_sec", default_value="12.0"),
        DeclareLaunchArgument("adaptation_enabled", default_value="true"),
        DeclareLaunchArgument("external_wrench_mode", default_value="none"),
        DeclareLaunchArgument("metrics_path", default_value=""),
    ]
    plant = Node(
        package="neuro_adaptive_control",
        executable="mujoco_ur5e_plant_node",
        name="mujoco_ur5e_plant",
        output="screen",
        emulate_tty=True,
        parameters=[
            config,
            {
                "scenario": scenario,
                "trajectory": trajectory,
                "duration_sec": ParameterValue(duration, value_type=float),
                "adaptation_enabled": ParameterValue(
                    adaptation, value_type=bool
                ),
                "external_wrench_mode": wrench_mode,
                "start_mujoco_viewer": ParameterValue(
                    start_viewer, value_type=bool
                ),
                "metrics_path": metrics_path,
            },
        ],
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
        output="screen",
    )
    bridge = Node(
        package="neuro_adaptive_control",
        executable="mujoco_rviz_bridge",
        name="mujoco_rviz_bridge",
        parameters=[{"use_sim_time": True}],
        output="screen",
    )
    gripper = Node(
        package="neuro_adaptive_control",
        executable="robotiq_gripper_action_server",
        name="robotiq_gripper_action_server",
        parameters=[config, {"use_sim_time": True}],
        output="screen",
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(start_rviz),
        output="screen",
    )
    shutdown = RegisterEventHandler(
        OnProcessExit(
            target_action=plant,
            on_exit=[EmitEvent(event=Shutdown(reason="MuJoCo plant exited"))],
        )
    )
    return LaunchDescription(
        arguments
        + [plant, robot_state_publisher, bridge, gripper, rviz, shutdown]
    )
