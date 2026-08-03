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

"""Launch the automated MuJoCo grasp/lift/hold demo with RViz."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Reuse the trajectory graph with the deterministic grasp scenario."""
    included = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("neuro_adaptive_control"),
                    "launch",
                    "ur5e_mujoco_rviz.launch.py",
                ]
            )
        ),
        launch_arguments={
            "start_rviz": LaunchConfiguration("start_rviz"),
            "start_mujoco_viewer": LaunchConfiguration("start_mujoco_viewer"),
            "duration_sec": "11.0",
            "adaptation_enabled": "true",
            "external_wrench_mode": "none",
            "metrics_path": LaunchConfiguration("metrics_path"),
            "scenario": "grasp",
        }.items(),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("start_rviz", default_value="true"),
            DeclareLaunchArgument(
                "start_mujoco_viewer", default_value="false"
            ),
            DeclareLaunchArgument("metrics_path", default_value=""),
            included,
        ]
    )
