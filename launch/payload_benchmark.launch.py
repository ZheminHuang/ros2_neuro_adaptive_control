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

"""Launch the six-DoF unknown-payload MuJoCo benchmark."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Build the single-owner MuJoCo benchmark launch graph."""
    arguments = [
        DeclareLaunchArgument("controller", default_value="adaptive_nac"),
        DeclareLaunchArgument("viewer", default_value="true"),
        DeclareLaunchArgument("realtime", default_value="true"),
        DeclareLaunchArgument("payload_mass_kg", default_value="0.75"),
        DeclareLaunchArgument(
            "payload_com_offset_m",
            default_value="[0.004, -0.003, 0.002]",
        ),
        DeclareLaunchArgument("payload_inertia_scale", default_value="1.15"),
        DeclareLaunchArgument("seed", default_value="41"),
    ]
    node = Node(
        package="neuro_adaptive_control",
        executable="payload_benchmark_node",
        name="payload_benchmark",
        output="screen",
        parameters=[
            {
                "controller": LaunchConfiguration("controller"),
                "viewer": LaunchConfiguration("viewer"),
                "realtime": LaunchConfiguration("realtime"),
                "payload_mass_kg": LaunchConfiguration("payload_mass_kg"),
                "payload_com_offset_m": LaunchConfiguration(
                    "payload_com_offset_m"
                ),
                "payload_inertia_scale": LaunchConfiguration(
                    "payload_inertia_scale"
                ),
                "seed": LaunchConfiguration("seed"),
            }
        ],
    )
    return LaunchDescription(arguments + [node])
