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

"""Structural launch contract for the native MuJoCo payload showcase."""

import importlib.util
from pathlib import Path

from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = ROOT / "launch" / "payload_benchmark.launch.py"


def test_payload_launch_exposes_viewer_realtime_and_controller_choices():
    specification = importlib.util.spec_from_file_location(
        "payload_benchmark_launch",
        LAUNCH_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    description = module.generate_launch_description()
    arguments = {
        entity.name: entity.default_value
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    nodes = [
        entity for entity in description.entities if isinstance(entity, Node)
    ]

    assert {"controller", "viewer", "realtime", "payload_mass_kg"}.issubset(
        arguments
    )
    assert len(nodes) == 1
    assert nodes[0].node_executable == "payload_benchmark_node"
