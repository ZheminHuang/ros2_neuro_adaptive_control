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

"""Static regressions for the ROS 2 Humble RViz display configuration."""

from pathlib import Path


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "rviz"
    / "ur5e_robotiq_mujoco.rviz"
)


def test_marker_arrays_use_humble_ros_topic_property() -> None:
    """Prevent a plausible-looking config that silently creates no subscriber."""
    content = CONFIG.read_text(encoding="utf-8")
    assert content.count("Class: rviz_default_plugins/MarkerArray") == 2
    assert "Marker Topic:" not in content
    assert "Value: /mujoco/scene_markers" in content
    assert "Value: /mujoco/contact_markers" in content


def test_tf_tree_does_not_cover_robot_with_every_frame_axis() -> None:
    """Keep TF enabled while a dedicated TCP display supplies the visible axes."""
    content = CONFIG.read_text(encoding="utf-8")
    assert "Name: TF Tree" in content
    assert "Show Arrows: false" in content
    assert "Show Axes: false" in content
    assert "Name: MuJoCo TCP Frame" in content
