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

"""Regression tests for safety behavior in the ROS MuJoCo plant owner."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


rclpy = pytest.importorskip("rclpy")
mujoco = pytest.importorskip("mujoco")
services = pytest.importorskip("std_srvs.srv")
simulation_module = pytest.importorskip(
    "neuro_adaptive_control.adapters.mujoco_simulation"
)
node_module = pytest.importorskip(
    "neuro_adaptive_control.nodes.mujoco_ur5e_plant_node"
)


Trigger = services.Trigger
SimulationState = simulation_module.SimulationState
MujocoUR5ePlantNode = node_module.MujocoUR5ePlantNode
pytestmark = pytest.mark.skipif(
    not hasattr(mujoco, "MjModel"),
    reason="official MuJoCo Python bindings are not installed",
)


@pytest.fixture()
def plant_node():
    """Create a real plant node without allowing its timer to run itself."""
    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init(args=[])
    node = MujocoUR5ePlantNode()
    node._timer.cancel()
    try:
        yield node
    finally:
        node.destroy_node()
        if owns_context and rclpy.ok():
            rclpy.shutdown()


def test_post_step_workspace_violation_faults_before_clean_finish(
    plant_node: MujocoUR5ePlantNode,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Guard the final post-integration sample before metrics or stopping."""
    real_advance = plant_node._plant.advance

    def violating_advance(*args, **kwargs):
        sample = real_advance(*args, **kwargs)
        return replace(sample, tcp_position=np.array((-0.90, 0.40, 0.40)))

    monkeypatch.setattr(plant_node._plant, "advance", violating_advance)
    metrics_path = tmp_path / "must_not_be_clean.json"
    plant_node._duration = 0.002
    plant_node._metrics_path = str(metrics_path)

    plant_node._on_control_tick()

    assert plant_node._step_index == 1
    assert plant_node._state == SimulationState.FAULT
    assert plant_node._controller.state.value == "fault"
    assert "workspace" in plant_node._reason.lower()
    assert not metrics_path.exists()


def test_stop_service_cannot_clear_latched_fault(
    plant_node: MujocoUR5ePlantNode,
) -> None:
    """Require an explicit reset before FAULT can leave the node or core."""
    plant_node._fault("injected latched fault")

    response = plant_node._on_stop(Trigger.Request(), Trigger.Response())

    assert response.success is False
    assert "reset required" in response.message
    assert plant_node._state == SimulationState.FAULT
    assert plant_node._controller.state.value == "fault"
    assert plant_node._reason == "injected latched fault"


def test_normal_stop_synchronizes_core_and_applies_bounded_safe_hold(
    plant_node: MujocoUR5ePlantNode,
) -> None:
    """Stop both state machines and remove all stale applied forces."""
    plant_node._plant.data.qfrc_applied[:] = 1.0
    plant_node._plant.data.xfrc_applied[:] = 1.0

    response = plant_node._on_stop(Trigger.Request(), Trigger.Response())

    assert response.success is True
    assert plant_node._state == SimulationState.STOPPED
    assert plant_node._controller.state.value == "stopped"
    assert np.all(np.isfinite(plant_node._plant.data.ctrl))
    np.testing.assert_array_less(
        np.abs(plant_node._plant.data.ctrl[:6]),
        plant_node._mapper.config.torque_limits + 1.0e-12,
    )
    np.testing.assert_array_equal(
        plant_node._plant.data.qfrc_applied,
        np.zeros_like(plant_node._plant.data.qfrc_applied),
    )
    np.testing.assert_array_equal(
        plant_node._plant.data.xfrc_applied,
        np.zeros_like(plant_node._plant.data.xfrc_applied),
    )
