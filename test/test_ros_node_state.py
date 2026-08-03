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

"""Regression tests for ROS wrapper initialization and reset state."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped

from neuro_adaptive_control.core.references import make_reference
from neuro_adaptive_control.core.safety import ControllerState
from neuro_adaptive_control.core.simulation import build_demo_controller
from neuro_adaptive_control.nodes.cartesian_demo_plant import (
    CartesianDemoPlant,
)
from neuro_adaptive_control.nodes.cartesian_demo_plant import (
    _finite_positive as plant_finite_positive,
)
from neuro_adaptive_control.nodes.nac_controller_node import (
    NACControllerNode,
)
from neuro_adaptive_control.nodes.nac_controller_node import (
    _finite_positive as controller_finite_positive,
)


def _bundle(stamp_ns: int = 0):
    pose = PoseStamped()
    twist = TwistStamped()
    wrench = WrenchStamped()
    for message in (pose, twist, wrench):
        message.header.stamp.sec = stamp_ns // 1_000_000_000
        message.header.stamp.nanosec = stamp_ns % 1_000_000_000
        message.header.frame_id = "world"
    pose.pose.orientation.w = 1.0
    key = (
        pose.header.stamp.sec,
        pose.header.stamp.nanosec,
    )
    return key, pose, twist, wrench


def _runtime_stub(controller):
    key, pose, twist, wrench = _bundle()
    published = []
    node = SimpleNamespace(
        pose_cache=OrderedDict([(key, pose)]),
        twist_cache=OrderedDict([(key, twist)]),
        wrench_cache=OrderedDict([(key, wrench)]),
        start_stamp_ns=None,
        last_processed_stamp_ns=None,
        last_bundle_stamp=None,
        last_receive_steady=None,
        last_actual_position=np.ones(3),
        last_actual_velocity=np.ones(3),
        processed_steps=0,
        saturation_count=0,
        stamp_mismatch_count=0,
        last_output=None,
        wall_start=-1.0,
        _duration_stopped=False,
        controller=controller,
        auto_start=False,
        reference=make_reference("fixed_point"),
        duration_sec=1.0,
        dt_ns=2_000_000,
        dt=0.002,
        telemetry_decimation=50,
        _steady_seconds=lambda: 0.01,
        _publish_command=lambda stamp, command: published.append(
            np.asarray(command).copy()
        ),
        _publish_telemetry=lambda *args: None,
    )
    return node, key, published


def test_manual_start_survives_first_bundle_initialization() -> None:
    """A pre-bundle start service request must not be reset back to start."""
    controller = build_demo_controller(adaptation_enabled=True, seed=7)
    controller.start(0.0)
    node, key, published = _runtime_stub(controller)

    NACControllerNode._try_process(node, key)

    assert controller.state == ControllerState.RUNNING
    assert node.start_stamp_ns == 0
    assert node.last_processed_stamp_ns == 0
    assert node.processed_steps == 1
    assert len(published) == 1
    assert np.all(np.isfinite(published[0]))


def test_runtime_history_clear_is_complete_and_repeatable() -> None:
    """Reset history must remove stale stamps, bundles, and counters."""
    controller = build_demo_controller(adaptation_enabled=True, seed=7)
    node, _, _ = _runtime_stub(controller)
    node.start_stamp_ns = 123
    node.last_processed_stamp_ns = 456
    node.last_bundle_stamp = object()
    node.last_receive_steady = 7.0
    node.processed_steps = 9
    node.saturation_count = 8
    node.stamp_mismatch_count = 7
    node.last_output = object()
    node._duration_stopped = True

    NACControllerNode._clear_runtime_history(node)
    first_wall_start = node.wall_start
    NACControllerNode._clear_runtime_history(node)

    assert not node.pose_cache
    assert not node.twist_cache
    assert not node.wrench_cache
    assert node.start_stamp_ns is None
    assert node.last_processed_stamp_ns is None
    assert node.last_bundle_stamp is None
    assert node.last_receive_steady is None
    assert node.processed_steps == 0
    assert node.saturation_count == 0
    assert node.stamp_mismatch_count == 0
    assert node.last_output is None
    assert not node._duration_stopped
    assert node.wall_start >= first_wall_start


def test_reset_service_clears_weights_and_all_wrapper_history() -> None:
    """A stopped controller reset must return a fresh start state."""
    controller = build_demo_controller(adaptation_enabled=True, seed=7)
    controller.network.weights[:] = 2.0
    controller.start(0.0)
    controller.stop("test")
    controller.safety.filter_command(np.zeros(3), 0.01)
    assert controller.state == ControllerState.STOPPED

    node, _, published = _runtime_stub(controller)
    node.start_stamp_ns = 123
    node.last_processed_stamp_ns = 456
    node.processed_steps = 9
    node.stamp_mismatch_count = 3
    node._duration_stopped = True
    node._publish_zero = lambda: published.append(np.zeros(3))
    node._clear_runtime_history = lambda: (
        NACControllerNode._clear_runtime_history(node)
    )
    response = SimpleNamespace(success=False, message="")

    result = NACControllerNode._reset(node, object(), response)

    assert result is response
    assert response.success
    assert response.message == "start"
    assert controller.state == ControllerState.START
    np.testing.assert_array_equal(
        controller.network.weights,
        np.zeros_like(controller.network.weights),
    )
    assert node.start_stamp_ns is None
    assert node.last_processed_stamp_ns is None
    assert node.processed_steps == 0
    assert node.stamp_mismatch_count == 0
    assert not node._duration_stopped
    assert len(published) == 1


@pytest.mark.parametrize(
    "validator",
    (controller_finite_positive, plant_finite_positive),
)
@pytest.mark.parametrize(
    "invalid",
    (0.0, -1.0, np.nan, np.inf, "not-a-number"),
)
def test_ros_rate_and_watchdog_parameter_validation(
    validator, invalid
) -> None:
    """ROS timing parameters must not disable safety through NaN or zero."""
    with pytest.raises(ValueError):
        validator(invalid, "test_parameter")
    assert validator("2.5", "test_parameter") == 2.5


def test_late_zero_replaces_pending_nonzero_plant_command() -> None:
    """A same-stamp stop/fault zero must override a queued plant command."""
    message = WrenchStamped()
    message.header.frame_id = "world"
    node = SimpleNamespace(
        awaiting_command=True,
        current_key=(0, 0),
        frame_id="world",
        stamp_mismatches=0,
        pending_command=np.array([3.0, -2.0, 1.0]),
        _fault=lambda reason: None,
    )

    CartesianDemoPlant._on_command(node, message)

    np.testing.assert_array_equal(node.pending_command, np.zeros(3))
    assert node.stamp_mismatches == 0
