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

"""Tests for lifecycle, watchdog, validation, and command limiting."""

import numpy as np
import pytest

from neuro_adaptive_control.core.safety import (
    ControllerState,
    SafetyConfig,
    SafetySupervisor,
)


def _supervisor(
    *,
    limits=(3.0, 2.0, 5.0),
    norm_limit=10.0,
    watchdog=0.2,
    maximum_dt=0.01,
):
    return SafetySupervisor(
        SafetyConfig(
            command_limits=np.asarray(limits, dtype=float),
            command_norm_limit=norm_limit,
            watchdog_timeout=watchdog,
            maximum_dt=maximum_dt,
        )
    )


def test_required_five_states_and_stop_cycle_emits_zero():
    safety = _supervisor()
    visited = [safety.state]

    safety.start(0.0)
    visited.append(safety.state)
    safety.request_stop("operator stop")
    visited.append(safety.state)

    command = safety.filter_command([1.0, -1.0, 0.5], now=0.01)
    visited.append(safety.state)

    np.testing.assert_array_equal(command, np.zeros(3))
    np.testing.assert_array_equal(safety.last_command, np.zeros(3))
    assert safety.reason == "operator stop"
    assert not safety.last_saturated

    safety.trigger_fault("post-stop diagnostic fault")
    visited.append(safety.state)
    assert set(visited) == set(ControllerState)
    assert [state.value for state in ControllerState] == [
        "start",
        "running",
        "stopping",
        "stopped",
        "fault",
    ]


def test_start_and_stopped_states_are_non_commanding():
    safety = _supervisor()
    np.testing.assert_array_equal(
        safety.filter_command([1.0, 1.0, 1.0], now=0.0),
        np.zeros(3),
    )

    safety.request_stop("never started")
    assert safety.state == ControllerState.STOPPED
    np.testing.assert_array_equal(
        safety.filter_command([1.0, 1.0, 1.0], now=0.01),
        np.zeros(3),
    )


def test_stopped_controller_can_restart_but_fault_requires_reset():
    safety = _supervisor()
    safety.start(0.0)
    safety.request_stop()
    safety.filter_command(np.ones(3), now=0.01)
    safety.start(0.02)
    assert safety.state == ControllerState.RUNNING

    safety.trigger_fault("latched")
    with pytest.raises(RuntimeError, match="reset is required"):
        safety.start(0.03)


def test_watchdog_uses_latest_coherent_measurement():
    safety = _supervisor(watchdog=0.2)
    safety.start(0.0)
    assert safety.note_measurement(0.15, np.zeros(3), np.ones(3))
    assert safety.tick(0.35) == ControllerState.RUNNING
    assert safety.tick(0.350001) == ControllerState.FAULT
    assert safety.reason == "state watchdog expired"
    np.testing.assert_array_equal(safety.last_command, np.zeros(3))


def test_watchdog_falls_back_to_start_time_before_first_measurement():
    safety = _supervisor(watchdog=0.05)
    safety.start(1.0)
    assert safety.tick(1.049) == ControllerState.RUNNING
    assert safety.tick(1.051) == ControllerState.FAULT
    assert safety.reason == "state watchdog expired"


def test_clock_rollback_faults_and_zeroes_command():
    safety = _supervisor()
    safety.start(1.0)
    assert safety.note_measurement(1.1, np.zeros(3))
    state = safety.tick(1.09)

    assert state == ControllerState.FAULT
    assert safety.reason == "controller time moved backwards"
    np.testing.assert_array_equal(safety.last_command, np.zeros(3))


def test_axis_saturation_is_applied_independently():
    safety = _supervisor(norm_limit=20.0)
    safety.start(0.0)

    command = safety.filter_command([-4.0, 1.0, 6.0], now=0.01)

    np.testing.assert_allclose(command, [-3.0, 1.0, 5.0])
    assert safety.last_saturated


def test_norm_saturation_follows_axis_saturation():
    safety = _supervisor(norm_limit=2.0)
    safety.start(0.0)

    command = safety.filter_command([4.0, 4.0, 0.0], now=0.01)
    clipped = np.array([3.0, 2.0, 0.0])
    expected = clipped * (2.0 / np.linalg.norm(clipped))

    np.testing.assert_allclose(command, expected)
    assert np.linalg.norm(command) == pytest.approx(2.0)
    assert safety.last_saturated


@pytest.mark.parametrize(
    "bad_command",
    [
        np.ones(2),
        np.ones((3, 1)),
        np.array([np.nan, 0.0, 0.0]),
        np.array([0.0, np.inf, 0.0]),
        ["not-a-number", 0.0, 0.0],
    ],
)
def test_invalid_command_faults_and_returns_zero(bad_command):
    safety = _supervisor()
    safety.start(0.0)

    command = safety.filter_command(bad_command, now=0.01)

    assert safety.state == ControllerState.FAULT
    np.testing.assert_array_equal(command, np.zeros(3))
    np.testing.assert_array_equal(safety.last_command, np.zeros(3))


@pytest.mark.parametrize(
    "bad_dt",
    [0.0, -0.001, 0.010001, np.nan, np.inf, "not-a-number"],
)
def test_invalid_dt_faults(bad_dt):
    safety = _supervisor(maximum_dt=0.01)
    safety.start(0.0)

    assert not safety.validate_dt(bad_dt)
    assert safety.state == ControllerState.FAULT
    np.testing.assert_array_equal(safety.last_command, np.zeros(3))


@pytest.mark.parametrize(
    "bad_sample",
    [
        np.array([np.nan, 0.0, 0.0]),
        np.array([0.0, np.inf, 0.0]),
        ["not-a-number"],
    ],
)
def test_nan_inf_or_nonnumeric_measurement_faults(bad_sample):
    safety = _supervisor()
    safety.start(0.0)

    assert not safety.note_measurement(0.01, bad_sample)
    assert safety.state == ControllerState.FAULT
    assert safety.last_measurement_time is None


def test_first_fault_reason_is_latched():
    safety = _supervisor()
    safety.start(0.0)
    safety.filter_command([1.0, 2.0], now=0.01)
    first_reason = safety.reason

    safety.trigger_fault("later fault")
    safety.note_measurement(0.02, [np.nan])

    assert safety.state == ControllerState.FAULT
    assert first_reason == "command must have shape (3,), got (2,)"
    assert safety.reason == first_reason


def test_reset_is_deterministic_and_clears_all_latches():
    safety = _supervisor()
    safety.start(0.0)
    safety.note_measurement(0.01, np.ones(3))
    safety.filter_command([20.0, 20.0, 20.0], now=0.02)
    safety.trigger_fault("test fault")

    safety.reset()
    first_snapshot = (
        safety.state,
        safety.reason,
        safety.last_measurement_time,
        safety.start_time,
        safety.last_observed_time,
        safety.last_command.copy(),
        safety.last_saturated,
    )
    safety.reset()
    second_snapshot = (
        safety.state,
        safety.reason,
        safety.last_measurement_time,
        safety.start_time,
        safety.last_observed_time,
        safety.last_command.copy(),
        safety.last_saturated,
    )

    assert first_snapshot[:5] == (
        ControllerState.START,
        "reset",
        None,
        None,
        None,
    )
    np.testing.assert_array_equal(first_snapshot[5], np.zeros(3))
    assert first_snapshot[6] is False
    assert first_snapshot[:5] == second_snapshot[:5]
    np.testing.assert_array_equal(first_snapshot[5], second_snapshot[5])
    assert first_snapshot[6] == second_snapshot[6]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"command_limits": [1.0, 2.0]}, "shape"),
        ({"command_limits": [1.0, 0.0, 1.0]}, "finite and positive"),
        ({"command_limits": [1.0, np.inf, 1.0]}, "finite and positive"),
        ({"command_norm_limit": 0.0}, "finite and positive"),
        ({"watchdog_timeout": np.nan}, "finite and positive"),
        ({"maximum_dt": "bad"}, "must be numeric"),
    ],
)
def test_invalid_safety_configuration_is_rejected(kwargs, match):
    defaults = {
        "command_limits": [1.0, 1.0, 1.0],
        "command_norm_limit": 2.0,
        "watchdog_timeout": 0.1,
        "maximum_dt": 0.01,
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=match):
        SafetyConfig(**defaults)
