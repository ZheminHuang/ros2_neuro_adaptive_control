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

"""Tests for six-dimensional pose impedance integration."""

import numpy as np
import pytest

from neuro_adaptive_control.core.pose_impedance_model import (
    PoseImpedanceModel,
    PoseImpedanceParameters,
)


def _parameters():
    return PoseImpedanceParameters.diagonal(
        mass=(2.0, 4.0, 5.0, 0.4, 0.5, 0.6),
        damping=(1.0, 2.0, 3.0, 0.2, 0.3, 0.4),
        stiffness=(10.0, 20.0, 30.0, 4.0, 5.0, 6.0),
        external_gain=(1.0, 2.0, 3.0, 0.5, 0.6, 0.7),
    )


def test_six_dimensional_auxiliary_input_and_semi_implicit_step():
    parameters = _parameters()
    model = PoseImpedanceModel(
        parameters,
        initial_position=np.linspace(0.1, 0.6, 6),
        initial_velocity=np.linspace(-0.2, 0.3, 6),
    )
    desired = np.linspace(0.3, -0.2, 6)
    desired_velocity = np.linspace(0.4, -0.1, 6)
    desired_acceleration = np.linspace(-0.3, 0.2, 6)
    wrench = np.linspace(1.0, -0.5, 6)
    previous = model.state
    dt = 0.002
    feedforward = (
        parameters.mass @ desired_acceleration
        + parameters.damping @ desired_velocity
        + parameters.stiffness @ desired
    )
    expected_acceleration = np.linalg.solve(
        parameters.mass,
        parameters.external_gain @ wrench
        + feedforward
        - parameters.damping @ previous.velocity
        - parameters.stiffness @ previous.position,
    )
    expected_velocity = previous.velocity + dt * expected_acceleration
    expected_position = previous.position + dt * expected_velocity

    state = model.step(
        desired,
        desired_velocity,
        desired_acceleration,
        wrench,
        dt,
    )

    np.testing.assert_allclose(model.auxiliary_input(
        desired, desired_velocity, desired_acceleration
    ), feedforward)
    np.testing.assert_allclose(state.acceleration, expected_acceleration)
    np.testing.assert_allclose(state.velocity, expected_velocity)
    np.testing.assert_allclose(state.position, expected_position)


def test_reset_is_deterministic_and_returns_defensive_state():
    model = PoseImpedanceModel(_parameters())
    expected_position = np.arange(6, dtype=float)
    expected_velocity = -np.arange(6, dtype=float)

    first = model.reset(expected_position, expected_velocity)
    first.position[:] = 99.0
    second = model.reset(expected_position, expected_velocity)

    np.testing.assert_array_equal(second.position, expected_position)
    np.testing.assert_array_equal(second.velocity, expected_velocity)
    np.testing.assert_array_equal(second.acceleration, np.zeros(6))


@pytest.mark.parametrize("bad_dt", [0.0, -0.1, np.nan, np.inf])
def test_invalid_integration_step_is_rejected(bad_dt):
    model = PoseImpedanceModel(_parameters())
    with pytest.raises(ValueError, match="dt must be finite"):
        model.step(np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), bad_dt)


def test_invalid_parameters_and_nonfinite_inputs_are_rejected():
    with pytest.raises(ValueError, match="positive definite"):
        PoseImpedanceParameters.diagonal(
            mass=(1.0, 1.0, 1.0, 1.0, 0.0, 1.0),
            damping=np.ones(6),
            stiffness=np.ones(6),
        )
    model = PoseImpedanceModel(_parameters())
    bad = np.zeros(6)
    bad[4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.step(np.zeros(6), np.zeros(6), np.zeros(6), bad, 0.01)
