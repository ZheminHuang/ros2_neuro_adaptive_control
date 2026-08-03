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

"""Unit tests for the 3D Cartesian impedance reference model."""

import numpy as np
import pytest

from neuro_adaptive_control.core.impedance_model import (
    CartesianImpedanceModel,
    ImpedanceParameters,
)


def _parameters(
    mass=(2.0, 3.0, 4.0),
    damping=(5.0, 6.0, 7.0),
    stiffness=(8.0, 9.0, 10.0),
    external_gain=(1.0, 1.0, 1.0),
):
    return ImpedanceParameters.diagonal(
        mass=mass,
        damping=damping,
        stiffness=stiffness,
        external_gain=external_gain,
    )


def _assert_same_state(left, right):
    np.testing.assert_array_equal(left.position, right.position)
    np.testing.assert_array_equal(left.velocity, right.velocity)
    np.testing.assert_array_equal(left.acceleration, right.acceleration)


def test_auxiliary_input_contains_full_reference_feedforward():
    parameters = _parameters()
    model = CartesianImpedanceModel(parameters)
    position = np.array([0.2, -0.3, 0.4])
    velocity = np.array([-0.5, 0.6, -0.7])
    acceleration = np.array([0.8, -0.9, 1.0])

    expected = (
        parameters.mass @ acceleration
        + parameters.damping @ velocity
        + parameters.stiffness @ position
    )

    np.testing.assert_allclose(
        model.auxiliary_input(position, velocity, acceleration),
        expected,
        rtol=0.0,
        atol=1e-15,
    )


def test_reference_state_produces_reference_acceleration():
    parameters = _parameters()
    desired_position = np.array([0.3, -0.2, 0.1])
    desired_velocity = np.array([-0.4, 0.5, -0.6])
    desired_acceleration = np.array([0.7, -0.8, 0.9])
    model = CartesianImpedanceModel(
        parameters,
        initial_position=desired_position,
        initial_velocity=desired_velocity,
    )

    state = model.step(
        desired_position,
        desired_velocity,
        desired_acceleration,
        external_wrench=np.zeros(3),
        dt=0.002,
    )

    np.testing.assert_allclose(
        state.acceleration, desired_acceleration, rtol=1e-14, atol=1e-14
    )


def test_step_uses_semi_implicit_euler_integration():
    parameters = _parameters(
        mass=(2.0, 4.0, 5.0),
        damping=(1.0, 2.0, 3.0),
        stiffness=(10.0, 20.0, 30.0),
        external_gain=(0.5, 1.0, 2.0),
    )
    initial_position = np.array([0.1, -0.2, 0.3])
    initial_velocity = np.array([-0.4, 0.5, -0.6])
    reference_position = np.array([0.7, -0.8, 0.9])
    reference_velocity = np.array([0.2, -0.1, 0.4])
    reference_acceleration = np.array([-0.3, 0.6, -0.5])
    wrench = np.array([1.2, -0.7, 0.25])
    dt = 0.01
    model = CartesianImpedanceModel(
        parameters, initial_position, initial_velocity
    )

    feedforward = (
        parameters.mass @ reference_acceleration
        + parameters.damping @ reference_velocity
        + parameters.stiffness @ reference_position
    )
    rhs = (
        parameters.external_gain @ wrench
        + feedforward
        - parameters.damping @ initial_velocity
        - parameters.stiffness @ initial_position
    )
    expected_acceleration = np.linalg.solve(parameters.mass, rhs)
    expected_velocity = initial_velocity + dt * expected_acceleration
    expected_position = initial_position + dt * expected_velocity

    state = model.step(
        reference_position,
        reference_velocity,
        reference_acceleration,
        wrench,
        dt,
    )

    np.testing.assert_allclose(
        state.acceleration, expected_acceleration, rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        state.velocity, expected_velocity, rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        state.position, expected_position, rtol=1e-14, atol=1e-14
    )
    explicit_position = initial_position + dt * initial_velocity
    assert not np.allclose(state.position, explicit_position)


def test_external_wrench_enters_with_positive_contract_sign():
    parameters = _parameters(
        mass=(2.0, 4.0, 8.0),
        damping=(0.0, 0.0, 0.0),
        stiffness=(0.0, 0.0, 0.0),
        external_gain=(1.0, 2.0, 4.0),
    )
    model = CartesianImpedanceModel(parameters)
    wrench = np.array([2.0, -4.0, 8.0])
    zeros = np.zeros(3)

    positive = model.step(zeros, zeros, zeros, wrench, dt=0.01)
    model.reset()
    negative = model.step(zeros, zeros, zeros, -wrench, dt=0.01)

    expected = np.linalg.solve(
        parameters.mass, parameters.external_gain @ wrench
    )
    np.testing.assert_allclose(positive.acceleration, expected)
    np.testing.assert_allclose(negative.acceleration, -expected)


def test_reset_is_deterministic_and_failed_reset_is_atomic():
    model = CartesianImpedanceModel(
        _parameters(), initial_position=[0.4, -0.5, 0.6]
    )
    inputs = (
        [0.2, 0.1, -0.2],
        [0.3, -0.4, 0.5],
        [-0.6, 0.7, -0.8],
        [1.0, -2.0, 3.0],
        0.01,
    )

    initial = model.reset([0.4, -0.5, 0.6], [0.1, -0.2, 0.3])
    first = model.step(*inputs)
    reset_state = model.reset(initial.position, initial.velocity)
    _assert_same_state(reset_state, initial)
    second = model.step(*inputs)
    _assert_same_state(first, second)

    before_failure = model.state
    with pytest.raises(ValueError):
        model.reset([9.0, 8.0, 7.0], [0.0, np.nan, 0.0])
    _assert_same_state(model.state, before_failure)


def test_state_property_returns_defensive_copies():
    model = CartesianImpedanceModel(_parameters(), [1.0, 2.0, 3.0])
    snapshot = model.state
    snapshot.position[:] = 99.0
    snapshot.velocity[:] = 99.0
    snapshot.acceleration[:] = 99.0

    np.testing.assert_array_equal(model.state.position, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(model.state.velocity, np.zeros(3))
    np.testing.assert_array_equal(model.state.acceleration, np.zeros(3))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ImpedanceParameters(
            mass=np.eye(2),
            damping=np.eye(3),
            stiffness=np.eye(3),
            external_gain=np.eye(3),
        ),
        lambda: ImpedanceParameters(
            mass=np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            damping=np.eye(3),
            stiffness=np.eye(3),
            external_gain=np.eye(3),
        ),
        lambda: _parameters(mass=(1.0, 0.0, 1.0)),
        lambda: _parameters(mass=(1.0, -1.0, 1.0)),
        lambda: _parameters(damping=(1.0, -1.0, 1.0)),
        lambda: _parameters(stiffness=(1.0, 1.0, -1.0)),
        lambda: _parameters(external_gain=(1.0, np.nan, 1.0)),
    ],
)
def test_invalid_impedance_parameters_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    "operation",
    [
        lambda model: model.reset([0.0, 0.0], [0.0, 0.0, 0.0]),
        lambda model: model.reset([0.0, np.inf, 0.0], [0.0, 0.0, 0.0]),
        lambda model: model.auxiliary_input(
            [0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0, 0.0]
        ),
        lambda model: model.step(
            [np.nan, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            0.01,
        ),
        lambda model: model.step(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, np.inf, 0.0],
            0.01,
        ),
        lambda model: model.step(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            0.0,
        ),
        lambda model: model.step(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            np.nan,
        ),
    ],
)
def test_invalid_runtime_inputs_are_atomic(operation):
    model = CartesianImpedanceModel(
        _parameters(), initial_position=[0.1, -0.2, 0.3]
    )
    before = model.state

    with pytest.raises(ValueError):
        operation(model)

    _assert_same_state(model.state, before)


def test_nonfinite_integration_result_is_trapped_atomically():
    parameters = _parameters(
        mass=(1.0, 1.0, 1.0),
        damping=(0.0, 0.0, 0.0),
        stiffness=(1e308, 1e308, 1e308),
    )
    model = CartesianImpedanceModel(
        parameters, initial_position=[1e308, 1e308, 1e308]
    )
    before = model.state
    zeros = np.zeros(3)

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError):
            model.step(zeros, zeros, zeros, zeros, dt=1.0)

    _assert_same_state(model.state, before)
