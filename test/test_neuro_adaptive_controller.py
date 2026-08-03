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

"""Contract tests for the pure NumPy neuro-adaptive controller."""

import numpy as np
import pytest

from neuro_adaptive_control.core.impedance_model import (
    CartesianImpedanceModel,
    ImpedanceParameters,
)
from neuro_adaptive_control.core.neuro_adaptive_controller import (
    NACParameters,
    NeuroAdaptiveController,
)
from neuro_adaptive_control.core.rbf_network import RBFNetwork
from neuro_adaptive_control.core.references import ReferenceSample
from neuro_adaptive_control.core.safety import (
    ControllerState,
    SafetyConfig,
    SafetySupervisor,
)


def _reference(position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
               acceleration=(0.0, 0.0, 0.0)):
    return ReferenceSample(
        position=np.asarray(position, dtype=float),
        velocity=np.asarray(velocity, dtype=float),
        acceleration=np.asarray(acceleration, dtype=float),
    )


def _controller(
    *,
    initial_position=(0.0, 0.0, 0.0),
    initial_velocity=(0.0, 0.0, 0.0),
    external_gain=(1.0, 1.0, 1.0),
    lambda_gain=(2.0, 3.0, 4.0),
    feedback_gain=(5.0, 6.0, 7.0),
    robust_gain=(1.0, 2.0, 3.0),
    robust_bias=0.5,
    adaptation_enabled=True,
    dynamics_feature_dim=6,
    learning_rate=2.5,
    leakage=0.0,
):
    impedance = CartesianImpedanceModel(
        ImpedanceParameters.diagonal(
            mass=(1.0, 1.0, 1.0),
            damping=(0.0, 0.0, 0.0),
            stiffness=(0.0, 0.0, 0.0),
            external_gain=external_gain,
        ),
        initial_position=initial_position,
        initial_velocity=initial_velocity,
    )
    input_dim = dynamics_feature_dim + 15
    network = RBFNetwork(
        input_dim=input_dim,
        output_dim=3,
        num_basis=2,
        centers=np.zeros((2, input_dim)),
        widths=(1.0, 2.0),
        input_scale=np.ones(input_dim),
        feature_clip=100.0,
        learning_rate=learning_rate,
        leakage=leakage,
        weight_limit=1.0e6,
        adaptation_enabled=adaptation_enabled,
    )
    parameters = NACParameters.diagonal(
        lambda_gain=lambda_gain,
        feedback_gain=feedback_gain,
        robust_gain=robust_gain,
        robust_bias=robust_bias,
    )
    safety = SafetySupervisor(
        SafetyConfig(
            command_limits=np.full(3, 1.0e6),
            command_norm_limit=1.0e6,
            watchdog_timeout=1.0,
            maximum_dt=0.2,
        )
    )
    return NeuroAdaptiveController(
        impedance,
        network,
        parameters,
        safety,
        dynamics_feature_dim=dynamics_feature_dim,
    )


def test_control_law_uses_em_xm_minus_x_and_positive_feedback_terms():
    controller = _controller(
        initial_position=(0.4, -0.3, 0.2),
        initial_velocity=(0.1, -0.2, 0.3),
        adaptation_enabled=False,
    )
    controller.network.weights[:] = np.array(
        [[1.0, -2.0, 0.5], [-0.25, 0.75, 1.5]]
    )
    position = np.array([0.1, -0.1, 0.3])
    velocity = np.array([-0.1, 0.1, 0.2])
    controller.start(0.0)

    output = controller.step(
        position,
        velocity,
        _reference(),
        np.zeros(3),
        dt=0.01,
        now=0.01,
    )

    expected_model_position = np.array([0.401, -0.302, 0.203])
    expected_error = expected_model_position - position
    expected_error_velocity = np.array([0.2, -0.3, 0.1])
    expected_r = expected_error_velocity + np.diag([2.0, 3.0, 4.0]) @ (
        expected_error
    )
    expected_features = np.concatenate(
        (
            position,
            velocity,
            expected_model_position,
            [0.1, -0.2, 0.3],
            np.zeros(3),
            expected_error,
            expected_error_velocity,
        )
    )
    phi = controller.network.activations(expected_features)
    expected_neural = controller.network.weights.T @ phi
    expected_feedback = np.diag([5.0, 6.0, 7.0]) @ expected_r
    expected_robust = (
        controller.network.weight_norm + 0.5
    ) * (np.diag([1.0, 2.0, 3.0]) @ expected_r)
    expected_raw = expected_neural + expected_feedback + expected_robust

    assert output.state == ControllerState.RUNNING
    np.testing.assert_allclose(output.model_error, expected_error)
    np.testing.assert_allclose(
        output.model_error_velocity, expected_error_velocity
    )
    np.testing.assert_allclose(output.sliding_error, expected_r)
    np.testing.assert_allclose(output.rbf_features, expected_features)
    np.testing.assert_allclose(output.neural_estimate, expected_neural)
    np.testing.assert_allclose(output.feedback_term, expected_feedback)
    np.testing.assert_allclose(output.robust_term, expected_robust)
    np.testing.assert_array_equal(output.external_term, np.zeros(3))
    np.testing.assert_allclose(output.raw_command, expected_raw)
    np.testing.assert_allclose(output.command, expected_raw)
    assert np.dot(output.feedback_term, expected_r) > 0.0
    assert np.dot(output.robust_term, expected_r) > 0.0
    assert not output.saturated


def test_external_wrench_enters_model_positive_and_command_as_minus_kh_wrench():
    controller = _controller(
        external_gain=(2.0, 3.0, 4.0),
        adaptation_enabled=False,
    )
    wrench = np.array([1.0, -2.0, 0.5])
    expected_acceleration = np.array([2.0, -6.0, 2.0])
    dt = 0.1
    expected_velocity = dt * expected_acceleration
    expected_position = dt * expected_velocity
    controller.start(0.0)

    output = controller.step(
        expected_position,
        expected_velocity,
        _reference(),
        wrench,
        dt=dt,
        now=dt,
    )

    np.testing.assert_allclose(
        output.model_state.acceleration, expected_acceleration
    )
    np.testing.assert_allclose(output.model_state.velocity, expected_velocity)
    np.testing.assert_allclose(output.model_state.position, expected_position)
    np.testing.assert_allclose(output.model_error, np.zeros(3), atol=1e-15)
    np.testing.assert_allclose(output.sliding_error, np.zeros(3), atol=1e-15)
    np.testing.assert_allclose(output.external_term, -expected_acceleration)
    np.testing.assert_allclose(output.raw_command, -expected_acceleration)
    np.testing.assert_allclose(output.command, -expected_acceleration)


def test_all_controller_output_terms_have_v01_translation_dimensions():
    controller = _controller(adaptation_enabled=False)
    controller.start(0.0)

    output = controller.step(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        _reference(),
        [0.0, 0.0, 0.0],
        dt=0.01,
        now=0.01,
    )

    for value in (
        output.command,
        output.raw_command,
        output.neural_estimate,
        output.feedback_term,
        output.robust_term,
        output.external_term,
        output.model_error,
        output.model_error_velocity,
        output.sliding_error,
        output.model_state.position,
        output.model_state.velocity,
        output.model_state.acceleration,
    ):
        assert value.shape == (3,)
    assert output.rbf_features.shape == (21,)


def test_zero_error_produces_zero_command_and_no_weight_update():
    controller = _controller(
        initial_position=(0.2, -0.1, 0.4),
        initial_velocity=(0.0, 0.0, 0.0),
    )
    controller.start(0.0)

    output = controller.step(
        [0.2, -0.1, 0.4],
        [0.0, 0.0, 0.0],
        _reference(position=(0.2, -0.1, 0.4)),
        [0.0, 0.0, 0.0],
        dt=0.01,
        now=0.01,
    )

    np.testing.assert_allclose(output.model_error, np.zeros(3), atol=1e-15)
    np.testing.assert_allclose(output.sliding_error, np.zeros(3), atol=1e-15)
    np.testing.assert_allclose(output.raw_command, np.zeros(3), atol=1e-15)
    np.testing.assert_allclose(output.command, np.zeros(3), atol=1e-15)
    np.testing.assert_array_equal(controller.network.weights, np.zeros((2, 3)))
    assert output.state == ControllerState.RUNNING


def test_step_applies_expected_rbf_weight_adaptation_after_command():
    learning_rate = 2.5
    dt = 0.01
    controller = _controller(
        initial_position=(0.4, -0.3, 0.2),
        initial_velocity=(0.1, -0.2, 0.3),
        learning_rate=learning_rate,
        leakage=0.0,
    )
    controller.start(0.0)

    output = controller.step(
        [0.1, -0.1, 0.3],
        [-0.1, 0.1, 0.2],
        _reference(),
        np.zeros(3),
        dt=dt,
        now=dt,
    )

    phi = controller.network.activations(output.rbf_features)
    expected_weights = dt * learning_rate * np.outer(
        phi, output.sliding_error
    )
    np.testing.assert_array_equal(output.neural_estimate, np.zeros(3))
    np.testing.assert_allclose(controller.network.weights, expected_weights)
    assert controller.network.weight_norm > 0.0


def test_disabled_adaptation_keeps_weights_fixed_for_baseline():
    controller = _controller(
        initial_position=(0.4, -0.3, 0.2),
        adaptation_enabled=False,
    )
    initial_weights = np.array(
        [[1.0, -2.0, 0.5], [-0.25, 0.75, 1.5]]
    )
    controller.network.weights[:] = initial_weights
    controller.start(0.0)

    controller.step(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        _reference(),
        np.zeros(3),
        dt=0.01,
        now=0.01,
    )

    np.testing.assert_array_equal(controller.network.weights, initial_weights)


def test_stop_completes_with_zero_command_before_any_new_control_update():
    controller = _controller(initial_position=(1.0, 1.0, 1.0))
    controller.start(0.0)
    controller.stop("test stop")

    output = controller.step(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        _reference(),
        np.zeros(3),
        dt=0.01,
        now=0.01,
    )

    assert output.state == ControllerState.STOPPED
    assert output.fault_reason == "test stop"
    np.testing.assert_array_equal(output.command, np.zeros(3))
    np.testing.assert_array_equal(output.raw_command, np.zeros(3))
    np.testing.assert_array_equal(controller.network.weights, np.zeros((2, 3)))
    np.testing.assert_array_equal(
        controller.impedance_model.state.position, np.ones(3)
    )


@pytest.mark.parametrize(
    "bad_dt",
    [0.0, -0.01, 0.200001, np.nan, np.inf, "not-a-number"],
)
def test_invalid_dt_faults_without_mutating_model_or_network(bad_dt):
    controller = _controller(initial_position=(0.1, 0.2, 0.3))
    initial_state = controller.impedance_model.state
    controller.start(0.0)

    output = controller.step(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        _reference(),
        np.zeros(3),
        dt=bad_dt,
        now=0.01,
    )

    assert output.state == ControllerState.FAULT
    np.testing.assert_array_equal(output.command, np.zeros(3))
    np.testing.assert_array_equal(
        controller.impedance_model.state.position, initial_state.position
    )
    np.testing.assert_array_equal(
        controller.impedance_model.state.velocity, initial_state.velocity
    )
    np.testing.assert_array_equal(controller.network.weights, np.zeros((2, 3)))


@pytest.mark.parametrize(
    "overrides",
    [
        {"actual_position": [np.nan, 0.0, 0.0]},
        {"actual_velocity": [0.0, np.inf, 0.0]},
        {"external_wrench": [0.0, 0.0, -np.inf]},
        {"reference": _reference(position=(0.0, np.nan, 0.0))},
        {"reference": _reference(velocity=(0.0, 0.0, np.inf))},
        {"reference": _reference(acceleration=(np.nan, 0.0, 0.0))},
    ],
)
def test_nonfinite_inputs_latch_fault_and_return_zero(overrides):
    controller = _controller()
    arguments = {
        "actual_position": [0.0, 0.0, 0.0],
        "actual_velocity": [0.0, 0.0, 0.0],
        "reference": _reference(),
        "external_wrench": [0.0, 0.0, 0.0],
    }
    arguments.update(overrides)
    controller.start(0.0)

    output = controller.step(**arguments, dt=0.01, now=0.01)

    assert output.state == ControllerState.FAULT
    assert "finite" in output.fault_reason or "NaN or Inf" in output.fault_reason
    np.testing.assert_array_equal(output.command, np.zeros(3))
    np.testing.assert_array_equal(controller.network.weights, np.zeros((2, 3)))


def test_nonfinite_internal_neural_output_faults_and_returns_zero():
    controller = _controller(adaptation_enabled=False)
    controller.network.weights[0, 0] = np.nan
    controller.start(0.0)

    output = controller.step(
        np.zeros(3),
        np.zeros(3),
        _reference(),
        np.zeros(3),
        dt=0.01,
        now=0.01,
    )

    assert output.state == ControllerState.FAULT
    assert output.fault_reason == "RBF output produced NaN or Inf."
    np.testing.assert_array_equal(output.command, np.zeros(3))


def test_nonfinite_dynamics_features_fault_during_rbf_validation():
    controller = _controller()
    controller.start(0.0)

    output = controller.step(
        np.zeros(3),
        np.zeros(3),
        _reference(),
        np.zeros(3),
        dt=0.01,
        now=0.01,
        dynamics_features=[0.0, 0.0, np.inf, 0.0, 0.0, 0.0],
    )

    assert output.state == ControllerState.FAULT
    assert output.fault_reason == "features must contain only finite values."
    np.testing.assert_array_equal(output.command, np.zeros(3))


def test_wrong_runtime_dynamics_feature_dimension_faults():
    controller = _controller(dynamics_feature_dim=4)
    controller.start(0.0)

    output = controller.step(
        np.zeros(3),
        np.zeros(3),
        _reference(),
        np.zeros(3),
        dt=0.01,
        now=0.01,
        dynamics_features=np.zeros(6),
    )

    assert output.state == ControllerState.FAULT
    assert output.fault_reason == (
        "dynamics_features must have shape (4,), got (6,)."
    )
    np.testing.assert_array_equal(output.command, np.zeros(3))


def test_controller_reset_replays_identically_and_clears_fault_and_weights():
    controller = _controller()
    reset_position = np.array([0.2, -0.3, 0.4])
    reset_velocity = np.array([0.1, 0.0, -0.1])

    def run_once():
        controller.reset(reset_position, reset_velocity)
        controller.start(0.0)
        output = controller.step(
            np.zeros(3),
            np.zeros(3),
            _reference(),
            np.zeros(3),
            dt=0.01,
            now=0.01,
        )
        return output, controller.network.weights.copy()

    first_output, first_weights = run_once()
    controller.safety.trigger_fault("injected fault")
    second_output, second_weights = run_once()

    assert second_output.state == ControllerState.RUNNING
    assert second_output.fault_reason == "started"
    for name in (
        "command",
        "raw_command",
        "neural_estimate",
        "feedback_term",
        "robust_term",
        "external_term",
        "model_error",
        "model_error_velocity",
        "sliding_error",
        "rbf_features",
    ):
        np.testing.assert_array_equal(
            getattr(first_output, name), getattr(second_output, name)
        )
    np.testing.assert_array_equal(
        first_output.model_state.position, second_output.model_state.position
    )
    np.testing.assert_array_equal(
        first_output.model_state.velocity, second_output.model_state.velocity
    )
    np.testing.assert_array_equal(
        first_output.model_state.acceleration,
        second_output.model_state.acceleration,
    )
    np.testing.assert_array_equal(first_weights, second_weights)


def test_constructor_validates_rbf_input_and_output_dimensions():
    impedance = CartesianImpedanceModel(
        ImpedanceParameters.diagonal(
            mass=(1.0, 1.0, 1.0),
            damping=(1.0, 1.0, 1.0),
            stiffness=(1.0, 1.0, 1.0),
        )
    )
    parameters = NACParameters.diagonal(
        lambda_gain=(1.0, 1.0, 1.0),
        feedback_gain=(1.0, 1.0, 1.0),
        robust_gain=(1.0, 1.0, 1.0),
        robust_bias=0.0,
    )
    safety = SafetySupervisor(
        SafetyConfig(np.ones(3), 1.0, 0.1, 0.01)
    )

    with pytest.raises(ValueError, match="RBF input_dim must be 21"):
        NeuroAdaptiveController(
            impedance,
            RBFNetwork(input_dim=20),
            parameters,
            safety,
        )
    with pytest.raises(ValueError, match="RBF output_dim must be 3"):
        NeuroAdaptiveController(
            impedance,
            RBFNetwork(input_dim=21, output_dim=2),
            parameters,
            safety,
        )
    with pytest.raises(ValueError, match="dynamics_feature_dim must be positive"):
        NeuroAdaptiveController(
            impedance,
            RBFNetwork(input_dim=15),
            parameters,
            safety,
            dynamics_feature_dim=0,
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"lambda_gain": np.ones((2, 2))}, "shape"),
        (
            {"feedback_gain": [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]]},
            "symmetric",
        ),
        ({"robust_gain": [-1.0, 1.0, 1.0]}, "positive semidefinite"),
        ({"robust_bias": -0.1}, "non-negative"),
        ({"robust_bias": np.nan}, "non-negative"),
    ],
)
def test_invalid_nac_parameters_are_rejected(kwargs, match):
    values = {
        "lambda_gain": np.eye(3),
        "feedback_gain": np.eye(3),
        "robust_gain": np.eye(3),
        "robust_bias": 0.1,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        NACParameters(**values)
