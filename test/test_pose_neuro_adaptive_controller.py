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

"""Contract tests for the six-DoF two-layer neuro-adaptive controller."""

import numpy as np

from neuro_adaptive_control.core.pose_impedance_model import (
    PoseImpedanceModel,
    PoseImpedanceParameters,
)
from neuro_adaptive_control.core.pose_neuro_adaptive_controller import (
    PoseNACParameters,
    PoseNeuroAdaptiveController,
    build_pose_nn_features,
)
from neuro_adaptive_control.core.pose_references import PoseReferenceSample
from neuro_adaptive_control.core.safety import (
    ControllerState,
    SafetyConfig,
    SafetySupervisor,
)
from neuro_adaptive_control.core.two_layer_network import TwoLayerAdaptiveNetwork


def _reference(position=None, velocity=None, acceleration=None):
    return PoseReferenceSample(
        position=np.zeros(6) if position is None else np.asarray(position),
        velocity=np.zeros(6) if velocity is None else np.asarray(velocity),
        acceleration=(
            np.zeros(6) if acceleration is None else np.asarray(acceleration)
        ),
    )


def _controller(*, adaptation_enabled=True, command_limits=None):
    impedance = PoseImpedanceModel(
        PoseImpedanceParameters.diagonal(
            mass=np.ones(6),
            damping=np.zeros(6),
            stiffness=np.zeros(6),
            external_gain=np.array([1.0, 2.0, 3.0, 0.5, 0.6, 0.7]),
        )
    )
    network = TwoLayerAdaptiveNetwork(
        input_dim=42,
        hidden_dim=3,
        output_dim=6,
        hidden_learning_rate=0.2,
        output_learning_rate=1.5,
        leakage=0.0,
        hidden_weight_limit=1.0e6,
        output_weight_limit=1.0e6,
        input_scale=np.ones(42),
        input_clip=100.0,
        initial_hidden_scale=0.02,
        seed=8,
        adaptation_enabled=adaptation_enabled,
    )
    parameters = PoseNACParameters.diagonal(
        lambda_gain=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        feedback_gain=(2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
        robust_gain=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        ideal_weight_bound=0.5,
    )
    limits = np.full(6, 1.0e6) if command_limits is None else command_limits
    safety = SafetySupervisor(
        SafetyConfig(
            command_limits=np.asarray(limits, dtype=float),
            command_norm_limit=1.0e7,
            watchdog_timeout=1.0,
            maximum_dt=0.2,
            command_dimension=6,
        )
    )
    return PoseNeuroAdaptiveController(impedance, network, parameters, safety)


def test_exact_42d_feature_order_and_dimensions():
    fields = [np.arange(6, dtype=float) + 10.0 * index for index in range(7)]

    features = build_pose_nn_features(*fields)

    assert features.shape == (42,)
    np.testing.assert_array_equal(features, np.concatenate(fields))


def test_control_law_signs_dimensions_and_update_timing():
    controller = _controller()
    controller.network.output_weights[:] = np.array(
        [
            [0.3, -0.2, 0.1, 0.4, -0.1, 0.2],
            [-0.2, 0.1, 0.4, -0.3, 0.2, 0.5],
            [0.1, 0.2, -0.3, 0.2, 0.3, -0.4],
        ]
    )
    actual = np.linspace(0.1, -0.2, 6)
    velocity = np.linspace(-0.1, 0.15, 6)
    joints = np.linspace(-0.5, 0.5, 6)
    joint_rates = np.linspace(0.2, -0.3, 6)
    wrench = np.linspace(0.5, -0.25, 6)
    old_hidden = controller.network.hidden_weights.copy()
    old_output = controller.network.output_weights.copy()
    old_norm = controller.network.combined_weight_norm
    controller.start(0.0)

    output = controller.step(
        actual,
        velocity,
        joints,
        joint_rates,
        _reference(),
        wrench,
        dt=0.01,
        now=0.01,
    )

    expected_acceleration = (
        controller.impedance_model.parameters.external_gain @ wrench
    )
    expected_model_velocity = 0.01 * expected_acceleration
    expected_model_position = 0.01 * expected_model_velocity
    expected_error = expected_model_position - actual
    expected_error_velocity = expected_model_velocity - velocity
    expected_sliding = (
        expected_error_velocity
        + controller.parameters.lambda_gain @ expected_error
    )
    expected_features = build_pose_nn_features(
        joints,
        joint_rates,
        expected_model_position,
        expected_model_velocity,
        expected_acceleration,
        expected_error,
        expected_error_velocity,
    )
    sigma = np.tanh(old_hidden.T @ expected_features)
    expected_neural = old_output.T @ sigma
    expected_feedback = controller.parameters.feedback_gain @ expected_sliding
    expected_robust = (
        old_norm + controller.parameters.ideal_weight_bound
    ) * (controller.parameters.robust_gain @ expected_sliding)
    expected_external = -wrench
    expected_raw = (
        expected_neural
        + expected_feedback
        + expected_robust
        + expected_external
    )

    assert output.state == ControllerState.RUNNING
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
    ):
        assert value.shape == (6,)
    assert output.nn_features.shape == (42,)
    np.testing.assert_allclose(output.nn_features, expected_features)
    np.testing.assert_allclose(output.neural_estimate, expected_neural)
    np.testing.assert_allclose(output.feedback_term, expected_feedback)
    np.testing.assert_allclose(output.robust_term, expected_robust)
    np.testing.assert_allclose(output.external_term, expected_external)
    np.testing.assert_allclose(output.raw_command, expected_raw)
    np.testing.assert_allclose(output.command, expected_raw)
    assert not np.array_equal(controller.network.output_weights, old_output)


def test_zero_error_has_zero_command_and_no_weight_change():
    controller = _controller()
    initial_hidden = controller.network.hidden_weights.copy()
    controller.start(0.0)

    output = controller.step(
        np.zeros(6),
        np.zeros(6),
        np.zeros(6),
        np.zeros(6),
        _reference(),
        np.zeros(6),
        dt=0.01,
        now=0.01,
    )

    np.testing.assert_array_equal(output.command, np.zeros(6))
    np.testing.assert_array_equal(output.raw_command, np.zeros(6))
    np.testing.assert_array_equal(
        controller.network.hidden_weights,
        initial_hidden,
    )
    np.testing.assert_array_equal(
        controller.network.output_weights,
        np.zeros((3, 6)),
    )


def test_six_dimensional_saturation_nan_fault_and_reset_are_deterministic():
    controller = _controller(
        adaptation_enabled=False,
        command_limits=np.ones(6),
    )
    controller.start(0.0)
    saturated = controller.step(
        -np.ones(6),
        np.zeros(6),
        np.zeros(6),
        np.zeros(6),
        _reference(),
        np.zeros(6),
        dt=0.01,
        now=0.01,
    )
    assert saturated.saturated
    assert np.all(np.abs(saturated.command) <= 1.0)

    invalid = np.zeros(6)
    invalid[2] = np.nan
    faulted = controller.step(
        invalid,
        np.zeros(6),
        np.zeros(6),
        np.zeros(6),
        _reference(),
        np.zeros(6),
        dt=0.01,
        now=0.02,
    )
    assert faulted.state == ControllerState.FAULT
    np.testing.assert_array_equal(faulted.command, np.zeros(6))

    controller.reset()
    first_hidden = controller.network.hidden_weights.copy()
    controller.network.hidden_weights.fill(3.0)
    controller.reset()
    assert controller.state == ControllerState.START
    np.testing.assert_array_equal(controller.network.hidden_weights, first_hidden)
