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

"""Tests for the two-adaptive-layer six-DoF neural approximator."""

import numpy as np
import pytest

from neuro_adaptive_control.core.two_layer_network import (
    TwoLayerAdaptiveNetwork,
    TwoLayerWeights,
)


def _network(**overrides):
    arguments = dict(
        input_dim=3,
        hidden_dim=2,
        output_dim=2,
        hidden_learning_rate=(0.2, 0.3, 0.4),
        output_learning_rate=(1.5, 2.0),
        leakage=0.1,
        hidden_weight_limit=100.0,
        output_weight_limit=100.0,
        input_scale=(2.0, 1.0, 0.5),
        input_clip=10.0,
        initial_hidden_scale=0.1,
        seed=4,
    )
    arguments.update(overrides)
    return TwoLayerAdaptiveNetwork(**arguments)


def test_forward_shape_and_numeric_value_use_tanh_two_layer_contract():
    network = _network(adaptation_enabled=False)
    network.hidden_weights[:] = np.array(
        [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5]]
    )
    network.output_weights[:] = np.array([[1.0, -2.0], [0.5, 0.25]])
    features = np.array([1.0, -0.5, 0.25])
    normalized = features / np.array([2.0, 1.0, 0.5])
    expected_sigma = np.tanh(network.hidden_weights.T @ normalized)

    estimate = network.forward(features)

    assert estimate.shape == (2,)
    np.testing.assert_allclose(
        estimate,
        network.output_weights.T @ expected_sigma,
    )


def test_update_matches_discrete_published_v_and_w_adaptation_laws():
    network = _network()
    network.hidden_weights[:] = np.array(
        [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5]]
    )
    network.output_weights[:] = np.array([[1.0, -2.0], [0.5, 0.25]])
    features = np.array([1.0, -0.5, 0.25])
    error = np.array([0.3, -0.7])
    dt = 0.01
    old_hidden = network.hidden_weights.copy()
    old_output = network.output_weights.copy()
    vector, sigma, derivative = network.activation(features)
    error_norm = np.linalg.norm(error)
    preactivation = old_hidden.T @ vector
    corrected = sigma - derivative * preactivation
    expected_output_dot = np.array([1.5, 2.0])[:, None] * (
        np.outer(corrected, error) - 0.1 * error_norm * old_output
    )
    hidden_error = derivative * (old_output @ error)
    expected_hidden_dot = np.array([0.2, 0.3, 0.4])[:, None] * (
        np.outer(vector, hidden_error) - 0.1 * error_norm * old_hidden
    )

    network.update(features, error, dt)

    np.testing.assert_allclose(
        network.output_weights,
        old_output + dt * expected_output_dot,
    )
    np.testing.assert_allclose(
        network.hidden_weights,
        old_hidden + dt * expected_hidden_dot,
    )


def test_checkpoint_freeze_restore_and_reset_are_deterministic():
    network = _network()
    initial_hidden = network.hidden_weights.copy()
    network.update([1.0, -0.5, 0.2], [0.3, -0.1], 0.01)
    checkpoint = network.checkpoint()
    network.adaptation_enabled = False
    network.update([-1.0, 0.8, 0.4], [-0.2, 0.5], 0.02)
    np.testing.assert_array_equal(network.hidden_weights, checkpoint.hidden)
    np.testing.assert_array_equal(network.output_weights, checkpoint.output)

    network.hidden_weights.fill(9.0)
    network.output_weights.fill(8.0)
    network.restore(checkpoint)
    np.testing.assert_array_equal(network.hidden_weights, checkpoint.hidden)
    np.testing.assert_array_equal(network.output_weights, checkpoint.output)

    network.reset()
    np.testing.assert_array_equal(network.hidden_weights, initial_hidden)
    np.testing.assert_array_equal(network.output_weights, np.zeros((2, 2)))


def test_nonzero_output_initialization_is_seeded_and_restored_by_reset():
    first = _network(initial_output_scale=0.01)
    second = _network(initial_output_scale=0.01)
    initial_hidden = first.hidden_weights.copy()
    initial_output = first.output_weights.copy()

    assert np.any(initial_output != 0.0)
    assert np.max(np.abs(initial_output)) <= 0.01
    np.testing.assert_array_equal(first.hidden_weights, second.hidden_weights)
    np.testing.assert_array_equal(first.output_weights, second.output_weights)

    first.update([0.4, -0.2, 0.1], [0.3, -0.1], 0.01)
    first.reset()
    np.testing.assert_array_equal(first.hidden_weights, initial_hidden)
    np.testing.assert_array_equal(first.output_weights, initial_output)


def test_invalid_shapes_nan_and_bad_checkpoint_are_rejected():
    network = _network()
    with pytest.raises(ValueError, match="features must have shape"):
        network.forward(np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        network.forward([0.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="invalid shape"):
        network.restore(TwoLayerWeights(np.zeros((2, 2)), np.zeros((2, 2))))


def test_invalid_output_initialization_scale_is_rejected():
    with pytest.raises(ValueError, match="initial_output_scale"):
        _network(initial_output_scale=-0.01)
