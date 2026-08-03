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

"""Unit tests for the ROS-independent Gaussian RBF approximator."""

import numpy as np
import pytest

from neuro_adaptive_control.core.rbf_network import RBFNetwork


def test_activation_and_forward_shapes_and_known_values():
    centers = np.array([[0.0, 0.0], [1.0, -1.0]])
    network = RBFNetwork(
        input_dim=2,
        output_dim=3,
        num_basis=2,
        centers=centers,
        widths=np.array([1.0, 2.0]),
        input_scale=np.array([2.0, 4.0]),
        feature_clip=10.0,
    )
    network.weights[:] = np.array([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]])

    # [2, -4] is normalized to [1, -1].  The squared distances from the
    # two centers are therefore 2 and 0, respectively.
    expected_phi = np.array([np.exp(-1.0), 1.0])
    phi = network.activations(np.array([2.0, -4.0]))
    estimate = network.forward(np.array([2.0, -4.0]))

    assert phi.shape == (2,)
    assert estimate.shape == (3,)
    np.testing.assert_allclose(phi, expected_phi, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        estimate,
        network.weights.T @ expected_phi,
        rtol=0.0,
        atol=1e-15,
    )


def test_feature_normalization_is_clipped_before_activation():
    network = RBFNetwork(
        input_dim=2,
        output_dim=1,
        num_basis=1,
        centers=np.zeros((1, 2)),
        widths=1.0,
        input_scale=np.array([2.0, 4.0]),
        feature_clip=0.5,
    )

    # Both normalized coordinates saturate at +/-0.5.
    expected = np.exp(-0.5 * (0.5**2 + (-0.5) ** 2))
    np.testing.assert_allclose(
        network.activations([20.0, -40.0]), [expected], rtol=0.0, atol=1e-15
    )


def test_weight_update_matches_explicit_euler_adaptation_law():
    network = RBFNetwork(
        input_dim=1,
        output_dim=2,
        num_basis=2,
        centers=np.array([[0.0], [1.0]]),
        widths=np.array([1.0, 0.5]),
        learning_rate=2.5,
        leakage=0.2,
        weight_limit=100.0,
    )
    initial = np.array([[0.2, -0.4], [1.0, 0.5]])
    network.weights[:] = initial
    features = np.array([0.25])
    sliding_error = np.array([0.6, -0.8])
    dt = 0.04

    phi = network.activations(features)
    derivative = 2.5 * (
        np.outer(phi, sliding_error)
        - 0.2 * np.linalg.norm(sliding_error) * initial
    )
    expected = initial + dt * derivative

    network.update(features, sliding_error, dt)

    np.testing.assert_allclose(
        network.weights, expected, rtol=1e-14, atol=1e-14
    )


def test_disabled_adaptation_leaves_weights_unchanged():
    network = RBFNetwork(
        input_dim=2,
        output_dim=3,
        num_basis=4,
        adaptation_enabled=False,
    )
    initial = np.arange(12, dtype=float).reshape(4, 3) / 10.0
    network.weights[:] = initial

    network.update([0.2, -0.3], [1.0, -2.0, 3.0], dt=0.01)

    np.testing.assert_array_equal(network.weights, initial)


def test_weight_update_projects_to_frobenius_norm_limit():
    network = RBFNetwork(
        input_dim=1,
        output_dim=2,
        num_basis=3,
        centers=np.zeros((3, 1)),
        widths=1.0,
        learning_rate=100.0,
        leakage=0.0,
        weight_limit=0.25,
    )

    network.update([0.0], [3.0, 4.0], dt=1.0)

    assert network.weight_norm == pytest.approx(0.25, abs=1e-15)
    assert np.all(np.isfinite(network.weights))


def test_reset_is_deterministic_and_restores_zero_estimate():
    first = RBFNetwork(input_dim=3, output_dim=3, num_basis=5, seed=19)
    second = RBFNetwork(input_dim=3, output_dim=3, num_basis=5, seed=19)
    features = np.array([0.2, -0.1, 0.3])
    first.update(features, [1.0, -2.0, 0.5], dt=0.02)
    assert first.weight_norm > 0.0

    first.reset()

    np.testing.assert_array_equal(first.weights, np.zeros((5, 3)))
    np.testing.assert_array_equal(first.forward(features), np.zeros(3))
    np.testing.assert_array_equal(first.centers, second.centers)
    np.testing.assert_array_equal(
        first.activations(features), second.activations(features)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_dim": 0},
        {"input_dim": 1, "output_dim": 0},
        {"input_dim": 1, "num_basis": 0},
        {"input_dim": 2, "num_basis": 2, "centers": np.zeros((2, 3))},
        {
            "input_dim": 2,
            "num_basis": 2,
            "centers": np.array([[0.0, 0.0], [np.nan, 0.0]]),
        },
        {"input_dim": 1, "num_basis": 2, "widths": [1.0]},
        {"input_dim": 1, "num_basis": 2, "widths": [1.0, 0.0]},
        {"input_dim": 1, "num_basis": 2, "widths": [1.0, np.inf]},
        {"input_dim": 2, "input_scale": [1.0]},
        {"input_dim": 2, "input_scale": [1.0, 0.0]},
        {"input_dim": 2, "input_scale": [1.0, np.nan]},
        {"input_dim": 1, "feature_clip": 0.0},
        {"input_dim": 1, "learning_rate": -1.0},
        {"input_dim": 1, "leakage": -1.0},
        {"input_dim": 1, "weight_limit": 0.0},
    ],
)
def test_invalid_constructor_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RBFNetwork(**kwargs)


@pytest.mark.parametrize(
    "operation",
    [
        lambda network: network.activations([0.0]),
        lambda network: network.activations([np.nan, 0.0]),
        lambda network: network.activations([0.0, np.inf]),
        lambda network: network.forward([0.0, 0.0, 0.0]),
        lambda network: network.update([0.0, 0.0], [1.0], 0.01),
        lambda network: network.update([0.0, 0.0], [1.0, np.nan], 0.01),
        lambda network: network.update([0.0, 0.0], [1.0, 2.0], 0.0),
        lambda network: network.update([0.0, 0.0], [1.0, 2.0], np.inf),
    ],
)
def test_invalid_runtime_inputs_are_rejected_without_mutating_weights(
    operation,
):
    network = RBFNetwork(input_dim=2, output_dim=2, num_basis=3)
    initial = np.arange(6, dtype=float).reshape(3, 2) / 10.0
    network.weights[:] = initial

    with pytest.raises(ValueError):
        operation(network)

    np.testing.assert_array_equal(network.weights, initial)


def test_nonfinite_weight_output_and_update_are_trapped():
    network = RBFNetwork(input_dim=1, output_dim=1, num_basis=1)
    network.weights[0, 0] = np.inf

    with pytest.raises(FloatingPointError):
        network.forward([0.0])
    with np.errstate(invalid="ignore"):
        with pytest.raises(FloatingPointError):
            network.update([0.0], [1.0], dt=0.01)
