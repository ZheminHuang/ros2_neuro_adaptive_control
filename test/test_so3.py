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

"""Tests for guarded rotation-vector coordinate operations."""

import numpy as np
import pytest

from neuro_adaptive_control.core.so3 import (
    coordinate_transform,
    exp,
    hat,
    left_jacobian,
    left_jacobian_inverse,
    log,
    vee,
)


@pytest.mark.parametrize(
    "rho",
    [
        np.zeros(3),
        np.array([1.0e-10, -2.0e-10, 3.0e-10]),
        np.array([0.2, -0.3, 0.4]),
        np.array([-1.2, 0.4, 0.3]),
    ],
)
def test_exp_log_round_trip_inside_principal_chart(rho):
    np.testing.assert_allclose(log(exp(rho)), rho, atol=2.0e-10)


def test_hat_and_vee_are_inverse_and_reproduce_cross_product():
    vector = np.array([0.3, -0.8, 1.2])
    operand = np.array([-0.4, 0.5, 0.7])

    np.testing.assert_array_equal(vee(hat(vector)), vector)
    np.testing.assert_allclose(hat(vector) @ operand, np.cross(vector, operand))


@pytest.mark.parametrize(
    "rho",
    [np.zeros(3), np.array([1.0e-9, 2.0e-9, -1.0e-9]), np.array([0.4, -0.2, 0.7])],
)
def test_left_jacobian_and_inverse_multiply_to_identity(rho):
    np.testing.assert_allclose(
        left_jacobian(rho) @ left_jacobian_inverse(rho),
        np.eye(3),
        atol=2.0e-12,
    )


def test_left_jacobian_matches_spatial_angular_velocity_finite_difference():
    rho = np.array([0.35, -0.22, 0.18])
    rho_dot = np.array([-0.12, 0.09, 0.07])
    step = 1.0e-7
    rotation = exp(rho)
    next_rotation = exp(rho + step * rho_dot)
    spatial_omega = vee(
        0.5
        * (
            (next_rotation - rotation) / step @ rotation.T
            - rotation @ ((next_rotation - rotation) / step).T
        )
    )

    np.testing.assert_allclose(
        spatial_omega,
        left_jacobian(rho) @ rho_dot,
        atol=2.0e-8,
    )


def test_coordinate_transform_contains_left_jacobian_block():
    rho = np.array([0.2, 0.1, -0.3])
    transform = coordinate_transform(rho)

    np.testing.assert_array_equal(transform[:3, :3], np.eye(3))
    np.testing.assert_array_equal(transform[:3, 3:], np.zeros((3, 3)))
    np.testing.assert_array_equal(transform[3:, :3], np.zeros((3, 3)))
    np.testing.assert_allclose(transform[3:, 3:], left_jacobian(rho))


def test_log_rejects_pi_branch_and_invalid_rotation():
    with pytest.raises(ValueError, match="outside the configured Log chart"):
        log(exp([np.pi, 0.0, 0.0]))
    with pytest.raises(ValueError, match="orthonormal"):
        log(np.diag([1.0, 1.0, 2.0]))
