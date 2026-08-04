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

"""Tests for power-consistent six-DoF torque realization."""

import numpy as np
import pytest

from neuro_adaptive_control.adapters.pose_wrench_to_torque import (
    PoseTorqueConfig,
    PoseWrenchToTorque,
)
from neuro_adaptive_control.core.so3 import coordinate_transform


def _adapter(*, torque_limits=None, torque_rate_limits=None):
    return PoseWrenchToTorque(
        PoseTorqueConfig.diagonal(
            torque_limits=(1000.0,) * 6 if torque_limits is None else torque_limits,
            torque_rate_limits=(
                (100000.0,) * 6
                if torque_rate_limits is None
                else torque_rate_limits
            ),
            safe_joint_damping=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
        )
    )


def test_running_mapping_matches_both_equivalent_contract_forms():
    generator = np.random.default_rng(12)
    adapter = _adapter()
    generalized_force = generator.normal(size=6)
    rho = np.array([0.25, -0.18, 0.31])
    geometric_jacobian = generator.normal(size=(6, 6))
    transform = coordinate_transform(rho)
    physical_wrench = np.linalg.solve(transform.T, generalized_force)
    analytical_jacobian = np.linalg.solve(transform, geometric_jacobian)
    expected = geometric_jacobian.T @ physical_wrench

    result = adapter.map_running_command(
        generalized_force,
        rho,
        geometric_jacobian,
        0.01,
    )

    np.testing.assert_allclose(result.physical_wrench, physical_wrench)
    np.testing.assert_allclose(result.analytical_jacobian, analytical_jacobian)
    np.testing.assert_allclose(result.raw_command, expected)
    np.testing.assert_allclose(
        result.raw_command,
        analytical_jacobian.T @ generalized_force,
    )
    np.testing.assert_allclose(result.command, expected)


def test_running_mapping_has_no_joint_damping_or_orientation_pd_term():
    adapter = _adapter()
    generalized_force = np.array([1.0, -2.0, 3.0, -0.5, 0.7, 0.2])
    jacobian = np.eye(6)
    first = adapter.map_running_command(
        generalized_force,
        np.zeros(3),
        jacobian,
        0.01,
    )
    adapter.reset()
    second = adapter.map_running_command(
        generalized_force,
        np.zeros(3),
        jacobian,
        0.01,
    )

    np.testing.assert_array_equal(first.raw_command, generalized_force)
    np.testing.assert_array_equal(second.raw_command, generalized_force)


def test_mapping_preserves_analytical_virtual_work():
    generator = np.random.default_rng(7)
    adapter = _adapter()
    generalized_force = generator.normal(size=6)
    rho = np.array([-0.4, 0.2, 0.1])
    geometric_jacobian = generator.normal(size=(6, 6))
    joint_velocity = generator.normal(size=6)
    result = adapter.map_running_command(
        generalized_force,
        rho,
        geometric_jacobian,
        0.01,
    )
    analytical_velocity = result.analytical_jacobian @ joint_velocity

    assert result.raw_command @ joint_velocity == pytest.approx(
        generalized_force @ analytical_velocity,
        abs=1.0e-12,
    )


def test_safe_damping_is_separate_dissipative_and_bounded():
    adapter = _adapter(
        torque_limits=(2.0,) * 6,
        torque_rate_limits=(1000.0,) * 6,
    )
    velocity = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])

    command = adapter.safe_stop_command(velocity, 0.01)

    assert command @ velocity < 0.0
    assert np.all(np.abs(command) <= 2.0)


def test_rate_and_absolute_limits_and_reset_are_deterministic():
    adapter = _adapter(
        torque_limits=(1.5,) * 6,
        torque_rate_limits=(10.0,) * 6,
    )
    arguments = (np.full(6, 100.0), np.zeros(3), np.eye(6), 0.1)
    first = adapter.map_running_command(*arguments)
    second = adapter.map_running_command(*arguments)
    np.testing.assert_allclose(first.command, np.ones(6))
    np.testing.assert_allclose(second.command, np.full(6, 1.5))
    assert first.rate_saturated and not first.torque_saturated
    assert second.rate_saturated and second.torque_saturated
    adapter.reset()
    repeated = adapter.map_running_command(*arguments)
    np.testing.assert_array_equal(repeated.command, first.command)


def test_invalid_inputs_and_nonfinite_raw_torque_are_rejected():
    adapter = _adapter()
    with pytest.raises(ValueError, match="rotation_vector"):
        adapter.map_running_command(np.zeros(6), np.zeros(2), np.eye(6), 0.01)
    bad_jacobian = np.eye(6)
    bad_jacobian[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        adapter.map_running_command(
            np.zeros(6), np.zeros(3), bad_jacobian, 0.01
        )
