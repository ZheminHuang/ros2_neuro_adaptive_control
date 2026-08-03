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

"""Tests for the pure UR5e Cartesian-force to joint-torque adapter."""

import numpy as np
import pytest

from neuro_adaptive_control.adapters.ur5e_wrench_to_torque import (
    ROBOT_RBF_INPUT_DIM,
    TorqueMappingConfig,
    UR5eWrenchToTorque,
    build_robot_rbf_features,
    orientation_distance,
    orientation_error_world,
    task_power_residual,
)


def _rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )


def _config(
    *,
    torque_limits=(1000.0,) * 6,
    torque_rate_limits=(100000.0,) * 6,
    orientation_error_limit=1.0,
) -> TorqueMappingConfig:
    return TorqueMappingConfig.diagonal(
        orientation_stiffness=(4.0, 5.0, 6.0),
        orientation_damping=(0.5, 0.6, 0.7),
        joint_damping=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        torque_limits=torque_limits,
        torque_rate_limits=torque_rate_limits,
        orientation_error_limit=orientation_error_limit,
    )


def test_robot_rbf_features_have_exact_order_and_dimension() -> None:
    blocks = (
        np.arange(0.0, 6.0),
        np.arange(10.0, 16.0),
        np.arange(20.0, 23.0),
        np.arange(30.0, 33.0),
        np.arange(40.0, 43.0),
        np.arange(50.0, 53.0),
        np.arange(60.0, 63.0),
    )
    features = build_robot_rbf_features(*blocks)
    assert features.shape == (ROBOT_RBF_INPUT_DIM,)
    np.testing.assert_array_equal(features, np.concatenate(blocks))


@pytest.mark.parametrize("bad_index", range(7))
def test_robot_rbf_features_reject_invalid_blocks(bad_index: int) -> None:
    blocks = [
        np.zeros(6),
        np.zeros(6),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
    ]
    blocks[bad_index] = np.array((np.nan,))
    with pytest.raises(ValueError):
        build_robot_rbf_features(*blocks)


def test_orientation_error_has_desired_minus_actual_sign() -> None:
    angle = 0.1
    error = orientation_error_world(np.eye(3), _rotation_z(angle))
    np.testing.assert_allclose(error, (0.0, 0.0, np.sin(angle)), atol=1e-14)
    assert orientation_distance(np.eye(3), _rotation_z(angle)) == pytest.approx(angle)


@pytest.mark.parametrize(
    "rotation",
    (np.ones((3, 3)), np.diag((1.0, 1.0, -1.0)), np.full((3, 3), np.nan)),
)
def test_orientation_helpers_reject_invalid_rotations(rotation: np.ndarray) -> None:
    with pytest.raises(ValueError):
        orientation_error_world(rotation, np.eye(3))


def test_mapping_terms_have_explicit_signs_and_dimensions() -> None:
    adapter = UR5eWrenchToTorque(_config())
    jacobian_v = np.arange(18.0).reshape(3, 6) / 10.0
    jacobian_w = np.flip(jacobian_v, axis=1) / 2.0
    force = np.array((1.0, -2.0, 3.0))
    omega = np.array((0.3, -0.2, 0.1))
    qdot = np.arange(1.0, 7.0) / 10.0
    desired = _rotation_z(0.1)
    output = adapter.map_command(
        force,
        jacobian_v,
        jacobian_w,
        np.eye(3),
        desired,
        omega,
        qdot,
        0.01,
    )
    error = np.array((0.0, 0.0, np.sin(0.1)))
    moment = np.diag((4.0, 5.0, 6.0)) @ error
    moment -= np.diag((0.5, 0.6, 0.7)) @ omega
    translation = jacobian_v.T @ force
    orientation = jacobian_w.T @ moment
    damping = -np.diag((0.1, 0.2, 0.3, 0.4, 0.5, 0.6)) @ qdot
    np.testing.assert_allclose(output.orientation_error, error)
    np.testing.assert_allclose(output.orientation_moment, moment)
    np.testing.assert_allclose(output.translation_term, translation)
    np.testing.assert_allclose(output.orientation_term, orientation)
    np.testing.assert_allclose(output.damping_term, damping)
    np.testing.assert_allclose(output.raw_command, translation + orientation + damping)
    np.testing.assert_allclose(output.command, output.raw_command)
    assert not output.torque_saturated
    assert not output.rate_saturated


def test_task_mapping_preserves_virtual_work_before_nonlinear_limits() -> None:
    generator = np.random.default_rng(17)
    jacobian_v = generator.normal(size=(3, 6))
    jacobian_w = generator.normal(size=(3, 6))
    qdot = generator.normal(size=6)
    force = generator.normal(size=3)
    moment = generator.normal(size=3)
    residual = task_power_residual(
        force, moment, jacobian_v, jacobian_w, qdot
    )
    assert abs(residual) < 1e-12


def test_joint_damping_is_dissipative() -> None:
    adapter = UR5eWrenchToTorque(_config())
    qdot = np.array((1.0, -2.0, 3.0, -4.0, 5.0, -6.0))
    command = adapter.damping_command(qdot, 0.01)
    assert float(command @ qdot) < 0.0


def test_rate_then_absolute_torque_limits_and_reset_are_deterministic() -> None:
    adapter = UR5eWrenchToTorque(
        _config(torque_limits=(1.5,) * 6, torque_rate_limits=(10.0,) * 6)
    )
    jacobian = np.vstack((np.ones(6), np.zeros(6), np.zeros(6)))
    arguments = (
        (100.0, 0.0, 0.0),
        jacobian,
        np.zeros((3, 6)),
        np.eye(3),
        np.eye(3),
        np.zeros(3),
        np.zeros(6),
        0.1,
    )
    first = adapter.map_command(*arguments)
    second = adapter.map_command(*arguments)
    np.testing.assert_allclose(first.command, np.ones(6))
    np.testing.assert_allclose(second.command, np.full(6, 1.5))
    assert first.rate_saturated and not first.torque_saturated
    assert second.rate_saturated and second.torque_saturated
    adapter.reset()
    repeated = adapter.map_command(*arguments)
    np.testing.assert_array_equal(repeated.command, first.command)


def test_orientation_distance_guard_rejects_unsafe_attitude() -> None:
    adapter = UR5eWrenchToTorque(_config(orientation_error_limit=0.2))
    with pytest.raises(ValueError, match="orientation error exceeds"):
        adapter.map_command(
            np.zeros(3),
            np.zeros((3, 6)),
            np.zeros((3, 6)),
            np.eye(3),
            _rotation_z(0.3),
            np.zeros(3),
            np.zeros(6),
            0.002,
        )


@pytest.mark.parametrize("dt", (0.0, -0.1, np.nan, np.inf, "bad"))
def test_mapping_rejects_invalid_dt(dt) -> None:
    adapter = UR5eWrenchToTorque(_config())
    with pytest.raises(ValueError):
        adapter.damping_command(np.zeros(6), dt)


def test_mapping_rejects_nonfinite_kinematic_input() -> None:
    adapter = UR5eWrenchToTorque(_config())
    bad_jacobian = np.zeros((3, 6))
    bad_jacobian[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        adapter.map_command(
            np.zeros(3),
            bad_jacobian,
            np.zeros((3, 6)),
            np.eye(3),
            np.eye(3),
            np.zeros(3),
            np.zeros(6),
            0.002,
        )


def test_mapping_rejects_overflow_before_saturation() -> None:
    adapter = UR5eWrenchToTorque(_config())
    jacobian = np.full((3, 6), 1e308)
    with np.errstate(over="ignore"):
        with pytest.raises(FloatingPointError, match="raw joint torque"):
            adapter.map_command(
                np.full(3, 1e308),
                jacobian,
                np.zeros((3, 6)),
                np.eye(3),
                np.eye(3),
                np.zeros(3),
                np.zeros(6),
                0.002,
            )


@pytest.mark.parametrize(
    "keyword,value",
    (
        ("torque_limits", (0.0,) * 6),
        ("torque_rate_limits", (-1.0,) * 6),
        ("orientation_error_limit", np.pi),
    ),
)
def test_mapping_config_rejects_invalid_limits(keyword: str, value) -> None:
    arguments = {
        "torque_limits": (10.0,) * 6,
        "torque_rate_limits": (100.0,) * 6,
        "orientation_error_limit": 1.0,
    }
    arguments[keyword] = value
    with pytest.raises(ValueError):
        _config(**arguments)
