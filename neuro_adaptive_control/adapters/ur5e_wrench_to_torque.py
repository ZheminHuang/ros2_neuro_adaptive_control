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

"""Pure NumPy mapping from a 3D TCP force to UR5e joint torque."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_ARM_DOF = 6
_TASK_DIM = 3
ROBOT_RBF_INPUT_DIM = 27


def _vector(value: Iterable[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def _matrix(
    value: Iterable[float] | np.ndarray,
    shape: tuple[int, int],
    name: str,
    *,
    diagonal_allowed: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if diagonal_allowed and array.shape == (shape[0],) and shape[0] == shape[1]:
        array = np.diag(array)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def _positive_limits(value: Iterable[float], name: str) -> np.ndarray:
    limits = _vector(value, _ARM_DOF, name)
    if np.any(limits <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")
    return limits


def _symmetric_psd(
    value: Iterable[float] | np.ndarray,
    size: int,
    name: str,
) -> np.ndarray:
    matrix = _matrix(value, (size, size), name, diagonal_allowed=True)
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    if np.min(np.linalg.eigvalsh(matrix)) < -1e-12:
        raise ValueError(f"{name} must be positive semidefinite.")
    return matrix


def _rotation_matrix(value: np.ndarray, name: str) -> np.ndarray:
    rotation = _matrix(value, (_TASK_DIM, _TASK_DIM), name)
    if not np.allclose(rotation.T @ rotation, np.eye(_TASK_DIM), atol=1e-8):
        raise ValueError(f"{name} must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise ValueError(f"{name} must have determinant +1.")
    return rotation


def build_robot_rbf_features(
    joint_position: Iterable[float],
    joint_velocity: Iterable[float],
    model_position: Iterable[float],
    model_velocity: Iterable[float],
    model_acceleration: Iterable[float],
    model_error: Iterable[float],
    model_error_velocity: Iterable[float],
) -> np.ndarray:
    """Build ``[q, qdot, x_m, xdot_m, xddot_m, e_m, edot_m]`` in order."""
    features = np.concatenate(
        (
            _vector(joint_position, _ARM_DOF, "joint_position"),
            _vector(joint_velocity, _ARM_DOF, "joint_velocity"),
            _vector(model_position, _TASK_DIM, "model_position"),
            _vector(model_velocity, _TASK_DIM, "model_velocity"),
            _vector(model_acceleration, _TASK_DIM, "model_acceleration"),
            _vector(model_error, _TASK_DIM, "model_error"),
            _vector(
                model_error_velocity,
                _TASK_DIM,
                "model_error_velocity",
            ),
        )
    )
    if features.shape != (ROBOT_RBF_INPUT_DIM,):
        raise AssertionError("the UR5e RBF feature contract must remain 27D")
    return features


def orientation_error_world(
    actual_rotation: np.ndarray,
    desired_rotation: np.ndarray,
) -> np.ndarray:
    """
    Return a desired-minus-actual local orientation error in the base frame.

    Both rotations map TCP-frame vectors into the base frame. The result is
    ``0.5 * vee(R_d R.T - R R_d.T)`` and is therefore expressed in the base
    frame used by a MuJoCo spatial angular Jacobian.
    """
    actual = _rotation_matrix(actual_rotation, "actual_rotation")
    desired = _rotation_matrix(desired_rotation, "desired_rotation")
    skew = desired @ actual.T - actual @ desired.T
    return 0.5 * np.array((skew[2, 1], skew[0, 2], skew[1, 0]))


def orientation_distance(
    actual_rotation: np.ndarray,
    desired_rotation: np.ndarray,
) -> float:
    """Return the geodesic rotation distance in radians."""
    actual = _rotation_matrix(actual_rotation, "actual_rotation")
    desired = _rotation_matrix(desired_rotation, "desired_rotation")
    cosine = 0.5 * (np.trace(desired.T @ actual) - 1.0)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


@dataclass(frozen=True)
class TorqueMappingConfig:
    """Gains and limits for the robot-specific wrench mapping."""

    orientation_stiffness: np.ndarray
    orientation_damping: np.ndarray
    joint_damping: np.ndarray
    torque_limits: np.ndarray
    torque_rate_limits: np.ndarray
    orientation_error_limit: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "orientation_stiffness",
            _symmetric_psd(
                self.orientation_stiffness,
                _TASK_DIM,
                "orientation_stiffness",
            ),
        )
        object.__setattr__(
            self,
            "orientation_damping",
            _symmetric_psd(
                self.orientation_damping,
                _TASK_DIM,
                "orientation_damping",
            ),
        )
        object.__setattr__(
            self,
            "joint_damping",
            _symmetric_psd(self.joint_damping, _ARM_DOF, "joint_damping"),
        )
        object.__setattr__(
            self,
            "torque_limits",
            _positive_limits(self.torque_limits, "torque_limits"),
        )
        object.__setattr__(
            self,
            "torque_rate_limits",
            _positive_limits(self.torque_rate_limits, "torque_rate_limits"),
        )
        try:
            angle_limit = float(self.orientation_error_limit)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("orientation_error_limit must be numeric.") from error
        if not np.isfinite(angle_limit) or not 0.0 < angle_limit < np.pi:
            raise ValueError("orientation_error_limit must lie in (0, pi).")
        object.__setattr__(self, "orientation_error_limit", angle_limit)

    @classmethod
    def diagonal(
        cls,
        orientation_stiffness: Iterable[float],
        orientation_damping: Iterable[float],
        joint_damping: Iterable[float],
        torque_limits: Iterable[float],
        torque_rate_limits: Iterable[float],
        orientation_error_limit: float,
    ) -> "TorqueMappingConfig":
        """Construct diagonal gains with per-joint actuator limits."""
        return cls(
            orientation_stiffness=np.asarray(
                tuple(orientation_stiffness), dtype=float
            ),
            orientation_damping=np.asarray(tuple(orientation_damping), dtype=float),
            joint_damping=np.asarray(tuple(joint_damping), dtype=float),
            torque_limits=np.asarray(tuple(torque_limits), dtype=float),
            torque_rate_limits=np.asarray(tuple(torque_rate_limits), dtype=float),
            orientation_error_limit=orientation_error_limit,
        )


@dataclass(frozen=True)
class TorqueMappingResult:
    """Expose the physical terms before and after actuator safety limits."""

    command: np.ndarray
    raw_command: np.ndarray
    translation_term: np.ndarray
    orientation_term: np.ndarray
    damping_term: np.ndarray
    orientation_moment: np.ndarray
    orientation_error: np.ndarray
    torque_saturated: bool
    rate_saturated: bool


class UR5eWrenchToTorque:
    """
    Map one base-frame TCP force into a bounded six-joint torque.

    ``J_v`` and ``J_w`` must be MuJoCo spatial Jacobians for the same TCP
    point and expressed in the base frame. Force, angular velocity, and the
    orientation moment use that same frame. This adapter uses kinematics only;
    it never reads a mass matrix, bias force, gravity term, or contact model.
    """

    def __init__(self, config: TorqueMappingConfig) -> None:
        self.config = config
        self._last_command = np.zeros(_ARM_DOF, dtype=float)

    @property
    def last_command(self) -> np.ndarray:
        """Return a defensive copy of the last bounded actuator command."""
        return self._last_command.copy()

    def reset(self) -> None:
        """Clear torque-rate history for a deterministic restart."""
        self._last_command.fill(0.0)

    def map_command(
        self,
        force: Iterable[float],
        translational_jacobian: np.ndarray,
        rotational_jacobian: np.ndarray,
        actual_rotation: np.ndarray,
        desired_rotation: np.ndarray,
        angular_velocity: Iterable[float],
        joint_velocity: Iterable[float],
        dt: float,
    ) -> TorqueMappingResult:
        """Add translation, orientation hold, and dissipative joint damping."""
        force_vector = _vector(force, _TASK_DIM, "force")
        jacobian_v = _matrix(
            translational_jacobian,
            (_TASK_DIM, _ARM_DOF),
            "translational_jacobian",
        )
        jacobian_w = _matrix(
            rotational_jacobian,
            (_TASK_DIM, _ARM_DOF),
            "rotational_jacobian",
        )
        omega = _vector(angular_velocity, _TASK_DIM, "angular_velocity")
        qdot = _vector(joint_velocity, _ARM_DOF, "joint_velocity")
        distance = orientation_distance(actual_rotation, desired_rotation)
        if distance > self.config.orientation_error_limit:
            raise ValueError(
                "orientation error exceeds configured limit: "
                f"{distance:.9g} > {self.config.orientation_error_limit:.9g} rad"
            )
        error = orientation_error_world(actual_rotation, desired_rotation)
        moment = (
            self.config.orientation_stiffness @ error
            - self.config.orientation_damping @ omega
        )
        translation_term = jacobian_v.T @ force_vector
        orientation_term = jacobian_w.T @ moment
        damping_term = -(self.config.joint_damping @ qdot)
        raw = translation_term + orientation_term + damping_term
        command, torque_saturated, rate_saturated = self._limit(raw, dt)
        return TorqueMappingResult(
            command=command,
            raw_command=raw.copy(),
            translation_term=translation_term.copy(),
            orientation_term=orientation_term.copy(),
            damping_term=damping_term.copy(),
            orientation_moment=moment.copy(),
            orientation_error=error.copy(),
            torque_saturated=torque_saturated,
            rate_saturated=rate_saturated,
        )

    def damping_command(
        self,
        joint_velocity: Iterable[float],
        dt: float,
    ) -> np.ndarray:
        """Produce the bounded damping-only command used while stopping."""
        qdot = _vector(joint_velocity, _ARM_DOF, "joint_velocity")
        raw = -(self.config.joint_damping @ qdot)
        command, _, _ = self._limit(raw, dt)
        return command

    def _limit(self, raw_command: np.ndarray, dt: float) -> tuple[np.ndarray, bool, bool]:
        if raw_command.shape != (_ARM_DOF,):
            raise ValueError(
                "raw_command must have shape "
                f"({_ARM_DOF},), got {raw_command.shape}."
            )
        if not np.all(np.isfinite(raw_command)):
            raise FloatingPointError("raw joint torque contains NaN or Inf.")
        try:
            step = float(dt)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("dt must be numeric.") from error
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be finite and strictly positive.")
        max_delta = self.config.torque_rate_limits * step
        delta = raw_command - self._last_command
        limited_delta = np.clip(delta, -max_delta, max_delta)
        rate_limited = self._last_command + limited_delta
        command = np.clip(
            rate_limited,
            -self.config.torque_limits,
            self.config.torque_limits,
        )
        rate_saturated = not np.array_equal(limited_delta, delta)
        torque_saturated = not np.array_equal(command, rate_limited)
        self._last_command = command.copy()
        return command.copy(), torque_saturated, rate_saturated


def task_power_residual(
    force: Iterable[float],
    moment: Iterable[float],
    translational_jacobian: np.ndarray,
    rotational_jacobian: np.ndarray,
    joint_velocity: Iterable[float],
) -> float:
    """Return the virtual-work residual before damping and safety limits."""
    force_vector = _vector(force, _TASK_DIM, "force")
    moment_vector = _vector(moment, _TASK_DIM, "moment")
    jacobian_v = _matrix(
        translational_jacobian,
        (_TASK_DIM, _ARM_DOF),
        "translational_jacobian",
    )
    jacobian_w = _matrix(
        rotational_jacobian,
        (_TASK_DIM, _ARM_DOF),
        "rotational_jacobian",
    )
    qdot = _vector(joint_velocity, _ARM_DOF, "joint_velocity")
    torque = jacobian_v.T @ force_vector + jacobian_w.T @ moment_vector
    task_power = force_vector @ (jacobian_v @ qdot)
    task_power += moment_vector @ (jacobian_w @ qdot)
    return float(torque @ qdot - task_power)
