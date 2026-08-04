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

"""Power-consistent six-DoF analytical-force to joint-torque mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from neuro_adaptive_control.core.so3 import coordinate_transform


_DIMENSION = 6


def _vector3(value: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def _vector6(value: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (_DIMENSION,):
        raise ValueError(f"{name} must have shape (6,), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def _matrix6(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (_DIMENSION, _DIMENSION):
        raise ValueError(f"{name} must have shape (6, 6), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


@dataclass(frozen=True)
class PoseTorqueConfig:
    """Actuator bounds and stopping/fault-only dissipative gain."""

    torque_limits: np.ndarray
    torque_rate_limits: np.ndarray
    safe_joint_damping: np.ndarray

    def __post_init__(self) -> None:
        for name in ("torque_limits", "torque_rate_limits"):
            vector = _vector6(getattr(self, name), name)
            if np.any(vector <= 0.0):
                raise ValueError(f"{name} must be strictly positive.")
            object.__setattr__(self, name, vector)
        damping = np.asarray(self.safe_joint_damping, dtype=float)
        if damping.shape == (6,):
            damping = np.diag(damping)
        if damping.shape != (6, 6):
            raise ValueError("safe_joint_damping must have shape (6, 6).")
        if not np.all(np.isfinite(damping)):
            raise ValueError("safe_joint_damping must contain only finite values.")
        if not np.allclose(damping, damping.T, atol=1.0e-12):
            raise ValueError("safe_joint_damping must be symmetric.")
        if np.min(np.linalg.eigvalsh(damping)) < 0.0:
            raise ValueError("safe_joint_damping must be positive semidefinite.")
        object.__setattr__(self, "safe_joint_damping", damping.copy())

    @classmethod
    def diagonal(
        cls,
        torque_limits: Iterable[float],
        torque_rate_limits: Iterable[float],
        safe_joint_damping: Iterable[float],
    ) -> "PoseTorqueConfig":
        """Construct per-joint limits and stopping damping."""
        return cls(
            np.asarray(tuple(torque_limits), dtype=float),
            np.asarray(tuple(torque_rate_limits), dtype=float),
            np.asarray(tuple(safe_joint_damping), dtype=float),
        )


@dataclass(frozen=True)
class PoseTorqueResult:
    """Expose coordinate conversion and actuator limiting for diagnostics."""

    command: np.ndarray
    raw_command: np.ndarray
    physical_wrench: np.ndarray
    analytical_jacobian: np.ndarray
    torque_saturated: bool
    rate_saturated: bool


class PoseWrenchToTorque:
    """Realize ``tau = J_g.T E(rho)^-T u`` without running damping/PD."""

    def __init__(self, config: PoseTorqueConfig) -> None:
        self.config = config
        self._last_command = np.zeros(6)

    @property
    def last_command(self) -> np.ndarray:
        """Return the most recently bounded joint command."""
        return self._last_command.copy()

    def reset(self) -> None:
        """Clear torque-rate history for deterministic replay."""
        self._last_command.fill(0.0)

    def map_running_command(
        self,
        generalized_force: Iterable[float],
        rotation_vector: Iterable[float],
        geometric_jacobian: np.ndarray,
        dt: float,
    ) -> PoseTorqueResult:
        """Map one analytical generalized force during running state."""
        force = _vector6(generalized_force, "generalized_force")
        rho = _vector3(rotation_vector, "rotation_vector")
        jacobian = _matrix6(geometric_jacobian, "geometric_jacobian")
        transform = coordinate_transform(rho)
        physical_wrench = np.linalg.solve(transform.T, force)
        analytical_jacobian = np.linalg.solve(transform, jacobian)
        raw = jacobian.T @ physical_wrench
        command, torque_saturated, rate_saturated = self._limit(raw, dt)
        return PoseTorqueResult(
            command=command,
            raw_command=raw.copy(),
            physical_wrench=physical_wrench.copy(),
            analytical_jacobian=analytical_jacobian.copy(),
            torque_saturated=torque_saturated,
            rate_saturated=rate_saturated,
        )

    def safe_stop_command(
        self,
        joint_velocity: Iterable[float],
        dt: float,
    ) -> np.ndarray:
        """Apply bounded ``-D_q_safe qdot`` only while stopping or faulted."""
        velocity = _vector6(joint_velocity, "joint_velocity")
        command, _, _ = self._limit(
            -(self.config.safe_joint_damping @ velocity),
            dt,
        )
        return command

    def _limit(self, raw: np.ndarray, dt: float) -> tuple[np.ndarray, bool, bool]:
        if not np.all(np.isfinite(raw)):
            raise FloatingPointError("raw joint torque contains NaN or Inf.")
        step = float(dt)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be finite and strictly positive.")
        maximum_delta = self.config.torque_rate_limits * step
        delta = raw - self._last_command
        limited_delta = np.clip(delta, -maximum_delta, maximum_delta)
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
