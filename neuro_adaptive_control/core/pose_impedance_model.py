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

"""Six-dimensional translation/rotation-vector impedance reference model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_DIMENSION = 6


def _vector6(value: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (_DIMENSION,):
        raise ValueError(f"{name} must have shape (6,), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def _matrix6(value: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (_DIMENSION,):
        array = np.diag(array)
    if array.shape != (_DIMENSION, _DIMENSION):
        raise ValueError(f"{name} must have shape (6, 6), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


@dataclass(frozen=True)
class PoseImpedanceParameters:
    """Constant matrices for the six-dimensional prescribed impedance."""

    mass: np.ndarray
    damping: np.ndarray
    stiffness: np.ndarray
    external_gain: np.ndarray

    def __post_init__(self) -> None:
        for name in ("mass", "damping", "stiffness", "external_gain"):
            object.__setattr__(self, name, _matrix6(getattr(self, name), name))
        for name in ("mass", "damping", "stiffness"):
            matrix = getattr(self, name)
            if not np.allclose(matrix, matrix.T, atol=1.0e-12):
                raise ValueError(f"{name} must be symmetric.")
        if np.min(np.linalg.eigvalsh(self.mass)) <= 0.0:
            raise ValueError("mass must be positive definite.")
        for name in ("damping", "stiffness"):
            if np.min(np.linalg.eigvalsh(getattr(self, name))) < 0.0:
                raise ValueError(f"{name} must be positive semidefinite.")

    @classmethod
    def diagonal(
        cls,
        mass: Iterable[float],
        damping: Iterable[float],
        stiffness: Iterable[float],
        external_gain: Iterable[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ) -> "PoseImpedanceParameters":
        """Construct independent translation and rotation-vector channels."""
        return cls(
            mass=np.asarray(tuple(mass), dtype=float),
            damping=np.asarray(tuple(damping), dtype=float),
            stiffness=np.asarray(tuple(stiffness), dtype=float),
            external_gain=np.asarray(tuple(external_gain), dtype=float),
        )


@dataclass(frozen=True)
class PoseImpedanceState:
    """Pose-coordinate position, velocity, and acceleration."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class PoseImpedanceModel:
    """Integrate the prescribed six-dimensional model by semi-implicit Euler."""

    def __init__(
        self,
        parameters: PoseImpedanceParameters,
        initial_position: Iterable[float] = (0.0,) * _DIMENSION,
        initial_velocity: Iterable[float] = (0.0,) * _DIMENSION,
    ) -> None:
        self.parameters = parameters
        self.reset(initial_position, initial_velocity)

    @property
    def state(self) -> PoseImpedanceState:
        """Return a defensive snapshot."""
        return PoseImpedanceState(
            self._position.copy(),
            self._velocity.copy(),
            self._acceleration.copy(),
        )

    def reset(
        self,
        position: Iterable[float] = (0.0,) * _DIMENSION,
        velocity: Iterable[float] = (0.0,) * _DIMENSION,
    ) -> PoseImpedanceState:
        """Restore a deterministic model state."""
        self._position = _vector6(position, "position")
        self._velocity = _vector6(velocity, "velocity")
        self._acceleration = np.zeros(_DIMENSION, dtype=float)
        return self.state

    def auxiliary_input(
        self,
        reference_position: Iterable[float],
        reference_velocity: Iterable[float],
        reference_acceleration: Iterable[float],
    ) -> np.ndarray:
        """Return ``M_m xdd_d + D_m xd_d + K_m x_d``."""
        position = _vector6(reference_position, "reference_position")
        velocity = _vector6(reference_velocity, "reference_velocity")
        acceleration = _vector6(reference_acceleration, "reference_acceleration")
        params = self.parameters
        return (
            params.mass @ acceleration
            + params.damping @ velocity
            + params.stiffness @ position
        )

    def step(
        self,
        reference_position: Iterable[float],
        reference_velocity: Iterable[float],
        reference_acceleration: Iterable[float],
        generalized_external_wrench: Iterable[float],
        dt: float,
    ) -> PoseImpedanceState:
        """Advance one coherent sample of the six-dimensional model."""
        step = float(dt)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be finite and strictly positive.")
        wrench = _vector6(
            generalized_external_wrench,
            "generalized_external_wrench",
        )
        params = self.parameters
        rhs = (
            params.external_gain @ wrench
            + self.auxiliary_input(
                reference_position,
                reference_velocity,
                reference_acceleration,
            )
            - params.damping @ self._velocity
            - params.stiffness @ self._position
        )
        acceleration = np.linalg.solve(params.mass, rhs)
        velocity = self._velocity + step * acceleration
        position = self._position + step * velocity
        if not all(np.all(np.isfinite(value)) for value in (acceleration, velocity, position)):
            raise FloatingPointError("pose impedance integration produced NaN or Inf.")
        self._acceleration = acceleration
        self._velocity = velocity
        self._position = position
        return self.state
