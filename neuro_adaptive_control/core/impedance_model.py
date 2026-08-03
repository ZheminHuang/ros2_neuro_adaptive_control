"""Three-dimensional Cartesian impedance reference model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_DIMENSION = 3


def _vector3(value: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (_DIMENSION,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def _matrix3(value: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (_DIMENSION,):
        array = np.diag(array)
    if array.shape != (_DIMENSION, _DIMENSION):
        raise ValueError(f"{name} must have shape (3, 3), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


@dataclass(frozen=True)
class ImpedanceParameters:
    """Constant matrices for the prescribed translational impedance."""

    mass: np.ndarray
    damping: np.ndarray
    stiffness: np.ndarray
    external_gain: np.ndarray

    def __post_init__(self) -> None:
        matrices = {
            "mass": _matrix3(self.mass, "mass"),
            "damping": _matrix3(self.damping, "damping"),
            "stiffness": _matrix3(self.stiffness, "stiffness"),
            "external_gain": _matrix3(self.external_gain, "external_gain"),
        }
        for name, matrix in matrices.items():
            object.__setattr__(self, name, matrix)

        for name in ("mass", "damping", "stiffness"):
            matrix = matrices[name]
            if not np.allclose(matrix, matrix.T, atol=1e-12):
                raise ValueError(f"{name} must be symmetric.")

        if np.min(np.linalg.eigvalsh(matrices["mass"])) <= 0.0:
            raise ValueError("mass must be positive definite.")
        for name in ("damping", "stiffness"):
            if np.min(np.linalg.eigvalsh(matrices[name])) < 0.0:
                raise ValueError(f"{name} must be positive semidefinite.")

    @classmethod
    def diagonal(
        cls,
        mass: Iterable[float],
        damping: Iterable[float],
        stiffness: Iterable[float],
        external_gain: Iterable[float] = (1.0, 1.0, 1.0),
    ) -> "ImpedanceParameters":
        """Construct the common independent-axis parameterization."""
        return cls(
            mass=np.asarray(tuple(mass), dtype=float),
            damping=np.asarray(tuple(damping), dtype=float),
            stiffness=np.asarray(tuple(stiffness), dtype=float),
            external_gain=np.asarray(tuple(external_gain), dtype=float),
        )


@dataclass(frozen=True)
class ImpedanceState:
    """Position, velocity, and acceleration of the impedance model."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class CartesianImpedanceModel:
    """
    Integrate the prescribed impedance with semi-implicit Euler.

    The continuous contract is

    ``M_m xdd_m + D_m xd_m + K_m x_m = K_h w_ext + f_a``

    with full moving-reference feedforward

    ``f_a = M_m xdd_d + D_m xd_d + K_m x_d``.
    """

    def __init__(
        self,
        parameters: ImpedanceParameters,
        initial_position: Iterable[float] = (0.0, 0.0, 0.0),
        initial_velocity: Iterable[float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.parameters = parameters
        self.reset(initial_position, initial_velocity)

    @property
    def state(self) -> ImpedanceState:
        """Return a defensive snapshot of the current model state."""
        return ImpedanceState(
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            acceleration=self._acceleration.copy(),
        )

    def reset(
        self,
        position: Iterable[float] = (0.0, 0.0, 0.0),
        velocity: Iterable[float] = (0.0, 0.0, 0.0),
    ) -> ImpedanceState:
        """Restore a deterministic model state."""
        validated_position = _vector3(position, "position")
        validated_velocity = _vector3(velocity, "velocity")
        self._position = validated_position
        self._velocity = validated_velocity
        self._acceleration = np.zeros(_DIMENSION, dtype=float)
        return self.state

    def auxiliary_input(
        self,
        reference_position: Iterable[float],
        reference_velocity: Iterable[float],
        reference_acceleration: Iterable[float],
    ) -> np.ndarray:
        """Compute full dynamic reference feedforward ``f_a``."""
        position = _vector3(reference_position, "reference_position")
        velocity = _vector3(reference_velocity, "reference_velocity")
        acceleration = _vector3(
            reference_acceleration, "reference_acceleration"
        )
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
        external_wrench: Iterable[float],
        dt: float,
    ) -> ImpedanceState:
        """Advance once using one coherent external-wrench snapshot."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive.")
        wrench = _vector3(external_wrench, "external_wrench")
        feedforward = self.auxiliary_input(
            reference_position,
            reference_velocity,
            reference_acceleration,
        )
        params = self.parameters
        rhs = (
            params.external_gain @ wrench
            + feedforward
            - params.damping @ self._velocity
            - params.stiffness @ self._position
        )
        acceleration = np.linalg.solve(params.mass, rhs)
        next_velocity = self._velocity + dt * acceleration
        next_position = self._position + dt * next_velocity
        if not (
            np.all(np.isfinite(acceleration))
            and np.all(np.isfinite(next_velocity))
            and np.all(np.isfinite(next_position))
        ):
            raise FloatingPointError("impedance integration produced NaN or Inf.")
        self._acceleration = acceleration
        self._velocity = next_velocity
        self._position = next_position
        return self.state
