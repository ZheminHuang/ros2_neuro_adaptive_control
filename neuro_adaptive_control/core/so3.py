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

"""Numerically guarded SO(3) operations for rotation-vector coordinates."""

from __future__ import annotations

from typing import Iterable

import numpy as np


_SMALL_ANGLE = 1.0e-7


def _vector3(value: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def validate_rotation_matrix(value: np.ndarray, name: str = "rotation") -> np.ndarray:
    """Return a defensive copy of a proper orthonormal rotation matrix."""
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {rotation.shape}.")
    if not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8):
        raise ValueError(f"{name} must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-8):
        raise ValueError(f"{name} must have determinant +1.")
    return rotation.copy()


def hat(vector: Iterable[float]) -> np.ndarray:
    """Return the cross-product matrix ``vector^``."""
    x, y, z = _vector3(vector, "vector")
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def vee(matrix: np.ndarray) -> np.ndarray:
    """Return the vector associated with a 3-by-3 skew matrix."""
    array = np.asarray(matrix, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must contain only finite values.")
    if not np.allclose(array + array.T, 0.0, atol=1.0e-10):
        raise ValueError("matrix must be skew-symmetric.")
    return np.array((array[2, 1], array[0, 2], array[1, 0]))


def exp(rotation_vector: Iterable[float]) -> np.ndarray:
    """Evaluate the SO(3) exponential map with a small-angle series."""
    rho = _vector3(rotation_vector, "rotation_vector")
    theta = float(np.linalg.norm(rho))
    rho_hat = hat(rho)
    if theta < _SMALL_ANGLE:
        theta2 = theta * theta
        a = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
        b = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * rho_hat + b * (rho_hat @ rho_hat)


def log(
    rotation: np.ndarray,
    *,
    chart_margin: float = 1.0e-6,
) -> np.ndarray:
    """Evaluate the principal SO(3) logarithm inside the guarded chart."""
    matrix = validate_rotation_matrix(rotation)
    margin = float(chart_margin)
    if not np.isfinite(margin) or not 0.0 < margin < np.pi:
        raise ValueError("chart_margin must lie in (0, pi).")
    cosine = float(np.clip(0.5 * (np.trace(matrix) - 1.0), -1.0, 1.0))
    theta = float(np.arccos(cosine))
    if theta >= np.pi - margin:
        raise ValueError("rotation lies outside the configured Log chart.")
    skew_vector = np.array(
        (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        )
    )
    if theta < _SMALL_ANGLE:
        scale = 0.5 + theta * theta / 12.0
    else:
        scale = theta / (2.0 * np.sin(theta))
    result = scale * skew_vector
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("SO(3) logarithm produced NaN or Inf.")
    return result


def left_jacobian(rotation_vector: Iterable[float]) -> np.ndarray:
    """Return the left Jacobian satisfying ``omega = J_l rho_dot``."""
    rho = _vector3(rotation_vector, "rotation_vector")
    theta = float(np.linalg.norm(rho))
    rho_hat = hat(rho)
    if theta < _SMALL_ANGLE:
        theta2 = theta * theta
        a = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
        b = 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0
    else:
        a = (1.0 - np.cos(theta)) / (theta * theta)
        b = (theta - np.sin(theta)) / (theta * theta * theta)
    return np.eye(3) + a * rho_hat + b * (rho_hat @ rho_hat)


def left_jacobian_inverse(rotation_vector: Iterable[float]) -> np.ndarray:
    """Return the inverse left Jacobian with a stable zero-angle series."""
    rho = _vector3(rotation_vector, "rotation_vector")
    theta = float(np.linalg.norm(rho))
    if theta >= np.pi - 1.0e-6:
        raise ValueError("rotation vector lies outside the supported chart.")
    rho_hat = hat(rho)
    if theta < _SMALL_ANGLE:
        theta2 = theta * theta
        coefficient = 1.0 / 12.0 + theta2 / 720.0
    else:
        coefficient = (
            1.0 / (theta * theta)
            - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        )
    return np.eye(3) - 0.5 * rho_hat + coefficient * (rho_hat @ rho_hat)


def coordinate_transform(rotation_vector: Iterable[float]) -> np.ndarray:
    """Return ``E = blkdiag(I, J_l)`` for geometric/analytical velocities."""
    transform = np.zeros((6, 6), dtype=float)
    transform[:3, :3] = np.eye(3)
    transform[3:, 3:] = left_jacobian(rotation_vector)
    return transform
