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

"""Six-dimensional pose-coordinate reference samples and trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _vector6(value: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (6,):
        raise ValueError(f"{name} must have shape (6,), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


@dataclass(frozen=True)
class PoseReferenceSample:
    """Desired ``[p, rho]`` and its first two analytical derivatives."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    def __post_init__(self) -> None:
        for name in ("position", "velocity", "acceleration"):
            object.__setattr__(self, name, _vector6(getattr(self, name), name))


def fixed_pose_reference(pose: Iterable[float]) -> PoseReferenceSample:
    """Return a constant six-dimensional pose-coordinate reference."""
    return PoseReferenceSample(
        position=_vector6(pose, "pose"),
        velocity=np.zeros(6),
        acceleration=np.zeros(6),
    )


def smooth_payload_reference(
    time_sec: float,
    center: Iterable[float],
    *,
    translation_amplitude: Iterable[float] = (0.025, 0.020, 0.018),
    rotation_amplitude: Iterable[float] = (0.10, 0.08, 0.12),
    frequency_hz: float = 0.12,
) -> PoseReferenceSample:
    """Generate a bounded 6D multi-sine trajectory for loaded tracking."""
    time_value = float(time_sec)
    if not np.isfinite(time_value) or time_value < 0.0:
        raise ValueError("time_sec must be finite and non-negative.")
    origin = _vector6(center, "center")
    translation = np.asarray(tuple(translation_amplitude), dtype=float)
    rotation = np.asarray(tuple(rotation_amplitude), dtype=float)
    if translation.shape != (3,) or rotation.shape != (3,):
        raise ValueError("translation_amplitude and rotation_amplitude must be 3D.")
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation)):
        raise ValueError("trajectory amplitudes must contain only finite values.")
    frequency = float(frequency_hz)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be finite and positive.")
    amplitude = np.concatenate((translation, rotation))
    harmonic = np.array((1.0, 1.0, 2.0, 1.0, 2.0, 1.5))
    phase = np.array((0.0, 0.5 * np.pi, 0.25 * np.pi, 0.0, 0.4, 0.8))
    omega = 2.0 * np.pi * frequency * harmonic
    angle = omega * time_value + phase
    return PoseReferenceSample(
        position=origin + amplitude * np.sin(angle),
        velocity=amplitude * omega * np.cos(angle),
        acceleration=-amplitude * omega * omega * np.sin(angle),
    )
