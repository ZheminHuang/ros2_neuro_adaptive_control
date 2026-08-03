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

"""Analytic 3D translational reference trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


def _vector3(value: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def _time(value: float) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("time must be finite and non-negative.")
    return float(value)


@dataclass(frozen=True)
class ReferenceSample:
    """Position plus analytic first and second derivatives."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class ReferenceTrajectory(Protocol):
    """Interface shared by all analytic reference generators."""

    def evaluate(self, time_sec: float) -> ReferenceSample:
        """Evaluate a trajectory at non-negative time."""


class FixedPointReference:
    """Constant position reference."""

    def __init__(self, position: Iterable[float]) -> None:
        self.position = _vector3(position, "position")

    def evaluate(self, time_sec: float) -> ReferenceSample:
        _time(time_sec)
        return ReferenceSample(
            self.position.copy(), np.zeros(3), np.zeros(3)
        )


class CircleReference:
    """Circle in the XY plane around a geometric center."""

    def __init__(
        self,
        center: Iterable[float],
        radius: float,
        frequency: float,
        phase: float = 0.0,
    ) -> None:
        self.center = _vector3(center, "center")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be finite and positive.")
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("frequency must be finite and positive.")
        if not np.isfinite(phase):
            raise ValueError("phase must be finite.")
        self.radius = float(radius)
        self.omega = 2.0 * np.pi * float(frequency)
        self.phase = float(phase)

    def evaluate(self, time_sec: float) -> ReferenceSample:
        theta = self.omega * _time(time_sec) + self.phase
        sine = np.sin(theta)
        cosine = np.cos(theta)
        position = self.center + self.radius * np.array([cosine, sine, 0.0])
        velocity = self.radius * self.omega * np.array([-sine, cosine, 0.0])
        acceleration = -self.radius * self.omega**2 * np.array(
            [cosine, sine, 0.0]
        )
        return ReferenceSample(position, velocity, acceleration)


class LineReference:
    """Smooth periodic traversal of a finite line segment."""

    def __init__(
        self,
        center: Iterable[float],
        length: float,
        frequency: float,
        axis: Iterable[float] = (1.0, 0.0, 0.0),
    ) -> None:
        self.center = _vector3(center, "center")
        direction = _vector3(axis, "axis")
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("axis must be non-zero.")
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("length must be finite and positive.")
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("frequency must be finite and positive.")
        self.axis = direction / norm
        self.amplitude = 0.5 * float(length)
        self.omega = 2.0 * np.pi * float(frequency)

    def evaluate(self, time_sec: float) -> ReferenceSample:
        theta = self.omega * _time(time_sec)
        position = self.center + self.axis * self.amplitude * np.sin(theta)
        velocity = self.axis * self.amplitude * self.omega * np.cos(theta)
        acceleration = (
            -self.axis * self.amplitude * self.omega**2 * np.sin(theta)
        )
        return ReferenceSample(position, velocity, acceleration)


class FigureEightReference:
    """Planar Gerono figure-eight with analytic derivatives."""

    def __init__(
        self,
        center: Iterable[float],
        width: float,
        height: float,
        frequency: float,
    ) -> None:
        self.center = _vector3(center, "center")
        for name, value in (
            ("width", width),
            ("height", height),
            ("frequency", frequency),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        self.amplitude_x = 0.5 * float(width)
        self.amplitude_y = 0.5 * float(height)
        self.omega = 2.0 * np.pi * float(frequency)

    def evaluate(self, time_sec: float) -> ReferenceSample:
        theta = self.omega * _time(time_sec)
        position = self.center + np.array(
            [
                self.amplitude_x * np.sin(theta),
                self.amplitude_y * np.sin(2.0 * theta),
                0.0,
            ]
        )
        velocity = np.array(
            [
                self.amplitude_x * self.omega * np.cos(theta),
                2.0 * self.amplitude_y * self.omega * np.cos(2.0 * theta),
                0.0,
            ]
        )
        acceleration = np.array(
            [
                -self.amplitude_x * self.omega**2 * np.sin(theta),
                -4.0 * self.amplitude_y * self.omega**2 * np.sin(2.0 * theta),
                0.0,
            ]
        )
        return ReferenceSample(position, velocity, acceleration)


def make_reference(
    kind: str,
    *,
    center: Iterable[float] = (0.0, 0.0, 0.0),
    frequency: float = 0.2,
    radius: float = 0.08,
    line_length: float = 0.16,
    line_axis: Iterable[float] = (1.0, 0.0, 0.0),
    figure8_width: float = 0.16,
    figure8_height: float = 0.10,
) -> ReferenceTrajectory:
    """Build one of the public v0.1 reference trajectories."""
    normalized = kind.strip().lower().replace("-", "_")
    if normalized in {"fixed", "fixed_point", "point"}:
        return FixedPointReference(center)
    if normalized == "circle":
        return CircleReference(center, radius, frequency)
    if normalized == "line":
        return LineReference(center, line_length, frequency, line_axis)
    if normalized in {"figure8", "figure_8"}:
        return FigureEightReference(
            center, figure8_width, figure8_height, frequency
        )
    raise ValueError(
        "kind must be one of: circle, line, figure8, fixed_point."
    )
