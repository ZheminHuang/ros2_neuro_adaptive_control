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

"""Robot adapter contracts; v0.1 intentionally ships no hardware adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class CartesianState:
    """
    Hold estimated translational state and optional dynamics features.

    A robot adapter is responsible for state estimation, forward kinematics,
    and Jacobian calculations. Those requirements are not removed by the
    controller's model-free treatment of ``M/C/F/G`` dynamics.
    """

    position: np.ndarray
    velocity: np.ndarray
    stamp_sec: float
    dynamics_features: np.ndarray


@runtime_checkable
class CartesianStateProvider(Protocol):
    """Supply coherent Cartesian state samples to the pure controller."""

    def read_cartesian_state(self) -> CartesianState:
        """Return the newest state estimate in one documented frame."""


@runtime_checkable
class WrenchCommandSink(Protocol):
    """Consume 3D force commands and provide an explicit zero path."""

    def send_force_command(self, force_xyz: np.ndarray, stamp_sec: float) -> None:
        """Send one finite, already-saturated translational command."""

    def send_zero_command(self, stamp_sec: float) -> None:
        """Send a zero force when stopping, faulted, or stale."""
