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

"""Pure command/state conversion for the dynamic Robotiq 2F-85 model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GripperLimits:
    """Physical API bounds independent of the MuJoCo actuator encoding."""

    maximum_opening_m: float = 0.085
    maximum_effort_n: float = 5.0
    driver_range_rad: float = 0.8
    actuator_maximum: float = 255.0

    def __post_init__(self) -> None:
        for name in (
            "maximum_opening_m",
            "maximum_effort_n",
            "driver_range_rad",
            "actuator_maximum",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class GripperState:
    """Normalized gripper feedback used by ROS actions and diagnostics."""

    opening_m: float
    effort_n: float
    target_opening_m: float
    left_contacts: int
    right_contacts: int
    contact_force_n: float
    reached_goal: bool
    stalled: bool
    stopped: bool


class RobotiqGripperAdapter:
    """Map metric opening goals to the Menagerie 0..255 tendon actuator."""

    def __init__(self, limits: GripperLimits = GripperLimits()) -> None:
        self.limits = limits
        self.reset()

    @property
    def target_opening_m(self) -> float:
        """Return the current clamped metric target."""
        return self._target_opening_m

    @property
    def maximum_effort_n(self) -> float:
        """Return the current clamped effort request."""
        return self._maximum_effort_n

    @property
    def stopped(self) -> bool:
        """Report whether actuator motion is explicitly stopped."""
        return self._stopped

    def reset(self) -> None:
        """Deterministically restore the default-open state."""
        self._target_opening_m = self.limits.maximum_opening_m
        self._maximum_effort_n = self.limits.maximum_effort_n
        self._stopped = False

    def command(self, opening_m: float, maximum_effort_n: float) -> None:
        """Accept one finite metric position/effort goal."""
        opening = float(opening_m)
        effort = float(maximum_effort_n)
        if not np.isfinite(opening) or not np.isfinite(effort):
            raise ValueError("gripper position and effort must be finite.")
        if effort <= 0.0:
            effort = self.limits.maximum_effort_n
        self._target_opening_m = float(
            np.clip(opening, 0.0, self.limits.maximum_opening_m)
        )
        self._maximum_effort_n = float(
            np.clip(effort, 0.0, self.limits.maximum_effort_n)
        )
        self._stopped = False

    def open(self, maximum_effort_n: float | None = None) -> None:
        """Request the maximum opening."""
        effort = self.limits.maximum_effort_n
        if maximum_effort_n is not None:
            effort = maximum_effort_n
        self.command(self.limits.maximum_opening_m, effort)

    def close(self, maximum_effort_n: float | None = None) -> None:
        """Request a fully closed actuator target."""
        effort = self.limits.maximum_effort_n
        if maximum_effort_n is not None:
            effort = maximum_effort_n
        self.command(0.0, effort)

    def stop(self, measured_opening_m: float) -> None:
        """Hold the measured opening and stop position-goal progression."""
        opening = float(measured_opening_m)
        if not np.isfinite(opening):
            raise ValueError("measured opening must be finite.")
        self._target_opening_m = float(
            np.clip(opening, 0.0, self.limits.maximum_opening_m)
        )
        self._stopped = True

    def actuator_control(self) -> float:
        """Return the 0..255 Menagerie position-actuator command."""
        fraction_closed = 1.0 - (
            self._target_opening_m / self.limits.maximum_opening_m
        )
        return float(self.limits.actuator_maximum * fraction_closed)

    def opening_from_driver_positions(
        self, right_driver_rad: float, left_driver_rad: float
    ) -> float:
        """Estimate physical opening from the coupled driver coordinates."""
        drivers = np.asarray((right_driver_rad, left_driver_rad), dtype=float)
        if not np.all(np.isfinite(drivers)):
            raise ValueError("driver positions must be finite.")
        mean_driver = float(np.mean(drivers))
        fraction_open = 1.0 - mean_driver / self.limits.driver_range_rad
        return float(
            self.limits.maximum_opening_m
            * np.clip(fraction_open, 0.0, 1.0)
        )

    def feedback(
        self,
        *,
        right_driver_rad: float,
        left_driver_rad: float,
        actuator_force_n: float,
        left_contacts: int,
        right_contacts: int,
        contact_force_n: float,
        position_tolerance_m: float = 0.001,
        stall_tolerance_m: float = 0.003,
    ) -> GripperState:
        """Build deterministic action feedback from physical state/contact."""
        for name, tolerance in (
            ("position_tolerance_m", position_tolerance_m),
            ("stall_tolerance_m", stall_tolerance_m),
        ):
            value = float(tolerance)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        opening = self.opening_from_driver_positions(
            right_driver_rad, left_driver_rad
        )
        effort = abs(float(actuator_force_n))
        force = float(contact_force_n)
        if not np.isfinite(effort) or not np.isfinite(force) or force < 0.0:
            raise ValueError("gripper feedback forces must be finite and valid.")
        error = abs(opening - self._target_opening_m)
        bilateral = int(left_contacts) > 0 and int(right_contacts) > 0
        stalled = bool(
            bilateral
            and error > stall_tolerance_m
            and effort >= 0.8 * self._maximum_effort_n
        )
        return GripperState(
            opening_m=opening,
            effort_n=effort,
            target_opening_m=self._target_opening_m,
            left_contacts=max(0, int(left_contacts)),
            right_contacts=max(0, int(right_contacts)),
            contact_force_n=force,
            reached_goal=bool(error <= position_tolerance_m),
            stalled=stalled,
            stopped=self._stopped,
        )
