"""Controller state machine and command safety filters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np


class ControllerState(str, Enum):
    """Required lifecycle states for a wrench-producing controller."""

    START = "start"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAULT = "fault"


@dataclass(frozen=True)
class SafetyConfig:
    """Static saturation, timing, and integration limits."""

    command_limits: np.ndarray
    command_norm_limit: float
    watchdog_timeout: float
    maximum_dt: float = 0.02

    def __post_init__(self) -> None:
        limits = np.asarray(self.command_limits, dtype=float)
        if limits.shape != (3,):
            raise ValueError("command_limits must have shape (3,).")
        if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
            raise ValueError("command_limits must be finite and positive.")
        object.__setattr__(self, "command_limits", limits.copy())
        for name in ("command_norm_limit", "watchdog_timeout", "maximum_dt"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name} must be numeric.") from error
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)


class SafetySupervisor:
    """Deterministic five-state safety supervisor."""

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Clear latches and return to the non-commanding start state."""
        self.state = ControllerState.START
        self.reason = "reset"
        self.last_measurement_time: float | None = None
        self.start_time: float | None = None
        self.last_observed_time: float | None = None
        self.last_command = np.zeros(3, dtype=float)
        self.last_saturated = False

    def start(self, now: float) -> None:
        """Enter running unless a fault still needs an explicit reset."""
        if self.state == ControllerState.FAULT:
            raise RuntimeError("reset is required before restarting from fault.")
        if self.state not in {ControllerState.START, ControllerState.STOPPED}:
            raise RuntimeError(f"cannot start from state {self.state.value}.")
        try:
            time_sec = self._observe_time(now)
        except (TypeError, ValueError, OverflowError) as error:
            self.trigger_fault(str(error))
            return
        self.state = ControllerState.RUNNING
        self.reason = "started"
        self.start_time = time_sec
        self.last_measurement_time = None
        self.last_command.fill(0.0)

    def request_stop(self, reason: str = "stop requested") -> None:
        """Enter stopping; the next filter cycle emits zero and stops."""
        if self.state == ControllerState.RUNNING:
            self.state = ControllerState.STOPPING
            self.reason = str(reason)
        elif self.state in {ControllerState.START, ControllerState.STOPPED}:
            self.state = ControllerState.STOPPED
            self.reason = str(reason)

    def complete_stop(self) -> None:
        """Latch the stopped state after a zero command was issued."""
        if self.state == ControllerState.STOPPING:
            self.state = ControllerState.STOPPED
            self.last_command.fill(0.0)

    def trigger_fault(self, reason: str) -> None:
        """Latch a fault and force subsequent commands to zero."""
        if self.state != ControllerState.FAULT:
            self.reason = str(reason)
        self.state = ControllerState.FAULT
        self.last_command.fill(0.0)
        self.last_saturated = False

    def note_measurement(self, now: float, *values: Iterable[float]) -> bool:
        """Validate one coherent state sample and refresh the watchdog."""
        try:
            time_sec = self._observe_time(now)
        except (TypeError, ValueError, OverflowError) as error:
            self.trigger_fault(str(error))
            return False
        for value in values:
            try:
                array = np.asarray(value, dtype=float)
            except (TypeError, ValueError, OverflowError):
                self.trigger_fault("state or wrench sample is not numeric")
                return False
            if not np.all(np.isfinite(array)):
                self.trigger_fault("state or wrench sample contains NaN or Inf")
                return False
        self.last_measurement_time = time_sec
        return True

    def validate_dt(self, dt: float) -> bool:
        """Reject invalid or unexpectedly large integration intervals."""
        try:
            value = float(dt)
        except (TypeError, ValueError, OverflowError):
            self.trigger_fault("dt must be a finite numeric value")
            return False
        if (
            not np.isfinite(value)
            or value <= 0.0
            or value > self.config.maximum_dt
        ):
            self.trigger_fault(
                f"dt {value!r} is outside (0, {self.config.maximum_dt}]"
            )
            return False
        return True

    def tick(self, now: float) -> ControllerState:
        """Advance watchdog logic without implying command publication."""
        try:
            time_sec = self._observe_time(now)
        except (TypeError, ValueError, OverflowError) as error:
            self.trigger_fault(str(error))
            return self.state
        if self.state == ControllerState.RUNNING:
            origin = (
                self.last_measurement_time
                if self.last_measurement_time is not None
                else self.start_time
            )
            if origin is not None and time_sec - origin > self.config.watchdog_timeout:
                self.trigger_fault("state watchdog expired")
        return self.state

    def filter_command(self, command: Iterable[float], now: float) -> np.ndarray:
        """Apply lifecycle gating, finite checks, and two-stage saturation."""
        self.tick(now)
        if self.state == ControllerState.STOPPING:
            self.last_command = np.zeros(3, dtype=float)
            self.last_saturated = False
            self.complete_stop()
            return self.last_command.copy()
        if self.state != ControllerState.RUNNING:
            self.last_command = np.zeros(3, dtype=float)
            self.last_saturated = False
            return self.last_command.copy()

        try:
            array = np.asarray(command, dtype=float)
        except (TypeError, ValueError, OverflowError):
            self.trigger_fault("command is not numeric")
            return np.zeros(3, dtype=float)
        if array.shape != (3,):
            self.trigger_fault(f"command must have shape (3,), got {array.shape}")
            return np.zeros(3, dtype=float)
        if not np.all(np.isfinite(array)):
            self.trigger_fault("command contains NaN or Inf")
            return np.zeros(3, dtype=float)

        limited = np.clip(
            array, -self.config.command_limits, self.config.command_limits
        )
        norm = float(np.linalg.norm(limited))
        if norm > self.config.command_norm_limit:
            limited *= self.config.command_norm_limit / norm
        self.last_saturated = not np.allclose(limited, array, atol=0.0, rtol=0.0)
        self.last_command = limited.copy()
        return limited

    def _observe_time(self, now: float) -> float:
        """Validate monotonic use of one time base across safety calls."""
        time_sec = float(now)
        if not np.isfinite(time_sec) or time_sec < 0.0:
            raise ValueError("time must be finite and non-negative")
        if (
            self.last_observed_time is not None
            and time_sec + 1e-12 < self.last_observed_time
        ):
            raise ValueError("controller time moved backwards")
        self.last_observed_time = time_sec
        return time_sec
