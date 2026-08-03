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

"""Pure NumPy neuro-adaptive Cartesian wrench controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .impedance_model import CartesianImpedanceModel, ImpedanceState
from .rbf_network import RBFNetwork
from .references import ReferenceSample
from .safety import ControllerState, SafetySupervisor


def _matrix3(value: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (3,):
        array = np.diag(array)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def _vector3(value: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    return array.copy()


@dataclass(frozen=True)
class NACParameters:
    """Feedback and robustifying gains for the NAC command."""

    lambda_gain: np.ndarray
    feedback_gain: np.ndarray
    robust_gain: np.ndarray
    robust_bias: float

    def __post_init__(self) -> None:
        for name in ("lambda_gain", "feedback_gain", "robust_gain"):
            matrix = _matrix3(getattr(self, name), name)
            if not np.allclose(matrix, matrix.T, atol=1e-12):
                raise ValueError(f"{name} must be symmetric.")
            if np.min(np.linalg.eigvalsh(matrix)) < 0.0:
                raise ValueError(f"{name} must be positive semidefinite.")
            object.__setattr__(self, name, matrix)
        try:
            robust_bias = float(self.robust_bias)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("robust_bias must be numeric.") from error
        if not np.isfinite(robust_bias) or robust_bias < 0.0:
            raise ValueError("robust_bias must be finite and non-negative.")
        object.__setattr__(self, "robust_bias", robust_bias)

    @classmethod
    def diagonal(
        cls,
        lambda_gain: Iterable[float],
        feedback_gain: Iterable[float],
        robust_gain: Iterable[float],
        robust_bias: float,
    ) -> "NACParameters":
        """Construct independent-axis gains."""
        return cls(
            np.asarray(tuple(lambda_gain), dtype=float),
            np.asarray(tuple(feedback_gain), dtype=float),
            np.asarray(tuple(robust_gain), dtype=float),
            robust_bias,
        )


@dataclass(frozen=True)
class ControllerOutput:
    """All terms needed for ROS publication and deterministic analysis."""

    command: np.ndarray
    raw_command: np.ndarray
    neural_estimate: np.ndarray
    feedback_term: np.ndarray
    robust_term: np.ndarray
    external_term: np.ndarray
    model_state: ImpedanceState
    model_error: np.ndarray
    model_error_velocity: np.ndarray
    sliding_error: np.ndarray
    rbf_features: np.ndarray
    state: ControllerState
    saturated: bool
    fault_reason: str


class NeuroAdaptiveController:
    """
    Coordinate impedance integration, RBF adaptation, and safety.

    The default dynamics context is ``[x, xdot]``. A robot adapter may supply
    a different fixed-size feature vector (for example ``[q, qdot]``), while
    the core remains independent of forward kinematics and ROS.
    """

    def __init__(
        self,
        impedance_model: CartesianImpedanceModel,
        network: RBFNetwork,
        parameters: NACParameters,
        safety: SafetySupervisor,
        *,
        dynamics_feature_dim: int = 6,
    ) -> None:
        if dynamics_feature_dim <= 0:
            raise ValueError("dynamics_feature_dim must be positive.")
        expected_input = int(dynamics_feature_dim) + 15
        if network.input_dim != expected_input:
            raise ValueError(
                f"RBF input_dim must be {expected_input} for the selected "
                f"dynamics feature size, got {network.input_dim}."
            )
        if network.output_dim != 3:
            raise ValueError("RBF output_dim must be 3 for v0.1 translation.")
        self.impedance_model = impedance_model
        self.network = network
        self.parameters = parameters
        self.safety = safety
        self.dynamics_feature_dim = int(dynamics_feature_dim)

    @property
    def state(self) -> ControllerState:
        return self.safety.state

    def start(self, now: float = 0.0) -> None:
        """Start command production."""
        self.safety.start(now)

    def stop(self, reason: str = "stop requested") -> None:
        """Request a zero-command transition through stopping."""
        self.safety.request_stop(reason)

    def reset(
        self,
        model_position: Iterable[float] = (0.0, 0.0, 0.0),
        model_velocity: Iterable[float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Deterministically reset model, RBF weights, and safety latches."""
        self.impedance_model.reset(model_position, model_velocity)
        self.network.reset()
        self.safety.reset()

    def _zero_output(self) -> ControllerOutput:
        zeros = np.zeros(3, dtype=float)
        return ControllerOutput(
            command=zeros.copy(),
            raw_command=zeros.copy(),
            neural_estimate=zeros.copy(),
            feedback_term=zeros.copy(),
            robust_term=zeros.copy(),
            external_term=zeros.copy(),
            model_state=self.impedance_model.state,
            model_error=zeros.copy(),
            model_error_velocity=zeros.copy(),
            sliding_error=zeros.copy(),
            rbf_features=np.zeros(self.network.input_dim),
            state=self.safety.state,
            saturated=False,
            fault_reason=self.safety.reason,
        )

    def step(
        self,
        actual_position: Iterable[float],
        actual_velocity: Iterable[float],
        reference: ReferenceSample,
        external_wrench: Iterable[float],
        *,
        dt: float,
        now: float,
        dynamics_features: Optional[Iterable[float]] = None,
    ) -> ControllerOutput:
        """Compute and safely filter one Cartesian wrench sample."""
        try:
            position = _vector3(actual_position, "actual_position")
            velocity = _vector3(actual_velocity, "actual_velocity")
            wrench = _vector3(external_wrench, "external_wrench")
            ref_position = _vector3(reference.position, "reference.position")
            ref_velocity = _vector3(reference.velocity, "reference.velocity")
            ref_acceleration = _vector3(
                reference.acceleration, "reference.acceleration"
            )
        except (AttributeError, TypeError, ValueError) as error:
            self.safety.trigger_fault(str(error))
            return self._zero_output()

        if not self.safety.note_measurement(
            now,
            position,
            velocity,
            wrench,
            ref_position,
            ref_velocity,
            ref_acceleration,
        ) or not self.safety.validate_dt(dt):
            return self._zero_output()
        if self.safety.state != ControllerState.RUNNING:
            if self.safety.state == ControllerState.STOPPING:
                self.safety.filter_command(np.zeros(3, dtype=float), now)
            else:
                self.safety.tick(now)
            return self._zero_output()

        try:
            model_state = self.impedance_model.step(
                ref_position,
                ref_velocity,
                ref_acceleration,
                wrench,
                dt,
            )
            error = model_state.position - position
            error_velocity = model_state.velocity - velocity
            sliding_error = (
                error_velocity + self.parameters.lambda_gain @ error
            )
            if dynamics_features is None:
                context = np.concatenate((position, velocity))
            else:
                context = np.asarray(dynamics_features, dtype=float)
            if context.shape != (self.dynamics_feature_dim,):
                raise ValueError(
                    "dynamics_features must have shape "
                    f"({self.dynamics_feature_dim},), got {context.shape}."
                )
            features = np.concatenate(
                (
                    context,
                    model_state.position,
                    model_state.velocity,
                    model_state.acceleration,
                    error,
                    error_velocity,
                )
            )
            neural_estimate = self.network.forward(features)
            feedback_term = self.parameters.feedback_gain @ sliding_error
            robust_term = (
                self.network.weight_norm + self.parameters.robust_bias
            ) * (self.parameters.robust_gain @ sliding_error)
            external_term = -(
                self.impedance_model.parameters.external_gain @ wrench
            )
            raw_command = (
                neural_estimate
                + feedback_term
                + robust_term
                + external_term
            )
            command = self.safety.filter_command(raw_command, now)
            if self.safety.state == ControllerState.RUNNING:
                self.network.update(features, sliding_error, dt)
        except (FloatingPointError, TypeError, ValueError) as error:
            self.safety.trigger_fault(str(error))
            return self._zero_output()

        return ControllerOutput(
            command=command.copy(),
            raw_command=raw_command.copy(),
            neural_estimate=neural_estimate.copy(),
            feedback_term=feedback_term.copy(),
            robust_term=robust_term.copy(),
            external_term=external_term.copy(),
            model_state=model_state,
            model_error=error.copy(),
            model_error_velocity=error_velocity.copy(),
            sliding_error=sliding_error.copy(),
            rbf_features=features.copy(),
            state=self.safety.state,
            saturated=self.safety.last_saturated,
            fault_reason=self.safety.reason,
        )
