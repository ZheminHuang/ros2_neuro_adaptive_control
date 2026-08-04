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

"""Pure NumPy six-DoF two-layer neuro-adaptive impedance controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .pose_impedance_model import PoseImpedanceModel, PoseImpedanceState
from .pose_references import PoseReferenceSample
from .safety import ControllerState, SafetySupervisor
from .two_layer_network import TwoLayerAdaptiveNetwork


POSE_NN_INPUT_DIM = 42


def _vector(value: Iterable[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def _matrix6(value: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (6,):
        array = np.diag(array)
    if array.shape != (6, 6):
        raise ValueError(f"{name} must have shape (6, 6), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(array, array.T, atol=1.0e-12):
        raise ValueError(f"{name} must be symmetric.")
    if np.min(np.linalg.eigvalsh(array)) < 0.0:
        raise ValueError(f"{name} must be positive semidefinite.")
    return array.copy()


def build_pose_nn_features(
    joint_position: Iterable[float],
    joint_velocity: Iterable[float],
    model_position: Iterable[float],
    model_velocity: Iterable[float],
    model_acceleration: Iterable[float],
    model_error: Iterable[float],
    model_error_velocity: Iterable[float],
) -> np.ndarray:
    """Build exact ``[q,qdot,xm,xmdot,xmddot,em,emdot]`` 42D ordering."""
    features = np.concatenate(
        (
            _vector(joint_position, 6, "joint_position"),
            _vector(joint_velocity, 6, "joint_velocity"),
            _vector(model_position, 6, "model_position"),
            _vector(model_velocity, 6, "model_velocity"),
            _vector(model_acceleration, 6, "model_acceleration"),
            _vector(model_error, 6, "model_error"),
            _vector(model_error_velocity, 6, "model_error_velocity"),
        )
    )
    if features.shape != (POSE_NN_INPUT_DIM,):
        raise AssertionError("the six-DoF NN feature contract must remain 42D")
    return features


@dataclass(frozen=True)
class PoseNACParameters:
    """Filtered-error, feedback, and robustification gains."""

    lambda_gain: np.ndarray
    feedback_gain: np.ndarray
    robust_gain: np.ndarray
    ideal_weight_bound: float

    def __post_init__(self) -> None:
        for name in ("lambda_gain", "feedback_gain", "robust_gain"):
            object.__setattr__(self, name, _matrix6(getattr(self, name), name))
        bound = float(self.ideal_weight_bound)
        if not np.isfinite(bound) or bound <= 0.0:
            raise ValueError("ideal_weight_bound must be finite and positive.")
        object.__setattr__(self, "ideal_weight_bound", bound)

    @classmethod
    def diagonal(
        cls,
        lambda_gain: Iterable[float],
        feedback_gain: Iterable[float],
        robust_gain: Iterable[float],
        ideal_weight_bound: float,
    ) -> "PoseNACParameters":
        """Construct independent translational and rotational channels."""
        return cls(
            np.asarray(tuple(lambda_gain), dtype=float),
            np.asarray(tuple(feedback_gain), dtype=float),
            np.asarray(tuple(robust_gain), dtype=float),
            ideal_weight_bound,
        )


@dataclass(frozen=True)
class PoseControllerOutput:
    """All analytical-force terms and states required for benchmark logging."""

    command: np.ndarray
    raw_command: np.ndarray
    neural_estimate: np.ndarray
    feedback_term: np.ndarray
    robust_term: np.ndarray
    external_term: np.ndarray
    model_state: PoseImpedanceState
    model_error: np.ndarray
    model_error_velocity: np.ndarray
    sliding_error: np.ndarray
    nn_features: np.ndarray
    state: ControllerState
    saturated: bool
    fault_reason: str


class PoseNeuroAdaptiveController:
    """Coordinate 6D impedance, two-layer V/W adaptation, and safety."""

    def __init__(
        self,
        impedance_model: PoseImpedanceModel,
        network: TwoLayerAdaptiveNetwork,
        parameters: PoseNACParameters,
        safety: SafetySupervisor,
    ) -> None:
        if network.input_dim != POSE_NN_INPUT_DIM or network.output_dim != 6:
            raise ValueError("six-DoF NAC requires a 42-input, 6-output network.")
        if safety.config.command_limits.shape != (6,):
            raise ValueError("six-DoF NAC requires six safety command limits.")
        self.impedance_model = impedance_model
        self.network = network
        self.parameters = parameters
        self.safety = safety

    @property
    def state(self) -> ControllerState:
        """Return the lifecycle state."""
        return self.safety.state

    def start(self, now: float = 0.0) -> None:
        """Start analytical-force command production."""
        self.safety.start(now)

    def stop(self, reason: str = "stop requested") -> None:
        """Request a zero-command stop transition."""
        self.safety.request_stop(reason)

    def reset(
        self,
        model_position: Iterable[float] = (0.0,) * 6,
        model_velocity: Iterable[float] = (0.0,) * 6,
    ) -> None:
        """Reset impedance state, both adaptive layers, and safety latches."""
        self.impedance_model.reset(model_position, model_velocity)
        self.network.reset()
        self.safety.reset()

    def _zero_output(self) -> PoseControllerOutput:
        zeros = np.zeros(6)
        return PoseControllerOutput(
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
            nn_features=np.zeros(POSE_NN_INPUT_DIM),
            state=self.safety.state,
            saturated=False,
            fault_reason=self.safety.reason,
        )

    def step(
        self,
        actual_position: Iterable[float],
        actual_velocity: Iterable[float],
        joint_position: Iterable[float],
        joint_velocity: Iterable[float],
        reference: PoseReferenceSample,
        generalized_external_wrench: Iterable[float],
        *,
        dt: float,
        now: float,
    ) -> PoseControllerOutput:
        """Compute one bounded generalized analytical-force command."""
        try:
            actual = _vector(actual_position, 6, "actual_position")
            velocity = _vector(actual_velocity, 6, "actual_velocity")
            joints = _vector(joint_position, 6, "joint_position")
            joint_rates = _vector(joint_velocity, 6, "joint_velocity")
            wrench = _vector(
                generalized_external_wrench,
                6,
                "generalized_external_wrench",
            )
            reference_position = _vector(reference.position, 6, "reference.position")
            reference_velocity = _vector(reference.velocity, 6, "reference.velocity")
            reference_acceleration = _vector(
                reference.acceleration, 6, "reference.acceleration"
            )
        except (AttributeError, TypeError, ValueError) as error:
            self.safety.trigger_fault(str(error))
            return self._zero_output()
        if not self.safety.note_measurement(
            now,
            actual,
            velocity,
            joints,
            joint_rates,
            wrench,
            reference_position,
            reference_velocity,
            reference_acceleration,
        ) or not self.safety.validate_dt(dt):
            return self._zero_output()
        if self.safety.state != ControllerState.RUNNING:
            if self.safety.state == ControllerState.STOPPING:
                self.safety.filter_command(np.zeros(6), now)
            else:
                self.safety.tick(now)
            return self._zero_output()
        try:
            model_state = self.impedance_model.step(
                reference_position,
                reference_velocity,
                reference_acceleration,
                wrench,
                dt,
            )
            error = model_state.position - actual
            error_velocity = model_state.velocity - velocity
            sliding_error = error_velocity + self.parameters.lambda_gain @ error
            features = build_pose_nn_features(
                joints,
                joint_rates,
                model_state.position,
                model_state.velocity,
                model_state.acceleration,
                error,
                error_velocity,
            )
            neural_estimate = self.network.forward(features)
            feedback_term = self.parameters.feedback_gain @ sliding_error
            robust_term = (
                self.network.combined_weight_norm
                + self.parameters.ideal_weight_bound
            ) * (self.parameters.robust_gain @ sliding_error)
            external_term = -wrench
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
        return PoseControllerOutput(
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
            nn_features=features.copy(),
            state=self.safety.state,
            saturated=self.safety.last_saturated,
            fault_reason=self.safety.reason,
        )
