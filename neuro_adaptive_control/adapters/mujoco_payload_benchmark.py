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

"""Deterministic unknown-payload acquisition benchmark in full MuJoCo dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter_ns
from typing import Callable, Dict

import numpy as np

from neuro_adaptive_control.core.pose_impedance_model import (
    PoseImpedanceModel,
    PoseImpedanceParameters,
)
from neuro_adaptive_control.core.pose_neuro_adaptive_controller import (
    PoseNACParameters,
    PoseNeuroAdaptiveController,
)
from neuro_adaptive_control.core.pose_references import PoseReferenceSample
from neuro_adaptive_control.core.safety import SafetyConfig, SafetySupervisor
from neuro_adaptive_control.core.so3 import left_jacobian_inverse, log
from neuro_adaptive_control.core.two_layer_network import TwoLayerAdaptiveNetwork

from .model_based_controller import MujocoModelBasedController
from .mujoco_simulation import SimulationState
from .mujoco_ur5e_adapter import MujocoUR5ePlant
from .pose_wrench_to_torque import (
    PoseTorqueConfig,
    PoseWrenchToTorque,
)


class BenchmarkController(str, Enum):
    """Controller variants evaluated under the identical physical scenario."""

    ADAPTIVE_NAC = "adaptive_nac"
    FROZEN_AT_PAYLOAD = "frozen_at_payload"
    NOMINAL_MODEL_BASED = "nominal_model_based"
    ORACLE_MODEL_BASED = "oracle_model_based"


@dataclass(frozen=True)
class PayloadCase:
    """Ground-truth MuJoCo payload properties hidden from the NAC."""

    name: str
    mass_kg: float
    com_offset_m: tuple[float, float, float]
    inertia_scale: float
    seed: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("payload name must not be empty")
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ValueError("payload mass must be finite and positive")
        offset = np.asarray(self.com_offset_m, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("payload COM offset must be a finite 3D vector")
        if not np.isfinite(self.inertia_scale) or self.inertia_scale <= 0.0:
            raise ValueError("payload inertia scale must be finite and positive")
        if self.seed < 0:
            raise ValueError("payload seed must be non-negative")


DEFAULT_PAYLOAD_CASE = PayloadCase(
    name="showcase_310g_offset",
    mass_kg=0.31,
    com_offset_m=(0.004, -0.003, 0.002),
    inertia_scale=1.15,
    seed=41,
)

HELD_OUT_PAYLOAD_CASES = (
    PayloadCase(
        name="light_240g",
        mass_kg=0.24,
        com_offset_m=(-0.003, 0.002, 0.0),
        inertia_scale=0.90,
        seed=53,
    ),
    DEFAULT_PAYLOAD_CASE,
    PayloadCase(
        name="heavy_360g",
        mass_kg=0.36,
        com_offset_m=(0.005, 0.004, 0.003),
        inertia_scale=1.25,
        seed=67,
    ),
)


@dataclass(frozen=True)
class PayloadBenchmarkConfig:
    """One controller/payload combination at the common 500 Hz schedule."""

    controller: BenchmarkController = BenchmarkController.ADAPTIVE_NAC
    payload: PayloadCase = DEFAULT_PAYLOAD_CASE
    duration_sec: float = 13.5
    control_period_sec: float = 0.002
    maximum_gripper_effort_n: float = 3.0
    maximum_contact_force_n: float = 250.0

    def __post_init__(self) -> None:
        controller = BenchmarkController(self.controller)
        object.__setattr__(self, "controller", controller)
        for name in (
            "duration_sec",
            "control_period_sec",
            "maximum_gripper_effort_n",
            "maximum_contact_force_n",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if not np.isclose(self.control_period_sec, 0.002, atol=1.0e-15):
            raise ValueError("control period must be exactly 0.002 s")
        if not np.isclose(self.duration_sec, 13.5, atol=1.0e-12):
            raise ValueError("duration must preserve the canonical 13.5 s schedule")


@dataclass(frozen=True)
class PayloadBenchmarkResult:
    """Canonical histories used by metrics, plots, and deterministic rendering."""

    config: PayloadBenchmarkConfig
    time: np.ndarray
    phase: tuple[str, ...]
    desired_pose: np.ndarray
    impedance_pose: np.ndarray
    actual_pose: np.ndarray
    generalized_command: np.ndarray
    neural_estimate: np.ndarray
    arm_torque: np.ndarray
    weight_norm: np.ndarray
    bilateral_contact: np.ndarray
    payload_acquired: np.ndarray
    contact_force: np.ndarray
    object_position: np.ndarray
    qpos: np.ndarray
    metrics: Dict[str, float | int | bool | str]


@dataclass(frozen=True)
class PayloadSuiteResult:
    """Held-out trials and aggregate adaptation acceptance metrics."""

    trials: tuple[PayloadBenchmarkResult, ...]
    metrics: Dict[str, float | int | bool | str]


def _quintic_segment(
    time_sec: float,
    start_time: float,
    end_time: float,
    start: np.ndarray,
    end: np.ndarray,
) -> PoseReferenceSample:
    duration = end_time - start_time
    if time_sec <= start_time:
        return PoseReferenceSample(start, np.zeros(6), np.zeros(6))
    if time_sec >= end_time:
        return PoseReferenceSample(end, np.zeros(6), np.zeros(6))
    normalized = (time_sec - start_time) / duration
    position_scale = (
        10.0 * normalized**3
        - 15.0 * normalized**4
        + 6.0 * normalized**5
    )
    velocity_scale = (
        30.0 * normalized**2
        - 60.0 * normalized**3
        + 30.0 * normalized**4
    ) / duration
    acceleration_scale = (
        60.0 * normalized
        - 180.0 * normalized**2
        + 120.0 * normalized**3
    ) / duration**2
    delta = end - start
    return PoseReferenceSample(
        start + position_scale * delta,
        velocity_scale * delta,
        acceleration_scale * delta,
    )


def _loaded_trajectory(time_sec: float, center: np.ndarray) -> PoseReferenceSample:
    duration = 4.0
    normalized = (time_sec - 6.5) / duration
    base_frequency = 2.0 * np.pi / duration
    harmonic = np.array((1.0, 1.0, 2.0, 1.0, 2.0, 1.0))
    sign = np.array((1.0, -1.0, 1.0, 1.0, -1.0, 1.0))
    amplitude = np.array((0.018, 0.014, 0.012, 0.055, 0.045, 0.065))
    omega = base_frequency * harmonic
    angle = 2.0 * np.pi * normalized * harmonic
    position = center + 0.5 * sign * amplitude * (1.0 - np.cos(angle))
    velocity = 0.5 * sign * amplitude * omega * np.sin(angle)
    acceleration = 0.5 * sign * amplitude * omega**2 * np.cos(angle)
    return PoseReferenceSample(position, velocity, acceleration)


def payload_schedule(
    time_sec: float,
    initial_pose: np.ndarray,
) -> tuple[str, PoseReferenceSample, bool]:
    """Return phase, six-dimensional reference, and gripper-close request."""
    home = initial_pose.copy()
    waypoint = home + np.array((0.018, 0.014, 0.012, 0.05, -0.04, 0.06))
    # The free-space NAC initially carries the robot+gripper gravity mismatch.
    # This command places the measured pinch site at the object center without
    # driving the gripper-base collision mesh into the payload before closing.
    grasp = home + np.array((0.0, 0.0, -0.060, 0.0, 0.0, 0.0))
    lifted = grasp + np.array((0.0, 0.0, 0.080, 0.0, 0.0, 0.0))
    if time_sec < 1.0:
        return "unloaded_out", _quintic_segment(
            time_sec, 0.0, 1.0, home, waypoint
        ), False
    if time_sec < 2.0:
        return "unloaded_return", _quintic_segment(
            time_sec, 1.0, 2.0, waypoint, home
        ), False
    if time_sec < 4.0:
        return "approach", _quintic_segment(
            time_sec, 2.0, 4.0, home, grasp
        ), False
    if time_sec < 5.0:
        return "grasp", PoseReferenceSample(
            grasp, np.zeros(6), np.zeros(6)
        ), True
    if time_sec < 6.5:
        return "lift", _quintic_segment(
            time_sec, 5.0, 6.5, grasp, lifted
        ), True
    if time_sec < 10.5:
        return "loaded_tracking", _loaded_trajectory(time_sec, lifted), True
    if time_sec < 12.0:
        return "lower", _quintic_segment(
            time_sec, 10.5, 12.0, lifted, grasp
        ), True
    if time_sec < 12.5:
        return "release", PoseReferenceSample(
            grasp, np.zeros(6), np.zeros(6)
        ), False
    return "retreat", _quintic_segment(
        time_sec, 12.5, 13.5, grasp, home
    ), False


def build_pose_controller(
    initial_pose: np.ndarray,
    *,
    seed: int,
) -> PoseNeuroAdaptiveController:
    """Construct the 42D two-adaptive-layer NAC without dynamics access."""
    impedance = PoseImpedanceModel(
        PoseImpedanceParameters.diagonal(
            mass=(1.0, 1.0, 1.0, 0.18, 0.18, 0.18),
            damping=(30.0, 30.0, 34.0, 3.2, 3.2, 3.2),
            stiffness=(110.0, 110.0, 125.0, 12.0, 12.0, 12.0),
            external_gain=np.ones(6),
        ),
        initial_position=initial_pose,
    )
    input_scale = np.concatenate(
        (
            np.full(6, 2.5),
            np.full(6, 2.0),
            np.array((0.6, 0.6, 0.6, 0.3, 0.3, 0.3)),
            np.array((0.4, 0.4, 0.4, 0.3, 0.3, 0.3)),
            np.array((2.0, 2.0, 2.0, 1.5, 1.5, 1.5)),
            np.array((0.10, 0.10, 0.10, 0.20, 0.20, 0.20)),
            np.array((0.5, 0.5, 0.5, 0.6, 0.6, 0.6)),
        )
    )
    network = TwoLayerAdaptiveNetwork(
        input_dim=42,
        hidden_dim=28,
        output_dim=6,
        hidden_learning_rate=0.08,
        output_learning_rate=60.0,
        leakage=0.01,
        hidden_weight_limit=80.0,
        output_weight_limit=140.0,
        input_scale=input_scale,
        input_clip=4.0,
        initial_hidden_scale=0.20,
        seed=seed,
    )
    parameters = PoseNACParameters.diagonal(
        lambda_gain=(13.0, 13.0, 15.0, 10.0, 10.0, 10.0),
        feedback_gain=(135.0, 135.0, 180.0, 24.0, 24.0, 24.0),
        robust_gain=(0.08, 0.08, 0.10, 0.10, 0.10, 0.10),
        ideal_weight_bound=4.0,
    )
    safety = SafetySupervisor(
        SafetyConfig(
            command_limits=np.array((150.0, 150.0, 170.0, 38.0, 38.0, 38.0)),
            command_norm_limit=230.0,
            watchdog_timeout=0.05,
            maximum_dt=0.002,
            command_dimension=6,
        )
    )
    return PoseNeuroAdaptiveController(impedance, network, parameters, safety)


def build_pose_mapper() -> PoseWrenchToTorque:
    """Construct running torque limits and stopping-only joint damping."""
    return PoseWrenchToTorque(
        PoseTorqueConfig.diagonal(
            torque_limits=(140.0, 140.0, 140.0, 27.0, 27.0, 27.0),
            torque_rate_limits=(
                8000.0,
                8000.0,
                8000.0,
                3000.0,
                3000.0,
                3000.0,
            ),
            safe_joint_damping=(0.8, 0.8, 0.7, 0.20, 0.20, 0.15),
        )
    )


def _rmse_norm(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(error * error, axis=1))))


class MujocoPayloadBenchmarkRunner:
    """Own one plant and run one controller through the canonical schedule."""

    def __init__(
        self,
        config: PayloadBenchmarkConfig,
        *,
        model_path: str | Path | None = None,
    ) -> None:
        self.config = config
        payload = config.payload
        self.plant = MujocoUR5ePlant(
            model_path=model_path,
            seed=payload.seed,
            payload_mass_kg=payload.mass_kg,
            payload_com_offset_m=payload.com_offset_m,
            payload_inertia_scale=payload.inertia_scale,
        )
        initial = self.plant.kinematic_state()
        self.reference_rotation = initial.tcp_rotation.copy()
        self.initial_pose = np.concatenate((initial.tcp_position, np.zeros(3)))
        self.pose_controller = build_pose_controller(
            self.initial_pose,
            seed=payload.seed,
        )
        self.pose_mapper = build_pose_mapper()
        oracle_mass = (
            payload.mass_kg
            if config.controller == BenchmarkController.ORACLE_MODEL_BASED
            else None
        )
        self.model_controller = MujocoModelBasedController(
            model_path=model_path,
            oracle_payload_mass_kg=oracle_mass
        )
        self.state = SimulationState.START
        self.reason = "created"

    def _actual_pose(self, sample) -> tuple[np.ndarray, np.ndarray]:
        rho = log(sample.tcp_rotation @ self.reference_rotation.T)
        rho_dot = left_jacobian_inverse(rho) @ sample.tcp_angular_velocity
        return (
            np.concatenate((sample.tcp_position, rho)),
            np.concatenate((sample.tcp_linear_velocity, rho_dot)),
        )

    def _safe_hold(self) -> None:
        self.pose_mapper.reset()
        velocity = self.plant.arm_velocity
        damping = np.zeros(6)
        if np.all(np.isfinite(velocity)):
            damping = self.pose_mapper.safe_stop_command(
                velocity,
                self.config.control_period_sec,
            )
        self.plant.apply_safe_hold(damping)

    def _validate(self, sample, contact, torque: np.ndarray) -> None:
        values = (
            sample.all_joint_position,
            sample.all_joint_velocity,
            sample.tcp_position,
            sample.tcp_rotation,
            sample.translational_jacobian,
            sample.rotational_jacobian,
            torque,
        )
        if not all(np.all(np.isfinite(value)) for value in values):
            raise FloatingPointError("benchmark state or command contains NaN/Inf")
        limits = self.plant.joint_limits
        tolerance = 6.0e-3
        if np.any(sample.all_joint_position < limits[:, 0] - tolerance) or np.any(
            sample.all_joint_position > limits[:, 1] + tolerance
        ):
            lower_error = limits[:, 0] - sample.all_joint_position
            upper_error = sample.all_joint_position - limits[:, 1]
            raise RuntimeError(
                f"benchmark joint-limit violation at t={sample.stamp_sec:.3f}s "
                f"(lower={float(np.max(lower_error)):.6g}, "
                f"upper={float(np.max(upper_error)):.6g})"
            )
        if np.max(np.abs(sample.arm_velocity)) > 4.0:
            raise RuntimeError("benchmark joint velocity limit exceeded")
        if np.max(np.abs(torque)) > 150.0:
            raise RuntimeError("benchmark torque hard limit exceeded")
        if contact.contact_force_norm_n > self.config.maximum_contact_force_n:
            raise RuntimeError("benchmark contact force limit exceeded")
        if contact.unexpected_contacts:
            raise RuntimeError(
                "unexpected robot-environment benchmark contact at "
                f"t={sample.stamp_sec:.3f}s "
                f"(count={contact.unexpected_contacts}, "
                f"tcp={sample.tcp_position.tolist()}, "
                f"object={sample.object_position.tolist()})"
            )

    def run(
        self,
        step_callback: Callable[[int, str], None] | None = None,
    ) -> PayloadBenchmarkResult:
        """Execute one deterministic controller/payload trial."""
        if self.state != SimulationState.START:
            raise RuntimeError("payload benchmark runner must be fresh")
        self.state = SimulationState.RUNNING
        dt = self.config.control_period_sec
        steps = int(round(self.config.duration_sec / dt))
        self.pose_controller.start(0.0)
        time_history = np.empty(steps)
        desired_history = np.empty((steps, 6))
        impedance_history = np.empty((steps, 6))
        actual_history = np.empty((steps, 6))
        command_history = np.zeros((steps, 6))
        neural_history = np.zeros((steps, 6))
        torque_history = np.empty((steps, 6))
        weight_history = np.zeros(steps)
        bilateral_history = np.zeros(steps, dtype=bool)
        acquired_history = np.zeros(steps, dtype=bool)
        contact_history = np.zeros(steps)
        object_history = np.empty((steps, 3))
        qpos_history = np.empty((steps, self.plant.model.nq))
        phases: list[str] = []
        payload_acquired = False
        acquisition_time = np.nan
        acquisition_height = float(self.plant.kinematic_state().object_position[2])
        saturation_count = 0
        wall_start = perf_counter_ns()
        for index in range(steps):
            stamp = index * dt
            sample = self.plant.kinematic_state()
            actual_pose, actual_velocity = self._actual_pose(sample)
            phase, reference, closing = payload_schedule(stamp, self.initial_pose)
            if closing:
                self.plant.gripper.close(self.config.maximum_gripper_effort_n)
            else:
                self.plant.gripper.open(self.config.maximum_gripper_effort_n)
            jacobian = np.vstack(
                (sample.translational_jacobian, sample.rotational_jacobian)
            )
            try:
                if self.config.controller in {
                    BenchmarkController.ADAPTIVE_NAC,
                    BenchmarkController.FROZEN_AT_PAYLOAD,
                }:
                    checkpoint = self.pose_controller.network.checkpoint()
                    output = self.pose_controller.step(
                        actual_pose,
                        actual_velocity,
                        sample.arm_position,
                        sample.arm_velocity,
                        reference,
                        np.zeros(6),
                        dt=dt,
                        now=stamp,
                    )
                    if output.state.value == "fault":
                        raise RuntimeError(output.fault_reason)
                    mapping = self.pose_mapper.map_running_command(
                        output.command,
                        actual_pose[3:],
                        jacobian,
                        dt,
                    )
                    if mapping.torque_saturated or mapping.rate_saturated:
                        self.pose_controller.network.restore(checkpoint)
                    torque = mapping.command
                    generalized = output.command
                    neural = output.neural_estimate
                    impedance = output.model_state.position
                    saturation_count += int(mapping.torque_saturated)
                else:
                    model_output = self.model_controller.command(
                        all_joint_position=sample.all_joint_position,
                        all_joint_velocity=sample.all_joint_velocity,
                        actual_pose=actual_pose,
                        actual_pose_velocity=actual_velocity,
                        reference=reference,
                        rotation_vector=actual_pose[3:],
                        geometric_jacobian=jacobian,
                        tcp_position=sample.tcp_position,
                        payload_position=sample.object_position,
                        payload_acquired=payload_acquired,
                        dt=dt,
                    )
                    torque = model_output.command
                    generalized = np.zeros(6)
                    neural = np.zeros(6)
                    impedance = reference.position
                contact = self.plant.contact_summary()
                self._validate(sample, contact, torque)
                next_sample = self.plant.advance(torque)
                next_contact = self.plant.contact_summary()
                self._validate(next_sample, next_contact, torque)
            except (FloatingPointError, RuntimeError, ValueError) as error:
                self.state = SimulationState.FAULT
                self.reason = str(error)
                self.pose_controller.safety.trigger_fault(self.reason)
                self._safe_hold()
                raise RuntimeError(self.reason) from error
            bilateral = bool(
                next_contact.left_finger_contacts > 0
                and next_contact.right_finger_contacts > 0
            )
            lifted = bool(
                next_sample.object_position[2] >= acquisition_height + 0.012
            )
            if not payload_acquired and bilateral and lifted:
                payload_acquired = True
                acquisition_time = next_sample.stamp_sec
                if self.config.controller == BenchmarkController.FROZEN_AT_PAYLOAD:
                    self.pose_controller.network.adaptation_enabled = False
            next_actual, _ = self._actual_pose(next_sample)
            next_reference = payload_schedule(
                next_sample.stamp_sec,
                self.initial_pose,
            )[1]
            time_history[index] = next_sample.stamp_sec
            desired_history[index] = next_reference.position
            impedance_history[index] = impedance
            actual_history[index] = next_actual
            command_history[index] = generalized
            neural_history[index] = neural
            torque_history[index] = torque
            weight_history[index] = self.pose_controller.network.combined_weight_norm
            bilateral_history[index] = bilateral
            acquired_history[index] = payload_acquired
            contact_history[index] = next_contact.contact_force_norm_n
            object_history[index] = next_sample.object_position
            qpos_history[index] = self.plant.data.qpos
            phases.append(phase)
            if step_callback is not None:
                step_callback(index, phase)
        wall_duration = (perf_counter_ns() - wall_start) * 1.0e-9
        self.state = SimulationState.STOPPING
        self._safe_hold()
        self.state = SimulationState.STOPPED
        self.reason = "benchmark complete; physics paused in safe hold"
        phase_array = np.asarray(phases)
        loaded_mask = (phase_array == "loaded_tracking") & acquired_history
        unloaded_mask = np.char.startswith(phase_array.astype(str), "unloaded")
        position_error = desired_history[:, :3] - actual_history[:, :3]
        orientation_error = desired_history[:, 3:] - actual_history[:, 3:]
        if not np.any(loaded_mask):
            raise RuntimeError("payload was not acquired before loaded tracking")
        loaded_position_error = position_error[loaded_mask]
        loaded_orientation_error = orientation_error[loaded_mask]
        loaded_bilateral_ratio = float(np.mean(bilateral_history[loaded_mask]))
        success = bool(
            payload_acquired
            and loaded_bilateral_ratio >= 0.90
            and np.max(contact_history) <= self.config.maximum_contact_force_n
            and self.state == SimulationState.STOPPED
        )
        metrics: Dict[str, float | int | bool | str] = {
            "controller": self.config.controller.value,
            "payload_case": self.config.payload.name,
            "success": success,
            "state": self.state.value,
            "payload_acquired": payload_acquired,
            "payload_acquisition_time_sec": float(acquisition_time),
            "loaded_position_rmse_m": _rmse_norm(loaded_position_error),
            "loaded_position_max_error_m": float(
                np.max(np.linalg.norm(loaded_position_error, axis=1))
            ),
            "loaded_orientation_rmse_rad": _rmse_norm(loaded_orientation_error),
            "loaded_orientation_max_error_rad": float(
                np.max(np.linalg.norm(loaded_orientation_error, axis=1))
            ),
            "unloaded_position_rmse_m": _rmse_norm(position_error[unloaded_mask]),
            "unloaded_orientation_rmse_rad": _rmse_norm(
                orientation_error[unloaded_mask]
            ),
            "loaded_bilateral_contact_ratio": loaded_bilateral_ratio,
            "maximum_contact_force_n": float(np.max(contact_history)),
            "maximum_arm_torque_abs_nm": float(np.max(np.abs(torque_history))),
            "maximum_neural_estimate_norm": float(
                np.max(np.linalg.norm(neural_history, axis=1))
            ),
            "final_combined_weight_norm": float(weight_history[-1]),
            "torque_saturation_count": saturation_count,
            "simulated_duration_sec": float(self.plant.data.time),
            "wall_duration_sec": wall_duration,
            "observed_step_rate_hz": float(steps / wall_duration),
            "fault_reason": "",
        }
        return PayloadBenchmarkResult(
            config=self.config,
            time=time_history,
            phase=tuple(phases),
            desired_pose=desired_history,
            impedance_pose=impedance_history,
            actual_pose=actual_history,
            generalized_command=command_history,
            neural_estimate=neural_history,
            arm_torque=torque_history,
            weight_norm=weight_history,
            bilateral_contact=bilateral_history,
            payload_acquired=acquired_history,
            contact_force=contact_history,
            object_position=object_history,
            qpos=qpos_history,
            metrics=metrics,
        )


def run_payload_benchmark(
    config: PayloadBenchmarkConfig = PayloadBenchmarkConfig(),
) -> PayloadBenchmarkResult:
    """Run one public deterministic unknown-payload trial."""
    return MujocoPayloadBenchmarkRunner(config).run()


def evaluate_adaptation_advantage(
    trials: tuple[PayloadBenchmarkResult, ...],
) -> Dict[str, float | int | bool | str]:
    """Evaluate the predeclared adaptive-versus-frozen acceptance gates."""
    grouped: dict[str, dict[BenchmarkController, PayloadBenchmarkResult]] = {}
    for trial in trials:
        grouped.setdefault(trial.config.payload.name, {})[
            trial.config.controller
        ] = trial
    paired = []
    for payload_name, controllers in grouped.items():
        if {
            BenchmarkController.ADAPTIVE_NAC,
            BenchmarkController.FROZEN_AT_PAYLOAD,
        }.issubset(controllers):
            paired.append(
                (
                    payload_name,
                    controllers[BenchmarkController.ADAPTIVE_NAC],
                    controllers[BenchmarkController.FROZEN_AT_PAYLOAD],
                )
            )
    if not paired:
        raise ValueError("at least one adaptive/frozen payload pair is required")
    adaptive_position = np.array(
        [pair[1].metrics["loaded_position_rmse_m"] for pair in paired],
        dtype=float,
    )
    frozen_position = np.array(
        [pair[2].metrics["loaded_position_rmse_m"] for pair in paired],
        dtype=float,
    )
    adaptive_orientation = np.array(
        [pair[1].metrics["loaded_orientation_rmse_rad"] for pair in paired],
        dtype=float,
    )
    frozen_orientation = np.array(
        [pair[2].metrics["loaded_orientation_rmse_rad"] for pair in paired],
        dtype=float,
    )
    adaptive_success = np.array(
        [bool(pair[1].metrics["success"]) for pair in paired]
    )
    frozen_success = np.array(
        [bool(pair[2].metrics["success"]) for pair in paired]
    )
    median_adaptive_position = float(np.median(adaptive_position))
    median_frozen_position = float(np.median(frozen_position))
    median_adaptive_orientation = float(np.median(adaptive_orientation))
    median_frozen_orientation = float(np.median(frozen_orientation))
    position_improvement = 1.0 - (
        median_adaptive_position / median_frozen_position
    )
    orientation_improvement = 1.0 - (
        median_adaptive_orientation / median_frozen_orientation
    )
    adaptive_completion = float(np.mean(adaptive_success))
    frozen_completion = float(np.mean(frozen_success))
    no_extra_safety_failures = bool(np.all(adaptive_success | ~frozen_success))
    gate_passed = bool(
        adaptive_completion >= 0.95
        and adaptive_completion >= frozen_completion
        and position_improvement >= 0.10
        and orientation_improvement >= 0.10
        and no_extra_safety_failures
    )
    return {
        "payload_pair_count": len(paired),
        "adaptive_completion_ratio": adaptive_completion,
        "frozen_completion_ratio": frozen_completion,
        "median_adaptive_loaded_position_rmse_m": median_adaptive_position,
        "median_frozen_loaded_position_rmse_m": median_frozen_position,
        "median_loaded_position_improvement_ratio": position_improvement,
        "median_adaptive_loaded_orientation_rmse_rad": (
            median_adaptive_orientation
        ),
        "median_frozen_loaded_orientation_rmse_rad": median_frozen_orientation,
        "median_loaded_orientation_improvement_ratio": orientation_improvement,
        "no_extra_adaptive_safety_failures": no_extra_safety_failures,
        "adaptation_advantage_gate_passed": gate_passed,
    }


def run_payload_suite(
    *,
    include_model_based_showcase: bool = True,
) -> PayloadSuiteResult:
    """Run held-out adaptive/frozen pairs and optional showcase baselines."""
    trials: list[PayloadBenchmarkResult] = []
    for payload in HELD_OUT_PAYLOAD_CASES:
        for controller in (
            BenchmarkController.ADAPTIVE_NAC,
            BenchmarkController.FROZEN_AT_PAYLOAD,
        ):
            trials.append(
                run_payload_benchmark(
                    PayloadBenchmarkConfig(
                        controller=controller,
                        payload=payload,
                    )
                )
            )
    if include_model_based_showcase:
        for controller in (
            BenchmarkController.NOMINAL_MODEL_BASED,
            BenchmarkController.ORACLE_MODEL_BASED,
        ):
            trials.append(
                run_payload_benchmark(
                    PayloadBenchmarkConfig(
                        controller=controller,
                        payload=DEFAULT_PAYLOAD_CASE,
                    )
                )
            )
    return PayloadSuiteResult(
        trials=tuple(trials),
        metrics=evaluate_adaptation_advantage(tuple(trials)),
    )
