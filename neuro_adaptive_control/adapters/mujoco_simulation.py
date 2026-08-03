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

"""Deterministic single-owner MuJoCo + NAC experiment loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import perf_counter_ns
from typing import Dict

import numpy as np

from neuro_adaptive_control.core.impedance_model import (
    CartesianImpedanceModel,
    ImpedanceParameters,
)
from neuro_adaptive_control.core.neuro_adaptive_controller import (
    NACParameters,
    NeuroAdaptiveController,
)
from neuro_adaptive_control.core.rbf_network import RBFNetwork
from neuro_adaptive_control.core.references import ReferenceTrajectory, make_reference
from neuro_adaptive_control.core.safety import SafetyConfig, SafetySupervisor

from .mujoco_ur5e_adapter import MujocoUR5ePlant
from .ur5e_wrench_to_torque import (
    TorqueMappingConfig,
    UR5eWrenchToTorque,
    orientation_distance,
)


class SimulationState(str, Enum):
    """Robot simulation lifecycle, including an explicit reset transition."""

    START = "start"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAULT = "fault"
    RESETTING = "resetting"


@dataclass(frozen=True)
class MujocoRunConfig:
    """Fixed scenario shared verbatim by adaptive and frozen-weight runs."""

    trajectory: str = "circle"
    duration_sec: float = 8.0
    control_period_sec: float = 0.002
    plant_substeps: int = 4
    adaptation_enabled: bool = True
    external_wrench_mode: str = "none"
    seed: int = 23
    frequency_hz: float = 0.10
    radius_m: float = 0.035
    line_length_m: float = 0.070
    figure8_width_m: float = 0.070
    figure8_height_m: float = 0.045

    def __post_init__(self) -> None:
        for name in (
            "duration_sec",
            "control_period_sec",
            "frequency_hz",
            "radius_m",
            "line_length_m",
            "figure8_width_m",
            "figure8_height_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if self.plant_substeps != 4:
            raise ValueError("plant_substeps must be four")
        if not np.isclose(self.control_period_sec, 0.002, atol=1e-15):
            raise ValueError("control_period_sec must be exactly 0.002 s")
        mode = self.external_wrench_mode.strip().lower()
        if mode not in {"none", "injected", "virtual_ft"}:
            raise ValueError(
                "external_wrench_mode must be none, injected, or virtual_ft"
            )
        object.__setattr__(self, "external_wrench_mode", mode)
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True)
class MujocoRunResult:
    """Histories and machine-readable acceptance metrics."""

    config: MujocoRunConfig
    time: np.ndarray
    desired: np.ndarray
    impedance: np.ndarray
    actual: np.ndarray
    command_force: np.ndarray
    neural_estimate: np.ndarray
    arm_torque: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    object_position: np.ndarray
    contact_force: np.ndarray
    metrics: Dict[str, float | int | bool | str]


def build_mujoco_controller(
    initial_position: np.ndarray,
    *,
    adaptation_enabled: bool,
    seed: int,
) -> NeuroAdaptiveController:
    """Construct the independent 27D robot-state RBF configuration."""
    impedance = CartesianImpedanceModel(
        ImpedanceParameters.diagonal(
            mass=(1.0, 1.0, 1.0),
            damping=(28.0, 28.0, 30.0),
            stiffness=(90.0, 90.0, 100.0),
            external_gain=(1.0, 1.0, 1.0),
        ),
        initial_position=initial_position,
    )
    input_scale = np.array(
        [
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            0.60,
            0.60,
            0.60,
            0.50,
            0.50,
            0.50,
            3.0,
            3.0,
            3.0,
            0.08,
            0.08,
            0.08,
            0.50,
            0.50,
            0.50,
        ],
        dtype=float,
    )
    network = RBFNetwork(
        input_dim=27,
        output_dim=3,
        num_basis=45,
        widths=3.5,
        input_scale=input_scale,
        feature_clip=3.0,
        learning_rate=18.0,
        leakage=0.01,
        weight_limit=120.0,
        seed=seed,
        adaptation_enabled=adaptation_enabled,
    )
    parameters = NACParameters.diagonal(
        lambda_gain=(14.0, 14.0, 16.0),
        feedback_gain=(75.0, 75.0, 90.0),
        robust_gain=(0.012, 0.012, 0.015),
        robust_bias=2.0,
    )
    safety = SafetySupervisor(
        SafetyConfig(
            command_limits=np.array((120.0, 120.0, 140.0)),
            command_norm_limit=180.0,
            watchdog_timeout=0.05,
            maximum_dt=0.002,
        )
    )
    return NeuroAdaptiveController(
        impedance,
        network,
        parameters,
        safety,
        dynamics_feature_dim=12,
    )


def build_torque_mapper() -> UR5eWrenchToTorque:
    """Build orientation hold, joint damping, and actuator limits."""
    return UR5eWrenchToTorque(
        TorqueMappingConfig.diagonal(
            orientation_stiffness=(45.0, 45.0, 35.0),
            orientation_damping=(6.0, 6.0, 5.0),
            joint_damping=(0.8, 0.8, 0.7, 0.20, 0.20, 0.15),
            torque_limits=(140.0, 140.0, 140.0, 27.0, 27.0, 27.0),
            torque_rate_limits=(8000.0, 8000.0, 8000.0, 3000.0, 3000.0, 3000.0),
            orientation_error_limit=np.deg2rad(35.0),
        )
    )


def _reference(
    config: MujocoRunConfig, initial_position: np.ndarray
) -> ReferenceTrajectory:
    center = initial_position.copy()
    if config.trajectory.strip().lower() == "circle":
        center[0] -= config.radius_m
    return make_reference(
        config.trajectory,
        center=center,
        frequency=config.frequency_hz,
        radius=config.radius_m,
        line_length=config.line_length_m,
        line_axis=(1.0, 0.0, 0.0),
        figure8_width=config.figure8_width_m,
        figure8_height=config.figure8_height_m,
    )


def _injected_force(time_sec: float) -> np.ndarray:
    omega = 2.0 * np.pi * 0.35
    return np.array(
        (
            1.0 * np.sin(omega * time_sec),
            0.7 * np.sin(0.83 * omega * time_sec + 0.4),
            0.5 * np.sin(1.13 * omega * time_sec - 0.2),
        ),
        dtype=float,
    )


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


class MujocoNACRunner:
    """Synchronous owner: state → NAC → torque → four MuJoCo steps."""

    def __init__(self, config: MujocoRunConfig) -> None:
        self.config = config
        self.state = SimulationState.START
        self.reason = "created"
        self.plant = MujocoUR5ePlant(seed=config.seed)
        self.controller: NeuroAdaptiveController
        self.mapper: UR5eWrenchToTorque
        self.reference: ReferenceTrajectory
        self.reset()

    def reset(self) -> None:
        """Restore plant, controller, mapper, stamps, and histories."""
        self.state = SimulationState.RESETTING
        initial = self.plant.reset()
        self.controller = build_mujoco_controller(
            initial.tcp_position,
            adaptation_enabled=self.config.adaptation_enabled,
            seed=self.config.seed,
        )
        self.mapper = build_torque_mapper()
        self.reference = _reference(self.config, initial.tcp_position)
        self.state = SimulationState.START
        self.reason = "reset complete"

    def _fault(self, reason: str) -> None:
        self.state = SimulationState.FAULT
        self.reason = str(reason)
        self._apply_safe_hold()
        self.controller.safety.trigger_fault(reason)

    def _apply_safe_hold(self) -> None:
        """Drop NAC history and replace it with bounded damping-only torque."""
        velocity = self.plant.arm_velocity
        self.mapper.reset()
        damping = np.zeros(6)
        if np.all(np.isfinite(velocity)):
            damping = self.mapper.damping_command(
                velocity, self.config.control_period_sec
            )
        self.plant.apply_safe_hold(damping)

    def _check_state(self, sample, contact, raw_torque: np.ndarray) -> None:
        if not (
            np.all(np.isfinite(sample.arm_position))
            and np.all(np.isfinite(sample.arm_velocity))
            and np.all(np.isfinite(sample.all_joint_position))
            and np.all(np.isfinite(sample.all_joint_velocity))
            and np.all(np.isfinite(sample.tcp_position))
            and np.all(np.isfinite(sample.tcp_rotation))
            and np.all(np.isfinite(sample.translational_jacobian))
            and np.all(np.isfinite(sample.rotational_jacobian))
            and np.all(np.isfinite(raw_torque))
        ):
            raise FloatingPointError(
                "robot state, FK, Jacobian, or torque contains NaN or Inf"
            )
        limits = self.plant.joint_limits
        # MuJoCo limits are soft constraints; tolerate the bounded release
        # transient of the closed-loop gripper, but fault larger excursions.
        tolerance = 5e-3
        if np.any(sample.all_joint_position < limits[:, 0] - tolerance) or np.any(
            sample.all_joint_position > limits[:, 1] + tolerance
        ):
            raise RuntimeError("arm or gripper joint-limit violation")
        if np.any(np.abs(sample.arm_velocity) > 3.5):
            raise RuntimeError("excessive joint velocity")
        if np.any(np.abs(raw_torque) > np.array((280, 280, 280, 54, 54, 54))):
            raise RuntimeError("raw joint torque exceeded hard fault limit")
        lower = np.array((-0.85, -0.25, 0.10))
        upper = np.array((0.70, 0.90, 1.25))
        if np.any(sample.tcp_position < lower) or np.any(sample.tcp_position > upper):
            raise RuntimeError("Cartesian workspace violation")
        angle = orientation_distance(
            sample.tcp_rotation, self.plant.desired_tcp_rotation
        )
        if angle > np.deg2rad(35.0):
            raise RuntimeError("orientation error exceeded hard limit")
        if contact.contact_force_norm_n > 250.0:
            raise RuntimeError("excessive contact force")
        if contact.unexpected_contacts:
            raise RuntimeError("unexpected robot-environment collision")
        gripper = self.plant.gripper_state(contact)
        if not np.all(
            np.isfinite(
                (
                    gripper.opening_m,
                    gripper.effort_n,
                    gripper.target_opening_m,
                    gripper.contact_force_n,
                    self.plant.gripper.actuator_control(),
                )
            )
        ):
            raise FloatingPointError("gripper actuator feedback is not finite")
        if abs(sample.all_joint_position[6] - sample.all_joint_position[10]) > 0.02:
            raise RuntimeError("gripper actuator/coupling failure")

    def run(self) -> MujocoRunResult:
        """Execute the configured fixed-step experiment without sleeping."""
        if self.state not in {SimulationState.START, SimulationState.STOPPED}:
            raise RuntimeError(f"cannot run from state {self.state.value}")
        if self.state == SimulationState.STOPPED:
            self.reset()
        self.state = SimulationState.RUNNING
        self.reason = "running"
        self.controller.start(0.0)
        dt = self.config.control_period_sec
        steps = int(round(self.config.duration_sec / dt))
        if not np.isclose(steps * dt, self.config.duration_sec, atol=1e-12):
            raise ValueError("duration must be an integer number of control periods")

        time_history = np.empty(steps)
        desired_history = np.empty((steps, 3))
        impedance_history = np.empty((steps, 3))
        actual_history = np.empty((steps, 3))
        command_history = np.empty((steps, 3))
        neural_history = np.empty((steps, 3))
        torque_history = np.empty((steps, 6))
        q_history = np.empty((steps, 6))
        qdot_history = np.empty((steps, 6))
        object_history = np.empty((steps, 3))
        contact_history = np.empty((steps, 6))
        nac_times: list[float] = []
        step_times: list[float] = []
        missed_deadlines = 0
        torque_saturation_count = 0
        rate_saturation_count = 0
        contact_max = 0.0
        penetration_max = 0.0
        wall_start = perf_counter_ns()

        for index in range(steps):
            iteration_start = perf_counter_ns()
            sample = self.plant.kinematic_state()
            expected_stamp = index * dt
            if sample.sequence_id != index or not np.isclose(
                sample.stamp_sec, expected_stamp, atol=2e-12
            ):
                self._fault("control stamp or sequence mismatch")
                raise RuntimeError(self.reason)
            contact = self.plant.contact_summary()
            if self.config.external_wrench_mode == "none":
                external = np.zeros(3)
                injected = np.zeros(3)
            elif self.config.external_wrench_mode == "injected":
                external = _injected_force(expected_stamp)
                injected = external.copy()
            else:
                external = contact.force_world.copy()
                injected = np.zeros(3)
            reference_sample = self.reference.evaluate(expected_stamp)
            weights_before = self.controller.network.weights.copy()
            nac_start = perf_counter_ns()
            try:
                output = self.controller.step(
                    sample.tcp_position,
                    sample.tcp_linear_velocity,
                    reference_sample,
                    external,
                    dt=dt,
                    now=expected_stamp,
                    dynamics_features=np.concatenate(
                        (sample.arm_position, sample.arm_velocity)
                    ),
                )
                nac_times.append((perf_counter_ns() - nac_start) * 1e-6)
                if output.state.value == "fault":
                    raise RuntimeError(output.fault_reason)
                mapping = self.mapper.map_command(
                    output.command,
                    sample.translational_jacobian,
                    sample.rotational_jacobian,
                    sample.tcp_rotation,
                    self.plant.desired_tcp_rotation,
                    sample.tcp_angular_velocity,
                    sample.arm_velocity,
                    dt,
                )
            except (FloatingPointError, RuntimeError, ValueError) as error:
                self._fault(str(error))
                raise RuntimeError(self.reason) from error
            if mapping.torque_saturated or mapping.rate_saturated:
                # Downstream-limited commands are not accepted adaptation data.
                self.controller.network.weights[:] = weights_before
            torque_saturation_count += int(mapping.torque_saturated)
            rate_saturation_count += int(mapping.rate_saturated)
            try:
                self._check_state(sample, contact, mapping.raw_command)
                step_start = perf_counter_ns()
                next_sample = self.plant.advance(
                    mapping.command,
                    injected_force_world=injected,
                    substeps=self.config.plant_substeps,
                )
                step_times.append((perf_counter_ns() - step_start) * 1e-6)
                next_contact = self.plant.contact_summary()
                self._check_state(
                    next_sample,
                    next_contact,
                    mapping.raw_command,
                )
            except (FloatingPointError, RuntimeError, ValueError) as error:
                self._fault(str(error))
                raise RuntimeError(self.reason) from error
            contact_max = max(contact_max, next_contact.contact_force_norm_n)
            penetration_max = max(
                penetration_max, next_contact.maximum_penetration_m
            )
            time_history[index] = next_sample.stamp_sec
            desired_history[index] = self.reference.evaluate(
                next_sample.stamp_sec
            ).position
            impedance_history[index] = output.model_state.position
            actual_history[index] = next_sample.tcp_position
            command_history[index] = output.command
            neural_history[index] = output.neural_estimate
            torque_history[index] = mapping.command
            q_history[index] = next_sample.arm_position
            qdot_history[index] = next_sample.arm_velocity
            object_history[index] = next_sample.object_position
            contact_history[index, :3] = next_contact.force_world
            contact_history[index, 3:] = next_contact.torque_world_at_tcp
            iteration_ms = (perf_counter_ns() - iteration_start) * 1e-6
            missed_deadlines += int(iteration_ms > dt * 1000.0)

        wall_duration = (perf_counter_ns() - wall_start) * 1e-9
        self.state = SimulationState.STOPPING
        self._apply_safe_hold()
        self.controller.stop("duration complete")
        self.state = SimulationState.STOPPED
        self.reason = "duration complete; physics paused in bounded safe hold"

        error = impedance_history - actual_history
        error_norm = np.linalg.norm(error, axis=1)
        torque_norm = np.linalg.norm(torque_history, axis=1)
        qvel_abs = np.abs(qdot_history)
        metrics: Dict[str, float | int | bool | str] = {
            "adaptation_enabled": self.config.adaptation_enabled,
            "trajectory": self.config.trajectory,
            "external_wrench_mode": self.config.external_wrench_mode,
            "state": self.state.value,
            "control_period_sec": dt,
            "mujoco_timestep_sec": float(self.plant.model.opt.timestep),
            "substeps_per_control": self.config.plant_substeps,
            "control_steps": steps,
            "mujoco_steps": int(self.plant.step_count * self.config.plant_substeps),
            "simulated_duration_sec": float(self.plant.data.time),
            "wall_duration_sec": wall_duration,
            "real_time_factor": float(self.plant.data.time / wall_duration),
            "observed_control_step_rate_hz": float(steps / wall_duration),
            "missed_wall_deadlines": missed_deadlines,
            "nac_time_median_ms": _percentile(nac_times, 50.0),
            "nac_time_p95_ms": _percentile(nac_times, 95.0),
            "nac_time_p99_ms": _percentile(nac_times, 99.0),
            "mujoco_step_time_median_ms": _percentile(step_times, 50.0),
            "mujoco_step_time_p95_ms": _percentile(step_times, 95.0),
            "mujoco_step_time_p99_ms": _percentile(step_times, 99.0),
            "impedance_tracking_rmse_m": float(
                np.sqrt(np.mean(error_norm**2))
            ),
            "impedance_tracking_max_error_m": float(np.max(error_norm)),
            "command_force_max_norm_n": float(
                np.max(np.linalg.norm(command_history, axis=1))
            ),
            "arm_torque_max_norm_nm": float(np.max(torque_norm)),
            "arm_torque_max_abs_nm": float(np.max(np.abs(torque_history))),
            "joint_velocity_max_abs_rad_s": float(np.max(qvel_abs)),
            "contact_force_max_n": contact_max,
            "contact_penetration_max_m": penetration_max,
            "torque_saturation_count": torque_saturation_count,
            "torque_saturation_ratio": torque_saturation_count / steps,
            "torque_rate_saturation_count": rate_saturation_count,
            "final_weight_norm": self.controller.network.weight_norm,
            "fault_reason": "",
        }
        return MujocoRunResult(
            config=self.config,
            time=time_history,
            desired=desired_history,
            impedance=impedance_history,
            actual=actual_history,
            command_force=command_history,
            neural_estimate=neural_history,
            arm_torque=torque_history,
            joint_position=q_history,
            joint_velocity=qdot_history,
            object_position=object_history,
            contact_force=contact_history,
            metrics=metrics,
        )


def run_mujoco_tracking(config: MujocoRunConfig) -> MujocoRunResult:
    """Run the fixed-step entry point used by tests, docs, and ROS."""
    return MujocoNACRunner(config).run()
