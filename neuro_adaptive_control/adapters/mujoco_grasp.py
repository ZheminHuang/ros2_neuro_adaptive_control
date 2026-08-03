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

"""Deterministic pregrasp, close, lift, hold, and replace scenario."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Dict

import numpy as np

from neuro_adaptive_control.core.references import ReferenceSample

from .mujoco_simulation import (
    SimulationState,
    build_mujoco_controller,
    build_torque_mapper,
)
from .mujoco_ur5e_adapter import MujocoUR5ePlant
from .ur5e_wrench_to_torque import orientation_distance


@dataclass(frozen=True)
class GraspRunConfig:
    """Fixed timing and acceptance geometry for the grasp demonstration."""

    duration_sec: float = 11.0
    control_period_sec: float = 0.002
    seed: int = 29
    descent_m: float = 0.100
    lift_m: float = 0.080
    maximum_gripper_effort_n: float = 2.0
    maximum_contact_force_n: float = 180.0

    def __post_init__(self) -> None:
        for name in (
            "duration_sec",
            "control_period_sec",
            "descent_m",
            "lift_m",
            "maximum_gripper_effort_n",
            "maximum_contact_force_n",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if not np.isclose(self.control_period_sec, 0.002, atol=1e-15):
            raise ValueError("control period must be exactly 0.002 s")
        if self.duration_sec < 10.5:
            raise ValueError("duration must include the complete grasp sequence")


@dataclass(frozen=True)
class GraspRunResult:
    """Time histories and quantitative grasp acceptance evidence."""

    time: np.ndarray
    phase: tuple[str, ...]
    tcp_position: np.ndarray
    object_position: np.ndarray
    gripper_opening: np.ndarray
    arm_torque: np.ndarray
    contact_force: np.ndarray
    bilateral_contact: np.ndarray
    metrics: Dict[str, float | int | bool | str]


def _quintic_segment(
    time_sec: float,
    start_time: float,
    end_time: float,
    start: np.ndarray,
    end: np.ndarray,
) -> ReferenceSample:
    duration = end_time - start_time
    if time_sec <= start_time:
        return ReferenceSample(start.copy(), np.zeros(3), np.zeros(3))
    if time_sec >= end_time:
        return ReferenceSample(end.copy(), np.zeros(3), np.zeros(3))
    u = (time_sec - start_time) / duration
    position_scale = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    velocity_scale = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
    acceleration_scale = (
        60.0 * u - 180.0 * u**2 + 120.0 * u**3
    ) / duration**2
    delta = end - start
    return ReferenceSample(
        start + position_scale * delta,
        velocity_scale * delta,
        acceleration_scale * delta,
    )


def _schedule(
    time_sec: float,
    initial: np.ndarray,
    descent_m: float,
    lift_m: float,
) -> tuple[str, ReferenceSample, bool]:
    pregrasp = initial.copy()
    grasp = initial + np.array((0.0, 0.0, -descent_m))
    lifted = grasp + np.array((0.0, 0.0, lift_m))
    if time_sec < 1.0:
        return "pregrasp", ReferenceSample(pregrasp, np.zeros(3), np.zeros(3)), False
    if time_sec < 3.0:
        return "descend", _quintic_segment(time_sec, 1.0, 3.0, pregrasp, grasp), False
    if time_sec < 4.0:
        return "close", ReferenceSample(grasp, np.zeros(3), np.zeros(3)), True
    if time_sec < 5.5:
        return "lift", _quintic_segment(time_sec, 4.0, 5.5, grasp, lifted), True
    if time_sec < 7.5:
        return "hold", ReferenceSample(lifted, np.zeros(3), np.zeros(3)), True
    if time_sec < 9.0:
        return "lower", _quintic_segment(time_sec, 7.5, 9.0, lifted, grasp), True
    if time_sec < 10.0:
        return "release", ReferenceSample(grasp, np.zeros(3), np.zeros(3)), False
    return "retreat", _quintic_segment(time_sec, 10.0, 11.0, grasp, pregrasp), False


class MujocoGraspRunner:
    """Run the automated grasp using the same plant, NAC, and torque adapter."""

    def __init__(self, config: GraspRunConfig = GraspRunConfig()) -> None:
        self.config = config
        self.plant = MujocoUR5ePlant(seed=config.seed)
        initial = self.plant.kinematic_state()
        self.initial_tcp = initial.tcp_position.copy()
        self.controller = build_mujoco_controller(
            self.initial_tcp, adaptation_enabled=True, seed=config.seed
        )
        self.mapper = build_torque_mapper()
        self.state = SimulationState.START
        self.reason = "created"

    def _safe_hold(self) -> None:
        """Replace the last task command by bounded damping and gripper hold."""
        self.mapper.reset()
        velocity = self.plant.arm_velocity
        damping = np.zeros(6)
        if np.all(np.isfinite(velocity)):
            damping = self.mapper.damping_command(
                velocity, self.config.control_period_sec
            )
        self.plant.apply_safe_hold(damping)

    def _fault(self, reason: str) -> None:
        self.state = SimulationState.FAULT
        self.reason = str(reason)
        self._safe_hold()
        self.controller.safety.trigger_fault(self.reason)

    def _check_state(self, sample, contact, raw_torque: np.ndarray) -> None:
        """Apply the same finite, kinematic, actuator, and contact guards."""
        if not (
            np.all(np.isfinite(sample.all_joint_position))
            and np.all(np.isfinite(sample.all_joint_velocity))
            and np.all(np.isfinite(sample.tcp_position))
            and np.all(np.isfinite(sample.tcp_rotation))
            and np.all(np.isfinite(sample.translational_jacobian))
            and np.all(np.isfinite(sample.rotational_jacobian))
            and np.all(np.isfinite(raw_torque))
        ):
            raise FloatingPointError(
                "grasp state, FK, Jacobian, or torque contains NaN or Inf"
            )
        limits = self.plant.joint_limits
        if np.any(sample.all_joint_position < limits[:, 0] - 5e-3) or np.any(
            sample.all_joint_position > limits[:, 1] + 5e-3
        ):
            raise RuntimeError("arm or gripper joint-limit violation")
        if np.any(np.abs(sample.arm_velocity) > 3.5):
            raise RuntimeError("excessive joint velocity during grasp")
        if np.any(np.abs(raw_torque) > np.array((280, 280, 280, 54, 54, 54))):
            raise RuntimeError("excessive raw joint torque during grasp")
        if np.any(sample.tcp_position < (-0.85, -0.25, 0.10)) or np.any(
            sample.tcp_position > (0.70, 0.90, 1.25)
        ):
            raise RuntimeError("Cartesian workspace violation during grasp")
        if orientation_distance(
            sample.tcp_rotation, self.plant.desired_tcp_rotation
        ) > np.deg2rad(35.0):
            raise RuntimeError("orientation guard during grasp")
        if contact.contact_force_norm_n > self.config.maximum_contact_force_n:
            raise RuntimeError("excessive contact force during grasp")
        if contact.unexpected_contacts:
            raise RuntimeError("unexpected robot-environment grasp collision")
        gripper = self.plant.gripper_state(contact)
        if not np.all(
            np.isfinite(
                (
                    gripper.opening_m,
                    gripper.effort_n,
                    gripper.target_opening_m,
                    self.plant.gripper.actuator_control(),
                )
            )
        ) or abs(sample.all_joint_position[6] - sample.all_joint_position[10]) > 0.02:
            raise RuntimeError("gripper actuator/coupling failure")

    def run(self) -> GraspRunResult:
        """Execute the grasp and return numerical success/failure evidence."""
        if self.state != SimulationState.START:
            raise RuntimeError("grasp runner requires a fresh deterministic state")
        self.state = SimulationState.RUNNING
        self.controller.start(0.0)
        dt = self.config.control_period_sec
        steps = int(round(self.config.duration_sec / dt))
        time_history = np.empty(steps)
        tcp_history = np.empty((steps, 3))
        object_history = np.empty((steps, 3))
        opening_history = np.empty(steps)
        torque_history = np.empty((steps, 6))
        force_history = np.empty(steps)
        bilateral_history = np.zeros(steps, dtype=bool)
        phases: list[str] = []
        maximum_penetration = 0.0
        maximum_effort = 0.0
        maximum_velocity = 0.0
        saturation_count = 0
        solver_warning_count = 0
        unexpected_contact_count = 0
        wall_start = perf_counter_ns()

        for index in range(steps):
            stamp = index * dt
            sample = self.plant.kinematic_state()
            if sample.sequence_id != index or not np.isclose(
                sample.stamp_sec, stamp, atol=2e-12
            ):
                self._fault("grasp control stamp mismatch")
                raise RuntimeError("grasp control stamp mismatch")
            phase, reference, closing = _schedule(
                stamp,
                self.initial_tcp,
                self.config.descent_m,
                self.config.lift_m,
            )
            if closing:
                self.plant.gripper.close(self.config.maximum_gripper_effort_n)
            else:
                self.plant.gripper.open(self.config.maximum_gripper_effort_n)
            weights_before = self.controller.network.weights.copy()
            try:
                output = self.controller.step(
                    sample.tcp_position,
                    sample.tcp_linear_velocity,
                    reference,
                    np.zeros(3),
                    dt=dt,
                    now=stamp,
                    dynamics_features=np.concatenate(
                        (sample.arm_position, sample.arm_velocity)
                    ),
                )
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
                self.controller.network.weights[:] = weights_before
            saturation_count += int(mapping.torque_saturated)
            try:
                self._check_state(
                    sample, self.plant.contact_summary(), mapping.raw_command
                )
            except (FloatingPointError, RuntimeError, ValueError) as error:
                self._fault(str(error))
                raise RuntimeError(self.reason) from error
            warning_before = sum(w.number for w in self.plant.data.warning)
            try:
                next_sample = self.plant.advance(mapping.command)
            except (FloatingPointError, RuntimeError, ValueError) as error:
                self._fault(str(error))
                raise RuntimeError(self.reason) from error
            warning_after = sum(w.number for w in self.plant.data.warning)
            solver_warning_count += max(0, warning_after - warning_before)
            next_contact = self.plant.contact_summary()
            try:
                self._check_state(next_sample, next_contact, mapping.raw_command)
            except (FloatingPointError, RuntimeError, ValueError) as error:
                self._fault(str(error))
                raise RuntimeError(self.reason) from error
            unexpected_contact_count += next_contact.unexpected_contacts
            gripper_state = self.plant.gripper_state(next_contact)
            maximum_penetration = max(
                maximum_penetration, next_contact.maximum_penetration_m
            )
            maximum_effort = max(maximum_effort, gripper_state.effort_n)
            maximum_velocity = max(
                maximum_velocity, float(np.max(np.abs(next_sample.arm_velocity)))
            )
            time_history[index] = next_sample.stamp_sec
            tcp_history[index] = next_sample.tcp_position
            object_history[index] = next_sample.object_position
            opening_history[index] = gripper_state.opening_m
            torque_history[index] = mapping.command
            force_history[index] = next_contact.contact_force_norm_n
            bilateral_history[index] = (
                next_contact.left_finger_contacts > 0
                and next_contact.right_finger_contacts > 0
            )
            phases.append(phase)

        wall_duration = (perf_counter_ns() - wall_start) * 1e-9
        self.state = SimulationState.STOPPING
        self._safe_hold()
        self.state = SimulationState.STOPPED
        self.reason = "grasp complete; physics paused in bounded safe hold"

        phase_array = np.asarray(phases)
        settle_mask = (time_history >= 2.8) & (time_history < 3.0)
        hold_mask = phase_array == "hold"
        lift_mask = (phase_array == "lift") | hold_mask
        release_mask = phase_array == "release"
        initial_object_z = float(np.median(object_history[settle_mask, 2]))
        maximum_object_z = float(np.max(object_history[lift_mask, 2]))
        lift_height = maximum_object_z - initial_object_z
        hold_duration = float(np.count_nonzero(hold_mask) * dt)
        hold_min_z = float(np.min(object_history[hold_mask, 2]))
        hold_drop = maximum_object_z - hold_min_z
        hold_contact_ratio = float(np.mean(bilateral_history[hold_mask]))
        bilateral_duration = float(np.count_nonzero(bilateral_history) * dt)
        returned_height_error = abs(
            float(np.median(object_history[release_mask, 2])) - initial_object_z
        )
        success = bool(
            np.any(bilateral_history)
            and bilateral_duration >= 0.1
            and lift_height >= 0.05
            and hold_duration >= 2.0
            and hold_drop <= 0.005
            and hold_contact_ratio >= 0.90
            and maximum_penetration <= 0.002
            and maximum_effort <= self.config.maximum_gripper_effort_n + 1e-9
            and float(np.max(force_history))
            <= self.config.maximum_contact_force_n
            and unexpected_contact_count == 0
            and solver_warning_count == 0
            and returned_height_error <= 0.01
        )
        metrics: Dict[str, float | int | bool | str] = {
            "state": self.state.value,
            "success": success,
            "simulated_duration_sec": float(self.plant.data.time),
            "wall_duration_sec": wall_duration,
            "real_time_factor": float(self.plant.data.time / wall_duration),
            "bilateral_contact_duration_sec": bilateral_duration,
            "hold_duration_sec": hold_duration,
            "hold_bilateral_contact_ratio": hold_contact_ratio,
            "object_lift_height_m": lift_height,
            "hold_drop_m": hold_drop,
            "returned_height_error_m": returned_height_error,
            "maximum_penetration_m": maximum_penetration,
            "maximum_contact_force_n": float(np.max(force_history)),
            "maximum_gripper_effort_n": maximum_effort,
            "configured_contact_force_limit_n": (
                self.config.maximum_contact_force_n
            ),
            "maximum_arm_torque_abs_nm": float(np.max(np.abs(torque_history))),
            "maximum_joint_velocity_rad_s": maximum_velocity,
            "torque_saturation_count": saturation_count,
            "solver_warning_count": solver_warning_count,
            "unexpected_contact_count": unexpected_contact_count,
            "fault_reason": "",
        }
        return GraspRunResult(
            time=time_history,
            phase=tuple(phases),
            tcp_position=tcp_history,
            object_position=object_history,
            gripper_opening=opening_history,
            arm_torque=torque_history,
            contact_force=force_history,
            bilateral_contact=bilateral_history,
            metrics=metrics,
        )


def run_grasp_demo(config: GraspRunConfig = GraspRunConfig()) -> GraspRunResult:
    """Run the public deterministic grasp scenario."""
    return MujocoGraspRunner(config).run()
