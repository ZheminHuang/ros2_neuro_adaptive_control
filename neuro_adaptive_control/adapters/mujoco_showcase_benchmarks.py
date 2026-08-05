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

"""
Deterministic MuJoCo scenarios for the concise public showcase.

MuJoCo owns every physical dynamics change in this module.  The controller
receives only the coherent state/kinematics sample and, for the compliance
scenario, the explicitly measured physical TCP wrench transformed into the
analytical rotation-vector coordinates required by the 6D contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict

import numpy as np

from neuro_adaptive_control.core.pose_impedance_model import (
    PoseImpedanceParameters,
)
from neuro_adaptive_control.core.pose_references import PoseReferenceSample
from neuro_adaptive_control.core.so3 import (
    coordinate_transform,
    left_jacobian_inverse,
    log,
)

from .mujoco_payload_benchmark import (
    DEFAULT_PAYLOAD_CASE,
    build_pose_controller,
    build_pose_mapper,
    payload_schedule,
)
from .model_based_controller import MujocoModelBasedController
from .mujoco_ur5e_adapter import MujocoUR5ePlant


class ComplianceVariant(str, Enum):
    """Impedance settings compared under the same physical wrench."""

    SOFT = "soft_impedance"
    STIFF = "stiff_impedance"


class DragVariant(str, Enum):
    """Controllers evaluated after the hidden plant change."""

    ADAPTIVE = "adaptive_nac"
    NOMINAL = "nominal_model_based"
    FROZEN = "frozen_at_disturbance"


@dataclass(frozen=True)
class ShowcaseResult:
    """Canonical history shared by metrics, plots, and MuJoCo rendering."""

    scenario: str
    variant: str
    time: np.ndarray
    phase: tuple[str, ...]
    desired_pose: np.ndarray
    impedance_pose: np.ndarray
    actual_pose: np.ndarray
    generalized_command: np.ndarray
    neural_estimate: np.ndarray
    arm_torque: np.ndarray
    weight_norm: np.ndarray
    physical_wrench: np.ndarray
    generalized_wrench: np.ndarray
    qpos: np.ndarray
    metrics: Dict[str, float | int | bool | str]


def _smooth_pulse(
    time_sec: float,
    start: float,
    ramp: float,
    hold: float,
) -> float:
    """Return a unit quintic ramp/hold/ramp pulse."""
    if time_sec < start or time_sec >= start + 2.0 * ramp + hold:
        return 0.0
    elapsed = time_sec - start
    if elapsed < ramp:
        s = elapsed / ramp
        return float(10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)
    if elapsed < ramp + hold:
        return 1.0
    s = (elapsed - ramp - hold) / ramp
    return float(1.0 - (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5))


def compliance_wrench(time_sec: float) -> tuple[str, np.ndarray]:
    """Return world-frame physical ``[force, moment]`` at the TCP."""
    force_scale = _smooth_pulse(time_sec, 7.0, 0.5, 1.0)
    moment_scale = _smooth_pulse(time_sec, 10.0, 0.5, 1.0)
    wrench = np.array(
        (0.0, 6.0 * force_scale, 0.0, 0.0, 0.0, 0.4 * moment_scale)
    )
    if force_scale > 0.0:
        phase = "lateral_push"
    elif moment_scale > 0.0:
        phase = "twist_moment"
    elif time_sec >= 6.5:
        phase = "recovery"
    else:
        phase = payload_schedule(time_sec, np.zeros(6))[0]
    return phase, wrench


def _actual_pose(sample, reference_rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rho = log(sample.tcp_rotation @ reference_rotation.T)
    rho_dot = left_jacobian_inverse(rho) @ sample.tcp_angular_velocity
    return (
        np.concatenate((sample.tcp_position, rho)),
        np.concatenate((sample.tcp_linear_velocity, rho_dot)),
    )


def _set_impedance_variant(controller, variant: ComplianceVariant) -> None:
    scale = 1.0 if variant == ComplianceVariant.SOFT else 2.0
    controller.impedance_model.parameters = PoseImpedanceParameters.diagonal(
        mass=(1.0, 1.0, 1.0, 0.18, 0.18, 0.18),
        damping=(30.0, 30.0 * np.sqrt(scale), 34.0, 3.2, 3.2, 3.2 * np.sqrt(scale)),
        stiffness=(110.0, 110.0 * scale, 125.0, 12.0, 12.0, 12.0 * scale),
        external_gain=np.ones(6),
    )


def _validate_sample(plant: MujocoUR5ePlant, sample, torque: np.ndarray) -> None:
    arrays = (
        sample.all_joint_position,
        sample.all_joint_velocity,
        sample.tcp_position,
        sample.tcp_rotation,
        torque,
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise FloatingPointError("showcase state or command contains NaN/Inf")
    if np.max(np.abs(sample.arm_velocity)) > 4.0:
        raise RuntimeError("showcase joint velocity limit exceeded")
    if np.max(np.abs(torque)) > 80.0 + 1.0e-12:
        raise RuntimeError("showcase joint torque limit exceeded")
    limits = plant.joint_limits
    if np.any(sample.all_joint_position < limits[:, 0] - 8.0e-3) or np.any(
        sample.all_joint_position > limits[:, 1] + 8.0e-3
    ):
        raise RuntimeError("showcase joint limit exceeded")


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(error), axis=1))))


def run_compliance_benchmark(
    variant: ComplianceVariant | str,
) -> ShowcaseResult:
    """Run grasp, lift, 6 N push, and 0.4 Nm twist with one impedance."""
    selected = ComplianceVariant(variant)
    payload = DEFAULT_PAYLOAD_CASE
    plant = MujocoUR5ePlant(
        seed=payload.seed,
        payload_mass_kg=payload.mass_kg,
        payload_com_offset_m=payload.com_offset_m,
        payload_inertia_scale=payload.inertia_scale,
    )
    initial = plant.kinematic_state()
    reference_rotation = initial.tcp_rotation.copy()
    initial_pose = np.concatenate((initial.tcp_position, np.zeros(3)))
    controller = build_pose_controller(initial_pose, seed=payload.seed)
    _set_impedance_variant(controller, selected)
    mapper = build_pose_mapper()
    controller.start(0.0)
    dt = 0.002
    steps = int(round(14.0 / dt))
    time = np.empty(steps)
    desired = np.empty((steps, 6))
    impedance = np.empty((steps, 6))
    actual = np.empty((steps, 6))
    command = np.empty((steps, 6))
    neural = np.empty((steps, 6))
    torque = np.empty((steps, 6))
    weights = np.empty(steps)
    physical = np.empty((steps, 6))
    generalized = np.empty((steps, 6))
    qpos = np.empty((steps, plant.model.nq))
    phases: list[str] = []
    saturation_count = 0
    lifted_reference = payload_schedule(6.5, initial_pose)[1]
    for index in range(steps):
        stamp = index * dt
        sample = plant.kinematic_state()
        actual_pose, actual_velocity = _actual_pose(sample, reference_rotation)
        if stamp < 6.5:
            phase, reference, closing = payload_schedule(stamp, initial_pose)
            wrench = np.zeros(6)
        else:
            phase, wrench = compliance_wrench(stamp)
            reference = lifted_reference
            closing = True
        if closing:
            plant.gripper.close(5.0)
        else:
            plant.gripper.open(5.0)
        analytical_wrench = coordinate_transform(actual_pose[3:]).T @ wrench
        checkpoint = controller.network.checkpoint()
        output = controller.step(
            actual_pose,
            actual_velocity,
            sample.arm_position,
            sample.arm_velocity,
            reference,
            analytical_wrench,
            dt=dt,
            now=stamp,
        )
        if output.state.value == "fault":
            raise RuntimeError(output.fault_reason)
        jacobian = np.vstack(
            (sample.translational_jacobian, sample.rotational_jacobian)
        )
        mapped = mapper.map_running_command(
            output.command, actual_pose[3:], jacobian, dt
        )
        if mapped.torque_saturated or mapped.rate_saturated:
            controller.network.restore(checkpoint)
        saturation_count += int(mapped.torque_saturated or mapped.rate_saturated)
        next_sample = plant.advance(
            mapped.command,
            injected_force_world=wrench[:3],
            injected_torque_world=wrench[3:],
        )
        _validate_sample(plant, next_sample, mapped.command)
        next_actual, _ = _actual_pose(next_sample, reference_rotation)
        time[index] = next_sample.stamp_sec
        desired[index] = reference.position
        impedance[index] = output.model_state.position
        actual[index] = next_actual
        command[index] = output.command
        neural[index] = output.neural_estimate
        torque[index] = mapped.command
        weights[index] = controller.network.combined_weight_norm
        physical[index] = wrench
        generalized[index] = analytical_wrench
        qpos[index] = plant.data.qpos
        phases.append(phase)
    phase_array = np.asarray(phases)
    push_hold = (time >= 7.5) & (time < 8.5)
    twist_hold = (time >= 10.5) & (time < 11.5)
    recovery = time >= 13.0
    y_deflection = actual[:, 1] - desired[:, 1]
    rz_deflection = actual[:, 5] - desired[:, 5]
    model_error = impedance - actual
    contact = plant.contact_summary()
    metrics: Dict[str, float | int | bool | str] = {
        "success": bool(
            np.any(phase_array == "lateral_push")
            and np.any(phase_array == "twist_moment")
            and contact.left_finger_contacts > 0
            and contact.right_finger_contacts > 0
        ),
        "force_n": 6.0,
        "moment_nm": 0.4,
        "push_hold_deflection_m": float(np.mean(y_deflection[push_hold])),
        "twist_hold_deflection_rad": float(np.mean(rz_deflection[twist_hold])),
        "apparent_translation_compliance_m_per_n": float(
            abs(np.mean(y_deflection[push_hold])) / 6.0
        ),
        "apparent_rotation_compliance_rad_per_nm": float(
            abs(np.mean(rz_deflection[twist_hold])) / 0.4
        ),
        "actual_to_impedance_position_rmse_m": _rmse(model_error[time >= 6.5, :3]),
        "actual_to_impedance_orientation_rmse_rad": _rmse(
            model_error[time >= 6.5, 3:]
        ),
        "actual_to_impedance_position_max_error_m": float(
            np.max(np.linalg.norm(model_error[time >= 6.5, :3], axis=1))
        ),
        "actual_to_impedance_orientation_max_error_rad": float(
            np.max(np.linalg.norm(model_error[time >= 6.5, 3:], axis=1))
        ),
        "recovery_position_error_m": float(
            np.mean(np.linalg.norm((desired - actual)[recovery, :3], axis=1))
        ),
        "recovery_orientation_error_rad": float(
            np.mean(np.linalg.norm((desired - actual)[recovery, 3:], axis=1))
        ),
        "maximum_arm_torque_abs_nm": float(np.max(np.abs(torque))),
        "saturated_control_samples": saturation_count,
        "completed_steps": steps,
    }
    return ShowcaseResult(
        scenario="push_and_twist_compliance",
        variant=selected.value,
        time=time,
        phase=tuple(phases),
        desired_pose=desired,
        impedance_pose=impedance,
        actual_pose=actual,
        generalized_command=command,
        neural_estimate=neural,
        arm_torque=torque,
        weight_norm=weights,
        physical_wrench=physical,
        generalized_wrench=generalized,
        qpos=qpos,
        metrics=metrics,
    )


def joint_drag_reference(time_sec: float, initial_pose: np.ndarray) -> PoseReferenceSample:
    """Return a smooth closed 6D spatial curve between 2 and 9 seconds."""
    if time_sec <= 2.0 or time_sec >= 9.0:
        return PoseReferenceSample(initial_pose, np.zeros(6), np.zeros(6))
    duration = 7.0
    s = (time_sec - 2.0) / duration
    sigma = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    sigma_dot = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
    sigma_ddot = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / duration**2
    angle = 2.0 * np.pi * sigma
    angle_dot = 2.0 * np.pi * sigma_dot
    angle_ddot = 2.0 * np.pi * sigma_ddot
    amplitude = np.array((0.025, 0.035, 0.018, 0.040, 0.035, 0.050))
    shape = np.array(
        (
            np.cos(angle) - 1.0,
            np.sin(angle),
            np.sin(2.0 * angle),
            np.sin(angle),
            1.0 - np.cos(angle),
            np.sin(2.0 * angle),
        )
    )
    dshape = np.array(
        (
            -np.sin(angle),
            np.cos(angle),
            2.0 * np.cos(2.0 * angle),
            np.cos(angle),
            np.sin(angle),
            2.0 * np.cos(2.0 * angle),
        )
    )
    ddshape = np.array(
        (
            -np.cos(angle),
            -np.sin(angle),
            -4.0 * np.sin(2.0 * angle),
            -np.sin(angle),
            np.cos(angle),
            -4.0 * np.sin(2.0 * angle),
        )
    )
    return PoseReferenceSample(
        initial_pose + amplitude * shape,
        amplitude * dshape * angle_dot,
        amplitude * (ddshape * angle_dot**2 + dshape * angle_ddot),
    )


def run_joint_drag_benchmark(variant: DragVariant | str) -> ShowcaseResult:
    """Run a 6D curve with hidden MuJoCo damping/friction added at 4 s."""
    selected = DragVariant(variant)
    plant = MujocoUR5ePlant(seed=83)
    initial = plant.kinematic_state()
    reference_rotation = initial.tcp_rotation.copy()
    initial_pose = np.concatenate((initial.tcp_position, np.zeros(3)))
    controller = build_pose_controller(initial_pose, seed=83)
    mapper = build_pose_mapper()
    model_controller = MujocoModelBasedController(
        torque_limits=(80.0, 80.0, 80.0, 28.0, 28.0, 28.0),
        torque_rate_limits=(
            8000.0,
            8000.0,
            8000.0,
            3000.0,
            3000.0,
            3000.0,
        ),
    )
    controller.start(0.0)
    dt = 0.002
    duration = 10.0
    event_time = 4.0
    steps = int(round(duration / dt))
    time = np.empty(steps)
    desired = np.empty((steps, 6))
    impedance = np.empty((steps, 6))
    actual = np.empty((steps, 6))
    command = np.empty((steps, 6))
    neural = np.empty((steps, 6))
    torque = np.empty((steps, 6))
    weights = np.empty(steps)
    qpos = np.empty((steps, plant.model.nq))
    phases: list[str] = []
    saturation_count = 0
    plant.gripper.open(5.0)
    for index in range(steps):
        stamp = index * dt
        if index == int(round(event_time / dt)):
            plant.apply_joint_drag(
                ("shoulder_lift_joint", "elbow_joint", "wrist_2_joint"),
                damping_scale=8.0,
                frictionloss_scale=6.0,
            )
            if selected == DragVariant.FROZEN:
                controller.network.adaptation_enabled = False
        sample = plant.kinematic_state()
        actual_pose, actual_velocity = _actual_pose(sample, reference_rotation)
        reference = joint_drag_reference(stamp, initial_pose)
        jacobian = np.vstack(
            (sample.translational_jacobian, sample.rotational_jacobian)
        )
        if selected == DragVariant.NOMINAL:
            model_output = model_controller.command(
                all_joint_position=sample.all_joint_position,
                all_joint_velocity=sample.all_joint_velocity,
                actual_pose=actual_pose,
                actual_pose_velocity=actual_velocity,
                reference=reference,
                rotation_vector=actual_pose[3:],
                geometric_jacobian=jacobian,
                tcp_position=sample.tcp_position,
                payload_position=sample.object_position,
                payload_acquired=False,
                dt=dt,
            )
            torque_command = model_output.command
            generalized_command = np.zeros(6)
            neural_estimate = np.zeros(6)
            impedance_position = reference.position
            weight_norm = 0.0
            saturated = (
                model_output.torque_saturated or model_output.rate_saturated
            )
        else:
            checkpoint = controller.network.checkpoint()
            output = controller.step(
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
            mapped = mapper.map_running_command(
                output.command, actual_pose[3:], jacobian, dt
            )
            if mapped.torque_saturated or mapped.rate_saturated:
                controller.network.restore(checkpoint)
            torque_command = mapped.command
            generalized_command = output.command
            neural_estimate = output.neural_estimate
            impedance_position = output.model_state.position
            weight_norm = controller.network.combined_weight_norm
            saturated = mapped.torque_saturated or mapped.rate_saturated
        saturation_count += int(saturated)
        next_sample = plant.advance(torque_command)
        _validate_sample(plant, next_sample, torque_command)
        if plant.contact_summary().total_robot_environment_contacts:
            raise RuntimeError("joint-drag scenario must remain collision-free")
        next_actual, _ = _actual_pose(next_sample, reference_rotation)
        time[index] = next_sample.stamp_sec
        desired[index] = reference.position
        impedance[index] = impedance_position
        actual[index] = next_actual
        command[index] = generalized_command
        neural[index] = neural_estimate
        torque[index] = torque_command
        weights[index] = weight_norm
        qpos[index] = plant.data.qpos
        phases.append("hidden_drag" if stamp >= event_time else "nominal_dynamics")
    evaluation = (time >= event_time + 0.25) & (time < 9.0)
    recovery = time >= 9.25
    error = desired - actual
    metrics: Dict[str, float | int | bool | str] = {
        "success": True,
        "event_time_sec": event_time,
        "damping_scale": 8.0,
        "frictionloss_scale": 6.0,
        "changed_joints": "shoulder_lift_joint,elbow_joint,wrist_2_joint",
        "post_event_position_rmse_m": _rmse(error[evaluation, :3]),
        "post_event_orientation_rmse_rad": _rmse(error[evaluation, 3:]),
        "post_event_position_max_error_m": float(
            np.max(np.linalg.norm(error[evaluation, :3], axis=1))
        ),
        "post_event_orientation_max_error_rad": float(
            np.max(np.linalg.norm(error[evaluation, 3:], axis=1))
        ),
        "recovery_position_error_m": float(
            np.mean(np.linalg.norm(error[recovery, :3], axis=1))
        ),
        "recovery_orientation_error_rad": float(
            np.mean(np.linalg.norm(error[recovery, 3:], axis=1))
        ),
        "weight_change_after_event": float(
            weights[-1] - weights[np.searchsorted(time, event_time)]
        ),
        "maximum_arm_torque_abs_nm": float(np.max(np.abs(torque))),
        "saturated_control_samples": saturation_count,
        "completed_steps": steps,
    }
    zeros = np.zeros((steps, 6))
    return ShowcaseResult(
        scenario="unknown_joint_drag",
        variant=selected.value,
        time=time,
        phase=tuple(phases),
        desired_pose=desired,
        impedance_pose=impedance,
        actual_pose=actual,
        generalized_command=command,
        neural_estimate=neural,
        arm_torque=torque,
        weight_norm=weights,
        physical_wrench=zeros.copy(),
        generalized_wrench=zeros.copy(),
        qpos=qpos,
        metrics=metrics,
    )


def compare_compliance(
    lower: ShowcaseResult, higher: ShowcaseResult
) -> Dict[str, float | bool]:
    """Summarize fixed-tuning tracking across two impedance settings."""
    maximum_position_rmse = max(
        float(lower.metrics["actual_to_impedance_position_rmse_m"]),
        float(higher.metrics["actual_to_impedance_position_rmse_m"]),
    )
    maximum_orientation_rmse = max(
        float(lower.metrics["actual_to_impedance_orientation_rmse_rad"]),
        float(higher.metrics["actual_to_impedance_orientation_rmse_rad"]),
    )
    return {
        "maximum_actual_to_impedance_position_rmse_m": maximum_position_rmse,
        "maximum_actual_to_impedance_orientation_rmse_rad": (
            maximum_orientation_rmse
        ),
        "controller_tuning_identical_between_trials": True,
        "online_adaptation_enabled_in_both_trials": True,
        "both_trials_completed": bool(
            lower.metrics["success"] and higher.metrics["success"]
        ),
        "fixed_tuning_tracking_gate_passed": bool(
            lower.metrics["success"]
            and higher.metrics["success"]
            and maximum_position_rmse <= 1.0e-3
            and maximum_orientation_rmse <= 1.0e-3
        ),
    }


def _tracking_improvement(
    adaptive: ShowcaseResult, baseline: ShowcaseResult
) -> tuple[float, float]:
    adaptive_position = float(adaptive.metrics["post_event_position_rmse_m"])
    baseline_position = float(baseline.metrics["post_event_position_rmse_m"])
    adaptive_orientation = float(
        adaptive.metrics["post_event_orientation_rmse_rad"]
    )
    baseline_orientation = float(
        baseline.metrics["post_event_orientation_rmse_rad"]
    )
    return (
        1.0 - adaptive_position / baseline_position,
        1.0 - adaptive_orientation / baseline_orientation,
    )


def compare_joint_drag(
    adaptive: ShowcaseResult,
    nominal: ShowcaseResult,
    frozen: ShowcaseResult,
) -> Dict[str, float | bool]:
    """Summarize the public nominal comparison and NN adaptation ablation."""
    nominal_position, nominal_orientation = _tracking_improvement(
        adaptive, nominal
    )
    frozen_position, frozen_orientation = _tracking_improvement(
        adaptive, frozen
    )
    return {
        "position_rmse_improvement_vs_nominal_ratio": nominal_position,
        "orientation_rmse_improvement_vs_nominal_ratio": nominal_orientation,
        "position_rmse_improvement_vs_frozen_ratio": frozen_position,
        "orientation_rmse_improvement_vs_frozen_ratio": frozen_orientation,
        "public_nominal_comparison_gate_passed": bool(
            nominal_position >= 0.10 and nominal_orientation >= 0.10
        ),
        "nn_adaptation_ablation_gate_passed": bool(
            frozen_position >= 0.10 and frozen_orientation >= 0.10
        ),
    }
