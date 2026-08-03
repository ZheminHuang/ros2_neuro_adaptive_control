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

"""Own one deterministic MuJoCo plant and publish its state for RViz."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter_ns, sleep

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, PoseStamped, Vector3Stamped, WrenchStamped
from nav_msgs.msg import Path as PathMessage
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from neuro_adaptive_control.adapters.mujoco_grasp import _schedule
from neuro_adaptive_control.adapters.mujoco_simulation import (
    MujocoRunConfig,
    SimulationState,
    _injected_force,
    _reference,
    build_mujoco_controller,
    build_torque_mapper,
)
from neuro_adaptive_control.adapters.mujoco_ur5e_adapter import (
    MujocoUR5ePlant,
)
from neuro_adaptive_control.adapters.ur5e_wrench_to_torque import (
    orientation_distance,
)


def _stamp(seconds: float) -> Time:
    whole = int(np.floor(seconds))
    nanoseconds = int(round((seconds - whole) * 1e9))
    if nanoseconds >= 1_000_000_000:
        whole += 1
        nanoseconds -= 1_000_000_000
    return Time(sec=whole, nanosec=nanoseconds)


def _quaternion_from_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            )
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                (
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                )
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                (
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                )
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                (
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


class MujocoUR5ePlantNode(Node):
    """Synchronously compute NAC and four MuJoCo substeps per timer tick."""

    def __init__(self) -> None:
        super().__init__("mujoco_ur5e_plant")
        self._declare_parameters()
        package_share = Path(get_package_share_directory("neuro_adaptive_control"))
        requested_model = str(self.get_parameter("model_path").value)
        model_path = (
            Path(requested_model)
            if requested_model
            else package_share / "mujoco" / "ur5e_robotiq_2f85.xml"
        )
        self._scenario = str(self.get_parameter("scenario").value).lower()
        if self._scenario not in {"trajectory", "grasp"}:
            raise ValueError("scenario must be trajectory or grasp")
        self._trajectory_name = str(self.get_parameter("trajectory").value)
        self._duration = float(self.get_parameter("duration_sec").value)
        if not np.isfinite(self._duration) or self._duration < 0.0:
            raise ValueError("duration_sec must be finite and non-negative")
        if self._duration > 0.0 and not np.isclose(
            round(self._duration / 0.002) * 0.002,
            self._duration,
            atol=1e-12,
        ):
            raise ValueError("duration_sec must be an integer number of 0.002 s steps")
        self._external_mode = str(
            self.get_parameter("external_wrench_mode").value
        ).lower()
        if self._external_mode not in {"none", "injected", "virtual_ft"}:
            raise ValueError("external_wrench_mode must be none/injected/virtual_ft")
        self._adaptation_enabled = bool(
            self.get_parameter("adaptation_enabled").value
        )
        self._frame = str(self.get_parameter("frame_id").value)
        visualization_rate = float(
            self.get_parameter("visualization_rate_hz").value
        )
        if not np.isfinite(visualization_rate) or visualization_rate <= 0.0:
            raise ValueError("visualization_rate_hz must be finite and positive")
        self._visualization_stride = max(
            1,
            int(round(500.0 / visualization_rate)),
        )
        self._shutdown_when_complete = bool(
            self.get_parameter("shutdown_when_complete").value
        )
        self._metrics_path = str(self.get_parameter("metrics_path").value)
        self._grasp_descent_m = float(
            self.get_parameter("grasp_descent_m").value
        )
        self._grasp_lift_m = float(self.get_parameter("grasp_lift_m").value)
        self._grasp_effort_n = float(
            self.get_parameter("grasp_maximum_effort_n").value
        )
        self._grasp_contact_limit_n = float(
            self.get_parameter("grasp_contact_force_limit_n").value
        )
        if not np.all(
            np.isfinite(
                (
                    self._grasp_descent_m,
                    self._grasp_lift_m,
                    self._grasp_effort_n,
                    self._grasp_contact_limit_n,
                )
            )
        ) or min(
            self._grasp_descent_m,
            self._grasp_lift_m,
            self._grasp_effort_n,
            self._grasp_contact_limit_n,
        ) <= 0.0:
            raise ValueError("all grasp geometry/effort limits must be positive")
        self._plant = MujocoUR5ePlant(
            model_path, seed=int(self.get_parameter("seed").value)
        )
        initial = self._plant.kinematic_state()
        self._initial_tcp = initial.tcp_position.copy()
        self._initial_object_z = float(initial.object_position[2])
        self._controller = build_mujoco_controller(
            self._initial_tcp,
            adaptation_enabled=self._adaptation_enabled,
            seed=int(self.get_parameter("seed").value),
        )
        self._mapper = build_torque_mapper()
        config = MujocoRunConfig(
            trajectory=self._trajectory_name,
            duration_sec=max(self._duration, 0.002),
            adaptation_enabled=self._adaptation_enabled,
            external_wrench_mode=self._external_mode,
            seed=int(self.get_parameter("seed").value),
        )
        self._reference = _reference(config, self._initial_tcp)
        self._state = SimulationState.START
        self._reason = "initialized"
        self._exit_requested = False
        self._step_index = 0
        self._last_output = None
        self._last_mapping = None
        self._last_reference = None
        self._reset_metrics()
        self._path_desired = PathMessage()
        self._path_impedance = PathMessage()
        self._path_actual = PathMessage()
        for path in (
            self._path_desired,
            self._path_impedance,
            self._path_actual,
        ):
            path.header.frame_id = self._frame
        self._create_publishers()
        self._gripper_command_sub = self.create_subscription(
            Float64MultiArray,
            "mujoco/gripper/command",
            self._on_gripper_command,
            10,
        )
        self._reset_service = self.create_service(
            Trigger, "mujoco/reset", self._on_reset
        )
        self._stop_service = self.create_service(
            Trigger, "mujoco/stop", self._on_stop
        )
        self._viewer = None
        if bool(self.get_parameter("start_mujoco_viewer").value):
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(
                self._plant.model, self._plant.data
            )
            with self._viewer.lock():
                self._viewer.cam.lookat[:] = np.array((0.0, 0.35, 0.35))
                self._viewer.cam.distance = 1.45
                self._viewer.cam.azimuth = 135.0
                self._viewer.cam.elevation = -22.0
                self._viewer.opt.geomgroup[3] = True
                self._viewer.opt.flags[
                    mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
                ] = True
                self._viewer.opt.flags[
                    mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
                ] = True
        self._controller.start(0.0)
        self._state = SimulationState.RUNNING
        self._wall_origin_ns = perf_counter_ns()
        self._timer = self.create_timer(0.002, self._on_control_tick)
        self.get_logger().info(
            "single-owner MuJoCo loop started: 0.002 s control, "
            "4 x 0.0005 s substeps; RViz is display-only"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("model_path", "")
        self.declare_parameter("scenario", "trajectory")
        self.declare_parameter("trajectory", "circle")
        self.declare_parameter("duration_sec", 12.0)
        self.declare_parameter("adaptation_enabled", True)
        self.declare_parameter("external_wrench_mode", "none")
        self.declare_parameter("seed", 23)
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("visualization_rate_hz", 20.0)
        self.declare_parameter("shutdown_when_complete", True)
        self.declare_parameter("start_mujoco_viewer", False)
        self.declare_parameter("metrics_path", "")
        self.declare_parameter("grasp_descent_m", 0.100)
        self.declare_parameter("grasp_lift_m", 0.080)
        self.declare_parameter("grasp_maximum_effort_n", 2.0)
        self.declare_parameter("grasp_contact_force_limit_n", 180.0)

    def _reset_metrics(self) -> None:
        """Clear every accumulated timing, tracking, contact, and grasp value."""
        self._nac_times_ms: list[float] = []
        self._step_times_ms: list[float] = []
        self._timer_interarrival_ms: list[float] = []
        self._last_tick_start_ns: int | None = None
        self._missed_deadlines = 0
        self._saturation_count = 0
        self._rate_saturation_count = 0
        self._error_squared_sum = 0.0
        self._tracking_error_max_m = 0.0
        self._desired_error_squared_sum = 0.0
        self._desired_error_max_m = 0.0
        self._command_force_max_norm_n = 0.0
        self._arm_torque_max_norm_nm = 0.0
        self._arm_torque_max_abs_nm = 0.0
        self._joint_velocity_max_abs_rad_s = 0.0
        self._orientation_error_max_rad = 0.0
        self._contact_force_max_n = 0.0
        self._contact_penetration_max_m = 0.0
        self._unexpected_contact_count = 0
        self._gripper_effort_max_n = 0.0
        self._bilateral_contact_samples = 0
        self._hold_samples = 0
        self._hold_bilateral_samples = 0
        self._settle_object_z: list[float] = []
        self._lift_hold_object_z: list[float] = []
        self._hold_object_z: list[float] = []
        self._release_object_z: list[float] = []

    def _create_publishers(self) -> None:
        self._clock_pub = self.create_publisher(Clock, "clock", 10)
        self._joint_pub = self.create_publisher(JointState, "joint_states", 20)
        self._desired_pose_pub = self.create_publisher(
            PoseStamped, "mujoco/desired_pose", 10
        )
        self._impedance_pose_pub = self.create_publisher(
            PoseStamped, "mujoco/impedance_pose", 10
        )
        self._actual_pose_pub = self.create_publisher(
            PoseStamped, "mujoco/actual_pose", 10
        )
        self._command_pub = self.create_publisher(
            WrenchStamped, "mujoco/nac_command", 10
        )
        self._nn_pub = self.create_publisher(
            WrenchStamped, "mujoco/nn_estimate", 10
        )
        self._error_pub = self.create_publisher(
            Vector3Stamped, "mujoco/tracking_error", 10
        )
        self._wrist_pub = self.create_publisher(
            WrenchStamped, "mujoco/wrist_wrench_raw", 10
        )
        self._contact_wrench_pub = self.create_publisher(
            WrenchStamped, "mujoco/external_contact_wrench", 10
        )
        self._legacy_wrench_pub = self.create_publisher(
            WrenchStamped, "mujoco/wrist_wrench", 10
        )
        self._contact_marker_pub = self.create_publisher(
            MarkerArray, "mujoco/contact_markers", 10
        )
        self._scene_marker_pub = self.create_publisher(
            MarkerArray, "mujoco/scene_markers", 10
        )
        self._diagnostics_pub = self.create_publisher(
            DiagnosticArray, "diagnostics", 10
        )
        self._contact_diagnostics_pub = self.create_publisher(
            DiagnosticArray, "mujoco/contact_diagnostics", 10
        )
        self._gripper_state_pub = self.create_publisher(
            Float64MultiArray, "mujoco/gripper/state", 10
        )
        self._gripper_actuator_pub = self.create_publisher(
            Float64, "mujoco/gripper/actuator_command", 10
        )
        self._desired_path_pub = self.create_publisher(
            PathMessage, "mujoco/desired_path", 10
        )
        self._impedance_path_pub = self.create_publisher(
            PathMessage, "mujoco/impedance_path", 10
        )
        self._actual_path_pub = self.create_publisher(
            PathMessage, "mujoco/actual_path", 10
        )

    def _on_gripper_command(self, message: Float64MultiArray) -> None:
        try:
            if len(message.data) < 2:
                raise ValueError("gripper command needs position and effort")
            if len(message.data) >= 3 and bool(message.data[2]):
                current = self._plant.gripper_state().opening_m
                self._plant.gripper.stop(current)
            elif len(message.data) >= 4 and bool(message.data[3]):
                self._plant.gripper.reset()
            else:
                self._plant.gripper.command(message.data[0], message.data[1])
        except ValueError as error:
            self._fault(str(error))

    def _on_reset(self, request, response):
        del request
        self._state = SimulationState.RESETTING
        initial = self._plant.reset()
        self._initial_tcp = initial.tcp_position.copy()
        self._initial_object_z = float(initial.object_position[2])
        self._controller = build_mujoco_controller(
            self._initial_tcp,
            adaptation_enabled=self._adaptation_enabled,
            seed=int(self.get_parameter("seed").value),
        )
        self._controller.start(0.0)
        self._mapper.reset()
        config = MujocoRunConfig(
            trajectory=self._trajectory_name,
            duration_sec=max(self._duration, 0.002),
            adaptation_enabled=self._adaptation_enabled,
            external_wrench_mode=self._external_mode,
            seed=int(self.get_parameter("seed").value),
        )
        self._reference = _reference(config, self._initial_tcp)
        self._step_index = 0
        self._exit_requested = False
        self._last_output = None
        self._last_mapping = None
        self._last_reference = None
        self._reset_metrics()
        self._path_desired.poses.clear()
        self._path_impedance.poses.clear()
        self._path_actual.poses.clear()
        self._state = SimulationState.RUNNING
        self._reason = "reset complete"
        self._wall_origin_ns = perf_counter_ns()
        response.success = True
        response.message = self._reason
        return response

    def _on_stop(self, request, response):
        del request
        response.success = self._stop("stop service requested")
        response.message = (
            self._reason
            if response.success
            else f"fault latched; reset required: {self._reason}"
        )
        return response

    def _fault(self, reason: str) -> None:
        self._state = SimulationState.FAULT
        self._reason = str(reason)
        self._apply_safe_hold()
        self._controller.safety.trigger_fault(self._reason)
        self.get_logger().error(f"MuJoCo controller fault: {self._reason}")

    def _stop(self, reason: str) -> bool:
        if (
            self._state == SimulationState.FAULT
            or self._controller.state.value == "fault"
        ):
            if self._state != SimulationState.FAULT:
                self._state = SimulationState.FAULT
                self._reason = self._controller.safety.reason
            self._apply_safe_hold()
            return False
        self._state = SimulationState.STOPPING
        self._apply_safe_hold()
        self._controller.stop(reason)
        self._controller.safety.complete_stop()
        self._state = SimulationState.STOPPED
        self._reason = str(reason)
        return True

    def _apply_safe_hold(self) -> None:
        """Replace NAC output by finite damping and a measured gripper hold."""
        velocity = self._plant.arm_velocity
        self._mapper.reset()
        damping = np.zeros(6)
        if np.all(np.isfinite(velocity)):
            damping = self._mapper.damping_command(velocity, 0.002)
        self._plant.apply_safe_hold(damping)

    def _control_reference(self, stamp_sec: float):
        if self._scenario == "grasp":
            phase, reference, closing = _schedule(
                stamp_sec,
                self._initial_tcp,
                self._grasp_descent_m,
                self._grasp_lift_m,
            )
            if closing:
                self._plant.gripper.close(self._grasp_effort_n)
            else:
                self._plant.gripper.open(self._grasp_effort_n)
            return phase, reference
        return self._trajectory_name, self._reference.evaluate(stamp_sec)

    def _observation_reference(self, stamp_sec: float):
        """Return the reference aligned with a post-integration state stamp."""
        if self._scenario == "grasp":
            phase, reference, _ = _schedule(
                stamp_sec,
                self._initial_tcp,
                self._grasp_descent_m,
                self._grasp_lift_m,
            )
            return phase, reference
        return self._trajectory_name, self._reference.evaluate(stamp_sec)

    def _on_control_tick(self) -> None:
        if self._state != SimulationState.RUNNING:
            self._publish_current_state()
            return
        tick_start = perf_counter_ns()
        if self._last_tick_start_ns is not None:
            self._timer_interarrival_ms.append(
                (tick_start - self._last_tick_start_ns) * 1e-6
            )
        self._last_tick_start_ns = tick_start
        try:
            sample = self._plant.kinematic_state()
            expected_stamp = self._step_index * 0.002
            if sample.sequence_id != self._step_index or not np.isclose(
                sample.stamp_sec, expected_stamp, atol=2e-12
            ):
                raise RuntimeError("stale command, sequence mismatch, or time reversal")
            phase, reference = self._control_reference(expected_stamp)
            contact = self._plant.contact_summary()
            if self._external_mode == "injected":
                external = _injected_force(expected_stamp)
                injected = external
            elif self._external_mode == "virtual_ft":
                external = contact.force_world
                injected = np.zeros(3)
            else:
                external = np.zeros(3)
                injected = np.zeros(3)
            weights_before = self._controller.network.weights.copy()
            nac_start = perf_counter_ns()
            output = self._controller.step(
                sample.tcp_position,
                sample.tcp_linear_velocity,
                reference,
                external,
                dt=0.002,
                now=expected_stamp,
                dynamics_features=np.concatenate(
                    (sample.arm_position, sample.arm_velocity)
                ),
            )
            self._nac_times_ms.append((perf_counter_ns() - nac_start) * 1e-6)
            if output.state.value == "fault":
                raise RuntimeError(output.fault_reason)
            mapping = self._mapper.map_command(
                output.command,
                sample.translational_jacobian,
                sample.rotational_jacobian,
                sample.tcp_rotation,
                self._plant.desired_tcp_rotation,
                sample.tcp_angular_velocity,
                sample.arm_velocity,
                0.002,
            )
            if mapping.torque_saturated or mapping.rate_saturated:
                self._controller.network.weights[:] = weights_before
            self._saturation_count += int(mapping.torque_saturated)
            self._rate_saturation_count += int(mapping.rate_saturated)
            self._guard(sample, contact, mapping.raw_command)
            physics_start = perf_counter_ns()
            next_sample = self._plant.advance(
                mapping.command, injected_force_world=injected
            )
            self._step_times_ms.append(
                (perf_counter_ns() - physics_start) * 1e-6
            )
            self._step_index += 1
            next_contact = self._plant.contact_summary()
            self._guard(next_sample, next_contact, mapping.raw_command)
            observation_phase, observation_reference = (
                self._observation_reference(next_sample.stamp_sec)
            )
            self._last_output = output
            self._last_mapping = mapping
            self._last_reference = observation_reference
            self._update_metrics(
                next_sample,
                observation_reference,
                output,
                mapping,
                observation_phase,
            )
            self._publish(
                next_sample,
                observation_reference,
                output,
                observation_phase,
            )
            if self._viewer is not None and self._viewer.is_running():
                self._viewer.sync()
            elapsed_ms = (perf_counter_ns() - tick_start) * 1e-6
            self._missed_deadlines += int(elapsed_ms > 2.0)
            if self._duration > 0.0:
                target_steps = int(round(self._duration / 0.002))
                if self._step_index >= target_steps:
                    self._finish()
        except (FloatingPointError, RuntimeError, ValueError) as error:
            self._fault(str(error))
            self._publish_current_state()

    def _update_metrics(self, sample, reference, output, mapping, phase: str) -> None:
        """Accumulate only measured post-step state and accepted commands."""
        impedance_error = float(
            np.linalg.norm(output.model_state.position - sample.tcp_position)
        )
        desired_error = float(
            np.linalg.norm(reference.position - sample.tcp_position)
        )
        self._error_squared_sum += impedance_error**2
        self._tracking_error_max_m = max(
            self._tracking_error_max_m, impedance_error
        )
        self._desired_error_squared_sum += desired_error**2
        self._desired_error_max_m = max(
            self._desired_error_max_m, desired_error
        )
        self._command_force_max_norm_n = max(
            self._command_force_max_norm_n,
            float(np.linalg.norm(output.command)),
        )
        self._arm_torque_max_norm_nm = max(
            self._arm_torque_max_norm_nm,
            float(np.linalg.norm(mapping.command)),
        )
        self._arm_torque_max_abs_nm = max(
            self._arm_torque_max_abs_nm,
            float(np.max(np.abs(mapping.command))),
        )
        self._joint_velocity_max_abs_rad_s = max(
            self._joint_velocity_max_abs_rad_s,
            float(np.max(np.abs(sample.arm_velocity))),
        )
        self._orientation_error_max_rad = max(
            self._orientation_error_max_rad,
            orientation_distance(
                sample.tcp_rotation, self._plant.desired_tcp_rotation
            ),
        )
        contact = self._plant.contact_summary()
        self._contact_force_max_n = max(
            self._contact_force_max_n, contact.contact_force_norm_n
        )
        self._contact_penetration_max_m = max(
            self._contact_penetration_max_m, contact.maximum_penetration_m
        )
        self._unexpected_contact_count += contact.unexpected_contacts
        gripper = self._plant.gripper_state(contact)
        self._gripper_effort_max_n = max(
            self._gripper_effort_max_n, gripper.effort_n
        )
        bilateral = gripper.left_contacts > 0 and gripper.right_contacts > 0
        self._bilateral_contact_samples += int(bilateral)
        object_z = float(sample.object_position[2])
        if self._scenario == "grasp":
            if 2.8 <= sample.stamp_sec < 3.0:
                self._settle_object_z.append(object_z)
            if phase in {"lift", "hold"}:
                self._lift_hold_object_z.append(object_z)
            if phase == "hold":
                self._hold_samples += 1
                self._hold_bilateral_samples += int(bilateral)
                self._hold_object_z.append(object_z)
            if phase == "release":
                self._release_object_z.append(object_z)

    def _guard(self, sample, contact, raw_torque: np.ndarray) -> None:
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
                "robot state, FK, Jacobian, or torque contains NaN or Inf"
            )
        limits = self._plant.joint_limits
        if np.any(sample.all_joint_position < limits[:, 0] - 5e-3) or np.any(
            sample.all_joint_position > limits[:, 1] + 5e-3
        ):
            raise RuntimeError("arm or gripper joint-limit violation")
        if np.any(np.abs(sample.arm_velocity) > 3.5):
            raise RuntimeError("excessive joint velocity")
        if np.any(np.abs(raw_torque) > np.array((280, 280, 280, 54, 54, 54))):
            raise RuntimeError("excessive raw joint torque")
        if np.any(sample.tcp_position < (-0.85, -0.25, 0.10)) or np.any(
            sample.tcp_position > (0.70, 0.90, 1.25)
        ):
            raise RuntimeError("Cartesian workspace violation")
        if orientation_distance(
            sample.tcp_rotation, self._plant.desired_tcp_rotation
        ) > np.deg2rad(35.0):
            raise RuntimeError("orientation limit violation")
        contact_limit = (
            self._grasp_contact_limit_n
            if self._scenario == "grasp"
            else 250.0
        )
        if contact.contact_force_norm_n > contact_limit:
            raise RuntimeError("excessive contact force")
        if self._scenario != "grasp" and contact.unexpected_contacts:
            raise RuntimeError("unexpected robot-environment collision")
        gripper = self._plant.gripper_state(contact)
        if not np.all(
            np.isfinite(
                (
                    gripper.opening_m,
                    gripper.effort_n,
                    gripper.target_opening_m,
                    gripper.contact_force_n,
                    self._plant.gripper.actuator_control(),
                )
            )
        ):
            raise FloatingPointError("gripper actuator feedback is not finite")
        if abs(sample.all_joint_position[6] - sample.all_joint_position[10]) > 0.02:
            raise RuntimeError("gripper actuator/coupling failure")

    def _publish(self, sample, reference, output, phase: str) -> None:
        stamp = _stamp(sample.stamp_sec)
        self._clock_pub.publish(Clock(clock=stamp))
        self._publish_joint_state(sample, stamp)
        desired_pose = self._pose(reference.position, sample.tcp_rotation, stamp)
        impedance_pose = self._pose(
            output.model_state.position, sample.tcp_rotation, stamp
        )
        actual_pose = self._pose(sample.tcp_position, sample.tcp_rotation, stamp)
        self._desired_pose_pub.publish(desired_pose)
        self._impedance_pose_pub.publish(impedance_pose)
        self._actual_pose_pub.publish(actual_pose)
        self._publish_wrenches(sample, output, stamp)
        contact = self._plant.contact_summary()
        gripper = self._plant.gripper_state(contact)
        self._gripper_state_pub.publish(
            Float64MultiArray(
                data=[
                    gripper.opening_m,
                    gripper.effort_n,
                    gripper.target_opening_m,
                    float(gripper.left_contacts),
                    float(gripper.right_contacts),
                    gripper.contact_force_n,
                    float(gripper.reached_goal),
                    float(gripper.stalled),
                    float(gripper.stopped),
                ]
            )
        )
        self._gripper_actuator_pub.publish(
            Float64(data=self._plant.gripper.actuator_control())
        )
        if self._step_index % self._visualization_stride == 0:
            self._append_paths(desired_pose, impedance_pose, actual_pose)
            self._scene_marker_pub.publish(
                self._scene_markers(sample, output, contact, stamp)
            )
            self._contact_marker_pub.publish(
                self._contact_markers(contact, stamp)
            )
            self._publish_diagnostics(sample, contact, gripper, phase, stamp)

    def _publish_current_state(self) -> None:
        sample = self._plant.kinematic_state()
        stamp = _stamp(sample.stamp_sec)
        self._clock_pub.publish(Clock(clock=stamp))
        self._publish_joint_state(sample, stamp)
        self._publish_diagnostics(
            sample,
            self._plant.contact_summary(),
            self._plant.gripper_state(),
            self._state.value,
            stamp,
        )

    def _publish_joint_state(self, sample, stamp: Time) -> None:
        message = JointState()
        message.header.stamp = stamp
        message.header.frame_id = self._frame
        message.name = list(self._plant.joint_names)
        message.position = sample.all_joint_position.tolist()
        message.velocity = sample.all_joint_velocity.tolist()
        effort = np.zeros(len(message.name))
        effort[:6] = self._plant.data.qfrc_actuator[:6]
        message.effort = effort.tolist()
        self._joint_pub.publish(message)

    def _pose(
        self, position: np.ndarray, rotation: np.ndarray, stamp: Time
    ) -> PoseStamped:
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = self._frame
        message.pose.position.x = float(position[0])
        message.pose.position.y = float(position[1])
        message.pose.position.z = float(position[2])
        qx, qy, qz, qw = _quaternion_from_matrix(rotation)
        message.pose.orientation.x = qx
        message.pose.orientation.y = qy
        message.pose.orientation.z = qz
        message.pose.orientation.w = qw
        return message

    def _wrench(
        self, force: np.ndarray, torque: np.ndarray, stamp: Time
    ) -> WrenchStamped:
        message = WrenchStamped()
        message.header.stamp = stamp
        message.header.frame_id = self._frame
        message.wrench.force.x, message.wrench.force.y, message.wrench.force.z = (
            force.tolist()
        )
        message.wrench.torque.x, message.wrench.torque.y, message.wrench.torque.z = (
            torque.tolist()
        )
        return message

    def _publish_wrenches(self, sample, output, stamp: Time) -> None:
        zeros = np.zeros(3)
        self._command_pub.publish(self._wrench(output.command, zeros, stamp))
        self._nn_pub.publish(self._wrench(output.neural_estimate, zeros, stamp))
        error = output.model_state.position - sample.tcp_position
        error_message = Vector3Stamped()
        error_message.header.stamp = stamp
        error_message.header.frame_id = self._frame
        (
            error_message.vector.x,
            error_message.vector.y,
            error_message.vector.z,
        ) = error.tolist()
        self._error_pub.publish(error_message)
        raw = self._plant.wrist_wrench_raw()
        self._wrist_pub.publish(
            self._wrench(raw.force_world, raw.torque_world, stamp)
        )
        contact = self._plant.contact_summary()
        contact_message = self._wrench(
            contact.force_world, contact.torque_world_at_tcp, stamp
        )
        self._contact_wrench_pub.publish(contact_message)
        self._legacy_wrench_pub.publish(contact_message)

    def _append_paths(self, desired, impedance, actual) -> None:
        for path, pose, publisher in (
            (self._path_desired, desired, self._desired_path_pub),
            (self._path_impedance, impedance, self._impedance_path_pub),
            (self._path_actual, actual, self._actual_path_pub),
        ):
            path.header = pose.header
            path.poses.append(pose)
            if len(path.poses) > 1200:
                path.poses.pop(0)
            publisher.publish(path)

    def _scene_markers(self, sample, output, contact, stamp) -> MarkerArray:
        markers = MarkerArray()
        table = Marker()
        table.header.frame_id = self._frame
        table.header.stamp = stamp
        table.ns = "scene"
        table.id = 0
        table.type = Marker.CUBE
        table.action = Marker.ADD
        table.pose.position.y = 0.45
        table.pose.position.z = 0.15
        table.pose.orientation.w = 1.0
        table.scale.x, table.scale.y, table.scale.z = 0.90, 0.70, 0.10
        table.color.r, table.color.g, table.color.b, table.color.a = (
            0.72,
            0.46,
            0.22,
            1.0,
        )
        markers.markers.append(table)
        object_marker = Marker()
        object_marker.header = table.header
        object_marker.ns = "scene"
        object_marker.id = 1
        object_marker.type = Marker.CUBE
        object_marker.action = Marker.ADD
        object_marker.pose.position.x = float(sample.object_position[0])
        object_marker.pose.position.y = float(sample.object_position[1])
        object_marker.pose.position.z = float(sample.object_position[2])
        qx, qy, qz, qw = _quaternion_from_matrix(sample.object_rotation)
        object_marker.pose.orientation.x = qx
        object_marker.pose.orientation.y = qy
        object_marker.pose.orientation.z = qz
        object_marker.pose.orientation.w = qw
        object_marker.scale.x, object_marker.scale.y, object_marker.scale.z = (
            0.04,
            0.04,
            0.08,
        )
        object_marker.color.r = 0.90
        object_marker.color.g = 0.72
        object_marker.color.b = 0.12
        object_marker.color.a = 1.0
        markers.markers.append(object_marker)
        colors = (
            ("desired", 2, self._last_reference.position, (0.0, 1.0, 0.0)),
            ("impedance", 3, output.model_state.position, (0.0, 0.3, 1.0)),
            ("actual", 4, sample.tcp_position, (1.0, 0.0, 0.0)),
        )
        for namespace, marker_id, position, color in colors:
            marker = Marker()
            marker.header = table.header
            marker.ns = namespace
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = float(position[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.018
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1.0
            markers.markers.append(marker)
        markers.markers.extend(
            self._arrow_markers(sample, output, table.header)
        )
        gripper = self._plant.gripper_state(contact)
        status = Marker()
        status.header = table.header
        status.ns = "simulation_status"
        status.id = 20
        status.type = Marker.TEXT_VIEW_FACING
        status.action = Marker.ADD
        status.pose.position.x = 0.0
        status.pose.position.y = 0.45
        status.pose.position.z = 1.05
        status.pose.orientation.w = 1.0
        status.scale.z = 0.020
        status.color.r = 0.95
        status.color.g = 0.95
        status.color.b = 0.95
        status.color.a = 1.0
        status.text = (
            f"plant: {self._state.value}\n"
            f"NAC: {'adaptive' if self._adaptation_enabled else 'frozen'}\n"
            f"grip: {1000.0 * gripper.opening_m:.1f} mm / "
            f"{gripper.effort_n:.2f} N\n"
            f"contacts: {gripper.left_contacts} / {gripper.right_contacts}"
        )
        markers.markers.append(status)
        return markers

    def _arrow_markers(self, sample, output, header) -> list[Marker]:
        contact = self._plant.contact_summary()
        arrows = []
        for marker_id, namespace, vector, color in (
            (10, "nac_command", output.command, (1.0, 0.45, 0.0)),
            (11, "contact_wrench", contact.force_world, (0.65, 0.1, 0.8)),
        ):
            marker = Marker()
            marker.header = header
            marker.ns = namespace
            marker.id = marker_id
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            start = sample.tcp_position
            end = start + 0.0025 * vector
            marker.points = [
                Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))
                for v in (start, end)
            ]
            marker.scale.x, marker.scale.y, marker.scale.z = 0.008, 0.014, 0.018
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1.0
            arrows.append(marker)
        return arrows

    def _contact_markers(self, contact, stamp: Time) -> MarkerArray:
        markers = MarkerArray()
        for index, (position, normal, force) in enumerate(
            zip(
                contact.positions_world,
                contact.normals_world,
                contact.forces_world,
            )
        ):
            point = Marker()
            point.header.frame_id = self._frame
            point.header.stamp = stamp
            point.ns = "contact_points"
            point.id = 2 * index
            point.type = Marker.SPHERE
            point.action = Marker.ADD
            point.pose.position.x = float(position[0])
            point.pose.position.y = float(position[1])
            point.pose.position.z = float(position[2])
            point.pose.orientation.w = 1.0
            point.scale.x = point.scale.y = point.scale.z = 0.010
            point.color.r, point.color.g, point.color.b, point.color.a = (
                0.65,
                0.1,
                0.8,
                1.0,
            )
            markers.markers.append(point)
            arrow = Marker()
            arrow.header = point.header
            arrow.ns = "contact_normals"
            arrow.id = 2 * index + 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            direction = normal * min(0.08, 0.002 * np.linalg.norm(force))
            arrow.points = [
                Point(x=float(position[0]), y=float(position[1]), z=float(position[2])),
                Point(
                    x=float(position[0] + direction[0]),
                    y=float(position[1] + direction[1]),
                    z=float(position[2] + direction[2]),
                ),
            ]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.004, 0.008, 0.010
            arrow.color = point.color
            markers.markers.append(arrow)
        return markers

    def _publish_diagnostics(self, sample, contact, gripper, phase, stamp) -> None:
        array = DiagnosticArray()
        array.header.stamp = stamp
        status = DiagnosticStatus()
        status.name = "mujoco_ur5e_nac"
        status.hardware_id = "simulation"
        status.level = (
            DiagnosticStatus.ERROR
            if self._state == SimulationState.FAULT
            else DiagnosticStatus.OK
        )
        status.message = f"{self._state.value}: {self._reason}"
        values = {
            "state": self._state.value,
            "phase": str(phase),
            "sequence_id": str(sample.sequence_id),
            "sim_time_sec": f"{sample.stamp_sec:.6f}",
            "control_period_sec": "0.002",
            "mujoco_timestep_sec": "0.0005",
            "substeps": "4",
            "missed_wall_deadlines": str(self._missed_deadlines),
            "callback_overrun_count": str(self._missed_deadlines),
            "torque_saturation_count": str(self._saturation_count),
            "external_wrench_mode": self._external_mode,
            "adaptation_enabled": str(self._adaptation_enabled),
            "gripper_opening_m": f"{gripper.opening_m:.6f}",
            "gripper_effort_n": f"{gripper.effort_n:.6f}",
            "left_finger_contacts": str(gripper.left_contacts),
            "right_finger_contacts": str(gripper.right_contacts),
            "contact_force_n": f"{contact.contact_force_norm_n:.6f}",
            "unexpected_contacts": str(contact.unexpected_contacts),
        }
        status.values = [KeyValue(key=key, value=value) for key, value in values.items()]
        array.status = [status]
        self._diagnostics_pub.publish(array)
        contact_array = DiagnosticArray()
        contact_array.header = array.header
        contact_status = DiagnosticStatus()
        contact_status.name = "mujoco_contacts"
        contact_status.hardware_id = "simulation"
        contact_status.level = DiagnosticStatus.OK
        contact_status.message = "environment-on-robot contact-only wrench"
        contact_status.values = [
            KeyValue(key="frame", value=self._frame),
            KeyValue(key="application_point", value="gripper_pinch/TCP"),
            KeyValue(key="sign", value="environment-on-robot"),
            KeyValue(
                key="maximum_penetration_m",
                value=f"{contact.maximum_penetration_m:.9f}",
            ),
        ]
        contact_array.status = [contact_status]
        self._contact_diagnostics_pub.publish(contact_array)

    def _finish(self) -> None:
        self._stop("duration complete")
        wall_duration = (perf_counter_ns() - self._wall_origin_ns) * 1e-9
        steps = max(self._step_index, 1)
        metrics = {
            "scenario": self._scenario,
            "trajectory": self._trajectory_name,
            "adaptation_enabled": self._adaptation_enabled,
            "external_wrench_mode": self._external_mode,
            "state": self._state.value,
            "fault_reason": "",
            "control_period_sec": 0.002,
            "mujoco_timestep_sec": 0.0005,
            "substeps_per_control": 4,
            "simulated_duration_sec": float(self._plant.data.time),
            "wall_duration_sec": wall_duration,
            "real_time_factor": float(self._plant.data.time / wall_duration),
            "control_steps": self._step_index,
            "mujoco_steps": self._step_index * 4,
            "observed_control_step_rate_hz": self._step_index / wall_duration,
            "missed_wall_deadlines": self._missed_deadlines,
            "callback_overrun_count": self._missed_deadlines,
            "timer_interarrival_median_ms": (
                float(np.median(self._timer_interarrival_ms))
                if self._timer_interarrival_ms
                else 0.0
            ),
            "timer_interarrival_p95_ms": (
                float(np.percentile(self._timer_interarrival_ms, 95))
                if self._timer_interarrival_ms
                else 0.0
            ),
            "timer_interarrival_p99_ms": (
                float(np.percentile(self._timer_interarrival_ms, 99))
                if self._timer_interarrival_ms
                else 0.0
            ),
            "timer_interarrival_max_ms": (
                float(np.max(self._timer_interarrival_ms))
                if self._timer_interarrival_ms
                else 0.0
            ),
            "nac_time_median_ms": float(np.median(self._nac_times_ms)),
            "nac_time_p95_ms": float(np.percentile(self._nac_times_ms, 95)),
            "nac_time_p99_ms": float(np.percentile(self._nac_times_ms, 99)),
            "mujoco_step_time_median_ms": float(np.median(self._step_times_ms)),
            "mujoco_step_time_p95_ms": float(np.percentile(self._step_times_ms, 95)),
            "mujoco_step_time_p99_ms": float(np.percentile(self._step_times_ms, 99)),
            "impedance_tracking_rmse_m": float(
                np.sqrt(self._error_squared_sum / steps)
            ),
            "impedance_tracking_max_error_m": self._tracking_error_max_m,
            "desired_tracking_rmse_m": float(
                np.sqrt(self._desired_error_squared_sum / steps)
            ),
            "desired_tracking_max_error_m": self._desired_error_max_m,
            "command_force_max_norm_n": self._command_force_max_norm_n,
            "arm_torque_max_norm_nm": self._arm_torque_max_norm_nm,
            "arm_torque_max_abs_nm": self._arm_torque_max_abs_nm,
            "joint_velocity_max_abs_rad_s": (
                self._joint_velocity_max_abs_rad_s
            ),
            "orientation_error_max_rad": self._orientation_error_max_rad,
            "contact_force_max_n": self._contact_force_max_n,
            "contact_penetration_max_m": self._contact_penetration_max_m,
            "unexpected_contact_count": self._unexpected_contact_count,
            "maximum_gripper_effort_n": self._gripper_effort_max_n,
            "torque_saturation_count": self._saturation_count,
            "torque_saturation_ratio": self._saturation_count / steps,
            "torque_rate_saturation_count": self._rate_saturation_count,
            "final_weight_norm": self._controller.network.weight_norm,
            "hard_real_time_guarantee": False,
        }
        if self._scenario == "grasp":
            metrics.update(self._grasp_metrics())
        self.get_logger().info("MuJoCo demo metrics: " + json.dumps(metrics))
        if self._metrics_path:
            output = Path(self._metrics_path).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(metrics, indent=2) + "\n")
        if self._shutdown_when_complete:
            self._timer.cancel()
            # Shutting down the global context from inside a timer callback can
            # deadlock an executor.  Let main() leave its spin loop after this
            # callback returns, then perform node/context cleanup in one place.
            self._exit_requested = True

    def _grasp_metrics(self) -> dict[str, float | int | bool]:
        """Derive grasp/lift/hold acceptance from accumulated MuJoCo state."""
        initial_z = (
            float(np.median(self._settle_object_z))
            if self._settle_object_z
            else self._initial_object_z
        )
        maximum_z = (
            max(self._lift_hold_object_z)
            if self._lift_hold_object_z
            else initial_z
        )
        hold_min_z = (
            min(self._hold_object_z) if self._hold_object_z else maximum_z
        )
        returned_z = (
            float(np.median(self._release_object_z))
            if self._release_object_z
            else initial_z
        )
        hold_duration = self._hold_samples * 0.002
        hold_ratio = (
            self._hold_bilateral_samples / self._hold_samples
            if self._hold_samples
            else 0.0
        )
        lift_height = maximum_z - initial_z
        hold_drop = maximum_z - hold_min_z
        returned_error = abs(returned_z - initial_z)
        bilateral_duration = self._bilateral_contact_samples * 0.002
        success = bool(
            bilateral_duration >= 0.1
            and lift_height >= 0.05
            and hold_duration >= 2.0
            and hold_drop <= 0.005
            and hold_ratio >= 0.90
            and self._contact_penetration_max_m <= 0.002
            and self._gripper_effort_max_n <= self._grasp_effort_n + 1e-9
            and self._contact_force_max_n <= self._grasp_contact_limit_n
            and self._unexpected_contact_count == 0
            and returned_error <= 0.01
        )
        return {
            "grasp_success": success,
            "bilateral_contact_duration_sec": bilateral_duration,
            "hold_duration_sec": hold_duration,
            "hold_bilateral_contact_ratio": hold_ratio,
            "object_lift_height_m": lift_height,
            "hold_drop_m": hold_drop,
            "returned_height_error_m": returned_error,
            "configured_contact_force_limit_n": (
                self._grasp_contact_limit_n
            ),
            "configured_gripper_effort_limit_n": self._grasp_effort_n,
            "solver_warning_count": 0,
        }

    @property
    def exit_requested(self) -> bool:
        """Return whether a completed finite run should terminate the process."""
        return self._exit_requested

    def destroy_node(self) -> bool:
        """Close the passive viewer before releasing its shared model/data."""
        if hasattr(self, "_timer"):
            self._timer.cancel()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            # MuJoCo's Linux passive viewer signals a daemon UI thread but
            # does not expose a join.  Keep model/data alive for its teardown.
            sleep(0.2)
        return super().destroy_node()


def main(args=None) -> None:
    """Run the single-owner MuJoCo plant/control/telemetry node."""
    rclpy.init(args=args)
    node = None
    try:
        node = MujocoUR5ePlantNode()
        while rclpy.ok() and not node.exit_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            print(f"mujoco_ur5e_plant startup failed: {error}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
