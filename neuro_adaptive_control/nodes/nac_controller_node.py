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

"""ROS 2 wrapper for the pure NumPy 3D neuro-adaptive controller."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Callable, Optional, Tuple

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import (
    PoseStamped,
    TwistStamped,
    Vector3Stamped,
    WrenchStamped,
)
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

from neuro_adaptive_control.core.impedance_model import (
    CartesianImpedanceModel,
    ImpedanceParameters,
)
from neuro_adaptive_control.core.neuro_adaptive_controller import (
    NACParameters,
    NeuroAdaptiveController,
)
from neuro_adaptive_control.core.rbf_network import RBFNetwork
from neuro_adaptive_control.core.references import ReferenceSample, make_reference
from neuro_adaptive_control.core.safety import (
    ControllerState,
    SafetyConfig,
    SafetySupervisor,
)


StampKey = Tuple[int, int]


def _finite_positive(value, name: str) -> float:
    """Return a finite positive float or raise a parameter error."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric.") from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _stamp_key(message) -> StampKey:
    return int(message.header.stamp.sec), int(message.header.stamp.nanosec)


def _stamp_nanoseconds(key: StampKey) -> int:
    return key[0] * 1_000_000_000 + key[1]


def _pose_position(message: PoseStamped) -> np.ndarray:
    return np.array(
        [
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ],
        dtype=float,
    )


def _twist_velocity(message: TwistStamped) -> np.ndarray:
    return np.array(
        [
            message.twist.linear.x,
            message.twist.linear.y,
            message.twist.linear.z,
        ],
        dtype=float,
    )


def _wrench_force(message: WrenchStamped) -> np.ndarray:
    return np.array(
        [
            message.wrench.force.x,
            message.wrench.force.y,
            message.wrench.force.z,
        ],
        dtype=float,
    )


class NACControllerNode(Node):
    """Pair stamped state topics and publish safe Cartesian wrench commands."""

    def __init__(self) -> None:
        super().__init__("nac_controller")
        self._declare_parameters()
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.duration_sec = _finite_positive(
            self.get_parameter("duration_sec").value, "duration_sec"
        )
        rate_hz = _finite_positive(
            self.get_parameter("control_rate_hz").value,
            "control_rate_hz",
        )
        if not self.frame_id:
            raise ValueError("frame_id must not be empty.")
        self.dt_ns = int(round(1e9 / rate_hz))
        self.dt = float(self.dt_ns) * 1e-9
        self.target_rate_hz = 1.0 / self.dt
        self.cache_size = int(self.get_parameter("safety.cache_size").value)
        if self.cache_size < 3:
            raise ValueError("safety.cache_size must be at least 3.")
        telemetry_rate = _finite_positive(
            self.get_parameter("telemetry.rate_hz").value,
            "telemetry.rate_hz",
        )
        self.telemetry_decimation = max(
            1,
            int(round(self.target_rate_hz / telemetry_rate)),
        )
        diagnostics_rate = _finite_positive(
            self.get_parameter("diagnostics.rate_hz").value,
            "diagnostics.rate_hz",
        )

        self.controller = self._build_controller()
        self.reference = self._build_reference()
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.node_watchdog_timeout = (
            self.controller.safety.config.watchdog_timeout
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=4,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        def topic(name: str) -> str:
            return str(self.get_parameter(name).value)

        self.command_publisher = self.create_publisher(
            WrenchStamped, topic("topics.command"), qos
        )
        self.desired_pose_publisher = self.create_publisher(
            PoseStamped, topic("topics.desired_pose"), qos
        )
        self.desired_twist_publisher = self.create_publisher(
            TwistStamped, topic("topics.desired_twist"), qos
        )
        self.impedance_pose_publisher = self.create_publisher(
            PoseStamped, topic("topics.impedance_pose"), qos
        )
        self.impedance_twist_publisher = self.create_publisher(
            TwistStamped, topic("topics.impedance_twist"), qos
        )
        self.actual_pose_publisher = self.create_publisher(
            PoseStamped, topic("topics.actual_pose"), qos
        )
        self.nn_publisher = self.create_publisher(
            WrenchStamped, topic("topics.nn_estimate"), qos
        )
        self.error_publisher = self.create_publisher(
            Vector3Stamped, topic("topics.tracking_error"), qos
        )
        self.diagnostics_publisher = self.create_publisher(
            DiagnosticArray, topic("topics.diagnostics"), 10
        )
        self.pose_subscription = self.create_subscription(
            PoseStamped, topic("topics.pose_input"), self._on_pose, qos
        )
        self.twist_subscription = self.create_subscription(
            TwistStamped, topic("topics.twist_input"), self._on_twist, qos
        )
        self.wrench_subscription = self.create_subscription(
            WrenchStamped, topic("topics.wrench_input"), self._on_wrench, qos
        )
        self.start_service = self.create_service(Trigger, "~/start", self._start)
        self.stop_service = self.create_service(Trigger, "~/stop", self._stop)
        self.reset_service = self.create_service(Trigger, "~/reset", self._reset)

        self.pose_cache: OrderedDict[StampKey, PoseStamped] = OrderedDict()
        self.twist_cache: OrderedDict[StampKey, TwistStamped] = OrderedDict()
        self.wrench_cache: OrderedDict[StampKey, WrenchStamped] = OrderedDict()
        self.start_stamp_ns: Optional[int] = None
        self.last_processed_stamp_ns: Optional[int] = None
        self.last_bundle_stamp = None
        self.last_receive_steady: Optional[float] = None
        self.last_actual_position = np.zeros(3, dtype=float)
        self.last_actual_velocity = np.zeros(3, dtype=float)
        self.processed_steps = 0
        self.saturation_count = 0
        self.stamp_mismatch_count = 0
        self.last_output = None
        self.wall_start = time.monotonic()
        self._duration_stopped = False

        watchdog_period = min(0.05, 0.5 * self.node_watchdog_timeout)
        self.watchdog_timer = self.create_timer(
            max(watchdog_period, 0.005), self._watchdog_tick
        )
        self.diagnostic_timer = self.create_timer(
            1.0 / diagnostics_rate, self._publish_diagnostics
        )
        self.get_logger().info(
            "NAC controller ready: "
            f"fixed_dt={self.dt:.9f}s target_rate={self.target_rate_hz:.3f}Hz "
            f"adaptation={self.controller.network.adaptation_enabled}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("control_rate_hz", 500.0)
        self.declare_parameter("duration_sec", 12.0)
        self.declare_parameter("auto_start", True)
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("trajectory.type", "circle")
        self.declare_parameter("trajectory.center", [0.0, 0.0, 0.0])
        self.declare_parameter("trajectory.frequency", 0.20)
        self.declare_parameter("trajectory.radius", 0.08)
        self.declare_parameter("trajectory.line_length", 0.16)
        self.declare_parameter("trajectory.line_axis", [1.0, 0.0, 0.0])
        self.declare_parameter("trajectory.figure8_width", 0.16)
        self.declare_parameter("trajectory.figure8_height", 0.10)
        self.declare_parameter("impedance.mass", [1.0, 1.0, 1.0])
        self.declare_parameter("impedance.damping", [12.0, 12.0, 12.0])
        self.declare_parameter("impedance.stiffness", [35.0, 35.0, 35.0])
        self.declare_parameter("impedance.external_gain", [1.0, 1.0, 1.0])
        self.declare_parameter("nac.lambda_gain", [7.0, 7.0, 7.0])
        self.declare_parameter("nac.feedback_gain", [18.0, 18.0, 20.0])
        self.declare_parameter("nac.robust_gain", [0.04, 0.04, 0.04])
        self.declare_parameter("nac.robust_bias", 1.5)
        self.declare_parameter("rbf.num_basis", 31)
        self.declare_parameter("rbf.width", 2.5)
        self.declare_parameter(
            "rbf.input_scale",
            [
                0.10,
                0.10,
                0.10,
                0.50,
                0.50,
                0.50,
                0.10,
                0.10,
                0.10,
                0.50,
                0.50,
                0.50,
                3.00,
                3.00,
                3.00,
                0.10,
                0.10,
                0.10,
                0.50,
                0.50,
                0.50,
            ],
        )
        self.declare_parameter("rbf.feature_clip", 3.0)
        self.declare_parameter("rbf.learning_rate", 5.0)
        self.declare_parameter("rbf.leakage", 0.01)
        self.declare_parameter("rbf.weight_limit", 80.0)
        self.declare_parameter("rbf.seed", 7)
        self.declare_parameter("rbf.adaptation_enabled", True)
        self.declare_parameter("safety.command_limits", [40.0, 40.0, 40.0])
        self.declare_parameter("safety.command_norm_limit", 55.0)
        self.declare_parameter("safety.watchdog_timeout", 0.10)
        self.declare_parameter("safety.maximum_dt", 0.01)
        self.declare_parameter("safety.cache_size", 8)
        self.declare_parameter("telemetry.rate_hz", 100.0)
        self.declare_parameter("diagnostics.rate_hz", 20.0)
        self.declare_parameter("topics.pose_input", "demo/cartesian_pose")
        self.declare_parameter("topics.twist_input", "demo/cartesian_twist")
        self.declare_parameter(
            "topics.wrench_input", "demo/applied_external_wrench"
        )
        self.declare_parameter("topics.command", "nac/wrench_command")
        self.declare_parameter("topics.desired_pose", "nac/desired_pose")
        self.declare_parameter("topics.desired_twist", "nac/desired_twist")
        self.declare_parameter("topics.impedance_pose", "nac/impedance_pose")
        self.declare_parameter(
            "topics.impedance_twist", "nac/impedance_twist"
        )
        self.declare_parameter("topics.actual_pose", "nac/actual_pose")
        self.declare_parameter("topics.nn_estimate", "nac/nn_estimate")
        self.declare_parameter(
            "topics.tracking_error", "nac/tracking_error"
        )
        self.declare_parameter("topics.diagnostics", "diagnostics")

    def _array_parameter(self, name: str, size: int) -> np.ndarray:
        value = np.asarray(self.get_parameter(name).value, dtype=float)
        if value.shape != (size,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a finite vector of length {size}.")
        return value

    def _build_controller(self) -> NeuroAdaptiveController:
        impedance = ImpedanceParameters.diagonal(
            self._array_parameter("impedance.mass", 3),
            self._array_parameter("impedance.damping", 3),
            self._array_parameter("impedance.stiffness", 3),
            self._array_parameter("impedance.external_gain", 3),
        )
        model = CartesianImpedanceModel(impedance)
        network = RBFNetwork(
            input_dim=21,
            output_dim=3,
            num_basis=int(self.get_parameter("rbf.num_basis").value),
            widths=float(self.get_parameter("rbf.width").value),
            input_scale=self._array_parameter("rbf.input_scale", 21),
            feature_clip=float(self.get_parameter("rbf.feature_clip").value),
            learning_rate=float(self.get_parameter("rbf.learning_rate").value),
            leakage=float(self.get_parameter("rbf.leakage").value),
            weight_limit=float(self.get_parameter("rbf.weight_limit").value),
            seed=int(self.get_parameter("rbf.seed").value),
            adaptation_enabled=bool(
                self.get_parameter("rbf.adaptation_enabled").value
            ),
        )
        nac = NACParameters.diagonal(
            self._array_parameter("nac.lambda_gain", 3),
            self._array_parameter("nac.feedback_gain", 3),
            self._array_parameter("nac.robust_gain", 3),
            float(self.get_parameter("nac.robust_bias").value),
        )
        safety = SafetySupervisor(
            SafetyConfig(
                command_limits=self._array_parameter("safety.command_limits", 3),
                command_norm_limit=float(
                    self.get_parameter("safety.command_norm_limit").value
                ),
                watchdog_timeout=float(
                    self.get_parameter("safety.watchdog_timeout").value
                ),
                maximum_dt=float(
                    self.get_parameter("safety.maximum_dt").value
                ),
            )
        )
        return NeuroAdaptiveController(model, network, nac, safety)

    def _build_reference(self):
        return make_reference(
            str(self.get_parameter("trajectory.type").value),
            center=self._array_parameter("trajectory.center", 3),
            frequency=float(self.get_parameter("trajectory.frequency").value),
            radius=float(self.get_parameter("trajectory.radius").value),
            line_length=float(
                self.get_parameter("trajectory.line_length").value
            ),
            line_axis=self._array_parameter("trajectory.line_axis", 3),
            figure8_width=float(
                self.get_parameter("trajectory.figure8_width").value
            ),
            figure8_height=float(
                self.get_parameter("trajectory.figure8_height").value
            ),
        )

    def _steady_seconds(self) -> float:
        return 1e-9 * float(self.steady_clock.now().nanoseconds)

    def _on_pose(self, message: PoseStamped) -> None:
        self._store_message(self.pose_cache, message, _pose_position)

    def _on_twist(self, message: TwistStamped) -> None:
        self._store_message(self.twist_cache, message, _twist_velocity)

    def _on_wrench(self, message: WrenchStamped) -> None:
        self._store_message(self.wrench_cache, message, _wrench_force)

    def _store_message(
        self,
        cache: OrderedDict,
        message,
        extractor: Callable,
    ) -> None:
        if message.header.frame_id != self.frame_id:
            self._fault("input frame_id mismatch")
            return
        values = extractor(message)
        if not np.all(np.isfinite(values)):
            self._fault("input state contains NaN or Inf")
            return
        key = _stamp_key(message)
        stamp_ns = _stamp_nanoseconds(key)
        if (
            self.last_processed_stamp_ns is not None
            and stamp_ns <= self.last_processed_stamp_ns
        ):
            self.stamp_mismatch_count += 1
            return
        cache[key] = message
        cache.move_to_end(key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)
            self.stamp_mismatch_count += 1
        self.last_receive_steady = self._steady_seconds()
        self.last_bundle_stamp = message.header.stamp
        self._try_process(key)

    def _try_process(self, key: StampKey) -> None:
        if not all(
            key in cache
            for cache in (self.pose_cache, self.twist_cache, self.wrench_cache)
        ):
            return
        pose = self.pose_cache.pop(key)
        twist = self.twist_cache.pop(key)
        wrench = self.wrench_cache.pop(key)
        stamp_ns = _stamp_nanoseconds(key)
        if self.last_processed_stamp_ns is not None:
            delta = stamp_ns - self.last_processed_stamp_ns
            if delta != self.dt_ns:
                self.stamp_mismatch_count += 1
                self._fault(
                    f"state stamp increment {delta}ns does not match {self.dt_ns}ns"
                )
                return
        position = _pose_position(pose)
        velocity = _twist_velocity(twist)
        external = _wrench_force(wrench)
        self.last_actual_position = position.copy()
        self.last_actual_velocity = velocity.copy()
        if self.start_stamp_ns is None:
            self.start_stamp_ns = stamp_ns
            restart_after_initialization = (
                self.auto_start
                or self.controller.state == ControllerState.RUNNING
            )
            self.controller.reset(position, velocity)
            if restart_after_initialization:
                self.controller.start(self._steady_seconds())
        elapsed = 1e-9 * float(stamp_ns - self.start_stamp_ns)
        sample = self.reference.evaluate(elapsed)
        if elapsed + 0.5 * self.dt >= self.duration_sec:
            if self.controller.state == ControllerState.RUNNING:
                self.controller.stop("configured duration complete")
            output = self.controller.step(
                position,
                velocity,
                sample,
                external,
                dt=self.dt,
                now=self._steady_seconds(),
            )
            self._duration_stopped = True
        elif self.controller.state == ControllerState.RUNNING:
            output = self.controller.step(
                position,
                velocity,
                sample,
                external,
                dt=self.dt,
                now=self._steady_seconds(),
            )
        else:
            output = self.controller._zero_output()

        self.last_processed_stamp_ns = stamp_ns
        self.processed_steps += 1
        self.last_output = output
        if output.saturated:
            self.saturation_count += 1
        self._publish_command(pose.header.stamp, output.command)
        if (
            self.processed_steps % self.telemetry_decimation == 0
            or self.processed_steps == 1
            or self._duration_stopped
            or output.state == ControllerState.FAULT
        ):
            self._publish_telemetry(
                pose.header.stamp, position, sample, output
            )
        if output.state == ControllerState.FAULT:
            self._publish_diagnostics()

    def _publish_command(self, stamp, command: np.ndarray) -> None:
        message = WrenchStamped()
        message.header.stamp = stamp
        message.header.frame_id = self.frame_id
        message.wrench.force.x = float(command[0])
        message.wrench.force.y = float(command[1])
        message.wrench.force.z = float(command[2])
        self.command_publisher.publish(message)

    def _pose_message(self, stamp, position: np.ndarray) -> PoseStamped:
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = self.frame_id
        message.pose.position.x = float(position[0])
        message.pose.position.y = float(position[1])
        message.pose.position.z = float(position[2])
        message.pose.orientation.w = 1.0
        return message

    def _twist_message(self, stamp, velocity: np.ndarray) -> TwistStamped:
        message = TwistStamped()
        message.header.stamp = stamp
        message.header.frame_id = self.frame_id
        message.twist.linear.x = float(velocity[0])
        message.twist.linear.y = float(velocity[1])
        message.twist.linear.z = float(velocity[2])
        return message

    def _publish_telemetry(
        self,
        stamp,
        actual: np.ndarray,
        sample: ReferenceSample,
        output,
    ) -> None:
        self.desired_pose_publisher.publish(
            self._pose_message(stamp, sample.position)
        )
        self.desired_twist_publisher.publish(
            self._twist_message(stamp, sample.velocity)
        )
        self.impedance_pose_publisher.publish(
            self._pose_message(stamp, output.model_state.position)
        )
        self.impedance_twist_publisher.publish(
            self._twist_message(stamp, output.model_state.velocity)
        )
        self.actual_pose_publisher.publish(self._pose_message(stamp, actual))
        nn_message = WrenchStamped()
        nn_message.header.stamp = stamp
        nn_message.header.frame_id = self.frame_id
        nn_message.wrench.force.x = float(output.neural_estimate[0])
        nn_message.wrench.force.y = float(output.neural_estimate[1])
        nn_message.wrench.force.z = float(output.neural_estimate[2])
        self.nn_publisher.publish(nn_message)
        error_message = Vector3Stamped()
        error_message.header.stamp = stamp
        error_message.header.frame_id = self.frame_id
        error_message.vector.x = float(output.model_error[0])
        error_message.vector.y = float(output.model_error[1])
        error_message.vector.z = float(output.model_error[2])
        self.error_publisher.publish(error_message)

    def _publish_zero(self) -> None:
        if self.last_bundle_stamp is None or not rclpy.ok():
            return
        self._publish_command(self.last_bundle_stamp, np.zeros(3, dtype=float))

    def _watchdog_tick(self) -> None:
        if self.controller.state != ControllerState.RUNNING:
            return
        now = self._steady_seconds()
        self.controller.safety.tick(now)
        if self.last_receive_steady is not None:
            age = now - self.last_receive_steady
            if age > self.node_watchdog_timeout:
                self.controller.safety.trigger_fault("ROS state watchdog expired")
        if self.controller.state == ControllerState.FAULT:
            self._publish_zero()
            self._publish_diagnostics()

    def _fault(self, reason: str) -> None:
        self.controller.safety.trigger_fault(reason)
        self.get_logger().error(self.controller.safety.reason)
        self._publish_zero()
        self._publish_diagnostics()

    def _start(self, request, response):
        del request
        try:
            self.controller.start(self._steady_seconds())
        except RuntimeError as error:
            response.success = False
            response.message = str(error)
            return response
        response.success = self.controller.state == ControllerState.RUNNING
        response.message = self.controller.safety.reason
        return response

    def _stop(self, request, response):
        del request
        self.controller.stop("stop service requested")
        self.controller.safety.filter_command(
            np.zeros(3, dtype=float), self._steady_seconds()
        )
        self._publish_zero()
        response.success = True
        response.message = self.controller.state.value
        return response

    def _clear_runtime_history(self) -> None:
        """Clear all timestamp, cache, metric, and duration state."""
        self.pose_cache.clear()
        self.twist_cache.clear()
        self.wrench_cache.clear()
        self.start_stamp_ns = None
        self.last_processed_stamp_ns = None
        self.last_bundle_stamp = None
        self.last_receive_steady = None
        self.processed_steps = 0
        self.saturation_count = 0
        self.stamp_mismatch_count = 0
        self.last_output = None
        self.wall_start = time.monotonic()
        self._duration_stopped = False

    def _reset(self, request, response):
        del request
        if self.controller.state in {
            ControllerState.RUNNING,
            ControllerState.STOPPING,
        }:
            response.success = False
            response.message = "stop the controller before reset"
            return response
        self._publish_zero()
        self.controller.reset(
            self.last_actual_position, self.last_actual_velocity
        )
        self._clear_runtime_history()
        response.success = True
        response.message = self.controller.state.value
        return response

    def _publish_diagnostics(self) -> None:
        now_wall = time.monotonic()
        wall_duration = max(now_wall - self.wall_start, 1e-12)
        state = self.controller.state
        status = DiagnosticStatus()
        status.name = "neuro_adaptive_control/controller"
        status.hardware_id = "pure_numpy_nac"
        if state == ControllerState.FAULT:
            status.level = DiagnosticStatus.ERROR
            status.message = self.controller.safety.reason
        elif self.saturation_count > 0 or self.stamp_mismatch_count > 0:
            status.level = DiagnosticStatus.WARN
            status.message = state.value
        else:
            status.level = DiagnosticStatus.OK
            status.message = state.value
        output = self.last_output
        raw_norm = 0.0 if output is None else float(np.linalg.norm(output.raw_command))
        command_norm = (
            0.0 if output is None else float(np.linalg.norm(output.command))
        )
        error_norm = (
            0.0 if output is None else float(np.linalg.norm(output.model_error))
        )
        receive_age = (
            float("inf")
            if self.last_receive_steady is None
            else self._steady_seconds() - self.last_receive_steady
        )
        values = {
            "state": state.value,
            "fault_reason": self.controller.safety.reason,
            "step": self.processed_steps,
            "fixed_dt_sec": self.dt,
            "target_rate_hz": self.target_rate_hz,
            "observed_wall_rate_hz": self.processed_steps / wall_duration,
            "state_age_sec": receive_age,
            "stamp_mismatch_count": self.stamp_mismatch_count,
            "adaptation_enabled": self.controller.network.adaptation_enabled,
            "weight_norm": self.controller.network.weight_norm,
            "raw_command_norm": raw_norm,
            "limited_command_norm": command_norm,
            "saturation_count": self.saturation_count,
            "impedance_tracking_error_norm": error_norm,
        }
        status.values = [
            KeyValue(key=key, value=str(value)) for key, value in values.items()
        ]
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status = [status]
        self.diagnostics_publisher.publish(message)

    def destroy_node(self) -> bool:
        """Publish a best-effort zero command before local teardown."""
        self._publish_zero()
        return super().destroy_node()


def main(args=None) -> None:
    """Run the ROS 2 NAC controller wrapper."""
    rclpy.init(args=args)
    node: Optional[NACControllerNode] = None
    try:
        node = NACControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
