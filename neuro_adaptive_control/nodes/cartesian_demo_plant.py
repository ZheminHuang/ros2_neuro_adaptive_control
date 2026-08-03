"""Deterministic unknown-dynamics Cartesian plant for the ROS 2 demo."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from neuro_adaptive_control.core.simulation import UnknownCartesianPlant


StampKey = Tuple[int, int]


def _key_from_stamp(stamp) -> StampKey:
    return int(stamp.sec), int(stamp.nanosec)


def _key_to_seconds(key: StampKey) -> float:
    return float(key[0]) + 1e-9 * float(key[1])


def _vector_from_wrench(message: WrenchStamped) -> np.ndarray:
    return np.array(
        [
            message.wrench.force.x,
            message.wrench.force.y,
            message.wrench.force.z,
        ],
        dtype=float,
    )


def _vector_from_pose(message: PoseStamped) -> np.ndarray:
    return np.array(
        [
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ],
        dtype=float,
    )


class CartesianDemoPlant(Node):
    """Publish fixed-step plant state and advance only on matching commands."""

    def __init__(self) -> None:
        super().__init__("cartesian_demo_plant")
        self.declare_parameter("control_rate_hz", 500.0)
        self.declare_parameter("duration_sec", 12.0)
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("initial_position", [0.0, 0.0, 0.0])
        self.declare_parameter("plant_substeps", 4)
        self.declare_parameter("external_wrench_enabled", False)
        self.declare_parameter("external_wrench_amplitude", [0.8, 0.6, 0.4])
        self.declare_parameter("external_wrench_frequency_hz", 0.35)
        self.declare_parameter("command_watchdog_sec", 0.25)
        self.declare_parameter("diagnostics_rate_hz", 20.0)
        self.declare_parameter("output_directory", "")

        rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be finite and positive.")
        if not np.isfinite(self.duration_sec) or self.duration_sec <= 0.0:
            raise ValueError("duration_sec must be finite and positive.")
        self.dt_ns = int(round(1e9 / rate_hz))
        self.dt = 1e-9 * float(self.dt_ns)
        self.target_rate_hz = 1.0 / self.dt
        self.frame_id = str(self.get_parameter("frame_id").value)
        if not self.frame_id:
            raise ValueError("frame_id must not be empty.")
        initial_position = np.asarray(
            self.get_parameter("initial_position").value, dtype=float
        )
        if initial_position.shape != (3,) or not np.all(np.isfinite(initial_position)):
            raise ValueError("initial_position must be a finite vector of length 3.")
        substeps = int(self.get_parameter("plant_substeps").value)
        self.plant = UnknownCartesianPlant(initial_position, substeps=substeps)
        self.initial_position = initial_position.copy()
        self.external_wrench_enabled = bool(
            self.get_parameter("external_wrench_enabled").value
        )
        self.external_amplitude = np.asarray(
            self.get_parameter("external_wrench_amplitude").value, dtype=float
        )
        if (
            self.external_amplitude.shape != (3,)
            or not np.all(np.isfinite(self.external_amplitude))
        ):
            raise ValueError(
                "external_wrench_amplitude must be a finite vector of length 3."
            )
        self.external_frequency = float(
            self.get_parameter("external_wrench_frequency_hz").value
        )
        self.command_watchdog_sec = float(
            self.get_parameter("command_watchdog_sec").value
        )
        diagnostics_rate = float(self.get_parameter("diagnostics_rate_hz").value)
        self.diagnostic_period_steps = max(
            1, int(round(self.target_rate_hz / diagnostics_rate))
        )
        self.output_directory = str(
            self.get_parameter("output_directory").value
        ).strip()

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=4,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped, "demo/cartesian_pose", qos
        )
        self.twist_publisher = self.create_publisher(
            TwistStamped, "demo/cartesian_twist", qos
        )
        self.wrench_publisher = self.create_publisher(
            WrenchStamped, "demo/applied_external_wrench", qos
        )
        self.diagnostics_publisher = self.create_publisher(
            DiagnosticArray, "diagnostics", 10
        )
        self.command_subscription = self.create_subscription(
            WrenchStamped, "nac/wrench_command", self._on_command, qos
        )
        self.external_subscription = self.create_subscription(
            WrenchStamped,
            "demo/external_wrench_input",
            self._on_external_input,
            qos,
        )
        self.desired_subscription = self.create_subscription(
            PoseStamped, "nac/desired_pose", self._on_desired_pose, qos
        )
        self.impedance_subscription = self.create_subscription(
            PoseStamped, "nac/impedance_pose", self._on_impedance_pose, qos
        )

        self.step_index = 0
        self.current_key: StampKey = (0, 0)
        self.awaiting_command = False
        self.pending_command: Optional[np.ndarray] = None
        self.pending_external_inputs: Dict[StampKey, np.ndarray] = {}
        self.current_external = np.zeros(3, dtype=float)
        self.last_publish_wall = time.monotonic()
        self.wall_start = self.last_publish_wall
        self.missed_deadlines = 0
        self.stamp_mismatches = 0
        self.state = "start"
        self.fault_reason = ""
        self._finished = False
        self._reported = False
        self._shutdown_scheduled = False

        self.actual_cache: Dict[StampKey, np.ndarray] = {}
        self.desired_cache: Dict[StampKey, np.ndarray] = {}
        self.impedance_cache: Dict[StampKey, np.ndarray] = {}
        self.recorded_keys: set[StampKey] = set()
        self.impedance_error_squared: list[float] = []
        self.desired_error_squared: list[float] = []
        self.impedance_error_max = 0.0
        self.desired_error_max = 0.0
        self.command_norm_squared: list[float] = []
        self.command_norm_max = 0.0
        self.command_samples = 0

        self.timer = self.create_timer(self.dt, self._on_timer)
        self.get_logger().info(
            "Unknown Cartesian plant ready: "
            f"fixed_dt={self.dt:.9f}s target_rate={self.target_rate_hz:.3f}Hz"
        )

    def _stamp_for_step(self, step: int):
        total_ns = int(step) * self.dt_ns
        stamp = PoseStamped().header.stamp
        stamp.sec = total_ns // 1_000_000_000
        stamp.nanosec = total_ns % 1_000_000_000
        return stamp

    def _generated_external_wrench(self, time_sec: float) -> np.ndarray:
        if not self.external_wrench_enabled:
            return np.zeros(3, dtype=float)
        omega = 2.0 * np.pi * self.external_frequency
        return self.external_amplitude * np.array(
            [
                np.sin(omega * time_sec),
                np.sin(0.83 * omega * time_sec + 0.7),
                np.sin(1.17 * omega * time_sec - 0.4),
            ],
            dtype=float,
        )

    def _external_for_current_step(self) -> np.ndarray:
        override = self.pending_external_inputs.pop(self.current_key, None)
        if override is not None:
            return override
        return self._generated_external_wrench(_key_to_seconds(self.current_key))

    def _publish_state_bundle(self) -> None:
        stamp = self._stamp_for_step(self.step_index)
        self.current_key = _key_from_stamp(stamp)
        self.current_external = self._external_for_current_step()

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = float(self.plant.position[0])
        pose.pose.position.y = float(self.plant.position[1])
        pose.pose.position.z = float(self.plant.position[2])
        pose.pose.orientation.w = 1.0

        twist = TwistStamped()
        twist.header.stamp = stamp
        twist.header.frame_id = self.frame_id
        twist.twist.linear.x = float(self.plant.velocity[0])
        twist.twist.linear.y = float(self.plant.velocity[1])
        twist.twist.linear.z = float(self.plant.velocity[2])

        wrench = WrenchStamped()
        wrench.header.stamp = stamp
        wrench.header.frame_id = self.frame_id
        wrench.wrench.force.x = float(self.current_external[0])
        wrench.wrench.force.y = float(self.current_external[1])
        wrench.wrench.force.z = float(self.current_external[2])

        self.actual_cache[self.current_key] = self.plant.position.copy()
        self._trim_caches()
        self.pose_publisher.publish(pose)
        self.twist_publisher.publish(twist)
        self.wrench_publisher.publish(wrench)
        self.awaiting_command = True
        self.pending_command = None
        self.last_publish_wall = time.monotonic()
        self.state = "running"

    def _on_command(self, message: WrenchStamped) -> None:
        key = _key_from_stamp(message.header.stamp)
        command = _vector_from_wrench(message)
        torque = np.array(
            [
                message.wrench.torque.x,
                message.wrench.torque.y,
                message.wrench.torque.z,
            ],
            dtype=float,
        )
        if (
            not self.awaiting_command
            or key != self.current_key
            or message.header.frame_id != self.frame_id
        ):
            self.stamp_mismatches += 1
            return
        if not np.all(np.isfinite(command)) or not np.all(np.isfinite(torque)):
            self._fault("received command contains NaN or Inf")
            return
        if self.pending_command is not None:
            return
        self.pending_command = command

    def _on_external_input(self, message: WrenchStamped) -> None:
        key = _key_from_stamp(message.header.stamp)
        wrench = _vector_from_wrench(message)
        if message.header.frame_id != self.frame_id or not np.all(np.isfinite(wrench)):
            self.stamp_mismatches += 1
            return
        self.pending_external_inputs[key] = wrench
        if len(self.pending_external_inputs) > 32:
            oldest = sorted(self.pending_external_inputs)[:-32]
            for old_key in oldest:
                self.pending_external_inputs.pop(old_key, None)

    def _on_desired_pose(self, message: PoseStamped) -> None:
        if message.header.frame_id != self.frame_id:
            self.stamp_mismatches += 1
            return
        key = _key_from_stamp(message.header.stamp)
        self.desired_cache[key] = _vector_from_pose(message)
        self._try_record_metrics(key)

    def _on_impedance_pose(self, message: PoseStamped) -> None:
        if message.header.frame_id != self.frame_id:
            self.stamp_mismatches += 1
            return
        key = _key_from_stamp(message.header.stamp)
        self.impedance_cache[key] = _vector_from_pose(message)
        self._try_record_metrics(key)

    def _try_record_metrics(self, key: StampKey) -> None:
        if key in self.recorded_keys:
            return
        if not all(
            key in cache
            for cache in (self.actual_cache, self.desired_cache, self.impedance_cache)
        ):
            return
        actual = self.actual_cache[key]
        desired_error = float(np.linalg.norm(self.desired_cache[key] - actual))
        impedance_error = float(
            np.linalg.norm(self.impedance_cache[key] - actual)
        )
        if not np.isfinite(desired_error) or not np.isfinite(impedance_error):
            self._fault("telemetry metric contains NaN or Inf")
            return
        self.desired_error_squared.append(desired_error**2)
        self.impedance_error_squared.append(impedance_error**2)
        self.desired_error_max = max(self.desired_error_max, desired_error)
        self.impedance_error_max = max(
            self.impedance_error_max, impedance_error
        )
        self.recorded_keys.add(key)
        self._trim_caches()

    def _trim_caches(self) -> None:
        for cache in (self.actual_cache, self.desired_cache, self.impedance_cache):
            if len(cache) > 64:
                for key in sorted(cache)[:-64]:
                    cache.pop(key, None)

    def _on_timer(self) -> None:
        if self._finished:
            return
        if not self.awaiting_command:
            self._publish_state_bundle()
            return
        if self.pending_command is None:
            self.missed_deadlines += 1
            age = time.monotonic() - self.last_publish_wall
            if age > self.command_watchdog_sec:
                self._fault(
                    f"command watchdog expired after {age:.3f}s at step "
                    f"{self.step_index}"
                )
            if self.missed_deadlines % self.diagnostic_period_steps == 0:
                self._publish_diagnostics()
            return

        command = self.pending_command.copy()
        command_norm = float(np.linalg.norm(command))
        self.command_norm_squared.append(command_norm**2)
        self.command_norm_max = max(self.command_norm_max, command_norm)
        self.command_samples += 1
        sim_time = _key_to_seconds(self.current_key)
        if sim_time + 0.5 * self.dt >= self.duration_sec:
            self.awaiting_command = False
            self.pending_command = None
            self.state = "stopped"
            self._finish_successfully()
            return

        try:
            self.plant.step(command, self.current_external, self.dt)
        except (FloatingPointError, TypeError, ValueError) as error:
            self._fault(str(error))
            return
        self.step_index += 1
        self.awaiting_command = False
        self.pending_command = None
        self._publish_state_bundle()
        if self.step_index % self.diagnostic_period_steps == 0:
            self._publish_diagnostics()

    def _fault(self, reason: str) -> None:
        if self._finished:
            return
        self.state = "fault"
        self.fault_reason = str(reason)
        self.pending_command = np.zeros(3, dtype=float)
        self.awaiting_command = False
        self._finished = True
        self._publish_diagnostics(error=True)
        self.get_logger().error(self.fault_reason)
        self._report_metrics()
        self._schedule_shutdown()

    def _finish_successfully(self) -> None:
        self._finished = True
        self._publish_diagnostics()
        self._report_metrics()
        self._schedule_shutdown()

    def _schedule_shutdown(self) -> None:
        """Request shutdown off-callback so the executor can unwind cleanly."""
        if self._shutdown_scheduled:
            return
        self._shutdown_scheduled = True

        def shutdown_after_callback() -> None:
            time.sleep(0.05)
            if rclpy.ok():
                rclpy.shutdown()

        threading.Thread(
            target=shutdown_after_callback,
            name="demo-plant-shutdown",
            daemon=True,
        ).start()

    def _metrics(self) -> dict:
        wall_duration = max(time.monotonic() - self.wall_start, 1e-12)

        def root_mean_square(samples: list[float]) -> float:
            if not samples:
                return float("nan")
            return float(math.sqrt(float(np.mean(samples))))

        return {
            "state": self.state,
            "fault_reason": self.fault_reason,
            "fixed_dt_sec": self.dt,
            "target_rate_hz": self.target_rate_hz,
            "observed_step_rate_hz": float(self.step_index / wall_duration),
            "simulated_duration_sec": float(self.step_index * self.dt),
            "wall_duration_sec": float(wall_duration),
            "steps": int(self.step_index),
            "metric_samples": int(len(self.impedance_error_squared)),
            "impedance_tracking_rmse_m": root_mean_square(
                self.impedance_error_squared
            ),
            "impedance_tracking_max_error_m": float(self.impedance_error_max),
            "desired_tracking_rmse_m": root_mean_square(
                self.desired_error_squared
            ),
            "desired_tracking_max_error_m": float(self.desired_error_max),
            "command_rms_norm_n": root_mean_square(self.command_norm_squared),
            "command_max_norm_n": float(self.command_norm_max),
            "command_samples": int(self.command_samples),
            "missed_wall_deadlines": int(self.missed_deadlines),
            "stamp_mismatches": int(self.stamp_mismatches),
            "external_wrench_enabled": bool(self.external_wrench_enabled),
        }

    def _report_metrics(self) -> None:
        if self._reported:
            return
        self._reported = True
        metrics = self._metrics()
        self.get_logger().info(
            "DEMO_METRICS " + json.dumps(metrics, sort_keys=True, allow_nan=True)
        )
        if self.output_directory:
            output_path = Path(self.output_directory).expanduser()
            output_path.mkdir(parents=True, exist_ok=True)
            metrics_path = output_path / "ros_demo_metrics.json"
            metrics_path.write_text(
                json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True)
                + "\n",
                encoding="utf-8",
            )
            self.get_logger().info(f"Saved metrics to {metrics_path}")

    def _publish_diagnostics(self, error: bool = False) -> None:
        metrics = self._metrics()
        status = DiagnosticStatus()
        status.name = "neuro_adaptive_control/demo_plant"
        status.hardware_id = "deterministic_numpy_plant"
        if error or self.state == "fault":
            status.level = DiagnosticStatus.ERROR
            status.message = self.fault_reason or "fault"
        elif self.missed_deadlines > 0:
            status.level = DiagnosticStatus.WARN
            status.message = "fixed-step target has missed wall deadlines"
        else:
            status.level = DiagnosticStatus.OK
            status.message = self.state
        keys = (
            "state",
            "fault_reason",
            "steps",
            "fixed_dt_sec",
            "target_rate_hz",
            "observed_step_rate_hz",
            "missed_wall_deadlines",
            "stamp_mismatches",
            "impedance_tracking_rmse_m",
            "desired_tracking_rmse_m",
            "command_max_norm_n",
            "external_wrench_enabled",
        )
        status.values = [
            KeyValue(key=key, value=str(metrics[key])) for key in keys
        ]
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status = [status]
        self.diagnostics_publisher.publish(message)

    def destroy_node(self) -> bool:
        """Zero the held input locally and report before destruction."""
        self.pending_command = np.zeros(3, dtype=float)
        self.awaiting_command = False
        self._report_metrics()
        return super().destroy_node()


def main(args=None) -> None:
    """Run the deterministic Cartesian demo plant."""
    rclpy.init(args=args)
    node: Optional[CartesianDemoPlant] = None
    try:
        node = CartesianDemoPlant()
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
