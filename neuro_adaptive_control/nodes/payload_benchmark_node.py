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

"""Run the 6D payload benchmark with an optional real-time MuJoCo viewer."""

from __future__ import annotations

import json
from pathlib import Path
from time import monotonic, sleep

from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import rclpy
from rclpy.node import Node

from neuro_adaptive_control.adapters.mujoco_payload_benchmark import (
    BenchmarkController,
    MujocoPayloadBenchmarkRunner,
    PayloadBenchmarkConfig,
    PayloadCase,
)


class PayloadBenchmarkNode(Node):
    """Own one benchmark runner and expose final metrics as diagnostics."""

    def __init__(self) -> None:
        super().__init__("payload_benchmark")
        self.declare_parameter("controller", "adaptive_nac")
        self.declare_parameter("viewer", True)
        self.declare_parameter("realtime", True)
        self.declare_parameter("payload_name", "showcase_750g_offset")
        self.declare_parameter("payload_mass_kg", 0.75)
        self.declare_parameter("payload_com_offset_m", [0.004, -0.003, 0.002])
        self.declare_parameter("payload_inertia_scale", 1.15)
        self.declare_parameter("seed", 41)
        self._diagnostics = self.create_publisher(
            DiagnosticArray,
            "/payload_benchmark/diagnostics",
            1,
        )

    def run(self) -> bool:
        """Run to completion, pacing the optional viewer at simulated time."""
        payload = PayloadCase(
            name=str(self.get_parameter("payload_name").value),
            mass_kg=float(self.get_parameter("payload_mass_kg").value),
            com_offset_m=tuple(
                float(value)
                for value in self.get_parameter("payload_com_offset_m").value
            ),
            inertia_scale=float(
                self.get_parameter("payload_inertia_scale").value
            ),
            seed=int(self.get_parameter("seed").value),
        )
        config = PayloadBenchmarkConfig(
            controller=BenchmarkController(
                str(self.get_parameter("controller").value)
            ),
            payload=payload,
        )
        model_path = (
            Path(get_package_share_directory("neuro_adaptive_control"))
            / "mujoco"
            / "ur5e_robotiq_2f85.xml"
        )
        runner = MujocoPayloadBenchmarkRunner(config, model_path=model_path)
        viewer_enabled = bool(self.get_parameter("viewer").value)
        realtime = bool(self.get_parameter("realtime").value)
        viewer = None
        if viewer_enabled:
            try:
                import mujoco.viewer

                viewer = mujoco.viewer.launch_passive(
                    runner.plant.model,
                    runner.plant.data,
                )
                viewer.cam.lookat[:] = (-0.08, 0.40, 0.30)
                viewer.cam.distance = 1.15
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -20.0
            except (ImportError, RuntimeError) as error:
                raise RuntimeError(
                    "MuJoCo viewer could not start; use viewer:=false for "
                    "headless execution"
                ) from error
        start = monotonic()
        period = config.control_period_sec

        def synchronize(index: int, phase: str) -> None:
            if viewer is not None and index % 5 == 0:
                if not viewer.is_running():
                    raise RuntimeError("MuJoCo viewer was closed")
                viewer.sync()
            if realtime:
                remaining = start + (index + 1) * period - monotonic()
                if remaining > 0.0:
                    sleep(remaining)
            if index % 500 == 0:
                self.get_logger().info(
                    f"t={(index + 1) * period:.1f}s phase={phase}"
                )

        try:
            result = runner.run(step_callback=synchronize)
        finally:
            if viewer is not None:
                viewer.close()
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = "six_dof_payload_benchmark"
        status.hardware_id = "mujoco_ur5e_robotiq_2f85"
        status.level = (
            DiagnosticStatus.OK
            if bool(result.metrics["success"])
            else DiagnosticStatus.ERROR
        )
        status.message = str(result.metrics["state"])
        status.values = [
            KeyValue(key=str(key), value=str(value))
            for key, value in result.metrics.items()
        ]
        message.status = [status]
        self._diagnostics.publish(message)
        self.get_logger().info(json.dumps(result.metrics, indent=2))
        return bool(result.metrics["success"])


def main(args=None) -> None:
    """Run the benchmark node once and exit with its acceptance status."""
    rclpy.init(args=args)
    node = PayloadBenchmarkNode()
    try:
        success = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not success:
        raise RuntimeError("payload benchmark did not satisfy acceptance")


if __name__ == "__main__":
    main()
