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

"""Soft 500 Hz timing guard for the pure NumPy six-DoF NAC step."""

from time import perf_counter_ns

import numpy as np

from neuro_adaptive_control.adapters.mujoco_payload_benchmark import (
    build_pose_controller,
)
from neuro_adaptive_control.core.pose_references import PoseReferenceSample


def test_six_dof_core_step_targets_500_hz_without_realtime_claim() -> None:
    controller = build_pose_controller(np.zeros(6), seed=41)
    controller.start(0.0)
    reference = PoseReferenceSample(
        position=np.array((0.02, -0.01, 0.03, 0.04, -0.03, 0.02)),
        velocity=np.array((0.01, 0.01, -0.01, 0.02, -0.01, 0.01)),
        acceleration=np.zeros(6),
    )
    joint_position = np.linspace(-1.0, 1.0, 6)
    joint_velocity = np.linspace(0.2, -0.2, 6)
    dt = 0.002
    now = 0.0
    for _ in range(250):
        now += dt
        controller.step(
            np.zeros(6),
            np.zeros(6),
            joint_position,
            joint_velocity,
            reference,
            np.zeros(6),
            dt=dt,
            now=now,
        )
    elapsed_ms = np.empty(5000)
    for index in range(elapsed_ms.size):
        now += dt
        started = perf_counter_ns()
        output = controller.step(
            np.zeros(6),
            np.zeros(6),
            joint_position,
            joint_velocity,
            reference,
            np.zeros(6),
            dt=dt,
            now=now,
        )
        elapsed_ms[index] = (perf_counter_ns() - started) * 1.0e-6
    median_ms = float(np.median(elapsed_ms))
    p95_ms = float(np.percentile(elapsed_ms, 95.0))
    p99_ms = float(np.percentile(elapsed_ms, 99.0))
    print(
        "six-DoF core step: "
        f"median={median_ms:.6f} ms, p95={p95_ms:.6f} ms, "
        f"p99={p99_ms:.6f} ms, median_rate={1000.0 / median_ms:.1f} Hz"
    )

    assert output.state.value == "running"
    assert np.all(np.isfinite(output.command))
    assert median_ms < 2.0
    assert p95_ms < 2.0
    assert p99_ms < 4.0
