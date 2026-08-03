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

"""Soft performance regression guard for the pure NumPy controller step."""

from time import perf_counter_ns

import numpy as np

from neuro_adaptive_control.core.references import ReferenceSample
from neuro_adaptive_control.core.simulation import build_demo_controller


def test_core_step_targets_500_hz_with_bounded_jitter() -> None:
    """Measure a loaded adaptive step without claiming hard real-time use."""
    controller = build_demo_controller(adaptation_enabled=True, seed=7)
    controller.start(0.0)
    reference = ReferenceSample(
        position=np.array([0.04, -0.03, 0.02]),
        velocity=np.array([0.02, 0.01, -0.01]),
        acceleration=np.array([-0.01, 0.02, 0.01]),
    )
    position = np.array([0.01, -0.01, 0.005])
    velocity = np.array([0.005, -0.003, 0.002])
    external_wrench = np.array([0.2, -0.1, 0.05])
    dt = 0.002
    now = 0.0

    for _ in range(250):
        now += dt
        controller.step(
            position,
            velocity,
            reference,
            external_wrench,
            dt=dt,
            now=now,
        )

    elapsed_ms = np.empty(5000, dtype=float)
    for index in range(elapsed_ms.size):
        now += dt
        started_ns = perf_counter_ns()
        output = controller.step(
            position,
            velocity,
            reference,
            external_wrench,
            dt=dt,
            now=now,
        )
        elapsed_ms[index] = (perf_counter_ns() - started_ns) / 1.0e6

    median_ms = float(np.median(elapsed_ms))
    p95_ms = float(np.percentile(elapsed_ms, 95.0))
    p99_ms = float(np.percentile(elapsed_ms, 99.0))
    median_rate_hz = 1000.0 / median_ms
    print(
        "core step: "
        f"median={median_ms:.6f} ms, p95={p95_ms:.6f} ms, "
        f"p99={p99_ms:.6f} ms, median_rate={median_rate_hz:.1f} Hz"
    )

    assert output.state.value == "running"
    assert np.all(np.isfinite(output.command))
    assert median_ms < 2.0
    assert p95_ms < 2.0
    assert p99_ms < 4.0
