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
End-to-end acceptance tests for deterministic MuJoCo NAC tracking.

The timing fields exercised here are wall-clock telemetry only.  They measure
the synchronous fixed-step runner and do not constitute a hard real-time
guarantee for the Python or ROS 2 execution path.
"""

from dataclasses import fields
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from neuro_adaptive_control.adapters.mujoco_simulation import (
    MujocoRunConfig,
    build_mujoco_controller,
    run_mujoco_tracking,
)


TRAJECTORIES = ("circle", "line", "figure8", "fixed_point")
CONTROL_PERIOD_SEC = 0.002
CONTROL_RATE_HZ = 500.0
PLANT_SUBSTEPS = 4
RUN_DURATION_SEC = 8.0
SEED = 23

WALL_TIMING_METRICS = {
    "wall_duration_sec",
    "real_time_factor",
    "observed_control_step_rate_hz",
    "missed_wall_deadlines",
    "nac_time_median_ms",
    "nac_time_p95_ms",
    "nac_time_p99_ms",
    "mujoco_step_time_median_ms",
    "mujoco_step_time_p95_ms",
    "mujoco_step_time_p99_ms",
}

V01_METRIC_HASHES = {
    "v0.1.0_demo_metrics.json": (
        "50b5707d71215fb6f94ed7710bd2143cda5c677e17b810954bda4f973c41bcbe"
    ),
    "v0.1.0_ros_metrics.json": (
        "8977e9425152d883c6761e1c37e8d55b1956a257284eb88cecdd99480a86bcf1"
    ),
}


def _require_mujoco_bindings():
    """Skip dynamic tests when the actual MuJoCo Python bindings are absent."""
    module = pytest.importorskip("mujoco")
    if not hasattr(module, "MjModel"):
        pytest.skip("MuJoCo Python bindings are not available")
    return module


def _config(trajectory, adaptation_enabled):
    """Return one canonical, deterministic eight-second run configuration."""
    return MujocoRunConfig(
        duration_sec=RUN_DURATION_SEC,
        control_period_sec=CONTROL_PERIOD_SEC,
        plant_substeps=PLANT_SUBSTEPS,
        trajectory=trajectory,
        adaptation_enabled=adaptation_enabled,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def tracking_runs():
    """Run all reference modes plus matched and repeated circle scenarios."""
    _require_mujoco_bindings()
    adaptive = {
        trajectory: run_mujoco_tracking(_config(trajectory, True))
        for trajectory in TRAJECTORIES
    }
    frozen = run_mujoco_tracking(_config("circle", False))
    repeated = run_mujoco_tracking(_config("circle", True))
    return adaptive, frozen, repeated


def _assert_finite_history(result):
    """Assert that every numerical result history is finite and nonempty."""
    histories = (
        result.time,
        result.desired,
        result.impedance,
        result.actual,
        result.command_force,
        result.neural_estimate,
        result.arm_torque,
        result.joint_position,
        result.joint_velocity,
        result.object_position,
        result.contact_force,
    )
    for history in histories:
        assert history.size > 0
        assert np.all(np.isfinite(history))


def test_all_reference_modes_finish_without_fault(tracking_runs):
    """Require all four trajectory modes to complete eight seconds safely."""
    adaptive, _, _ = tracking_runs
    assert set(adaptive) == set(TRAJECTORIES)

    for trajectory, result in adaptive.items():
        assert result.metrics["state"] == "stopped"
        assert result.metrics["fault_reason"] == ""
        assert result.metrics["trajectory"] == trajectory
        assert result.metrics["adaptation_enabled"] is True
        assert result.metrics["control_steps"] == 4000
        assert result.metrics["mujoco_steps"] == 16000
        _assert_finite_history(result)


def test_circle_tracking_and_identical_frozen_baseline(tracking_runs):
    """Enforce circle accuracy and at least ten-percent adaptive improvement."""
    adaptive_runs, frozen, _ = tracking_runs
    adaptive = adaptive_runs["circle"]

    for config_field in fields(MujocoRunConfig):
        if config_field.name != "adaptation_enabled":
            assert getattr(adaptive.config, config_field.name) == getattr(
                frozen.config, config_field.name
            )

    assert adaptive.config.adaptation_enabled is True
    assert frozen.config.adaptation_enabled is False
    assert np.array_equal(adaptive.time, frozen.time)
    assert np.array_equal(adaptive.desired, frozen.desired)
    assert np.array_equal(adaptive.actual[0], frozen.actual[0])

    adaptive_rmse = adaptive.metrics["impedance_tracking_rmse_m"]
    frozen_rmse = frozen.metrics["impedance_tracking_rmse_m"]
    improvement = (frozen_rmse - adaptive_rmse) / frozen_rmse

    assert adaptive_rmse <= 0.03
    assert adaptive.metrics["impedance_tracking_max_error_m"] <= 0.08
    assert frozen_rmse > 0.0
    assert improvement >= 0.10
    assert frozen.metrics["final_weight_norm"] == 0.0
    assert np.array_equal(
        frozen.neural_estimate, np.zeros_like(frozen.neural_estimate)
    )
    assert adaptive.metrics["final_weight_norm"] > 0.0


def test_exact_500_hz_four_substep_and_stamp_contract(tracking_runs):
    """Verify exact control/substep counts and deterministic simulation stamps."""
    adaptive, frozen, _ = tracking_runs
    results = tuple(adaptive.values()) + (frozen,)
    expected_steps = int(round(RUN_DURATION_SEC / CONTROL_PERIOD_SEC))
    expected_time = (
        np.arange(1, expected_steps + 1, dtype=float) * CONTROL_PERIOD_SEC
    )

    for result in results:
        metrics = result.metrics
        assert metrics["control_period_sec"] == CONTROL_PERIOD_SEC
        assert 1.0 / metrics["control_period_sec"] == CONTROL_RATE_HZ
        assert metrics["substeps_per_control"] == PLANT_SUBSTEPS
        assert metrics["mujoco_timestep_sec"] == (
            CONTROL_PERIOD_SEC / PLANT_SUBSTEPS
        )
        assert metrics["control_steps"] == expected_steps
        assert metrics["mujoco_steps"] == expected_steps * PLANT_SUBSTEPS
        assert metrics["simulated_duration_sec"] == pytest.approx(
            RUN_DURATION_SEC, abs=2.0e-12
        )
        assert result.time.shape == (expected_steps,)
        assert np.allclose(result.time, expected_time, rtol=0.0, atol=2.0e-12)
        assert np.allclose(
            np.diff(result.time),
            CONTROL_PERIOD_SEC,
            rtol=0.0,
            atol=2.0e-12,
        )


def test_repeated_run_is_bitwise_deterministic_except_wall_timing(tracking_runs):
    """Require seeded physics and controller histories to repeat bit for bit."""
    adaptive, _, repeated = tracking_runs
    original = adaptive["circle"]

    assert original.config == repeated.config
    for attribute in (
        "time",
        "desired",
        "impedance",
        "actual",
        "command_force",
        "neural_estimate",
        "arm_torque",
        "joint_position",
        "joint_velocity",
        "object_position",
        "contact_force",
    ):
        assert np.array_equal(
            getattr(original, attribute), getattr(repeated, attribute)
        )

    deterministic_metrics = set(original.metrics) - WALL_TIMING_METRICS
    assert deterministic_metrics == set(repeated.metrics) - WALL_TIMING_METRICS
    for metric in deterministic_metrics:
        assert original.metrics[metric] == repeated.metrics[metric]


def test_timing_fields_are_valid_soft_measurements(tracking_runs):
    """Validate timing telemetry without treating it as a real-time guarantee."""
    adaptive, _, _ = tracking_runs
    metrics = adaptive["circle"].metrics

    assert WALL_TIMING_METRICS <= set(metrics)
    for metric in WALL_TIMING_METRICS:
        assert np.isfinite(metrics[metric])
        assert metrics[metric] >= 0.0

    assert metrics["wall_duration_sec"] > 0.0
    assert metrics["real_time_factor"] > 0.0
    assert metrics["observed_control_step_rate_hz"] > 0.0
    assert isinstance(metrics["missed_wall_deadlines"], int)
    assert (
        metrics["nac_time_median_ms"]
        <= metrics["nac_time_p95_ms"]
        <= metrics["nac_time_p99_ms"]
    )
    assert (
        metrics["mujoco_step_time_median_ms"]
        <= metrics["mujoco_step_time_p95_ms"]
        <= metrics["mujoco_step_time_p99_ms"]
    )
    assert all("hard" not in key.lower() for key in metrics)
    text_values = " ".join(
        value.lower() for value in metrics.values() if isinstance(value, str)
    )
    assert "hard realtime" not in text_values
    assert "hard real-time" not in text_values


def test_robot_controller_is_27d_and_v01_metrics_are_unchanged():
    """Lock the robot RBF input at 27D and preserve published 21D evidence."""
    controller = build_mujoco_controller(
        initial_position=np.zeros(3),
        adaptation_enabled=True,
        seed=SEED,
    )
    network = controller.network

    assert network.input_dim == 27
    assert network.output_dim == 3
    assert network.centers.shape == (network.num_basis, 27)
    assert controller.dynamics_feature_dim == 12

    project_root = Path(__file__).resolve().parents[1]
    metrics_dir = project_root / "docs" / "metrics"
    for filename, expected_hash in V01_METRIC_HASHES.items():
        metric_path = metrics_dir / filename
        assert metric_path.is_file()
        assert sha256(metric_path.read_bytes()).hexdigest() == expected_hash
