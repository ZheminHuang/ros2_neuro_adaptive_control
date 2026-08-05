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

"""Integration contracts for unknown-payload acquisition and adaptation."""

import inspect

import numpy as np

from neuro_adaptive_control.adapters.mujoco_payload_benchmark import (
    BenchmarkController,
    MujocoPayloadBenchmarkRunner,
    PayloadBenchmarkConfig,
    SHOWCASE_PAYLOAD_CASE,
    build_pose_mapper,
    evaluate_adaptation_advantage,
    payload_schedule,
    run_payload_benchmark,
)
from neuro_adaptive_control.core.pose_neuro_adaptive_controller import (
    PoseNeuroAdaptiveController,
)


def test_schedule_is_continuous_and_excites_all_six_pose_coordinates():
    initial = np.array((-0.13, 0.49, 0.33, 0.0, 0.0, 0.0))
    for boundary in (1.0, 2.0, 4.0, 5.0, 6.5, 11.5, 13.0, 13.5):
        left = payload_schedule(boundary - 1.0e-8, initial)[1]
        right = payload_schedule(boundary + 1.0e-8, initial)[1]
        np.testing.assert_allclose(left.position, right.position, atol=1.0e-7)
    samples = np.array(
        [
            payload_schedule(time_sec, initial)[1].position
            for time_sec in np.linspace(6.5, 11.5, 201)
        ]
    )
    xy = samples[:, :2]
    circle_center = np.array((initial[0] - 0.04, initial[1]))
    np.testing.assert_allclose(
        np.linalg.norm(xy - circle_center, axis=1),
        0.04,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(samples[0], samples[-1], atol=1.0e-12)
    assert np.all(np.ptp(samples[:, 3:], axis=0) > 0.0)


def test_model_free_core_has_no_mujoco_dynamics_truth_access():
    source = inspect.getsource(PoseNeuroAdaptiveController)
    for forbidden in (
        "qM",
        "qfrc_bias",
        "mj_fullM",
        "body_mass",
        "payload_mass",
        "contact_model",
    ):
        assert forbidden not in source


def test_public_nominal_payload_baseline_uses_identical_torque_limits():
    runner = MujocoPayloadBenchmarkRunner(
        PayloadBenchmarkConfig(
            controller=BenchmarkController.NOMINAL_MODEL_BASED,
            payload=SHOWCASE_PAYLOAD_CASE,
        )
    )
    nac_limits = build_pose_mapper().config.torque_limits
    nac_rate_limits = build_pose_mapper().config.torque_rate_limits
    np.testing.assert_array_equal(
        runner.model_controller.torque_limits, nac_limits
    )
    np.testing.assert_array_equal(
        runner.model_controller.torque_rate_limits, nac_rate_limits
    )


def test_adaptive_nac_beats_payload_time_freeze_on_showcase_case():
    adaptive = run_payload_benchmark(
        PayloadBenchmarkConfig(controller=BenchmarkController.ADAPTIVE_NAC)
    )
    frozen = run_payload_benchmark(
        PayloadBenchmarkConfig(
            controller=BenchmarkController.FROZEN_AT_PAYLOAD
        )
    )
    summary = evaluate_adaptation_advantage((adaptive, frozen))

    assert adaptive.metrics["success"]
    assert frozen.metrics["success"]
    assert summary["adaptive_completion_ratio"] == 1.0
    assert summary["median_loaded_position_improvement_ratio"] >= 0.10
    assert summary["median_loaded_orientation_improvement_ratio"] >= 0.10
    assert summary["adaptation_advantage_gate_passed"]
    np.testing.assert_array_equal(
        adaptive.payload_acquired,
        frozen.payload_acquired,
    )
