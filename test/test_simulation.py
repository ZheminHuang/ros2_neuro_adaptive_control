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

"""Integration tests for the deterministic unknown-dynamics demo."""

from inspect import signature

import numpy as np
import pytest

from neuro_adaptive_control.core.simulation import build_demo_controller
from neuro_adaptive_control.core.simulation import run_comparison
from neuro_adaptive_control.core.simulation import run_simulation
from neuro_adaptive_control.core.simulation import SimulationConfig
from neuro_adaptive_control.core.simulation import UnknownCartesianPlant


_COMMAND_AXIS_LIMIT_N = 40.0
_COMMAND_NORM_LIMIT_N = 55.0


def _assert_finite_bounded_result(result) -> None:
    """Check the public histories and the configured command envelope."""
    steps = result.time.size
    vector_histories = (
        result.desired,
        result.impedance,
        result.actual,
        result.velocity,
        result.command,
        result.raw_command,
        result.neural_estimate,
        result.tracking_error,
        result.desired_error,
        result.external_wrench,
    )
    assert result.time.shape == (steps,)
    assert result.saturated.shape == (steps,)
    assert np.all(np.isfinite(result.time))
    for history in vector_histories:
        assert history.shape == (steps, 3)
        assert np.all(np.isfinite(history))

    tolerance = 10.0 * np.finfo(float).eps
    assert np.max(np.abs(result.command)) <= (
        _COMMAND_AXIS_LIMIT_N + tolerance
    )
    assert np.max(np.linalg.norm(result.command, axis=1)) <= (
        _COMMAND_NORM_LIMIT_N + tolerance
    )
    assert result.metrics["controller_state"] == "running"


def test_demo_is_bitwise_deterministic() -> None:
    """The same seed and fixed-step scenario must reproduce every sample."""
    config = SimulationConfig(
        trajectory="circle",
        duration_sec=0.4,
        external_wrench_enabled=True,
        seed=23,
    )

    first = run_simulation(config, adaptation_enabled=True)
    second = run_simulation(config, adaptation_enabled=True)

    history_names = (
        "time",
        "desired",
        "impedance",
        "actual",
        "velocity",
        "command",
        "raw_command",
        "neural_estimate",
        "tracking_error",
        "desired_error",
        "external_wrench",
        "saturated",
    )
    for name in history_names:
        np.testing.assert_array_equal(
            getattr(first, name),
            getattr(second, name),
        )
    assert first.metrics == second.metrics


def test_controller_is_isolated_from_unknown_plant_parameters() -> None:
    """Controller construction must not receive the plant or true dynamics."""
    controller_arguments = set(signature(build_demo_controller).parameters)
    forbidden_arguments = {
        "plant",
        "plant_mass",
        "plant_damping",
        "plant_bias",
        "true_dynamics",
    }
    assert controller_arguments.isdisjoint(forbidden_arguments)

    plant = UnknownCartesianPlant((0.0, 0.0, 0.0))
    controller = build_demo_controller(adaptation_enabled=True, seed=7)

    assert all(value is not plant for value in vars(controller).values())
    assert not np.shares_memory(
        plant._mass,
        controller.impedance_model.parameters.mass,
    )
    assert not np.shares_memory(
        plant._damping,
        controller.impedance_model.parameters.damping,
    )
    assert not np.allclose(
        plant._mass,
        controller.impedance_model.parameters.mass,
    )


def test_unknown_plant_applies_configured_external_gain() -> None:
    """The physical wrench path must use the same explicit K_h contract."""
    unit = UnknownCartesianPlant(
        (0.0, 0.0, 0.0),
        substeps=1,
        external_gain=(1.0, 1.0, 1.0),
    )
    doubled = UnknownCartesianPlant(
        (0.0, 0.0, 0.0),
        substeps=1,
        external_gain=(2.0, 2.0, 2.0),
    )
    command = np.zeros(3)
    wrench = np.array([0.4, -0.2, 0.1])
    dt = 0.002

    unit.step(command, wrench, dt)
    doubled.step(command, wrench, dt)

    expected_velocity_difference = (
        dt * np.linalg.solve(unit._mass, wrench)
    )
    np.testing.assert_allclose(
        doubled.velocity - unit.velocity,
        expected_velocity_difference,
        rtol=1e-13,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    "external_gain",
    (
        (1.0, 1.0),
        (1.0, np.nan, 1.0),
        np.ones((2, 2)),
    ),
)
def test_unknown_plant_rejects_invalid_external_gain(external_gain) -> None:
    """The known wrench mapping must be finite and dimensionally explicit."""
    with pytest.raises(ValueError, match="external_gain"):
        UnknownCartesianPlant(
            (0.0, 0.0, 0.0),
            external_gain=external_gain,
        )


@pytest.mark.parametrize(
    "trajectory",
    ("circle", "line", "figure8", "fixed_point"),
)
def test_all_reference_trajectories_run_at_fixed_step(trajectory: str) -> None:
    """Every documented v0.1 reference must produce a valid demo history."""
    config = SimulationConfig(
        trajectory=trajectory,
        duration_sec=0.2,
        dt=0.002,
        seed=11,
    )

    result = run_simulation(config, adaptation_enabled=True)

    assert result.time.size == 100
    np.testing.assert_allclose(np.diff(result.time), config.dt, atol=1e-15)
    _assert_finite_bounded_result(result)
    desired_motion = np.ptp(result.desired, axis=0)
    if trajectory == "fixed_point":
        np.testing.assert_array_equal(desired_motion, np.zeros(3))
    else:
        assert np.linalg.norm(desired_motion) > 0.0


def test_optional_external_wrench_changes_the_same_demo_scenario() -> None:
    """Enabling the disturbance must populate input and affect the response."""
    common = dict(
        trajectory="line",
        duration_sec=0.5,
        dt=0.002,
        seed=17,
    )
    without_wrench = run_simulation(
        SimulationConfig(**common, external_wrench_enabled=False),
        adaptation_enabled=True,
    )
    with_wrench = run_simulation(
        SimulationConfig(**common, external_wrench_enabled=True),
        adaptation_enabled=True,
    )

    np.testing.assert_array_equal(without_wrench.time, with_wrench.time)
    np.testing.assert_array_equal(without_wrench.desired, with_wrench.desired)
    np.testing.assert_array_equal(
        without_wrench.external_wrench,
        np.zeros_like(without_wrench.external_wrench),
    )
    assert np.max(np.linalg.norm(with_wrench.external_wrench, axis=1)) > 0.5
    assert not np.array_equal(without_wrench.command, with_wrench.command)
    assert not np.array_equal(without_wrench.actual, with_wrench.actual)
    _assert_finite_bounded_result(with_wrench)


def test_adaptation_significantly_improves_identical_figure8_scenario() -> None:
    """NAC must outperform frozen weights in one paired dynamic scenario."""
    comparison = run_comparison(
        SimulationConfig(
            trajectory="figure8",
            duration_sec=2.0,
            dt=0.002,
            seed=7,
        )
    )
    baseline = comparison.baseline
    nac = comparison.nac

    np.testing.assert_array_equal(baseline.time, nac.time)
    np.testing.assert_array_equal(baseline.desired, nac.desired)
    np.testing.assert_array_equal(
        baseline.external_wrench,
        nac.external_wrench,
    )
    np.testing.assert_array_equal(baseline.command[0], nac.command[0])
    assert baseline.metrics["final_weight_norm"] == 0.0
    assert nac.metrics["final_weight_norm"] > 0.0

    metrics = comparison.metrics
    assert metrics["impedance_rmse_improvement_percent"] > 50.0
    assert metrics["desired_rmse_improvement_percent"] > 40.0
    assert metrics["nac_impedance_tracking_rmse_m"] < (
        0.5 * metrics["baseline_impedance_tracking_rmse_m"]
    )
    assert metrics["nac_desired_tracking_rmse_m"] < (
        0.65 * metrics["baseline_desired_tracking_rmse_m"]
    )
    _assert_finite_bounded_result(baseline)
    _assert_finite_bounded_result(nac)
