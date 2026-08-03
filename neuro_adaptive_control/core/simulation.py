"""Deterministic, ROS-independent unknown-dynamics demonstration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .impedance_model import CartesianImpedanceModel, ImpedanceParameters
from .neuro_adaptive_controller import NACParameters, NeuroAdaptiveController
from .rbf_network import RBFNetwork
from .references import ReferenceTrajectory, make_reference
from .safety import SafetyConfig, SafetySupervisor


@dataclass(frozen=True)
class SimulationConfig:
    """Reproducible scenario parameters shared by NAC and baseline runs."""

    trajectory: str = "circle"
    duration_sec: float = 12.0
    dt: float = 0.002
    plant_substeps: int = 4
    external_wrench_enabled: bool = False
    seed: int = 7
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    frequency_hz: float = 0.20
    radius_m: float = 0.08
    line_length_m: float = 0.16
    figure8_width_m: float = 0.16
    figure8_height_m: float = 0.10

    def __post_init__(self) -> None:
        for name in ("duration_sec", "dt", "frequency_hz"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if self.plant_substeps <= 0:
            raise ValueError("plant_substeps must be positive.")
        if self.duration_sec < self.dt:
            raise ValueError("duration_sec must be at least one time step.")
        center = np.asarray(self.center, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center must contain three finite coordinates.")


@dataclass(frozen=True)
class SimulationResult:
    """Time histories and scalar metrics for one controller setting."""

    adaptation_enabled: bool
    time: np.ndarray
    desired: np.ndarray
    impedance: np.ndarray
    actual: np.ndarray
    velocity: np.ndarray
    command: np.ndarray
    raw_command: np.ndarray
    neural_estimate: np.ndarray
    tracking_error: np.ndarray
    desired_error: np.ndarray
    external_wrench: np.ndarray
    saturated: np.ndarray
    metrics: Dict[str, float | int | bool | str]


@dataclass(frozen=True)
class ComparisonResult:
    """Paired baseline and adaptive results from the identical scenario."""

    baseline: SimulationResult
    nac: SimulationResult
    metrics: Dict[str, float]


class UnknownCartesianPlant:
    """
    Simulate fixed but controller-inaccessible Cartesian dynamics.

    The plant contains coupled inertia/damping, bias, nonlinear position terms,
    and quadratic drag. No plant parameter is passed to the controller.
    """

    def __init__(
        self,
        initial_position: np.ndarray | tuple[float, float, float],
        *,
        substeps: int = 4,
    ) -> None:
        position = np.asarray(initial_position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("initial_position must be a finite 3-vector.")
        if substeps <= 0:
            raise ValueError("substeps must be positive.")
        self.initial_position = position.copy()
        self.substeps = int(substeps)
        self._mass = np.array(
            [[1.75, 0.10, 0.04], [0.10, 1.30, 0.06], [0.04, 0.06, 2.05]],
            dtype=float,
        )
        self._damping = np.array(
            [[2.20, 0.12, 0.00], [0.12, 1.75, 0.08], [0.00, 0.08, 2.55]],
            dtype=float,
        )
        self._bias = np.array([0.70, -0.50, 0.35], dtype=float)
        self.reset()

    def reset(self) -> None:
        """Restore the deterministic initial state."""
        self.position = self.initial_position.copy()
        self.velocity = np.zeros(3, dtype=float)

    def _unknown_force(self) -> np.ndarray:
        nonlinear = 0.65 * np.sin(
            3.0 * self.position + np.array([0.2, -0.4, 0.7])
        )
        drag = 0.18 * self.velocity * np.abs(self.velocity)
        coupling = np.array(
            [
                0.30 * np.sin(self.position[1] - 0.5 * self.position[2]),
                0.25 * np.sin(self.position[2] + self.position[0]),
                0.28 * np.sin(self.position[0] - self.position[1]),
            ],
            dtype=float,
        )
        return self._bias + nonlinear + drag + coupling

    def step(
        self,
        command: np.ndarray,
        external_wrench: np.ndarray,
        dt: float,
    ) -> None:
        """Advance with held force and external wrench over fixed substeps."""
        force = np.asarray(command, dtype=float)
        external = np.asarray(external_wrench, dtype=float)
        if (
            force.shape != (3,)
            or external.shape != (3,)
            or not np.all(np.isfinite(force))
            or not np.all(np.isfinite(external))
            or not np.isfinite(dt)
            or dt <= 0.0
        ):
            raise ValueError("plant inputs and dt must be finite and well formed.")
        substep_dt = float(dt) / float(self.substeps)
        for _ in range(self.substeps):
            acceleration = np.linalg.solve(
                self._mass,
                force
                + external
                - self._damping @ self.velocity
                - self._unknown_force(),
            )
            self.velocity += substep_dt * acceleration
            self.position += substep_dt * self.velocity
        if not (
            np.all(np.isfinite(self.position))
            and np.all(np.isfinite(self.velocity))
        ):
            raise FloatingPointError("unknown plant produced NaN or Inf.")


def external_wrench_at(time_sec: float, enabled: bool) -> np.ndarray:
    """Return the optional deterministic physical disturbance."""
    if not enabled:
        return np.zeros(3, dtype=float)
    omega = 2.0 * np.pi * 0.35
    return np.array(
        [
            0.8 * np.sin(omega * time_sec),
            0.6 * np.sin(0.83 * omega * time_sec + 0.7),
            0.4 * np.sin(1.17 * omega * time_sec - 0.4),
        ],
        dtype=float,
    )


def build_demo_reference(config: SimulationConfig) -> ReferenceTrajectory:
    """Create the configured analytic trajectory."""
    return make_reference(
        config.trajectory,
        center=config.center,
        frequency=config.frequency_hz,
        radius=config.radius_m,
        line_length=config.line_length_m,
        figure8_width=config.figure8_width_m,
        figure8_height=config.figure8_height_m,
    )


def build_demo_controller(
    *,
    adaptation_enabled: bool,
    initial_position: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    initial_velocity: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    seed: int = 7,
) -> NeuroAdaptiveController:
    """Construct the documented v0.1 controller without plant parameters."""
    impedance_parameters = ImpedanceParameters.diagonal(
        mass=(1.0, 1.0, 1.0),
        damping=(12.0, 12.0, 12.0),
        stiffness=(35.0, 35.0, 35.0),
        external_gain=(1.0, 1.0, 1.0),
    )
    model = CartesianImpedanceModel(
        impedance_parameters,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
    )
    input_scale = np.array(
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
        dtype=float,
    )
    network = RBFNetwork(
        input_dim=21,
        output_dim=3,
        num_basis=31,
        widths=2.5,
        input_scale=input_scale,
        feature_clip=3.0,
        learning_rate=5.0,
        leakage=0.01,
        weight_limit=80.0,
        seed=seed,
        adaptation_enabled=adaptation_enabled,
    )
    nac_parameters = NACParameters.diagonal(
        lambda_gain=(7.0, 7.0, 7.0),
        feedback_gain=(18.0, 18.0, 20.0),
        robust_gain=(0.04, 0.04, 0.04),
        robust_bias=1.5,
    )
    safety = SafetySupervisor(
        SafetyConfig(
            command_limits=np.array([40.0, 40.0, 40.0]),
            command_norm_limit=55.0,
            watchdog_timeout=0.10,
            maximum_dt=0.01,
        )
    )
    return NeuroAdaptiveController(model, network, nac_parameters, safety)


def _metrics(
    *,
    adaptation_enabled: bool,
    dt: float,
    impedance_error: np.ndarray,
    desired_error: np.ndarray,
    command: np.ndarray,
    saturated: np.ndarray,
    controller: NeuroAdaptiveController,
) -> Dict[str, float | int | bool | str]:
    impedance_norm = np.linalg.norm(impedance_error, axis=1)
    desired_norm = np.linalg.norm(desired_error, axis=1)
    command_norm = np.linalg.norm(command, axis=1)
    return {
        "adaptation_enabled": bool(adaptation_enabled),
        "fixed_dt_sec": float(dt),
        "target_rate_hz": float(1.0 / dt),
        "steps": int(impedance_error.shape[0]),
        "duration_sec": float(dt * impedance_error.shape[0]),
        "impedance_tracking_rmse_m": float(
            np.sqrt(np.mean(impedance_norm**2))
        ),
        "impedance_tracking_max_error_m": float(np.max(impedance_norm)),
        "desired_tracking_rmse_m": float(np.sqrt(np.mean(desired_norm**2))),
        "desired_tracking_max_error_m": float(np.max(desired_norm)),
        "command_rms_norm_n": float(np.sqrt(np.mean(command_norm**2))),
        "command_max_norm_n": float(np.max(command_norm)),
        "saturation_count": int(np.count_nonzero(saturated)),
        "final_weight_norm": float(controller.network.weight_norm),
        "controller_state": controller.state.value,
    }


def run_simulation(
    config: SimulationConfig,
    *,
    adaptation_enabled: bool,
) -> SimulationResult:
    """Run one fixed-step scenario without wall-clock or ROS dependencies."""
    reference = build_demo_reference(config)
    initial_position = np.asarray(config.center, dtype=float)
    plant = UnknownCartesianPlant(
        initial_position, substeps=config.plant_substeps
    )
    controller = build_demo_controller(
        adaptation_enabled=adaptation_enabled,
        initial_position=initial_position,
        seed=config.seed,
    )
    controller.start(0.0)
    steps = int(round(config.duration_sec / config.dt))
    if steps <= 0:
        raise ValueError("simulation requires at least one step.")

    time_history = np.empty(steps, dtype=float)
    desired_history = np.empty((steps, 3), dtype=float)
    impedance_history = np.empty((steps, 3), dtype=float)
    actual_history = np.empty((steps, 3), dtype=float)
    velocity_history = np.empty((steps, 3), dtype=float)
    command_history = np.empty((steps, 3), dtype=float)
    raw_history = np.empty((steps, 3), dtype=float)
    neural_history = np.empty((steps, 3), dtype=float)
    impedance_error_history = np.empty((steps, 3), dtype=float)
    desired_error_history = np.empty((steps, 3), dtype=float)
    external_history = np.empty((steps, 3), dtype=float)
    saturated_history = np.empty(steps, dtype=bool)

    for step in range(steps):
        time_sec = float(step) * config.dt
        sample = reference.evaluate(time_sec)
        external = external_wrench_at(
            time_sec, config.external_wrench_enabled
        )
        output = controller.step(
            plant.position,
            plant.velocity,
            sample,
            external,
            dt=config.dt,
            now=time_sec,
        )
        if output.state.value == "fault":
            raise RuntimeError(f"controller fault: {output.fault_reason}")
        plant.step(output.command, external, config.dt)
        next_time = float(step + 1) * config.dt
        desired_next = reference.evaluate(next_time).position
        model_next = output.model_state.position

        time_history[step] = next_time
        desired_history[step] = desired_next
        impedance_history[step] = model_next
        actual_history[step] = plant.position
        velocity_history[step] = plant.velocity
        command_history[step] = output.command
        raw_history[step] = output.raw_command
        neural_history[step] = output.neural_estimate
        impedance_error_history[step] = model_next - plant.position
        desired_error_history[step] = desired_next - plant.position
        external_history[step] = external
        saturated_history[step] = output.saturated

    metrics = _metrics(
        adaptation_enabled=adaptation_enabled,
        dt=config.dt,
        impedance_error=impedance_error_history,
        desired_error=desired_error_history,
        command=command_history,
        saturated=saturated_history,
        controller=controller,
    )
    return SimulationResult(
        adaptation_enabled=adaptation_enabled,
        time=time_history,
        desired=desired_history,
        impedance=impedance_history,
        actual=actual_history,
        velocity=velocity_history,
        command=command_history,
        raw_command=raw_history,
        neural_estimate=neural_history,
        tracking_error=impedance_error_history,
        desired_error=desired_error_history,
        external_wrench=external_history,
        saturated=saturated_history,
        metrics=metrics,
    )


def run_comparison(config: SimulationConfig) -> ComparisonResult:
    """Compare frozen-weight baseline and NAC under identical conditions."""
    baseline = run_simulation(config, adaptation_enabled=False)
    nac = run_simulation(config, adaptation_enabled=True)
    baseline_rmse = float(baseline.metrics["impedance_tracking_rmse_m"])
    nac_rmse = float(nac.metrics["impedance_tracking_rmse_m"])
    improvement = 100.0 * (baseline_rmse - nac_rmse) / baseline_rmse
    desired_baseline = float(baseline.metrics["desired_tracking_rmse_m"])
    desired_nac = float(nac.metrics["desired_tracking_rmse_m"])
    desired_improvement = (
        100.0 * (desired_baseline - desired_nac) / desired_baseline
    )
    return ComparisonResult(
        baseline=baseline,
        nac=nac,
        metrics={
            "impedance_rmse_improvement_percent": float(improvement),
            "desired_rmse_improvement_percent": float(desired_improvement),
            "baseline_impedance_tracking_rmse_m": baseline_rmse,
            "nac_impedance_tracking_rmse_m": nac_rmse,
            "baseline_desired_tracking_rmse_m": desired_baseline,
            "nac_desired_tracking_rmse_m": desired_nac,
        },
    )
