#!/usr/bin/env python3
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

"""Generate traceable full-dynamics MuJoCo tracking benchmark artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from hashlib import sha256
import json
from pathlib import Path
from platform import python_version
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuro_adaptive_control import __version__  # noqa: E402
from neuro_adaptive_control.adapters.mujoco_simulation import (  # noqa: E402
    MujocoRunConfig,
    MujocoRunResult,
    run_mujoco_tracking,
)


TRAJECTORIES = ("circle", "line", "figure8", "fixed_point")
MODEL_RELATIVE_PATH = Path("mujoco/ur5e_robotiq_2f85.xml")
MANIFEST_RELATIVE_PATH = Path("mujoco/SHA256SUMS")
RUNNER_RELATIVE_PATH = Path(
    "neuro_adaptive_control/adapters/mujoco_simulation.py"
)
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    Path("neuro_adaptive_control/adapters/mujoco_ur5e_adapter.py"),
    Path("neuro_adaptive_control/adapters/robotiq_gripper_adapter.py"),
    Path("neuro_adaptive_control/adapters/ur5e_wrench_to_torque.py"),
    Path("neuro_adaptive_control/core/impedance_model.py"),
    Path("neuro_adaptive_control/core/neuro_adaptive_controller.py"),
    Path("neuro_adaptive_control/core/rbf_network.py"),
    Path("neuro_adaptive_control/core/references.py"),
    Path("neuro_adaptive_control/core/safety.py"),
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"


def _file_sha256(relative_path: Path) -> str:
    """Return a source-tree file digest without recording an absolute path."""
    return sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()


def _history_sha256(result: MujocoRunResult) -> str:
    """Digest deterministic histories while excluding wall-clock telemetry."""
    digest = sha256()
    for name in (
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
        values = np.ascontiguousarray(getattr(result, name))
        digest.update(name.encode("utf-8"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _mujoco_version() -> str:
    """Return the loaded official binding version or fail actionably."""
    import mujoco

    if not hasattr(mujoco, "MjModel"):
        raise RuntimeError(
            "Official MuJoCo bindings are required; install mujoco==3.9.0."
        )
    return str(mujoco.__version__)


def _validate_matched_baseline(
    adaptive: MujocoRunResult,
    baseline: MujocoRunResult,
) -> None:
    """Require circle runs to differ only by the adaptation switch."""
    for config_field in fields(MujocoRunConfig):
        if config_field.name != "adaptation_enabled":
            if getattr(adaptive.config, config_field.name) != getattr(
                baseline.config, config_field.name
            ):
                raise RuntimeError(
                    f"unmatched baseline field: {config_field.name}"
                )
    if not np.array_equal(adaptive.time, baseline.time):
        raise RuntimeError("adaptive and baseline stamps differ")
    if not np.array_equal(adaptive.desired, baseline.desired):
        raise RuntimeError("adaptive and baseline references differ")
    if not np.array_equal(adaptive.actual[0], baseline.actual[0]):
        raise RuntimeError("adaptive and baseline initial states differ")


def _run_benchmark() -> tuple[dict[str, MujocoRunResult], MujocoRunResult]:
    """Run four adaptive modes and a matched frozen-weight circle baseline."""
    adaptive = {
        trajectory: run_mujoco_tracking(
            MujocoRunConfig(
                trajectory=trajectory,
                duration_sec=8.0,
                control_period_sec=0.002,
                plant_substeps=4,
                adaptation_enabled=True,
                external_wrench_mode="none",
                seed=23,
            )
        )
        for trajectory in TRAJECTORIES
    }
    baseline = run_mujoco_tracking(
        MujocoRunConfig(
            trajectory="circle",
            duration_sec=8.0,
            control_period_sec=0.002,
            plant_substeps=4,
            adaptation_enabled=False,
            external_wrench_mode="none",
            seed=23,
        )
    )
    _validate_matched_baseline(adaptive["circle"], baseline)
    return adaptive, baseline


def _build_report(
    adaptive: dict[str, MujocoRunResult],
    baseline: MujocoRunResult,
) -> dict[str, object]:
    """Build the machine-readable report from measured runner outputs."""
    circle = adaptive["circle"]
    adaptive_rmse = float(circle.metrics["impedance_tracking_rmse_m"])
    baseline_rmse = float(baseline.metrics["impedance_tracking_rmse_m"])
    improvement = 100.0 * (baseline_rmse - adaptive_rmse) / baseline_rmse
    all_stopped = all(
        result.metrics["state"] == "stopped"
        and result.metrics["fault_reason"] == ""
        for result in adaptive.values()
    )
    trajectory_reports = {
        trajectory: {
            "adaptive_metrics": result.metrics,
            "deterministic_history_sha256": _history_sha256(result),
        }
        for trajectory, result in adaptive.items()
    }
    trajectory_reports["circle"]["frozen_baseline_metrics"] = baseline.metrics
    trajectory_reports["circle"][
        "frozen_baseline_history_sha256"
    ] = _history_sha256(baseline)
    trajectory_reports["circle"]["comparison"] = {
        "impedance_rmse_improvement_percent": improvement,
        "matched_except_adaptation": True,
    }
    config = asdict(circle.config)
    return {
        "schema_version": 1,
        "artifact_kind": "full_dynamics_tracking_benchmark",
        "traceability": {
            "generator": "examples/run_mujoco_benchmark.py",
            "generator_sha256": _file_sha256(
                Path("examples/run_mujoco_benchmark.py")
            ),
            "runner": str(RUNNER_RELATIVE_PATH),
            "runner_sha256": _file_sha256(RUNNER_RELATIVE_PATH),
            "source_files_sha256": {
                str(path): _file_sha256(path) for path in SOURCE_RELATIVE_PATHS
            },
            "model": str(MODEL_RELATIVE_PATH),
            "model_sha256": _file_sha256(MODEL_RELATIVE_PATH),
            "model_manifest": str(MANIFEST_RELATIVE_PATH),
            "model_manifest_sha256": _file_sha256(MANIFEST_RELATIVE_PATH),
            "package_version": __version__,
            "python_version": python_version(),
            "numpy_version": np.__version__,
            "mujoco_version": _mujoco_version(),
        },
        "scenario": {
            **config,
            "trajectories": list(TRAJECTORIES),
            "target_control_rate_hz": 500.0,
            "rbf_input_dimension": 27,
        },
        "results": trajectory_reports,
        "acceptance": {
            "all_four_trajectories_stopped_without_fault": all_stopped,
            "circle_rmse_limit_m": 0.03,
            "circle_rmse_pass": adaptive_rmse <= 0.03,
            "circle_maximum_error_limit_m": 0.08,
            "circle_maximum_error_pass": (
                circle.metrics["impedance_tracking_max_error_m"] <= 0.08
            ),
            "minimum_adaptive_improvement_percent": 10.0,
            "adaptive_improvement_pass": improvement >= 10.0,
        },
        "timing_scope": (
            "Fixed-step target and observed wall-clock telemetry; not a hard "
            "real-time guarantee."
        ),
    }


def _plot_benchmark(
    path: Path,
    adaptive: dict[str, MujocoRunResult],
    baseline: MujocoRunResult,
) -> None:
    """Plot measured trajectories, errors, metrics, and timing telemetry."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    circle = adaptive["circle"]
    circle_error = np.linalg.norm(circle.impedance - circle.actual, axis=1)
    baseline_error = np.linalg.norm(
        baseline.impedance - baseline.actual, axis=1
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    axis = axes[0, 0]
    axis.plot(
        circle.desired[:, 0],
        circle.desired[:, 1],
        "--",
        color="#2ca02c",
        label="desired",
    )
    axis.plot(
        circle.impedance[:, 0],
        circle.impedance[:, 1],
        color="#1f77b4",
        label="impedance",
    )
    axis.plot(
        circle.actual[:, 0],
        circle.actual[:, 1],
        color="#d62728",
        label="NAC actual",
    )
    axis.plot(
        baseline.actual[:, 0],
        baseline.actual[:, 1],
        color="#7f7f7f",
        alpha=0.8,
        label="frozen actual",
    )
    axis.set_title("Circle trajectory from MuJoCo state")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    axis.plot(
        baseline.time,
        baseline_error,
        color="#7f7f7f",
        label="frozen baseline",
    )
    axis.plot(circle.time, circle_error, color="#d62728", label="NAC")
    axis.set_title("Circle impedance tracking error")
    axis.set_xlabel("simulation time [s]")
    axis.set_ylabel(r"$\|x_m-x\|$ [m]")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = axes[1, 0]
    labels = ["circle\nfrozen"] + list(TRAJECTORIES)
    values = [baseline.metrics["impedance_tracking_rmse_m"]] + [
        adaptive[name].metrics["impedance_tracking_rmse_m"]
        for name in TRAJECTORIES
    ]
    colors = ["#7f7f7f"] + ["#d62728"] * len(TRAJECTORIES)
    axis.bar(labels, values, color=colors)
    axis.axhline(0.03, color="#333333", linestyle="--", label="circle limit")
    axis.set_title("Full-dynamics impedance RMSE")
    axis.set_ylabel("RMSE [m]")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    percentiles = ("median", "p95", "p99")
    nac_times = [circle.metrics[f"nac_time_{name}_ms"] for name in percentiles]
    step_times = [
        circle.metrics[f"mujoco_step_time_{name}_ms"] for name in percentiles
    ]
    locations = np.arange(len(percentiles))
    width = 0.36
    axis.bar(locations - width / 2, nac_times, width, label="NAC compute")
    axis.bar(locations + width / 2, step_times, width, label="4 MuJoCo steps")
    axis.set_xticks(locations, percentiles)
    axis.set_title("Observed wall-clock timing (not hard real time)")
    axis.set_ylabel("elapsed [ms]")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)

    figure.suptitle(
        "UR5e + Robotiq full-dynamics NAC benchmark: 500 Hz fixed-step target"
    )
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=170,
        metadata={
            "Title": "UR5e Robotiq MuJoCo tracking benchmark",
            "Software": "ros2_neuro_adaptive_control",
        },
    )
    plt.close(figure)


def main() -> int:
    """Run canonical scenarios and write JSON plus a plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="artifact directory (default: project docs/assets)",
    )
    args = parser.parse_args()
    adaptive, baseline = _run_benchmark()
    report = _build_report(adaptive, baseline)
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "mujoco_tracking_benchmark.json"
    plot_path = output / "mujoco_tracking_benchmark.png"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot_benchmark(plot_path, adaptive, baseline)
    print(json.dumps(report["acceptance"], indent=2, sort_keys=True))
    print("Wrote mujoco_tracking_benchmark.json and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
