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

"""Generate traceable MuJoCo grasp, lift, hold, and release artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from neuro_adaptive_control.adapters.mujoco_grasp import (  # noqa: E402
    GraspRunConfig,
    GraspRunResult,
    run_grasp_demo,
)


MODEL_RELATIVE_PATH = Path("mujoco/ur5e_robotiq_2f85.xml")
MANIFEST_RELATIVE_PATH = Path("mujoco/SHA256SUMS")
RUNNER_RELATIVE_PATH = Path("neuro_adaptive_control/adapters/mujoco_grasp.py")
SOURCE_RELATIVE_PATHS = (
    RUNNER_RELATIVE_PATH,
    Path("neuro_adaptive_control/adapters/mujoco_simulation.py"),
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


def _history_sha256(result: GraspRunResult) -> str:
    """Digest deterministic grasp histories without wall-clock telemetry."""
    digest = sha256()
    digest.update("phase".encode("ascii"))
    digest.update("\0".join(result.phase).encode("utf-8"))
    for name in (
        "time",
        "tcp_position",
        "object_position",
        "gripper_opening",
        "arm_torque",
        "contact_force",
        "bilateral_contact",
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


def _build_report(
    config: GraspRunConfig,
    result: GraspRunResult,
) -> dict[str, object]:
    """Build a traceable machine-readable grasp evidence report."""
    metrics = result.metrics
    return {
        "schema_version": 1,
        "artifact_kind": "full_dynamics_grasp_benchmark",
        "traceability": {
            "generator": "examples/run_mujoco_grasp.py",
            "generator_sha256": _file_sha256(
                Path("examples/run_mujoco_grasp.py")
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
            "deterministic_history_sha256": _history_sha256(result),
        },
        "scenario": {
            **asdict(config),
            "target_control_rate_hz": 500.0,
            "mujoco_timestep_sec": 0.0005,
            "substeps_per_control": 4,
            "phases": [
                "pregrasp",
                "descend",
                "close",
                "lift",
                "hold",
                "lower",
                "release",
                "retreat",
            ],
        },
        "metrics": metrics,
        "acceptance": {
            "runner_success": metrics["success"],
            "stopped_without_fault": (
                metrics["state"] == "stopped"
                and metrics["fault_reason"] == ""
            ),
            "minimum_lift_height_m": 0.05,
            "lift_height_pass": metrics["object_lift_height_m"] >= 0.05,
            "minimum_hold_duration_sec": 2.0,
            "hold_duration_pass": metrics["hold_duration_sec"] >= 2.0,
            "maximum_hold_drop_m": 0.005,
            "hold_drop_pass": metrics["hold_drop_m"] <= 0.005,
            "minimum_hold_bilateral_contact_ratio": 0.90,
            "bilateral_contact_pass": (
                metrics["hold_bilateral_contact_ratio"] >= 0.90
            ),
            "contact_force_limit_n": config.maximum_contact_force_n,
            "contact_force_pass": (
                metrics["maximum_contact_force_n"]
                <= config.maximum_contact_force_n
            ),
            "gripper_effort_limit_n": config.maximum_gripper_effort_n,
            "gripper_effort_pass": (
                metrics["maximum_gripper_effort_n"]
                <= config.maximum_gripper_effort_n + 1.0e-9
            ),
        },
        "timing_scope": (
            "Fixed-step simulation with observed wall-clock telemetry; not a "
            "hard real-time guarantee."
        ),
    }


def _phase_spans(result: GraspRunResult) -> list[tuple[str, float, float]]:
    """Convert per-sample phases into contiguous plot spans."""
    spans: list[tuple[str, float, float]] = []
    start = 0
    for index in range(1, len(result.phase) + 1):
        if index == len(result.phase) or result.phase[index] != result.phase[start]:
            spans.append(
                (
                    result.phase[start],
                    float(result.time[start]),
                    float(result.time[index - 1]),
                )
            )
            start = index
    return spans


def _shade_phases(axis, result: GraspRunResult) -> None:
    """Add restrained phase shading to one time-series axis."""
    colors = ("#f3f3f3", "#e8eef7")
    for index, (phase, start, end) in enumerate(_phase_spans(result)):
        axis.axvspan(start, end, color=colors[index % 2], alpha=0.5, zorder=0)
        axis.text(
            (start + end) / 2.0,
            0.98,
            phase,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            color="#555555",
            fontsize=6,
            rotation=90,
        )


def _plot_grasp(path: Path, result: GraspRunResult) -> None:
    """Plot measured object motion, gripper state, contact, and arm effort."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)

    axis = axes[0, 0]
    _shade_phases(axis, result)
    axis.plot(result.time, result.tcp_position[:, 2], label="TCP z")
    axis.plot(
        result.time,
        result.object_position[:, 2],
        color="#d62728",
        label="object z",
    )
    axis.set_title("Measured lift and return")
    axis.set_ylabel("world z [m]")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = axes[0, 1]
    _shade_phases(axis, result)
    axis.plot(
        result.time,
        result.gripper_opening * 1000.0,
        color="#1f77b4",
        label="opening",
    )
    axis.fill_between(
        result.time,
        0.0,
        1.0,
        where=result.bilateral_contact,
        transform=axis.get_xaxis_transform(),
        color="#9467bd",
        alpha=0.18,
        label="bilateral contact",
    )
    axis.set_title("Dynamic gripper state")
    axis.set_ylabel("opening [mm]")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = axes[1, 0]
    _shade_phases(axis, result)
    axis.plot(result.time, result.contact_force, color="#9467bd")
    axis.set_title("Robot-environment contact force")
    axis.set_xlabel("simulation time [s]")
    axis.set_ylabel("summed contact norm [N]")
    axis.grid(alpha=0.25)

    axis = axes[1, 1]
    _shade_phases(axis, result)
    axis.plot(
        result.time,
        np.linalg.norm(result.arm_torque, axis=1),
        color="#ff7f0e",
    )
    axis.set_title("Applied UR5e arm torque norm")
    axis.set_xlabel("simulation time [s]")
    axis.set_ylabel(r"$\|\tau\|$ [N m]")
    axis.grid(alpha=0.25)

    figure.suptitle("MuJoCo UR5e + Robotiq grasp/lift/hold evidence")
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=170,
        metadata={
            "Title": "UR5e Robotiq MuJoCo grasp benchmark",
            "Software": "ros2_neuro_adaptive_control",
        },
    )
    plt.close(figure)


def main() -> int:
    """Run the canonical grasp and write JSON plus a plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="artifact directory (default: project docs/assets)",
    )
    args = parser.parse_args()
    config = GraspRunConfig()
    result = run_grasp_demo(config)
    report = _build_report(config, result)
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "mujoco_grasp_benchmark.json"
    plot_path = output / "mujoco_grasp_benchmark.png"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot_grasp(plot_path, result)
    print(json.dumps(report["acceptance"], indent=2, sort_keys=True))
    print("Wrote mujoco_grasp_benchmark.json and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
