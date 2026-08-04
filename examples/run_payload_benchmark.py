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

"""Generate traceable unknown-payload metrics, result plot, and MuJoCo GIF."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from platform import python_version
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuro_adaptive_control import __version__  # noqa: E402
from neuro_adaptive_control.adapters.mujoco_payload_benchmark import (  # noqa: E402
    BenchmarkController,
    DEFAULT_PAYLOAD_CASE,
    PayloadBenchmarkResult,
    run_payload_suite,
)
from neuro_adaptive_control.adapters.mujoco_ur5e_adapter import (  # noqa: E402
    MujocoUR5ePlant,
)


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
MODEL_PATH = Path("mujoco/ur5e_robotiq_2f85.xml")
RUNNER_PATH = Path(
    "neuro_adaptive_control/adapters/mujoco_payload_benchmark.py"
)
SOURCE_PATHS = (
    RUNNER_PATH,
    Path("neuro_adaptive_control/adapters/model_based_controller.py"),
    Path("neuro_adaptive_control/adapters/mujoco_ur5e_adapter.py"),
    Path("neuro_adaptive_control/adapters/pose_wrench_to_torque.py"),
    Path("neuro_adaptive_control/core/pose_impedance_model.py"),
    Path("neuro_adaptive_control/core/pose_neuro_adaptive_controller.py"),
    Path("neuro_adaptive_control/core/so3.py"),
    Path("neuro_adaptive_control/core/two_layer_network.py"),
)


def _file_hash(path: Path) -> str:
    return sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()


def _history_hash(result: PayloadBenchmarkResult) -> str:
    digest = sha256("\0".join(result.phase).encode("utf-8"))
    for name in (
        "time",
        "desired_pose",
        "impedance_pose",
        "actual_pose",
        "generalized_command",
        "neural_estimate",
        "arm_torque",
        "weight_norm",
        "bilateral_contact",
        "payload_acquired",
        "contact_force",
        "object_position",
        "qpos",
    ):
        array = np.ascontiguousarray(getattr(result, name))
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _select(
    results: tuple[PayloadBenchmarkResult, ...],
    controller: BenchmarkController,
) -> PayloadBenchmarkResult:
    for result in results:
        if (
            result.config.payload == DEFAULT_PAYLOAD_CASE
            and result.config.controller == controller
        ):
            return result
    raise RuntimeError(f"missing showcase result for {controller.value}")


def _report(results, metrics: dict[str, object]) -> dict[str, object]:
    import mujoco

    trials = []
    for result in results:
        payload = result.config.payload
        trials.append(
            {
                "controller": result.config.controller.value,
                "payload": {
                    "name": payload.name,
                    "mass_kg": payload.mass_kg,
                    "com_offset_m": list(payload.com_offset_m),
                    "inertia_scale": payload.inertia_scale,
                    "seed": payload.seed,
                },
                "metrics": result.metrics,
                "deterministic_history_sha256": _history_hash(result),
            }
        )
    nominal = _select(results, BenchmarkController.NOMINAL_MODEL_BASED)
    adaptive = _select(results, BenchmarkController.ADAPTIVE_NAC)
    nominal_position_degradation = (
        float(nominal.metrics["loaded_position_rmse_m"])
        / float(nominal.metrics["unloaded_position_rmse_m"])
    )
    nominal_orientation_degradation = (
        float(nominal.metrics["loaded_orientation_rmse_rad"])
        / float(nominal.metrics["unloaded_orientation_rmse_rad"])
    )
    return {
        "schema_version": 1,
        "artifact_kind": "six_dof_unknown_payload_benchmark",
        "traceability": {
            "generator": "examples/run_payload_benchmark.py",
            "generator_sha256": _file_hash(
                Path("examples/run_payload_benchmark.py")
            ),
            "runner": str(RUNNER_PATH),
            "runner_sha256": _file_hash(RUNNER_PATH),
            "source_files_sha256": {
                str(path): _file_hash(path) for path in SOURCE_PATHS
            },
            "model": str(MODEL_PATH),
            "model_sha256": _file_hash(MODEL_PATH),
            "package_version": __version__,
            "python_version": python_version(),
            "numpy_version": np.__version__,
            "mujoco_version": mujoco.__version__,
        },
        "scenario": {
            "target_control_rate_hz": 500.0,
            "mujoco_timestep_sec": 0.0005,
            "substeps_per_control": 4,
            "external_wrench_mode": "none",
            "payload_parameters_visible_to_adaptive_nac": False,
            "phases": [
                "unloaded_out",
                "unloaded_return",
                "approach",
                "grasp",
                "lift",
                "loaded_tracking",
                "lower",
                "release",
                "retreat",
            ],
        },
        "aggregate_adaptation_gate": metrics,
        "showcase_dynamics_change": {
            "adaptive_loaded_position_rmse_m": adaptive.metrics[
                "loaded_position_rmse_m"
            ],
            "adaptive_loaded_orientation_rmse_rad": adaptive.metrics[
                "loaded_orientation_rmse_rad"
            ],
            "nominal_loaded_position_rmse_m": nominal.metrics[
                "loaded_position_rmse_m"
            ],
            "nominal_loaded_orientation_rmse_rad": nominal.metrics[
                "loaded_orientation_rmse_rad"
            ],
            "nominal_position_rmse_loaded_to_unloaded_ratio": (
                nominal_position_degradation
            ),
            "nominal_orientation_rmse_loaded_to_unloaded_ratio": (
                nominal_orientation_degradation
            ),
        },
        "trials": trials,
        "claim_scope": (
            "Empirical deterministic MuJoCo evidence for these held-out "
            "payloads; no hard-real-time or universal superiority claim."
        ),
    }


def _plot_results(
    path: Path,
    adaptive: PayloadBenchmarkResult,
    frozen: PayloadBenchmarkResult,
    nominal: PayloadBenchmarkResult,
    oracle: PayloadBenchmarkResult,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ros2_nac_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex="col")
    time = adaptive.time
    acquired = float(adaptive.metrics["payload_acquisition_time_sec"])
    colors = {
        "adaptive": "#d62728",
        "frozen": "#7f7f7f",
        "nominal": "#1f77b4",
        "oracle": "#2ca02c",
    }
    for dimension, label in enumerate(("x", "y", "z")):
        axes[0, 0].plot(
            time,
            adaptive.desired_pose[:, dimension],
            "--",
            linewidth=1.0,
            label=f"desired {label}",
        )
        axes[0, 0].plot(
            time,
            adaptive.actual_pose[:, dimension],
            linewidth=1.0,
            label=f"NAC {label}",
        )
    for dimension, label in enumerate(("rx", "ry", "rz"), start=3):
        axes[0, 1].plot(
            time,
            adaptive.desired_pose[:, dimension],
            "--",
            linewidth=1.0,
            label=f"desired {label}",
        )
        axes[0, 1].plot(
            time,
            adaptive.actual_pose[:, dimension],
            linewidth=1.0,
            label=f"NAC {label}",
        )
    controllers = (
        ("adaptive", adaptive),
        ("frozen", frozen),
        ("nominal", nominal),
        ("oracle", oracle),
    )
    for name, result in controllers:
        position_error = np.linalg.norm(
            result.desired_pose[:, :3] - result.actual_pose[:, :3], axis=1
        )
        orientation_error = np.linalg.norm(
            result.desired_pose[:, 3:] - result.actual_pose[:, 3:], axis=1
        )
        axes[1, 0].plot(
            result.time,
            1000.0 * position_error,
            label=name,
            color=colors[name],
        )
        axes[1, 1].plot(
            result.time,
            180.0 / np.pi * orientation_error,
            label=name,
            color=colors[name],
        )
    axes[2, 0].plot(
        time,
        np.linalg.norm(adaptive.neural_estimate, axis=1),
        color=colors["adaptive"],
        label="adaptive NN estimate",
    )
    axes[2, 0].plot(
        time,
        np.linalg.norm(frozen.neural_estimate, axis=1),
        color=colors["frozen"],
        label="frozen NN estimate",
    )
    labels = ("adaptive", "frozen", "nominal", "oracle")
    position_values = [
        1000.0 * float(result.metrics["loaded_position_rmse_m"])
        for _, result in controllers
    ]
    orientation_values = [
        180.0 / np.pi * float(result.metrics["loaded_orientation_rmse_rad"])
        for _, result in controllers
    ]
    locations = np.arange(len(labels))
    width = 0.38
    axes[2, 1].bar(
        locations - width / 2.0,
        position_values,
        width,
        label="position [mm]",
    )
    axes[2, 1].bar(
        locations + width / 2.0,
        orientation_values,
        width,
        label="orientation [deg]",
    )
    axes[2, 1].set_xticks(locations, labels, rotation=15)
    axes[2, 1].set_title("Loaded-phase RMSE")
    axes[2, 1].legend(fontsize=8)
    axes[0, 0].set_title("Adaptive NAC XYZ tracking")
    axes[0, 1].set_title("Adaptive NAC rotation-vector tracking")
    axes[1, 0].set_title("Position error norm")
    axes[1, 1].set_title("Orientation error norm")
    axes[2, 0].set_title("Learned dynamics compensation")
    axes[0, 0].set_ylabel("position [m]")
    axes[0, 1].set_ylabel("rotation vector [rad]")
    axes[1, 0].set_ylabel("error [mm]")
    axes[1, 1].set_ylabel("error [deg]")
    axes[2, 0].set_ylabel("NN output norm")
    axes[2, 0].set_xlabel("simulation time [s]")
    for axis in axes.flat:
        axis.grid(alpha=0.22)
        if axis is not axes[2, 1]:
            axis.axvline(acquired, color="#9467bd", linestyle=":", linewidth=1)
    axes[0, 0].legend(ncol=2, fontsize=7)
    axes[0, 1].legend(ncol=2, fontsize=7)
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    axes[2, 0].legend(fontsize=8)
    figure.suptitle(
        "UR5e + Robotiq: dynamics change after physical payload acquisition"
    )
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=160,
        metadata={
            "Title": "Six-DoF unknown-payload benchmark",
            "Software": "ros2_neuro_adaptive_control",
        },
    )
    plt.close(figure)


def _render_frames(result: PayloadBenchmarkResult, indices: np.ndarray) -> list:
    import mujoco
    from PIL import Image

    payload = result.config.payload
    plant = MujocoUR5ePlant(
        seed=payload.seed,
        payload_mass_kg=payload.mass_kg,
        payload_com_offset_m=payload.com_offset_m,
        payload_inertia_scale=payload.inertia_scale,
    )
    plant.model.vis.headlight.active = 1
    plant.model.vis.headlight.ambient[:] = (0.4, 0.4, 0.4)
    plant.model.vis.headlight.diffuse[:] = (0.8, 0.8, 0.8)
    renderer = mujoco.Renderer(plant.model, height=240, width=320)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = np.array((-0.08, 0.40, 0.30))
    camera.distance = 1.15
    camera.azimuth = 135.0
    camera.elevation = -20.0
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 1
    frames = []
    try:
        for index in indices:
            plant.data.qpos[:] = result.qpos[index]
            plant.data.qvel[:] = 0.0
            mujoco.mj_forward(plant.model, plant.data)
            renderer.update_scene(
                plant.data,
                camera=camera,
                scene_option=scene_option,
            )
            frames.append(Image.fromarray(renderer.render().copy()))
    finally:
        renderer.close()
    return frames


def _draw_plot_panel(image, adaptive, nominal, index: int) -> None:
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    origin_x = 654
    top = 38
    width = 292
    height = 82
    draw.text((origin_x, 8), "Synchronized tracking error", fill="white", font=font)
    draw.rectangle(
        (origin_x, top, origin_x + width, top + height),
        outline=(150, 150, 150),
    )
    time = adaptive.time[: index + 1]
    traces = (
        (
            np.linalg.norm(
                adaptive.desired_pose[: index + 1, :3]
                - adaptive.actual_pose[: index + 1, :3],
                axis=1,
            ),
            (255, 80, 80),
        ),
        (
            np.linalg.norm(
                nominal.desired_pose[: index + 1, :3]
                - nominal.actual_pose[: index + 1, :3],
                axis=1,
            ),
            (80, 160, 255),
        ),
    )
    for values, color in traces:
        if len(values) < 2:
            continue
        points = []
        for stamp, value in zip(time, values):
            x = origin_x + int(width * stamp / adaptive.time[-1])
            y = top + height - int(height * min(float(value) / 0.08, 1.0))
            points.append((x, y))
        draw.line(points, fill=color, width=2)
    second_top = 156
    second_height = 70
    draw.rectangle(
        (origin_x, second_top, origin_x + width, second_top + second_height),
        outline=(150, 150, 150),
    )
    nn_values = np.linalg.norm(adaptive.neural_estimate[: index + 1], axis=1)
    if len(nn_values) >= 2:
        points = []
        for stamp, value in zip(time, nn_values):
            x = origin_x + int(width * stamp / adaptive.time[-1])
            y = second_top + second_height - int(
                second_height * min(float(value) / 55.0, 1.0)
            )
            points.append((x, y))
        draw.line(points, fill=(255, 190, 60), width=2)
    event_time = float(adaptive.metrics["payload_acquisition_time_sec"])
    event_x = origin_x + int(width * event_time / adaptive.time[-1])
    draw.line((event_x, top, event_x, second_top + second_height), fill=(190, 90, 220))
    draw.text((origin_x, 126), "red NAC | blue nominal | position error", fill="white")
    draw.text((origin_x, 232), "yellow NN dynamics estimate", fill="white")
    if adaptive.time[index] >= event_time:
        draw.text((origin_x + 70, 142), "PAYLOAD ACQUIRED", fill=(220, 120, 255))


def _write_gif(
    path: Path,
    adaptive: PayloadBenchmarkResult,
    nominal: PayloadBenchmarkResult,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    target_fps = 8.0
    frame_count = int(np.floor(adaptive.time[-1] * target_fps)) + 1
    indices = np.linspace(
        0,
        len(adaptive.time) - 1,
        frame_count,
        dtype=int,
    )
    adaptive_frames = _render_frames(adaptive, indices)
    nominal_frames = _render_frames(nominal, indices)
    font = ImageFont.load_default()
    frames = []
    for output_index, history_index in enumerate(indices):
        canvas = Image.new("RGB", (960, 260), color=(20, 22, 26))
        canvas.paste(adaptive_frames[output_index], (0, 20))
        canvas.paste(nominal_frames[output_index], (320, 20))
        draw = ImageDraw.Draw(canvas)
        draw.text((105, 5), "Adaptive NAC", fill="white", font=font)
        draw.text((400, 5), "Nominal model-based", fill="white", font=font)
        draw.text(
            (8, 244),
            f"t={adaptive.time[history_index]:05.2f}s",
            fill="white",
            font=font,
        )
        _draw_plot_panel(canvas, adaptive, nominal, int(history_index))
        frames.append(
            canvas.convert(
                "P",
                palette=Image.ADAPTIVE,
                colors=128,
            )
        )
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000.0 / target_fps)),
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--skip-gif", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    suite = run_payload_suite(include_model_based_showcase=True)
    adaptive = _select(suite.trials, BenchmarkController.ADAPTIVE_NAC)
    frozen = _select(suite.trials, BenchmarkController.FROZEN_AT_PAYLOAD)
    nominal = _select(suite.trials, BenchmarkController.NOMINAL_MODEL_BASED)
    oracle = _select(suite.trials, BenchmarkController.ORACLE_MODEL_BASED)
    report_path = output / "payload_benchmark_metrics.json"
    plot_path = output / "payload_benchmark_results.png"
    gif_path = output / "payload_benchmark_comparison.gif"
    report_path.write_text(
        json.dumps(_report(suite.trials, suite.metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_results(plot_path, adaptive, frozen, nominal, oracle)
    if not arguments.skip_gif:
        _write_gif(gif_path, adaptive, nominal)
    print(json.dumps(suite.metrics, indent=2))
    print(f"wrote {report_path.relative_to(PROJECT_ROOT)}")
    print(f"wrote {plot_path.relative_to(PROJECT_ROOT)}")
    if not arguments.skip_gif:
        print(f"wrote {gif_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
