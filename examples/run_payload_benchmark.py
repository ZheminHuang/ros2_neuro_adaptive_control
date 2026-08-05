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

"""Generate metrics, a result plot, and synchronized MuJoCo animations."""

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
            "duration_sec": 15.0,
            "loaded_trajectory": "one_smooth_xy_circle",
            "loaded_circle_radius_m": 0.04,
            "held_out_payload_mass_kg": [0.50, 0.75, 1.00],
            "gripper_effort_n": 5.0,
            "external_wrench_mode": "none",
            "payload_parameters_visible_to_adaptive_nac": False,
            "presentation": {
                "rendered_geom_groups": [2],
                "hidden_collision_geom_groups": [3],
                "primary_animation": "lossless_animated_webp",
                "compatibility_animation": "palette_quantized_gif",
            },
            "nac": {
                "input_dim": 42,
                "hidden_dim": 120,
                "lambda_diagonal": [10.0] * 6,
                "feedback_diagonal": [250.0, 250.0, 350.0, 50.0, 50.0, 50.0],
                "F0_diagonal": 200.0,
                "F1_diagonal": 200.0,
                "kappa": 0.05,
                "robust_diagonal": [0.3, 0.3, 0.3, 0.2, 0.2, 0.2],
                "ideal_weight_bound": 100.0,
                "initial_weight_scale": 0.01,
                "both_layers_nonzero_at_reset": True,
                "effective_torque_limits_nm": [80.0, 80.0, 80.0, 28.0, 28.0, 28.0],
            },
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
    plant.model.vis.headlight.ambient[:] = (0.48, 0.48, 0.48)
    plant.model.vis.headlight.diffuse[:] = (0.86, 0.86, 0.86)
    plant.model.vis.headlight.specular[:] = (0.25, 0.25, 0.25)
    plant.model.vis.quality.offsamples = 4
    plant.model.vis.quality.shadowsize = 2048
    renderer = mujoco.Renderer(plant.model, height=360, width=480)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = np.array((-0.09, 0.43, 0.31))
    camera.distance = 1.02
    camera.azimuth = 138.0
    camera.elevation = -18.0
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 0
    scene_option.geomgroup[2] = 1
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
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
    origin_x = 980
    top = 42
    width = 285
    height = 112
    draw.text(
        (origin_x, 12),
        "Synchronized tracking error",
        fill="white",
        font=title_font,
    )
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
            y = top + height - int(height * min(float(value) / 0.05, 1.0))
            points.append((x, y))
        draw.line(points, fill=color, width=3)
    second_top = 212
    second_height = 105
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
                second_height * min(float(value) / 65.0, 1.0)
            )
            points.append((x, y))
        draw.line(points, fill=(255, 190, 60), width=3)
    event_time = float(adaptive.metrics["payload_acquisition_time_sec"])
    event_x = origin_x + int(width * event_time / adaptive.time[-1])
    draw.line((event_x, top, event_x, second_top + second_height), fill=(190, 90, 220))
    draw.text(
        (origin_x, 162),
        "red NAC | blue nominal | position error",
        fill="white",
        font=font,
    )
    draw.text(
        (origin_x, 325),
        "yellow NN dynamics estimate",
        fill="white",
        font=font,
    )
    if adaptive.time[index] >= event_time:
        draw.text(
            (origin_x + 55, 188),
            "PAYLOAD ACQUIRED",
            fill=(220, 120, 255),
            font=title_font,
        )


def _write_animations(
    webp_path: Path,
    gif_path: Path,
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
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
    frames = []
    for output_index, history_index in enumerate(indices):
        canvas = Image.new("RGB", (1280, 390), color=(20, 22, 26))
        canvas.paste(adaptive_frames[output_index], (0, 30))
        canvas.paste(nominal_frames[output_index], (480, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text((185, 7), "Adaptive NAC", fill="white", font=title_font)
        draw.text(
            (635, 7),
            "Nominal model-based",
            fill="white",
            font=title_font,
        )
        draw.text(
            (10, 368),
            (
                f"t={adaptive.time[history_index]:05.2f}s | "
                f"phase={adaptive.phase[history_index]} | "
                f"payload={adaptive.config.payload.mass_kg:.2f} kg"
            ),
            fill="white",
            font=font,
        )
        _draw_plot_panel(canvas, adaptive, nominal, int(history_index))
        frames.append(canvas)
    frames[0].save(
        webp_path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000.0 / target_fps)),
        loop=0,
        lossless=True,
        method=6,
    )
    compatibility_frames = [
        frame.convert(
            "P",
            palette=Image.ADAPTIVE,
            colors=192,
        )
        for frame in frames
    ]
    compatibility_frames[0].save(
        gif_path,
        save_all=True,
        append_images=compatibility_frames[1:],
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
    parser.add_argument(
        "--skip-animations",
        "--skip-gif",
        dest="skip_animations",
        action="store_true",
        help="skip both the primary WebP and compatibility GIF",
    )
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
    webp_path = output / "payload_benchmark_comparison.webp"
    gif_path = output / "payload_benchmark_comparison.gif"
    report_path.write_text(
        json.dumps(_report(suite.trials, suite.metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_results(plot_path, adaptive, frozen, nominal, oracle)
    if not arguments.skip_animations:
        _write_animations(webp_path, gif_path, adaptive, nominal)
    print(json.dumps(suite.metrics, indent=2))
    print(f"wrote {report_path.relative_to(PROJECT_ROOT)}")
    print(f"wrote {plot_path.relative_to(PROJECT_ROOT)}")
    if not arguments.skip_animations:
        print(f"wrote {webp_path.relative_to(PROJECT_ROOT)}")
        print(f"wrote {gif_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
