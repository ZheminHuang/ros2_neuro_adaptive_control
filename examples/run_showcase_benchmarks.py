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

"""Generate compliance and hidden-joint-drag metrics and animations."""

from __future__ import annotations

import argparse
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
from neuro_adaptive_control.adapters.mujoco_payload_benchmark import (  # noqa: E402
    DEFAULT_PAYLOAD_CASE,
)
from neuro_adaptive_control.adapters.mujoco_showcase_benchmarks import (  # noqa: E402
    ComplianceVariant,
    DragVariant,
    ShowcaseResult,
    compare_compliance,
    compare_joint_drag,
    run_compliance_benchmark,
    run_joint_drag_benchmark,
)
from neuro_adaptive_control.adapters.mujoco_ur5e_adapter import (  # noqa: E402
    MujocoUR5ePlant,
)


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
MODEL_PATH = Path("mujoco/ur5e_robotiq_2f85.xml")
SOURCE_PATHS = (
    Path("examples/run_showcase_benchmarks.py"),
    Path("neuro_adaptive_control/adapters/mujoco_showcase_benchmarks.py"),
    Path("neuro_adaptive_control/adapters/mujoco_ur5e_adapter.py"),
    Path("neuro_adaptive_control/adapters/mujoco_payload_benchmark.py"),
    Path("neuro_adaptive_control/adapters/pose_wrench_to_torque.py"),
    Path("neuro_adaptive_control/core/pose_impedance_model.py"),
    Path("neuro_adaptive_control/core/pose_neuro_adaptive_controller.py"),
    Path("neuro_adaptive_control/core/so3.py"),
)


def _file_hash(path: Path) -> str:
    return sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()


def _history_hash(result: ShowcaseResult) -> str:
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
        "physical_wrench",
        "generalized_wrench",
        "qpos",
    ):
        array = np.ascontiguousarray(getattr(result, name))
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _report(
    soft: ShowcaseResult,
    stiff: ShowcaseResult,
    adaptive: ShowcaseResult,
    frozen: ShowcaseResult,
) -> dict[str, object]:
    import mujoco

    results = (soft, stiff, adaptive, frozen)
    return {
        "schema_version": 1,
        "artifact_kind": "six_dof_nac_showcase_scenarios",
        "traceability": {
            "generator": "examples/run_showcase_benchmarks.py",
            "generator_sha256": _file_hash(
                Path("examples/run_showcase_benchmarks.py")
            ),
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
        "compliance": {
            "contract": (
                "World-frame physical wrench at gripper_pinch; generalized "
                "input h=E(rho)^T w; the same w is applied once by MuJoCo."
            ),
            "comparison": compare_compliance(soft, stiff),
            "soft_metrics": soft.metrics,
            "stiff_metrics": stiff.metrics,
        },
        "joint_drag": {
            "contract": (
                "MuJoCo plant changes selected DOF damping/friction at 4 s; "
                "coefficients and event are absent from the 42D NN input."
            ),
            "comparison": compare_joint_drag(adaptive, frozen),
            "adaptive_metrics": adaptive.metrics,
            "frozen_metrics": frozen.metrics,
        },
        "histories": [
            {
                "scenario": result.scenario,
                "variant": result.variant,
                "deterministic_history_sha256": _history_hash(result),
            }
            for result in results
        ],
        "claim_scope": (
            "Empirical deterministic MuJoCo evidence for the bundled model; "
            "no passivity, hardware-safety, hard-real-time, or universal claim."
        ),
    }


def _render_frames(
    result: ShowcaseResult,
    indices: np.ndarray,
    *,
    payload: bool,
) -> list:
    import mujoco
    from PIL import Image

    kwargs = {}
    if payload:
        case = DEFAULT_PAYLOAD_CASE
        kwargs = {
            "seed": case.seed,
            "payload_mass_kg": case.mass_kg,
            "payload_com_offset_m": case.com_offset_m,
            "payload_inertia_scale": case.inertia_scale,
        }
    else:
        kwargs = {"seed": 83}
    plant = MujocoUR5ePlant(**kwargs)
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


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _dashed_vertical(draw, x: int, top: int, bottom: int, color) -> None:
    for dash_top in range(top + 2, bottom - 1, 10):
        draw.line(
            (x, dash_top, x, min(dash_top + 5, bottom - 1)),
            fill=color,
            width=2,
        )


def _trace(
    draw,
    times: np.ndarray,
    values: np.ndarray,
    *,
    bounds: tuple[int, int, int, int],
    time_range: tuple[float, float],
    value_limit: float,
    color,
) -> None:
    if len(times) < 2:
        return
    left, top, width, height = bounds
    start, end = time_range
    points = []
    for stamp, value in zip(times, values):
        x = left + int(width * (float(stamp) - start) / (end - start))
        normalized = np.clip(float(value) / value_limit, -1.0, 1.0)
        y = top + height // 2 - int(0.46 * height * normalized)
        points.append((x, y))
    draw.line(points, fill=color, width=3)


def _draw_force_overlay(draw, phase: str, offset_x: int) -> None:
    if phase == "lateral_push":
        start = (offset_x + 365, 115)
        end = (offset_x + 285, 115)
        draw.line((start, end), fill=(255, 155, 35), width=7)
        draw.polygon(
            (
                (end[0], end[1]),
                (end[0] + 20, end[1] - 11),
                (end[0] + 20, end[1] + 11),
            ),
            fill=(255, 155, 35),
        )
        draw.text(
            (offset_x + 315, 80),
            "6 N push",
            fill=(255, 190, 70),
            font=_font(14, bold=True),
        )
    if phase == "twist_moment":
        box = (offset_x + 290, 65, offset_x + 390, 165)
        draw.arc(box, start=35, end=315, fill=(255, 155, 35), width=7)
        draw.polygon(
            ((offset_x + 385, 116), (offset_x + 372, 105), (offset_x + 370, 123)),
            fill=(255, 155, 35),
        )
        draw.text(
            (offset_x + 300, 42),
            "0.4 Nm twist",
            fill=(255, 190, 70),
            font=_font(14, bold=True),
        )


def _save_animation(frames: list, webp_path: Path, gif_path: Path, fps: float) -> None:
    from PIL import Image

    duration = int(round(1000.0 / fps))
    frames[0].save(
        webp_path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        lossless=True,
        method=6,
    )
    compatibility = [
        frame.convert("P", palette=Image.ADAPTIVE, colors=192) for frame in frames
    ]
    compatibility[0].save(
        gif_path,
        save_all=True,
        append_images=compatibility[1:],
        duration=duration,
        loop=0,
        optimize=True,
        disposal=2,
    )


def _write_compliance_animation(
    webp_path: Path,
    gif_path: Path,
    soft: ShowcaseResult,
    stiff: ShowcaseResult,
) -> None:
    from PIL import Image, ImageDraw

    fps = 8.0
    start_time = 6.5
    end_time = float(soft.time[-1])
    frame_count = int(np.floor((end_time - start_time) * fps)) + 1
    candidate = np.flatnonzero(soft.time >= start_time)
    indices = np.linspace(candidate[0], candidate[-1], frame_count, dtype=int)
    soft_frames = _render_frames(soft, indices, payload=True)
    stiff_frames = _render_frames(stiff, indices, payload=True)
    frames = []
    top_plot = (980, 42, 285, 112)
    lower_plot = (980, 212, 285, 105)
    for output_index, history_index in enumerate(indices):
        canvas = Image.new("RGB", (1280, 390), color=(20, 22, 26))
        canvas.paste(soft_frames[output_index], (0, 30))
        canvas.paste(stiff_frames[output_index], (480, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (180, 7), "Soft impedance", fill="white", font=_font(15, bold=True)
        )
        draw.text(
            (640, 7), "Stiff impedance", fill="white", font=_font(15, bold=True)
        )
        phase = soft.phase[history_index]
        _draw_force_overlay(draw, phase, 0)
        _draw_force_overlay(draw, phase, 480)
        draw.text(
            (980, 12),
            "Measured compliance",
            fill="white",
            font=_font(14, bold=True),
        )
        for bounds in (top_plot, lower_plot):
            left, top, width, height = bounds
            draw.rectangle(
                (left, top, left + width, top + height),
                outline=(150, 150, 150),
            )
            draw.line(
                (left, top + height // 2, left + width, top + height // 2),
                fill=(80, 80, 80),
            )
        visible = slice(indices[0], history_index + 1)
        _trace(
            draw,
            soft.time[visible],
            1000.0
            * (soft.actual_pose[visible, 1] - soft.desired_pose[visible, 1]),
            bounds=top_plot,
            time_range=(start_time, end_time),
            value_limit=60.0,
            color=(255, 80, 80),
        )
        _trace(
            draw,
            stiff.time[visible],
            1000.0
            * (stiff.actual_pose[visible, 1] - stiff.desired_pose[visible, 1]),
            bounds=top_plot,
            time_range=(start_time, end_time),
            value_limit=60.0,
            color=(80, 160, 255),
        )
        _trace(
            draw,
            soft.time[visible],
            180.0
            / np.pi
            * (soft.actual_pose[visible, 5] - soft.desired_pose[visible, 5]),
            bounds=lower_plot,
            time_range=(start_time, end_time),
            value_limit=2.5,
            color=(255, 80, 80),
        )
        _trace(
            draw,
            stiff.time[visible],
            180.0
            / np.pi
            * (stiff.actual_pose[visible, 5] - stiff.desired_pose[visible, 5]),
            bounds=lower_plot,
            time_range=(start_time, end_time),
            value_limit=2.5,
            color=(80, 160, 255),
        )
        draw.text(
            (980, 160),
            "Y deflection [mm] | red soft | blue stiff",
            fill="white",
            font=_font(12),
        )
        draw.text(
            (980, 325), "Rz deflection [deg]", fill="white", font=_font(12)
        )
        draw.text(
            (10, 368),
            f"t={soft.time[history_index]:05.2f}s | {phase} | same measured wrench",
            fill="white",
            font=_font(13),
        )
        frames.append(canvas)
    _save_animation(frames, webp_path, gif_path, fps)


def _write_drag_animation(
    webp_path: Path,
    gif_path: Path,
    adaptive: ShowcaseResult,
    frozen: ShowcaseResult,
) -> None:
    from PIL import Image, ImageDraw

    fps = 8.0
    frame_count = int(np.floor(adaptive.time[-1] * fps)) + 1
    indices = np.linspace(0, len(adaptive.time) - 1, frame_count, dtype=int)
    adaptive_frames = _render_frames(adaptive, indices, payload=False)
    frozen_frames = _render_frames(frozen, indices, payload=False)
    frames = []
    top_plot = (980, 42, 285, 112)
    lower_plot = (980, 212, 285, 105)
    event_time = float(adaptive.metrics["event_time_sec"])
    event_x = top_plot[0] + int(top_plot[2] * event_time / adaptive.time[-1])
    for output_index, history_index in enumerate(indices):
        canvas = Image.new("RGB", (1280, 390), color=(20, 22, 26))
        canvas.paste(adaptive_frames[output_index], (0, 30))
        canvas.paste(frozen_frames[output_index], (480, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (180, 7), "Adaptive NAC", fill="white", font=_font(15, bold=True)
        )
        draw.text(
            (620, 7),
            "Frozen at disturbance",
            fill="white",
            font=_font(15, bold=True),
        )
        draw.text(
            (980, 12),
            "Tracking after hidden drag",
            fill="white",
            font=_font(14, bold=True),
        )
        for bounds in (top_plot, lower_plot):
            left, top, width, height = bounds
            draw.rectangle(
                (left, top, left + width, top + height),
                outline=(150, 150, 150),
            )
            _dashed_vertical(draw, event_x, top, top + height, (190, 90, 220))
        visible = slice(0, history_index + 1)
        adaptive_position = 1000.0 * np.linalg.norm(
            adaptive.desired_pose[visible, :3]
            - adaptive.actual_pose[visible, :3],
            axis=1,
        )
        frozen_position = 1000.0 * np.linalg.norm(
            frozen.desired_pose[visible, :3] - frozen.actual_pose[visible, :3],
            axis=1,
        )
        adaptive_rotation = 1000.0 * np.linalg.norm(
            adaptive.desired_pose[visible, 3:]
            - adaptive.actual_pose[visible, 3:],
            axis=1,
        )
        frozen_rotation = 1000.0 * np.linalg.norm(
            frozen.desired_pose[visible, 3:] - frozen.actual_pose[visible, 3:],
            axis=1,
        )
        for values, bounds, color in (
            (adaptive_position, top_plot, (255, 80, 80)),
            (frozen_position, top_plot, (80, 160, 255)),
            (adaptive_rotation, lower_plot, (255, 80, 80)),
            (frozen_rotation, lower_plot, (80, 160, 255)),
        ):
            _trace(
                draw,
                adaptive.time[visible],
                values,
                bounds=bounds,
                time_range=(0.0, adaptive.time[-1]),
                value_limit=2.5,
                color=color,
            )
        draw.text(
            (980, 160),
            "Position error [mm] | red adaptive | blue frozen",
            fill="white",
            font=_font(12),
        )
        draw.text(
            (980, 325),
            "Rotation-vector error [mrad]",
            fill="white",
            font=_font(12),
        )
        if adaptive.time[history_index] >= event_time:
            for offset in (0, 480):
                draw.ellipse(
                    (offset + 210, 125, offset + 265, 180),
                    outline=(255, 145, 35),
                    width=5,
                )
                draw.text(
                    (offset + 270, 145),
                    "SIMULATED DRAG",
                    fill=(255, 175, 70),
                    font=_font(13, bold=True),
                )
        draw.text(
            (10, 368),
            (
                f"t={adaptive.time[history_index]:05.2f}s | "
                f"{adaptive.phase[history_index]} | plant change hidden from NN input"
            ),
            fill="white",
            font=_font(13),
        )
        frames.append(canvas)
    _save_animation(frames, webp_path, gif_path, fps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--skip-animations", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    soft = run_compliance_benchmark(ComplianceVariant.SOFT)
    stiff = run_compliance_benchmark(ComplianceVariant.STIFF)
    adaptive = run_joint_drag_benchmark(DragVariant.ADAPTIVE)
    frozen = run_joint_drag_benchmark(DragVariant.FROZEN)
    report = _report(soft, stiff, adaptive, frozen)
    report_path = output / "showcase_scenarios_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not arguments.skip_animations:
        _write_compliance_animation(
            output / "compliance_comparison.webp",
            output / "compliance_comparison.gif",
            soft,
            stiff,
        )
        _write_drag_animation(
            output / "joint_drag_comparison.webp",
            output / "joint_drag_comparison.gif",
            adaptive,
            frozen,
        )
    print(
        json.dumps(
            {
                "compliance": report["compliance"]["comparison"],
                "joint_drag": report["joint_drag"]["comparison"],
            },
            indent=2,
        )
    )
    print(f"wrote {report_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
