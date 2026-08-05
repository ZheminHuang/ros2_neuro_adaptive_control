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
from examples.showcase_rendering import (  # noqa: E402
    EVENT_COLOR,
    HIGHER_COLOR,
    LOWER_COLOR,
    NAC_COLOR,
    NOMINAL_COLOR,
    WRENCH_COLOR,
    Trace,
    draw_metric_panel,
    nice_upper_limit,
    wrench_connectors,
)


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
MODEL_PATH = Path("mujoco/ur5e_robotiq_2f85.xml")
SOURCE_PATHS = (
    Path("examples/run_showcase_benchmarks.py"),
    Path("examples/showcase_rendering.py"),
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
    lower: ShowcaseResult,
    higher: ShowcaseResult,
    adaptive: ShowcaseResult,
    nominal: ShowcaseResult,
    frozen: ShowcaseResult,
) -> dict[str, object]:
    import mujoco

    results = (lower, higher, adaptive, nominal, frozen)
    return {
        "schema_version": 2,
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
            "visualization": {
                "source": "recorded physical_wrench applied by MuJoCo",
                "application_site": "gripper_pinch",
                "force_marker": "world-direction 3D arrow",
                "moment_marker": "right-hand 3D ring",
                "visual_geometry_affects_dynamics": False,
            },
            "comparison": compare_compliance(lower, higher),
            "lower_stiffness_metrics": lower.metrics,
            "higher_stiffness_metrics": higher.metrics,
        },
        "joint_drag": {
            "contract": (
                "MuJoCo plant changes selected DOF damping/friction at 4 s; "
                "the coefficients and event are absent from controller "
                "observations, and the nominal controller model stays fixed."
            ),
            "comparison": compare_joint_drag(adaptive, nominal, frozen),
            "adaptive_metrics": adaptive.metrics,
            "nominal_model_based_metrics": nominal.metrics,
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
    tcp_site_id = mujoco.mj_name2id(
        plant.model,
        mujoco.mjtObj.mjOBJ_SITE,
        "gripper_pinch",
    )
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
            _add_recorded_wrench_visual(
                renderer,
                plant.data.site_xpos[tcp_site_id],
                result.physical_wrench[index],
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


def _add_recorded_wrench_visual(renderer, point, physical_wrench) -> None:
    """Append MuJoCo decorations derived from the exact applied wrench."""
    import mujoco

    rgba = np.asarray(
        (*np.asarray(WRENCH_COLOR) / 255.0, 1.0),
        dtype=np.float32,
    )
    identity = np.eye(3).reshape(-1)
    for connector in wrench_connectors(point, physical_wrench):
        if renderer.scene.ngeom >= renderer.scene.maxgeom:
            raise RuntimeError("MuJoCo scene has no room for wrench geometry")
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        geom_type = (
            mujoco.mjtGeom.mjGEOM_ARROW
            if connector.kind == "arrow"
            else mujoco.mjtGeom.mjGEOM_LINE
        )
        mujoco.mjv_initGeom(
            geom,
            geom_type,
            np.zeros(3),
            np.zeros(3),
            identity,
            rgba,
        )
        mujoco.mjv_connector(
            geom,
            geom_type,
            connector.width,
            connector.start,
            connector.end,
        )
        geom.emission = 1.0
        renderer.scene.ngeom += 1


def _draw_status_badge(draw, text: str, offset_x: int) -> None:
    draw.rounded_rectangle(
        (offset_x + 286, 45, offset_x + 461, 72),
        radius=7,
        fill=(70, 47, 22),
        outline=WRENCH_COLOR,
        width=2,
    )
    draw.text(
        (offset_x + 297, 51),
        text,
        fill=WRENCH_COLOR,
        font=_font(11, bold=True),
    )


def _draw_wrench_badge(draw, phase: str, offset_x: int) -> None:
    if phase == "lateral_push":
        _draw_status_badge(draw, "6 N TCP PUSH", offset_x)
    elif phase == "twist_moment":
        _draw_status_badge(draw, "1.0 N·m TCP TWIST", offset_x)


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
    lower: ShowcaseResult,
    higher: ShowcaseResult,
) -> None:
    from PIL import Image, ImageDraw

    fps = 8.0
    start_time = 6.5
    end_time = float(lower.time[-1])
    frame_count = int(np.floor((end_time - start_time) * fps)) + 1
    candidate = np.flatnonzero(lower.time >= start_time)
    indices = np.linspace(candidate[0], candidate[-1], frame_count, dtype=int)
    lower_frames = _render_frames(lower, indices, payload=True)
    higher_frames = _render_frames(higher, indices, payload=True)
    frames = []
    lower_position_error = 1000.0 * np.linalg.norm(
        lower.impedance_pose[:, :3] - lower.actual_pose[:, :3], axis=1
    )
    higher_position_error = 1000.0 * np.linalg.norm(
        higher.impedance_pose[:, :3] - higher.actual_pose[:, :3], axis=1
    )
    lower_orientation_error = 1000.0 * np.linalg.norm(
        lower.impedance_pose[:, 3:] - lower.actual_pose[:, 3:], axis=1
    )
    higher_orientation_error = 1000.0 * np.linalg.norm(
        higher.impedance_pose[:, 3:] - higher.actual_pose[:, 3:], axis=1
    )
    first_index = int(candidate[0])
    for output_index, history_index in enumerate(indices):
        canvas = Image.new("RGB", (1280, 390), color=(20, 22, 26))
        canvas.paste(lower_frames[output_index], (0, 30))
        canvas.paste(higher_frames[output_index], (480, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (174, 7), "Lower stiffness", fill="white", font=_font(15, bold=True)
        )
        draw.text(
            (642, 7), "Higher stiffness", fill="white", font=_font(15, bold=True)
        )
        phase = lower.phase[history_index]
        _draw_wrench_badge(draw, phase, 0)
        _draw_wrench_badge(draw, phase, 480)
        draw_metric_panel(
            canvas,
            title="Fixed-tuning impedance tracking",
            time=lower.time,
            current_index=int(history_index),
            first_index=first_index,
            upper_title="Actual-to-impedance error [mm]",
            upper_traces=(
                Trace("Lower K", lower_position_error, LOWER_COLOR),
                Trace("Higher K", higher_position_error, HIGHER_COLOR),
            ),
            upper_limit=nice_upper_limit(
                lower_position_error[first_index:],
                higher_position_error[first_index:],
                floor=0.1,
            ),
            lower_title="Rotation-vector error [mrad]",
            lower_traces=(
                Trace("Lower K", lower_orientation_error, LOWER_COLOR),
                Trace("Higher K", higher_orientation_error, HIGHER_COLOR),
            ),
            lower_limit=nice_upper_limit(
                lower_orientation_error[first_index:],
                higher_orientation_error[first_index:],
                floor=0.1,
            ),
            events=((7.0, WRENCH_COLOR), (10.0, WRENCH_COLOR)),
            summary="Same NAC tuning | adaptation enabled",
        )
        draw.text(
            (10, 368),
            f"t={lower.time[history_index]:05.2f}s | {phase} | same measured wrench",
            fill="white",
            font=_font(13),
        )
        frames.append(canvas)
    _save_animation(frames, webp_path, gif_path, fps)


def _write_drag_animation(
    webp_path: Path,
    gif_path: Path,
    adaptive: ShowcaseResult,
    nominal: ShowcaseResult,
) -> None:
    from PIL import Image, ImageDraw

    fps = 8.0
    frame_count = int(np.floor(adaptive.time[-1] * fps)) + 1
    indices = np.linspace(0, len(adaptive.time) - 1, frame_count, dtype=int)
    adaptive_frames = _render_frames(adaptive, indices, payload=False)
    nominal_frames = _render_frames(nominal, indices, payload=False)
    frames = []
    event_time = float(adaptive.metrics["event_time_sec"])
    adaptive_position = 1000.0 * np.linalg.norm(
        adaptive.desired_pose[:, :3] - adaptive.actual_pose[:, :3], axis=1
    )
    nominal_position = 1000.0 * np.linalg.norm(
        nominal.desired_pose[:, :3] - nominal.actual_pose[:, :3], axis=1
    )
    adaptive_rotation = 1000.0 * np.linalg.norm(
        adaptive.desired_pose[:, 3:] - adaptive.actual_pose[:, 3:], axis=1
    )
    nominal_rotation = 1000.0 * np.linalg.norm(
        nominal.desired_pose[:, 3:] - nominal.actual_pose[:, 3:], axis=1
    )
    for output_index, history_index in enumerate(indices):
        canvas = Image.new("RGB", (1280, 390), color=(20, 22, 26))
        canvas.paste(adaptive_frames[output_index], (0, 30))
        canvas.paste(nominal_frames[output_index], (480, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (180, 7), "Adaptive NAC", fill="white", font=_font(15, bold=True)
        )
        draw.text(
            (617, 7),
            "Fixed nominal model-based",
            fill="white",
            font=_font(15, bold=True),
        )
        draw_metric_panel(
            canvas,
            title="Tracking after hidden joint drag",
            time=adaptive.time,
            current_index=int(history_index),
            first_index=0,
            upper_title="Position tracking error [mm]",
            upper_traces=(
                Trace("NAC", adaptive_position, NAC_COLOR),
                Trace("Nominal", nominal_position, NOMINAL_COLOR),
            ),
            upper_limit=nice_upper_limit(
                adaptive_position, nominal_position, floor=1.0
            ),
            lower_title="Rotation-vector error [mrad]",
            lower_traces=(
                Trace("NAC", adaptive_rotation, NAC_COLOR),
                Trace("Nominal", nominal_rotation, NOMINAL_COLOR),
            ),
            lower_limit=nice_upper_limit(
                adaptive_rotation, nominal_rotation, floor=1.0
            ),
            events=((event_time, EVENT_COLOR),),
            summary="Purple: hidden plant change | fixed model",
        )
        if adaptive.time[history_index] >= event_time:
            for offset in (0, 480):
                _draw_status_badge(draw, "HIDDEN JOINT DRAG", offset)
        draw.text(
            (10, 368),
            (
                f"t={adaptive.time[history_index]:05.2f}s | "
                f"{adaptive.phase[history_index]} | plant change hidden from both controllers"
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
    lower = run_compliance_benchmark(ComplianceVariant.SOFT)
    higher = run_compliance_benchmark(ComplianceVariant.STIFF)
    adaptive = run_joint_drag_benchmark(DragVariant.ADAPTIVE)
    nominal = run_joint_drag_benchmark(DragVariant.NOMINAL)
    frozen = run_joint_drag_benchmark(DragVariant.FROZEN)
    report = _report(lower, higher, adaptive, nominal, frozen)
    report_path = output / "showcase_scenarios_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not arguments.skip_animations:
        _write_compliance_animation(
            output / "compliance_comparison.webp",
            output / "compliance_comparison.gif",
            lower,
            higher,
        )
        _write_drag_animation(
            output / "joint_drag_comparison.webp",
            output / "joint_drag_comparison.gif",
            adaptive,
            nominal,
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
