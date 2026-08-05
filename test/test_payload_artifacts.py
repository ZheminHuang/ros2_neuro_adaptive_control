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

"""Traceability and presentation checks for the v0.3 payload artifacts."""

from hashlib import sha256
import json
from pathlib import Path
import re

from PIL import Image

from neuro_adaptive_control import __version__


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "assets" / "payload_benchmark_metrics.json"
PLOT = ROOT / "docs" / "assets" / "payload_benchmark_results.png"
GIF = ROOT / "docs" / "assets" / "payload_benchmark_comparison.gif"
WEBP = ROOT / "docs" / "assets" / "payload_benchmark_comparison.webp"
README = ROOT / "README.md"


def _digest(relative: str) -> str:
    return sha256((ROOT / relative).read_bytes()).hexdigest()


def test_payload_report_matches_current_sources_model_and_version():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    trace = report["traceability"]

    assert trace["package_version"] == __version__
    assert trace["mujoco_version"] == "3.9.0"
    assert trace["numpy_version"] == "1.24.4"
    assert trace["generator_sha256"] == _digest(trace["generator"])
    assert trace["runner_sha256"] == _digest(trace["runner"])
    assert trace["model_sha256"] == _digest(trace["model"])
    for relative, expected in trace["source_files_sha256"].items():
        assert expected == _digest(relative), relative
    for trial in report["trials"]:
        assert re.fullmatch(
            r"[0-9a-f]{64}",
            trial["deterministic_history_sha256"],
        )


def test_payload_gate_and_readme_values_match_committed_report():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    metrics = report["aggregate_adaptation_gate"]
    readme = README.read_text(encoding="utf-8")

    assert metrics["adaptation_advantage_gate_passed"]
    assert metrics["adaptive_completion_ratio"] == 1.0
    assert metrics["frozen_completion_ratio"] == 1.0
    showcase = report["showcase_dynamics_change"]
    scenario = report["scenario"]
    assert scenario["public_showcase_payload_mass_kg"] == 1.0
    assert not scenario["payload_parameters_visible_to_nominal_controller"]
    assert not scenario["nominal_controller_model_updated_after_pickup"]
    assert all(scenario["comparison_contract"].values())
    assert showcase["adaptive_loaded_position_rmse_m"] < (
        showcase["nominal_loaded_position_rmse_m"]
    )
    assert showcase["adaptive_loaded_orientation_rmse_rad"] < (
        showcase["nominal_loaded_orientation_rmse_rad"]
    )
    for text in ("0.230", "26.381", "0.236", "133.329"):
        assert text in readme


def test_showcase_images_are_real_bounded_animated_artifacts():
    assert WEBP.stat().st_size <= 30 * 1024 * 1024
    with Image.open(WEBP) as animation:
        assert animation.format == "WEBP"
        assert animation.mode in {"RGB", "RGBA"}
        assert animation.size == (1280, 390)
        assert animation.n_frames >= 120
        animation.seek(animation.n_frames // 2)
        assert len(animation.convert("RGB").getcolors(maxcolors=1_000_000)) > 256
    assert GIF.stat().st_size <= 10 * 1024 * 1024
    with Image.open(GIF) as animation:
        assert animation.format == "GIF"
        assert animation.size == (1280, 390)
        assert animation.n_frames >= 120
    with Image.open(PLOT) as plot:
        assert plot.format == "PNG"
        assert plot.width >= 1600
        assert plot.height >= 1200

    readme = README.read_text(encoding="utf-8")
    assert "docs/assets/payload_benchmark_comparison.webp" in readme


def test_payload_event_marker_is_dashed_and_confined_to_each_plot():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    adaptive = next(
        trial
        for trial in report["trials"]
        if trial["controller"] == "adaptive_nac"
        and trial["payload"]["mass_kg"] == 1.0
    )
    event_time = adaptive["metrics"]["payload_acquisition_time_sec"]
    first_time = 0.002
    event_x = 988 + int(274 * (event_time - first_time) / (15.0 - first_time))
    purple = (198, 105, 224)

    def close_to_purple(pixel):
        return max(
            abs(channel - expected)
            for channel, expected in zip(pixel, purple)
        ) <= 20

    with Image.open(WEBP) as animation:
        animation.seek(animation.n_frames // 2)
        frame = animation.convert("RGB")
        assert close_to_purple(frame.getpixel((event_x, 69)))
        assert not close_to_purple(frame.getpixel((event_x, 170)))
        assert close_to_purple(frame.getpixel((event_x, 209)))


def test_payload_artifacts_contain_no_absolute_workspace_path():
    report = REPORT.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    assert "/home/" not in report
    assert "/home/" not in readme
