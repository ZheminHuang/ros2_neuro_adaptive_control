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
    assert "0.239 mm" in readme
    assert "1.143 mm" in readme
    assert "79.1% lower" in readme
    assert "0.236 mrad" in readme
    assert "1.057 mrad" in readme
    assert "77.7% lower" in readme


def test_showcase_images_are_real_bounded_animated_artifacts():
    assert GIF.stat().st_size <= 10 * 1024 * 1024
    with Image.open(GIF) as animation:
        assert animation.format == "GIF"
        assert animation.size == (1280, 390)
        assert animation.n_frames >= 120
    with Image.open(PLOT) as plot:
        assert plot.format == "PNG"
        assert plot.width >= 1600
        assert plot.height >= 1200


def test_payload_artifacts_contain_no_absolute_workspace_path():
    report = REPORT.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    assert "/home/" not in report
    assert "/home/" not in readme
