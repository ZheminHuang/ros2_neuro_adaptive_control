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

"""Traceability and presentation checks for the three-scene showcase."""

from hashlib import sha256
import json
from pathlib import Path
import re

from PIL import Image

from neuro_adaptive_control import __version__


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "assets" / "showcase_scenarios_metrics.json"
README = ROOT / "README.md"


def _digest(relative: str) -> str:
    return sha256((ROOT / relative).read_bytes()).hexdigest()


def test_showcase_report_matches_current_sources_model_and_version():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    trace = report["traceability"]
    assert trace["package_version"] == __version__
    assert trace["mujoco_version"] == "3.9.0"
    assert trace["numpy_version"] == "1.24.4"
    assert trace["generator_sha256"] == _digest(trace["generator"])
    assert trace["model_sha256"] == _digest(trace["model"])
    for relative, expected in trace["source_files_sha256"].items():
        assert expected == _digest(relative), relative
    for history in report["histories"]:
        assert re.fullmatch(
            r"[0-9a-f]{64}", history["deterministic_history_sha256"]
        )


def test_showcase_metrics_support_only_the_concise_readme_claims():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    readme = README.read_text(encoding="utf-8")
    compliance = report["compliance"]["comparison"]
    drag = report["joint_drag"]["comparison"]
    assert compliance["both_trials_completed"]
    assert compliance["controller_tuning_identical_between_trials"]
    assert compliance["online_adaptation_enabled_in_both_trials"]
    assert compliance["fixed_tuning_tracking_gate_passed"]
    assert compliance["maximum_actual_to_impedance_position_rmse_m"] <= 0.001
    assert compliance["maximum_actual_to_impedance_orientation_rmse_rad"] <= 0.001
    assert drag["public_nominal_comparison_gate_passed"]
    assert drag["nn_adaptation_ablation_gate_passed"]
    assert drag["position_rmse_improvement_vs_nominal_ratio"] >= 0.10
    assert drag["orientation_rmse_improvement_vs_nominal_ratio"] >= 0.10
    for text in ("0.283", "0.120", "4.121", "41.529"):
        assert text in readme
    assert "1.89" not in readme
    assert "1.90" not in readme


def test_two_new_scenes_are_full_color_bounded_animations():
    readme = README.read_text(encoding="utf-8")
    for stem, minimum_frames in (
        ("compliance_comparison", 60),
        ("joint_drag_comparison", 80),
    ):
        webp = ROOT / "docs" / "assets" / f"{stem}.webp"
        gif = ROOT / "docs" / "assets" / f"{stem}.gif"
        assert webp.stat().st_size <= 30 * 1024 * 1024
        assert gif.stat().st_size <= 10 * 1024 * 1024
        with Image.open(webp) as animation:
            assert animation.format == "WEBP"
            assert animation.size == (1280, 390)
            assert animation.n_frames >= minimum_frames
            animation.seek(animation.n_frames // 2)
            colors = animation.convert("RGB").getcolors(maxcolors=1_000_000)
            assert colors is not None and len(colors) > 256
        assert f"docs/assets/{stem}.webp" in readme
        assert f"docs/assets/{stem}.gif" in readme


def test_showcase_artifacts_contain_no_absolute_workspace_path():
    assert "/home/" not in REPORT.read_text(encoding="utf-8")
    assert "/home/" not in README.read_text(encoding="utf-8")
