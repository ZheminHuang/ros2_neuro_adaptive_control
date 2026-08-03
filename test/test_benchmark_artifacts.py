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

"""Keep committed MuJoCo benchmark evidence tied to its exact implementation."""

from hashlib import sha256
import json
from pathlib import Path
import re

from neuro_adaptive_control import __version__


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = (
    PROJECT_ROOT / "docs" / "assets" / "mujoco_tracking_benchmark.json",
    PROJECT_ROOT / "docs" / "assets" / "mujoco_grasp_benchmark.json",
)
ROS_TIMING_REPORTS = (
    PROJECT_ROOT / "docs" / "assets" / "mujoco_ros_trajectory_timing.json",
    PROJECT_ROOT / "docs" / "assets" / "mujoco_ros_grasp_timing.json",
)
RESULTS_DOCUMENT = PROJECT_ROOT / "docs" / "mujoco_demo_results.md"
ARTIFACTS = (
    PROJECT_ROOT / "docs" / "assets" / "mujoco_tracking_benchmark.json",
    PROJECT_ROOT / "docs" / "assets" / "mujoco_tracking_benchmark.png",
    PROJECT_ROOT / "docs" / "assets" / "mujoco_grasp_benchmark.json",
    PROJECT_ROOT / "docs" / "assets" / "mujoco_grasp_benchmark.png",
    *ROS_TIMING_REPORTS,
    PROJECT_ROOT / "docs" / "images" / "ur5e_robotiq_mujoco_rviz.png",
    PROJECT_ROOT
    / "docs"
    / "images"
    / "ur5e_robotiq_mujoco_viewer.png",
)


def _digest(relative_path: str) -> str:
    return sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()


def test_benchmark_reports_match_all_hashed_sources_and_model() -> None:
    """Fail whenever evidence was not regenerated after implementation edits."""
    for path in REPORTS:
        report = json.loads(path.read_text(encoding="utf-8"))
        trace = report["traceability"]
        assert trace["package_version"] == __version__
        assert trace["mujoco_version"] == "3.9.0"
        assert trace["numpy_version"] == "1.24.4"
        assert trace["generator_sha256"] == _digest(trace["generator"])
        assert trace["runner_sha256"] == _digest(trace["runner"])
        assert trace["model_sha256"] == _digest(trace["model"])
        assert trace["model_manifest_sha256"] == _digest(
            trace["model_manifest"]
        )
        for relative, expected in trace["source_files_sha256"].items():
            assert expected == _digest(relative), relative


def test_benchmark_reports_record_passing_acceptance_without_nonfinite_json() -> None:
    """Require every committed acceptance flag to reflect a passing run."""
    for path in REPORTS:
        content = path.read_text(encoding="utf-8")
        assert "NaN" not in content
        assert "Infinity" not in content
        acceptance = json.loads(content)["acceptance"]
        boolean_results = [
            value for value in acceptance.values() if isinstance(value, bool)
        ]
        assert boolean_results
        assert all(boolean_results)


def test_documented_artifact_hashes_match_committed_files() -> None:
    """Keep the human-readable evidence page tied to JSON and PNG bytes."""
    document = RESULTS_DOCUMENT.read_text(encoding="utf-8")
    for artifact in ARTIFACTS:
        relative = artifact.relative_to(RESULTS_DOCUMENT.parent).as_posix()
        match = re.search(
            rf"\]\({re.escape(relative)}\), SHA-256\s+`([0-9a-f]{{64}})`",
            document,
        )
        assert match is not None, relative
        assert match.group(1) == sha256(artifact.read_bytes()).hexdigest()


def test_ros_timing_reports_are_machine_readable_non_realtime_evidence() -> None:
    """Tie documented ROS wall-rate numbers to sanitized launch metrics."""
    document = RESULTS_DOCUMENT.read_text(encoding="utf-8")
    expected_scenarios = ("trajectory", "grasp")
    for path, scenario in zip(ROS_TIMING_REPORTS, expected_scenarios):
        content = path.read_text(encoding="utf-8")
        assert "NaN" not in content
        assert "Infinity" not in content
        report = json.loads(content)
        assert report["artifact_kind"] == "ros_launch_wall_clock_measurement"
        assert report["environment"]["mujoco_version"] == "3.9.0"
        assert report["environment"]["numpy_version"] == "1.24.4"
        metrics = report["metrics"]
        assert metrics["scenario"] == scenario
        assert metrics["state"] == "stopped"
        assert metrics["fault_reason"] == ""
        assert metrics["hard_real_time_guarantee"] is False
        assert metrics["callback_overrun_count"] == metrics[
            "missed_wall_deadlines"
        ]
        rate = metrics["observed_control_step_rate_hz"]
        assert f"{rate:.3f} Hz" in document
