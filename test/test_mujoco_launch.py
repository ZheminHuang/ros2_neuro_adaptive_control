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

"""Headless ROS launch acceptance tests for MuJoCo trajectory and grasp."""

from collections import deque
from functools import partial
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import signal
import subprocess
from time import monotonic

import pytest


rclpy = pytest.importorskip("rclpy")
mujoco = pytest.importorskip("mujoco")
action_messages = pytest.importorskip("action_msgs.msg")
ament_packages = pytest.importorskip("ament_index_python.packages")
control_actions = pytest.importorskip("control_msgs.action")
diagnostic_messages = pytest.importorskip("diagnostic_msgs.msg")
nav_messages = pytest.importorskip("nav_msgs.msg")
rclpy_action = pytest.importorskip("rclpy.action")
rclpy_executors = pytest.importorskip("rclpy.executors")
sensor_messages = pytest.importorskip("sensor_msgs.msg")
standard_messages = pytest.importorskip("std_msgs.msg")
standard_services = pytest.importorskip("std_srvs.srv")
tf2_messages = pytest.importorskip("tf2_msgs.msg")
visualization_messages = pytest.importorskip("visualization_msgs.msg")

GoalStatus = action_messages.GoalStatus
GripperCommand = control_actions.GripperCommand
DiagnosticArray = diagnostic_messages.DiagnosticArray
PathMessage = nav_messages.Path
ActionClient = rclpy_action.ActionClient
JointState = sensor_messages.JointState
Float64 = standard_messages.Float64
Float64MultiArray = standard_messages.Float64MultiArray
Trigger = standard_services.Trigger
TFMessage = tf2_messages.TFMessage
MarkerArray = visualization_messages.MarkerArray

try:
    ament_packages.get_package_prefix("neuro_adaptive_control")
    _PACKAGE_DISCOVERABLE = True
except ament_packages.PackageNotFoundError:
    _PACKAGE_DISCOVERABLE = False

pytestmark = pytest.mark.skipif(
    not hasattr(mujoco, "MjModel")
    or shutil.which("ros2") is None
    or not _PACKAGE_DISCOVERABLE,
    reason=(
        "official MuJoCo bindings, ROS 2 CLI, and a sourced package are required"
    ),
)

_TOPICS = {
    "joint_states": (JointState, "/joint_states"),
    "tf": (TFMessage, "/tf"),
    "desired_path": (PathMessage, "/mujoco/desired_path"),
    "impedance_path": (PathMessage, "/mujoco/impedance_path"),
    "actual_path": (PathMessage, "/mujoco/actual_path"),
    "diagnostics": (DiagnosticArray, "/diagnostics"),
    "scene_markers": (MarkerArray, "/mujoco/scene_markers"),
    "contact_markers": (MarkerArray, "/mujoco/contact_markers"),
}


class _TopicObserver:
    """Keep bounded evidence that each visualization interface was received."""

    def __init__(self, context) -> None:
        self.node = rclpy.create_node(
            f"mujoco_launch_acceptance_{os.getpid()}", context=context
        )
        self.executor = rclpy_executors.SingleThreadedExecutor(context=context)
        self.executor.add_node(self.node)
        self.counts = {key: 0 for key in _TOPICS}
        self.last_messages = {}
        self.subscriptions = []
        self.gripper_states = deque(maxlen=500)
        self.gripper_actuator_commands = deque(maxlen=500)
        self.gripper_state_count = 0
        self.gripper_actuator_command_count = 0
        for key, (message_type, topic) in _TOPICS.items():
            self.subscriptions.append(
                self.node.create_subscription(
                    message_type,
                    topic,
                    partial(self._record, key),
                    20,
                )
            )
        self.subscriptions.append(
            self.node.create_subscription(
                Float64MultiArray,
                "/mujoco/gripper/state",
                self._record_gripper_state,
                20,
            )
        )
        self.subscriptions.append(
            self.node.create_subscription(
                Float64,
                "/mujoco/gripper/actuator_command",
                self._record_gripper_actuator_command,
                20,
            )
        )

    def _record(self, key: str, message) -> None:
        self.counts[key] += 1
        self.last_messages[key] = message

    def _record_gripper_state(self, message: Float64MultiArray) -> None:
        if len(message.data) >= 9:
            self.gripper_state_count += 1
            self.gripper_states.append(
                tuple(float(value) for value in message.data[:9])
            )

    def _record_gripper_actuator_command(self, message: Float64) -> None:
        self.gripper_actuator_command_count += 1
        self.gripper_actuator_commands.append(float(message.data))

    def all_topics_seen(self) -> bool:
        return all(count > 0 for count in self.counts.values())

    def spin_once(self, timeout_sec: float) -> None:
        self.executor.spin_once(timeout_sec=timeout_sec)

    def close(self) -> None:
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        self.executor.shutdown()


def _launch_environment(tmp_path: Path) -> dict[str, str]:
    """Expose the active official MuJoCo wheel to ROS console scripts."""
    environment = os.environ.copy()
    binding_site = str(Path(mujoco.__file__).resolve().parents[1])
    python_path = environment.get("PYTHONPATH", "")
    path_entries = [binding_site]
    if python_path:
        path_entries.append(python_path)
    environment["PYTHONPATH"] = os.pathsep.join(path_entries)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["ROS_LOG_DIR"] = str(tmp_path / "ros_logs")
    environment["RCUTILS_COLORIZED_OUTPUT"] = "0"
    return environment


def _start_launch(
    launch_file: str,
    arguments: list[str],
    tmp_path: Path,
) -> subprocess.Popen:
    command = [
        "ros2",
        "launch",
        "neuro_adaptive_control",
        launch_file,
        *arguments,
    ]
    return subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=_launch_environment(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _stop_launch(process: subprocess.Popen) -> str:
    """Stop only the launch process group created by this test."""
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        output, _ = process.communicate(timeout=8.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate(timeout=3.0)
    return output


def _spin_until_exit(
    observer: _TopicObserver,
    process: subprocess.Popen,
    *,
    timeout_sec: float,
) -> str:
    deadline = monotonic() + timeout_sec
    while process.poll() is None and monotonic() < deadline:
        observer.spin_once(timeout_sec=0.05)
    if process.poll() is None:
        output = _stop_launch(process)
        pytest.fail("launch did not exit before timeout:\n" + output[-8000:])
    for _ in range(10):
        observer.spin_once(timeout_sec=0.02)
    output, _ = process.communicate(timeout=3.0)
    return output


def _spin_until(
    observer: _TopicObserver,
    process: subprocess.Popen,
    predicate,
    *,
    timeout_sec: float,
) -> bool:
    """Spin one acceptance node until evidence arrives or launch exits."""
    deadline = monotonic() + timeout_sec
    while monotonic() < deadline:
        if predicate():
            return True
        if process.poll() is not None:
            output, _ = process.communicate(timeout=3.0)
            pytest.fail(
                "launch exited while awaiting ROS evidence:\n" + output[-8000:]
            )
        observer.spin_once(timeout_sec=0.05)
    return bool(predicate())


@pytest.fixture()
def ros_observer(monkeypatch, tmp_path: Path):
    """Create an isolated ROS context before any launch publishers start."""
    domain_id = 40 + os.getpid() % 180
    monkeypatch.setenv("ROS_DOMAIN_ID", str(domain_id))
    monkeypatch.delenv("ROS_LOCALHOST_ONLY", raising=False)
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "observer_ros_logs"))
    context = rclpy.context.Context()
    rclpy.init(context=context)
    observer = _TopicObserver(context)
    try:
        yield observer
    finally:
        observer.close()
        if rclpy.ok(context=context):
            context.shutdown()


def test_headless_trajectory_launch_exits_with_exact_steps_and_topics(
    tmp_path: Path, ros_observer: _TopicObserver
) -> None:
    """Run 1 s headless and verify exact physics plus RViz telemetry."""
    metrics_path = tmp_path / "trajectory_metrics.json"
    process = _start_launch(
        "ur5e_mujoco_rviz.launch.py",
        [
            "start_rviz:=false",
            "start_mujoco_viewer:=false",
            "duration_sec:=1.0",
            "scenario:=trajectory",
            "trajectory:=circle",
            f"metrics_path:={metrics_path}",
        ],
        tmp_path,
    )
    output = _spin_until_exit(ros_observer, process, timeout_sec=25.0)

    assert process.returncode == 0, output[-8000:]
    assert metrics_path.is_file(), output[-8000:]
    metrics = json.loads(metrics_path.read_text())
    assert metrics["state"] == "stopped"
    assert metrics["control_steps"] == 500
    assert metrics["mujoco_steps"] == 2000
    assert metrics["simulated_duration_sec"] == pytest.approx(
        1.0, abs=2.0e-12
    )
    assert metrics["hard_real_time_guarantee"] is False
    for key in (
        "impedance_tracking_rmse_m",
        "impedance_tracking_max_error_m",
        "command_force_max_norm_n",
        "arm_torque_max_abs_nm",
        "joint_velocity_max_abs_rad_s",
        "contact_force_max_n",
        "torque_saturation_ratio",
        "callback_overrun_count",
        "timer_interarrival_median_ms",
        "timer_interarrival_p95_ms",
        "timer_interarrival_p99_ms",
        "timer_interarrival_max_ms",
    ):
        assert isfinite(metrics[key]) and metrics[key] >= 0.0
    assert metrics["missed_wall_deadlines"] == metrics["callback_overrun_count"]
    assert (
        metrics["timer_interarrival_median_ms"]
        <= metrics["timer_interarrival_p95_ms"]
        <= metrics["timer_interarrival_p99_ms"]
        <= metrics["timer_interarrival_max_ms"]
    )

    assert ros_observer.all_topics_seen(), (
        f"missing topics: {ros_observer.counts}\n{output[-8000:]}"
    )
    joint_state = ros_observer.last_messages["joint_states"]
    assert len(joint_state.name) == 14
    assert len(joint_state.position) == 14
    assert len(joint_state.velocity) == 14
    assert joint_state.header.frame_id == "world"
    assert ros_observer.last_messages["tf"].transforms

    for key in ("desired_path", "impedance_path", "actual_path"):
        path = ros_observer.last_messages[key]
        assert path.header.frame_id == "world"
        assert path.poses
    diagnostics = ros_observer.last_messages["diagnostics"]
    assert diagnostics.status
    assert diagnostics.status[0].name == "mujoco_ur5e_nac"
    scene = ros_observer.last_messages["scene_markers"]
    assert len(scene.markers) >= 7
    assert ros_observer.counts["contact_markers"] > 0


def test_headless_grasp_launch_starts_and_publishes_state(
    tmp_path: Path, ros_observer: _TopicObserver
) -> None:
    """Start the grasp graph, observe its nodes/topics, then stop cleanly."""
    metrics_path = tmp_path / "grasp_metrics.json"
    process = _start_launch(
        "ur5e_mujoco_grasp_demo.launch.py",
        [
            "start_rviz:=false",
            "start_mujoco_viewer:=false",
            f"metrics_path:={metrics_path}",
        ],
        tmp_path,
    )
    deadline = monotonic() + 15.0
    expected_nodes = {
        "mujoco_ur5e_plant",
        "mujoco_rviz_bridge",
        "robotiq_gripper_action_server",
        "robot_state_publisher",
    }
    observed_nodes: set[str] = set()
    while process.poll() is None and monotonic() < deadline:
        ros_observer.spin_once(timeout_sec=0.05)
        observed_nodes = set(ros_observer.node.get_node_names())
        if (
            ros_observer.counts["joint_states"] > 0
            and ros_observer.counts["diagnostics"] > 0
            and expected_nodes <= observed_nodes
        ):
            break

    started = (
        process.poll() is None
        and ros_observer.counts["joint_states"] > 0
        and ros_observer.counts["diagnostics"] > 0
        and expected_nodes <= observed_nodes
    )
    if not started:
        output = _stop_launch(process)
        pytest.fail(
            f"grasp graph failed to become observable; nodes={observed_nodes}, "
            f"topics={ros_observer.counts}\n{output[-8000:]}"
        )
    # GitHub's container can run contact dynamics below 0.4 real-time factor;
    # keep all 5,500-step assertions while allowing the finite run to finish.
    output = _spin_until_exit(ros_observer, process, timeout_sec=90.0)
    assert process.returncode == 0, output[-8000:]
    assert "mujoco_ur5e_plant startup failed" not in output
    assert metrics_path.is_file(), output[-8000:]
    metrics = json.loads(metrics_path.read_text())
    assert metrics["state"] == "stopped"
    assert metrics["grasp_success"] is True
    assert metrics["control_steps"] == 5500
    assert metrics["mujoco_steps"] == 22000
    assert metrics["object_lift_height_m"] >= 0.05
    assert metrics["hold_duration_sec"] >= 2.0
    assert metrics["hold_drop_m"] <= 0.005
    assert metrics["hold_bilateral_contact_ratio"] >= 0.90
    assert metrics["contact_force_max_n"] <= (
        metrics["configured_contact_force_limit_n"]
    )
    assert metrics["maximum_gripper_effort_n"] <= (
        metrics["configured_gripper_effort_limit_n"] + 1e-9
    )


def test_headless_gripper_action_drives_plant_and_reset(
    tmp_path: Path, ros_observer: _TopicObserver
) -> None:
    """Exercise GripperCommand, physical plant feedback, and reset service."""
    process = _start_launch(
        "ur5e_mujoco_rviz.launch.py",
        [
            "start_rviz:=false",
            "start_mujoco_viewer:=false",
            "duration_sec:=12.0",
            "scenario:=trajectory",
            "trajectory:=fixed_point",
            f"metrics_path:={tmp_path / 'gripper_action_metrics.json'}",
        ],
        tmp_path,
    )
    action_client = None
    reset_client = None
    output = ""
    try:
        action_client = ActionClient(
            ros_observer.node,
            GripperCommand,
            "/robotiq_gripper/gripper_command",
        )
        reset_client = ros_observer.node.create_client(
            Trigger, "/robotiq_gripper/reset"
        )
        assert _spin_until(
            ros_observer,
            process,
            lambda: (
                action_client.server_is_ready()
                and reset_client.service_is_ready()
                and bool(ros_observer.gripper_states)
                and bool(ros_observer.gripper_actuator_commands)
            ),
            timeout_sec=12.0,
        ), "gripper ROS interfaces did not become ready"
        initial_state = ros_observer.gripper_states[-1]
        initial_opening = initial_state[0]
        initial_actuator = ros_observer.gripper_actuator_commands[-1]
        assert all(isfinite(value) for value in initial_state)
        assert isfinite(initial_actuator)

        feedback = []
        goal = GripperCommand.Goal()
        goal.command.position = 0.070
        goal.command.max_effort = 3.0
        send_future = action_client.send_goal_async(
            goal,
            feedback_callback=lambda message: feedback.append(message.feedback),
        )
        assert _spin_until(
            ros_observer,
            process,
            send_future.done,
            timeout_sec=5.0,
        ), "GripperCommand goal response timed out"
        goal_handle = send_future.result()
        assert goal_handle is not None and goal_handle.accepted

        result_future = goal_handle.get_result_async()
        assert _spin_until(
            ros_observer,
            process,
            result_future.done,
            timeout_sec=6.0,
        ), "GripperCommand result timed out"
        for _ in range(5):
            ros_observer.spin_once(timeout_sec=0.02)
        wrapped_result = result_future.result()
        assert wrapped_result.status == GoalStatus.STATUS_SUCCEEDED
        assert isfinite(wrapped_result.result.position)
        assert isfinite(wrapped_result.result.effort)
        assert wrapped_result.result.reached_goal is True
        assert wrapped_result.result.stalled is False
        assert abs(wrapped_result.result.position - 0.070) <= 0.001
        assert feedback
        assert all(
            isfinite(sample.position) and isfinite(sample.effort)
            for sample in feedback
        )
        assert feedback[-1].reached_goal is True
        assert feedback[-1].stalled is False
        assert abs(feedback[-1].position - 0.070) <= 0.001

        assert _spin_until(
            ros_observer,
            process,
            lambda: (
                any(
                    abs(state[2] - 0.070) < 1.0e-9
                    for state in ros_observer.gripper_states
                )
                and any(
                    command > initial_actuator + 5.0
                    for command in ros_observer.gripper_actuator_commands
                )
                and any(
                    abs(state[0] - initial_opening) > 2.0e-4
                    for state in ros_observer.gripper_states
                )
            ),
            timeout_sec=4.0,
        ), "accepted action did not move the MuJoCo gripper"
        assert all(
            all(isfinite(value) for value in state)
            for state in ros_observer.gripper_states
        )
        assert all(
            isfinite(command)
            for command in ros_observer.gripper_actuator_commands
        )

        state_count_before_reset = ros_observer.gripper_state_count
        actuator_count_before_reset = (
            ros_observer.gripper_actuator_command_count
        )
        reset_future = reset_client.call_async(Trigger.Request())
        assert _spin_until(
            ros_observer,
            process,
            reset_future.done,
            timeout_sec=4.0,
        ), "gripper reset service response timed out"
        reset_response = reset_future.result()
        assert reset_response is not None and reset_response.success
        assert reset_response.message == "gripper reset requested"
        assert _spin_until(
            ros_observer,
            process,
            lambda: (
                ros_observer.gripper_state_count > state_count_before_reset
                and ros_observer.gripper_actuator_command_count
                > actuator_count_before_reset
                and abs(
                    ros_observer.gripper_states[-1][2] - 0.085
                )
                < 1.0e-9
                and abs(ros_observer.gripper_actuator_commands[-1]) < 1.0e-9
            ),
            timeout_sec=4.0,
        ), "reset service did not restore the plant gripper target"
    finally:
        if action_client is not None:
            action_client.destroy()
        if reset_client is not None:
            ros_observer.node.destroy_client(reset_client)
        output = _stop_launch(process)
    assert "mujoco_ur5e_plant startup failed" not in output
