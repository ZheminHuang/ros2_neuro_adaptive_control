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

"""Goal-admission regressions for the Robotiq action bridge."""

from types import SimpleNamespace

import pytest


rclpy = pytest.importorskip("rclpy")
action_module = pytest.importorskip("rclpy.action")
control_actions = pytest.importorskip("control_msgs.action")
server_module = pytest.importorskip(
    "neuro_adaptive_control.nodes.robotiq_gripper_action_server"
)

GoalResponse = action_module.GoalResponse
CancelResponse = action_module.CancelResponse
GripperCommand = control_actions.GripperCommand
RobotiqGripperActionServer = server_module.RobotiqGripperActionServer
Trigger = server_module.Trigger


@pytest.fixture()
def action_server():
    """Create an isolated server without spinning its executor."""
    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init(args=[])
    node = RobotiqGripperActionServer()
    try:
        yield node
    finally:
        node.destroy_node()
        if owns_context and rclpy.ok():
            rclpy.shutdown()


def _goal(position: float, effort: float) -> GripperCommand.Goal:
    goal = GripperCommand.Goal()
    goal.command.position = position
    goal.command.max_effort = effort
    return goal


@pytest.mark.parametrize(
    ("position", "effort"),
    ((-0.001, 1.0), (0.086, 1.0), (0.050, -0.1)),
)
def test_invalid_metric_goals_are_rejected(
    action_server: RobotiqGripperActionServer,
    position: float,
    effort: float,
) -> None:
    """Reject position bounds and negative effort before actuator command."""
    assert action_server._goal(_goal(position, effort)) == GoalResponse.REJECT


def test_only_one_gripper_goal_can_be_active(
    action_server: RobotiqGripperActionServer,
) -> None:
    """Keep synchronous execute callbacks from starving physical feedback."""
    assert action_server._goal(_goal(0.070, 3.0)) == GoalResponse.ACCEPT
    assert action_server._goal(_goal(0.075, 2.0)) == GoalResponse.REJECT


class _FakeGoalHandle:
    """Minimal action handle for terminal-behavior unit tests."""

    def __init__(self, goal, *, cancel_requested: bool = False) -> None:
        self.request = goal
        self.is_cancel_requested = cancel_requested
        self.canceled_called = False
        self.aborted_called = False
        self.succeeded_called = False
        self.feedback_count = 0

    def publish_feedback(self, feedback) -> None:
        del feedback
        self.feedback_count += 1

    def canceled(self) -> None:
        self.canceled_called = True

    def abort(self) -> None:
        self.aborted_called = True

    def succeed(self) -> None:
        self.succeeded_called = True


def test_cancel_sends_stop_and_reports_canceled_result(
    action_server: RobotiqGripperActionServer,
    monkeypatch,
) -> None:
    """Do not leave a tendon command active after action cancellation."""
    commands = []
    monkeypatch.setattr(
        action_server,
        "_publish_command",
        lambda *args, **kwargs: commands.append((args, kwargs)),
    )
    handle = _FakeGoalHandle(_goal(0.020, 2.0), cancel_requested=True)

    result = action_server._execute_accepted_goal(handle)

    assert handle.canceled_called
    assert not handle.aborted_called
    assert result.reached_goal is False
    assert result.stalled is False
    assert commands[-1] == ((0.0, 0.0), {"stop": True})


def test_timeout_sends_stop_and_reports_stalled_abort(
    action_server: RobotiqGripperActionServer,
    monkeypatch,
) -> None:
    """Abort an unacknowledged command without retaining actuator effort."""
    commands = []
    monkeypatch.setattr(
        action_server,
        "_publish_command",
        lambda *args, **kwargs: commands.append((args, kwargs)),
    )
    monkeypatch.setattr(
        action_server,
        "get_parameter",
        lambda name: SimpleNamespace(value=0.0),
    )
    handle = _FakeGoalHandle(_goal(0.020, 2.0))

    result = action_server._execute_accepted_goal(handle)

    assert handle.aborted_called
    assert not handle.succeeded_called
    assert result.reached_goal is False
    assert result.stalled is True
    assert commands[-1] == ((0.0, 0.0), {"stop": True})


def test_reset_is_rejected_while_action_goal_is_active(
    action_server: RobotiqGripperActionServer,
    monkeypatch,
) -> None:
    """Prevent reset from silently overriding an accepted action command."""
    published = []
    monkeypatch.setattr(
        action_server,
        "_publish_command",
        lambda *args, **kwargs: published.append((args, kwargs)),
    )
    assert action_server._goal(_goal(0.020, 2.0)) == GoalResponse.ACCEPT

    response = action_server._on_reset(Trigger.Request(), Trigger.Response())

    assert response.success is False
    assert "active" in response.message
    assert published == []


def test_cancel_callback_accepts_and_stops_immediately(
    action_server: RobotiqGripperActionServer,
    monkeypatch,
) -> None:
    """Expose action cancel as the explicit gripper stop interface."""
    published = []
    monkeypatch.setattr(
        action_server,
        "_publish_command",
        lambda *args, **kwargs: published.append((args, kwargs)),
    )

    assert action_server._cancel(object()) == CancelResponse.ACCEPT
    assert published == [((0.0, 0.0), {"stop": True})]
