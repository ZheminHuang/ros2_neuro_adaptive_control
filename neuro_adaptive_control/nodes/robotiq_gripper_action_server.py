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

"""Expose the dynamic MuJoCo gripper through GripperCommand."""

from __future__ import annotations

from threading import Lock
from time import monotonic, sleep

import numpy as np
import rclpy
from control_msgs.action import GripperCommand
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger


class RobotiqGripperActionServer(Node):
    """Translate standard metric action goals into plant command samples."""

    def __init__(self) -> None:
        super().__init__("robotiq_gripper_action_server")
        self.declare_parameter("action_name", "robotiq_gripper/gripper_command")
        self.declare_parameter("goal_timeout_sec", 8.0)
        self.declare_parameter("maximum_opening_m", 0.085)
        self.declare_parameter("maximum_effort_n", 5.0)
        self._lock = Lock()
        self._state = np.array((0.085, 0.0, 0.085, 0, 0, 0, 1, 0, 0), dtype=float)
        self._state_sequence = 0
        self._goal_active = False
        self._command_pub = self.create_publisher(
            Float64MultiArray, "mujoco/gripper/command", 10
        )
        callback_group = ReentrantCallbackGroup()
        self._state_sub = self.create_subscription(
            Float64MultiArray,
            "mujoco/gripper/state",
            self._on_state,
            10,
            callback_group=callback_group,
        )
        self._reset_service = self.create_service(
            Trigger,
            "robotiq_gripper/reset",
            self._on_reset,
            callback_group=callback_group,
        )
        self._action = ActionServer(
            self,
            GripperCommand,
            str(self.get_parameter("action_name").value),
            execute_callback=self._execute,
            goal_callback=self._goal,
            cancel_callback=self._cancel,
            callback_group=callback_group,
        )

    def _on_state(self, message: Float64MultiArray) -> None:
        if len(message.data) >= 9 and np.all(np.isfinite(message.data[:9])):
            with self._lock:
                self._state = np.asarray(message.data[:9], dtype=float)
                self._state_sequence += 1

    def _goal(self, goal_request) -> GoalResponse:
        position = float(goal_request.command.position)
        effort = float(goal_request.command.max_effort)
        maximum_opening = float(self.get_parameter("maximum_opening_m").value)
        if (
            not np.isfinite(position)
            or not np.isfinite(effort)
            or effort < 0.0
            or position < 0.0
            or position > maximum_opening
        ):
            self.get_logger().warning("rejecting invalid gripper action goal")
            return GoalResponse.REJECT
        with self._lock:
            if self._goal_active:
                self.get_logger().warning(
                    "rejecting concurrent gripper goal; one goal is active"
                )
                return GoalResponse.REJECT
            self._goal_active = True
        return GoalResponse.ACCEPT

    def _cancel(self, goal_handle) -> CancelResponse:
        del goal_handle
        self._publish_command(0.0, 0.0, stop=True)
        return CancelResponse.ACCEPT

    def _publish_command(
        self,
        position: float,
        effort: float,
        *,
        stop: bool = False,
        reset: bool = False,
    ) -> None:
        self._command_pub.publish(
            Float64MultiArray(
                data=[position, effort, float(stop), float(reset)]
            )
        )

    def _execute(self, goal_handle):
        try:
            return self._execute_accepted_goal(goal_handle)
        finally:
            with self._lock:
                self._goal_active = False

    def _execute_accepted_goal(self, goal_handle):
        """Run the one accepted goal while state callbacks use another thread."""
        command = goal_handle.request.command
        position = float(command.position)
        effort = float(command.max_effort)
        if effort <= 0.0:
            effort = float(self.get_parameter("maximum_effort_n").value)
        with self._lock:
            state_sequence_before_command = self._state_sequence
        self._publish_command(position, effort)
        start = monotonic()
        timeout = float(self.get_parameter("goal_timeout_sec").value)
        result = GripperCommand.Result()
        terminal_reached_goal = False
        terminal_stalled = False
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self._publish_command(0.0, 0.0, stop=True)
                goal_handle.canceled()
                break
            with self._lock:
                state = self._state.copy()
                state_sequence = self._state_sequence
            command_acknowledged = bool(
                state_sequence > state_sequence_before_command
                and abs(float(state[2]) - position) <= 1.0e-9
            )
            feedback = GripperCommand.Feedback()
            feedback.position = float(state[0])
            feedback.effort = float(state[1])
            feedback.reached_goal = bool(command_acknowledged and state[6])
            feedback.stalled = bool(command_acknowledged and state[7])
            goal_handle.publish_feedback(feedback)
            if feedback.reached_goal or feedback.stalled:
                terminal_reached_goal = feedback.reached_goal
                terminal_stalled = feedback.stalled
                goal_handle.succeed()
                break
            if monotonic() - start > timeout:
                self._publish_command(0.0, 0.0, stop=True)
                goal_handle.abort()
                terminal_stalled = True
                break
            sleep(0.05)
        with self._lock:
            final = self._state.copy()
        result.position = float(final[0])
        result.effort = float(final[1])
        result.reached_goal = terminal_reached_goal
        result.stalled = terminal_stalled
        return result

    def _on_reset(self, request, response):
        del request
        with self._lock:
            if self._goal_active:
                response.success = False
                response.message = "cannot reset while a gripper goal is active"
                return response
            self._publish_command(0.085, 5.0, reset=True)
            response.success = True
            response.message = "gripper reset requested"
        return response

    def destroy_node(self) -> bool:
        """Destroy the action server before its ROS node."""
        self._action.destroy()
        return super().destroy_node()


def main(args=None) -> None:
    """Run the standard Robotiq GripperCommand action bridge."""
    rclpy.init(args=args)
    node = RobotiqGripperActionServer()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
