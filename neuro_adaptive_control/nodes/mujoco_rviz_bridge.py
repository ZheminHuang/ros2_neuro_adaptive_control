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

"""Broadcast display-only TCP transforms from MuJoCo pose telemetry."""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class MujocoRvizBridge(Node):
    """Convert three pose topics into RViz-friendly TF display frames."""

    def __init__(self) -> None:
        super().__init__("mujoco_rviz_bridge")
        self._broadcaster = TransformBroadcaster(self)
        for topic, child in (
            ("mujoco/actual_pose", "mujoco_tcp"),
            ("mujoco/desired_pose", "desired_tcp"),
            ("mujoco/impedance_pose", "impedance_tcp"),
        ):
            self.create_subscription(
                PoseStamped,
                topic,
                lambda message, frame=child: self._broadcast(message, frame),
                10,
            )

    def _broadcast(self, message: PoseStamped, child_frame: str) -> None:
        transform = TransformStamped()
        transform.header = message.header
        transform.child_frame_id = child_frame
        transform.transform.translation.x = message.pose.position.x
        transform.transform.translation.y = message.pose.position.y
        transform.transform.translation.z = message.pose.position.z
        transform.transform.rotation = message.pose.orientation
        self._broadcaster.sendTransform(transform)


def main(args=None) -> None:
    """Run the display-only MuJoCo-to-RViz transform bridge."""
    rclpy.init(args=args)
    node = MujocoRvizBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
