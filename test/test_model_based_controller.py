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

"""Tests for fair nominal and oracle model-based benchmark controllers."""

import numpy as np

from neuro_adaptive_control.adapters.model_based_controller import (
    MujocoModelBasedController,
)
from neuro_adaptive_control.adapters.mujoco_ur5e_adapter import MujocoUR5ePlant
from neuro_adaptive_control.core.pose_references import PoseReferenceSample
from neuro_adaptive_control.core.so3 import log


def _arguments(plant, *, payload_acquired):
    sample = plant.kinematic_state()
    rho = log(sample.tcp_rotation @ plant.desired_tcp_rotation.T)
    jacobian = np.vstack(
        (sample.translational_jacobian, sample.rotational_jacobian)
    )
    return dict(
        all_joint_position=sample.all_joint_position,
        all_joint_velocity=sample.all_joint_velocity,
        actual_pose=np.concatenate((sample.tcp_position, rho)),
        actual_pose_velocity=np.concatenate(
            (sample.tcp_linear_velocity, sample.tcp_angular_velocity)
        ),
        reference=PoseReferenceSample(
            np.concatenate((sample.tcp_position, rho)),
            np.zeros(6),
            np.zeros(6),
        ),
        rotation_vector=rho,
        geometric_jacobian=jacobian,
        tcp_position=sample.tcp_position,
        payload_position=sample.object_position + np.array([0.01, 0.0, 0.0]),
        payload_acquired=payload_acquired,
        dt=0.002,
    )


def test_nominal_controller_does_not_use_payload_truth():
    plant = MujocoUR5ePlant()
    controller = MujocoModelBasedController()
    arguments = _arguments(plant, payload_acquired=False)

    unloaded = controller.command(**arguments)
    controller.reset()
    arguments["payload_acquired"] = True
    loaded = controller.command(**arguments)

    np.testing.assert_array_equal(
        unloaded.nominal_inverse_dynamics,
        loaded.nominal_inverse_dynamics,
    )
    np.testing.assert_array_equal(loaded.oracle_payload_compensation, np.zeros(6))


def test_nominal_controller_model_does_not_follow_plant_joint_drag():
    plant = MujocoUR5ePlant()
    controller = MujocoModelBasedController()
    model_damping = controller.model.dof_damping.copy()
    model_friction = controller.model.dof_frictionloss.copy()

    plant.apply_joint_drag(
        ("shoulder_lift_joint", "elbow_joint", "wrist_2_joint"),
        damping_scale=8.0,
        frictionloss_scale=6.0,
    )

    np.testing.assert_array_equal(controller.model.dof_damping, model_damping)
    np.testing.assert_array_equal(
        controller.model.dof_frictionloss, model_friction
    )


def test_oracle_adds_known_payload_gravity_only_after_acquisition():
    plant = MujocoUR5ePlant()
    controller = MujocoModelBasedController(
        oracle_payload_mass_kg=0.31,
        torque_limits=(1000.0,) * 6,
        torque_rate_limits=(1.0e9,) * 6,
    )
    arguments = _arguments(plant, payload_acquired=False)

    before = controller.command(**arguments)
    controller.reset()
    arguments["payload_acquired"] = True
    after = controller.command(**arguments)

    np.testing.assert_array_equal(before.oracle_payload_compensation, np.zeros(6))
    assert np.linalg.norm(after.oracle_payload_compensation) > 0.0
    np.testing.assert_allclose(
        after.command - before.command,
        after.oracle_payload_compensation,
        atol=1.0e-12,
    )
