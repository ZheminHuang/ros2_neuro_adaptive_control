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

"""Validate deterministic grasp/contact dynamics with optional MuJoCo."""

from dataclasses import replace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mujoco_bindings():
    module = pytest.importorskip(
        "mujoco",
        reason="the optional official MuJoCo Python bindings are unavailable",
    )
    if not hasattr(module, "MjModel"):
        pytest.skip(
            "the local mujoco asset directory is not the Python MuJoCo package"
        )
    return module


def _manual_contact_wrench(mujoco_bindings, plant) -> dict[str, object]:
    tcp = plant.data.site_xpos[plant._tcp_site_id].copy()
    robot_force = np.zeros(3)
    robot_torque = np.zeros(3)
    environment_force = np.zeros(3)
    environment_torque = np.zeros(3)
    positions = []
    forces = []
    count = 0
    norm_sum = 0.0
    for index in range(plant.data.ncon):
        contact = plant.data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(plant.model.geom_bodyid[geom1])
        body2 = int(plant.model.geom_bodyid[geom2])
        robot1 = body1 in plant._robot_bodies
        robot2 = body2 in plant._robot_bodies
        if robot1 == robot2:
            continue
        local_wrench = np.zeros(6)
        mujoco_bindings.mj_contactForce(
            plant.model,
            plant.data,
            index,
            local_wrench,
        )
        contact_to_world = np.asarray(contact.frame).reshape(3, 3).T
        force_on_geom2 = contact_to_world @ local_wrench[:3]
        torque_on_geom2 = contact_to_world @ local_wrench[3:]
        if robot2:
            force_on_robot = force_on_geom2
            torque_on_robot = torque_on_geom2
        else:
            force_on_robot = -force_on_geom2
            torque_on_robot = -torque_on_geom2
        position = np.asarray(contact.pos).copy()
        torque_at_tcp = torque_on_robot + np.cross(
            position - tcp,
            force_on_robot,
        )
        robot_force += force_on_robot
        robot_torque += torque_at_tcp
        environment_force -= force_on_robot
        environment_torque -= torque_at_tcp
        positions.append(position)
        forces.append(force_on_robot.copy())
        norm_sum += float(np.linalg.norm(force_on_robot))
        count += 1
    return {
        "tcp": tcp,
        "robot_force": robot_force,
        "robot_torque": robot_torque,
        "environment_force": environment_force,
        "environment_torque": environment_torque,
        "positions": tuple(positions),
        "forces": tuple(forces),
        "count": count,
        "norm_sum": norm_sum,
    }


@pytest.fixture(scope="module")
def grasp_evidence(mujoco_bindings):
    from neuro_adaptive_control.adapters.mujoco_grasp import (
        MujocoGraspRunner,
        run_grasp_demo,
    )

    runner = MujocoGraspRunner()
    captured: dict[str, object] = {}
    original_contact_summary = runner.plant.contact_summary

    def contact_summary_with_evidence():
        summary = original_contact_summary()
        if (
            "summary" not in captured
            and summary.left_finger_contacts > 0
            and summary.right_finger_contacts > 0
        ):
            captured["summary"] = summary
            captured["manual"] = _manual_contact_wrench(
                mujoco_bindings,
                runner.plant,
            )
        return summary

    runner.plant.contact_summary = contact_summary_with_evidence
    first = runner.run()
    second = run_grasp_demo()
    assert "summary" in captured
    assert "manual" in captured
    return first, second, captured


def _phase_array(result) -> np.ndarray:
    return np.asarray(result.phase)


def test_default_grasp_meets_contact_lift_and_hold_contract(grasp_evidence) -> None:
    result = grasp_evidence[0]
    metrics = result.metrics
    phases = _phase_array(result)
    dt = 0.002
    hold = phases == "hold"
    lift_or_hold = (phases == "lift") | hold
    settle = (result.time >= 2.8) & (result.time < 3.0)
    initial_object_z = float(np.median(result.object_position[settle, 2]))
    maximum_object_z = float(np.max(result.object_position[lift_or_hold, 2]))
    lift_height = maximum_object_z - initial_object_z
    hold_drop = maximum_object_z - float(
        np.min(result.object_position[hold, 2])
    )
    contact_duration = float(np.count_nonzero(result.bilateral_contact) * dt)
    contact_ratio = float(np.mean(result.bilateral_contact[hold]))

    assert metrics["success"] is True
    assert metrics["state"] == "stopped"
    assert contact_duration >= 0.1
    assert lift_height >= 0.05
    assert float(np.count_nonzero(hold) * dt) >= 2.0
    assert hold_drop <= 0.005
    assert contact_ratio >= 0.90
    assert metrics["bilateral_contact_duration_sec"] == pytest.approx(
        contact_duration
    )
    assert metrics["object_lift_height_m"] == pytest.approx(lift_height)
    assert metrics["hold_drop_m"] == pytest.approx(hold_drop)
    assert metrics["hold_bilateral_contact_ratio"] == pytest.approx(
        contact_ratio
    )


def test_default_grasp_respects_contact_actuator_and_motion_bounds(
    grasp_evidence,
) -> None:
    result = grasp_evidence[0]
    metrics = result.metrics
    torque_limits = np.array((140.0, 140.0, 140.0, 27.0, 27.0, 27.0))
    assert metrics["maximum_penetration_m"] <= 0.002
    assert metrics["maximum_gripper_effort_n"] <= 2.0 + 1e-9
    assert metrics["maximum_contact_force_n"] <= 180.0
    assert metrics["maximum_joint_velocity_rad_s"] <= 3.5
    assert np.all(np.abs(result.arm_torque) <= torque_limits + 1e-12)
    assert metrics["maximum_arm_torque_abs_nm"] == pytest.approx(
        float(np.max(np.abs(result.arm_torque)))
    )
    assert metrics["torque_saturation_count"] == 0
    assert np.all(result.contact_force >= 0.0)
    assert float(np.max(result.contact_force)) == pytest.approx(
        metrics["maximum_contact_force_n"]
    )


def test_default_grasp_has_no_fault_warning_nan_or_unexpected_collision(
    grasp_evidence,
) -> None:
    result = grasp_evidence[0]
    metrics = result.metrics
    for values in (
        result.time,
        result.tcp_position,
        result.object_position,
        result.gripper_opening,
        result.arm_torque,
        result.contact_force,
    ):
        assert np.all(np.isfinite(values))
    assert metrics["solver_warning_count"] == 0
    assert metrics["unexpected_contact_count"] == 0
    assert metrics["fault_reason"] == ""
    assert metrics["simulated_duration_sec"] == pytest.approx(11.0, abs=3e-12)
    assert np.all(np.diff(result.time) > 0.0)


def test_release_returns_object_opens_gripper_and_retreats(grasp_evidence) -> None:
    result = grasp_evidence[0]
    phases = _phase_array(result)
    release = phases == "release"
    settle = (result.time >= 2.8) & (result.time < 3.0)
    initial_object_z = float(np.median(result.object_position[settle, 2]))
    returned_object_z = float(np.median(result.object_position[release, 2]))
    assert np.any(release)
    assert np.any(phases == "retreat")
    assert abs(returned_object_z - initial_object_z) <= 0.005
    assert result.metrics["returned_height_error_m"] <= 0.005
    assert result.gripper_opening[-1] >= 0.080
    assert not result.bilateral_contact[-1]
    assert np.linalg.norm(result.tcp_position[-1] - result.tcp_position[0]) <= 0.005


def test_default_grasp_is_exactly_deterministic_except_wall_timing(
    grasp_evidence,
) -> None:
    first, second, _ = grasp_evidence
    assert first.phase == second.phase
    for name in (
        "time",
        "tcp_position",
        "object_position",
        "gripper_opening",
        "arm_torque",
        "contact_force",
        "bilateral_contact",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    nondeterministic_metrics = {"wall_duration_sec", "real_time_factor"}
    for name, value in first.metrics.items():
        if name not in nondeterministic_metrics:
            assert second.metrics[name] == value


def test_contact_only_wrench_uses_world_frame_tcp_shift_and_robot_sign(
    grasp_evidence,
) -> None:
    _, _, captured = grasp_evidence
    summary = captured["summary"]
    manual = captured["manual"]
    assert summary.left_finger_contacts > 0
    assert summary.right_finger_contacts > 0
    assert summary.expected_object_contacts > 0
    assert summary.unexpected_contacts == 0
    assert summary.total_robot_environment_contacts == manual["count"]
    np.testing.assert_allclose(
        summary.force_world,
        manual["robot_force"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        summary.torque_world_at_tcp,
        manual["robot_torque"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        manual["robot_force"] + manual["environment_force"],
        np.zeros(3),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        manual["robot_torque"] + manual["environment_torque"],
        np.zeros(3),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(summary.forces_world), axis=0),
        summary.force_world,
        rtol=1e-12,
        atol=1e-12,
    )
    assert summary.contact_force_norm_n == pytest.approx(manual["norm_sum"])
    assert np.linalg.norm(summary.force_world) > 0.0


def test_virtual_ft_uses_contact_force_without_reapplying_it(
    mujoco_bindings,
) -> None:
    from neuro_adaptive_control.adapters.mujoco_simulation import (
        MujocoNACRunner,
        MujocoRunConfig,
    )

    runner = MujocoNACRunner(
        MujocoRunConfig(
            trajectory="fixed_point",
            duration_sec=0.002,
            adaptation_enabled=False,
            external_wrench_mode="virtual_ft",
        )
    )
    real_summary = runner.plant.contact_summary()
    contact_force = np.array((3.0, -2.0, 1.0))
    virtual_summary = replace(
        real_summary,
        force_world=contact_force,
        contact_force_norm_n=float(np.linalg.norm(contact_force)),
    )
    runner.plant.contact_summary = lambda: virtual_summary
    controller_external = []
    injected_force = []
    original_step = runner.controller.step
    original_advance = runner.plant.advance

    def recorded_step(*args, **kwargs):
        controller_external.append(np.asarray(args[3]).copy())
        return original_step(*args, **kwargs)

    def recorded_advance(*args, **kwargs):
        injected_force.append(
            np.asarray(kwargs.get("injected_force_world", np.zeros(3))).copy()
        )
        return original_advance(*args, **kwargs)

    runner.controller.step = recorded_step
    runner.plant.advance = recorded_advance
    result = runner.run()
    assert result.metrics["external_wrench_mode"] == "virtual_ft"
    assert len(controller_external) == 1
    assert len(injected_force) == 1
    np.testing.assert_array_equal(controller_external[0], contact_force)
    np.testing.assert_array_equal(injected_force[0], np.zeros(3))
    np.testing.assert_array_equal(runner.plant.data.qfrc_applied, np.zeros_like(
        runner.plant.data.qfrc_applied
    ))
