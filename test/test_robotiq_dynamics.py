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

"""Verify the Robotiq metric adapter and optional MuJoCo dynamics model."""

from pathlib import Path

import numpy as np
import pytest

from neuro_adaptive_control.adapters.robotiq_gripper_adapter import (
    GripperLimits,
    RobotiqGripperAdapter,
)


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "mujoco"
    / "ur5e_robotiq_2f85.xml"
)
GRIPPER_JOINT_NAMES = (
    "gripper_right_driver_joint",
    "gripper_right_coupler_joint",
    "gripper_right_spring_link_joint",
    "gripper_right_follower_joint",
    "gripper_left_driver_joint",
    "gripper_left_coupler_joint",
    "gripper_left_spring_link_joint",
    "gripper_left_follower_joint",
)


def _driver_position(opening_m: float) -> float:
    limits = GripperLimits()
    return limits.driver_range_rad * (
        1.0 - opening_m / limits.maximum_opening_m
    )


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


@pytest.fixture
def dynamic_plant(mujoco_bindings):
    from neuro_adaptive_control.adapters.mujoco_ur5e_adapter import (
        MujocoUR5ePlant,
    )

    plant = MujocoUR5ePlant(MODEL_PATH)
    plant.model.opt.gravity[:] = 0.0
    plant.model.opt.disableflags |= int(
        mujoco_bindings.mjtDisableBit.mjDSBL_CONTACT
    )
    plant.reset()
    return plant


def test_default_open_and_close_commands_use_menagerie_endpoints() -> None:
    adapter = RobotiqGripperAdapter()
    assert adapter.target_opening_m == pytest.approx(0.085)
    assert adapter.actuator_control() == pytest.approx(0.0)
    adapter.close()
    assert adapter.target_opening_m == pytest.approx(0.0)
    assert adapter.actuator_control() == pytest.approx(255.0)
    adapter.open()
    assert adapter.target_opening_m == pytest.approx(0.085)
    assert adapter.actuator_control() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "opening_m,expected_control",
    ((0.0, 255.0), (0.02125, 191.25), (0.0425, 127.5), (0.085, 0.0)),
)
def test_metric_opening_maps_linearly_to_actuator_control(
    opening_m: float,
    expected_control: float,
) -> None:
    adapter = RobotiqGripperAdapter()
    adapter.command(opening_m, 2.0)
    assert adapter.actuator_control() == pytest.approx(expected_control)


@pytest.mark.parametrize(
    "driver_position,expected_opening",
    ((0.0, 0.085), (0.2, 0.06375), (0.4, 0.0425), (0.8, 0.0)),
)
def test_driver_coordinates_map_back_to_metric_opening(
    driver_position: float,
    expected_opening: float,
) -> None:
    adapter = RobotiqGripperAdapter()
    assert adapter.opening_from_driver_positions(
        driver_position, driver_position
    ) == pytest.approx(expected_opening)


def test_driver_mapping_and_metric_command_clip_to_physical_range() -> None:
    adapter = RobotiqGripperAdapter()
    adapter.command(-1.0, 100.0)
    assert adapter.target_opening_m == pytest.approx(0.0)
    assert adapter.maximum_effort_n == pytest.approx(5.0)
    assert adapter.opening_from_driver_positions(-1.0, -1.0) == pytest.approx(
        0.085
    )
    adapter.command(1.0, 2.5)
    assert adapter.target_opening_m == pytest.approx(0.085)
    assert adapter.maximum_effort_n == pytest.approx(2.5)
    assert adapter.opening_from_driver_positions(2.0, 2.0) == pytest.approx(0.0)


@pytest.mark.parametrize("effort", (0.0, -1.0, -100.0))
def test_nonpositive_effort_requests_select_the_safe_global_limit(
    effort: float,
) -> None:
    adapter = RobotiqGripperAdapter()
    adapter.command(0.04, effort)
    assert adapter.maximum_effort_n == pytest.approx(5.0)


def test_open_and_close_accept_a_bounded_per_goal_effort() -> None:
    adapter = RobotiqGripperAdapter()
    adapter.close(1.25)
    assert adapter.maximum_effort_n == pytest.approx(1.25)
    adapter.open(50.0)
    assert adapter.maximum_effort_n == pytest.approx(5.0)


def test_stop_holds_measured_opening_and_new_command_resumes() -> None:
    adapter = RobotiqGripperAdapter()
    adapter.close(2.0)
    adapter.stop(0.034)
    assert adapter.stopped
    assert adapter.target_opening_m == pytest.approx(0.034)
    expected = 255.0 * (1.0 - 0.034 / 0.085)
    assert adapter.actuator_control() == pytest.approx(expected)
    adapter.open()
    assert not adapter.stopped
    assert adapter.actuator_control() == pytest.approx(0.0)


def test_stop_clips_measurement_and_reset_is_deterministic() -> None:
    adapter = RobotiqGripperAdapter()
    adapter.command(0.01, 1.0)
    adapter.stop(1.0)
    assert adapter.target_opening_m == pytest.approx(0.085)
    adapter.reset()
    assert adapter.target_opening_m == pytest.approx(0.085)
    assert adapter.maximum_effort_n == pytest.approx(5.0)
    assert not adapter.stopped
    assert adapter.actuator_control() == pytest.approx(0.0)


def test_feedback_reports_metric_state_and_normalizes_counts_and_effort() -> None:
    adapter = RobotiqGripperAdapter()
    adapter.command(0.04, 2.0)
    driver = _driver_position(0.04)
    state = adapter.feedback(
        right_driver_rad=driver,
        left_driver_rad=driver,
        actuator_force_n=-1.5,
        left_contacts=-2,
        right_contacts=3,
        contact_force_n=2.25,
    )
    assert state.opening_m == pytest.approx(0.04)
    assert state.effort_n == pytest.approx(1.5)
    assert state.target_opening_m == pytest.approx(0.04)
    assert state.left_contacts == 0
    assert state.right_contacts == 3
    assert state.contact_force_n == pytest.approx(2.25)
    assert state.reached_goal
    assert not state.stalled
    assert not state.stopped


def test_feedback_propagates_stopped_and_reached_state() -> None:
    adapter = RobotiqGripperAdapter()
    adapter.stop(0.05)
    driver = _driver_position(0.05)
    state = adapter.feedback(
        right_driver_rad=driver,
        left_driver_rad=driver,
        actuator_force_n=0.0,
        left_contacts=0,
        right_contacts=0,
        contact_force_n=0.0,
    )
    assert state.reached_goal
    assert state.stopped


def test_bilateral_contact_and_effort_define_stall() -> None:
    adapter = RobotiqGripperAdapter()
    adapter.close(2.0)
    driver = _driver_position(0.02)

    def feedback(*, left_contacts=1, right_contacts=1, effort=1.6):
        return adapter.feedback(
            right_driver_rad=driver,
            left_driver_rad=driver,
            actuator_force_n=effort,
            left_contacts=left_contacts,
            right_contacts=right_contacts,
            contact_force_n=3.0,
        )

    stalled = feedback()
    assert not stalled.reached_goal
    assert stalled.stalled
    assert not feedback(right_contacts=0).stalled
    assert not feedback(effort=1.59).stalled


@pytest.mark.parametrize(
    "field,value",
    (
        ("maximum_opening_m", 0.0),
        ("maximum_effort_n", -1.0),
        ("driver_range_rad", np.nan),
        ("actuator_maximum", np.inf),
    ),
)
def test_limits_reject_nonpositive_or_nonfinite_values(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        GripperLimits(**{field: value})


@pytest.mark.parametrize(
    "opening,effort",
    (
        (np.nan, 1.0),
        (np.inf, 1.0),
        (0.04, np.nan),
        (0.04, np.inf),
        ("bad", 1.0),
    ),
)
def test_command_rejects_nonfinite_or_nonnumeric_input(opening, effort) -> None:
    with pytest.raises((TypeError, ValueError)):
        RobotiqGripperAdapter().command(opening, effort)


@pytest.mark.parametrize("measurement", (np.nan, np.inf, -np.inf, "bad"))
def test_stop_rejects_invalid_measurement(measurement) -> None:
    with pytest.raises((TypeError, ValueError)):
        RobotiqGripperAdapter().stop(measurement)


@pytest.mark.parametrize(
    "right_driver,left_driver",
    ((np.nan, 0.0), (0.0, np.inf), (-np.inf, 0.0)),
)
def test_driver_mapping_rejects_nonfinite_input(
    right_driver: float,
    left_driver: float,
) -> None:
    with pytest.raises(ValueError):
        RobotiqGripperAdapter().opening_from_driver_positions(
            right_driver, left_driver
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"actuator_force_n": np.nan},
        {"actuator_force_n": np.inf},
        {"contact_force_n": np.nan},
        {"contact_force_n": -0.1},
        {"position_tolerance_m": np.nan},
        {"position_tolerance_m": -0.1},
        {"stall_tolerance_m": np.inf},
        {"stall_tolerance_m": -0.1},
    ),
)
def test_feedback_rejects_invalid_force_or_tolerance(overrides) -> None:
    arguments = {
        "right_driver_rad": 0.0,
        "left_driver_rad": 0.0,
        "actuator_force_n": 0.0,
        "left_contacts": 0,
        "right_contacts": 0,
        "contact_force_n": 0.0,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError):
        RobotiqGripperAdapter().feedback(**arguments)


def test_mujoco_model_exposes_all_eight_gripper_joint_states(
    mujoco_bindings,
    dynamic_plant,
) -> None:
    joint_ids = [
        mujoco_bindings.mj_name2id(
            dynamic_plant.model,
            mujoco_bindings.mjtObj.mjOBJ_JOINT,
            name,
        )
        for name in GRIPPER_JOINT_NAMES
    ]
    assert all(identifier >= 0 for identifier in joint_ids)
    qpos_addresses = dynamic_plant.model.jnt_qposadr[joint_ids]
    dof_addresses = dynamic_plant.model.jnt_dofadr[joint_ids]
    assert len(set(map(int, qpos_addresses))) == 8
    assert len(set(map(int, dof_addresses))) == 8
    state = dynamic_plant.kinematic_state()
    assert dynamic_plant.joint_names[-8:] == GRIPPER_JOINT_NAMES
    assert state.all_joint_position.shape == (14,)
    assert state.all_joint_velocity.shape == (14,)
    assert np.all(np.isfinite(state.all_joint_position[-8:]))
    assert np.all(np.isfinite(state.all_joint_velocity[-8:]))


def test_mujoco_driver_equality_has_unit_coupling(
    mujoco_bindings,
    dynamic_plant,
) -> None:
    model = dynamic_plant.model
    right = mujoco_bindings.mj_name2id(
        model,
        mujoco_bindings.mjtObj.mjOBJ_JOINT,
        "gripper_right_driver_joint",
    )
    left = mujoco_bindings.mj_name2id(
        model,
        mujoco_bindings.mjtObj.mjOBJ_JOINT,
        "gripper_left_driver_joint",
    )
    joint_type = int(mujoco_bindings.mjtEq.mjEQ_JOINT)
    matches = [
        index
        for index in range(model.neq)
        if int(model.eq_type[index]) == joint_type
        and {int(model.eq_obj1id[index]), int(model.eq_obj2id[index])}
        == {right, left}
    ]
    assert len(matches) == 1
    np.testing.assert_allclose(
        model.eq_data[matches[0], :5],
        (0.0, 1.0, 0.0, 0.0, 0.0),
        atol=1e-12,
    )


def test_mujoco_actuator_declares_position_and_global_force_limits(
    mujoco_bindings,
    dynamic_plant,
) -> None:
    model = dynamic_plant.model
    actuator = mujoco_bindings.mj_name2id(
        model,
        mujoco_bindings.mjtObj.mjOBJ_ACTUATOR,
        "gripper_fingers_actuator",
    )
    assert actuator >= 0
    assert bool(model.actuator_ctrllimited[actuator])
    assert bool(model.actuator_forcelimited[actuator])
    np.testing.assert_allclose(model.actuator_ctrlrange[actuator], (0.0, 255.0))
    np.testing.assert_allclose(model.actuator_forcerange[actuator], (-5.0, 5.0))


def test_plant_applies_per_goal_effort_limit_to_mujoco_actuator(
    mujoco_bindings,
    dynamic_plant,
) -> None:
    actuator = mujoco_bindings.mj_name2id(
        dynamic_plant.model,
        mujoco_bindings.mjtObj.mjOBJ_ACTUATOR,
        "gripper_fingers_actuator",
    )
    dynamic_plant.gripper.close(1.25)
    for _ in range(20):
        dynamic_plant.advance(np.zeros(6))
        assert abs(dynamic_plant.data.actuator_force[actuator]) <= 1.25 + 1e-12
    np.testing.assert_allclose(
        dynamic_plant.model.actuator_forcerange[actuator], (-1.25, 1.25)
    )
    dynamic_plant.reset()
    np.testing.assert_allclose(
        dynamic_plant.model.actuator_forcerange[actuator], (-5.0, 5.0)
    )


def test_mujoco_dynamics_map_255_to_close_and_0_to_open(
    mujoco_bindings,
    dynamic_plant,
) -> None:
    model = dynamic_plant.model
    actuator = mujoco_bindings.mj_name2id(
        model,
        mujoco_bindings.mjtObj.mjOBJ_ACTUATOR,
        "gripper_fingers_actuator",
    )
    right_joint = mujoco_bindings.mj_name2id(
        model,
        mujoco_bindings.mjtObj.mjOBJ_JOINT,
        "gripper_right_driver_joint",
    )
    left_joint = mujoco_bindings.mj_name2id(
        model,
        mujoco_bindings.mjtObj.mjOBJ_JOINT,
        "gripper_left_driver_joint",
    )
    right_qpos = int(model.jnt_qposadr[right_joint])
    left_qpos = int(model.jnt_qposadr[left_joint])

    dynamic_plant.gripper.close()
    maximum_force = 0.0
    for _ in range(500):
        dynamic_plant.advance(np.zeros(6))
        maximum_force = max(
            maximum_force,
            abs(float(dynamic_plant.data.actuator_force[actuator])),
        )
    closed = dynamic_plant.gripper_state()
    assert dynamic_plant.data.ctrl[actuator] == pytest.approx(255.0)
    assert closed.opening_m <= 0.003
    assert closed.reached_goal
    assert maximum_force <= 5.0 + 1e-12
    assert maximum_force > 0.0
    assert abs(
        dynamic_plant.data.qpos[right_qpos]
        - dynamic_plant.data.qpos[left_qpos]
    ) < 1e-5

    dynamic_plant.gripper.open()
    for _ in range(500):
        dynamic_plant.advance(np.zeros(6))
    opened = dynamic_plant.gripper_state()
    assert dynamic_plant.data.ctrl[actuator] == pytest.approx(0.0)
    assert opened.opening_m >= 0.082
    assert opened.reached_goal
    assert abs(
        dynamic_plant.data.qpos[right_qpos]
        - dynamic_plant.data.qpos[left_qpos]
    ) < 1e-5
    assert np.all(np.isfinite(dynamic_plant.kinematic_state().all_joint_position))
