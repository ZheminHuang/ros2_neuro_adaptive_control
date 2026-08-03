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

"""Dynamic contract tests for the UR5e/2F-85 MuJoCo plant model."""

from pathlib import Path

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.skipif(
    not hasattr(mujoco, "MjModel"),
    reason="official MuJoCo Python bindings are not installed",
)

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "mujoco"
    / "ur5e_robotiq_2f85.xml"
)
ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@pytest.fixture()
def model():
    """Compile an isolated model for each test that may mutate parameters."""
    assert MODEL_PATH.is_file()
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def _id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, f"missing {object_type}: {name}"
    return object_id


def _arm_dof_addresses(model):
    addresses = []
    for joint_name in ARM_JOINT_NAMES:
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        addresses.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(addresses, dtype=int)


def _gripper_subtree_body_ids(model):
    root_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base_mount")
    subtree_ids = []
    for body_id in range(model.nbody):
        ancestor_id = body_id
        while ancestor_id != 0 and ancestor_id != root_id:
            ancestor_id = int(model.body_parentid[ancestor_id])
        if ancestor_id == root_id:
            subtree_ids.append(body_id)
    assert root_id in subtree_ids
    return np.asarray(subtree_ids, dtype=int)


def _set_payload_test_configuration(model, data):
    arm_configuration = np.array((0.2, -1.0, 1.2, -0.7, 0.8, 0.4))
    for joint_name, value in zip(ARM_JOINT_NAMES, arm_configuration):
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
    data.qvel[:] = 0.0


def _warning_counts(data):
    return np.asarray([warning.number for warning in data.warning], dtype=int)


def test_half_millisecond_timestep_and_four_substep_schedule(model):
    """Four fixed MuJoCo steps must equal one 500 Hz control period."""
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.timestep == pytest.approx(0.0005, rel=0.0, abs=1.0e-15)

    data = mujoco.MjData(model)
    start_time = float(data.time)
    for _ in range(4):
        mujoco.mj_step(model, data)

    assert data.time - start_time == pytest.approx(
        0.002, rel=0.0, abs=1.0e-15
    )
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))


def test_arm_motor_controls_map_directly_to_generalized_actuator_force(model):
    """Unit-gear arm motors must put commanded torque on their joint DOFs."""
    data = mujoco.MjData(model)
    commands = np.array((25.0, -20.0, 15.0, -10.0, 8.0, -6.0))
    data.ctrl[:6] = commands
    mujoco.mj_forward(model, data)

    arm_dofs = _arm_dof_addresses(model)
    np.testing.assert_allclose(
        data.actuator_force[:6], commands, rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        data.qfrc_actuator[arm_dofs], commands, rtol=0.0, atol=1.0e-12
    )

    gripper_actuator_id = _id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_fingers_actuator"
    )
    data.ctrl[:] = 0.0
    data.ctrl[gripper_actuator_id] = 100.0
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(
        data.qfrc_actuator[arm_dofs], np.zeros(6), rtol=0.0, atol=1.0e-12
    )
    assert np.linalg.norm(data.qfrc_actuator[6:14]) > 0.0


def test_gripper_payload_changes_arm_bias_torque_and_free_acceleration():
    """Prove the articulated gripper inertia participates in arm dynamics."""
    loaded_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    reduced_payload_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    payload_body_ids = _gripper_subtree_body_ids(reduced_payload_model)
    reduced_payload_model.body_mass[payload_body_ids] *= 1.0e-6
    reduced_payload_model.body_inertia[payload_body_ids] *= 1.0e-6

    loaded_model.opt.disableflags |= int(
        mujoco.mjtDisableBit.mjDSBL_CONTACT
    )
    reduced_payload_model.opt.disableflags |= int(
        mujoco.mjtDisableBit.mjDSBL_CONTACT
    )
    loaded_data = mujoco.MjData(loaded_model)
    reduced_payload_data = mujoco.MjData(reduced_payload_model)
    mujoco.mj_setConst(reduced_payload_model, reduced_payload_data)
    _set_payload_test_configuration(loaded_model, loaded_data)
    _set_payload_test_configuration(
        reduced_payload_model, reduced_payload_data
    )

    mujoco.mj_forward(loaded_model, loaded_data)
    mujoco.mj_forward(reduced_payload_model, reduced_payload_data)
    arm_dofs = _arm_dof_addresses(loaded_model)
    bias_delta = (
        loaded_data.qfrc_bias[arm_dofs]
        - reduced_payload_data.qfrc_bias[arm_dofs]
    )
    acceleration_delta = (
        loaded_data.qacc[arm_dofs] - reduced_payload_data.qacc[arm_dofs]
    )

    assert np.all(np.isfinite(bias_delta))
    assert np.all(np.isfinite(acceleration_delta))
    assert np.linalg.norm(bias_delta) > 1.0
    assert np.linalg.norm(acceleration_delta) > 0.1


def test_nominal_contact_dynamics_remain_finite_without_solver_warnings(model):
    """Exercise falling-object contact and gripper constraints for one second."""
    data = mujoco.MjData(model)
    for _ in range(2000):
        mujoco.mj_step(model, data)

    assert data.time == pytest.approx(1.0, rel=0.0, abs=1.0e-12)
    assert data.ncon > 0
    for array in (
        data.qpos,
        data.qvel,
        data.qacc,
        data.qfrc_bias,
        data.qfrc_actuator,
        data.sensordata,
    ):
        assert np.all(np.isfinite(array))
    np.testing.assert_array_equal(
        _warning_counts(data), np.zeros(len(data.warning), dtype=int)
    )


@pytest.mark.parametrize(
    ("field_name", "warning_name"),
    (
        ("qpos", "mjWARN_BADQPOS"),
        ("ctrl", "mjWARN_BADCTRL"),
    ),
)
def test_nonfinite_inputs_are_reported_by_mujoco_warning_counters(
    model, field_name, warning_name, monkeypatch, tmp_path
):
    """A NaN state or command must never pass through without a warning."""
    # MuJoCo writes MUJOCO_LOG.TXT for these intentional warnings. Keep that
    # runtime artifact out of the release source tree.
    monkeypatch.chdir(tmp_path)
    data = mujoco.MjData(model)
    getattr(data, field_name)[0] = np.nan
    mujoco.mj_step(model, data)

    warning_type = getattr(mujoco.mjtWarning, warning_name)
    assert data.warning[int(warning_type)].number > 0
