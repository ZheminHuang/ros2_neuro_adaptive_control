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

"""Structural and physical contract tests for the UR5e/2F-85 MJCF."""

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
ARM_ACTUATOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)


@pytest.fixture()
def model():
    """Compile a fresh model so tests cannot leak mutable MuJoCo state."""
    assert MODEL_PATH.is_file()
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def _name(model, object_type, object_id):
    return mujoco.mj_id2name(model, object_type, int(object_id))


def _id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, f"missing {object_type}: {name}"
    return object_id


def test_model_parses_with_expected_dimensions_and_names(model):
    """Lock the compiled generalized-coordinate and interface dimensions."""
    assert model.nq == 21
    assert model.nv == 20
    assert model.njnt == 15
    assert model.nu == 7
    assert model.ntendon == 1
    assert model.neq == 3
    assert model.nsensor == 2

    joint_names = tuple(
        _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
    )
    assert joint_names == (
        *ARM_JOINT_NAMES,
        *GRIPPER_JOINT_NAMES,
        "object_freejoint",
    )

    actuator_names = tuple(
        _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        for actuator_id in range(model.nu)
    )
    assert actuator_names == (*ARM_ACTUATOR_NAMES, "gripper_fingers_actuator")


def test_joint_and_actuator_limits_are_explicit(model):
    """Require finite joint, command, and force bounds for every actuator."""
    expected_joint_ranges = {
        "shoulder_pan_joint": (-6.28319, 6.28319),
        "shoulder_lift_joint": (-6.28319, 6.28319),
        "elbow_joint": (-3.1415, 3.1415),
        "wrist_1_joint": (-6.28319, 6.28319),
        "wrist_2_joint": (-6.28319, 6.28319),
        "wrist_3_joint": (-6.28319, 6.28319),
        "gripper_right_driver_joint": (0.0, 0.8),
        "gripper_right_coupler_joint": (-1.57, 0.0),
        "gripper_right_spring_link_joint": (-0.296706, 0.8),
        "gripper_right_follower_joint": (-0.872664, 0.872664),
        "gripper_left_driver_joint": (0.0, 0.8),
        "gripper_left_coupler_joint": (-1.57, 0.0),
        "gripper_left_spring_link_joint": (-0.296706, 0.8),
        "gripper_left_follower_joint": (-0.872664, 0.872664),
    }
    for joint_name, expected_range in expected_joint_ranges.items():
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        assert bool(model.jnt_limited[joint_id])
        np.testing.assert_allclose(
            model.jnt_range[joint_id], expected_range, rtol=0.0, atol=1.0e-12
        )

    free_joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
    assert model.jnt_type[free_joint_id] == mujoco.mjtJoint.mjJNT_FREE
    assert not bool(model.jnt_limited[free_joint_id])

    expected_motor_limits = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)
    for actuator_name, joint_name, limit in zip(
        ARM_ACTUATOR_NAMES, ARM_JOINT_NAMES, expected_motor_limits
    ):
        actuator_id = _id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT
        assert model.actuator_trnid[actuator_id, 0] == joint_id
        assert bool(model.actuator_ctrllimited[actuator_id])
        assert bool(model.actuator_forcelimited[actuator_id])
        np.testing.assert_allclose(
            model.actuator_ctrlrange[actuator_id],
            (-limit, limit),
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            model.actuator_forcerange[actuator_id],
            (-limit, limit),
            rtol=0.0,
            atol=1.0e-12,
        )

    gripper_id = _id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_fingers_actuator"
    )
    tendon_id = _id(model, mujoco.mjtObj.mjOBJ_TENDON, "gripper_split")
    assert model.actuator_trntype[gripper_id] == mujoco.mjtTrn.mjTRN_TENDON
    assert model.actuator_trnid[gripper_id, 0] == tendon_id
    assert bool(model.actuator_ctrllimited[gripper_id])
    assert bool(model.actuator_forcelimited[gripper_id])
    np.testing.assert_allclose(
        model.actuator_ctrlrange[gripper_id], (0.0, 255.0)
    )
    np.testing.assert_allclose(
        model.actuator_forcerange[gripper_id], (-5.0, 5.0)
    )


def test_all_bodies_with_degrees_of_freedom_have_physical_inertia(model):
    """Require positive physical mass and realizable principal inertias."""
    dynamic_body_ids = np.flatnonzero(model.body_dofnum > 0)
    assert dynamic_body_ids.size == 15

    for body_id in dynamic_body_ids:
        body_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        mass = float(model.body_mass[body_id])
        inertia = np.asarray(model.body_inertia[body_id], dtype=float)
        assert np.isfinite(mass) and mass > 0.0, body_name
        assert np.all(np.isfinite(inertia)) and np.all(inertia > 0.0), body_name

        principal = np.sort(inertia)
        assert principal[2] < principal[0] + principal[1], body_name


def test_generalized_mass_matrix_is_symmetric_positive_definite(model):
    """Check the compiled inertia matrix over deterministic legal states."""
    rng = np.random.default_rng(19)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

    for _ in range(12):
        data = mujoco.MjData(model)
        for joint_id in range(model.njnt):
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            low, high = model.jnt_range[joint_id]
            data.qpos[model.jnt_qposadr[joint_id]] = rng.uniform(low, high)

        object_joint_id = _id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint"
        )
        object_qpos_address = model.jnt_qposadr[object_joint_id]
        quaternion = rng.normal(size=4)
        data.qpos[object_qpos_address + 3:object_qpos_address + 7] = (
            quaternion / np.linalg.norm(quaternion)
        )

        mujoco.mj_forward(model, data)
        mass_matrix = np.empty((model.nv, model.nv), dtype=float)
        mujoco.mj_fullM(model, mass_matrix, data.qM)

        assert np.all(np.isfinite(mass_matrix))
        np.testing.assert_allclose(
            mass_matrix, mass_matrix.T, rtol=0.0, atol=1.0e-12
        )
        eigenvalues = np.linalg.eigvalsh(mass_matrix)
        assert float(eigenvalues[0]) > 0.0


def test_gripper_tendon_and_equality_constraints_are_compiled(model):
    """Verify the two-finger tendon split and closed-chain constraints."""
    tendon_id = _id(model, mujoco.mjtObj.mjOBJ_TENDON, "gripper_split")
    tendon_address = model.tendon_adr[tendon_id]
    tendon_count = model.tendon_num[tendon_id]
    tendon_slice = slice(tendon_address, tendon_address + tendon_count)
    assert tendon_count == 2
    assert np.all(
        model.wrap_type[tendon_slice] == mujoco.mjtWrap.mjWRAP_JOINT
    )
    wrapped_joint_names = {
        _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in model.wrap_objid[tendon_slice]
    }
    assert wrapped_joint_names == {
        "gripper_right_driver_joint",
        "gripper_left_driver_joint",
    }
    np.testing.assert_allclose(
        model.wrap_prm[tendon_slice], (0.5, 0.5), rtol=0.0, atol=1.0e-12
    )

    connect_ids = np.flatnonzero(model.eq_type == mujoco.mjtEq.mjEQ_CONNECT)
    joint_ids = np.flatnonzero(model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
    assert connect_ids.size == 2
    assert joint_ids.size == 1
    assert np.all(model.eq_active0)

    connect_pairs = {
        frozenset(
            (
                _name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    model.eq_obj1id[equality_id],
                ),
                _name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    model.eq_obj2id[equality_id],
                ),
            )
        )
        for equality_id in connect_ids
    }
    assert connect_pairs == {
        frozenset(("gripper_right_follower", "gripper_right_coupler")),
        frozenset(("gripper_left_follower", "gripper_left_coupler")),
    }

    joint_equality_id = int(joint_ids[0])
    coupled_joint_names = {
        _name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            model.eq_obj1id[joint_equality_id],
        ),
        _name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            model.eq_obj2id[joint_equality_id],
        ),
    }
    assert coupled_joint_names == {
        "gripper_right_driver_joint",
        "gripper_left_driver_joint",
    }
    np.testing.assert_allclose(
        model.eq_data[joint_equality_id, :5],
        (0.0, 1.0, 0.0, 0.0, 0.0),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        model.eq_solref,
        np.tile((0.005, 1.0), (model.neq, 1)),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        model.eq_solimp[:, :3],
        np.tile((0.95, 0.99, 0.001), (model.neq, 1)),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_gripper_collisions_friction_and_contact_solver_parameters(model):
    """Require active finger collisions and explicit contact parameters."""
    expected_collision_bodies = (
        "gripper_base_mount",
        "gripper_base",
        "gripper_right_driver",
        "gripper_right_coupler",
        "gripper_right_spring_link",
        "gripper_right_follower",
        "gripper_right_pad",
        "gripper_left_driver",
        "gripper_left_coupler",
        "gripper_left_spring_link",
        "gripper_left_follower",
        "gripper_left_pad",
    )
    contact_enabled = (model.geom_contype != 0) & (
        model.geom_conaffinity != 0
    )
    assert np.all(model.geom_group[contact_enabled] == 3)
    assert np.all(model.geom_group[~contact_enabled] == 2)
    for body_name in expected_collision_bodies:
        body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        body_contact_geoms = contact_enabled & (model.geom_bodyid == body_id)
        assert np.any(body_contact_geoms), body_name
        assert np.all(model.geom_friction[body_contact_geoms] > 0.0)
        assert np.all(model.geom_solref[body_contact_geoms] > 0.0)
        assert np.all(model.geom_solimp[body_contact_geoms, :3] > 0.0)

    expected_pad_friction = {
        "gripper_right_pad1": 0.7,
        "gripper_right_pad2": 0.6,
        "gripper_left_pad1": 0.7,
        "gripper_left_pad2": 0.6,
    }
    for geom_name, sliding_friction in expected_pad_friction.items():
        geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        assert contact_enabled[geom_id]
        np.testing.assert_allclose(
            model.geom_friction[geom_id],
            (sliding_friction, 0.005, 0.0001),
        )
        np.testing.assert_allclose(model.geom_solref[geom_id], (0.004, 1.0))
        np.testing.assert_allclose(
            model.geom_solimp[geom_id, :3], (0.95, 0.99, 0.001)
        )

    for geom_name in ("ground", "table_top", "grasp_object_collision"):
        geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        assert contact_enabled[geom_id]
        assert model.geom_group[geom_id] == 3
        assert model.geom_rgba[geom_id, 3] == 0.0
        assert np.all(model.geom_friction[geom_id] > 0.0)
        assert np.all(model.geom_solref[geom_id] > 0.0)
        assert np.all(model.geom_solimp[geom_id, :3] > 0.0)

    for geom_name in (
        "ground_visual",
        "table_top_visual",
        "grasp_object_visual",
    ):
        geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        assert not contact_enabled[geom_id]
        assert model.geom_group[geom_id] == 2
        assert model.geom_rgba[geom_id, 3] == 1.0

    expected_exclusions = (
        ("gripper_base", "gripper_right_driver"),
        ("gripper_base", "gripper_right_spring_link"),
        ("gripper_base", "gripper_left_driver"),
        ("gripper_base", "gripper_left_spring_link"),
        ("gripper_right_coupler", "gripper_right_follower"),
        ("gripper_left_coupler", "gripper_left_follower"),
    )
    expected_signatures = set()
    for first_name, second_name in expected_exclusions:
        first_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, first_name)
        second_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, second_name)
        low, high = sorted((first_id, second_id))
        expected_signatures.add((low << 16) + high)
    assert set(model.exclude_signature.tolist()) == expected_signatures
