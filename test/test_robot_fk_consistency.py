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

"""Cross-check the RViz URDF tree against the authoritative MuJoCo model."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import xacro


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MJCF_PATH = PROJECT_ROOT / "mujoco" / "ur5e_robotiq_2f85.xml"
URDF_PATH = PROJECT_ROOT / "urdf" / "ur5e_robotiq_2f85.urdf.xacro"
MESH_PREFIX = "package://neuro_adaptive_control/mujoco/assets/"

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
GRIPPER_JOINTS = (
    "gripper_right_driver_joint",
    "gripper_right_coupler_joint",
    "gripper_right_spring_link_joint",
    "gripper_right_follower_joint",
    "gripper_left_driver_joint",
    "gripper_left_coupler_joint",
    "gripper_left_spring_link_joint",
    "gripper_left_follower_joint",
)
JOINT_CONTRACT = {
    "shoulder_pan_joint": ((0.0, 0.0, 1.0), -6.28319, 6.28319),
    "shoulder_lift_joint": ((0.0, 1.0, 0.0), -6.28319, 6.28319),
    "elbow_joint": ((0.0, 1.0, 0.0), -3.1415, 3.1415),
    "wrist_1_joint": ((0.0, 1.0, 0.0), -6.28319, 6.28319),
    "wrist_2_joint": ((0.0, 0.0, 1.0), -6.28319, 6.28319),
    "wrist_3_joint": ((0.0, 1.0, 0.0), -6.28319, 6.28319),
    "gripper_right_driver_joint": ((1.0, 0.0, 0.0), 0.0, 0.8),
    "gripper_right_coupler_joint": ((1.0, 0.0, 0.0), -1.57, 0.0),
    "gripper_right_spring_link_joint": (
        (1.0, 0.0, 0.0),
        -0.296706,
        0.8,
    ),
    "gripper_right_follower_joint": (
        (1.0, 0.0, 0.0),
        -0.872664,
        0.872664,
    ),
    "gripper_left_driver_joint": ((1.0, 0.0, 0.0), 0.0, 0.8),
    "gripper_left_coupler_joint": ((1.0, 0.0, 0.0), -1.57, 0.0),
    "gripper_left_spring_link_joint": (
        (1.0, 0.0, 0.0),
        -0.296706,
        0.8,
    ),
    "gripper_left_follower_joint": (
        (1.0, 0.0, 0.0),
        -0.872664,
        0.872664,
    ),
}


def _numbers(value: str | None, default: str) -> np.ndarray:
    return np.asarray([float(item) for item in (value or default).split()])


def _translation(vector: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = vector
    return transform


def _rotation_transform(rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


def _rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cosine_roll, sine_roll = np.cos(roll), np.sin(roll)
    cosine_pitch, sine_pitch = np.cos(pitch), np.sin(pitch)
    cosine_yaw, sine_yaw = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array(
        (
            (1.0, 0.0, 0.0),
            (0.0, cosine_roll, -sine_roll),
            (0.0, sine_roll, cosine_roll),
        )
    )
    rotation_y = np.array(
        (
            (cosine_pitch, 0.0, sine_pitch),
            (0.0, 1.0, 0.0),
            (-sine_pitch, 0.0, cosine_pitch),
        )
    )
    rotation_z = np.array(
        (
            (cosine_yaw, -sine_yaw, 0.0),
            (sine_yaw, cosine_yaw, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    return rotation_z @ rotation_y @ rotation_x


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    unit_axis = axis / np.linalg.norm(axis)
    axis_cross = np.array(
        (
            (0.0, -unit_axis[2], unit_axis[1]),
            (unit_axis[2], 0.0, -unit_axis[0]),
            (-unit_axis[1], unit_axis[0], 0.0),
        )
    )
    return (
        np.eye(3)
        + np.sin(angle) * axis_cross
        + (1.0 - np.cos(angle)) * (axis_cross @ axis_cross)
    )


def _origin_transform(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    if origin is None:
        return np.eye(4)
    position = _numbers(origin.get("xyz"), "0 0 0")
    rotation = _rpy_rotation(_numbers(origin.get("rpy"), "0 0 0"))
    return _translation(position) @ _rotation_transform(rotation)


def _expanded_urdf() -> ET.Element:
    document = xacro.process_file(str(URDF_PATH))
    return ET.fromstring(document.toxml())


def _urdf_forward_kinematics(
    robot: ET.Element,
    joint_positions: dict[str, float],
) -> dict[str, np.ndarray]:
    joints = list(robot.findall("joint"))
    links = {link.get("name") for link in robot.findall("link")}
    child_links = {joint.find("child").get("link") for joint in joints}
    root_links = links - child_links
    assert root_links == {"world"}

    transforms = {"world": np.eye(4)}
    unresolved = joints.copy()
    while unresolved:
        progressed = False
        for joint in unresolved.copy():
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            child = joint.find("child").get("link")
            transform = transforms[parent] @ _origin_transform(joint)
            if joint.get("type") in {"revolute", "continuous"}:
                axis = _numbers(joint.find("axis").get("xyz"), "1 0 0")
                angle = joint_positions.get(joint.get("name"), 0.0)
                transform = transform @ _rotation_transform(
                    _axis_angle_rotation(axis, angle)
                )
            transforms[child] = transform
            unresolved.remove(joint)
            progressed = True
        assert progressed, "URDF contains a disconnected link or a kinematic cycle"
    return transforms


def _orientation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _mujoco_bindings():
    module = pytest.importorskip("mujoco")
    if not hasattr(module, "MjModel"):
        pytest.skip("the MuJoCo Python bindings are not installed")
    return module


def _named_transform(
    mujoco_module,
    model,
    data,
    object_type,
    name: str,
) -> np.ndarray:
    object_id = mujoco_module.mj_name2id(model, object_type, name)
    assert object_id >= 0, f"MuJoCo object is missing: {name}"
    if object_type == mujoco_module.mjtObj.mjOBJ_BODY:
        position = data.xpos[object_id]
        rotation = data.xmat[object_id].reshape(3, 3)
    elif object_type == mujoco_module.mjtObj.mjOBJ_SITE:
        position = data.site_xpos[object_id]
        rotation = data.site_xmat[object_id].reshape(3, 3)
    else:
        raise AssertionError(f"unsupported transform object type: {object_type}")
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def _assert_pose_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    position_limit: float = 0.001,
    angle_limit: float = np.deg2rad(0.5),
) -> None:
    position_error = np.linalg.norm(actual[:3, 3] - expected[:3, 3])
    angle_error = _orientation_distance(actual[:3, :3], expected[:3, :3])
    assert position_error <= position_limit
    assert angle_error <= angle_limit


def _joint_qpos_address(mujoco_module, model, name: str) -> int:
    joint_id = mujoco_module.mj_name2id(
        model,
        mujoco_module.mjtObj.mjOBJ_JOINT,
        name,
    )
    assert joint_id >= 0
    return int(model.jnt_qposadr[joint_id])


def _joint_dof_address(mujoco_module, model, name: str) -> int:
    joint_id = mujoco_module.mj_name2id(
        model,
        mujoco_module.mjtObj.mjOBJ_JOINT,
        name,
    )
    assert joint_id >= 0
    return int(model.jnt_dofadr[joint_id])


def _settle_gripper(mujoco_module, control: float) -> dict[str, float]:
    model = mujoco_module.MjModel.from_xml_path(str(MJCF_PATH))
    model.opt.gravity[:] = 0.0
    model.opt.disableflags |= int(mujoco_module.mjtDisableBit.mjDSBL_CONTACT)
    data = mujoco_module.MjData(model)
    arm_target = np.array(
        (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
    )
    arm_actuators = tuple(
        mujoco_module.mj_name2id(
            model,
            mujoco_module.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        for name in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
    )
    gripper_actuator = mujoco_module.mj_name2id(
        model,
        mujoco_module.mjtObj.mjOBJ_ACTUATOR,
        "gripper_fingers_actuator",
    )
    for index, name in enumerate(ARM_JOINTS):
        data.qpos[_joint_qpos_address(mujoco_module, model, name)] = arm_target[index]
    mujoco_module.mj_forward(model, data)

    for _ in range(3000):
        for index, name in enumerate(ARM_JOINTS):
            qpos_address = _joint_qpos_address(mujoco_module, model, name)
            dof_address = _joint_dof_address(mujoco_module, model, name)
            actuator = arm_actuators[index]
            effort = data.qfrc_bias[dof_address]
            effort += 100.0 * (arm_target[index] - data.qpos[qpos_address])
            effort -= 10.0 * data.qvel[dof_address]
            lower, upper = model.actuator_ctrlrange[actuator]
            data.ctrl[actuator] = np.clip(effort, lower, upper)
        data.ctrl[gripper_actuator] = control
        mujoco_module.mj_step(model, data)

    gripper_velocity = np.array(
        [
            data.qvel[_joint_dof_address(mujoco_module, model, name)]
            for name in GRIPPER_JOINTS
        ]
    )
    assert np.max(np.abs(gripper_velocity)) < 1e-3
    equality_mask = data.efc_type[: data.nefc] == int(
        mujoco_module.mjtConstraint.mjCNSTR_EQUALITY
    )
    assert np.max(np.abs(data.efc_pos[: data.nefc][equality_mask])) < 1e-4
    return {
        name: float(data.qpos[_joint_qpos_address(mujoco_module, model, name)])
        for name in GRIPPER_JOINTS
    }


def test_description_mirrors_joint_contract_and_all_robot_bodies() -> None:
    """Require every simulated robot body and movable joint in the URDF."""
    robot = _expanded_urdf()
    urdf_joints = {joint.get("name"): joint for joint in robot.findall("joint")}
    nonfixed = {
        name for name, joint in urdf_joints.items() if joint.get("type") != "fixed"
    }
    assert nonfixed == set(ARM_JOINTS + GRIPPER_JOINTS)
    assert not robot.findall(".//mimic")

    mjcf_root = ET.parse(MJCF_PATH).getroot()
    mjcf_base = mjcf_root.find("./worldbody/body[@name='base']")
    mjcf_bodies = {
        body.get("name") for body in mjcf_base.iter("body") if body.get("name")
    }
    urdf_links = {link.get("name") for link in robot.findall("link")}
    assert mjcf_bodies <= urdf_links
    assert {"attachment_site", "wrist_ft_site", "gripper_pinch"} <= urdf_links

    mjcf_axes = {
        joint.get("name"): _numbers(joint.get("axis"), "0 0 1")
        for joint in mjcf_base.iter("joint")
    }
    for name, (axis, lower, upper) in JOINT_CONTRACT.items():
        joint = urdf_joints[name]
        np.testing.assert_allclose(
            _numbers(joint.find("axis").get("xyz"), "0 0 1"),
            axis,
            atol=0.0,
        )
        np.testing.assert_allclose(mjcf_axes[name], axis, atol=0.0)
        limit = joint.find("limit")
        assert float(limit.get("lower")) == pytest.approx(lower)
        assert float(limit.get("upper")) == pytest.approx(upper)


def test_visual_meshes_are_complete_package_local_assets() -> None:
    """Require every visual to resolve to the pinned package-local mesh set."""
    robot = _expanded_urdf()
    mesh_elements = robot.findall(".//visual/geometry/mesh")
    filenames = {mesh.get("filename") for mesh in mesh_elements}
    expected_assets = {
        f"{MESH_PREFIX}{path.name}"
        for path in (PROJECT_ROOT / "mujoco" / "assets").iterdir()
        if path.suffix in {".obj", ".stl"}
    }
    assert filenames == expected_assets
    for mesh in mesh_elements:
        filename = mesh.get("filename")
        assert filename.startswith(MESH_PREFIX)
        asset = PROJECT_ROOT / "mujoco" / "assets" / Path(filename).name
        assert asset.is_file()
        scale = _numbers(mesh.get("scale"), "1 1 1")
        expected_scale = (0.001, 0.001, 0.001) if asset.suffix == ".stl" else (1, 1, 1)
        np.testing.assert_allclose(scale, expected_scale, atol=0.0)


def test_random_arm_fk_matches_mujoco_tcp_and_attachment() -> None:
    """Match the flange and pinch frames over deterministic random arm poses."""
    mujoco_module = _mujoco_bindings()
    model = mujoco_module.MjModel.from_xml_path(str(MJCF_PATH))
    robot = _expanded_urdf()
    generator = np.random.default_rng(20260803)
    for _ in range(32):
        data = mujoco_module.MjData(model)
        joint_positions = {}
        for name in ARM_JOINTS:
            joint_id = mujoco_module.mj_name2id(
                model,
                mujoco_module.mjtObj.mjOBJ_JOINT,
                name,
            )
            lower, upper = model.jnt_range[joint_id]
            value = float(generator.uniform(0.45 * lower, 0.45 * upper))
            data.qpos[model.jnt_qposadr[joint_id]] = value
            joint_positions[name] = value
        mujoco_module.mj_forward(model, data)
        urdf_transforms = _urdf_forward_kinematics(robot, joint_positions)
        for name in ("attachment_site", "gripper_pinch"):
            actual = _named_transform(
                mujoco_module,
                model,
                data,
                mujoco_module.mjtObj.mjOBJ_SITE,
                name,
            )
            _assert_pose_close(actual, urdf_transforms[name])


def test_constraint_settled_gripper_fk_and_pad_endpoints_match_mujoco() -> None:
    """Match both four-bar trees after MuJoCo settles their constraints."""
    mujoco_module = _mujoco_bindings()
    settled = tuple(
        _settle_gripper(mujoco_module, control) for control in (32.0, 128.0, 224.0)
    )
    model = mujoco_module.MjModel.from_xml_path(str(MJCF_PATH))
    robot = _expanded_urdf()
    generator = np.random.default_rng(85)
    compared_bodies = (
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
    for gripper_positions in settled:
        for _ in range(8):
            data = mujoco_module.MjData(model)
            joint_positions = dict(gripper_positions)
            for name in ARM_JOINTS:
                joint_id = mujoco_module.mj_name2id(
                    model,
                    mujoco_module.mjtObj.mjOBJ_JOINT,
                    name,
                )
                lower, upper = model.jnt_range[joint_id]
                value = float(generator.uniform(0.35 * lower, 0.35 * upper))
                joint_positions[name] = value
            for name, value in joint_positions.items():
                address = _joint_qpos_address(mujoco_module, model, name)
                data.qpos[address] = value
            mujoco_module.mj_forward(model, data)
            urdf_transforms = _urdf_forward_kinematics(robot, joint_positions)

            for name in compared_bodies:
                actual = _named_transform(
                    mujoco_module,
                    model,
                    data,
                    mujoco_module.mjtObj.mjOBJ_BODY,
                    name,
                )
                _assert_pose_close(actual, urdf_transforms[name])

            for side in ("right", "left"):
                geom_name = f"gripper_{side}_pad1"
                geom_id = mujoco_module.mj_name2id(
                    model,
                    mujoco_module.mjtObj.mjOBJ_GEOM,
                    geom_name,
                )
                geom_rotation = data.geom_xmat[geom_id].reshape(3, 3)
                local_endpoint = np.array(
                    (0.0, -model.geom_size[geom_id, 1], model.geom_size[geom_id, 2])
                )
                endpoint = data.geom_xpos[geom_id] + geom_rotation @ local_endpoint
                urdf_endpoint = urdf_transforms[f"gripper_{side}_pad_tip"][:3, 3]
                assert np.linalg.norm(endpoint - urdf_endpoint) <= 0.001
