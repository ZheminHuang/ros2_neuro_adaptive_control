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

"""
MuJoCo plant boundary for the UR5e + dynamic Robotiq 2F-85 model.

This module is deliberately outside :mod:`neuro_adaptive_control.core`.
MuJoCo owns dynamics, constraints, contacts, and sensors.  The NAC receives
only measured state, FK/Jacobian kinematics, and the selected external force.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .robotiq_gripper_adapter import GripperState, RobotiqGripperAdapter

try:  # Keep the v0.1 NumPy demo importable without the optional dependency.
    import mujoco
except ImportError:  # pragma: no cover - exercised in the minimal ROS image.
    mujoco = None
if mujoco is not None and not hasattr(mujoco, "MjModel"):
    mujoco = None


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
HOME_ARM_Q = np.array(
    (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0), dtype=float
)


def require_mujoco() -> None:
    """Raise an actionable error when the optional physics package is absent."""
    if mujoco is None:
        raise RuntimeError(
            "MuJoCo is required for this demo; install the pinned dependency "
            "with `python3 -m pip install mujoco==3.9.0`."
        )


def default_model_path() -> Path:
    """Return the source-tree model path for direct Python use."""
    return Path(__file__).resolve().parents[2] / "mujoco" / "ur5e_robotiq_2f85.xml"


def _finite_vector(value: Iterable[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},).")
    return array.copy()


@dataclass(frozen=True)
class MujocoKinematicState:
    """One coherent state sample at a MuJoCo control stamp."""

    stamp_sec: float
    sequence_id: int
    arm_position: np.ndarray
    arm_velocity: np.ndarray
    all_joint_position: np.ndarray
    all_joint_velocity: np.ndarray
    tcp_position: np.ndarray
    tcp_rotation: np.ndarray
    tcp_linear_velocity: np.ndarray
    tcp_angular_velocity: np.ndarray
    translational_jacobian: np.ndarray
    rotational_jacobian: np.ndarray
    object_position: np.ndarray
    object_rotation: np.ndarray


@dataclass(frozen=True)
class ContactSummary:
    """Contact-only environment-on-robot wrench, shifted to the TCP."""

    force_world: np.ndarray
    torque_world_at_tcp: np.ndarray
    left_finger_contacts: int
    right_finger_contacts: int
    total_robot_environment_contacts: int
    expected_object_contacts: int
    unexpected_contacts: int
    contact_force_norm_n: float
    maximum_penetration_m: float
    positions_world: tuple[np.ndarray, ...]
    normals_world: tuple[np.ndarray, ...]
    forces_world: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class WristWrench:
    """Raw MuJoCo cut-wrench sensor transformed into the world frame."""

    force_world: np.ndarray
    torque_world: np.ndarray


class MujocoUR5ePlant:
    """Own the only MuJoCo model/data instance used by control and viewers."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        seed: int = 23,
        gripper: RobotiqGripperAdapter | None = None,
        payload_mass_kg: float | None = None,
        payload_com_offset_m: Iterable[float] = (0.0, 0.0, 0.0),
        payload_inertia_scale: float = 1.0,
    ) -> None:
        require_mujoco()
        path = Path(model_path) if model_path is not None else default_model_path()
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {path}")
        self.model_path = path.resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        if not np.isclose(self.model.opt.timestep, 0.0005, atol=1e-15):
            raise ValueError("model timestep must be exactly 0.0005 s")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.gripper = gripper or RobotiqGripperAdapter()
        self.sequence_id = 0
        self.step_count = 0
        self._arm_joint_ids = self._joint_ids(ARM_JOINT_NAMES)
        self._gripper_joint_ids = self._joint_ids(GRIPPER_JOINT_NAMES)
        self._arm_qpos = self._addresses(ARM_JOINT_NAMES, qpos=True)
        self._arm_dof = self._addresses(ARM_JOINT_NAMES, qpos=False)
        self._gripper_qpos = self._addresses(GRIPPER_JOINT_NAMES, qpos=True)
        self._gripper_dof = self._addresses(GRIPPER_JOINT_NAMES, qpos=False)
        self._right_driver_qpos = self._joint_address(
            "gripper_right_driver_joint", qpos=True
        )
        self._left_driver_qpos = self._joint_address(
            "gripper_left_driver_joint", qpos=True
        )
        self._tcp_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "gripper_pinch")
        self._ft_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "wrist_ft_site")
        self._object_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "grasp_object"
        )
        nominal_mass = float(self.model.body_mass[self._object_body_id])
        selected_mass = nominal_mass if payload_mass_kg is None else float(
            payload_mass_kg
        )
        inertia_scale = float(payload_inertia_scale)
        com_offset = _finite_vector(
            payload_com_offset_m,
            3,
            "payload_com_offset_m",
        )
        if not np.isfinite(selected_mass) or selected_mass <= 0.0:
            raise ValueError("payload_mass_kg must be finite and positive")
        if not np.isfinite(inertia_scale) or inertia_scale <= 0.0:
            raise ValueError("payload_inertia_scale must be finite and positive")
        mass_ratio = selected_mass / nominal_mass
        self.model.body_mass[self._object_body_id] = selected_mass
        self.model.body_inertia[self._object_body_id] *= (
            mass_ratio * inertia_scale
        )
        self.model.body_ipos[self._object_body_id] += com_offset
        mujoco.mj_setConst(self.model, self.data)
        self._injection_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "gripper_base_mount"
        )
        self._gripper_actuator_id = self._id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_fingers_actuator"
        )
        self._force_sensor_id = self._id(
            mujoco.mjtObj.mjOBJ_SENSOR, "wrist_force"
        )
        self._torque_sensor_id = self._id(
            mujoco.mjtObj.mjOBJ_SENSOR, "wrist_torque"
        )
        self._robot_bodies = self._descendants(
            self._id(mujoco.mjtObj.mjOBJ_BODY, "base")
        )
        self._object_bodies = self._descendants(self._object_body_id)
        self._left_finger_bodies = self._descendants(
            self._id(mujoco.mjtObj.mjOBJ_BODY, "gripper_left_spring_link")
        ) | self._descendants(
            self._id(mujoco.mjtObj.mjOBJ_BODY, "gripper_left_driver")
        )
        self._right_finger_bodies = self._descendants(
            self._id(mujoco.mjtObj.mjOBJ_BODY, "gripper_right_spring_link")
        ) | self._descendants(
            self._id(mujoco.mjtObj.mjOBJ_BODY, "gripper_right_driver")
        )
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qpos[self._arm_qpos] = HOME_ARM_Q
        self._initial_qpos[self._gripper_qpos] = 0.0
        self._initial_object_qpos = self._free_joint_qpos("object_freejoint")
        self._initial_ctrl = np.zeros(self.model.nu, dtype=float)
        self._initial_ctrl[self._gripper_actuator_id] = 0.0
        self.reset()

    @property
    def control_period(self) -> float:
        """Return the mandated 4-substep control period."""
        return 4.0 * float(self.model.opt.timestep)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Return all articulated arm and gripper joints in qpos order."""
        return ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES

    @property
    def joint_limits(self) -> np.ndarray:
        """Return lower/upper limits for every published articulated joint."""
        ids = np.concatenate((self._arm_joint_ids, self._gripper_joint_ids))
        if not np.all(self.model.jnt_limited[ids]):
            raise RuntimeError("every arm and gripper joint must be limited")
        return self.model.jnt_range[ids].copy()

    @property
    def arm_velocity(self) -> np.ndarray:
        """Return the measured six-joint velocity without exposing dynamics."""
        return self.data.qvel[self._arm_dof].copy()

    def _id(self, kind, name: str) -> int:
        identifier = int(mujoco.mj_name2id(self.model, kind, name))
        if identifier < 0:
            raise ValueError(f"MuJoCo model is missing required name {name!r}")
        return identifier

    def _joint_address(self, name: str, *, qpos: bool) -> int:
        joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
        addresses = self.model.jnt_qposadr if qpos else self.model.jnt_dofadr
        return int(addresses[joint_id])

    def _joint_ids(self, names: tuple[str, ...]) -> np.ndarray:
        return np.asarray(
            [self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in names],
            dtype=int,
        )

    def _addresses(self, names: tuple[str, ...], *, qpos: bool) -> np.ndarray:
        return np.asarray(
            [self._joint_address(name, qpos=qpos) for name in names], dtype=int
        )

    def _free_joint_qpos(self, name: str) -> np.ndarray:
        start = self._joint_address(name, qpos=True)
        return self._initial_qpos[start:start + 7].copy()

    def _descendants(self, root_body: int) -> set[int]:
        descendants = {root_body}
        changed = True
        while changed:
            changed = False
            for body_id in range(1, self.model.nbody):
                if (
                    int(self.model.body_parentid[body_id]) in descendants
                    and body_id not in descendants
                ):
                    descendants.add(body_id)
                    changed = True
        return descendants

    def reset(self) -> MujocoKinematicState:
        """Reset every dynamic, actuator, applied-force, and RNG state."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self._initial_ctrl
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.gripper.reset()
        self.model.actuator_forcerange[self._gripper_actuator_id] = (
            -self.gripper.maximum_effort_n,
            self.gripper.maximum_effort_n,
        )
        self.sequence_id = 0
        self.step_count = 0
        self.rng = np.random.default_rng(self.seed)
        mujoco.mj_forward(self.model, self.data)
        self._desired_tcp_rotation = self._tcp_rotation().copy()
        return self.kinematic_state()

    def _tcp_rotation(self) -> np.ndarray:
        return self.data.site_xmat[self._tcp_site_id].reshape(3, 3).copy()

    def kinematic_state(self) -> MujocoKinematicState:
        """Read FK, spatial Jacobian, and joint state without dynamics truth."""
        mujoco.mj_forward(self.model, self.data)
        jacobian_v = np.zeros((3, self.model.nv), dtype=float)
        jacobian_w = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacobian_v,
            jacobian_w,
            self._tcp_site_id,
        )
        arm_velocity = self.data.qvel[self._arm_dof].copy()
        tcp_linear = jacobian_v[:, self._arm_dof] @ arm_velocity
        tcp_angular = jacobian_w[:, self._arm_dof] @ arm_velocity
        return MujocoKinematicState(
            stamp_sec=float(self.data.time),
            sequence_id=self.sequence_id,
            arm_position=self.data.qpos[self._arm_qpos].copy(),
            arm_velocity=arm_velocity,
            all_joint_position=self.data.qpos[
                np.concatenate((self._arm_qpos, self._gripper_qpos))
            ].copy(),
            all_joint_velocity=self.data.qvel[
                np.concatenate((self._arm_dof, self._gripper_dof))
            ].copy(),
            tcp_position=self.data.site_xpos[self._tcp_site_id].copy(),
            tcp_rotation=self._tcp_rotation(),
            tcp_linear_velocity=tcp_linear,
            tcp_angular_velocity=tcp_angular,
            translational_jacobian=jacobian_v[:, self._arm_dof].copy(),
            rotational_jacobian=jacobian_w[:, self._arm_dof].copy(),
            object_position=self.data.xpos[self._object_body_id].copy(),
            object_rotation=self.data.xmat[self._object_body_id]
            .reshape(3, 3)
            .copy(),
        )

    @property
    def desired_tcp_rotation(self) -> np.ndarray:
        """Return the fixed orientation captured by deterministic reset."""
        return self._desired_tcp_rotation.copy()

    def wrist_wrench_raw(self) -> WristWrench:
        """Return the sensor cut-wrench; it is never the default NAC input."""
        force_adr = int(self.model.sensor_adr[self._force_sensor_id])
        torque_adr = int(self.model.sensor_adr[self._torque_sensor_id])
        local_force = self.data.sensordata[force_adr:force_adr + 3]
        local_torque = self.data.sensordata[torque_adr:torque_adr + 3]
        rotation = self.data.site_xmat[self._ft_site_id].reshape(3, 3)
        return WristWrench(
            force_world=(rotation @ local_force).copy(),
            torque_world=(rotation @ local_torque).copy(),
        )

    def contact_summary(self) -> ContactSummary:
        """Sum only external environment-on-robot contacts at the TCP point."""
        tcp = self.data.site_xpos[self._tcp_site_id]
        force_sum = np.zeros(3, dtype=float)
        torque_sum = np.zeros(3, dtype=float)
        positions: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        forces: list[np.ndarray] = []
        left_contacts = 0
        right_contacts = 0
        count = 0
        expected_object_contacts = 0
        unexpected_contacts = 0
        maximum_penetration = 0.0
        norm_sum = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            robot1 = body1 in self._robot_bodies
            robot2 = body2 in self._robot_bodies
            maximum_penetration = max(
                maximum_penetration, max(0.0, -float(contact.dist))
            )
            if robot1 == robot2:
                continue
            local = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(self.model, self.data, index, local)
            frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
            world_force_on_geom2 = frame.T @ local[:3]
            world_torque_on_geom2 = frame.T @ local[3:]
            if robot2:
                world_force = world_force_on_geom2
                world_torque = world_torque_on_geom2
                robot_body = body2
                environment_body = body1
            else:
                world_force = -world_force_on_geom2
                world_torque = -world_torque_on_geom2
                robot_body = body1
                environment_body = body2
            position = np.asarray(contact.pos, dtype=float).copy()
            force_sum += world_force
            torque_sum += world_torque + np.cross(position - tcp, world_force)
            force_norm = float(np.linalg.norm(world_force))
            norm_sum += force_norm
            positions.append(position)
            normal = frame[0].copy()
            if robot1:
                normal = -normal
            normals.append(normal)
            forces.append(world_force.copy())
            count += 1
            if environment_body in self._object_bodies:
                if robot_body in self._left_finger_bodies:
                    left_contacts += 1
                if robot_body in self._right_finger_bodies:
                    right_contacts += 1
                if robot_body in (
                    self._left_finger_bodies | self._right_finger_bodies
                ):
                    expected_object_contacts += 1
                else:
                    unexpected_contacts += 1
            else:
                unexpected_contacts += 1
        return ContactSummary(
            force_world=force_sum,
            torque_world_at_tcp=torque_sum,
            left_finger_contacts=left_contacts,
            right_finger_contacts=right_contacts,
            total_robot_environment_contacts=count,
            expected_object_contacts=expected_object_contacts,
            unexpected_contacts=unexpected_contacts,
            contact_force_norm_n=norm_sum,
            maximum_penetration_m=maximum_penetration,
            positions_world=tuple(positions),
            normals_world=tuple(normals),
            forces_world=tuple(forces),
        )

    def gripper_state(self, contacts: ContactSummary | None = None) -> GripperState:
        """Return metric gripper state from dynamic joints and actuator force."""
        summary = contacts if contacts is not None else self.contact_summary()
        return self.gripper.feedback(
            right_driver_rad=float(self.data.qpos[self._right_driver_qpos]),
            left_driver_rad=float(self.data.qpos[self._left_driver_qpos]),
            actuator_force_n=float(
                self.data.actuator_force[self._gripper_actuator_id]
            ),
            left_contacts=summary.left_finger_contacts,
            right_contacts=summary.right_finger_contacts,
            contact_force_n=summary.contact_force_norm_n,
        )

    def apply_safe_hold(self, arm_damping_torque: Iterable[float]) -> None:
        """Replace every old command with bounded damping and gripper hold."""
        damping = _finite_vector(
            arm_damping_torque, 6, "arm_damping_torque"
        )
        lower = self.model.actuator_ctrlrange[:6, 0]
        upper = self.model.actuator_ctrlrange[:6, 1]
        self.data.ctrl[:] = 0.0
        self.data.ctrl[:6] = np.clip(damping, lower, upper)
        try:
            self.gripper.stop(self.gripper_state().opening_m)
            self.data.ctrl[self._gripper_actuator_id] = (
                self.gripper.actuator_control()
            )
        except (FloatingPointError, RuntimeError, ValueError):
            # A non-finite gripper measurement cannot be trusted for holding.
            self.data.ctrl[self._gripper_actuator_id] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

    def advance(
        self,
        arm_torque: Iterable[float],
        *,
        injected_force_world: Iterable[float] = (0.0, 0.0, 0.0),
        substeps: int = 4,
    ) -> MujocoKinematicState:
        """Advance exactly four ZOH substeps using one accepted command stamp."""
        if substeps != 4:
            raise ValueError("the v0.2 contract requires exactly four substeps")
        torque = _finite_vector(arm_torque, 6, "arm_torque")
        injected = _finite_vector(
            injected_force_world, 3, "injected_force_world"
        )
        if self.sequence_id != self.step_count:
            raise RuntimeError("stale or duplicate command sequence")
        self.data.ctrl[:6] = torque
        self.data.ctrl[self._gripper_actuator_id] = self.gripper.actuator_control()
        effort = self.gripper.maximum_effort_n
        self.model.actuator_forcerange[self._gripper_actuator_id] = (-effort, effort)
        warning_counts = np.asarray(
            [warning.number for warning in self.data.warning], dtype=int
        )
        for _ in range(substeps):
            self.data.qfrc_applied[:] = 0.0
            if np.any(injected):
                point = self.data.site_xpos[self._tcp_site_id].copy()
                mujoco.mj_applyFT(
                    self.model,
                    self.data,
                    injected,
                    np.zeros(3),
                    point,
                    self._injection_body_id,
                    self.data.qfrc_applied,
                )
            mujoco.mj_step(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0
        if not (
            np.all(np.isfinite(self.data.qpos))
            and np.all(np.isfinite(self.data.qvel))
            and np.all(np.isfinite(self.data.qacc))
        ):
            raise FloatingPointError("MuJoCo state contains NaN or Inf")
        new_warnings = np.asarray(
            [warning.number for warning in self.data.warning], dtype=int
        )
        if np.any(new_warnings > warning_counts):
            raise FloatingPointError("MuJoCo reported a solver warning")
        self.step_count += 1
        self.sequence_id += 1
        return self.kinematic_state()
