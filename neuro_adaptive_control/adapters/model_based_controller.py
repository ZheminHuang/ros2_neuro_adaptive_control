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

"""Nominal and payload-aware model-based benchmark controllers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from neuro_adaptive_control.core.pose_references import PoseReferenceSample
from neuro_adaptive_control.core.so3 import coordinate_transform

from .mujoco_ur5e_adapter import (
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    default_model_path,
    mujoco,
    require_mujoco,
)


def _vector(value: Iterable[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


@dataclass(frozen=True)
class ModelBasedOutput:
    """Joint command and the terms used to construct it."""

    command: np.ndarray
    task_acceleration: np.ndarray
    nominal_inverse_dynamics: np.ndarray
    oracle_payload_compensation: np.ndarray
    torque_saturated: bool
    rate_saturated: bool


class MujocoModelBasedController:
    """Computed-torque baseline using a nominal robot+gripper model."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        position_gain: Iterable[float] = (80.0, 80.0, 90.0, 35.0, 35.0, 35.0),
        velocity_gain: Iterable[float] = (18.0, 18.0, 20.0, 10.0, 10.0, 10.0),
        torque_limits: Iterable[float] = (140.0, 140.0, 140.0, 27.0, 27.0, 27.0),
        torque_rate_limits: Iterable[float] = (
            8000.0,
            8000.0,
            8000.0,
            3000.0,
            3000.0,
            3000.0,
        ),
        oracle_payload_mass_kg: float | None = None,
        gravity_m_s2: float = 9.81,
    ) -> None:
        require_mujoco()
        path = Path(model_path) if model_path is not None else default_model_path()
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.position_gain = _vector(position_gain, 6, "position_gain")
        self.velocity_gain = _vector(velocity_gain, 6, "velocity_gain")
        self.torque_limits = _vector(torque_limits, 6, "torque_limits")
        self.torque_rate_limits = _vector(
            torque_rate_limits,
            6,
            "torque_rate_limits",
        )
        if np.any(self.position_gain < 0.0) or np.any(self.velocity_gain < 0.0):
            raise ValueError("feedback gains must be non-negative")
        if np.any(self.torque_limits <= 0.0) or np.any(
            self.torque_rate_limits <= 0.0
        ):
            raise ValueError("torque limits must be positive")
        if oracle_payload_mass_kg is None:
            self.oracle_payload_mass_kg = None
        else:
            mass = float(oracle_payload_mass_kg)
            if not np.isfinite(mass) or mass <= 0.0:
                raise ValueError("oracle_payload_mass_kg must be positive")
            self.oracle_payload_mass_kg = mass
        self.gravity_m_s2 = float(gravity_m_s2)
        if not np.isfinite(self.gravity_m_s2) or self.gravity_m_s2 <= 0.0:
            raise ValueError("gravity_m_s2 must be finite and positive")
        joint_names = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES
        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in joint_names
            ],
            dtype=int,
        )
        self._qpos_addresses = self.model.jnt_qposadr[joint_ids].copy()
        self._dof_addresses = self.model.jnt_dofadr[joint_ids].copy()
        self._arm_dof = self._dof_addresses[:6]
        self._last_command = np.zeros(6)

    def reset(self) -> None:
        """Clear model data and torque-rate history."""
        mujoco.mj_resetData(self.model, self.data)
        self._last_command.fill(0.0)

    def command(
        self,
        *,
        all_joint_position: Iterable[float],
        all_joint_velocity: Iterable[float],
        actual_pose: Iterable[float],
        actual_pose_velocity: Iterable[float],
        reference: PoseReferenceSample,
        rotation_vector: Iterable[float],
        geometric_jacobian: np.ndarray,
        tcp_position: Iterable[float],
        payload_position: Iterable[float],
        payload_acquired: bool,
        dt: float,
    ) -> ModelBasedOutput:
        """Compute nominal inverse dynamics plus optional oracle payload term."""
        joint_position = _vector(all_joint_position, 14, "all_joint_position")
        joint_velocity = _vector(all_joint_velocity, 14, "all_joint_velocity")
        pose = _vector(actual_pose, 6, "actual_pose")
        pose_velocity = _vector(
            actual_pose_velocity,
            6,
            "actual_pose_velocity",
        )
        rho = _vector(rotation_vector, 3, "rotation_vector")
        jacobian = np.asarray(geometric_jacobian, dtype=float)
        if jacobian.shape != (6, 6) or not np.all(np.isfinite(jacobian)):
            raise ValueError("geometric_jacobian must be a finite 6-by-6 matrix")
        self.data.qpos[self._qpos_addresses] = joint_position
        self.data.qvel[self._dof_addresses] = joint_velocity
        mujoco.mj_forward(self.model, self.data)
        analytical_jacobian = np.linalg.solve(
            coordinate_transform(rho),
            jacobian,
        )
        desired_acceleration = (
            reference.acceleration
            + self.velocity_gain * (reference.velocity - pose_velocity)
            + self.position_gain * (reference.position - pose)
        )
        joint_acceleration = np.linalg.solve(
            analytical_jacobian,
            desired_acceleration,
        )
        mass_matrix = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, mass_matrix, self.data.qM)
        nominal = (
            mass_matrix[np.ix_(self._arm_dof, self._arm_dof)]
            @ joint_acceleration
            + self.data.qfrc_bias[self._arm_dof]
        )
        oracle = np.zeros(6)
        if payload_acquired and self.oracle_payload_mass_kg is not None:
            tcp = _vector(tcp_position, 3, "tcp_position")
            payload = _vector(payload_position, 3, "payload_position")
            support_force = np.array(
                (0.0, 0.0, self.oracle_payload_mass_kg * self.gravity_m_s2)
            )
            support_moment = np.cross(payload - tcp, support_force)
            oracle = jacobian[:3].T @ support_force
            oracle += jacobian[3:].T @ support_moment
        command, torque_saturated, rate_saturated = self._limit(
            nominal + oracle, dt
        )
        return ModelBasedOutput(
            command=command,
            task_acceleration=desired_acceleration.copy(),
            nominal_inverse_dynamics=nominal.copy(),
            oracle_payload_compensation=oracle.copy(),
            torque_saturated=torque_saturated,
            rate_saturated=rate_saturated,
        )

    def _limit(
        self, raw: np.ndarray, dt: float
    ) -> tuple[np.ndarray, bool, bool]:
        if not np.all(np.isfinite(raw)):
            raise FloatingPointError("model-based torque contains NaN or Inf")
        step = float(dt)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be finite and positive")
        delta = raw - self._last_command
        maximum_delta = self.torque_rate_limits * step
        limited_delta = np.clip(
            delta,
            -maximum_delta,
            maximum_delta,
        )
        rate_limited = self._last_command + limited_delta
        command = np.clip(rate_limited, -self.torque_limits, self.torque_limits)
        rate_saturated = not np.array_equal(limited_delta, delta)
        torque_saturated = not np.array_equal(command, rate_limited)
        self._last_command = command.copy()
        return command, torque_saturated, rate_saturated
