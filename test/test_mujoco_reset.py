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

"""Reset, sequencing, and hard-safety tests for the MuJoCo plant owner."""

from dataclasses import replace

import numpy as np
import pytest

from neuro_adaptive_control.adapters import mujoco_ur5e_adapter as plant_module
from neuro_adaptive_control.adapters.mujoco_simulation import (
    MujocoNACRunner,
    MujocoRunConfig,
    SimulationState,
    build_torque_mapper,
)
from neuro_adaptive_control.adapters.mujoco_ur5e_adapter import (
    MujocoUR5ePlant,
)


mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.skipif(
    not hasattr(mujoco, "MjModel"),
    reason="official MuJoCo Python bindings are not installed",
)


def _one_step_runner() -> MujocoNACRunner:
    return MujocoNACRunner(MujocoRunConfig(duration_sec=0.002, seed=41))


def _rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )


def test_plant_reset_replays_all_dynamic_command_and_rng_state() -> None:
    """Reset q, qdot, object, time, command, gripper, RNG, and sequence."""
    plant = MujocoUR5ePlant(seed=117)
    first = plant.reset()
    expected_qpos = plant.data.qpos.copy()
    expected_object_position = first.object_position.copy()
    expected_object_rotation = first.object_rotation.copy()
    expected_random = plant.rng.standard_normal(12)

    plant.gripper.close(maximum_effort_n=2.5)
    for _ in range(3):
        plant.advance(
            np.full(6, 0.25),
            injected_force_world=(0.5, -0.25, 0.1),
        )
    assert plant.data.time > 0.0
    assert plant.sequence_id == 3
    assert plant.step_count == 3
    assert plant.gripper.target_opening_m == 0.0
    assert np.any(plant.data.ctrl != 0.0)

    repeated = plant.reset()
    repeated_random = plant.rng.standard_normal(12)
    np.testing.assert_array_equal(plant.data.qpos, expected_qpos)
    np.testing.assert_array_equal(plant.data.qvel, np.zeros(plant.model.nv))
    np.testing.assert_array_equal(
        repeated.object_position, expected_object_position
    )
    np.testing.assert_array_equal(
        repeated.object_rotation, expected_object_rotation
    )
    assert plant.data.time == 0.0
    np.testing.assert_array_equal(plant.data.ctrl, np.zeros(plant.model.nu))
    np.testing.assert_array_equal(
        plant.data.qfrc_applied, np.zeros(plant.model.nv)
    )
    np.testing.assert_array_equal(
        plant.data.xfrc_applied, np.zeros((plant.model.nbody, 6))
    )
    assert plant.gripper.target_opening_m == pytest.approx(
        plant.gripper.limits.maximum_opening_m
    )
    assert plant.gripper.maximum_effort_n == pytest.approx(
        plant.gripper.limits.maximum_effort_n
    )
    assert not plant.gripper.stopped
    assert plant.gripper.actuator_control() == 0.0
    assert plant.sequence_id == 0
    assert plant.step_count == 0
    np.testing.assert_array_equal(repeated_random, expected_random)


def test_plant_advances_exactly_four_substeps() -> None:
    """One accepted command advances one sequence and exactly 2 ms."""
    plant = MujocoUR5ePlant()
    before = plant.kinematic_state()
    after = plant.advance(np.zeros(6), substeps=4)

    assert before.sequence_id == 0
    assert before.stamp_sec == 0.0
    assert after.sequence_id == 1
    assert after.stamp_sec == pytest.approx(0.002, abs=1.0e-15)
    assert plant.sequence_id == 1
    assert plant.step_count == 1


@pytest.mark.parametrize("substeps", (0, 1, 2, 3, 5, 8))
def test_plant_rejects_any_substep_count_other_than_four(substeps: int) -> None:
    """An invalid scheduler request must not mutate simulation time."""
    plant = MujocoUR5ePlant()
    with pytest.raises(ValueError, match="exactly four substeps"):
        plant.advance(np.zeros(6), substeps=substeps)
    assert plant.data.time == 0.0
    assert plant.sequence_id == 0
    assert plant.step_count == 0


def test_plant_rejects_stale_or_duplicate_sequence() -> None:
    """A command is accepted only when sequence and completed steps agree."""
    plant = MujocoUR5ePlant()
    plant.sequence_id = 1
    with pytest.raises(RuntimeError, match="stale or duplicate"):
        plant.advance(np.zeros(6))
    assert plant.data.time == 0.0
    assert plant.step_count == 0


@pytest.mark.parametrize(
    ("torque", "injected_force", "injected_torque"),
    (
        (
            (np.nan, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        ((0.0,) * 6, (0.0, np.inf, 0.0), (0.0, 0.0, 0.0)),
        ((0.0,) * 6, (0.0, 0.0, 0.0), (0.0, np.nan, 0.0)),
    ),
)
def test_plant_rejects_nonfinite_torque_or_injected_wrench(
    torque, injected_force, injected_torque
) -> None:
    """NaN/Inf commands must be rejected before any physics step."""
    plant = MujocoUR5ePlant()
    with pytest.raises(ValueError, match="finite vector"):
        plant.advance(
            torque,
            injected_force_world=injected_force,
            injected_torque_world=injected_torque,
        )
    assert plant.data.time == 0.0
    assert plant.sequence_id == 0
    assert plant.step_count == 0


def test_hidden_joint_drag_is_baseline_referenced_and_resettable() -> None:
    """Plant-only damping/friction changes must never accumulate across calls."""
    plant = MujocoUR5ePlant()
    joint_names = ("shoulder_lift_joint", "elbow_joint", "wrist_2_joint")
    addresses = plant._addresses(joint_names, qpos=False)
    baseline_damping = plant.model.dof_damping.copy()
    baseline_friction = plant.model.dof_frictionloss.copy()

    plant.apply_joint_drag(
        joint_names,
        damping_scale=8.0,
        frictionloss_scale=6.0,
    )
    np.testing.assert_allclose(
        plant.model.dof_damping[addresses],
        8.0 * baseline_damping[addresses],
    )
    np.testing.assert_allclose(
        plant.model.dof_frictionloss[addresses],
        6.0 * baseline_friction[addresses],
    )
    plant.apply_joint_drag(
        joint_names,
        damping_scale=8.0,
        frictionloss_scale=6.0,
    )
    np.testing.assert_allclose(
        plant.model.dof_damping[addresses],
        8.0 * baseline_damping[addresses],
    )

    plant.reset()
    np.testing.assert_array_equal(plant.model.dof_damping, baseline_damping)
    np.testing.assert_array_equal(
        plant.model.dof_frictionloss, baseline_friction
    )


@pytest.mark.parametrize(
    ("names", "damping", "friction"),
    (
        ((), 2.0, 2.0),
        (("not_a_joint",), 2.0, 2.0),
        (("elbow_joint",), 0.0, 2.0),
        (("elbow_joint",), 2.0, np.inf),
    ),
)
def test_hidden_joint_drag_rejects_invalid_parameters(
    names, damping, friction
) -> None:
    plant = MujocoUR5ePlant()
    with pytest.raises(ValueError):
        plant.apply_joint_drag(
            names,
            damping_scale=damping,
            frictionloss_scale=friction,
        )


def test_new_mujoco_solver_warning_faults_the_runner(monkeypatch) -> None:
    """A newly incremented MuJoCo warning counter must latch system fault."""
    original_step = plant_module.mujoco.mj_step
    warning_id = int(mujoco.mjtWarning.mjWARN_BADQPOS)

    def step_with_warning(model, data) -> None:
        original_step(model, data)
        data.warning[warning_id].number += 1

    monkeypatch.setattr(plant_module.mujoco, "mj_step", step_with_warning)
    runner = _one_step_runner()
    with pytest.raises(RuntimeError, match="solver warning"):
        runner.run()
    assert runner.state == SimulationState.FAULT
    assert "solver warning" in runner.reason
    assert runner.controller.safety.state.value == "fault"


def test_runner_post_step_guard_faults_final_unsafe_sample(monkeypatch) -> None:
    """Guard the final integrated state before recording clean metrics."""
    runner = _one_step_runner()
    real_advance = runner.plant.advance

    def violating_advance(*args, **kwargs):
        sample = real_advance(*args, **kwargs)
        return replace(sample, tcp_position=np.array((-0.90, 0.40, 0.40)))

    monkeypatch.setattr(runner.plant, "advance", violating_advance)

    with pytest.raises(RuntimeError, match="workspace"):
        runner.run()

    assert runner.state == SimulationState.FAULT
    assert runner.controller.safety.state.value == "fault"


def test_torque_mapper_reports_rate_and_absolute_saturation() -> None:
    """Exercise both downstream rate and actuator magnitude limits."""
    mapper = build_torque_mapper()
    arguments = (
        (10000.0, 0.0, 0.0),
        np.vstack((np.ones(6), np.zeros(6), np.zeros(6))),
        np.zeros((3, 6)),
        np.eye(3),
        np.eye(3),
        np.zeros(3),
        np.zeros(6),
        0.002,
    )

    first = mapper.map_command(*arguments)
    assert first.rate_saturated
    assert not first.torque_saturated
    later = first
    for _ in range(24):
        later = mapper.map_command(*arguments)
    assert later.rate_saturated
    assert later.torque_saturated
    assert np.all(np.abs(later.command) <= mapper.config.torque_limits)


def test_runner_hard_guards_reject_velocity_workspace_orientation_and_contact() -> None:
    """Each physical hard guard must reject a finite but unsafe sample."""
    runner = _one_step_runner()
    sample = runner.plant.kinematic_state()
    contact = runner.plant.contact_summary()
    safe_torque = np.zeros(6)
    runner._check_state(sample, contact, safe_torque)

    fast = replace(sample, arm_velocity=np.array((3.5001, 0, 0, 0, 0, 0)))
    with pytest.raises(RuntimeError, match="joint velocity"):
        runner._check_state(fast, contact, safe_torque)

    outside = replace(sample, tcp_position=np.array((-0.851, 0.4, 0.4)))
    with pytest.raises(RuntimeError, match="workspace"):
        runner._check_state(outside, contact, safe_torque)

    unsafe_rotation = _rotation_z(np.deg2rad(35.1)) @ sample.tcp_rotation
    tilted = replace(sample, tcp_rotation=unsafe_rotation)
    with pytest.raises(RuntimeError, match="orientation error"):
        runner._check_state(tilted, contact, safe_torque)

    excessive_contact = replace(contact, contact_force_norm_n=250.001)
    with pytest.raises(RuntimeError, match="contact force"):
        runner._check_state(sample, excessive_contact, safe_torque)

    excessive_torque = np.array((280.001, 0.0, 0.0, 0.0, 0.0, 0.0))
    with pytest.raises(RuntimeError, match="joint torque"):
        runner._check_state(sample, contact, excessive_torque)

    invalid_joint = sample.all_joint_position.copy()
    invalid_joint[0] = runner.plant.joint_limits[0, 1] + 0.006
    outside_joint_limit = replace(sample, all_joint_position=invalid_joint)
    with pytest.raises(RuntimeError, match="joint-limit"):
        runner._check_state(outside_joint_limit, contact, safe_torque)

    broken_coupling = sample.all_joint_position.copy()
    broken_coupling[10] += 0.021
    actuator_failure = replace(sample, all_joint_position=broken_coupling)
    with pytest.raises(RuntimeError, match="actuator/coupling failure"):
        runner._check_state(actuator_failure, contact, safe_torque)


def test_all_simulation_states_are_observable_and_reachable(monkeypatch) -> None:
    """Observe every lifecycle state through reset, run, stop, and fault."""
    expected_states = {
        SimulationState.START,
        SimulationState.RUNNING,
        SimulationState.STOPPING,
        SimulationState.STOPPED,
        SimulationState.FAULT,
        SimulationState.RESETTING,
    }
    assert set(SimulationState) == expected_states

    runner = _one_step_runner()
    observed = {runner.state}
    original_reset = runner.plant.reset

    def observing_reset():
        observed.add(runner.state)
        return original_reset()

    monkeypatch.setattr(runner.plant, "reset", observing_reset)
    runner.reset()
    observed.add(runner.state)

    original_kinematic_state = runner.plant.kinematic_state

    def observing_kinematic_state():
        observed.add(runner.state)
        return original_kinematic_state()

    monkeypatch.setattr(
        runner.plant, "kinematic_state", observing_kinematic_state
    )
    original_stop = runner.controller.stop

    def observing_stop(reason):
        observed.add(runner.state)
        return original_stop(reason)

    monkeypatch.setattr(runner.controller, "stop", observing_stop)
    runner.run()
    observed.add(runner.state)
    runner._fault("injected lifecycle fault")
    observed.add(runner.state)

    assert observed == expected_states


def test_fault_clears_every_actuator_and_applied_force_buffer() -> None:
    """Fault entry must prevent every old command from affecting physics."""
    runner = _one_step_runner()
    runner.plant.data.ctrl[:] = 1.0
    runner.plant.data.qfrc_applied[:] = 2.0
    runner.plant.data.xfrc_applied[:] = 3.0

    runner._fault("injected safety fault")

    assert runner.state == SimulationState.FAULT
    assert runner.reason == "injected safety fault"
    np.testing.assert_array_equal(
        runner.plant.data.ctrl, np.zeros(runner.plant.model.nu)
    )
    np.testing.assert_array_equal(
        runner.plant.data.qfrc_applied,
        np.zeros(runner.plant.model.nv),
    )
    np.testing.assert_array_equal(
        runner.plant.data.xfrc_applied,
        np.zeros((runner.plant.model.nbody, 6)),
    )


def test_fault_replaces_old_nac_command_with_bounded_joint_damping() -> None:
    """A moving robot must enter damping-only hold, never retain NAC torque."""
    runner = _one_step_runner()
    runner.plant.data.qvel[runner.plant._arm_dof] = (
        0.50,
        -0.40,
        0.30,
        -0.20,
        0.10,
        -0.05,
    )
    runner.plant.data.ctrl[:6] = (20.0, 19.0, 18.0, 7.0, 6.0, 5.0)

    runner._fault("damping transition")

    expected = -runner.mapper.config.joint_damping @ runner.plant.arm_velocity
    np.testing.assert_allclose(runner.plant.data.ctrl[:6], expected)
    assert np.all(
        np.abs(runner.plant.data.ctrl[:6]) <= runner.mapper.config.torque_limits
    )
