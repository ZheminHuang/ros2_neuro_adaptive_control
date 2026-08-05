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

"""Schedule and coordinate-contract tests for the public showcase scenes."""

import numpy as np
import pytest

from neuro_adaptive_control.adapters.mujoco_showcase_benchmarks import (
    compliance_wrench,
    joint_drag_reference,
)
from neuro_adaptive_control.core.so3 import coordinate_transform


def test_compliance_wrench_has_smooth_separate_force_and_moment_events():
    phase, wrench = compliance_wrench(7.75)
    assert phase == "lateral_push"
    np.testing.assert_allclose(wrench, (0.0, 6.0, 0.0, 0.0, 0.0, 0.0))

    phase, wrench = compliance_wrench(10.75)
    assert phase == "twist_moment"
    np.testing.assert_allclose(wrench, (0.0, 0.0, 0.0, 0.0, 0.0, 0.4))

    for stamp in (6.99, 9.0, 12.0, 13.5):
        _, wrench = compliance_wrench(stamp)
        np.testing.assert_allclose(wrench, np.zeros(6), atol=1.0e-14)


def test_physical_wrench_uses_power_consistent_analytical_transform():
    rho = np.array((0.17, -0.09, 0.12))
    physical = np.array((1.0, -2.0, 3.0, 0.4, -0.2, 0.3))
    generalized = coordinate_transform(rho).T @ physical
    analytical_velocity = np.array((0.2, -0.1, 0.3, 0.1, 0.2, -0.15))
    geometric_velocity = coordinate_transform(rho) @ analytical_velocity
    assert generalized @ analytical_velocity == pytest.approx(
        physical @ geometric_velocity
    )


def test_joint_drag_reference_is_closed_and_six_dof_exciting():
    home = np.array((0.1, 0.2, 0.3, 0.0, 0.0, 0.0))
    before = joint_drag_reference(2.0, home)
    after = joint_drag_reference(9.0, home)
    np.testing.assert_array_equal(before.position, home)
    np.testing.assert_array_equal(after.position, home)
    np.testing.assert_array_equal(before.velocity, np.zeros(6))
    np.testing.assert_array_equal(after.velocity, np.zeros(6))

    samples = np.asarray(
        [joint_drag_reference(stamp, home).position for stamp in np.linspace(2.1, 8.9, 80)]
    )
    assert np.all(np.ptp(samples, axis=0) > 0.01)
