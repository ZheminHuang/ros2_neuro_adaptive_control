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

"""Tests for physical held-out payload variations in the MuJoCo plant."""

import numpy as np
import pytest

from neuro_adaptive_control.adapters.mujoco_ur5e_adapter import MujocoUR5ePlant


def test_payload_mass_com_and_inertia_are_physical_model_properties():
    plant = MujocoUR5ePlant(
        payload_mass_kg=0.31,
        payload_com_offset_m=(0.004, -0.003, 0.002),
        payload_inertia_scale=1.2,
    )
    body_id = plant._object_body_id

    assert plant.model.body_mass[body_id] == pytest.approx(0.31)
    np.testing.assert_allclose(
        plant.model.body_ipos[body_id],
        (0.004, -0.003, 0.002),
    )
    expected_nominal_inertia = np.array(
        (0.000148333, 0.000148333, 0.0000833333)
    )
    np.testing.assert_allclose(
        plant.model.body_inertia[body_id],
        expected_nominal_inertia * (0.31 / 0.20) * 1.2,
    )
    assert np.all(np.isfinite(plant.kinematic_state().object_position))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"payload_mass_kg": 0.0},
        {"payload_mass_kg": np.nan},
        {"payload_inertia_scale": -1.0},
        {"payload_com_offset_m": (0.0, np.inf, 0.0)},
    ],
)
def test_invalid_payload_properties_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MujocoUR5ePlant(**kwargs)
