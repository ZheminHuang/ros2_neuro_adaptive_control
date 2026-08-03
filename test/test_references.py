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

"""Tests for analytic Cartesian reference positions and derivatives."""

import numpy as np
import pytest

from neuro_adaptive_control.core.references import (
    CircleReference,
    FigureEightReference,
    FixedPointReference,
    LineReference,
    make_reference,
)


def test_fixed_point_has_zero_derivatives_and_returns_copies():
    reference = FixedPointReference([1.0, -2.0, 3.0])

    first = reference.evaluate(0.0)
    second = reference.evaluate(123.0)

    np.testing.assert_array_equal(first.position, [1.0, -2.0, 3.0])
    np.testing.assert_array_equal(second.position, first.position)
    np.testing.assert_array_equal(first.velocity, np.zeros(3))
    np.testing.assert_array_equal(first.acceleration, np.zeros(3))
    first.position[:] = 99.0
    np.testing.assert_array_equal(
        reference.evaluate(0.0).position, [1.0, -2.0, 3.0]
    )


def test_circle_position_velocity_and_acceleration_are_analytic():
    center = np.array([1.0, 2.0, 3.0])
    radius = 2.0
    frequency = 0.25
    omega = 2.0 * np.pi * frequency
    reference = CircleReference(center, radius, frequency, phase=np.pi / 2.0)

    # At t=1, theta=pi.
    sample = reference.evaluate(1.0)
    expected_position = center + np.array([-radius, 0.0, 0.0])
    expected_velocity = np.array([0.0, -radius * omega, 0.0])
    expected_acceleration = np.array([radius * omega**2, 0.0, 0.0])

    np.testing.assert_allclose(sample.position, expected_position, atol=1e-14)
    np.testing.assert_allclose(sample.velocity, expected_velocity, atol=1e-14)
    np.testing.assert_allclose(
        sample.acceleration, expected_acceleration, atol=1e-14
    )


def test_line_position_velocity_and_acceleration_are_analytic():
    center = np.array([1.0, -2.0, 3.0])
    axis = np.array([0.0, 3.0, 4.0])
    unit_axis = axis / np.linalg.norm(axis)
    length = 4.0
    amplitude = length / 2.0
    frequency = 0.25
    omega = 2.0 * np.pi * frequency
    reference = LineReference(center, length, frequency, axis)

    # At t=1, theta=pi/2.
    sample = reference.evaluate(1.0)

    np.testing.assert_allclose(sample.position, center + amplitude * unit_axis)
    np.testing.assert_allclose(sample.velocity, np.zeros(3), atol=1e-14)
    np.testing.assert_allclose(
        sample.acceleration, -amplitude * omega**2 * unit_axis
    )


def test_figure_eight_position_velocity_and_acceleration_are_analytic():
    center = np.array([0.5, -0.25, 1.0])
    width = 4.0
    height = 2.0
    amplitude_x = width / 2.0
    amplitude_y = height / 2.0
    frequency = 0.25
    omega = 2.0 * np.pi * frequency
    theta = np.pi / 4.0
    reference = FigureEightReference(center, width, height, frequency)

    sample = reference.evaluate(0.5)
    expected_position = center + np.array(
        [amplitude_x * np.sin(theta), amplitude_y * np.sin(2.0 * theta), 0.0]
    )
    expected_velocity = np.array(
        [
            amplitude_x * omega * np.cos(theta),
            2.0 * amplitude_y * omega * np.cos(2.0 * theta),
            0.0,
        ]
    )
    expected_acceleration = np.array(
        [
            -amplitude_x * omega**2 * np.sin(theta),
            -4.0 * amplitude_y * omega**2 * np.sin(2.0 * theta),
            0.0,
        ]
    )

    np.testing.assert_allclose(sample.position, expected_position, atol=1e-14)
    np.testing.assert_allclose(sample.velocity, expected_velocity, atol=1e-14)
    np.testing.assert_allclose(
        sample.acceleration, expected_acceleration, atol=1e-14
    )


@pytest.mark.parametrize(
    "reference",
    [
        FixedPointReference([0.1, -0.2, 0.3]),
        CircleReference(
            [0.1, -0.2, 0.3], radius=0.4, frequency=0.2, phase=0.3
        ),
        LineReference(
            [0.1, -0.2, 0.3], length=0.4, frequency=0.2, axis=[1.0, 2.0, -1.0]
        ),
        FigureEightReference(
            [0.1, -0.2, 0.3], width=0.4, height=0.2, frequency=0.2
        ),
    ],
)
def test_reported_derivatives_match_centered_finite_differences(reference):
    time_sec = 0.73
    epsilon = 1e-5
    before = reference.evaluate(time_sec - epsilon)
    sample = reference.evaluate(time_sec)
    after = reference.evaluate(time_sec + epsilon)

    numerical_velocity = (after.position - before.position) / (2.0 * epsilon)
    numerical_acceleration = (after.velocity - before.velocity) / (
        2.0 * epsilon
    )

    np.testing.assert_allclose(
        sample.velocity, numerical_velocity, rtol=2e-8, atol=2e-10
    )
    np.testing.assert_allclose(
        sample.acceleration, numerical_acceleration, rtol=2e-8, atol=2e-10
    )


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    [
        ("circle", CircleReference),
        ("line", LineReference),
        ("figure8", FigureEightReference),
        (" Figure-8 ", FigureEightReference),
        ("fixed", FixedPointReference),
        ("fixed_point", FixedPointReference),
    ],
)
def test_reference_factory_selects_supported_trajectories(kind, expected_type):
    assert isinstance(make_reference(kind), expected_type)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FixedPointReference([0.0, 0.0]),
        lambda: FixedPointReference([0.0, np.nan, 0.0]),
        lambda: CircleReference([0.0, 0.0, 0.0], 0.0, 1.0),
        lambda: CircleReference([0.0, 0.0, 0.0], np.inf, 1.0),
        lambda: CircleReference([0.0, 0.0, 0.0], 1.0, 0.0),
        lambda: CircleReference([0.0, 0.0, 0.0], 1.0, 1.0, np.nan),
        lambda: LineReference([0.0, 0.0, 0.0], 1.0, 1.0, [0.0, 0.0, 0.0]),
        lambda: LineReference([0.0, 0.0, 0.0], -1.0, 1.0),
        lambda: LineReference([0.0, 0.0, 0.0], 1.0, np.inf),
        lambda: FigureEightReference([0.0, 0.0, 0.0], 0.0, 1.0, 1.0),
        lambda: FigureEightReference([0.0, 0.0, 0.0], 1.0, -1.0, 1.0),
        lambda: FigureEightReference([0.0, 0.0, 0.0], 1.0, 1.0, np.nan),
        lambda: make_reference("helix"),
    ],
)
def test_invalid_trajectory_parameters_and_kind_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    "reference",
    [
        FixedPointReference([0.0, 0.0, 0.0]),
        CircleReference([0.0, 0.0, 0.0], 1.0, 1.0),
        LineReference([0.0, 0.0, 0.0], 1.0, 1.0),
        FigureEightReference([0.0, 0.0, 0.0], 1.0, 1.0, 1.0),
    ],
)
@pytest.mark.parametrize("invalid_time", [-0.01, np.nan, np.inf])
def test_invalid_evaluation_time_is_rejected(reference, invalid_time):
    with pytest.raises(ValueError):
        reference.evaluate(invalid_time)
