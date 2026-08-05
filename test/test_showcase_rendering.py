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

"""Unit tests for the shared antialiased showcase metric panel."""

import numpy as np
from PIL import Image

from examples.showcase_rendering import (
    EVENT_COLOR,
    NAC_COLOR,
    Trace,
    draw_metric_panel,
    nice_upper_limit,
)


def test_nice_upper_limit_uses_readable_one_two_five_steps():
    assert nice_upper_limit(np.array((0.0, 0.27)), floor=0.1) == 0.5
    assert nice_upper_limit(np.array((0.0, 4.1)), floor=1.0) == 5.0
    assert nice_upper_limit(np.array((0.0, 41.6)), floor=1.0) == 50.0


def test_metric_panel_fills_third_column_and_confines_event_markers():
    canvas = Image.new("RGB", (1280, 390), (0, 0, 0))
    time = np.linspace(0.0, 10.0, 101)
    values = np.linspace(0.0, 4.0, 101)
    trace = Trace("NAC", values, NAC_COLOR)
    draw_metric_panel(
        canvas,
        title="Test panel",
        time=time,
        current_index=100,
        first_index=0,
        upper_title="Position error [mm]",
        upper_traces=(trace,),
        upper_limit=5.0,
        lower_title="Rotation error [mrad]",
        lower_traces=(trace,),
        lower_limit=5.0,
        events=((4.0, EVENT_COLOR),),
        summary="Compact status",
    )
    panel = np.asarray(canvas)[:, 970:]
    assert np.mean(np.all(panel == 0, axis=2)) < 0.01
    # The gap between cards is not part of either event marker.
    event_x = int(10 + 274 * 0.4)
    assert tuple(panel[173, event_x]) != EVENT_COLOR
