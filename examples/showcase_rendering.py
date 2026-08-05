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

"""Shared antialiased metric-panel rendering for the three showcase scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


NAC_COLOR = (255, 92, 92)
NOMINAL_COLOR = (83, 170, 255)
LOWER_COLOR = (255, 92, 92)
HIGHER_COLOR = (83, 170, 255)
IMPEDANCE_COLOR = (86, 210, 143)
EVENT_COLOR = (198, 105, 224)
WRENCH_COLOR = (255, 162, 45)


@dataclass(frozen=True)
class Trace:
    """One labeled scalar history in a metric card."""

    label: str
    values: np.ndarray
    color: tuple[int, int, int]
    dashed: bool = False


def nice_upper_limit(*arrays: np.ndarray, floor: float = 1.0) -> float:
    """Return a stable 1/2/5-scaled positive chart limit."""
    maximum = max(float(np.max(np.asarray(array))) for array in arrays)
    maximum = max(maximum * 1.08, float(floor))
    exponent = np.floor(np.log10(maximum))
    scale = 10.0**exponent
    normalized = maximum / scale
    if normalized <= 1.0:
        step = 1.0
    elif normalized <= 2.0:
        step = 2.0
    elif normalized <= 5.0:
        step = 5.0
    else:
        step = 10.0
    return float(step * scale)


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _dashed_line(draw, points, color, width: int) -> None:
    for start in range(0, len(points) - 1, 4):
        draw.line(points[start:start + 3], fill=color, width=width)


def _draw_card(
    draw,
    *,
    top: int,
    title: str,
    time: np.ndarray,
    current_index: int,
    first_index: int,
    traces: Sequence[Trace],
    upper_limit: float,
    events: Sequence[tuple[float, tuple[int, int, int]]],
    scale: int,
) -> None:
    left = 8 * scale
    width = 294 * scale
    height = 132 * scale
    card_top = top * scale
    draw.rounded_rectangle(
        (left, card_top, left + width, card_top + height),
        radius=8 * scale,
        fill=(27, 31, 37),
        outline=(77, 86, 98),
        width=1 * scale,
    )
    draw.text(
        (left + 10 * scale, card_top + 7 * scale),
        title,
        fill=(232, 236, 241),
        font=_font(11 * scale, bold=True),
    )
    plot_left = left + 10 * scale
    plot_top = card_top + 29 * scale
    plot_width = width - 20 * scale
    plot_height = height - 39 * scale
    for fraction in (0.0, 0.5, 1.0):
        y = plot_top + int(plot_height * fraction)
        draw.line(
            (plot_left, y, plot_left + plot_width, y),
            fill=(57, 64, 73),
            width=1 * scale,
        )
    start_time = float(time[first_index])
    end_time = float(time[-1])
    for fraction in (0.0, 0.5, 1.0):
        x = plot_left + int(plot_width * fraction)
        draw.line(
            (x, plot_top, x, plot_top + plot_height),
            fill=(48, 54, 62),
            width=1 * scale,
        )
    for stamp, color in events:
        if start_time <= stamp <= end_time:
            x = plot_left + int(
                plot_width * (float(stamp) - start_time) / (end_time - start_time)
            )
            for dash_top in range(plot_top, plot_top + plot_height, 10 * scale):
                draw.line(
                    (
                        x,
                        dash_top,
                        x,
                        min(dash_top + 5 * scale, plot_top + plot_height),
                    ),
                    fill=color,
                    width=2 * scale,
                )
    stop = current_index + 1
    stamps = time[first_index:stop]
    for trace in traces:
        values = np.asarray(trace.values[first_index:stop], dtype=float)
        if len(values) < 2:
            continue
        points = []
        for stamp, value in zip(stamps, values):
            x = plot_left + int(
                plot_width
                * (float(stamp) - start_time)
                / (end_time - start_time)
            )
            normalized = np.clip(float(value) / upper_limit, 0.0, 1.0)
            y = plot_top + plot_height - int(plot_height * normalized)
            points.append((x, y))
        if trace.dashed:
            _dashed_line(draw, points, trace.color, 2 * scale)
        else:
            draw.line(points, fill=trace.color, width=2 * scale)
    draw.text(
        (plot_left + 2 * scale, plot_top + 2 * scale),
        f"{upper_limit:g}",
        fill=(160, 168, 178),
        font=_font(8 * scale),
    )
    draw.text(
        (plot_left + 2 * scale, plot_top + plot_height - 11 * scale),
        "0",
        fill=(160, 168, 178),
        font=_font(8 * scale),
    )


def draw_metric_panel(
    canvas,
    *,
    title: str,
    time: np.ndarray,
    current_index: int,
    first_index: int,
    upper_title: str,
    upper_traces: Sequence[Trace],
    upper_limit: float,
    lower_title: str,
    lower_traces: Sequence[Trace],
    lower_limit: float,
    events: Sequence[tuple[float, tuple[int, int, int]]],
    summary: str,
) -> None:
    """Draw a full-height, two-card antialiased panel at canvas x=970."""
    from PIL import Image, ImageDraw

    scale = 2
    panel = Image.new("RGB", (310 * scale, 390 * scale), (20, 23, 28))
    draw = ImageDraw.Draw(panel)
    draw.text(
        (10 * scale, 10 * scale),
        title,
        fill=(244, 246, 249),
        font=_font(13 * scale, bold=True),
    )
    _draw_card(
        draw,
        top=38,
        title=upper_title,
        time=time,
        current_index=current_index,
        first_index=first_index,
        traces=upper_traces,
        upper_limit=upper_limit,
        events=events,
        scale=scale,
    )
    _draw_card(
        draw,
        top=178,
        title=lower_title,
        time=time,
        current_index=current_index,
        first_index=first_index,
        traces=lower_traces,
        upper_limit=lower_limit,
        events=events,
        scale=scale,
    )
    draw.rounded_rectangle(
        (8 * scale, 318 * scale, 302 * scale, 380 * scale),
        radius=8 * scale,
        fill=(31, 35, 42),
        outline=(77, 86, 98),
        width=1 * scale,
    )
    x = 18 * scale
    for trace in upper_traces:
        draw.line(
            (x, 335 * scale, x + 14 * scale, 335 * scale),
            fill=trace.color,
            width=3 * scale,
        )
        draw.text(
            (x + 18 * scale, 328 * scale),
            trace.label,
            fill=(224, 228, 234),
            font=_font(9 * scale),
        )
        x += (25 + 8 * len(trace.label)) * scale
    draw.text(
        (18 * scale, 352 * scale),
        summary,
        fill=(185, 194, 205),
        font=_font(9 * scale),
    )
    resampling = getattr(Image, "Resampling", Image)
    panel = panel.resize((310, 390), resample=resampling.LANCZOS)
    canvas.paste(panel, (970, 0))
