#!/usr/bin/env python3
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

"""Run and optionally plot the deterministic NAC/baseline comparison."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuro_adaptive_control.core.simulation import (  # noqa: E402
    ComparisonResult,
    SimulationConfig,
    SimulationResult,
    run_comparison,
)


def _write_csv(path: Path, result: SimulationResult) -> None:
    columns: list[tuple[str, Iterable[float]]] = [("time_sec", result.time)]
    for name, values in (
        ("desired", result.desired),
        ("impedance", result.impedance),
        ("actual", result.actual),
        ("velocity", result.velocity),
        ("command", result.command),
        ("raw_command", result.raw_command),
        ("nn_estimate", result.neural_estimate),
        ("impedance_error", result.tracking_error),
        ("desired_error", result.desired_error),
        ("external_wrench", result.external_wrench),
    ):
        for axis, suffix in enumerate(("x", "y", "z")):
            columns.append((f"{name}_{suffix}", values[:, axis]))
    columns.append(("saturated", result.saturated.astype(int)))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([name for name, _ in columns])
        writer.writerows(zip(*(values for _, values in columns)))


def _plot(path: Path, comparison: ComparisonResult) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = comparison.baseline
    nac = comparison.nac
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axis = axes[0, 0]
    axis.plot(nac.desired[:, 0], nac.desired[:, 1], "k--", label="desired")
    axis.plot(nac.impedance[:, 0], nac.impedance[:, 1], label="impedance")
    axis.plot(nac.actual[:, 0], nac.actual[:, 1], label="NAC actual")
    axis.plot(
        baseline.actual[:, 0], baseline.actual[:, 1], alpha=0.75,
        label="baseline actual"
    )
    axis.set_title("XY trajectory")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    axis.legend()

    axis = axes[0, 1]
    axis.plot(
        baseline.time,
        np.linalg.norm(baseline.tracking_error, axis=1),
        label="baseline",
    )
    axis.plot(
        nac.time,
        np.linalg.norm(nac.tracking_error, axis=1),
        label="NAC",
    )
    axis.set_title("Impedance model-following error")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("||x_m - x|| [m]")
    axis.grid(True, alpha=0.3)
    axis.legend()

    axis = axes[1, 0]
    axis.plot(
        baseline.time,
        np.linalg.norm(baseline.command, axis=1),
        label="baseline",
    )
    axis.plot(nac.time, np.linalg.norm(nac.command, axis=1), label="NAC")
    axis.set_title("Cartesian command norm")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("||f_c|| [N]")
    axis.grid(True, alpha=0.3)
    axis.legend()

    axis = axes[1, 1]
    labels = ["baseline", "NAC"]
    values = [
        baseline.metrics["impedance_tracking_rmse_m"],
        nac.metrics["impedance_tracking_rmse_m"],
    ]
    axis.bar(labels, values, color=["#7f8c8d", "#2e86de"])
    axis.set_title(
        "Impedance RMSE "
        f"({comparison.metrics['impedance_rmse_improvement_percent']:.1f}% change)"
    )
    axis.set_ylabel("RMSE [m]")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        choices=("circle", "line", "figure8", "fixed_point"),
        default="circle",
    )
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--external-wrench", action="store_true")
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()

    config = SimulationConfig(
        trajectory=args.trajectory,
        duration_sec=args.duration,
        dt=args.dt,
        external_wrench_enabled=args.external_wrench,
    )
    comparison = run_comparison(config)
    report = {
        "scenario": {
            "trajectory": config.trajectory,
            "duration_sec": config.duration_sec,
            "fixed_dt_sec": config.dt,
            "external_wrench_enabled": config.external_wrench_enabled,
            "seed": config.seed,
        },
        "baseline": comparison.baseline.metrics,
        "nac": comparison.nac.metrics,
        "comparison": comparison.metrics,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.output_directory is not None:
        output = args.output_directory
        output.mkdir(parents=True, exist_ok=True)
        (output / "comparison_metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_csv(output / "baseline.csv", comparison.baseline)
        _write_csv(output / "nac.csv", comparison.nac)
        _plot(output / "demo_results.png", comparison)
        print(f"Saved deterministic artifacts in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
