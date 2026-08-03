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

"""Deterministic Gaussian RBF dynamics approximator."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


class RBFNetwork:
    """
    Implement a fixed-basis RBF network with adaptive output weights.

    Centers and widths are fixed. Only ``weights`` in
    ``G_hat = weights.T @ phi(z)`` are adapted online. This is intentionally
    different from the two-adaptive-layer network used by the source paper.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 3,
        num_basis: int = 25,
        *,
        centers: Optional[np.ndarray] = None,
        widths: float | Iterable[float] = 2.5,
        input_scale: float | Iterable[float] = 1.0,
        feature_clip: float = 3.0,
        learning_rate: float = 4.0,
        leakage: float = 0.01,
        weight_limit: float = 80.0,
        seed: int = 7,
        adaptation_enabled: bool = True,
    ) -> None:
        if input_dim <= 0 or output_dim <= 0 or num_basis <= 0:
            raise ValueError("input_dim, output_dim, and num_basis must be positive.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_basis = int(num_basis)

        if centers is None:
            rng = np.random.default_rng(seed)
            center_array = rng.uniform(
                -0.75, 0.75, size=(self.num_basis, self.input_dim)
            )
            center_array[0, :] = 0.0
        else:
            center_array = np.asarray(centers, dtype=float)
        if center_array.shape != (self.num_basis, self.input_dim):
            raise ValueError(
                "centers must have shape "
                f"({self.num_basis}, {self.input_dim}), got {center_array.shape}."
            )
        if not np.all(np.isfinite(center_array)):
            raise ValueError("centers must contain only finite values.")
        self.centers = center_array.copy()

        width_array = np.asarray(widths, dtype=float)
        if width_array.ndim == 0:
            width_array = np.full(self.num_basis, float(width_array))
        if width_array.shape != (self.num_basis,):
            raise ValueError(
                f"widths must be scalar or shape ({self.num_basis},)."
            )
        if not np.all(np.isfinite(width_array)) or np.any(width_array <= 0.0):
            raise ValueError("widths must be finite and strictly positive.")
        self.widths = width_array.copy()

        scale_array = np.asarray(input_scale, dtype=float)
        if scale_array.ndim == 0:
            scale_array = np.full(self.input_dim, float(scale_array))
        if scale_array.shape != (self.input_dim,):
            raise ValueError(
                f"input_scale must be scalar or shape ({self.input_dim},)."
            )
        if not np.all(np.isfinite(scale_array)) or np.any(scale_array <= 0.0):
            raise ValueError("input_scale must be finite and strictly positive.")
        self.input_scale = scale_array.copy()

        for name, value, lower_bound in (
            ("feature_clip", feature_clip, 0.0),
            ("learning_rate", learning_rate, 0.0),
            ("leakage", leakage, 0.0),
            ("weight_limit", weight_limit, 0.0),
        ):
            if not np.isfinite(value) or value <= lower_bound:
                if name in {"learning_rate", "leakage"} and value == 0.0:
                    continue
                raise ValueError(f"{name} has an invalid value: {value}.")
        self.feature_clip = float(feature_clip)
        self.learning_rate = float(learning_rate)
        self.leakage = float(leakage)
        self.weight_limit = float(weight_limit)
        self.adaptation_enabled = bool(adaptation_enabled)
        self.weights = np.zeros((self.num_basis, self.output_dim), dtype=float)

    @property
    def weight_norm(self) -> float:
        """Return the Frobenius norm used by the robust term."""
        return float(np.linalg.norm(self.weights, ord="fro"))

    def reset(self) -> None:
        """Restore deterministic zero output weights."""
        self.weights.fill(0.0)

    def _features(self, features: Iterable[float]) -> np.ndarray:
        array = np.asarray(features, dtype=float)
        if array.shape != (self.input_dim,):
            raise ValueError(
                f"features must have shape ({self.input_dim},), got {array.shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("features must contain only finite values.")
        return np.clip(
            array / self.input_scale,
            -self.feature_clip,
            self.feature_clip,
        )

    def activations(self, features: Iterable[float]) -> np.ndarray:
        """Evaluate ``phi_i = exp(-||z-c_i||^2 / (2 b_i^2))``."""
        normalized = self._features(features)
        squared_distance = np.sum((self.centers - normalized) ** 2, axis=1)
        phi = np.exp(-0.5 * squared_distance / (self.widths**2))
        if not np.all(np.isfinite(phi)):
            raise FloatingPointError("RBF activation produced NaN or Inf.")
        return phi

    def forward(self, features: Iterable[float]) -> np.ndarray:
        """Return the current approximation of lumped Cartesian dynamics."""
        estimate = self.weights.T @ self.activations(features)
        if not np.all(np.isfinite(estimate)):
            raise FloatingPointError("RBF output produced NaN or Inf.")
        return estimate

    def update(
        self,
        features: Iterable[float],
        sliding_error: Iterable[float],
        dt: float,
    ) -> None:
        """Apply explicit Euler output-weight adaptation with projection."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive.")
        error = np.asarray(sliding_error, dtype=float)
        if error.shape != (self.output_dim,):
            raise ValueError(
                "sliding_error must have shape "
                f"({self.output_dim},), got {error.shape}."
            )
        if not np.all(np.isfinite(error)):
            raise ValueError("sliding_error must contain only finite values.")
        if not self.adaptation_enabled:
            return

        phi = self.activations(features)
        error_norm = float(np.linalg.norm(error))
        derivative = self.learning_rate * (
            np.outer(phi, error) - self.leakage * error_norm * self.weights
        )
        candidate = self.weights + dt * derivative
        candidate_norm = float(np.linalg.norm(candidate, ord="fro"))
        if not np.isfinite(candidate_norm):
            raise FloatingPointError("RBF weight update produced NaN or Inf.")
        if candidate_norm > self.weight_limit:
            candidate *= self.weight_limit / candidate_norm
        self.weights = candidate
