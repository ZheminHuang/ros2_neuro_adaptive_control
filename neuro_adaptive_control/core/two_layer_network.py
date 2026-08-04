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

"""Two-adaptive-layer neural approximation from the six-DoF contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _positive_diagonal(
    value: float | Iterable[float],
    size: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(size, float(array))
    if array.shape != (size,):
        raise ValueError(f"{name} must be scalar or shape ({size},).")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive.")
    return array.copy()


@dataclass(frozen=True)
class TwoLayerWeights:
    """Serializable deterministic checkpoint for both adaptive layers."""

    hidden: np.ndarray
    output: np.ndarray


class TwoLayerAdaptiveNetwork:
    """Approximate ``G(z) = W.T tanh(V.T z)`` and adapt both ``V`` and ``W``."""

    def __init__(
        self,
        input_dim: int = 42,
        hidden_dim: int = 32,
        output_dim: int = 6,
        *,
        hidden_learning_rate: float | Iterable[float] = 0.2,
        output_learning_rate: float | Iterable[float] = 2.0,
        leakage: float = 0.01,
        hidden_weight_limit: float = 40.0,
        output_weight_limit: float = 80.0,
        input_scale: float | Iterable[float] = 1.0,
        input_clip: float = 5.0,
        initial_hidden_scale: float = 0.05,
        seed: int = 17,
        adaptation_enabled: bool = True,
    ) -> None:
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("network dimensions must be positive.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.hidden_learning_rate = _positive_diagonal(
            hidden_learning_rate, self.input_dim, "hidden_learning_rate"
        )
        self.output_learning_rate = _positive_diagonal(
            output_learning_rate, self.hidden_dim, "output_learning_rate"
        )
        for name, value, allow_zero in (
            ("leakage", leakage, True),
            ("hidden_weight_limit", hidden_weight_limit, False),
            ("output_weight_limit", output_weight_limit, False),
            ("input_clip", input_clip, False),
            ("initial_hidden_scale", initial_hidden_scale, False),
        ):
            numeric = float(value)
            if not np.isfinite(numeric) or numeric < 0.0 or (not allow_zero and numeric == 0.0):
                raise ValueError(f"{name} has an invalid value: {value}.")
            setattr(self, name, numeric)
        scale = np.asarray(input_scale, dtype=float)
        if scale.ndim == 0:
            scale = np.full(self.input_dim, float(scale))
        if scale.shape != (self.input_dim,):
            raise ValueError(
                f"input_scale must be scalar or shape ({self.input_dim},)."
            )
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("input_scale must be finite and strictly positive.")
        self.input_scale = scale.copy()
        self.seed = int(seed)
        self.adaptation_enabled = bool(adaptation_enabled)
        generator = np.random.default_rng(self.seed)
        self._initial_hidden = generator.uniform(
            -self.initial_hidden_scale,
            self.initial_hidden_scale,
            size=(self.input_dim, self.hidden_dim),
        )
        self.hidden_weights = self._initial_hidden.copy()
        self.output_weights = np.zeros((self.hidden_dim, self.output_dim))

    @property
    def combined_weight_norm(self) -> float:
        """Return the Frobenius norm of ``blkdiag(V_hat, W_hat)``."""
        return float(
            np.sqrt(
                np.linalg.norm(self.hidden_weights, ord="fro") ** 2
                + np.linalg.norm(self.output_weights, ord="fro") ** 2
            )
        )

    def reset(self) -> None:
        """Restore deterministic nondegenerate hidden weights and zero output."""
        self.hidden_weights = self._initial_hidden.copy()
        self.output_weights.fill(0.0)

    def checkpoint(self) -> TwoLayerWeights:
        """Copy both adaptive layers for a frozen-controller comparison."""
        return TwoLayerWeights(
            self.hidden_weights.copy(),
            self.output_weights.copy(),
        )

    def restore(self, checkpoint: TwoLayerWeights) -> None:
        """Restore a validated two-layer checkpoint."""
        hidden = np.asarray(checkpoint.hidden, dtype=float)
        output = np.asarray(checkpoint.output, dtype=float)
        if hidden.shape != (self.input_dim, self.hidden_dim):
            raise ValueError("checkpoint hidden weights have an invalid shape.")
        if output.shape != (self.hidden_dim, self.output_dim):
            raise ValueError("checkpoint output weights have an invalid shape.")
        if not np.all(np.isfinite(hidden)) or not np.all(np.isfinite(output)):
            raise ValueError("checkpoint weights must contain only finite values.")
        self.hidden_weights = hidden.copy()
        self.output_weights = output.copy()

    def _normalized_input(self, features: Iterable[float]) -> np.ndarray:
        vector = np.asarray(features, dtype=float)
        if vector.shape != (self.input_dim,):
            raise ValueError(
                f"features must have shape ({self.input_dim},), got {vector.shape}."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("features must contain only finite values.")
        return np.clip(
            vector / self.input_scale,
            -self.input_clip,
            self.input_clip,
        )

    def activation(self, features: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return normalized input, tanh activation, and its derivative."""
        vector = self._normalized_input(features)
        preactivation = self.hidden_weights.T @ vector
        sigma = np.tanh(preactivation)
        derivative = 1.0 - sigma * sigma
        return vector, sigma, derivative

    def forward(self, features: Iterable[float]) -> np.ndarray:
        """Return ``W_hat.T sigma(V_hat.T z)``."""
        _, sigma, _ = self.activation(features)
        estimate = self.output_weights.T @ sigma
        if not np.all(np.isfinite(estimate)):
            raise FloatingPointError("two-layer network produced NaN or Inf.")
        return estimate

    @staticmethod
    def _project(matrix: np.ndarray, limit: float, name: str) -> np.ndarray:
        norm = float(np.linalg.norm(matrix, ord="fro"))
        if not np.isfinite(norm):
            raise FloatingPointError(f"{name} update produced NaN or Inf.")
        if norm > limit:
            matrix *= limit / norm
        return matrix

    def update(
        self,
        features: Iterable[float],
        sliding_error: Iterable[float],
        dt: float,
    ) -> None:
        """Apply explicit-Euler discretization of the published V/W laws."""
        step = float(dt)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be finite and strictly positive.")
        error = np.asarray(sliding_error, dtype=float)
        if error.shape != (self.output_dim,):
            raise ValueError(
                "sliding_error must have shape "
                f"({self.output_dim},), got {error.shape}."
            )
        if not np.all(np.isfinite(error)):
            raise ValueError("sliding_error must contain only finite values.")
        vector, sigma, derivative = self.activation(features)
        if not self.adaptation_enabled:
            return
        error_norm = float(np.linalg.norm(error))
        preactivation = self.hidden_weights.T @ vector
        corrected_activation = sigma - derivative * preactivation
        output_derivative = self.output_learning_rate[:, None] * (
            np.outer(corrected_activation, error)
            - self.leakage * error_norm * self.output_weights
        )
        hidden_error = derivative * (self.output_weights @ error)
        hidden_derivative = self.hidden_learning_rate[:, None] * (
            np.outer(vector, hidden_error)
            - self.leakage * error_norm * self.hidden_weights
        )
        self.output_weights = self._project(
            self.output_weights + step * output_derivative,
            self.output_weight_limit,
            "output-weight",
        )
        self.hidden_weights = self._project(
            self.hidden_weights + step * hidden_derivative,
            self.hidden_weight_limit,
            "hidden-weight",
        )
