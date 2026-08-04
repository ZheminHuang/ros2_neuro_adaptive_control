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

"""ROS-independent NumPy controller core."""

from .impedance_model import (
    CartesianImpedanceModel,
    ImpedanceParameters,
    ImpedanceState,
)
from .neuro_adaptive_controller import (
    ControllerOutput,
    NACParameters,
    NeuroAdaptiveController,
)
from .rbf_network import RBFNetwork
from .pose_impedance_model import (
    PoseImpedanceModel,
    PoseImpedanceParameters,
    PoseImpedanceState,
)
from .pose_neuro_adaptive_controller import (
    POSE_NN_INPUT_DIM,
    PoseControllerOutput,
    PoseNACParameters,
    PoseNeuroAdaptiveController,
    build_pose_nn_features,
)
from .pose_references import (
    PoseReferenceSample,
    fixed_pose_reference,
    smooth_payload_reference,
)
from .references import ReferenceSample, make_reference
from .safety import ControllerState, SafetyConfig, SafetySupervisor
from .simulation import (
    ComparisonResult,
    SimulationConfig,
    SimulationResult,
    run_comparison,
    run_simulation,
)
from .so3 import (
    coordinate_transform,
    exp,
    hat,
    left_jacobian,
    left_jacobian_inverse,
    log,
    validate_rotation_matrix,
    vee,
)
from .two_layer_network import TwoLayerAdaptiveNetwork, TwoLayerWeights

__all__ = [
    "CartesianImpedanceModel",
    "ComparisonResult",
    "ControllerOutput",
    "ControllerState",
    "ImpedanceParameters",
    "ImpedanceState",
    "NACParameters",
    "NeuroAdaptiveController",
    "POSE_NN_INPUT_DIM",
    "PoseControllerOutput",
    "PoseImpedanceModel",
    "PoseImpedanceParameters",
    "PoseImpedanceState",
    "PoseNACParameters",
    "PoseNeuroAdaptiveController",
    "PoseReferenceSample",
    "RBFNetwork",
    "ReferenceSample",
    "SafetyConfig",
    "SafetySupervisor",
    "SimulationConfig",
    "SimulationResult",
    "TwoLayerAdaptiveNetwork",
    "TwoLayerWeights",
    "build_pose_nn_features",
    "coordinate_transform",
    "exp",
    "fixed_pose_reference",
    "hat",
    "left_jacobian",
    "left_jacobian_inverse",
    "log",
    "make_reference",
    "run_comparison",
    "run_simulation",
    "smooth_payload_reference",
    "validate_rotation_matrix",
    "vee",
]
