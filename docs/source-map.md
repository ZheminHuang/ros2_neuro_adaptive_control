# Source map and provenance

The controller and ROS implementation were written independently from
equations and behavior checks. No source file, figure, table, log,
calibration, hardware detail, or prose was copied from either legacy reference
repository. The separately audited UR5e and Robotiq model assets now vendored
for the v0.2.0 candidate come from a pinned MuJoCo Menagerie revision, not
from either legacy repository.

## Mathematical sources checked

| Source | Revision / fingerprint | Material checked | Distribution decision |
|---|---|---|---|
| Local pHRI manuscript | SHA-256 `5e1f4d8fc74939bbbde149d342ea26463193be848ab2ce8f19efc48276476f74` | Cartesian dynamics; impedance model; error, command, robust and adaptation signs | Not distributed: no open license was present and the file contains private manuscript metadata |
| `ZheminHuang/ur5e_nac_mujoco` | commit `38d6c29ec7c8cd48c9f5117e380ab2eaecc28a2c` | numerical impedance integration, NN implementation, reference feedforward, command composition | Read-only behavior reference; controller files had no repository-level license, so no code was copied |
| Existing ROS2 real wrapper | commit `e6cbcbe381fb94931415a7af1c235a4f5f8c38df` | wall-clock timing, translational force-mode boundary, wrench processing, stop/watchdog gaps | Read-only engineering comparison; no code, configuration, logs, or hardware details were copied |

The detailed equation-by-equation resolution is in
[`math_contract.md`](math_contract.md).

## Important discrepancies found

- The manuscript uses a two-layer sigmoid NN, while the MuJoCo implementation
  uses a two-layer tanh NN. Neither defines an RBF network.
- The manuscript prints a decreasing logistic expression but a positive
  logistic derivative; those two signs are inconsistent.
- The manuscript does not specify inner-loop numerical integration.
- The MuJoCo controller initializes weights nondeterministically and updates
  both NN layers. The real wrapper's reference reset does not clear NN weights.
- The existing real path sends only translational force even though internal
  arrays include an unvalidated orientation approximation.

Accordingly, Gaussian bases, center generation, widths, normalization,
output-weight-only adaptation, projection, saturation, formal lifecycle, and
deterministic reset are explicit new project contracts.

## License boundary

All first-party code in this repository is newly authored for this project and
released under Apache-2.0. Apache-2.0 does not relicense the manuscript,
either legacy implementation, ROS/Ubuntu dependencies, or vendored model
assets. The pinned UR5e assets retain BSD-3-Clause terms and the pinned
Robotiq assets retain BSD-2-Clause terms; their exact revision, transformations,
hashes, and fidelity limits are recorded in
[`ur5e_robotiq_model_provenance.md`](ur5e_robotiq_model_provenance.md),
[`../mujoco/SOURCE.yml`](../mujoco/SOURCE.yml), and
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

The release dependency and first-party license review is recorded in
[`license-audit.md`](license-audit.md).

The v0.1 plot and v0.2 benchmark plots are generated from repository-owned
simulation histories. The v0.2 candidate also includes 28 audited Menagerie
mesh files and two direct GUI captures of this bundled model. No participant
data, calibration, private log, video, manuscript figure, or other external
media is included.
