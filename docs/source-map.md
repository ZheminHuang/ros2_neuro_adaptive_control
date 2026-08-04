# Source map and provenance

The controller and ROS implementation were written independently from
equations and behavior checks. No source file, figure, table, log,
calibration, hardware detail, or prose was copied from either legacy reference
repository. The separately audited UR5e and Robotiq model assets vendored for
the v0.2.0 candidate and reused by v0.3 come from a pinned MuJoCo Menagerie
revision, not from either legacy repository.

## Mathematical sources checked

| Source | Revision / fingerprint | Material checked | Distribution decision |
|---|---|---|---|
| Local pHRI manuscript | SHA-256 `5e1f4d8fc74939bbbde149d342ea26463193be848ab2ce8f19efc48276476f74` | Cartesian dynamics; impedance model; error, command, robust and adaptation signs | Not distributed: no open license was present and the file contains private manuscript metadata |
| User-supplied 6D rotation-vector derivation | SHA-256 `4f02d6c7b2f89671e4059922dba02acbf4ab99afbc6a62ef556ab669327195b3` | (SO(3)) chart, left Jacobian, analytical/geometric wrench duality, 6D impedance, 42D regressor, two-layer V/W laws, robust term, torque realization, and UUB assumptions | Not distributed; equations were checked and independently implemented in NumPy |
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

Accordingly, the retained 3D Gaussian bases, center generation, widths,
output-weight-only adaptation, projection, saturation, formal lifecycle, and
deterministic reset remain explicit project contracts. The v0.3 path instead
uses the supplied 6D rotation-vector/two-layer equations with separately
documented normalization, discretization, projection, chart guards, and
sampled/contact proof boundary.

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

The v0.1, v0.2, and v0.3 benchmark images are generated from repository-owned
simulation histories. The animated v0.3 comparison is deterministic offscreen
MuJoCo rendering of those histories, not participant video or a manually
spliced screen recording. The repository includes 28 audited Menagerie mesh
files. No participant data, calibration, private log, manuscript figure, or
other external media is included.
