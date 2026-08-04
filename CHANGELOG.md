# Changelog

All notable changes are documented here. The project follows semantic versioning.

## [Unreleased] - v0.3.0 candidate

### Added

- Rotation-vector XYZ/RX/RY/RZ impedance coordinates with guarded SO(3)
  Exp/Log, left Jacobian, inverse Jacobian, and analytical/geometric
  power-conjugate transformations.
- A 42D two-layer tanh network with online hidden/output V/W adaptation,
  leakage, projection, deterministic reset, and exact payload-time checkpoint.
- Six-dimensional NAC generalized-force output and the running torque contract
  `J_g.T @ E^-T @ u`, without independent orientation PD or running joint
  damping; bounded joint damping is stopping/fault-only.
- Physical held-out payload variants that change MuJoCo mass, COM, and inertia
  before reset while remaining hidden from the NAC.
- Matched adaptive, payload-time-frozen, nominal model-based, and payload-aware
  oracle controllers on one grasp/lift/loaded-trajectory scenario.
- A native MuJoCo benchmark launch, aggregate acceptance gates, machine-readable
  metrics, synchronized comparison GIF, and trajectory/error/NN result plot.

### Candidate evidence

- Three of three adaptive and frozen held-out payload trials completed without
  an added adaptive safety failure.
- Median loaded position RMSE was 6.497 mm adaptive versus 13.779 mm frozen
  (52.8% lower).
- Median loaded rotation-vector RMSE was 1.368 mrad adaptive versus 4.916 mrad
  frozen (72.2% lower).
- The 310 g nominal baseline remains visible alongside a payload-aware oracle;
  metric-specific results are reported without a universal superiority claim.

### Release gate

- This section is intentionally undated and unreleased. Only the feature
  branch may be pushed until the user reviews the final GIF and metrics.
- `CITATION.cff` and the public release tag remain at v0.1.0. Do not merge,
  tag, or create v0.3.0 before explicit user approval and final CI/audit.

## v0.2.0 candidate (superseded, unreleased)

### Added

- Single-owner MuJoCo 3.9.0 simulation of a six-joint UR5e, articulated
  eight-joint Robotiq 2F-85, table, and dynamic grasp object.
- A 27D robot regressor using six arm positions, six arm velocities, and the
  existing impedance/error features while retaining a 3D translational NAC
  output.
- Frame-explicit Cartesian-force-to-joint-torque mapping through the
  translational Jacobian, plus independent non-adaptive orientation hold,
  joint damping, torque-rate limits, and torque limits.
- Standard `control_msgs/action/GripperCommand` bridge with measured opening,
  effort, contact, reached-goal, stalled, cancel, timeout, and reset behavior.
- Contact-only environment-on-robot wrench, separate raw wrist cut-wrench,
  injected and virtual-FT external-force modes, contact diagnostics, and
  display markers.
- Display-only RViz launch for trajectory tracking and an automated dynamic
  grasp/lift/hold launch, with an optional native MuJoCo passive viewer and
  headless operation.
- Deterministic four-reference tracking benchmark, matched frozen-weight
  circle baseline, grasp benchmark, machine-readable JSON, generated plots,
  source/model/history hashes, and candidate release audit.
- Vendored, pinned MuJoCo Menagerie model inputs and meshes with manifests,
  provenance, fidelity limitations, and retained BSD license notices.

### Changed

- Package candidate metadata is 0.2.0 while the latest published release and
  citation metadata remain v0.1.0 until publication.
- The MuJoCo loop uses a fixed 0.002 s controller period and exactly four
  0.0005 s `implicitfast` physics substeps per accepted command.
- Downstream torque or torque-rate saturation restores the pre-step RBF
  weights; Cartesian force saturation remains a separate core limiter.
- README now distinguishes MuJoCo dynamics ownership from RViz visualization,
  3D adaptive translation from fixed orientation hold, and articulated model
  completeness from manufacturer-calibrated digital-twin fidelity.

### Candidate evidence

- Matched 8 s circle impedance RMSE: 0.0328492 m frozen weights versus
  0.00385451 m adaptive NAC (88.266% lower); all four adaptive references
  stopped without fault.
- The 11 s grasp run lifted the object 0.0776457 m, held for 2.0 s with
  0.000396843 m drop, used at most 2.0 N gripper effort, and stopped without
  unexpected contact, solver warning, torque saturation, or fault.
- Results are deterministic bundled-simulation evidence only, not hardware,
  general-performance, formal-stability, or hard-real-time claims.

### Release gate

- This section is intentionally undated and unreleased. Do not create the
  v0.2.0 tag or public release until the user visually accepts both RViz demos
  and the final Humble build, test, privacy, license, provenance, and
  clean-tree audits pass.

## [0.1.0] - 2026-08-03

### Added

- ROS-independent NumPy implementation of a 3D translational impedance model.
- Model-free neuro-adaptive wrench controller with fixed Gaussian RBF bases, online output-weight adaptation, leakage, projection, feedback, and robust terms.
- Circle, line, figure-eight, and fixed-point analytic references.
- Five-state lifecycle, command saturation, watchdog, finite-value validation, fault latching, and deterministic reset.
- Standard-message ROS 2 controller wrapper and deterministic unknown-dynamics plant.
- Optional external wrench, frozen-adaptation baseline, telemetry, diagnostics, metrics, and generated result plot.
- Unit, integration, comparison, lint, licensing, and soft 500 Hz core performance tests.
- Mathematical contract, architecture, provenance, contribution, security, citation, and release documentation.

### Excluded by design

- Physical-robot and UR5e `force_mode` adapters.
- Validated 6D orientation control.
- ADP impedance optimization, human identification, and participant/experiment media workflows.
