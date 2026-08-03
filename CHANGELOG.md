# Changelog

All notable changes are documented here. The project follows semantic versioning.

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
