# Security policy

## Supported versions

Security fixes are provided for the latest `0.1.x` release. Development snapshots may change without notice.

## Reporting a vulnerability

Use GitHub's private security-advisory reporting for this repository. Do not publish credentials, robot addresses, calibration, participant data, or a working exploit in a public issue. Include the affected version, impact, reproduction steps that do not require physical hardware, and any proposed mitigation.

## Robotics safety boundary

This repository is research software and ships no hardware adapter. Its watchdog, state machine, finite-value checks, and command saturation are software safeguards, not a certified functional-safety system. Do not connect its output directly to a physical robot. A deployment requires independent stops and robot-specific collision, workspace, velocity, force, singularity, frame, timing, and recovery validation.
