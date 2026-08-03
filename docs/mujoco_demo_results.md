# MuJoCo v0.2.0-candidate results

This page records the committed, machine-readable evidence for the
full-dynamics UR5e + Robotiq candidate. Values below are transcribed from the
linked JSON files; terminal output and RViz appearance are not used as metric
sources.

## Environment and traceability

Both reports identify:

| Field | Value |
|---|---|
| package candidate | 0.2.0 |
| Python environment | ROS 2 Humble system Python |
| Python | 3.10.12 |
| NumPy | 1.24.4 |
| MuJoCo | 3.9.0 |
| composite model | `mujoco/ur5e_robotiq_2f85.xml` |
| model SHA-256 | `24c8d1318feb78e0319d24bcbb6f768820b85026c64c86b19a9b2d2decab8ce6` |
| model-manifest SHA-256 | `00818775e8bc4c74488c36cc9e3fa904e5cd767c58805b0391fc5467279ea848` |

The reports also contain hashes for every controller/adapter source file used
by the run. Wall-clock fields are intentionally excluded from deterministic
history hashes.

## Trajectory benchmark

Artifacts:

- [JSON](assets/mujoco_tracking_benchmark.json), SHA-256
  `277793b062a0197ad7b64a815b63797de5b3ce1bae0eccc2c9fd67804e5cd6a8`;
- [PNG](assets/mujoco_tracking_benchmark.png), SHA-256
  `aabe4f74ea6d910a22ad948fb6d36a4e63f2bc61db7f69de3b44b3e5bde230ef`;
- generator: `examples/run_mujoco_benchmark.py`;
- runner: `neuro_adaptive_control/adapters/mujoco_simulation.py`.

The common scenario is 8.0 s, seed 23, 0.002 s control period, four 0.0005 s
MuJoCo substeps per command, no external wrench, and a 27D RBF regressor. Four
adaptive trajectories are run. The circle is repeated with frozen output
weights; the generator verifies that every other configuration field, time
history, desired history, and initial actual state matches.

| Circle metric | Frozen weights | Adaptive NAC |
|---|---:|---:|
| control steps / MuJoCo steps | 4,000 / 16,000 | 4,000 / 16,000 |
| impedance tracking RMSE | 0.0328492086 m | 0.00385450524 m |
| impedance maximum error | 0.0359411736 m | 0.0253601350 m |
| maximum command-force norm | 56.5299358 N | 62.3283990 N |
| maximum arm-torque norm | 36.8368226 N m | 40.1894396 N m |
| maximum absolute arm torque | 26.9360031 N m | 29.6645631 N m |
| maximum absolute joint speed | 0.807564313 rad/s | 0.762956789 rad/s |
| final RBF weight norm | 0 | 9.95066318 |
| torque saturation count | 0 | 0 |
| torque-rate saturation count | 0 | 0 |
| state / fault | `stopped` / empty | `stopped` / empty |

The matched impedance-RMSE improvement is

$$100\frac{0.0328492086-0.00385450524}{0.0328492086}
=88.2660636\%.$$

That percentage describes this single deterministic circle comparison; it is
not an across-task or hardware performance claim.

| Adaptive reference | RMSE | Maximum error | Maximum absolute torque | Final weight norm |
|---|---:|---:|---:|---:|
| circle | 0.00385450524 m | 0.0253601350 m | 29.6645631 N m | 9.95066318 |
| line | 0.00384195035 m | 0.0254873099 m | 30.1320552 N m | 9.96481001 |
| figure8 | 0.00387925387 m | 0.0253290039 m | 29.5184913 N m | 9.77930226 |
| fixed_point | 0.00383689112 m | 0.0254865847 m | 30.1477212 N m | 9.96021015 |

All four runs stopped without fault. The acceptance record passes the 0.03 m
circle RMSE limit, 0.08 m circle maximum-error limit, and minimum 10%
matched-baseline improvement.

The adaptive circle wall duration was 4.179872640 s for 8.0 s simulated time,
a real-time factor of 1.91393. Its measured NAC times were 0.139414 ms median,
0.151849 ms p95, and 0.161215 ms p99. MuJoCo four-substep times were
0.317160 ms median, 0.337921 ms p95, and 0.358439 ms p99. These are
standalone synchronous-runner measurements on one host, not ROS wall-rate or
hard-real-time guarantees.

The adaptive circle deterministic-history SHA-256 is
`c8ae3d971103261a37219653e1227e4f9b3afdb287e5746c64b1af416e3f7eee`;
the frozen baseline history is
`fdcc9bbe356132bbe89a6acb9ef13f9653a501826a2d8adb95b2f27cc29d8b8b`.

## Grasp, lift, and hold benchmark

Artifacts:

- [JSON](assets/mujoco_grasp_benchmark.json), SHA-256
  `35e9e69818abb1d39d9c10c207621e3ccd945431aab5d4175d1a4c591b15888b`;
- [PNG](assets/mujoco_grasp_benchmark.png), SHA-256
  `52d3921985c1dd85d225f9b3ae9a31b8a1ff7be0698dca14704089d27296a9bd`;
- generator: `examples/run_mujoco_grasp.py`;
- runner: `neuro_adaptive_control/adapters/mujoco_grasp.py`.

The 11.0 s, seed-29 schedule is `pregrasp -> descend -> close -> lift ->
hold -> lower -> release -> retreat`. The controller still runs at a fixed
0.002 s period with four 0.0005 s MuJoCo substeps.

| Metric | Measured | Acceptance boundary |
|---|---:|---:|
| object lift height | 0.0776457370 m | at least 0.05 m |
| hold duration | 2.000 s | at least 2.0 s |
| hold drop | 0.000396843 m | at most 0.005 m |
| bilateral-contact duration | 5.916 s | informational |
| hold bilateral-contact ratio | 1.000 | at least 0.90 |
| maximum contact-force norm sum | 143.047668 N | at most 180 N |
| maximum gripper effort | 2.000 N | at most 2.0 N |
| maximum absolute arm torque | 30.0241608 N m | guard-dependent |
| maximum absolute joint speed | 0.724289501 rad/s | guard-dependent |
| maximum penetration | 0.000660195 m | informational |
| unexpected contacts | 0 | zero |
| solver warnings | 0 | zero |
| torque saturation | 0 | zero |
| returned-height error | 2.77556e-17 m | informational |
| state / fault | `stopped` / empty | stopped without fault |

The deterministic-history SHA-256 is
`e7c8bc75fa4b8bd61efa83efc4527ed60d04022874efb43e8966596981d00303`.
The measured wall duration was 7.624596971 s and real-time factor 1.44270;
again, this is not a hard-real-time guarantee.

## ROS launch wall-clock measurements

These additional measurements exercise the ROS nodes and launch graph. They
are host-load observations and are intentionally not included in deterministic
artifact hashes. Both no-GUI runs used the same 0.002 s control period and
four 0.0005 s substeps as the synchronous evidence above.

Artifacts:

- [trajectory timing JSON](assets/mujoco_ros_trajectory_timing.json), SHA-256
  `50a4d8ed23e29f6819557579cfd2758e827c6e3ea7025bfd32ede58c64ce4e25`;
- [grasp timing JSON](assets/mujoco_ros_grasp_timing.json), SHA-256
  `c28d2354625c70f50fc1c30c6cac95cd6684f2e3d54a021b0abeb69c8882303a`.

| Measurement | Trajectory, no GUI | Grasp, no GUI |
|---|---:|---:|
| simulated duration | 8.000 s | 11.000 s |
| wall duration | 8.04557 s | 12.95994 s |
| observed control-step rate | 497.168 Hz | 424.385 Hz |
| callback runtime over 2 ms | 203 / 4,000 | 3,088 / 5,500 |
| timer inter-arrival median | 1.99810 ms | 2.47623 ms |
| timer inter-arrival p95 | 2.18833 ms | 2.76055 ms |
| timer inter-arrival p99 | 3.27665 ms | 4.66062 ms |
| timer inter-arrival maximum | 3.89486 ms | 5.60808 ms |
| NAC median / p95 / p99 | 0.15919 / 0.17627 / 0.21965 ms | 0.15492 / 0.16814 / 0.21292 ms |
| four-substep median / p95 / p99 | 0.38825 / 0.42447 / 0.48006 ms | 0.40945 / 0.44954 / 0.50057 ms |

The no-GUI trajectory stopped without fault or torque saturation and reached
the same deterministic tracking metrics as the synchronous artifact. The
grasp launch reported successful bilateral contact, lift, and hold, with no
fault, solver warning, unexpected contact, or torque saturation. RViz and the
passive viewer are visual-debug options rather than part of the target-rate
path. None of these measurements establishes a hard-real-time guarantee.

## Visual captures

- [RViz capture](images/ur5e_robotiq_mujoco_rviz.png), SHA-256
  `ed69bd62002b806edb7dcd6fea32fa23692a2706eefce67c215f9b979dd66e0d`;
- [MuJoCo viewer capture](images/ur5e_robotiq_mujoco_viewer.png), SHA-256
  `0da637b6eb531a5d87611387a3bbc31c590d44835b780c5c0aa1d99dd95baeae`.

Both files are direct captures from the running candidate. They are visual
evidence only; all numerical claims come from JSON or launch metrics.

## Reproduction

Run from the repository root with the official MuJoCo 3.9.0 binding:

```bash
python3 examples/run_mujoco_benchmark.py \
  --output-directory results/mujoco_tracking

python3 examples/run_mujoco_grasp.py \
  --output-directory results/mujoco_grasp
```

Generated artifacts include fresh source/model hashes. If implementation or
model files change, regenerate rather than copying old numbers. A different
host may change wall-clock telemetry while deterministic state-history hashes
should remain stable for the pinned software stack.

## Evidence boundary

These results verify deterministic numerical behavior and bounded metrics for
the bundled simulation scenarios. They do not prove formal sampled-data
stability, hard-real-time scheduling, model fidelity to a specific physical
robot, contact generalization, hardware safety, or a 6D adaptive controller.
RViz visual acceptance is a separate release gate because the PNG files record
benchmark histories rather than the interactive RViz rendering.
