# Unknown-payload benchmark protocol

This internal protocol prevents the showcase from leaking MuJoCo truth into
the adaptive controller or comparing unmatched scenarios. The public README
contains only the generated animation and measured results.

## Physical scenario

- One MuJoCo owner advances the complete UR5e, articulated Robotiq 2F-85,
  table, and free object at 0.0005 s.
- The controller period is 0.002 s with exactly four zero-order-hold MuJoCo
  substeps.
- The object exists from reset. Pickup does not hot-change a mass parameter;
  support transfers through physical bilateral contact.
- Held-out cases are 0.24, 0.31, and 0.36 kg with committed COM offsets,
  inertia scales, and deterministic seeds.
- The schedule is unloaded 6D motion, approach, grasp, lift, loaded 6D
  tracking, lower, release, and retreat.

## Information boundary

Adaptive and frozen NAC may read only arm (q,\dot q), measured TCP pose and
twist, FK/Jacobian kinematics, reference/impedance state, and configured
gains/limits. They may not read payload identity, mass, COM, inertia, contact
parameters, acquisition schedule, MuJoCo `qM`, `qfrc_bias`, live inverse
dynamics, or the oracle term. External wrench input is zero in this benchmark.

Contact is used by the experiment owner for safety, acquisition labeling, and
the frozen comparison switch. The main adaptive NAC does not receive the
event. A payload is acquired after bilateral finger contact and 12 mm object
lift.

## Controllers

1. `adaptive_nac`: two-layer V/W adaptation remains enabled.
2. `frozen_at_payload`: identical NAC and history until acquisition, then both
   V and W are held at their exact event checkpoint.
3. `nominal_model_based`: computed torque from the known nominal robot and
   gripper model, without payload compensation.
4. `oracle_model_based`: the nominal baseline plus known payload gravity/COM
   compensation after acquisition.

All variants share the same plant case, reference, initial state, gripper
effort, actuator limits, safety gates, simulation rate, and visualization
camera. The oracle is an upper reference, not evidence that model-based
control cannot address payload change when the payload is known.

## Metrics and claim gate

Translation and rotation-vector errors are never mixed into one scaled 6D
number. Loaded-phase position RMSE is reported in metres and orientation RMSE
in radians. Completion, bilateral contact, maximum contact force, maximum
torque, and faults are reported separately.

The adaptive-advantage gate passes only if:

- adaptive completion is at least 95% and no worse than frozen;
- median loaded position RMSE is at least 10% below frozen;
- median loaded orientation RMSE is at least 10% below frozen; and
- adaptive introduces no extra failed safety completion.

Claims against nominal model-based control are axis/metric specific. The
README reports nominal loaded-to-unloaded degradation and shows the oracle;
it does not claim adaptive NAC has lower error in every metric.

## Artifact contract

`examples/run_payload_benchmark.py` runs every held-out adaptive/frozen pair
and the showcase nominal/oracle trials. It writes:

- `payload_benchmark_metrics.json`: versions, source/model/history hashes,
  per-trial metrics, and aggregate gate;
- `payload_benchmark_results.png`: XYZ, rotation-vector, error, NN, and RMSE
  evidence; and
- `payload_benchmark_comparison.gif`: synchronized adaptive and nominal
  MuJoCo states rendered from canonical qpos histories, plus error/NN traces
  and a `PAYLOAD ACQUIRED` marker.

The GIF uses one camera, timeline, reference, and payload. It is not a screen
recording. Current evidence is deterministic simulation evidence only.
