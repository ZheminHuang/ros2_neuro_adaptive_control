# Unknown-payload benchmark protocol

This internal protocol prevents the showcase from leaking MuJoCo truth into
the adaptive controller or comparing unmatched scenarios. The public README
contains only the generated 1.00 kg animation and measured
adaptive-versus-nominal results.

## Physical scenario

- One MuJoCo owner advances the complete UR5e, articulated Robotiq 2F-85,
  table, and free object at 0.0005 s.
- The controller period is 0.002 s with exactly four zero-order-hold MuJoCo
  substeps.
- The object exists from reset. Pickup does not hot-change a mass parameter;
  support transfers through physical bilateral contact.
- Held-out cases are 0.50, 0.75, and 1.00 kg with committed COM offsets,
  inertia scales, and deterministic seeds.
- The 15 s schedule is unloaded 6D motion, approach, grasp, 80 mm lift, one
  smooth 40 mm-radius XY circle with bounded RX/RY/RZ excitation, lower,
  release, and retreat. The phase law has zero velocity and acceleration at
  the circle endpoints.
- The gripper uses its modelled 5 N effort ceiling. Object weight, inertia,
  collision, friction, and support transfer remain physical MuJoCo dynamics.

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
3. `nominal_model_based`: computed torque from the fixed pre-pickup robot and
   gripper model, without payload mass, COM, or inertia compensation. Its
   internal model is never updated from the MuJoCo plant.
4. `oracle_model_based`: the nominal baseline plus known payload gravity/COM
   compensation after acquisition.

All variants share the same plant case, reference, initial state,
deterministic seed, gripper effort, torque and torque-rate limits, safety
gates, simulation rate, and visualization camera. The public comparison uses
only the 1.00 kg case. The three adaptive/frozen payload pairs remain an
online-adaptation ablation, and the oracle remains a supplementary reference
rather than a public animation baseline. A model-based controller can
compensate the change when the payload is identified and its model is updated;
that is deliberately outside the fixed-nominal baseline tested here.

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

The public claim compares loaded-phase position and rotation-vector RMSE for
adaptive NAC against the fixed nominal controller in the 1.00 kg case. It does
not claim universal superiority over identified or payload-aware model-based
control.

## Artifact contract

`examples/run_payload_benchmark.py` runs every held-out adaptive/frozen pair
and the showcase nominal/oracle trials. It writes:

- `payload_benchmark_metrics.json`: versions, source/model/history hashes,
  per-trial metrics, and aggregate gate;
- `payload_benchmark_results.png`: XYZ, rotation-vector, error, NN, and RMSE
  evidence; and
- `payload_benchmark_comparison.webp`: full-color synchronized adaptive and
  fixed-nominal MuJoCo states rendered from canonical qpos histories, plus
  position/orientation error traces and a dashed pickup marker confined to
  both metric cards;
- `payload_benchmark_comparison.gif`: palette-quantized compatibility copy of
  the same synchronized animation.

Both animations use one camera, timeline, reference, and payload. Collision
geometries remain active in dynamics but only group-2 visual geometries are
rendered. These are not screen recordings. Current evidence is deterministic
simulation evidence only.
