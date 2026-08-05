# UR5e + Robotiq model provenance

## Frozen sources

The repository vendors only the files required from Google DeepMind's MuJoCo
Menagerie commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`:

- `universal_robots_ur5e`: BSD-3-Clause, Copyright 2018 ROS Industrial
  Consortium;
- `robotiq_2f85`: BSD-2-Clause, Copyright 2013 ROS-Industrial.

The pristine upstream XML, README, and license files are under
`mujoco/vendor/`. Twenty UR OBJ files and eight Robotiq STL files are under
`mujoco/assets/`. `mujoco/SHA256SUMS` fixes their byte identity. No model,
mesh, calibration, or controller source was copied from the older local ROS or
UR5e projects.

MuJoCo is an Apache-2.0 runtime dependency pinned to 3.9.0. It is not vendored.
The project code remains Apache-2.0, while the model files retain their BSD
licenses as listed in `THIRD_PARTY_NOTICES.md`.

## Project-authored derivation

`tools/generate_combined_model.py` loads both pristine descriptions, attaches
the gripper at the UR5e `attachment_site`, prefixes all gripper symbols, and
writes `mujoco/ur5e_robotiq_2f85.xml`. The deliberate changes are:

1. replace the six upstream position servos with unit-gear, bounded torque
   motors using the upstream force limits;
2. set `0.0005 s`, `implicitfast`, elliptic friction cone, `impratio=10`, 100
   solver iterations, and deterministic tolerances;
3. add finite arm and gripper damping/friction loss;
4. preserve the Robotiq split tendon, two loop-closure connects, driver
   equality, collision exclusions, pad friction, `solref`, and `solimp`;
5. add a wrist sensor site, table, ground, and an explicitly inertialized
   first-party 0.20 kg benchmark box, with group-2 presentation geoms
   separated from the unchanged group-3 collision geoms;
6. add force/torque sensors at the gripper mount.

The RViz URDF is independently authored from the generated MJCF's exact joint
tree, axes, zero transforms, limits, attachment, and mesh URIs. It contains
eight independent Robotiq visualization joints because URDF cannot represent
the MuJoCo closed-loop equality constraints. Their positions always come from
MuJoCo `/joint_states`; no RViz mimic or IK animation fabricates motion.

The benchmark object is a project-defined primitive, not a copied or
manufacturer-calibrated asset. Its nominal inertial envelope is a uniform
0.05 m by 0.05 m by 0.08 m box with mass 0.20 kg. Applying
$I_{xx}=m(b^2+c^2)/12$, $I_{yy}=m(a^2+c^2)/12$, and
$I_{zz}=m(a^2+b^2)/12$ gives the committed diagonal inertia
`0.000148333 0.000148333 0.0000833333` kg m^2. The contact box is deliberately
inset to 0.04 m by 0.04 m by 0.08 m (5 mm per lateral face); this is a stated
benchmark contact approximation, not an unidentified physical-object
parameter. Tracking and grasp claims apply only to this bundled definition.

## Fidelity classification

This is a structurally complete articulated simulation model: all six arm
joints, all eight gripper joints, masses, centers of mass, inertias, actuators,
constraints, collision geometries, contact parameters, table, and object
participate in MuJoCo dynamics. It is not a manufacturer-calibrated digital
twin.

Both Menagerie READMEs call their descriptions *simplified*. In MuJoCo 3.9.0,
the original 2F-85's eleven explicit inertials sum to about 0.900000 kg, while
compiler-inferred mass from the mount and silicone mesh bodies raises the
compiled total to about 1.052608 kg. The Robotiq 2018 instruction manual gives
0.900 kg for the gripper including coupling, but it does not publish verified
per-link inertias. The same public model's compiled center of mass and pinch
offset also differ from the manual's aggregate reference values. This project
does not invent replacement link inertias, and reports this discrepancy as a
known limitation.

The public model also idealizes the gripper motor as a bounded position/tendon
actuator. Motor electrical dynamics, gear backlash, self-locking, identified
joint friction, soft-pad material behavior, and hardware-calibrated contact
parameters are not modeled. The collision shapes and contact solver settings
are simulation approximations whose stability is tested here.

Consequently, “full dynamics” in this repository means that the complete
articulated model and payload participate in the simulated equations of motion;
it does not mean manufacturer-validated parameter fidelity or validated
sim-to-real transfer.

## Verification boundary

Automated tests compile the composite, check body and composite inertia,
joint/actuator inventory, constraints, collision/contact settings, torque
actuator virtual work, payload influence, deterministic stepping, and
MuJoCo/URDF FK. Random legal arm states and constraint-settled gripper states
must remain within 1 mm TCP/pad position and 0.5 degree TCP orientation error.

The NAC never receives MuJoCo mass matrices, bias force, gravity, solver, or
contact parameters. Tests may read those values only to audit the plant and to
prove that the gripper payload changes arm dynamics.

## External references

- MuJoCo Menagerie UR5e:
  https://github.com/google-deepmind/mujoco_menagerie/tree/71f066ad0be9cd271f7ed58c030243ef157af9f4/universal_robots_ur5e
- MuJoCo Menagerie Robotiq 2F-85:
  https://github.com/google-deepmind/mujoco_menagerie/tree/71f066ad0be9cd271f7ed58c030243ef157af9f4/robotiq_2f85
- MuJoCo 3.9.0:
  https://github.com/google-deepmind/mujoco/releases/tag/3.9.0
- Robotiq 2F-85/2F-140 instruction manual (aggregate comparison only):
  https://assets.robotiq.com/website-assets/support_archives/document_en/2F-85_2F-140_Instruction_Manual_PDF_20181130.pdf
