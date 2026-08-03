# Third-party notices

The project source code is Apache-2.0. The vendored robot descriptions and
meshes below retain their own permissive licenses; they are not relicensed as
Apache-2.0.

## Universal Robots UR5e model

- Source: MuJoCo Menagerie `universal_robots_ur5e`
- Pinned commit: `71f066ad0be9cd271f7ed58c030243ef157af9f4`
- Copyright: 2018 ROS Industrial Consortium
- License: BSD-3-Clause
- Notice: [mujoco/vendor/universal_robots_ur5e/LICENSE](mujoco/vendor/universal_robots_ur5e/LICENSE)

The BSD-3-Clause no-endorsement condition applies. Universal Robots, ROS
Industrial Consortium, and the model contributors do not endorse this project.

## Robotiq 2F-85 model

- Source: MuJoCo Menagerie `robotiq_2f85`
- Pinned commit: `71f066ad0be9cd271f7ed58c030243ef157af9f4`
- Copyright: 2013 ROS-Industrial
- License: BSD-2-Clause
- Notice: [mujoco/vendor/robotiq_2f85/LICENSE](mujoco/vendor/robotiq_2f85/LICENSE)

The composite MJCF and RViz URDF are project-authored derivative works built
from those pinned inputs. See
[docs/ur5e_robotiq_model_provenance.md](docs/ur5e_robotiq_model_provenance.md)
for the exact transformations and fidelity limitations.

## MuJoCo runtime

- Version: 3.9.0
- Upstream tag commit: `237c17e48539b6c90bf90d3161547cbdcbfaa1e0`
- Copyright: 2021–2026 Google DeepMind and contributors
- License: Apache-2.0

MuJoCo is installed as a dependency and is not vendored in this repository.

