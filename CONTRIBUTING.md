<!--
Copyright 2026 Zhemin Huang

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contributing

Thank you for improving ROS 2 Neuro-Adaptive Control. Contributions must preserve the project's narrow v0.1 safety and mathematical claims.

## Development workflow

1. Open an issue describing the behavior, mathematical sign, or interface to change.
2. Keep the pure Python/NumPy algorithm in `neuro_adaptive_control/core/`; do not import ROS there.
3. Update `docs/math_contract.md` before changing an error, external-wrench, robust, or adaptation sign.
4. Add focused unit tests and run the corresponding test file while developing.
5. Run the complete ROS 2 build and test sequence from the README before opening a pull request.
6. Update documentation and `CHANGELOG.md` for user-visible behavior.

Do not submit robot IP addresses, calibration, credentials, participant information, private logs, video, experiment datasets, build products, or code of unclear provenance. Contributions derived from a paper or another implementation must identify the source and its license. When a source license is unclear, implement from the mathematical contract independently and document that decision.

## Pull-request checklist

- [ ] Core remains ROS-independent and the robot adapter boundary remains explicit.
- [ ] Mathematical definitions, units, frames, signs, and discretization are documented.
- [ ] Unit/integration tests cover success, invalid input, reset, and fault behavior.
- [ ] NAC and frozen-adaptation comparisons use identical deterministic scenarios.
- [ ] No hard real-time, hardware-safety, 6D-control, or general performance claim is added without evidence.
- [ ] Copyright/license headers and Apache-2.0 compatibility have been checked.
- [ ] Secret, privacy, absolute-path, and generated-artifact scans are clean.

Any contribution that you make to this repository will
be under the Apache 2 License, as dictated by that
[license](http://www.apache.org/licenses/LICENSE-2.0.html):

~~~
5. Submission of Contributions. Unless You explicitly state otherwise,
   any Contribution intentionally submitted for inclusion in the Work
   by You to the Licensor shall be under the terms and conditions of
   this License, without any additional terms or conditions.
   Notwithstanding the above, nothing herein shall supersede or modify
   the terms of any separate license agreement you may have executed
   with Licensor regarding such Contributions.
~~~
