#!/usr/bin/env python3
# Copyright 2026 Zhemin Huang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Regenerate the pinned UR5e + Robotiq 2F-85 combined MJCF.

The two input descriptions are unmodified files from MuJoCo Menagerie commit
71f066ad0be9cd271f7ed58c030243ef157af9f4.  Their BSD notices live beside the
inputs.  This script attaches the models through the UR5e attachment site and
applies this project's explicit torque-control and scene contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


SOURCE_COMMIT = "71f066ad0be9cd271f7ed58c030243ef157af9f4"
ARM_JOINTS = (
    ("shoulder_pan", "shoulder_pan_joint", 150.0),
    ("shoulder_lift", "shoulder_lift_joint", 150.0),
    ("elbow", "elbow_joint", 150.0),
    ("wrist_1", "wrist_1_joint", 28.0),
    ("wrist_2", "wrist_2_joint", 28.0),
    ("wrist_3", "wrist_3_joint", 28.0),
)


def _load_spec(xml_path: Path, asset_dir: Path) -> mujoco.MjSpec:
    """Load a pristine vendor XML through an in-memory asset VFS."""
    assets = {
        f"assets/{path.name}": path.read_bytes()
        for path in sorted(asset_dir.iterdir())
        if path.is_file()
    }
    return mujoco.MjSpec.from_string(xml_path.read_text(), assets=assets)


def _default_by_class(root: ET.Element, class_name: str) -> ET.Element:
    for element in root.findall(".//default"):
        if element.get("class") == class_name:
            return element
    raise RuntimeError(f"missing default class {class_name!r}")


def _body(root: ET.Element, name: str) -> ET.Element:
    for element in root.findall(".//body"):
        if element.get("name") == name:
            return element
    raise RuntimeError(f"missing body {name!r}")


def _build_tree(project_root: Path) -> ET.ElementTree:
    vendor = project_root / "mujoco" / "vendor"
    assets = project_root / "mujoco" / "assets"
    arm = _load_spec(
        vendor / "universal_robots_ur5e" / "ur5e.xml", assets
    )
    gripper = _load_spec(vendor / "robotiq_2f85" / "2f85.xml", assets)
    arm.attach(gripper, prefix="gripper_", site="attachment_site")
    arm.compile()

    root = ET.fromstring(arm.to_xml())
    root.set("model", "ur5e_robotiq_2f85")
    root.insert(
        0,
        ET.Comment(
            " Derived from MuJoCo Menagerie commit "
            f"{SOURCE_COMMIT}; see mujoco/vendor/*/LICENSE. "
        ),
    )
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", "assets")
    compiler.set("autolimits", "true")
    compiler.set("balanceinertia", "false")
    for mesh in root.findall("./asset/mesh"):
        filename = mesh.get("file", "")
        if mesh.get("name", "").startswith("gripper_") and filename.startswith(
            "gripper_"
        ):
            mesh.set("file", filename.removeprefix("gripper_"))

    option = root.find("option")
    assert option is not None
    option.attrib.update(
        {
            "timestep": "0.0005",
            "integrator": "implicitfast",
            "cone": "elliptic",
            "impratio": "10",
            "gravity": "0 0 -9.81",
            "iterations": "100",
            "tolerance": "1e-10",
        }
    )

    arm_default = _default_by_class(root, "ur5e")
    arm_joint = arm_default.find("joint")
    assert arm_joint is not None
    arm_joint.attrib.update({"damping": "0.5", "frictionloss": "0.05"})
    for class_name in (
        "gripper_driver",
        "gripper_follower",
        "gripper_spring_link",
        "gripper_coupler",
    ):
        gripper_joint = _default_by_class(root, class_name).find("joint")
        assert gripper_joint is not None
        gripper_joint.set("frictionloss", "0.001")
        if "damping" not in gripper_joint.attrib:
            gripper_joint.set("damping", "0.002")

    wrist_mount = _body(root, "gripper_base_mount")
    ET.SubElement(
        wrist_mount,
        "site",
        {
            "name": "wrist_ft_site",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
            "size": "0.003",
            "rgba": "0.6 0.1 0.8 1",
            "group": "4",
        },
    )

    worldbody = root.find("worldbody")
    assert worldbody is not None
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "ground",
            "type": "plane",
            "size": "2 2 0.05",
            "pos": "0 0 0",
            "rgba": "0.18 0.20 0.22 1",
            "friction": "0.9 0.02 0.002",
            "solref": "0.01 1",
            "solimp": "0.95 0.99 0.001",
            "group": "3",
        },
    )
    table = ET.SubElement(worldbody, "body", {"name": "table", "pos": "0 0.45 0.15"})
    ET.SubElement(
        table,
        "geom",
        {
            "name": "table_top",
            "type": "box",
            "size": "0.45 0.35 0.05",
            "rgba": "0.50 0.34 0.20 1",
            "friction": "0.8 0.02 0.002",
            "solref": "0.008 1",
            "solimp": "0.95 0.99 0.001",
            "group": "3",
        },
    )
    object_body = ET.SubElement(
        worldbody,
        "body",
        {"name": "grasp_object", "pos": "-0.134 0.492 0.245"},
    )
    ET.SubElement(object_body, "freejoint", {"name": "object_freejoint"})
    ET.SubElement(
        object_body,
        "inertial",
        {
            "mass": "0.20",
            "pos": "0 0 0",
            "diaginertia": "0.000148333 0.000148333 0.0000833333",
        },
    )
    ET.SubElement(
        object_body,
        "geom",
        {
            "name": "grasp_object_collision",
            "type": "box",
            "size": "0.020 0.020 0.040",
            "rgba": "0.90 0.72 0.12 1",
            "friction": "1.1 0.03 0.003",
            "solref": "0.006 1",
            "solimp": "0.95 0.99 0.001",
            "priority": "2",
            "group": "3",
        },
    )
    ET.SubElement(
        object_body,
        "site",
        {
            "name": "object_site",
            "type": "sphere",
            "size": "0.004",
            "rgba": "1 0.8 0 1",
            "group": "4",
        },
    )

    actuator = root.find("actuator")
    assert actuator is not None
    actuator.clear()
    for name, joint, limit in ARM_JOINTS:
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": name,
                "joint": joint,
                "gear": "1",
                "ctrllimited": "true",
                "ctrlrange": f"{-limit:g} {limit:g}",
                "forcerange": f"{-limit:g} {limit:g}",
            },
        )
    ET.SubElement(
        actuator,
        "general",
        {
            "name": "gripper_fingers_actuator",
            "class": "gripper_2f85",
            "tendon": "gripper_split",
            "ctrllimited": "true",
            "forcelimited": "true",
            "ctrlrange": "0 255",
            "forcerange": "-5 5",
            "gainprm": "0.3137255 0 0",
            "biasprm": "0 -100 -10",
        },
    )

    old_keyframe = root.find("keyframe")
    if old_keyframe is not None:
        root.remove(old_keyframe)
    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "force", {"name": "wrist_force", "site": "wrist_ft_site"})
    ET.SubElement(sensor, "torque", {"name": "wrist_torque", "site": "wrist_ft_site"})
    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MJCF; defaults to mujoco/ur5e_robotiq_2f85.xml",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output = args.output or project_root / "mujoco" / "ur5e_robotiq_2f85.xml"
    tree = _build_tree(project_root)
    tree.write(output, encoding="unicode", xml_declaration=False)
    output.write_text(output.read_text() + "\n")
    model = mujoco.MjModel.from_xml_path(str(output))
    if (model.nq, model.nv, model.nu) != (21, 20, 7):
        raise RuntimeError(
            f"unexpected compiled dimensions {(model.nq, model.nv, model.nu)}"
        )
    print(f"wrote {output} (nq={model.nq}, nv={model.nv}, nu={model.nu})")


if __name__ == "__main__":
    main()
