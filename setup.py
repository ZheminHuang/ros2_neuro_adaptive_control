# Copyright 2026 Zhemin Huang
# SPDX-License-Identifier: Apache-2.0

from glob import glob

from setuptools import find_packages, setup


package_name = "neuro_adaptive_control"


setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["THIRD_PARTY_NOTICES.md"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/mujoco", glob("mujoco/*.*")),
        ("share/" + package_name + "/mujoco", ["mujoco/SHA256SUMS"]),
        (
            "share/" + package_name + "/mujoco/assets",
            glob("mujoco/assets/*"),
        ),
        (
            "share/" + package_name + "/mujoco/vendor/robotiq_2f85",
            glob("mujoco/vendor/robotiq_2f85/*"),
        ),
        (
            "share/" + package_name + "/mujoco/vendor/universal_robots_ur5e",
            glob("mujoco/vendor/universal_robots_ur5e/*"),
        ),
        ("share/" + package_name + "/urdf", glob("urdf/*")),
        ("share/" + package_name + "/rviz", glob("rviz/*")),
    ],
    install_requires=["setuptools", "numpy>=1.21"],
    zip_safe=True,
    maintainer="Zhemin Huang",
    maintainer_email="zheminhuang@users.noreply.github.com",
    description=(
        "Model-free six-DoF neuro-adaptive impedance tracking with "
        "unknown-payload MuJoCo benchmarks for ROS 2."
    ),
    license="Apache-2.0",
    extras_require={
        "plot": ["matplotlib>=3.5", "Pillow>=9.0"],
        "mujoco": ["numpy==1.24.4", "mujoco==3.9.0"],
        "test": ["pytest"],
    },
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            (
                "nac_controller_node = "
                "neuro_adaptive_control.nodes.nac_controller_node:main"
            ),
            (
                "cartesian_demo_plant = "
                "neuro_adaptive_control.nodes.cartesian_demo_plant:main"
            ),
            (
                "mujoco_ur5e_plant_node = "
                "neuro_adaptive_control.nodes.mujoco_ur5e_plant_node:main"
            ),
            (
                "robotiq_gripper_action_server = "
                "neuro_adaptive_control.nodes."
                "robotiq_gripper_action_server:main"
            ),
            (
                "mujoco_rviz_bridge = "
                "neuro_adaptive_control.nodes.mujoco_rviz_bridge:main"
            ),
            (
                "payload_benchmark_node = "
                "neuro_adaptive_control.nodes.payload_benchmark_node:main"
            ),
        ],
    },
)
