# Copyright 2026 Zhemin Huang
# SPDX-License-Identifier: Apache-2.0

from glob import glob

from setuptools import find_packages, setup


package_name = "neuro_adaptive_control"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy>=1.21"],
    zip_safe=True,
    maintainer="Zhemin Huang",
    maintainer_email="zheminhuang@users.noreply.github.com",
    description=(
        "Model-free 3D Cartesian neuro-adaptive impedance trajectory "
        "tracking for ROS 2."
    ),
    license="Apache-2.0",
    extras_require={
        "plot": ["matplotlib>=3.5"],
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
        ],
    },
)
