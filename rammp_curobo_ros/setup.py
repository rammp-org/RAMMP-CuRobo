import os
from glob import glob

from setuptools import find_packages, setup

package_name = "rammp_curobo_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RAMMP",
    maintainer_email="chrisman4247@gmail.com",
    description="ROS 2 wrapper for the rammp_curobo planning core",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "planner_node = rammp_curobo_ros.planner_node:main",
            "tour_demo = rammp_curobo_ros.tour_demo:main",
            "dance_demo = rammp_curobo_ros.dance_demo:main",
            "cameras = rammp_curobo_ros.cameras:main",
        ],
    },
)
