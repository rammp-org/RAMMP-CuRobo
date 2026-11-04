import os
from glob import glob

from setuptools import setup

package_name = "mebot_sim"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RAMMP",
    maintainer_email="chrisman4247@gmail.com",
    description="The MEBot sim stack: a MuJoCo arm beside the planner nodes",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "sim_arm = mebot_sim.sim_arm:main",
            "try_goal = mebot_sim.try_goal:main",
            "check_layers = mebot_sim.check_layers:main",
            "find_pairs = mebot_sim.find_pairs:main",
            "run_suite = mebot_sim.run_suite:main",
        ],
    },
)
