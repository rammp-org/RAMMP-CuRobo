#!/bin/bash
# Source ROS + the built workspace, then run whatever was asked (default
# CMD: the planning-only planner node on gen3_real.yaml).
set -e
source /opt/ros/humble/setup.bash
source /opt/rammp_curobo/setup.bash
exec "$@"
