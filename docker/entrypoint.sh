#!/bin/bash
# Source ROS + the built workspace, then run whatever was asked (default
# CMD: the planning-only planner node on gen3_real.yaml).
set -e
# Both files exist only inside the image, so shellcheck cannot follow them from
# the repo. SC1091 here is unfollowable-path, not an unchecked script.
# shellcheck source=/dev/null
source /opt/ros/humble/setup.bash
# shellcheck source=/dev/null
source /opt/rammp_curobo/setup.bash
exec "$@"
