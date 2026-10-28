#!/bin/bash
# Source ROS + the built workspace, then run whatever was asked (default
# CMD: the planner node on /config/robot/curobo/planner.yaml). The one verb is
# a first argument of exactly `example`: it is dropped and the planner node runs
# with no `config` (example mode: the image's Gen3 + kitchen world, never a
# robot), any further arguments passed to the node.
set -e
# Both files exist only inside the image, so shellcheck cannot follow them from
# the repo. SC1091 here is unfollowable-path, not an unchecked script.
# shellcheck source=/dev/null
source /opt/ros/humble/setup.bash
# shellcheck source=/dev/null
source /opt/rammp_curobo/setup.bash
if [ "${1:-}" = example ]; then
    shift
    exec /opt/rammp_curobo/rammp_curobo_ros/lib/rammp_curobo_ros/planner_node "$@"
fi
exec "$@"
