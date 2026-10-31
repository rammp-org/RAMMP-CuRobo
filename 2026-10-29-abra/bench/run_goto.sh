#!/bin/bash
# Run track_goto.py in the go-to-many driver image (it has the new actions): run_goto.sh <out> [args]
out=$1; shift
mkdir -p ~/hil-blend/out/$out
CY=/home/abra/.config/rammp-bench/cyclonedds-local.xml
exec docker run --rm --name goto-bench --network host --ipc host -e ROS_DOMAIN_ID=0 \
  -e CYCLONEDDS_URI=file://$CY -v $CY:$CY:ro -v /home/abra/hil-blend/bench:/bench:ro \
  -v /home/abra/hil-blend/out/$out:/out --entrypoint bash kinova-gen3-ros2:go-to-many-kortex \
  -c 'source /opt/ros/humble/setup.bash && source /module_ws/install/setup.bash && cd /bench && python3 -u track_goto.py --out /out "$@"' _ "$@"
