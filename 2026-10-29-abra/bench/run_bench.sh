#!/bin/bash
# Run track.py in the blend image on abra's host network: run_bench.sh <out dir name> [track.py args]
out=$1; shift
mkdir -p ~/hil-blend/out/$out
exec docker run --rm --name blend-bench --network host --ipc host \
  -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI=file:///home/abra/.config/rammp-bench/cyclonedds-local.xml \
  -v /home/abra/.config/rammp-bench/cyclonedds-local.xml:/home/abra/.config/rammp-bench/cyclonedds-local.xml:ro \
  -v /home/abra/hil-blend/bench:/bench:ro -v /home/abra/hil-blend/out/$out:/out \
  rammp-curobo:blend bash -c 'source /opt/driver_ws/setup.bash && source /opt/rammp_curobo/setup.bash && python3 -u /bench/track.py --out /out "$@"' _ "$@"
