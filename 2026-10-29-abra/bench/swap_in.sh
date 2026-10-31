#!/bin/bash
# Put the go-to-many driver and a blend-capable planner in place of the stack's (abra).
# Stops sheppy-planner and sheppy-arm (sheppy does not restart them); swap_out.sh restores.
set -e
CY=/home/abra/.config/rammp-bench/cyclonedds-local.xml
docker stop sheppy-planner
docker run -d --name goto-planner --runtime nvidia --network host --ipc host \
  -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI=file://$CY -v $CY:$CY:ro -v curobo_kernel_cache:/root/.cache \
  -v /home/abra/hil-blend/bench:/bench:ro rammp-curobo:blend \
  /opt/rammp_curobo/rammp_curobo_ros/lib/rammp_curobo_ros/planner_node \
  --ros-args -p config:=gen3_real.yaml -p world:=/bench/world_keepout.yaml
for i in $(seq 1 60); do docker logs goto-planner 2>&1 | grep -q "rammp_curobo ready" && break; sleep 5; done
docker logs goto-planner 2>&1 | grep "rammp_curobo ready" | cut -c1-160
docker stop sheppy-arm
# as sheppy runs it (docker inspect sheppy-arm), from the branch's KORTEX image; no --rm (CONTRIBUTING)
docker run -d --name goto-arm --network host --ipc host --cap-add SYS_NICE \
  --ulimit memlock=-1:-1 --ulimit rtprio=99:99 \
  -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI=file://$CY -v $CY:$CY:ro \
  -v /home/abra/.sheppy/scratch/arm/params.yaml:/sheppy/params.yaml:ro \
  kinova-gen3-ros2:go-to-many-kortex \
  /module_ws/install/kinova_gen3_ros2/lib/kinova_gen3_ros2/kinova_gen3_node --ip 192.168.1.10 --cpu 11 \
  --ros-args --params-file /sheppy/params.yaml
sleep 8
docker logs goto-arm 2>&1 | grep -i "up (real)\|error\|fault" | cut -c1-200 | tail -5
