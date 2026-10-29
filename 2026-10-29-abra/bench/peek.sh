#!/bin/bash
# One look at the arm's state from the blend image: joints, control status, action status.
exec docker run --rm --network host --ipc host \
  -e ROS_DOMAIN_ID=0 -e CYCLONEDDS_URI=file:///home/abra/.config/rammp-bench/cyclonedds-local.xml \
  -v /home/abra/.config/rammp-bench/cyclonedds-local.xml:/home/abra/.config/rammp-bench/cyclonedds-local.xml:ro \
  rammp-curobo:blend bash -c 'source /opt/driver_ws/setup.bash && source /opt/rammp_curobo/setup.bash &&
    timeout 5 ros2 topic echo --once /joint_states sensor_msgs/msg/JointState --field position;
    timeout 5 ros2 topic echo --once /control_status;
    timeout 5 ros2 topic echo --once /execute_joint_trajectory/_action/status'
