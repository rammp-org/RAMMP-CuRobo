cmake_minimum_required(VERSION 3.8)
project(rammp_curobo_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/PlanToPose.action"
  "action/PlanToJoints.action"
  "action/ExecuteTrajectory.action"
  "srv/SetWorld.srv"
  "srv/UpdateWorldBoxes.srv"
  "srv/CheckTrajectory.srv"
  "srv/SetIgnoreRegion.srv"
  "srv/SetTarget.srv"
  DEPENDENCIES geometry_msgs sensor_msgs trajectory_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
