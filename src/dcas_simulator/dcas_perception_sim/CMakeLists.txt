cmake_minimum_required(VERSION 3.0.2)
project(dcas_perception_sim)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  dcas_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs dcas_msgs
)

catkin_install_python(PROGRAMS
  scripts/env_lane_node.py
  scripts/env_traffic_light_node.py
  scripts/env_guardrail_node.py
  scripts/env_building_node.py
  scripts/env_sign_node.py
  scripts/env_surrounding_vehicle_node.py
  scripts/sensor_motion_node.py
  scripts/sensor_gnss_node.py
  scripts/sensor_imu_node.py
  scripts/sensor_lidar_node.py
  scripts/sensor_radar_node.py
  scripts/sensor_camera_node.py
  scripts/vehicle_kinematic_node.py
  scripts/teleop_keyboard_node.py
  scripts/tf_sensor_frames_node.py
  scripts/final_project/control_node.py
  scripts/final_project/evaluation_node.py
  scripts/final_project/lane_detection_node.py
  scripts/final_project/scc_node.py
  
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


