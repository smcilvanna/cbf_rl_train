cmake_minimum_required(VERSION 3.0.2)
project(cbf_rl_train)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  src/dummy_trainer.py
  src/nmpc_cbf_train_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)