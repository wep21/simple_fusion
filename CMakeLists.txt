cmake_minimum_required(VERSION 3.8)
project(simple_fusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(message_filters REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Signal.msg"
  DEPENDENCIES std_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")

add_library(${PROJECT_NAME}_node SHARED
  src/simple_fusion_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node
  message_filters::message_filters
  rclcpp::rclcpp
  rclcpp_components::component
  tf2_ros::tf2_ros
  ${cpp_typesupport_target}
  ${sensor_msgs_TARGETS}
)

rclcpp_components_register_node(${PROJECT_NAME}_node
  PLUGIN "simple_fusion::SimpleFusionNode"
  EXECUTABLE ${PROJECT_NAME}_exe
)

install(
  TARGETS
    ${PROJECT_NAME}_exe
    ${PROJECT_NAME}_node
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY
  launch
  rviz
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
