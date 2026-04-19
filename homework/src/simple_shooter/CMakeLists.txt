cmake_minimum_required(VERSION 3.8)
project(simple_shooter)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)



add_executable(shooter src/shooter.cpp src/shooter.h)
ament_target_dependencies(shooter rclcpp sensor_msgs cv_bridge OpenCV) #一键添加所有依赖项


install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(TARGETS shooter DESTINATION lib/${PROJECT_NAME})
ament_package()