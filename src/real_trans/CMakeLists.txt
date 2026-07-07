cmake_minimum_required(VERSION 3.8)
project(real_trans)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

add_executable(real_trans src/real_trans.cpp)
ament_target_dependencies(real_trans
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  pcl_conversions
)

add_executable(voxel_map src/voxel_map.cpp)
ament_target_dependencies(voxel_map
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  pcl_conversions
  nav_msgs
)

add_executable(kd_tree src/kd_tree.cpp)
ament_target_dependencies(kd_tree
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  pcl_conversions
  nav_msgs
)

add_executable(ikd_tree 
  src/ikd_tree.cpp
  third_party/ikd-Tree/ikd-Tree/ikd_Tree.cpp
)
ament_target_dependencies(ikd_tree
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  pcl_conversions
  nav_msgs
)

target_link_libraries(real_trans ${PCL_LIBRARIES} Eigen3::Eigen)
target_link_libraries(voxel_map ${PCL_LIBRARIES} Eigen3::Eigen)
target_link_libraries(kd_tree ${PCL_LIBRARIES} Eigen3::Eigen)

target_include_directories(ikd_tree PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  third_party/ikd-Tree/ikd-Tree
)
target_link_libraries(ikd_tree 
  ${PCL_LIBRARIES} 
  Eigen3::Eigen
  Threads::Threads
)

install(TARGETS real_trans voxel_map kd_tree ikd_tree DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
