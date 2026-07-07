# 启动官方传感器示例
- gz sim ~/study/imu_ws/sim/sensor_tutorial.sdf

# 官方小车运动示例
- ~/study/sensor/sensor_launch.sh

# 桥接传感器数据，IMU从Gazebo到ROS2，还有lidar数据，位置数据
- ros2 run ros_gz_bridge parameter_bridge imu@sensor_msgs/msg/Imu[gz.msgs.IMU

- ros2 run ros_gz_bridge parameter_bridge /lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked imu@sensor_msgs/msg/Imu[gz.msgs.IMU /model/vehicle_blue/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose

# 测试发布静态数据变换
- ros2 run tf2_ros static_transform_publisher 0.8 0 0.5 0 0 0 world "vehicle_blue/chassis/gpu_lidar"

- ros2 run tf2_ros static_transform_publisher 0.8 0 0.5 0 0 0 "vehicle_blue/chassis" "vehicle_blue/chassis/gpu_lidar"

# 让车子转圈
- gz topic -t /cmd_vel -m gz.msgs.Twist -p 'linear: {x: 0.0}, angular: {z: 1.0}'

# 启动
- ros2 run deskew_test deskew_test
- ros2 run deskew_test deskew_test --ros-args -p use_sim_time:=true


# 新测试
- 