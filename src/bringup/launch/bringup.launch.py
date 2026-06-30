from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # 2. 桥接节点（桥接 LiDAR, IMU, Pose）
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
                '/model/vehicle_blue/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose'
            ],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),

        # 3. TF 广播节点（订阅 PoseStamped 发布 TF）
        Node(
            package='bringup',   # 替换为你的包名
            executable='tf_broadcaster',
            parameters=[
                {'use_sim_time': True},
                {'pose_topic': '/model/vehicle_blue/pose'},
                {'odom_frame': 'world'},
                {'base_frame': 'vehicle_blue/chassis'}
            ],
            output='screen'
        ),

        # 4. 静态 TF（雷达相对底盘）
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['1.3', '0', '0.9', '0', '0', '0', 'vehicle_blue/chassis', 'vehicle_blue/chassis/gpu_lidar'],
            parameters=[{'use_sim_time': True}],
        ),
    ])