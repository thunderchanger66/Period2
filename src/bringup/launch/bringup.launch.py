from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    bridge_node = Node(
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
    )

    TF_node = Node(
        package='bringup',
        executable='tf_broadcaster',
        parameters=[
            {'use_sim_time': True},
            {'pose_topic': '/model/vehicle_blue/pose'},
            {'odom_frame': 'world'},
            {'base_frame': 'vehicle_blue/chassis'}
        ],
        output='screen'
    )

    static_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['1.3', '0', '0.9', '0', '0', '0', 'vehicle_blue/chassis', 'vehicle_blue/chassis/gpu_lidar'],
        parameters=[{'use_sim_time': True}],
    )

    deskew_node = Node(
        package='deskew_test',
        executable='deskew_test',
        parameters=[
            {'use_sim_time': True},
        ],
        output='screen'
    )

    real_trans_node = Node(
        package='real_trans',
        executable='real_trans',
        parameters=[
            {'use_sim_time': True},
        ],
        output='screen'
    )

    return LaunchDescription([
        bridge_node,
        TF_node,
        static_node,
        deskew_node,
        real_trans_node,
    ])