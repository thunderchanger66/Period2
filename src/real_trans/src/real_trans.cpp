#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2/exceptions.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Dense>

#include <string>

class RealTrans : public rclcpp::Node
{
public:
    RealTrans() : Node("real_trans_node") {
        this->declare_parameter<std::string>("cloud_topic", "/lidar/points_deskewed");
        this->declare_parameter<std::string>("map_frame", "world");
        this->declare_parameter<double>("voxel_leaf_size", 0.2);
        cloud_topic_ = this->get_parameter("cloud_topic").as_string();
        map_frame_ = this->get_parameter("map_frame").as_string();
        voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(&RealTrans::cloudCallback, this, std::placeholders::_1)
        );
        map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/global_cloud_map",
            rclcpp::SensorDataQoS()
        );

        global_map_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

        RCLCPP_INFO(this->get_logger(), "TF cloud map builder started.");
        RCLCPP_INFO(this->get_logger(), "Cloud topic: %s", cloud_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Map frame: %s", map_frame_.c_str());
    }

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    Eigen::Matrix4f transformToEigen(const geometry_msgs::msg::TransformStamped& tf);
    void downsampleGlobalMap();
    void publishMap(const rclcpp::Time& stamp);

    std::string cloud_topic_;
    std::string map_frame_;
    double voxel_leaf_size_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr global_map_;   
};

void RealTrans::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (msg->header.frame_id.empty()) {
        RCLCPP_WARN(this->get_logger(), "PointCloud2 frame_id is empty.");
        return;
    }

    geometry_msgs::msg::TransformStamped tf_map_cloud;
    try {
        tf_map_cloud = tf_buffer_->lookupTransform(
            map_frame_,
            msg->header.frame_id,
            msg->header.stamp,
            rclcpp::Duration::from_seconds(0.1)
        );
    } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "TF lookup failed: %s -> %s, reason: %s",
            msg->header.frame_id.c_str(),
            map_frame_.c_str(),
            ex.what()
        );
        return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_lidar(
        new pcl::PointCloud<pcl::PointXYZ>
    );

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map(
        new pcl::PointCloud<pcl::PointXYZ>
    );

    pcl::fromROSMsg(*msg, *cloud_lidar);
    if (cloud_lidar->empty()) return;

    Eigen::Matrix4f T_map_cloud = transformToEigen(tf_map_cloud);
    pcl::transformPointCloud(
        *cloud_lidar,
        *cloud_map,
        T_map_cloud
    );

    *global_map_ += *cloud_map;
    downsampleGlobalMap();
    publishMap(msg->header.stamp);
}

Eigen::Matrix4f RealTrans::transformToEigen(const geometry_msgs::msg::TransformStamped& tf) {
    const auto & t = tf.transform.translation;
    const auto & q_msg = tf.transform.rotation;

    Eigen::Quaternionf q(
        static_cast<float>(q_msg.w),
        static_cast<float>(q_msg.x),
        static_cast<float>(q_msg.y),
        static_cast<float>(q_msg.z)
    );

    q.normalize();

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3,3>(0,0) = q.toRotationMatrix();
    T(0,3) = static_cast<float>(t.x);
    T(1,3) = static_cast<float>(t.y);
    T(2,3) = static_cast<float>(t.z);

    return T;
}

void RealTrans::downsampleGlobalMap() {
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setLeafSize(
        static_cast<float>(voxel_leaf_size_),
        static_cast<float>(voxel_leaf_size_),
        static_cast<float>(voxel_leaf_size_)
    );

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(
        new pcl::PointCloud<pcl::PointXYZ>
    );

    voxel.setInputCloud(global_map_);
    voxel.filter(*filtered);

    global_map_ = filtered;
}

void RealTrans::publishMap(const rclcpp::Time& stamp) {
    sensor_msgs::msg::PointCloud2 map_msg;
    pcl::toROSMsg(*global_map_, map_msg);

    map_msg.header.stamp = stamp;
    map_msg.header.frame_id = map_frame_;

    map_pub_->publish(map_msg);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<RealTrans> node = std::make_shared<RealTrans>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}