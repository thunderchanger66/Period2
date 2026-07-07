#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>
#include <tf2/time.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>

#include "ikd_Tree.h"

#include <cmath>
#include <string>
#include <vector>
#include <chrono>

class IKDTreeNode : public rclcpp::Node
{
public:
    IKDTreeNode() : Node("ikd_tree_node") {
        cloud_topic_ = this->declare_parameter<std::string>(
            "cloud_topic", "/lidar/points_deskewed"
        );
        map_frame_ = this->declare_parameter<std::string>(
            "map_frame", "world"
        );
        base_frame_ = this->declare_parameter<std::string>(
            "base_frame", "vehicle_blue/chassis"
        );
        scan_voxel_size_ = this->declare_parameter<double>(
            "scan_voxel_size", 0.2
        );
        ikd_downsample_size_ = this->declare_parameter<double>(
            "ikd_downsample_size", 0.2
        );
        local_map_length_ = this->declare_parameter<double>(
            "local_map_length", 20.0
        );
        local_map_height_ = this->declare_parameter<double>(
            "local_map_height", 8.0
        );
         radius_search_size_ = this->declare_parameter<double>(
            "radius_search_size", 3.0
        );
        k_neighbors_ = this->declare_parameter<int>(
            "k_neighbors", 5
        );

        ikd_tree_.InitializeKDTree(0.5, 0.7, ikd_downsample_size_);
        ikd_tree_.set_downsample_param(ikd_downsample_size_);

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(&IKDTreeNode::cloudCallback, this, std::placeholders::_1)
        );
        local_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/ikd_local_map", 10
        );
        radius_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/ikd_radius_map", 10
        );
        nearest_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/ikd_nearest_points", 10
        );

        RCLCPP_INFO(this->get_logger(), "ikd-tree mapping node started.");
        RCLCPP_INFO(this->get_logger(), "cloud_topic: %s", cloud_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "map_frame: %s", map_frame_.c_str());
        RCLCPP_INFO(this->get_logger(), "base_frame: %s", base_frame_.c_str());
    }

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    Eigen::Matrix4f transformToEigen(const geometry_msgs::msg::TransformStamped& tf);
    BoxPointType makeLocalBox(const ikdTree_PointType& center);
    bool isValidBox(const BoxPointType& box);
    void deleteOutsideLocalBox(const BoxPointType& keep_box);
    void publishLocalBoxMap(const BoxPointType& local_box, const rclcpp::Time& stamp);
    void publishRadiusSearch(const ikdTree_PointType& query, const rclcpp::Time& stamp);
    void publishNearestSearch(const ikdTree_PointType& query, const rclcpp::Time& stamp);
    void publishIKDPoints(const KD_TREE<ikdTree_PointType>::PointVector& points,
        const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
        const rclcpp::Time& stamp);

    std::string cloud_topic_;
    std::string map_frame_;
    std::string base_frame_;

    double scan_voxel_size_;
    double ikd_downsample_size_;
    double local_map_length_;
    double local_map_height_;
    double radius_search_size_;
    int k_neighbors_;

    bool has_built_tree_ = false;

    KD_TREE<ikdTree_PointType> ikd_tree_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr radius_map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nearest_pub_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

void IKDTreeNode::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (msg->header.frame_id.empty()) {
        RCLCPP_WARN(this->get_logger(), "Input cloud frame_id is empty.");
        return;
    }

    geometry_msgs::msg::TransformStamped tf_map_lidar;
    geometry_msgs::msg::TransformStamped tf_map_base;

    try {
        tf_map_lidar = tf_buffer_->lookupTransform(
            map_frame_,
            msg->header.frame_id,
            tf2::TimePointZero
        );

        tf_map_base = tf_buffer_->lookupTransform(
            map_frame_,
            base_frame_,
            tf2::TimePointZero
        );
    } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "TF lookup failed: %s",
            ex.what()
        );
        return;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_lidar(
        new pcl::PointCloud<pcl::PointXYZI>
    );
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_map(
        new pcl::PointCloud<pcl::PointXYZI>
    );
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ds(
        new pcl::PointCloud<pcl::PointXYZI>
    );
    pcl::fromROSMsg(*msg, *cloud_lidar);
    if (cloud_lidar->empty()) {
        return;
    }

    Eigen::Matrix4f T_map_lidar = transformToEigen(tf_map_lidar);
    pcl::transformPointCloud(
        *cloud_lidar,
        *cloud_map,
        T_map_lidar
    );

    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setLeafSize(
        static_cast<float>(scan_voxel_size_),
        static_cast<float>(scan_voxel_size_),
        static_cast<float>(scan_voxel_size_)
    );
    voxel.setInputCloud(cloud_map);
    voxel.filter(*cloud_ds);

    KD_TREE<ikdTree_PointType>::PointVector points_to_add;
    points_to_add.reserve(cloud_ds->size());

    for (const auto & p : cloud_ds->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) 
            continue;
        points_to_add.emplace_back(p.x, p.y, p.z);
    }
    if (points_to_add.empty())
        return;
    
    if (!has_built_tree_) {
        ikd_tree_.Build(points_to_add);
        has_built_tree_ = true;
    } else {
        bool downsample_on = true;
        ikd_tree_.Add_Points(points_to_add, downsample_on);
    }
    ikdTree_PointType robot_pos(
        static_cast<float>(tf_map_base.transform.translation.x),
        static_cast<float>(tf_map_base.transform.translation.y),
        static_cast<float>(tf_map_base.transform.translation.z)
    );

    BoxPointType local_box = makeLocalBox(robot_pos);
    deleteOutsideLocalBox(local_box);
    publishLocalBoxMap(local_box, msg->header.stamp);
    publishRadiusSearch(robot_pos, msg->header.stamp);
    publishNearestSearch(robot_pos, msg->header.stamp);

    RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "input=%zu, ds=%zu, added=%zu, ikd_size=%d, valid=%d",
        cloud_lidar->size(),
        cloud_ds->size(),
        points_to_add.size(),
        ikd_tree_.size(),
        ikd_tree_.validnum()
    );
}

Eigen::Matrix4f IKDTreeNode::transformToEigen(const geometry_msgs::msg::TransformStamped& tf) {
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
    T.block<3, 3>(0, 0) = q.toRotationMatrix();
    T(0, 3) = static_cast<float>(t.x);
    T(1, 3) = static_cast<float>(t.y);
    T(2, 3) = static_cast<float>(t.z);

    return T;
}

BoxPointType IKDTreeNode::makeLocalBox(const ikdTree_PointType& center) {
    BoxPointType box;

    float half_xy = static_cast<float>(local_map_length_ * 0.5);
    float half_z = static_cast<float>(local_map_height_ * 0.5);

    box.vertex_min[0] = center.x - half_xy;
    box.vertex_max[0] = center.x + half_xy;

    box.vertex_min[1] = center.y - half_xy;
    box.vertex_max[1] = center.y + half_xy;

    box.vertex_min[2] = center.z - half_z;
    box.vertex_max[2] = center.z + half_z;

    return box;
}

bool IKDTreeNode::isValidBox(const BoxPointType& box) {
    return box.vertex_min[0] < box.vertex_max[0] &&
            box.vertex_min[1] < box.vertex_max[1] &&
            box.vertex_min[2] < box.vertex_max[2];
}

void IKDTreeNode::deleteOutsideLocalBox(const BoxPointType& keep_box) {
    if (!has_built_tree_ || ikd_tree_.validnum() <= 0)
        return;

    BoxPointType range = ikd_tree_.tree_range();
    std::vector<BoxPointType> delete_boxes;

    float rx_min = range.vertex_min[0];
    float ry_min = range.vertex_min[1];
    float rz_min = range.vertex_min[2];

    float rx_max = range.vertex_max[0];
    float ry_max = range.vertex_max[1];
    float rz_max = range.vertex_max[2];

    float kx_min = keep_box.vertex_min[0];
    float ky_min = keep_box.vertex_min[1];
    float kz_min = keep_box.vertex_min[2];

    float kx_max = keep_box.vertex_max[0];
    float ky_max = keep_box.vertex_max[1];
    float kz_max = keep_box.vertex_max[2];

    // x 方向左侧
    if (rx_min < kx_min) {
        BoxPointType box = range;
        box.vertex_max[0] = kx_min;
        if (isValidBox(box)) {
            delete_boxes.push_back(box);
        }
    }
    // x 方向右侧
    if (rx_max > kx_max) {
        BoxPointType box = range;
        box.vertex_min[0] = kx_max;
        if (isValidBox(box)) {
            delete_boxes.push_back(box);
        }
    }
    // y 方向下侧，只删除 x 保留范围内的外部区域，避免重复太多
    if (ry_min < ky_min) {
        BoxPointType box;
        box.vertex_min[0] = std::max(rx_min, kx_min);
        box.vertex_max[0] = std::min(rx_max, kx_max);
        box.vertex_min[1] = ry_min;
        box.vertex_max[1] = ky_min;
        box.vertex_min[2] = rz_min;
        box.vertex_max[2] = rz_max;
        if (isValidBox(box)) {
            delete_boxes.push_back(box);
        }
    }
    // y 方向上侧
    if (ry_max > ky_max) {
        BoxPointType box;
        box.vertex_min[0] = std::max(rx_min, kx_min);
        box.vertex_max[0] = std::min(rx_max, kx_max);
        box.vertex_min[1] = ky_max;
        box.vertex_max[1] = ry_max;
        box.vertex_min[2] = rz_min;
        box.vertex_max[2] = rz_max;
        if (isValidBox(box)) {
            delete_boxes.push_back(box);
        }
    }
    // z 方向下侧
    if (rz_min < kz_min) {
        BoxPointType box;
        box.vertex_min[0] = std::max(rx_min, kx_min);
        box.vertex_max[0] = std::min(rx_max, kx_max);
        box.vertex_min[1] = std::max(ry_min, ky_min);
        box.vertex_max[1] = std::min(ry_max, ky_max);
        box.vertex_min[2] = rz_min;
        box.vertex_max[2] = kz_min;
        if (isValidBox(box)) {
            delete_boxes.push_back(box);
        }
    }
    // z 方向上侧
    if (rz_max > kz_max) {
        BoxPointType box;
        box.vertex_min[0] = std::max(rx_min, kx_min);
        box.vertex_max[0] = std::min(rx_max, kx_max);
        box.vertex_min[1] = std::max(ry_min, ky_min);
        box.vertex_max[1] = std::min(ry_max, ky_max);
        box.vertex_min[2] = kz_max;
        box.vertex_max[2] = rz_max;
        if (isValidBox(box)) {
            delete_boxes.push_back(box);
        }
    }

    if (!delete_boxes.empty()) {
        int deleted = ikd_tree_.Delete_Point_Boxes(delete_boxes);

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "delete boxes=%zu, deleted points=%d",
            delete_boxes.size(),
            deleted
        );
    }
}

void IKDTreeNode::publishLocalBoxMap(const BoxPointType& local_box, const rclcpp::Time& stamp) {
    KD_TREE<ikdTree_PointType>::PointVector local_points;

    ikd_tree_.Box_Search(local_box, local_points);

    publishIKDPoints(local_points, local_map_pub_, stamp);

    RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "local box points=%zu",
        local_points.size()
    );
}

void IKDTreeNode::publishRadiusSearch(const ikdTree_PointType& query, const rclcpp::Time& stamp) {
    KD_TREE<ikdTree_PointType>::PointVector radius_points;

    ikd_tree_.Radius_Search(
        query,
        static_cast<float>(radius_search_size_),
        radius_points
    );

    publishIKDPoints(radius_points, radius_map_pub_, stamp);

    RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "radius search points=%zu, radius=%.2f",
        radius_points.size(),
        radius_search_size_
    );
}

void IKDTreeNode::publishNearestSearch(const ikdTree_PointType& query, const rclcpp::Time& stamp) {
    KD_TREE<ikdTree_PointType>::PointVector nearest_points;
    std::vector<float> distances;

    ikd_tree_.Nearest_Search(
        query,
        k_neighbors_,
        nearest_points,
        distances
    );

    publishIKDPoints(nearest_points, nearest_pub_, stamp);

    double nearest_dist = -1.0;

    if (!nearest_points.empty()) {
        double dx = nearest_points[0].x - query.x;
        double dy = nearest_points[0].y - query.y;
        double dz = nearest_points[0].z - query.z;
        nearest_dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "nearest found=%zu, nearest distance=%.3f",
        nearest_points.size(),
        nearest_dist
    );
}

void IKDTreeNode::publishIKDPoints(const KD_TREE<ikdTree_PointType>::PointVector& points,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
    const rclcpp::Time& stamp) {

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZI>
    );

    cloud->reserve(points.size());

    for (const auto & p : points) {
        pcl::PointXYZI q;
        q.x = p.x;
        q.y = p.y;
        q.z = p.z;
        q.intensity = 1.0f;
        cloud->push_back(q);
    }

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);

    msg.header.stamp = stamp;
    msg.header.frame_id = map_frame_;

    pub->publish(msg);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<IKDTreeNode> node = std::make_shared<IKDTreeNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}