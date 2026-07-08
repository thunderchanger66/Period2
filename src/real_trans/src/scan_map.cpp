#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>
#include <tf2/time.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>

#include <Eigen/Dense>

#include <mutex>
#include <string>
#include <cmath>
#include <algorithm>

class ScanMapNode : public rclcpp::Node
{
public:
    ScanMapNode() : Node("scan_map_node") {
        scan_topic_ = this->declare_parameter<std::string>(
            "scan_topic", "/lidar/points_deskewed"
        );
        local_map_topic_ = this->declare_parameter<std::string>(
            "local_map_topic", "/local_map_crop"
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
        map_voxel_size_ = this->declare_parameter<double>(
            "map_voxel_size", 0.2
        );
        max_correspondence_distance_ = this->declare_parameter<double>(
            "max_correspondence_distance", 1.0
        );
        max_iterations_ = this->declare_parameter<int>(
            "max_iterations", 40
        );
        transformation_epsilon_ = this->declare_parameter<double>(
            "transformation_epsilon", 1e-6
        );
        fitness_epsilon_ = this->declare_parameter<double>(
            "fitness_epsilon", 1e-6
        );

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        scan_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            scan_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(&ScanMapNode::scanCallback, this, std::placeholders::_1)
        );
        local_map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            local_map_topic_,
            10,
            std::bind(&ScanMapNode::localMapCallback, this, std::placeholders::_1)
        );
        aligned_scan_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/icp_aligned_scan",
            10
        );
        est_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/icp_est_pose",
            10
        );

        local_map_ = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

        RCLCPP_INFO(this->get_logger(), "Scan-to-local-map ICP node started.");
        RCLCPP_INFO(this->get_logger(), "scan_topic: %s", scan_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "local_map_topic: %s", local_map_topic_.c_str());
    }
private:
    void localMapCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void scanCallback(const sensor_msgs::msg::PointCloud2::SharedPtr scan_msg);
    Eigen::Matrix4f transformToEigen(const geometry_msgs::msg::TransformStamped& tf);
    double computeTranslationError(const Eigen::Matrix4f& T_est, const Eigen::Matrix4f& T_gt);
    double computeRotationErrorDeg(const Eigen::Matrix4f& T_est, const Eigen::Matrix4f& T_gt);
    void publishAlignedScan(const pcl::PointCloud<pcl::PointXYZI>& cloud, const rclcpp::Time& stamp);
    void publishEstimatedPose(const Eigen::Matrix4f& T_map_base, const rclcpp::Time& stamp);

    std::string scan_topic_;
    std::string local_map_topic_;
    std::string map_frame_;
    std::string base_frame_;

    double scan_voxel_size_;
    double map_voxel_size_;
    double max_correspondence_distance_;
    int max_iterations_;
    double transformation_epsilon_;
    double fitness_epsilon_;

    bool has_local_map_ = false;

    std::mutex map_mutex_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr local_map_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr scan_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_sub_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_scan_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr est_pose_pub_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

void ScanMapNode::localMapCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZI>
    );

    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty()) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Received empty local map."
        );
        return;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ds(
        new pcl::PointCloud<pcl::PointXYZI>
    );

    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setLeafSize(
        static_cast<float>(map_voxel_size_),
        static_cast<float>(map_voxel_size_),
        static_cast<float>(map_voxel_size_)
    );
    voxel.setInputCloud(cloud);
    voxel.filter(*cloud_ds);

    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        local_map_ = cloud_ds;

        if (!msg->header.frame_id.empty()) {
            map_frame_ = msg->header.frame_id;
        }

        has_local_map_ = true;
    }

    RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "Local map updated: raw=%zu, ds=%zu, frame=%s",
        cloud->size(),
        cloud_ds->size(),
        map_frame_.c_str()
    );
}

void ScanMapNode::scanCallback(const sensor_msgs::msg::PointCloud2::SharedPtr scan_msg) {
    if (!has_local_map_) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Waiting for local map..."
        );
        return;
    }

    if (scan_msg->header.frame_id.empty()) {
        RCLCPP_WARN(this->get_logger(), "Scan frame_id is empty.");
        return;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr scan_raw(
        new pcl::PointCloud<pcl::PointXYZI>
    );
    pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ds(
        new pcl::PointCloud<pcl::PointXYZI>
    );
    pcl::fromROSMsg(*scan_msg, *scan_raw);
    if (scan_raw->empty()) {
        return;
    }

    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setLeafSize(
        static_cast<float>(scan_voxel_size_),
        static_cast<float>(scan_voxel_size_),
        static_cast<float>(scan_voxel_size_)
    );
    voxel.setInputCloud(scan_raw);
    voxel.filter(*scan_ds);
    if (scan_ds->empty()) {
        return;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr target_map(
        new pcl::PointCloud<pcl::PointXYZI>
    );

    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        *target_map = *local_map_;
    }

    if (target_map->size() < 30) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Local map too small: %zu points. Skip ICP.",
            target_map->size()
        );
        return;
    }

    geometry_msgs::msg::TransformStamped tf_map_lidar_init;
    geometry_msgs::msg::TransformStamped tf_map_base_gt;
    geometry_msgs::msg::TransformStamped tf_base_lidar;

    try {
        tf_map_lidar_init = tf_buffer_->lookupTransform(
            map_frame_,
            scan_msg->header.frame_id,
            tf2::TimePointZero
        );

        tf_map_base_gt = tf_buffer_->lookupTransform(
            map_frame_,
            base_frame_,
            tf2::TimePointZero
        );

        tf_base_lidar = tf_buffer_->lookupTransform(
            base_frame_,
            scan_msg->header.frame_id,
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

    Eigen::Matrix4f T_map_lidar_init = transformToEigen(tf_map_lidar_init);
    Eigen::Matrix4f T_map_base_gt = transformToEigen(tf_map_base_gt);
    Eigen::Matrix4f T_base_lidar = transformToEigen(tf_base_lidar);

    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;

    icp.setInputSource(scan_ds);
    icp.setInputTarget(target_map);

    icp.setMaxCorrespondenceDistance(
        static_cast<float>(max_correspondence_distance_)
    );
    icp.setMaximumIterations(max_iterations_);
    icp.setTransformationEpsilon(transformation_epsilon_);
    icp.setEuclideanFitnessEpsilon(fitness_epsilon_);

    pcl::PointCloud<pcl::PointXYZI> aligned_scan;

    icp.align(aligned_scan, T_map_lidar_init);

    if (!icp.hasConverged()) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "ICP did not converge. fitness=%.6f",
            icp.getFitnessScore()
        );
        return;
    }

    Eigen::Matrix4f T_map_lidar_est = icp.getFinalTransformation();

    Eigen::Matrix4f T_map_base_est =
        T_map_lidar_est * T_base_lidar.inverse();

    double trans_error = computeTranslationError(
        T_map_base_est,
        T_map_base_gt
    );

    double rot_error_deg = computeRotationErrorDeg(
        T_map_base_est,
        T_map_base_gt
    );

    publishAlignedScan(aligned_scan, scan_msg->header.stamp);
    publishEstimatedPose(T_map_base_est, scan_msg->header.stamp);

    RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "ICP converged. scan_raw=%zu, scan_ds=%zu, map=%zu, fitness=%.6f, trans_err=%.4f m, rot_err=%.4f deg",
        scan_raw->size(),
        scan_ds->size(),
        target_map->size(),
        icp.getFitnessScore(),
        trans_error,
        rot_error_deg
    );
}

Eigen::Matrix4f ScanMapNode::transformToEigen(const geometry_msgs::msg::TransformStamped& tf) {
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

double ScanMapNode::computeTranslationError(const Eigen::Matrix4f& T_est, const Eigen::Matrix4f& T_gt) {
    Eigen::Vector3f t_est = T_est.block<3, 1>(0, 3);
    Eigen::Vector3f t_gt = T_gt.block<3, 1>(0, 3);

    return static_cast<double>((t_est - t_gt).norm());
}

double ScanMapNode::computeRotationErrorDeg(const Eigen::Matrix4f& T_est, const Eigen::Matrix4f& T_gt) {
    Eigen::Matrix3f R_est = T_est.block<3, 3>(0, 0);
    Eigen::Matrix3f R_gt = T_gt.block<3, 3>(0, 0);

    Eigen::Matrix3f R_err = R_gt.transpose() * R_est;

    float trace_value = R_err.trace();
    float cos_theta = (trace_value - 1.0f) * 0.5f;

    cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta));

    double angle_rad = std::acos(cos_theta);
    double angle_deg = angle_rad * 180.0 / M_PI;

    return angle_deg;
}

void ScanMapNode::publishAlignedScan(const pcl::PointCloud<pcl::PointXYZI>& cloud, const rclcpp::Time& stamp) {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);

    msg.header.stamp = stamp;
    msg.header.frame_id = map_frame_;

    aligned_scan_pub_->publish(msg);
}

void ScanMapNode::publishEstimatedPose(const Eigen::Matrix4f& T_map_base, const rclcpp::Time& stamp) {
    geometry_msgs::msg::PoseStamped pose_msg;

    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = map_frame_;

    Eigen::Matrix3f R = T_map_base.block<3, 3>(0, 0);
    Eigen::Quaternionf q(R);
    q.normalize();

    pose_msg.pose.position.x = T_map_base(0, 3);
    pose_msg.pose.position.y = T_map_base(1, 3);
    pose_msg.pose.position.z = T_map_base(2, 3);

    pose_msg.pose.orientation.w = q.w();
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();

    est_pose_pub_->publish(pose_msg);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ScanMapNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}