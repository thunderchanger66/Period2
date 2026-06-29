#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/transforms.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <filesystem>
#include <string>
#include <vector>

class MyTestNode : public rclcpp::Node
{
public:
    using PointT = pcl::PointXYZ;
    using CloudT = pcl::PointCloud<PointT>;

    MyTestNode()
    : Node("my_test_node")
    {
        this->declare_parameter<std::string>("input_topic", "/lidar/points");
        this->declare_parameter<std::string>("output_topic", "/lidar/filtered_points");
        this->declare_parameter<std::string>("save_dir", "results");
        this->declare_parameter<double>("leaf_size", 0.10);
        this->declare_parameter<int>("mean_k", 20);
        this->declare_parameter<double>("stddev_mul", 1.0);
        this->declare_parameter<bool>("save_once", true);

        input_topic_ = this->get_parameter("input_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        save_dir_ = this->get_parameter("save_dir").as_string();
        leaf_size_ = this->get_parameter("leaf_size").as_double();
        mean_k_ = this->get_parameter("mean_k").as_int();
        stddev_mul_ = this->get_parameter("stddev_mul").as_double();
        save_once_ = this->get_parameter("save_once").as_bool();

        std::filesystem::create_directories(save_dir_);

        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(&MyTestNode::cloudCallback, this, std::placeholders::_1)
        );
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            output_topic_,
            rclcpp::SensorDataQoS()
        );

        RCLCPP_INFO(this->get_logger(), "Subscribe: %s", input_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Publish: %s", output_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Save dir: %s", save_dir_.c_str());
    }

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

    std::string input_topic_;
    std::string output_topic_;
    std::string save_dir_;

    double leaf_size_;
    int mean_k_;
    double stddev_mul_;

    bool save_once_;
    bool saved_ = false;
};

void MyTestNode::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    CloudT::Ptr cloud_raw = std::make_shared<CloudT>();
    CloudT::Ptr cloud_no_nan = std::make_shared<CloudT>();
    std::shared_ptr<CloudT> cloud_down = std::make_shared<CloudT>();
    auto cloud_filtered = std::make_shared<CloudT>();
    CloudT::Ptr cloud_transformed(new CloudT);

    pcl::fromROSMsg(*msg, *cloud_raw);
    if (cloud_raw->empty()) {
        RCLCPP_WARN(this->get_logger(), "Received empty cloud");
        return;
    }

    std::vector<int> valid_indices;
    pcl::removeNaNFromPointCloud(*cloud_raw, *cloud_no_nan, valid_indices);

    // 先降采样再滤波
    pcl::VoxelGrid<PointT> voxel;
    voxel.setInputCloud(cloud_no_nan);
    Eigen::Vector4f leaf(static_cast<float>(leaf_size_), 
        static_cast<float>(leaf_size_), static_cast<float>(leaf_size_), 0.0f);
    voxel.setLeafSize(leaf);
    voxel.filter(*cloud_down);

    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(cloud_down);
    sor.setMeanK(mean_k_);
    sor.setStddevMulThresh(stddev_mul_);
    sor.filter(*cloud_filtered);

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    pcl::transformPointCloud(*cloud_filtered, *cloud_transformed, T);

    std::shared_ptr<sensor_msgs::msg::PointCloud2> out_msg =
        std::make_shared<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(*cloud_transformed, *out_msg);
    out_msg->header = msg->header;
    pub_->publish(*out_msg);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<MyTestNode> node = std::make_shared<MyTestNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}