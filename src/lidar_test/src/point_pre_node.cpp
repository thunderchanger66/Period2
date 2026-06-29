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

class PointCloudPreprocessNode : public rclcpp::Node
{
public:
    using PointT = pcl::PointXYZ;
    using CloudT = pcl::PointCloud<PointT>;

    PointCloudPreprocessNode()
    : Node("pointcloud_preprocess_node")
    {
        this->declare_parameter<std::string>("input_topic", "/lidar/points");
        this->declare_parameter<std::string>("output_topic", "/lidar/points_filtered");
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

        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            output_topic_,
            rclcpp::SensorDataQoS()
        );

        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(&PointCloudPreprocessNode::cloudCallback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), "Subscribe: %s", input_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Publish:   %s", output_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Save dir:  %s", save_dir_.c_str());
    }

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        CloudT::Ptr cloud_raw(new CloudT);
        CloudT::Ptr cloud_no_nan(new CloudT);
        CloudT::Ptr cloud_down(new CloudT);
        CloudT::Ptr cloud_filtered(new CloudT);
        CloudT::Ptr cloud_transformed(new CloudT);

        // 1. ROS PointCloud2 -> PCL PointCloud
        pcl::fromROSMsg(*msg, *cloud_raw);

        if (cloud_raw->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Received empty cloud.");
            return;
        }

        // 2. 去除 NaN 点
        std::vector<int> valid_indices;
        pcl::removeNaNFromPointCloud(*cloud_raw, *cloud_no_nan, valid_indices);

        // 3. 体素降采样
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(cloud_no_nan);
        voxel.setLeafSize(
            static_cast<float>(leaf_size_),
            static_cast<float>(leaf_size_),
            static_cast<float>(leaf_size_)
        );
        voxel.filter(*cloud_down);

        // 4. 统计离群点滤波
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(cloud_down);
        sor.setMeanK(mean_k_);
        sor.setStddevMulThresh(stddev_mul_);
        sor.filter(*cloud_filtered);

        // 5. 坐标变换示例
        // 这里先给一个单位变换。你后面可以替换成真正的 T_WL。
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();

        // 示例：如果想测试平移 1 米，可以取消下面这行注释
        // T(0, 3) = 1.0f;

        pcl::transformPointCloud(*cloud_filtered, *cloud_transformed, T);

        // 6. PCL -> ROS PointCloud2 并发布
        sensor_msgs::msg::PointCloud2 out_msg;
        pcl::toROSMsg(*cloud_transformed, out_msg);
        out_msg.header = msg->header;

        pub_->publish(out_msg);

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "raw: %zu, no_nan: %zu, down: %zu, filtered: %zu",
            cloud_raw->size(),
            cloud_no_nan->size(),
            cloud_down->size(),
            cloud_filtered->size()
        );

        // 7. 保存一次 PCD
        if (!saved_)
        {
            std::string raw_path = save_dir_ + "/raw_cloud.pcd";
            std::string down_path = save_dir_ + "/downsampled_cloud.pcd";
            std::string filtered_path = save_dir_ + "/filtered_cloud.pcd";
            std::string transformed_path = save_dir_ + "/transformed_cloud.pcd";

            pcl::io::savePCDFileBinary(raw_path, *cloud_raw);
            pcl::io::savePCDFileBinary(down_path, *cloud_down);
            pcl::io::savePCDFileBinary(filtered_path, *cloud_filtered);
            pcl::io::savePCDFileBinary(transformed_path, *cloud_transformed);

            RCLCPP_INFO(this->get_logger(), "Saved PCD files:");
            RCLCPP_INFO(this->get_logger(), "  %s", raw_path.c_str());
            RCLCPP_INFO(this->get_logger(), "  %s", down_path.c_str());
            RCLCPP_INFO(this->get_logger(), "  %s", filtered_path.c_str());
            RCLCPP_INFO(this->get_logger(), "  %s", transformed_path.c_str());

            if (save_once_)
            {
                saved_ = true;
            }
        }
    }

private:
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

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudPreprocessNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}