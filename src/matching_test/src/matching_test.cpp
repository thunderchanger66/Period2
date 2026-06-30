#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <filesystem>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>

class MatchingNode : public rclcpp::Node
{
public:
    using PointT = pcl::PointXYZ;
    using CloudT = pcl::PointCloud<PointT>;

    MatchingNode() : Node("matching_node") {
        declare_parameter<std::string>("input_topic", "/lidar/points_deskewed");
        declare_parameter<std::string>("output_topic", "/two_frame_icp_map");
        declare_parameter<std::string>("save_dir", "results");

        declare_parameter<double>("leaf_size", 0.15);
        declare_parameter<double>("max_corr_dist", 1.0);
        declare_parameter<int>("max_iter", 50);
        declare_parameter<double>("trans_eps", 1e-6);
        declare_parameter<double>("fitness_eps", 1e-6);

        input_topic_ = get_parameter("input_topic").as_string();
        output_topic_ = get_parameter("output_topic").as_string();
        save_dir_ = get_parameter("save_dir").as_string();

        leaf_size_ = get_parameter("leaf_size").as_double();
        max_corr_dist_ = get_parameter("max_corr_dist").as_double();
        max_iter_ = get_parameter("max_iter").as_int();
        trans_eps_ = get_parameter("trans_eps").as_double();
        fitness_eps_ = get_parameter("fitness_eps").as_double();

        std::filesystem::create_directories(save_dir_);

        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();

        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_, 
            qos,
            std::bind(&MatchingNode::cloudCallback, this, std::placeholders::_1)
        );
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            output_topic_,
            qos
        );

        last_cloud_ = std::make_shared<CloudT>();

        RCLCPP_INFO(get_logger(), "Subscribe: %s", input_topic_.c_str());
        RCLCPP_INFO(get_logger(), "Publish:   %s", output_topic_.c_str());
        RCLCPP_INFO(get_logger(), "leaf_size: %.3f", leaf_size_);
        RCLCPP_INFO(get_logger(), "max_corr_dist: %.3f", max_corr_dist_);
    }
private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    std::shared_ptr<CloudT> preprocessCloud(const std::shared_ptr<CloudT>& cloud_in);
    void printMatrix(const Eigen::Matrix4f& T);

    std::string input_topic_;
    std::string output_topic_;
    std::string save_dir_;

    double leaf_size_;
    double max_corr_dist_;
    int max_iter_;
    double trans_eps_;
    double fitness_eps_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;

    std::shared_ptr<CloudT> last_cloud_;
    std_msgs::msg::Header last_header_;

    bool has_last_cloud_ = false;
    int frame_count_ = 0;
};

std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> 
    MatchingNode::preprocessCloud(const std::shared_ptr<CloudT>& cloud_in) {

    CloudT::Ptr cloud_no_nan = std::make_shared<CloudT>();
    std::shared_ptr<CloudT> cloud_down = std::make_shared<CloudT>();

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud_in, *cloud_no_nan, indices);

    pcl::VoxelGrid<PointT> voxel;
    voxel.setInputCloud(cloud_no_nan);
    voxel.setLeafSize(
        static_cast<float>(leaf_size_),
        static_cast<float>(leaf_size_),
        static_cast<float>(leaf_size_)
    );
    voxel.filter(*cloud_down);

    return cloud_down;
}

void MatchingNode::printMatrix(const Eigen::Matrix4f& T) {
    std::cout << "\nICP delta_T = \n" << T << std::endl;

    float dx = T(0, 3);
    float dy = T(1, 3);
    float dz = T(2, 3);

    Eigen::Matrix3f R = T.block<3,3>(0, 0);
    Eigen::Vector3f euler = R.eulerAngles(2, 1, 0); //留作测试
    float yaw0 = euler(0);

    float yaw = std::atan2(T(1, 0), T(0, 0));

    std::cout << "translation: "
                  << "x = " << dx << ", "
                  << "y = " << dy << ", "
                  << "z = " << dz << std::endl;
    std::cout << "approx yaw: " << yaw << " rad" << std::endl;
    std::cout << "approx yaw0: " << yaw0 << " rad" << std::endl;
}

void MatchingNode::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    CloudT::Ptr cloud_raw = std::make_shared<CloudT>();
    pcl::fromROSMsg(*msg, *cloud_raw);
    if (cloud_raw->empty()) {
        RCLCPP_WARN(this->get_logger(), "Received empty cloud.");
        return;
    }

    CloudT::Ptr cloud_curr = preprocessCloud(cloud_raw);
    if (cloud_curr->empty()) {
        RCLCPP_WARN(get_logger(), "Current cloud empty after preprocessing.");
        return;
    }

    frame_count_++;

    if (!has_last_cloud_){
        *last_cloud_ = *cloud_curr;
        last_header_ = msg->header;
        has_last_cloud_ = true;

        // pcl::io::savePCDFileBinary(save_dir_ + "/frame_0.pcd", *last_cloud_);

        RCLCPP_INFO(
            get_logger(),
            "Received first frame. points raw: %zu, down: %zu",
            cloud_raw->size(),
            cloud_curr->size()
        );

        return;
    }

    // 当前帧 cloud_curr 配准到上一帧 last_cloud_
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(cloud_curr);
    icp.setInputTarget(last_cloud_);

    icp.setMaxCorrespondenceDistance(max_corr_dist_);
    icp.setMaximumIterations(max_iter_);
    icp.setTransformationEpsilon(trans_eps_);
    icp.setEuclideanFitnessEpsilon(fitness_eps_);

    CloudT::Ptr cloud_curr_aligned = std::make_shared<CloudT>();
    icp.align(*cloud_curr_aligned);

    if (!icp.hasConverged()) {
        RCLCPP_WARN(get_logger(), "ICP did not converge. Skip this frame.");
        return;
    }

    Eigen::Matrix4f delta_T = icp.getFinalTransformation();
    double score = icp.getFitnessScore();

    RCLCPP_INFO(
        get_logger(),
        "ICP converged. score = %.6f, curr points = %zu, prev points = %zu",
        score,
        cloud_curr->size(),
        last_cloud_->size()
    );
    printMatrix(delta_T);

    // 构造两帧叠加点云：
    // 上一帧保持不动，当前帧使用 ICP 后的 cloud_curr_aligned
    CloudT::Ptr two_frame_map = std::make_shared<CloudT>();
    *two_frame_map += *last_cloud_;
    *two_frame_map += *cloud_curr_aligned;
    sensor_msgs::msg::PointCloud2 out_msg;
    pcl::toROSMsg(*two_frame_map, out_msg);
    out_msg.header = msg->header;
    pub_->publish(out_msg);

    // pcl::io::savePCDFileBinary(save_dir_ + "/frame_curr_down.pcd", *cloud_curr);
    // pcl::io::savePCDFileBinary(save_dir_ + "/frame_curr_aligned.pcd", *cloud_curr_aligned);
    // pcl::io::savePCDFileBinary(save_dir_ + "/two_frame_icp_map.pcd", *two_frame_map);

    // RCLCPP_INFO(
    //     get_logger(),
    //     "Saved two-frame ICP result to %s",
    //     (save_dir_ + "/two_frame_icp_map.pcd").c_str()
    // );

    *last_cloud_ = *cloud_curr;
    last_header_ = msg->header;
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<MatchingNode> node = std::make_shared<MatchingNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}