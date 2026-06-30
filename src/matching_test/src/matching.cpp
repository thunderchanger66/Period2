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

#include <filesystem>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

class MultiFrameRegistrationNode : public rclcpp::Node
{
public:
    using PointT = pcl::PointXYZ;
    using CloudT = pcl::PointCloud<PointT>;

    MultiFrameRegistrationNode()
    : Node("multi_frame_registration_node")
    {
        declare_parameter<std::string>("input_topic", "/lidar/points_deskewed");
        declare_parameter<std::string>("map_topic", "/multi_frame_map");
        declare_parameter<std::string>("registered_scan_topic", "/registered_current_scan");
        declare_parameter<std::string>("map_frame", "map");
        declare_parameter<std::string>("save_dir", "results");

        declare_parameter<double>("scan_leaf_size", 0.25);
        declare_parameter<double>("map_leaf_size", 0.20);
        declare_parameter<double>("max_corr_dist", 0.8);
        declare_parameter<int>("max_iter", 20);
        declare_parameter<double>("trans_eps", 1e-6);
        declare_parameter<double>("fitness_eps", 1e-6);
        declare_parameter<double>("max_fitness_score", 2.0);

        declare_parameter<int>("process_every_n", 1);
        declare_parameter<int>("publish_every_n", 1);
        declare_parameter<int>("save_every_n", 50);

        input_topic_ = get_parameter("input_topic").as_string();
        map_topic_ = get_parameter("map_topic").as_string();
        registered_scan_topic_ = get_parameter("registered_scan_topic").as_string();
        map_frame_ = get_parameter("map_frame").as_string();
        save_dir_ = get_parameter("save_dir").as_string();

        scan_leaf_size_ = get_parameter("scan_leaf_size").as_double();
        map_leaf_size_ = get_parameter("map_leaf_size").as_double();
        max_corr_dist_ = get_parameter("max_corr_dist").as_double();
        max_iter_ = get_parameter("max_iter").as_int();
        trans_eps_ = get_parameter("trans_eps").as_double();
        fitness_eps_ = get_parameter("fitness_eps").as_double();
        max_fitness_score_ = get_parameter("max_fitness_score").as_double();

        process_every_n_ = get_parameter("process_every_n").as_int();
        publish_every_n_ = get_parameter("publish_every_n").as_int();
        save_every_n_ = get_parameter("save_every_n").as_int();

        std::filesystem::create_directories(save_dir_);

        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();

        cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_,
            qos,
            std::bind(&MultiFrameRegistrationNode::cloudCallback, this, std::placeholders::_1)
        );

        map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            map_topic_,
            qos
        );

        registered_scan_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            registered_scan_topic_,
            qos
        );

        last_cloud_ = std::make_shared<CloudT>();
        map_cloud_ = std::make_shared<CloudT>();

        T_map_lidar_ = Eigen::Matrix4f::Identity();

        RCLCPP_INFO(get_logger(), "Multi-frame registration node started.");
        RCLCPP_INFO(get_logger(), "Input topic: %s", input_topic_.c_str());
        RCLCPP_INFO(get_logger(), "Map topic: %s", map_topic_.c_str());
        RCLCPP_INFO(get_logger(), "Map frame: %s", map_frame_.c_str());
    }

private:
    CloudT::Ptr preprocessCloud(const CloudT::Ptr& cloud_in)
    {
        CloudT::Ptr cloud_no_nan = std::make_shared<CloudT>();
        CloudT::Ptr cloud_down = std::make_shared<CloudT>();

        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud_in, *cloud_no_nan, indices);

        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(cloud_no_nan);
        voxel.setLeafSize(
            static_cast<float>(scan_leaf_size_),
            static_cast<float>(scan_leaf_size_),
            static_cast<float>(scan_leaf_size_)
        );
        voxel.filter(*cloud_down);

        return cloud_down;
    }

    void downsampleMap()
    {
        if (map_cloud_->empty())
            return;

        CloudT::Ptr map_down = std::make_shared<CloudT>();

        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(map_cloud_);
        voxel.setLeafSize(
            static_cast<float>(map_leaf_size_),
            static_cast<float>(map_leaf_size_),
            static_cast<float>(map_leaf_size_)
        );
        voxel.filter(*map_down);

        map_cloud_ = map_down;
    }

    void publishCloud(
        const CloudT::Ptr& cloud,
        const rclcpp::Time& stamp,
        const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub)
    {
        sensor_msgs::msg::PointCloud2 out_msg;
        pcl::toROSMsg(*cloud, out_msg);

        out_msg.header.stamp = stamp;
        out_msg.header.frame_id = map_frame_;

        pub->publish(out_msg);
    }

    void printPoseAndDelta(const Eigen::Matrix4f& delta_T, double score)
    {
        float dx = delta_T(0, 3);
        float dy = delta_T(1, 3);
        float dz = delta_T(2, 3);
        float dyaw = std::atan2(delta_T(1, 0), delta_T(0, 0));

        float x = T_map_lidar_(0, 3);
        float y = T_map_lidar_(1, 3);
        float z = T_map_lidar_(2, 3);
        float yaw = std::atan2(T_map_lidar_(1, 0), T_map_lidar_(0, 0));

        RCLCPP_INFO(
            get_logger(),
            "used frame: %d | score: %.6f | delta: x %.3f y %.3f z %.3f yaw %.3f | global: x %.3f y %.3f z %.3f yaw %.3f | map points: %zu",
            used_frame_count_,
            score,
            dx, dy, dz, dyaw,
            x, y, z, yaw,
            map_cloud_->size()
        );
    }

    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        auto t_begin = std::chrono::steady_clock::now();

        frame_count_++;

        if (process_every_n_ > 1 && frame_count_ % process_every_n_ != 0)
        {
            return;
        }

        CloudT::Ptr cloud_raw(new CloudT);
        pcl::fromROSMsg(*msg, *cloud_raw);

        if (cloud_raw->empty())
        {
            RCLCPP_WARN(get_logger(), "Received empty cloud.");
            return;
        }

        CloudT::Ptr cloud_curr = preprocessCloud(cloud_raw);

        if (cloud_curr->empty())
        {
            RCLCPP_WARN(get_logger(), "Cloud empty after preprocessing.");
            return;
        }

        rclcpp::Time stamp(msg->header.stamp);

        if (!has_last_cloud_)
        {
            // 第一帧作为 map 坐标系原点
            T_map_lidar_ = Eigen::Matrix4f::Identity();

            CloudT::Ptr cloud_first_in_map(new CloudT);
            pcl::transformPointCloud(*cloud_curr, *cloud_first_in_map, T_map_lidar_);

            *map_cloud_ += *cloud_first_in_map;
            downsampleMap();

            *last_cloud_ = *cloud_curr;
            last_stamp_ = stamp;
            has_last_cloud_ = true;
            used_frame_count_ = 1;

            publishCloud(map_cloud_, stamp, map_pub_);
            publishCloud(cloud_first_in_map, stamp, registered_scan_pub_);

            pcl::io::savePCDFileBinary(save_dir_ + "/multi_frame_map.pcd", *map_cloud_);

            RCLCPP_INFO(
                get_logger(),
                "First frame inserted into map. raw points: %zu, down points: %zu",
                cloud_raw->size(),
                cloud_curr->size()
            );

            return;
        }

        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(cloud_curr);
        icp.setInputTarget(last_cloud_);

        icp.setMaxCorrespondenceDistance(max_corr_dist_);
        icp.setMaximumIterations(max_iter_);
        icp.setTransformationEpsilon(trans_eps_);
        icp.setEuclideanFitnessEpsilon(fitness_eps_);

        CloudT::Ptr cloud_curr_aligned_to_last(new CloudT);
        icp.align(*cloud_curr_aligned_to_last);

        if (!icp.hasConverged())
        {
            RCLCPP_WARN(get_logger(), "ICP did not converge. Skip this frame.");
            return;
        }

        double score = icp.getFitnessScore();

        if (score > max_fitness_score_)
        {
            RCLCPP_WARN(
                get_logger(),
                "ICP score too large: %.6f > %.6f. Skip this frame.",
                score,
                max_fitness_score_
            );
            return;
        }

        Eigen::Matrix4f delta_T = icp.getFinalTransformation();

        // 关键：
        // delta_T 表示 当前帧 LiDAR -> 上一帧 LiDAR
        // T_map_lidar_ 表示 当前 LiDAR -> map
        // 所以累积：
        T_map_lidar_ = T_map_lidar_ * delta_T;

        CloudT::Ptr cloud_curr_in_map(new CloudT);
        pcl::transformPointCloud(*cloud_curr, *cloud_curr_in_map, T_map_lidar_);

        *map_cloud_ += *cloud_curr_in_map;

        // 地图必须降采样，否则点数会越来越多，RViz 和节点都会越来越慢
        downsampleMap();

        used_frame_count_++;

        if (publish_every_n_ <= 1 || used_frame_count_ % publish_every_n_ == 0)
        {
            publishCloud(map_cloud_, stamp, map_pub_);
            publishCloud(cloud_curr_in_map, stamp, registered_scan_pub_);
        }

        // if (save_every_n_ > 0 && used_frame_count_ % save_every_n_ == 0)
        // {
        //     pcl::io::savePCDFileBinary(save_dir_ + "/multi_frame_map.pcd", *map_cloud_);
        //     RCLCPP_INFO(
        //         get_logger(),
        //         "Saved map to %s",
        //         (save_dir_ + "/multi_frame_map.pcd").c_str()
        //     );
        // }

        printPoseAndDelta(delta_T, score);

        // 更新上一帧
        *last_cloud_ = *cloud_curr;
        last_stamp_ = stamp;

        auto t_end = std::chrono::steady_clock::now();
        double cost_ms =
            std::chrono::duration<double, std::milli>(t_end - t_begin).count();

        RCLCPP_INFO_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,
            "Multi-frame processing cost: %.2f ms",
            cost_ms
        );
    }

private:
    std::string input_topic_;
    std::string map_topic_;
    std::string registered_scan_topic_;
    std::string map_frame_;
    std::string save_dir_;

    double scan_leaf_size_;
    double map_leaf_size_;
    double max_corr_dist_;
    int max_iter_;
    double trans_eps_;
    double fitness_eps_;
    double max_fitness_score_;

    int process_every_n_;
    int publish_every_n_;
    int save_every_n_;

    int frame_count_ = 0;
    int used_frame_count_ = 0;

    bool has_last_cloud_ = false;

    Eigen::Matrix4f T_map_lidar_;

    CloudT::Ptr last_cloud_;
    CloudT::Ptr map_cloud_;

    rclcpp::Time last_stamp_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr registered_scan_pub_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MultiFrameRegistrationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}