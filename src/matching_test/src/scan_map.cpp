#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>

#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <Eigen/Dense>

#include <filesystem>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

class ScanToMapNode : public rclcpp::Node
{
public:
    using PointT = pcl::PointXYZ;
    using CloudT = pcl::PointCloud<PointT>;

    ScanToMapNode()
    : Node("scan_to_map_node")
    {
        declare_parameter<std::string>("input_topic", "/lidar/points_deskewed");
        declare_parameter<std::string>("map_topic", "/scan_to_map_map");
        declare_parameter<std::string>("registered_scan_topic", "/scan_to_map_current_scan");
        declare_parameter<std::string>("map_frame", "map");
        declare_parameter<std::string>("save_dir", "results");

        declare_parameter<double>("scan_leaf_size", 0.25);
        declare_parameter<double>("map_leaf_size", 0.25);
        declare_parameter<double>("local_map_radius", 20.0);

        declare_parameter<double>("max_corr_dist", 1.0);
        declare_parameter<int>("max_iter", 30);
        declare_parameter<double>("trans_eps", 1e-6);
        declare_parameter<double>("fitness_eps", 1e-6);
        declare_parameter<double>("max_fitness_score", 2.0);

        declare_parameter<int>("process_every_n", 1);
        declare_parameter<int>("publish_every_n", 1);
        declare_parameter<int>("save_every_n", 50);
        declare_parameter<int>("min_target_points", 30);

        input_topic_ = get_parameter("input_topic").as_string();
        map_topic_ = get_parameter("map_topic").as_string();
        registered_scan_topic_ = get_parameter("registered_scan_topic").as_string();
        map_frame_ = get_parameter("map_frame").as_string();
        save_dir_ = get_parameter("save_dir").as_string();

        scan_leaf_size_ = get_parameter("scan_leaf_size").as_double();
        map_leaf_size_ = get_parameter("map_leaf_size").as_double();
        local_map_radius_ = get_parameter("local_map_radius").as_double();

        max_corr_dist_ = get_parameter("max_corr_dist").as_double();
        max_iter_ = get_parameter("max_iter").as_int();
        trans_eps_ = get_parameter("trans_eps").as_double();
        fitness_eps_ = get_parameter("fitness_eps").as_double();
        max_fitness_score_ = get_parameter("max_fitness_score").as_double();

        process_every_n_ = get_parameter("process_every_n").as_int();
        publish_every_n_ = get_parameter("publish_every_n").as_int();
        save_every_n_ = get_parameter("save_every_n").as_int();
        min_target_points_ = get_parameter("min_target_points").as_int();

        std::filesystem::create_directories(save_dir_);

        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();

        cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_,
            qos,
            std::bind(&ScanToMapNode::cloudCallback, this, std::placeholders::_1)
        );

        map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            map_topic_,
            qos
        );

        registered_scan_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            registered_scan_topic_,
            qos
        );

        map_cloud_.reset(new CloudT);
        T_map_lidar_ = Eigen::Matrix4f::Identity();

        RCLCPP_INFO(get_logger(), "Scan-to-map node started.");
        RCLCPP_INFO(get_logger(), "Input: %s", input_topic_.c_str());
        RCLCPP_INFO(get_logger(), "Map topic: %s", map_topic_.c_str());
        RCLCPP_INFO(get_logger(), "Map frame: %s", map_frame_.c_str());
    }

private:
    CloudT::Ptr voxelDownsample(const CloudT::Ptr& cloud_in, double leaf_size)
    {
        CloudT::Ptr cloud_out(new CloudT);

        if (cloud_in->empty())
        {
            return cloud_out;
        }

        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(cloud_in);
        voxel.setLeafSize(
            static_cast<float>(leaf_size),
            static_cast<float>(leaf_size),
            static_cast<float>(leaf_size)
        );
        voxel.filter(*cloud_out);

        return cloud_out;
    }

    CloudT::Ptr preprocessScan(const CloudT::Ptr& cloud_in)
    {
        CloudT::Ptr cloud_no_nan(new CloudT);

        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud_in, *cloud_no_nan, indices);

        return voxelDownsample(cloud_no_nan, scan_leaf_size_);
    }

    void downsampleMap()
    {
        map_cloud_ = voxelDownsample(map_cloud_, map_leaf_size_);
    }

    CloudT::Ptr getLocalMap(const Eigen::Matrix4f& T_predict)
    {
        CloudT::Ptr local_map(new CloudT);

        if (map_cloud_->empty())
        {
            return local_map;
        }

        float cx = T_predict(0, 3);
        float cy = T_predict(1, 3);
        float cz = T_predict(2, 3);

        pcl::CropBox<PointT> crop;
        crop.setInputCloud(map_cloud_);
        crop.setMin(Eigen::Vector4f(
            cx - static_cast<float>(local_map_radius_),
            cy - static_cast<float>(local_map_radius_),
            cz - static_cast<float>(local_map_radius_),
            1.0f
        ));
        crop.setMax(Eigen::Vector4f(
            cx + static_cast<float>(local_map_radius_),
            cy + static_cast<float>(local_map_radius_),
            cz + static_cast<float>(local_map_radius_),
            1.0f
        ));

        crop.filter(*local_map);

        if (static_cast<int>(local_map->size()) < min_target_points_)
        {
            RCLCPP_WARN(
                get_logger(),
                "Local map too small: %zu points. Use full map instead.",
                local_map->size()
            );

            return map_cloud_;
        }

        return local_map;
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

    void printPose(double score)
    {
        float x = T_map_lidar_(0, 3);
        float y = T_map_lidar_(1, 3);
        float z = T_map_lidar_(2, 3);
        float yaw = std::atan2(T_map_lidar_(1, 0), T_map_lidar_(0, 0));

        RCLCPP_INFO(
            get_logger(),
            "frame: %d | score: %.6f | pose in map: x %.3f y %.3f z %.3f yaw %.3f | map points: %zu",
            used_frame_count_,
            score,
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

        CloudT::Ptr cloud_curr = preprocessScan(cloud_raw);

        if (cloud_curr->empty())
        {
            RCLCPP_WARN(get_logger(), "Current cloud empty after preprocessing.");
            return;
        }

        rclcpp::Time stamp(msg->header.stamp);

        if (!has_map_)
        {
            // 第一帧：定义 map 坐标系
            T_map_lidar_ = Eigen::Matrix4f::Identity();

            CloudT::Ptr cloud_first_in_map(new CloudT);
            pcl::transformPointCloud(*cloud_curr, *cloud_first_in_map, T_map_lidar_);

            *map_cloud_ += *cloud_first_in_map;
            downsampleMap();

            has_map_ = true;
            used_frame_count_ = 1;

            publishCloud(map_cloud_, stamp, map_pub_);
            publishCloud(cloud_first_in_map, stamp, registered_scan_pub_);

            pcl::io::savePCDFileBinary(save_dir_ + "/scan_to_map_map.pcd", *map_cloud_);

            RCLCPP_INFO(
                get_logger(),
                "First frame inserted. raw: %zu, down: %zu, map: %zu",
                cloud_raw->size(),
                cloud_curr->size(),
                map_cloud_->size()
            );

            return;
        }

        // 预测位姿：第一版先用上一帧 scan-to-map 的结果作为当前初值
        Eigen::Matrix4f T_predict = T_map_lidar_;

        CloudT::Ptr local_map = getLocalMap(T_predict);

        if (static_cast<int>(local_map->size()) < min_target_points_)
        {
            RCLCPP_WARN(
                get_logger(),
                "Target map has too few points: %zu. Skip frame.",
                local_map->size()
            );
            return;
        }

        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(cloud_curr);
        icp.setInputTarget(local_map);

        icp.setMaxCorrespondenceDistance(max_corr_dist_);
        icp.setMaximumIterations(max_iter_);
        icp.setTransformationEpsilon(trans_eps_);
        icp.setEuclideanFitnessEpsilon(fitness_eps_);

        CloudT::Ptr cloud_curr_in_map(new CloudT);

        // 关键：
        // cloud_curr 是 LiDAR 坐标系
        // local_map 是 map 坐标系
        // T_predict 是 LiDAR -> map 的初值
        icp.align(*cloud_curr_in_map, T_predict);

        if (!icp.hasConverged())
        {
            RCLCPP_WARN(get_logger(), "Scan-to-map ICP did not converge. Skip frame.");
            return;
        }

        double score = icp.getFitnessScore();

        if (score > max_fitness_score_)
        {
            RCLCPP_WARN(
                get_logger(),
                "ICP score too large: %.6f > %.6f. Skip frame.",
                score,
                max_fitness_score_
            );
            return;
        }

        // 关键：
        // scan-to-map 的最终结果就是 当前 LiDAR -> map
        // 不要再和上一帧累乘
        T_map_lidar_ = icp.getFinalTransformation();

        // 为了确保使用最终 T，再变换一次当前帧
        cloud_curr_in_map.reset(new CloudT);
        pcl::transformPointCloud(*cloud_curr, *cloud_curr_in_map, T_map_lidar_);

        *map_cloud_ += *cloud_curr_in_map;
        downsampleMap();

        used_frame_count_++;

        if (publish_every_n_ <= 1 || used_frame_count_ % publish_every_n_ == 0)
        {
            publishCloud(map_cloud_, stamp, map_pub_);
            publishCloud(cloud_curr_in_map, stamp, registered_scan_pub_);
        }

        // if (save_every_n_ > 0 && used_frame_count_ % save_every_n_ == 0)
        // {
        //     pcl::io::savePCDFileBinary(save_dir_ + "/scan_to_map_map.pcd", *map_cloud_);

        //     RCLCPP_INFO(
        //         get_logger(),
        //         "Saved map to %s",
        //         (save_dir_ + "/scan_to_map_map.pcd").c_str()
        //     );
        // }

        printPose(score);

        auto t_end = std::chrono::steady_clock::now();
        double cost_ms =
            std::chrono::duration<double, std::milli>(t_end - t_begin).count();

        RCLCPP_INFO_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,
            "Scan-to-map processing cost: %.2f ms",
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
    double local_map_radius_;

    double max_corr_dist_;
    int max_iter_;
    double trans_eps_;
    double fitness_eps_;
    double max_fitness_score_;

    int process_every_n_;
    int publish_every_n_;
    int save_every_n_;
    int min_target_points_;

    int frame_count_ = 0;
    int used_frame_count_ = 0;

    bool has_map_ = false;

    Eigen::Matrix4f T_map_lidar_;

    CloudT::Ptr map_cloud_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr registered_scan_pub_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ScanToMapNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}