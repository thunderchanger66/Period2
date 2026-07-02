#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/io.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <filesystem>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <deque>
#include <mutex>
#include <algorithm>

class ScanToMapNode : public rclcpp::Node
{
public:
    using PointT = pcl::PointXYZ;
    using CloudT = pcl::PointCloud<PointT>;

    ScanToMapNode()
    : Node("scan_to_map_node_optimized")
    {
        // 参数声明（新增转弯优化相关参数）
        declare_parameter<std::string>("input_topic", "/lidar/points_deskewed");
        declare_parameter<std::string>("imu_topic", "/imu");
        declare_parameter<std::string>("map_topic", "/scan_to_map_map");
        declare_parameter<std::string>("registered_scan_topic", "/scan_to_map_current_scan");
        declare_parameter<std::string>("map_frame", "map");
        declare_parameter<std::string>("save_dir", "results");

        declare_parameter<double>("scan_leaf_size", 0.25);
        declare_parameter<double>("map_leaf_size", 0.25);
        declare_parameter<double>("local_map_radius", 30.0);
        declare_parameter<double>("max_corr_dist", 0.8);
        declare_parameter<int>("max_iter", 30);
        declare_parameter<double>("trans_eps", 1e-6);
        declare_parameter<double>("fitness_eps", 1e-6);
        declare_parameter<double>("max_fitness_score", 1.5);
        declare_parameter<int>("process_every_n", 1);
        declare_parameter<int>("publish_every_n", 1);
        declare_parameter<int>("save_every_n", 50);
        declare_parameter<int>("min_target_points", 30);
        declare_parameter<int>("max_map_points", 80000);
        declare_parameter<bool>("enable_constant_velocity_fallback", true);
        declare_parameter<double>("imu_timeout", 0.2);
        // 转弯优化参数
        declare_parameter<double>("turn_angle_threshold", 0.15);      // 累积旋转 > 0.15 rad 视为转弯
        declare_parameter<double>("turn_radius_scale", 1.5);          // 转弯时局部地图半径放大倍数
        declare_parameter<double>("turn_corr_dist_scale", 1.2);       // 转弯时 max_corr_dist 放大倍数
        declare_parameter<int>("imu_init_frames", 100);               // 用于零偏估计的 IMU 帧数

        // 读取参数
        input_topic_ = get_parameter("input_topic").as_string();
        imu_topic_ = get_parameter("imu_topic").as_string();
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
        max_map_points_ = get_parameter("max_map_points").as_int();
        enable_constant_velocity_fallback_ = get_parameter("enable_constant_velocity_fallback").as_bool();
        imu_timeout_ = get_parameter("imu_timeout").as_double();

        turn_angle_threshold_ = get_parameter("turn_angle_threshold").as_double();
        turn_radius_scale_ = get_parameter("turn_radius_scale").as_double();
        turn_corr_dist_scale_ = get_parameter("turn_corr_dist_scale").as_double();
        imu_init_frames_ = get_parameter("imu_init_frames").as_int();

        std::filesystem::create_directories(save_dir_);

        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
        cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_, qos,
            std::bind(&ScanToMapNode::cloudCallback, this, std::placeholders::_1));
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, rclcpp::SensorDataQoS(),
            std::bind(&ScanToMapNode::imuCallback, this, std::placeholders::_1));

        map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(map_topic_, qos);
        registered_scan_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(registered_scan_topic_, qos);

        map_cloud_.reset(new CloudT);
        T_map_lidar_ = Eigen::Matrix4f::Identity();
        last_T_ = Eigen::Matrix4f::Identity();

        // IMU 零偏估计状态
        gyro_bias_.setZero();
        imu_bias_initialized_ = false;
        imu_init_counter_ = 0;

        RCLCPP_INFO(get_logger(), "Scan-to-map node (optimized) started.");
        RCLCPP_INFO(get_logger(), "Turn angle threshold: %.3f rad", turn_angle_threshold_);
    }

private:
    // -------------------- IMU 数据结构与处理 --------------------
    struct ImuData {
        double t;
        Eigen::Vector3d gyro;
    };
    std::deque<ImuData> imu_buffer_;
    std::mutex imu_mutex_;
    Eigen::Vector3d gyro_bias_;
    bool imu_bias_initialized_ = false;
    int imu_init_counter_ = 0;

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        ImuData data;
        data.t = rclcpp::Time(msg->header.stamp).seconds();
        data.gyro << msg->angular_velocity.x,
                     msg->angular_velocity.y,
                     msg->angular_velocity.z;

        // 零偏估计：前 imu_init_frames_ 帧静止（假设启动时静止）
        if (!imu_bias_initialized_ && imu_init_counter_ < imu_init_frames_) {
            gyro_bias_ += data.gyro;
            imu_init_counter_++;
            if (imu_init_counter_ == imu_init_frames_) {
                gyro_bias_ /= static_cast<double>(imu_init_frames_);
                imu_bias_initialized_ = true;
                RCLCPP_INFO(get_logger(), "IMU bias estimated: [%.4f, %.4f, %.4f]",
                            gyro_bias_.x(), gyro_bias_.y(), gyro_bias_.z());
            }
            return; // 零偏估计阶段不存入缓存，但为了简单也可存入，只是不参与积分
        }

        // 扣除零偏
        if (imu_bias_initialized_) {
            data.gyro -= gyro_bias_;
        }

        std::lock_guard<std::mutex> lock(imu_mutex_);
        imu_buffer_.push_back(data);
        if (imu_buffer_.size() > 10000) imu_buffer_.pop_front();
    }

    // 积分 IMU 旋转部分，返回旋转矩阵 (3x3)
    Eigen::Matrix3f integrateImuRotation(double t_start, double t_end)
    {
        Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
        if (t_end <= t_start) return R;

        std::lock_guard<std::mutex> lock(imu_mutex_);
        if (imu_buffer_.size() < 2) return R;

        auto it_start = std::lower_bound(imu_buffer_.begin(), imu_buffer_.end(), t_start,
            [](const ImuData& d, double t) { return d.t < t; });
        auto it_end = std::upper_bound(it_start, imu_buffer_.end(), t_end,
            [](double t, const ImuData& d) { return t < d.t; });

        if (it_start == imu_buffer_.end() || it_end == it_start) return R;

        for (auto it = it_start; it != it_end; ++it) {
            auto next = it + 1;
            if (next == imu_buffer_.end()) break;
            double dt = next->t - it->t;
            if (dt <= 0.0 || dt > 0.1) continue;
            Eigen::Vector3f omega = it->gyro.cast<float>();
            float angle = omega.norm() * dt;
            if (angle < 1e-6) continue;
            Eigen::AngleAxisf rot(angle, omega.normalized());
            R = R * rot.toRotationMatrix();
        }
        return R;
    }

    // -------------------- 点云处理工具 --------------------
    CloudT::Ptr voxelDownsample(const CloudT::Ptr& cloud_in, double leaf_size)
    {
        CloudT::Ptr cloud_out(new CloudT);
        if (cloud_in->empty()) return cloud_out;
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(cloud_in);
        voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
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

    void addCloudToMap(const CloudT::Ptr& cloud_in_map)
    {
        if (cloud_in_map->empty()) return;
        CloudT::Ptr cloud_down = voxelDownsample(cloud_in_map, map_leaf_size_);
        *map_cloud_ += *cloud_down;
        if (static_cast<int>(map_cloud_->size()) > max_map_points_) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                "Map too large (%zu), global downsample.", map_cloud_->size());
            map_cloud_ = voxelDownsample(map_cloud_, map_leaf_size_);
        }
    }

    CloudT::Ptr getLocalMap(const Eigen::Matrix4f& T_predict, double radius)
    {
        CloudT::Ptr local_map(new CloudT);
        if (map_cloud_->empty()) return local_map;

        pcl::search::KdTree<PointT> tree;
        tree.setInputCloud(map_cloud_);

        Eigen::Vector4f center = T_predict * Eigen::Vector4f(0, 0, 0, 1);
        PointT search_point{center[0], center[1], center[2]};

        std::vector<int> indices;
        std::vector<float> distances;
        int num_found = tree.radiusSearch(search_point, radius, indices, distances);

        if (num_found > 0)
            pcl::copyPointCloud(*map_cloud_, indices, *local_map);

        if (static_cast<int>(local_map->size()) < min_target_points_) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
                "Local map too small (%zu), use full map.", local_map->size());
            return map_cloud_;
        }
        return local_map;
    }

    void publishCloud(const CloudT::Ptr& cloud, const rclcpp::Time& stamp,
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
        float x = T_map_lidar_(0,3), y = T_map_lidar_(1,3), z = T_map_lidar_(2,3);
        float yaw = std::atan2(T_map_lidar_(1,0), T_map_lidar_(0,0));
        RCLCPP_INFO(get_logger(), "frame %d | score %.6f | pose: x %.3f y %.3f z %.3f yaw %.3f | map %zu",
                    used_frame_count_, score, x, y, z, yaw, map_cloud_->size());
    }

    // -------------------- 预测函数（含转弯检测） --------------------
    Eigen::Matrix4f predictPose(double current_time, double& turn_angle_accum)
    {
        turn_angle_accum = 0.0;
        Eigen::Matrix4f T_pred = T_map_lidar_;

        if (has_last_cloud_time_) {
            // 用 IMU 积分旋转
            Eigen::Matrix3f R_delta = integrateImuRotation(last_cloud_time_, current_time);
            // 计算累积旋转角
            Eigen::AngleAxisf aa(R_delta);
            turn_angle_accum = aa.angle();

            if (R_delta.isIdentity()) {
                // IMU 积分失败，用常速度模型
                if (enable_constant_velocity_fallback_ && has_previous_pose_) {
                    Eigen::Matrix4f delta = last_T_.inverse() * T_map_lidar_;
                    T_pred = T_map_lidar_ * delta;
                    // 常速度无法给出旋转角度，设为零
                    turn_angle_accum = 0.0;
                }
            } else {
                // 成功积分，构造 4x4 变换（平移不变）
                Eigen::Matrix4f delta_T = Eigen::Matrix4f::Identity();
                delta_T.block<3,3>(0,0) = R_delta;
                T_pred = T_map_lidar_ * delta_T;
            }
        }

        return T_pred;
    }

    // ======================== 主回调 ========================
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        auto t_begin = std::chrono::steady_clock::now();
        frame_count_++;
        if (process_every_n_ > 1 && frame_count_ % process_every_n_ != 0) return;

        // 1. 加载与预处理
        CloudT::Ptr cloud_raw(new CloudT);
        pcl::fromROSMsg(*msg, *cloud_raw);
        if (cloud_raw->empty()) { RCLCPP_WARN(get_logger(), "Empty cloud."); return; }

        CloudT::Ptr cloud_curr = preprocessScan(cloud_raw);
        if (cloud_curr->empty()) { RCLCPP_WARN(get_logger(), "Cloud empty after preprocess."); return; }

        double current_time = rclcpp::Time(msg->header.stamp).seconds();
        rclcpp::Time stamp(msg->header.stamp);

        // 2. 第一帧初始化
        if (!has_map_) {
            T_map_lidar_ = Eigen::Matrix4f::Identity();
            CloudT::Ptr cloud_first_in_map(new CloudT);
            pcl::transformPointCloud(*cloud_curr, *cloud_first_in_map, T_map_lidar_);
            addCloudToMap(cloud_first_in_map);
            has_map_ = true;
            used_frame_count_ = 1;
            has_previous_pose_ = false;
            has_last_cloud_time_ = true;
            last_cloud_time_ = current_time;

            publishCloud(map_cloud_, stamp, map_pub_);
            publishCloud(cloud_first_in_map, stamp, registered_scan_pub_);
            pcl::io::savePCDFileBinary(save_dir_ + "/scan_to_map_map.pcd", *map_cloud_);
            RCLCPP_INFO(get_logger(), "First frame inserted.");
            return;
        }

        // 3. 预测位姿，并获取累积旋转角（用于转弯检测）
        double turn_angle = 0.0;
        Eigen::Matrix4f T_predict = predictPose(current_time, turn_angle);

        // 动态调整局部地图半径和 ICP max_corr_dist
        double radius_scale = 1.0;
        double corr_scale = 1.0;
        bool is_turning = turn_angle > turn_angle_threshold_;
        if (is_turning) {
            radius_scale = turn_radius_scale_;
            corr_scale = turn_corr_dist_scale_;
            RCLCPP_DEBUG(get_logger(), "Turning detected, angle=%.3f rad, scale radius=%.2f, corr=%.2f",
                         turn_angle, radius_scale, corr_scale);
        }

        double local_radius = local_map_radius_ * radius_scale;
        double max_corr_dist = max_corr_dist_ * corr_scale;

        // 4. 提取局部地图
        CloudT::Ptr local_map = getLocalMap(T_predict, local_radius);
        if (static_cast<int>(local_map->size()) < min_target_points_) {
            RCLCPP_WARN(get_logger(), "Target map too small. Skip.");
            return;
        }

        // 5. ICP 配准
        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(cloud_curr);
        icp.setInputTarget(local_map);
        icp.setMaxCorrespondenceDistance(max_corr_dist);
        icp.setMaximumIterations(max_iter_);
        icp.setTransformationEpsilon(trans_eps_);
        icp.setEuclideanFitnessEpsilon(fitness_eps_);

        CloudT::Ptr cloud_curr_in_map(new CloudT);
        icp.align(*cloud_curr_in_map, T_predict);

        if (!icp.hasConverged()) {
            RCLCPP_WARN(get_logger(), "ICP did not converge. Skip.");
            return;
        }

        double score = icp.getFitnessScore();
        if (score > max_fitness_score_) {
            RCLCPP_WARN(get_logger(), "ICP score too high: %.6f > %.6f. Skip.", score, max_fitness_score_);
            return;
        }

        // 6. 校验变换合理性
        Eigen::Matrix4f T_new = icp.getFinalTransformation();
        Eigen::Matrix4f delta = T_map_lidar_.inverse() * T_new;
        float dx = delta(0,3), dy = delta(1,3), dz = delta(2,3);
        float dyaw = std::atan2(delta(1,0), delta(0,0));
        if (std::abs(dx) > 3.0 || std::abs(dy) > 3.0 || std::abs(dz) > 2.0 || std::abs(dyaw) > 0.5) {
            RCLCPP_WARN(get_logger(), "Abnormal transform (dx=%.2f dy=%.2f dyaw=%.3f). Skip.",
                        dx, dy, dyaw);
            return;
        }

        // 7. 更新状态
        last_T_ = T_map_lidar_;
        T_map_lidar_ = T_new;
        has_previous_pose_ = true;
        has_last_cloud_time_ = true;
        last_cloud_time_ = current_time;

        addCloudToMap(cloud_curr_in_map);
        used_frame_count_++;

        // 8. 发布
        if (publish_every_n_ <= 1 || used_frame_count_ % publish_every_n_ == 0) {
            publishCloud(map_cloud_, stamp, map_pub_);
            publishCloud(cloud_curr_in_map, stamp, registered_scan_pub_);
        }

        if (save_every_n_ > 0 && used_frame_count_ % save_every_n_ == 0) {
            pcl::io::savePCDFileBinary(save_dir_ + "/scan_to_map_map.pcd", *map_cloud_);
            RCLCPP_INFO(get_logger(), "Map saved.");
        }

        printPose(score);
        auto t_end = std::chrono::steady_clock::now();
        double cost_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Cost: %.2f ms", cost_ms);
    }

private:
    // 参数
    std::string input_topic_, imu_topic_, map_topic_, registered_scan_topic_, map_frame_, save_dir_;
    double scan_leaf_size_, map_leaf_size_, local_map_radius_;
    double max_corr_dist_, trans_eps_, fitness_eps_, max_fitness_score_;
    int max_iter_, process_every_n_, publish_every_n_, save_every_n_, min_target_points_;
    int max_map_points_;
    bool enable_constant_velocity_fallback_;
    double imu_timeout_;
    double turn_angle_threshold_, turn_radius_scale_, turn_corr_dist_scale_;
    int imu_init_frames_;

    // 状态
    int frame_count_ = 0, used_frame_count_ = 0;
    bool has_map_ = false;
    bool has_previous_pose_ = false;
    bool has_last_cloud_time_ = false;
    double last_cloud_time_ = 0.0;
    Eigen::Matrix4f T_map_lidar_;
    Eigen::Matrix4f last_T_;
    CloudT::Ptr map_cloud_;

    // ROS
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
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