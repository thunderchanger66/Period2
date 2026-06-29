#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <deque>
#include <mutex>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <algorithm>

/*
订阅 /imu
    ↓
缓存 IMU 角速度

订阅 /lidar/points
    ↓
读取 PointCloud2
    ↓
估计每个点采样时间
    ↓
用 IMU gyro 积分相对旋转
    ↓
旋转补偿每个点
    ↓
发布 /lidar/points_deskewed
    ↓
保存 raw_cloud.pcd 和 deskewed_cloud.pcd
*/
struct ImuData
{
    double t;
    Eigen::Vector3d gyro;
};

struct RotData
{
    double t;
    Eigen::Matrix3d R;
};

class DeskewTestNode : public rclcpp::Node
{
public:
    using PointT = pcl::PointXYZ;
    using CloudT = pcl::PointCloud<PointT>;

    DeskewTestNode() : Node("DeskewTestNode") {
        declare_parameter<std::string>("imu_topic", "/imu");
        declare_parameter<std::string>("cloud_topic", "/lidar/points");
        declare_parameter<std::string>("deskewed_topic", "/lidar/points_deskewed");

        declare_parameter<double>("scan_period", 0.1);
        declare_parameter<double>("gyro_bias_x", 0.0);
        declare_parameter<double>("gyro_bias_y", 0.0);
        declare_parameter<double>("gyro_bias_z", 0.0);
        declare_parameter<double>("time_offset_imu_to_lidar", 0.0);

        // 如果补偿后点云更差，可以把这个参数设为 true 试一次
        declare_parameter<bool>("use_inverse_rotation", false);

        declare_parameter<std::string>("save_dir", "results");
        declare_parameter<bool>("save_once", true);

        // ---------- 读取参数 ----------
        imu_topic_ = get_parameter("imu_topic").as_string();
        cloud_topic_ = get_parameter("cloud_topic").as_string();
        deskewed_topic_ = get_parameter("deskewed_topic").as_string();

        scan_period_ = get_parameter("scan_period").as_double();

        gyro_bias_ = Eigen::Vector3d(
            get_parameter("gyro_bias_x").as_double(),
            get_parameter("gyro_bias_y").as_double(),
            get_parameter("gyro_bias_z").as_double()
        );

        time_offset_imu_to_lidar_ =
            get_parameter("time_offset_imu_to_lidar").as_double();

        use_inverse_rotation_ =
            get_parameter("use_inverse_rotation").as_bool();

        save_dir_ = get_parameter("save_dir").as_string();
        save_once_ = get_parameter("save_once").as_bool();

        std::filesystem::create_directories(save_dir_);

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(&DeskewTestNode::imuCallback, this, std::placeholders::_1)
        );

        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(&DeskewTestNode::cloudCallback, this, std::placeholders::_1)
        );

        deskewed_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            deskewed_topic_,
            rclcpp::SensorDataQoS()
        );

        RCLCPP_INFO(get_logger(), "IMU topic:      %s", imu_topic_.c_str());
        RCLCPP_INFO(get_logger(), "Cloud topic:    %s", cloud_topic_.c_str());
        RCLCPP_INFO(get_logger(), "Deskewed topic: %s", deskewed_topic_.c_str());
        RCLCPP_INFO(get_logger(), "Scan period:    %.6f s", scan_period_);
    }

private:
    std::vector<RotData> buildRotationTable(double t_start, double t_end)
    {
        std::deque<ImuData> imu_copy;

        {
            std::lock_guard<std::mutex> lock(imu_mutex_);
            imu_copy = imu_buffer_;
        }

        std::vector<RotData> rot_table;

        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

        rot_table.push_back({t_start, R});

        if (imu_copy.size() < 2)
        {
            rot_table.push_back({t_end, R});
            return rot_table;
        }

        for (size_t i = 1; i < imu_copy.size(); ++i)
        {
            double ta = imu_copy[i - 1].t;
            double tb = imu_copy[i].t;

            if (tb < t_start) continue;

            if (ta > t_end) break;

            double seg_start = std::max(ta, t_start);
            double seg_end = std::min(tb, t_end);
            double dt = seg_end - seg_start;

            if (dt <= 0.0) continue;

            Eigen::Vector3d omega = imu_copy[i - 1].gyro - gyro_bias_;
            Eigen::Vector3d delta_theta = omega * dt;

            R = R * ExpSO3(delta_theta);

            rot_table.push_back({seg_end, R});
        }

        if (rot_table.back().t < t_end) rot_table.push_back({t_end, R});

        return rot_table;
    }

    Eigen::Matrix3d getRotationAtTime(
        const std::vector<RotData>& rot_table,
        double t,
        size_t& idx)
    {
        if (rot_table.empty()) return Eigen::Matrix3d::Identity();

        if (t <= rot_table.front().t) return rot_table.front().R;

        while (idx + 1 < rot_table.size() && rot_table[idx + 1].t < t) idx++;

        if (idx + 1 >= rot_table.size()) return rot_table.back().R;

        const RotData& a = rot_table[idx];
        const RotData& b = rot_table[idx + 1];

        double dt = b.t - a.t;

        if (dt <= 1e-9) return a.R;

        double alpha = (t - a.t) / dt;
        alpha = std::clamp(alpha, 0.0, 1.0);

        Eigen::Quaterniond qa(a.R);
        Eigen::Quaterniond qb(b.R);

        Eigen::Quaterniond q = qa.slerp(alpha, qb);
        q.normalize();

        return q.toRotationMatrix();
    }

    /**
     * @brief 指数映射：旋转向量 → 旋转矩阵
     * @param phi 旋转向量（轴角表示），方向为旋转轴，模长为旋转角度
     * @return 对应的 3x3 旋转矩阵
     */
    Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& phi) {
        double angle = phi.norm();
        if (angle < 1e-12) return Eigen::Matrix3d::Identity();
        Eigen::Vector3d axis = phi / angle;
        return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    }
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    std::string imu_topic_;
    std::string cloud_topic_;
    std::string deskewed_topic_;
    std::string save_dir_;

    double scan_period_;                  // 一帧扫描时长（秒）
    double time_offset_imu_to_lidar_;     // IMU 时间戳相对于 LiDAR 的偏移

    Eigen::Vector3d gyro_bias_;           // 陀螺仪零偏（静止时角速度偏移）

    bool use_inverse_rotation_;           // 是否使用反向旋转补偿
    bool save_once_;                      // 是否只保存第一帧
    bool saved_ = false;                  // 保存状态标记

    std::mutex imu_mutex_;                // 保护 IMU 缓存的互斥锁
    std::deque<ImuData> imu_buffer_;      // IMU 数据环形缓存

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_pub_;

    /**
     * @brief 检查 IMU 缓存是否覆盖指定的时间区间
     */
    bool hasImuCoverage(double t_start, double t_end) {
        std::lock_guard<std::mutex> lock(imu_mutex_);

        if (imu_buffer_.size() < 2) return false;

        double imu_start = imu_buffer_.front().t;
        double imu_end = imu_buffer_.back().t;

        return imu_start <= t_start && imu_end >= t_end;
    }

    /**
     * @brief 对一帧点云进行去畸变处理
     * @param cloud_in        输入点云（已去除 NaN）
     * @param cloud_start_time 帧起始时刻（秒）
     * @return 去畸变后的点云（所有点被旋转到帧起始时刻的坐标系）
     */
    CloudT::Ptr deskewCloudByImu(
        const CloudT::Ptr& cloud_in,
        double cloud_start_time)
    {
        CloudT::Ptr cloud_out(new CloudT);
        cloud_out->reserve(cloud_in->size());

        size_t N = cloud_in->size();

        if (N == 0) return cloud_out;

        double cloud_end_time = cloud_start_time + scan_period_;

        // 关键：一帧点云只构建一次旋转表
        std::vector<RotData> rot_table =
            buildRotationTable(cloud_start_time, cloud_end_time);

        size_t rot_idx = 0;

        for (size_t i = 0; i < N; ++i)
        {
            const PointT& p = cloud_in->points[i];

            if (!std::isfinite(p.x) ||
                !std::isfinite(p.y) ||
                !std::isfinite(p.z)) continue;

            double ratio = 0.0;

            if (N > 1) ratio = static_cast<double>(i) / static_cast<double>(N - 1);

            double point_time = cloud_start_time + ratio * scan_period_;

            Eigen::Matrix3d R_0i =
                getRotationAtTime(rot_table, point_time, rot_idx);

            Eigen::Vector3d p_i(p.x, p.y, p.z);

            Eigen::Vector3d p_deskew;

            if (use_inverse_rotation_)
            {
                p_deskew = R_0i.transpose() * p_i;
            }
            else
            {
                p_deskew = R_0i * p_i;
            }

            PointT q;
            q.x = static_cast<float>(p_deskew.x());
            q.y = static_cast<float>(p_deskew.y());
            q.z = static_cast<float>(p_deskew.z());

            cloud_out->points.push_back(q);
        }

        cloud_out->width = cloud_out->points.size();
        cloud_out->height = 1;
        cloud_out->is_dense = false;

        return cloud_out;
    }
};

/**
 * @brief IMU 订阅回调：将 IMU 数据存入缓存
 */
void DeskewTestNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    double t = rclcpp::Time(msg->header.stamp).seconds();
    if (t <= 0.0) t = this->get_clock()->now().seconds();

    ImuData data;
    data.t = t;
    data.gyro = Eigen::Vector3d(
        msg->angular_velocity.x,
        msg->angular_velocity.y,
        msg->angular_velocity.z
    );

    std::lock_guard<std::mutex> lock(imu_mutex_);

    imu_buffer_.push_back(data);
    // 只保留最近 2 秒 IMU 数据
    while (!imu_buffer_.empty() && data.t - imu_buffer_.front().t > 2.0)
        imu_buffer_.pop_front();
}

void DeskewTestNode::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    double cloud_time = rclcpp::Time(msg->header.stamp).seconds();

    if (cloud_time <= 0.0) cloud_time = this->get_clock()->now().seconds();

    // 应用时间偏移（如果 IMU 和 LiDAR 时间戳不同步）
    double cloud_start_time = cloud_time + time_offset_imu_to_lidar_;
    double cloud_end_time = cloud_start_time - scan_period_;

    // 检查 IMU 数据是否覆盖该帧范围
    if (!hasImuCoverage(cloud_start_time, cloud_end_time)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "IMU buffer does not cover lidar scan. Need [%.6f, %.6f]",
            cloud_start_time, cloud_end_time
        );
        return;
    }

    // 转换 ROS 点云 → PCL 点云
    CloudT::Ptr cloud_raw(new CloudT);
    pcl::fromROSMsg(*msg, *cloud_raw);

    if (cloud_raw->empty()) {
        RCLCPP_WARN(get_logger(), "Received empty point cloud.");
        return;
    }

    // 去除 NaN 点
    CloudT::Ptr cloud_no_nan(new CloudT);
    std::vector<int> indices;   // 用于存储有效点索引（此处未使用）
    pcl::removeNaNFromPointCloud(*cloud_raw, *cloud_no_nan, indices);

    // 执行去畸变
    CloudT::Ptr cloud_deskewed = deskewCloudByImu(cloud_no_nan, cloud_start_time);

    // 转换回 ROS 消息并发布
    sensor_msgs::msg::PointCloud2 out_msg;
    pcl::toROSMsg(*cloud_deskewed, out_msg);
    out_msg.header = msg->header;   // 保留原始消息头
    out_msg.header.stamp = msg->header.stamp; // 时间戳不变

    deskewed_pub_->publish(out_msg);

    // 打印点云数量变化
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
        "raw: %zu, no_nan: %zu, deskewed: %zu",
        cloud_raw->size(), cloud_no_nan->size(), cloud_deskewed->size()
    );

    // 保存点云（仅第一帧或每次）
    if (!saved_ || !save_once_) {
        std::string raw_path = save_dir_ + "/raw_cloud.pcd";
        std::string deskewed_path = save_dir_ + "/deskewed_cloud.pcd";

        pcl::io::savePCDFileBinary(raw_path, *cloud_no_nan);
        pcl::io::savePCDFileBinary(deskewed_path, *cloud_deskewed);

        RCLCPP_INFO(get_logger(), "Saved: %s", raw_path.c_str());
        RCLCPP_INFO(get_logger(), "Saved: %s", deskewed_path.c_str());

        saved_ = true;
    }
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<DeskewTestNode> node = std::make_shared<DeskewTestNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}