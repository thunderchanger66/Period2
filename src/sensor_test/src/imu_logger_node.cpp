#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <fstream>
#include <iomanip>
#include <cmath>

/**
 * @class ImuLoggerNode
 * @brief ROS2 节点，用于订阅 IMU 话题，记录数据到 CSV 文件，并利用角速度积分实时估计姿态。
 *
 * 本节点实现了以下功能：
 * 1. 订阅 sensor_msgs/Imu 消息，获取角速度、线加速度和姿态四元数。
 * 2. 将原始数据及计算得到的欧拉角、范数等记录到 CSV 文件。
 * 3. 基于角速度进行递推积分（航位推算），更新从 IMU 坐标系到世界坐标系的旋转四元数。
 *
 * 注意：此处的姿态积分仅使用陀螺仪数据，未融合加速度计或磁力计，因此会随时间漂移。
 */
class ImuLoggerNode : public rclcpp::Node
{
public:
    ImuLoggerNode()
    : Node("imu_logger_node")
    {
        // 1. 声明并读取 ROS2 参数
        this->declare_parameter<std::string>("imu_topic", "/imu");
        this->declare_parameter<std::string>("output_csv", "imu_processed.csv");

        std::string imu_topic = this->get_parameter("imu_topic").as_string();
        std::string output_csv = this->get_parameter("output_csv").as_string();

        // 2. 打开 CSV 文件用于写入
        file_.open(output_csv);
        if (!file_.is_open())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open output csv: %s", output_csv.c_str());
            rclcpp::shutdown();
            return;
        }

        // 写入 CSV 表头（各列含义见注释）
        file_ << "t,"                 // 时间戳 (秒)
              << "qx,qy,qz,qw,"       // IMU 原始姿态四元数（来自消息）
              << "gx,gy,gz,gyro_norm,"// 角速度三轴及范数
              << "ax,ay,az,acc_norm," // 线加速度三轴及范数
              << "roll,pitch,yaw\n";  // 由积分姿态计算出的欧拉角（ZYX 顺序）

        // 3. 初始化姿态四元数（单位四元数，表示无旋转）
        q_WI_ = Eigen::Quaterniond::Identity();
        has_last_time_ = false;       // 尚未收到第一帧数据

        // 4. 创建 IMU 订阅者（使用传感器数据 QoS）
        sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic,
            rclcpp::SensorDataQoS(),
            std::bind(&ImuLoggerNode::imuCallback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), "Subscribe IMU topic: %s", imu_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Output CSV: %s", output_csv.c_str());
    }

    ~ImuLoggerNode()
    {
        if (file_.is_open())
        {
            file_.close();
        }
    }

private:
    /**
     * @brief 将旋转向量 (so3) 转换为旋转矩阵 (SO3)
     * @param phi 旋转向量（角度轴表示，方向为旋转轴，模长为旋转角度）
     * @return 对应的 3x3 旋转矩阵
     *
     * 使用指数映射 exp(phi^) = I + sinθ/θ * phi^ + (1-cosθ)/θ^2 * (phi^)^2
     * 这里通过 Eigen::AngleAxisd 实现，与李代数指数映射等价。
     */
    Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& phi)
    {
        double angle = phi.norm();

        if (angle < 1e-12)   // 避免除以零
        {
            return Eigen::Matrix3d::Identity();
        }

        Eigen::Vector3d axis = phi / angle;
        return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    }

    /**
     * @brief IMU 消息回调函数
     * 每收到一帧 IMU 数据，执行：
     * - 提取时间戳、角速度、线加速度
     * - 利用上一帧角速度与时间差进行姿态积分（陀螺积分）
     * - 将原始数据和计算出的欧拉角写入 CSV 文件
     */
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        double t = rclcpp::Time(msg->header.stamp).seconds();

        Eigen::Vector3d gyro(
            msg->angular_velocity.x,
            msg->angular_velocity.y,
            msg->angular_velocity.z
        );

        Eigen::Vector3d acc(
            msg->linear_acceleration.x,
            msg->linear_acceleration.y,
            msg->linear_acceleration.z
        );

        // 3. 姿态积分（使用上一帧角速度和两帧之间的时间差）
        if (has_last_time_)
        {
            double dt = t - last_time_;

            // 只处理合理的时间间隔（避免异常跳变）
            if (dt > 0.0 && dt < 1.0)
            {
                // 假设在 dt 内角速度恒定，计算角度增量
                Eigen::Vector3d delta_theta = last_gyro_ * dt;

                // 将角度增量转换为旋转矩阵（指数映射）
                Eigen::Matrix3d delta_R = ExpSO3(delta_theta);

                // 更新姿态：q_WI_ = q_WI_ * delta_q
                // 含义：从上一时刻 IMU 姿态继续旋转 delta_R
                q_WI_ = q_WI_ * Eigen::Quaterniond(delta_R);
                q_WI_.normalize();   // 防止数值漂移导致四元数非单位
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Abnormal dt: %.9f", dt);
            }
        }

        // 4. 更新“上一帧”信息供下次使用
        has_last_time_ = true;
        last_time_ = t;
        last_gyro_ = gyro;

        // 5. 将当前估计的姿态（四元数）转换为旋转矩阵，再提取欧拉角
        // 注意：这里的 q_WI_ 是积分得到的，而 msg->orientation 是 IMU 硬件给出的原始姿态（如果有）
        // 我们同时记录原始四元数和积分得到的欧拉角，用于对比。
        Eigen::Matrix3d R_WI = q_WI_.toRotationMatrix();

        // Eigen 的 eulerAngles(2,1,0) 表示 ZYX 顺序（即先绕 Z 轴 yaw，再 Y 轴 pitch，最后 X 轴 roll）
        Eigen::Vector3d ypr = R_WI.eulerAngles(2, 1, 0);

        double yaw   = ypr[0];
        double pitch = ypr[1];
        double roll  = ypr[2];

        // 6. 计算角速度和加速度的范数（用于分析）
        double gyro_norm = gyro.norm();
        double acc_norm  = acc.norm();

        // 7. 将所有数据写入 CSV 文件
        file_ << std::fixed << std::setprecision(9)
              << t << ","
              << msg->orientation.x << ","
              << msg->orientation.y << ","
              << msg->orientation.z << ","
              << msg->orientation.w << ","
              << gyro.x() << ","
              << gyro.y() << ","
              << gyro.z() << ","
              << gyro_norm << ","
              << acc.x() << ","
              << acc.y() << ","
              << acc.z() << ","
              << acc_norm << ","
              << roll << ","
              << pitch << ","
              << yaw << "\n";
    }

private:
    // ROS2 订阅器
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;

    // 输出文件流
    std::ofstream file_;

    // 用于姿态积分的状态变量
    bool has_last_time_;       // 是否已收到上一帧数据
    double last_time_;         // 上一帧的时间戳
    Eigen::Vector3d last_gyro_; // 上一帧的角速度

    // 姿态四元数：表示从 IMU 坐标系 (I) 到世界坐标系 (W) 的旋转
    // 即：将 IMU 坐标系下的向量旋转到世界坐标系下。
    // 初始化时为单位四元数（无旋转）
    Eigen::Quaterniond q_WI_;
};

/**
 * @brief 主函数
 * 初始化 ROS2，创建 ImuLoggerNode 节点，并进入自旋循环。
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ImuLoggerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}