#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <geometry_msgs/msg/transform.hpp>

class TFBroadcaster : public rclcpp::Node
{
public:
    TFBroadcaster() : Node("tf_broadcaster") {
         // 声明参数：固定坐标系和子坐标系名称
        this->declare_parameter<std::string>("odom_frame", "odom");
        this->declare_parameter<std::string>("base_frame", "vehicle_blue/chassis");
        this->declare_parameter<std::string>("pose_topic", "/model/vehicle_blue/pose");

        odom_frame_ = this->get_parameter("odom_frame").as_string();
        base_frame_ = this->get_parameter("base_frame").as_string();
        std::string pose_topic = this->get_parameter("pose_topic").as_string();

        // 创建 TF 广播器
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        // 订阅 PoseStamped（由 ros_gz_bridge 桥接）
        sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            pose_topic,
            rclcpp::SensorDataQoS(),
            std::bind(&TFBroadcaster::poseCallback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), "TF Broadcaster started. Odom: %s, Base: %s", 
                    odom_frame_.c_str(), base_frame_.c_str());
    }

private:
    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

    std::string odom_frame_;
    std::string base_frame_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_;
};

void TFBroadcaster::poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    geometry_msgs::msg::TransformStamped tf;

    // 设置时间戳（使用仿真时间）
    tf.header.stamp = msg->header.stamp;
    tf.header.frame_id = odom_frame_;          // 父坐标系
    tf.child_frame_id = base_frame_;           // 子坐标系
    // 平移
    tf.transform.translation.x = msg->pose.position.x;
    tf.transform.translation.y = msg->pose.position.y;
    tf.transform.translation.z = msg->pose.position.z;

    // 旋转
    tf.transform.rotation = msg->pose.orientation;

    // 广播 TF
    tf_broadcaster_->sendTransform(tf);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<TFBroadcaster> node = std::make_shared<TFBroadcaster>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}