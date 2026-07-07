#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>
#include <tf2/time.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>

#include <mutex>
#include <string>
#include <chrono>

class LocalMapCropNode : public rclcpp::Node
{
public:
    LocalMapCropNode() : Node("local_map_crop_node")
    {
        input_map_topic_ = this->declare_parameter<std::string>(
            "input_map_topic", "/voxel_cloud"
        );

        map_frame_ = this->declare_parameter<std::string>(
            "map_frame", "world"
        );

        base_frame_ = this->declare_parameter<std::string>(
            "base_frame", "vehicle_blue/chassis"
        );

        crop_mode_ = this->declare_parameter<std::string>(
            "crop_mode", "pose"
        );

        length_x_ = this->declare_parameter<double>(
            "length_x", 10.0
        );

        length_y_ = this->declare_parameter<double>(
            "length_y", 10.0
        );

        length_z_ = this->declare_parameter<double>(
            "length_z", 4.0
        );

        fixed_x_min_ = this->declare_parameter<double>("fixed_x_min", -5.0);
        fixed_x_max_ = this->declare_parameter<double>("fixed_x_max", 5.0);
        fixed_y_min_ = this->declare_parameter<double>("fixed_y_min", -5.0);
        fixed_y_max_ = this->declare_parameter<double>("fixed_y_max", 5.0);
        fixed_z_min_ = this->declare_parameter<double>("fixed_z_min", -2.0);
        fixed_z_max_ = this->declare_parameter<double>("fixed_z_max", 2.0);

        query_rate_hz_ = this->declare_parameter<double>(
            "query_rate_hz", 5.0
        );

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_map_topic_,
            10,
            std::bind(&LocalMapCropNode::mapCallback, this, std::placeholders::_1)
        );

        local_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/local_map_crop",
            10
        );

        double period_ms = 1000.0 / query_rate_hz_;
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(period_ms)),
            std::bind(&LocalMapCropNode::timerCallback, this)
        );

        global_map_.reset(new pcl::PointCloud<pcl::PointXYZI>);

        RCLCPP_INFO(this->get_logger(), "Local map crop node started.");
        RCLCPP_INFO(this->get_logger(), "Input map topic: %s", input_map_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Crop mode: %s", crop_mode_.c_str());
    }

private:
    void mapCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZI>
        );

        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty()) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "Received empty map."
            );
            return;
        }

        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            global_map_ = cloud;

            if (!msg->header.frame_id.empty()) {
                map_frame_ = msg->header.frame_id;
            }

            has_map_ = true;
        }

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Received map: points=%zu, frame=%s",
            cloud->size(),
            map_frame_.c_str()
        );
    }

    void timerCallback()
    {
        if (!has_map_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "Waiting for input map..."
            );
            return;
        }

        double x_min, x_max;
        double y_min, y_max;
        double z_min, z_max;

        if (crop_mode_ == "pose") {
            geometry_msgs::msg::TransformStamped tf_map_base;

            try {
                tf_map_base = tf_buffer_->lookupTransform(
                    map_frame_,
                    base_frame_,
                    tf2::TimePointZero
                );
            } catch (const tf2::TransformException & ex) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(),
                    *this->get_clock(),
                    1000,
                    "TF lookup failed: %s -> %s, reason: %s",
                    base_frame_.c_str(),
                    map_frame_.c_str(),
                    ex.what()
                );
                return;
            }

            double cx = tf_map_base.transform.translation.x;
            double cy = tf_map_base.transform.translation.y;
            double cz = tf_map_base.transform.translation.z;

            x_min = cx - length_x_ * 0.5;
            x_max = cx + length_x_ * 0.5;

            y_min = cy - length_y_ * 0.5;
            y_max = cy + length_y_ * 0.5;

            z_min = cz - length_z_ * 0.5;
            z_max = cz + length_z_ * 0.5;
        } else {
            x_min = fixed_x_min_;
            x_max = fixed_x_max_;
            y_min = fixed_y_min_;
            y_max = fixed_y_max_;
            z_min = fixed_z_min_;
            z_max = fixed_z_max_;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr local_map(
            new pcl::PointCloud<pcl::PointXYZI>
        );

        pcl::PointCloud<pcl::PointXYZI>::Ptr map_copy(
            new pcl::PointCloud<pcl::PointXYZI>
        );

        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            *map_copy = *global_map_;
        }

        pcl::CropBox<pcl::PointXYZI> crop;
        crop.setInputCloud(map_copy);

        crop.setMin(Eigen::Vector4f(
            static_cast<float>(x_min),
            static_cast<float>(y_min),
            static_cast<float>(z_min),
            1.0f
        ));

        crop.setMax(Eigen::Vector4f(
            static_cast<float>(x_max),
            static_cast<float>(y_max),
            static_cast<float>(z_max),
            1.0f
        ));

        crop.filter(*local_map);

        publishLocalMap(local_map);

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Crop box: x[%.2f, %.2f], y[%.2f, %.2f], z[%.2f, %.2f], global=%zu, local=%zu",
            x_min, x_max,
            y_min, y_max,
            z_min, z_max,
            map_copy->size(),
            local_map->size()
        );
    }

    void publishLocalMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud)
    {
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);

        msg.header.stamp = this->now();
        msg.header.frame_id = map_frame_;

        local_map_pub_->publish(msg);
    }

private:
    std::string input_map_topic_;
    std::string map_frame_;
    std::string base_frame_;
    std::string crop_mode_;

    double length_x_;
    double length_y_;
    double length_z_;

    double fixed_x_min_;
    double fixed_x_max_;
    double fixed_y_min_;
    double fixed_y_max_;
    double fixed_z_min_;
    double fixed_z_max_;

    double query_rate_hz_;

    bool has_map_ = false;

    std::mutex map_mutex_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr global_map_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LocalMapCropNode>());
    rclcpp::shutdown();
    return 0;
}