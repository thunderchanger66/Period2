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
#include <pcl/kdtree/kdtree_flann.h>

#include <mutex>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>

class KDTreeNode : public rclcpp::Node
{
public:
    KDTreeNode() : Node("kd_tree_node") {
        map_topic_ = this->declare_parameter<std::string>(
            "map_topic", "/voxel_cloud");
        map_frame_ = this->declare_parameter<std::string>(
            "map_frame", "world");
        base_frame_ = this->declare_parameter<std::string>(
            "base_frame", "vehicle_blue/chassis");
        k_neighbors_ = this->declare_parameter<int>(
            "k_neighbors", 5);
        search_radius_ = this->declare_parameter<double>(
            "search_radius", 3.0);
        query_rate_hz_ = this->declare_parameter<double>(
            "query_rate_hz", 5.0);
        
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            map_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(&KDTreeNode::mapCallback, this, std::placeholders::_1)
        );
        local_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/local_map_kdtree",
            10
        );
        nearest_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/nearest_points",
            10
        );

        double period_ms = 1000.0 / query_rate_hz_;
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(period_ms)),
            std::bind(&KDTreeNode::queryTimerCallback, this)
        );

        map_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

        RCLCPP_INFO(this->get_logger(), "KD-tree query node started.");
        RCLCPP_INFO(this->get_logger(), "Map topic: %s", map_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Base frame: %s", base_frame_.c_str());
        RCLCPP_INFO(this->get_logger(), "Search radius: %.2f m", search_radius_);
        RCLCPP_INFO(this->get_logger(), "K neighbors: %d", k_neighbors_);
    };
private:
    void mapCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void queryTimerCallback();
    void publishCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
        const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub);

    std::string map_topic_;
    std::string map_frame_;
    std::string base_frame_;

    int k_neighbors_;
    double search_radius_;
    double query_rate_hz_;

    bool has_map_ = false;

    std::mutex map_mutex_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud_;
    pcl::KdTreeFLANN<pcl::PointXYZI> kd_tree_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nearest_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

void KDTreeNode::mapCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr new_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::fromROSMsg(*msg, *new_cloud);
    if (new_cloud->empty()) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Received empty map cloud."
        );
        return;
    }

    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        map_cloud_ = new_cloud;
        kd_tree_.setInputCloud(map_cloud_);
        if (!msg->header.frame_id.empty()) {
                map_frame_ = msg->header.frame_id;
            }
        has_map_ = true;
    }

    RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "KD-tree rebuilt. Map points: %zu, frame: %s",
        new_cloud->size(),
        map_frame_.c_str()
    );
}

void KDTreeNode::queryTimerCallback() {
    if (!has_map_) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Waiting for map cloud..."
        );
        return;
    }

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

    pcl::PointXYZI query_point;
    query_point.x = static_cast<float>(tf_map_base.transform.translation.x);
    query_point.y = static_cast<float>(tf_map_base.transform.translation.y);
    query_point.z = static_cast<float>(tf_map_base.transform.translation.z);
    query_point.intensity = 0.0f;

    std::vector<int> knn_indices;
    std::vector<float> knn_squared_distances;
    std::vector<int> radius_indices;
    std::vector<float> radius_squared_distances;

    pcl::PointCloud<pcl::PointXYZI>::Ptr nearest_cloud(
        new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr local_cloud(
        new pcl::PointCloud<pcl::PointXYZI>);

    int found_knn = 0;
    int found_radius = 0;
    float nearest_distance = -1.0f;

    {
        std::lock_guard<std::mutex> lock(map_mutex_);

        if (!map_cloud_ || map_cloud_->empty()) return;

        found_knn = kd_tree_.nearestKSearch(
            query_point,
            k_neighbors_,
            knn_indices,
            knn_squared_distances
        );

        if (found_knn > 0) {
            nearest_cloud->reserve(found_knn);

            for (int i = 0; i < found_knn; ++i)
                nearest_cloud->push_back(map_cloud_->points[knn_indices[i]]);

            nearest_distance = std::sqrt(knn_squared_distances[0]);
        }

        found_radius = kd_tree_.radiusSearch(
            query_point,
            search_radius_,
            radius_indices,
            radius_squared_distances
        );

        if (found_radius > 0) {
            local_cloud->reserve(found_radius);
            for (int idx : radius_indices)
                local_cloud->push_back(map_cloud_->points[idx]);
        }
    }

    publishCloud(nearest_cloud, nearest_pub_);
    publishCloud(local_cloud, local_map_pub_);

    RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "Query position: [%.2f, %.2f, %.2f], nearest=%d, nearest_dist=%.3f, radius_points=%d",
        query_point.x,
        query_point.y,
        query_point.z,
        found_knn,
        nearest_distance,
        found_radius
    );
}

void KDTreeNode::publishCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub) {

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);

    msg.header.stamp = this->now();
    msg.header.frame_id = map_frame_;

    pub->publish(msg);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<KDTreeNode> node = std::make_shared<KDTreeNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}