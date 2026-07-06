#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>

#include <unordered_map>
#include <cmath>
#include <limits>
#include <string>

struct VoxelKey
{
    int ix, iy, iz;
    bool operator==(const VoxelKey& other) const {
        return ix == other.ix && iy ==  other.iy && iz == other.iz;
    }
};

struct VoxelKeyHash
{
    std::size_t operator()(const VoxelKey& key) const {
        std::size_t h1 = std::hash<int>()(key.ix);
        std::size_t h2 = std::hash<int>()(key.iy);
        std::size_t h3 = std::hash<int>()(key.iz);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct VoxelData
{
    int count = 0;
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
};

struct GridKey
{
    int ix, iy;

    bool operator==(const GridKey & other) const
    {
        return ix == other.ix && iy == other.iy;
    }
};

struct GridKeyHash
{
    std::size_t operator()(const GridKey & key) const {
        std::size_t h1 = std::hash<int>()(key.ix);
        std::size_t h2 = std::hash<int>()(key.iy);
        return h1 ^ (h2 << 1);
    }
};

class VoxelMapNode : public rclcpp::Node
{
public:
    VoxelMapNode() : Node("voxel_map_node") {
        input_cloud_topic_ = this->declare_parameter<std::string>(
            "input_cloud_topic", "/global_cloud_map"
        );
        voxel_size_ = this->declare_parameter<double>(
            "voxel_size", 0.2
        );
        grid_resolution_ = this->declare_parameter<double>(
            "grid_resolution", 0.2
        );
        z_min_ = this->declare_parameter<double>(
            "z_min", -1.0
        );
        z_max_ = this->declare_parameter<double>(
            "z_max", 2.0
        );
        map_frame_ = this->declare_parameter<std::string>(
            "map_frame", "world"
        );
        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_cloud_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(&VoxelMapNode::cloudCallback, this, std::placeholders::_1)
        );
        voxel_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/voxel_cloud",
            10
        );
        grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "/grid_map",
            10
        );
        RCLCPP_INFO(this->get_logger(), "Map representation node started.");
        RCLCPP_INFO(this->get_logger(), "Input cloud: %s", input_cloud_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Voxel size: %.3f", voxel_size_);
        RCLCPP_INFO(this->get_logger(), "Grid resolution: %.3f", grid_resolution_);
    }
private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void buildVoxelMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);
    void publishVoxelCloud(const rclcpp::Time& stamp);
    void buildAndPublishGridMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
        const rclcpp::Time& stamp);

    std::string input_cloud_topic_;
    std::string map_frame_;

    double voxel_size_;
    double grid_resolution_;
    double z_min_;
    double z_max_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr voxel_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;

    std::unordered_map<VoxelKey, VoxelData, VoxelKeyHash> voxel_map_;
};

void VoxelMapNode::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    std::shared_ptr<pcl::PointCloud<pcl::PointXYZI>> cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty()) {
        RCLCPP_WARN(this->get_logger(), "Input cloud is empty.");
        return;
    }

    if (!msg->header.frame_id.empty()) {
        map_frame_ = msg->header.frame_id;
    }

    buildVoxelMap(cloud);
    publishVoxelCloud(msg->header.stamp);
    buildAndPublishGridMap(cloud, msg->header.stamp);
}

void VoxelMapNode::buildVoxelMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    voxel_map_.clear();

    for (const auto & p : cloud->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }

        VoxelKey key;
        key.ix = static_cast<int>(std::floor(p.x / voxel_size_));
        key.iy = static_cast<int>(std::floor(p.y / voxel_size_));
        key.iz = static_cast<int>(std::floor(p.z / voxel_size_));

        auto & voxel = voxel_map_[key];
        voxel.count++;
        voxel.sum += Eigen::Vector3d(p.x, p.y, p.z);
    }

    RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "Raw points: %zu, occupied voxels: %zu",
        cloud->size(),
        voxel_map_.size()
    );
}

void VoxelMapNode::publishVoxelCloud(const rclcpp::Time& stamp) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr voxel_cloud(
        new pcl::PointCloud<pcl::PointXYZI>
    );

    voxel_cloud->reserve(voxel_map_.size());

    for (const auto & item : voxel_map_) {
        const VoxelKey & key = item.first;
        const VoxelData & data = item.second;

        pcl::PointXYZI p;

        // 使用体素中心作为代表点
        // p.x = static_cast<float>((key.ix + 0.5) * voxel_size_);
        // p.y = static_cast<float>((key.iy + 0.5) * voxel_size_);
        // p.z = static_cast<float>((key.iz + 0.5) * voxel_size_);
        p.x = static_cast<float>(data.sum.x() / data.count);
        p.y = static_cast<float>(data.sum.y() / data.count);
        p.z = static_cast<float>(data.sum.z() / data.count);

        // intensity 表示这个体素里的点数
        p.intensity = static_cast<float>(data.count);

        voxel_cloud->push_back(p);
    }

    sensor_msgs::msg::PointCloud2 out_msg;
    pcl::toROSMsg(*voxel_cloud, out_msg);

    out_msg.header.stamp = stamp;
    out_msg.header.frame_id = map_frame_;

    voxel_pub_->publish(out_msg);
}

void VoxelMapNode::buildAndPublishGridMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const rclcpp::Time& stamp) {

    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();

    std::unordered_map<GridKey, int, GridKeyHash> grid_count;

    for (const auto & p : cloud->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }

        // 高度过滤
        if (p.z < z_min_ || p.z > z_max_) {
            continue;
        }

        min_x = std::min(min_x, static_cast<double>(p.x));
        min_y = std::min(min_y, static_cast<double>(p.y));
        max_x = std::max(max_x, static_cast<double>(p.x));
        max_y = std::max(max_y, static_cast<double>(p.y));
    }

    if (min_x > max_x || min_y > max_y) {
        RCLCPP_WARN(this->get_logger(), "No valid points for grid map.");
        return;
    }

    // 稍微扩大边界，避免点刚好落在边界外
    double padding = 1.0;
    min_x -= padding;
    min_y -= padding;
    max_x += padding;
    max_y += padding;

    int width = static_cast<int>(std::ceil((max_x - min_x) / grid_resolution_));
    int height = static_cast<int>(std::ceil((max_y - min_y) / grid_resolution_));

    if (width <= 0 || height <= 0) {
        return;
    }

    nav_msgs::msg::OccupancyGrid grid_msg;
    grid_msg.header.stamp = stamp;
    grid_msg.header.frame_id = map_frame_;

    grid_msg.info.resolution = static_cast<float>(grid_resolution_);
    grid_msg.info.width = width;
    grid_msg.info.height = height;

    grid_msg.info.origin.position.x = min_x;
    grid_msg.info.origin.position.y = min_y;
    grid_msg.info.origin.position.z = 0.0;
    grid_msg.info.origin.orientation.w = 1.0;

    // 默认未知
    grid_msg.data.assign(width * height, -1);

    for (const auto & p : cloud->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }

        if (p.z < z_min_ || p.z > z_max_) {
            continue;
        }

        int ix = static_cast<int>(std::floor((p.x - min_x) / grid_resolution_));
        int iy = static_cast<int>(std::floor((p.y - min_y) / grid_resolution_));

        if (ix < 0 || iy < 0 || ix >= width || iy >= height) {
            continue;
        }

        int index = iy * width + ix;

        // 有点的位置认为是占据
        grid_msg.data[index] = 100;

        GridKey key{ix, iy};
        grid_count[key]++;
    }

    grid_pub_->publish(grid_msg);

    RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "Grid map: width=%d, height=%d, occupied cells=%zu",
        width,
        height,
        grid_count.size()
    );
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<VoxelMapNode> node = std::make_shared<VoxelMapNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}