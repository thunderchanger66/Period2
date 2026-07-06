#include <rclcpp/rclcpp.hpp>

// ROS2 消息类型
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

// TF2 相关
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>
#include <tf2/time.h>

// PCL 与 ROS 桥接、点云类型以及 KD 树
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <mutex>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>

// 主节点类
class KDTreeNode : public rclcpp::Node
{
public:
    // 构造函数：声明参数、创建订阅/发布者、定时器以及 TF 监听器
    KDTreeNode() : Node("kd_tree_node") {
        // 声明并获取参数，提供默认值
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

        // TF 缓冲区和监听器，用于获取 base_frame 到 map_frame 的变换
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // 订阅全局点云地图（ voxel 地图 )
        map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            map_topic_,
            rclcpp::SensorDataQoS(),
            std::bind(&KDTreeNode::mapCallback, this, std::placeholders::_1)
        );
        // 发布局部地图（半径搜索得到的点云）
        local_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/local_map_kdtree",
            10
        );
        // 发布最近点云（KNN 搜索结果）
        nearest_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/nearest_points",
            10
        );

        // 创建定时器，定时执行查询回调
        double period_ms = 1000.0 / query_rate_hz_;
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(period_ms)),
            std::bind(&KDTreeNode::queryTimerCallback, this)
        );

        // 初始化地图指针
        map_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

        // 输出启动信息
        RCLCPP_INFO(this->get_logger(), "KD-tree query node started.");
        RCLCPP_INFO(this->get_logger(), "Map topic: %s", map_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Base frame: %s", base_frame_.c_str());
        RCLCPP_INFO(this->get_logger(), "Search radius: %.2f m", search_radius_);
        RCLCPP_INFO(this->get_logger(), "K neighbors: %d", k_neighbors_);
    };
private:
    // 回调函数声明：处理传入的点云地图、定时查询、发布点云
    void mapCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void queryTimerCallback();
    void publishCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
        const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub);

    // 参数成员
    std::string map_topic_;
    std::string map_frame_;
    std::string base_frame_;

    int k_neighbors_;
    double search_radius_;
    double query_rate_hz_;

    // 标志位：是否已经收到地图点云
    bool has_map_ = false;

    // 互斥锁，保护点云和 KD 树的并发访问
    std::mutex map_mutex_;

    // 全局地图点云（由订阅回调更新）
    pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud_;
    // KD 树结构，用于快速最近邻和半径搜索
    pcl::KdTreeFLANN<pcl::PointXYZI> kd_tree_;

    // ROS 订阅者和发布者、定时器
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nearest_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // TF 缓冲区和监听器（使用 unique_ptr/shared_ptr 管理生命周期）
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

// -----------------------------------------------------------------------------
// 回调函数实现
// -----------------------------------------------------------------------------

// 地图点云回调：接收新的点云，更新内部点云并重建 KD 树
void KDTreeNode::mapCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // 将 ROS PointCloud2 转换为 PCL PointCloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr new_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::fromROSMsg(*msg, *new_cloud);
    if (new_cloud->empty()) {
        // 如果收到空点云，输出警告并直接返回
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Received empty map cloud."
        );
        return;
    }

    {
        // 加锁以安全地更新共享资源
        std::lock_guard<std::mutex> lock(map_mutex_);
        map_cloud_ = new_cloud;                     // 替换为新点云
        kd_tree_.setInputCloud(map_cloud_);         // 重新构建 KD 树的输入
        if (!msg->header.frame_id.empty()) {
            map_frame_ = msg->header.frame_id;      // 更新参考坐标系（如有变化）
        }
        has_map_ = true;                            // 标记已有地图
    }

    // 输出日志（每秒最多一次），展示点云大小和所在坐标系
    RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "KD-tree rebuilt. Map points: %zu, frame: %s",
        new_cloud->size(),
        map_frame_.c_str()
    );
}

// 定时查询回调：根据当前车辆位置查询最近点和半径内点云，并发布结果
void KDTreeNode::queryTimerCallback() {
    // 若尚未收到地图点云，则等待
    if (!has_map_) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Waiting for map cloud..."
        );
        return;
    }

    // 尝试从 TF 获取 base_frame 到 map_frame 的变换
    geometry_msgs::msg::TransformStamped tf_map_base;
    try {
        tf_map_base = tf_buffer_->lookupTransform(
            map_frame_,       // 目标坐标系（地图）
            base_frame_,      // 源坐标系（车体）
            tf2::TimePointZero // 取最新可用的变换
        );
    } catch (const tf2::TransformException & ex) {
        // TF 查找失败时输出警告并退出本次回调
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

    // 将 TF 平移部分转换为 PCL 查询点（强制转为 float）
    pcl::PointXYZI query_point;
    query_point.x = static_cast<float>(tf_map_base.transform.translation.x);
    query_point.y = static_cast<float>(tf_map_base.transform.translation.y);
    query_point.z = static_cast<float>(tf_map_base.transform.translation.z);
    query_point.intensity = 0.0f; // 强度字段暂不用，设为 0

    // 用于存储最近邻搜索结果的索引和平方距离
    std::vector<int> knn_indices;
    std::vector<float> knn_squared_distances;
    // 用于存储半径搜索结果的索引和平方距离
    std::vector<int> radius_indices;
    std::vector<float> radius_squared_distances;

    // 创建用于发布的点云指针：最近点云和局部地图（半径搜索结果）
    pcl::PointCloud<pcl::PointXYZI>::Ptr nearest_cloud(
        new pcl::PointCloud<pcl::PointXYZI>);
    // local_cloud 用于存储在半径搜索范围内的点
    pcl::PointCloud<pcl::PointXYZI>::Ptr local_cloud(
        new pcl::PointCloud<pcl::PointXYZI>);

    // 搜索结果计数和最近点距离
    int found_knn = 0;
    int found_radius = 0;
    float nearest_distance = -1.0f;

    // 临界区：保护对共享点云和 KD 树的访问
    {
        std::lock_guard<std::mutex> lock(map_mutex_);

        // 再次检查地图是否有效（防止在此期间被置空）
        if (!map_cloud_ || map_cloud_->empty()) return;

        // ------- K 最近邻搜索 -------
        found_knn = kd_tree_.nearestKSearch(
            query_point,      // 查询点
            k_neighbors_,     // 最近邻点数目
            knn_indices,              // 输出：找到的点的索引
            knn_squared_distances     // 输出：对应的平方距离
        );

        if (found_knn > 0) {
            // 为最近点云预分配内存并填充点
            nearest_cloud->reserve(found_knn);
            for (int i = 0; i < found_knn; ++i)
                nearest_cloud->push_back(map_cloud_->points[knn_indices[i]]);

            // 最近点的实际欧氏距离（开方）
            nearest_distance = std::sqrt(knn_squared_distances[0]);
        }

        // ------- 半径搜索 -------
        found_radius = kd_tree_.radiusSearch(
            query_point,          // 查询点
            search_radius_,       // 搜索半径
            radius_indices,               // 输出：在半径内的点索引
            radius_squared_distances      // 输出：对应的平方距离
        );

        // 将半径搜索得到的点存入 local_cloud
        if (found_radius > 0) {
            local_cloud->reserve(found_radius);
            for (int idx : radius_indices)
                local_cloud->push_back(map_cloud_->points[idx]);
        }
    }   // 锁在此处自动释放

    // 发布最近点云和局部地图点云
    publishCloud(nearest_cloud, nearest_pub_);
    publishCloud(local_cloud, local_map_pub_);

    // 周期性打印查询结果（每秒最多一次），便于调试
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

// 点云发布辅助函数：将 PCL 点云转为 ROS 消息并发布
void KDTreeNode::publishCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub) {

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);   // PCL -> ROS 消息转换

    msg.header.stamp = this->now();        // 时间戳为当前 ROS 时间
    msg.header.frame_id = map_frame_;      // 坐标系使用地图框架

    pub->publish(msg);                     // 实际发布
}

// -----------------------------------------------------------------------------
// 主函数
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);                         // 初始化 ROS 2
    std::shared_ptr<KDTreeNode> node = std::make_shared<KDTreeNode>();
    rclcpp::spin(node);                               // 进入事件循环
    rclcpp::shutdown();                               // 关闭 ROS 2
    return 0;
}