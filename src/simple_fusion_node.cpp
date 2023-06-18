#include <limits>
#include <memory>
#include <string>

#include "message_filters/pass_through.h"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/sync_policies/approximate_epsilon_time.h"
#include "message_filters/sync_policies/exact_time.h"
#include "simple_fusion/msg/signal.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/create_timer_ros.h"
#include "tf2_ros/message_filter.h"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace simple_fusion
{
using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::PointCloud2;
using simple_fusion::msg::Signal;

class SimpleFusionNode : public rclcpp::Node
{
public:
  explicit SimpleFusionNode(const rclcpp::NodeOptions & options);

private:
  message_filters::PassThrough<Image> pass_through_;
  rclcpp::Publisher<Signal>::SharedPtr lidar_signal_pub_;
  std::unique_ptr<message_filters::Subscriber<PointCloud2>> point_cloud_sub_;
  std::unique_ptr<message_filters::Subscriber<Signal>> lidar_signal_sub_;
  std::unique_ptr<tf2_ros::MessageFilter<PointCloud2>> point_cloud_filter_;
  using ApproximateEpsilonImageSyncPolicy = message_filters::sync_policies::ApproximateEpsilonTime<
    Image, Image, Image, Image, Image, Image, Image, Image, Image>;
  using ApproximateEpsilonImageSync =
    message_filters::Synchronizer<ApproximateEpsilonImageSyncPolicy>;
  std::unique_ptr<ApproximateEpsilonImageSync> image_sync_;
  using ApproximateEpsilonFusionSyncPolicy = message_filters::sync_policies::ApproximateEpsilonTime<
    Signal, Signal>;
  using ApproximateEpsilonFusionSync =
    message_filters::Synchronizer<ApproximateEpsilonFusionSyncPolicy>;
  std::unique_ptr<ApproximateEpsilonFusionSync> fusion_sync_;

  std::vector<CameraInfo> camera_infos_;
  std::vector<rclcpp::Subscription<CameraInfo>::SharedPtr> camera_info_subs_;
  std::vector<std::unique_ptr<message_filters::Subscriber<Image>>> image_subs_;
  rclcpp::Publisher<Signal>::SharedPtr camera_signal_pub_;
  std::unique_ptr<message_filters::Subscriber<Signal>> camera_signal_sub_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_{std::make_unique<tf2_ros::Buffer>(get_clock())};
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_{std::make_unique<tf2_ros::TransformListener>(
      *tf_buffer_)};

  void onCameraInfo(
    const CameraInfo::ConstSharedPtr camera_info,
    const size_t camera_id)
  {
    camera_infos_.at(camera_id) = *camera_info;
  }

  void onImage(
    const Image::ConstSharedPtr & image0,
    const Image::ConstSharedPtr & image1,
    const Image::ConstSharedPtr & image2,
    const Image::ConstSharedPtr & image3,
    const Image::ConstSharedPtr & image4,
    const Image::ConstSharedPtr & image5,
    const Image::ConstSharedPtr & image6,
    const Image::ConstSharedPtr & image7,
    const Image::ConstSharedPtr & image8);

  void onPointCloud(
    const PointCloud2::ConstSharedPtr point_cloud)
  {
    Signal signal;
    signal.header = point_cloud->header;
    lidar_signal_pub_->publish(signal);
  }

  void onMultiSensorSignal(
    const simple_fusion::msg::Signal::ConstSharedPtr & lidar_signal,
    const simple_fusion::msg::Signal::ConstSharedPtr & camera_signal);

  void onDummy(Image::ConstSharedPtr input)
  {
    pass_through_.add(input);
  }
};

SimpleFusionNode::SimpleFusionNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("simple_fusion_node", options)
{
  using namespace std::placeholders;

  size_t camera_number = static_cast<size_t>(declare_parameter("camera_number", 1));
  if (camera_number < 1) {
    RCLCPP_WARN(
      this->get_logger(), "minimum camera_number is 1. current camera_number is %zu",
      camera_number);
    camera_number = 1;
  }
  if (camera_number > 9) {
    RCLCPP_WARN(
      this->get_logger(), "maximum camera_number is 9. current camera_number is %zu",
      camera_number);
    camera_number = 9;
  }

  auto sub_opts = rclcpp::SubscriptionOptions();
  sub_opts.qos_overriding_options = rclcpp::QosOverridingOptions::with_default_policies();
  int queue_size = declare_parameter("queue_size", 5);
  std::string base_frame = declare_parameter("base_frame", "base_link");

  lidar_signal_pub_ = create_publisher<Signal>(
    "~/input/lidar/signal", rclcpp::SensorDataQoS());
  point_cloud_sub_ =
    std::make_unique<message_filters::Subscriber<PointCloud2>>(
    this, "~/input/point_cloud", rmw_qos_profile_sensor_data, sub_opts);
  point_cloud_filter_ =
    std::make_unique<tf2_ros::MessageFilter<PointCloud2>>(
    *point_cloud_sub_, *tf_buffer_, base_frame, queue_size,
    get_node_logging_interface(), get_node_clock_interface(),
    tf2::durationFromSec(1.0));
  point_cloud_filter_->registerCallback(
    std::bind(&SimpleFusionNode::onPointCloud, this, std::placeholders::_1));

  lidar_signal_sub_ =
    std::make_unique<message_filters::Subscriber<Signal>>(
    this, "~/input/lidar/signal", rmw_qos_profile_sensor_data, sub_opts);

  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
    get_node_base_interface(),
    get_node_timers_interface());
  tf_buffer_->setCreateTimerInterface(timer_interface);

  camera_signal_pub_ = create_publisher<Signal>(
    "~/input/camera/signal", rclcpp::SensorDataQoS());
  camera_signal_sub_ =
    std::make_unique<message_filters::Subscriber<Signal>>(
    this, "~/input/camera/signal", rmw_qos_profile_sensor_data, sub_opts);
  camera_infos_.resize(camera_number);
  camera_info_subs_.resize(camera_number);
  image_subs_.resize(camera_number);
  for (size_t i = 0; i < camera_number; ++i) {
    std::function<void(const CameraInfo::ConstSharedPtr msg)> camera_info_func = std::bind(
      &SimpleFusionNode::onCameraInfo, this, std::placeholders::_1, i);
    camera_info_subs_.at(i) = create_subscription<CameraInfo>(
      "~/input/camera_info" + std::to_string(i), rclcpp::SensorDataQoS(), camera_info_func,
      sub_opts);
    image_subs_.at(i) =
      std::make_unique<message_filters::Subscriber<Image>>(
      this, "~/input/image" + std::to_string(i), rmw_qos_profile_sensor_data, sub_opts);
  }

  double approx_sync_epsilon = declare_parameter("approximate_sync_tolerance_seconds", 0.0);

  image_subs_.at(0)->registerCallback(
    std::bind(&SimpleFusionNode::onDummy, this, std::placeholders::_1));
  switch (camera_number) {
    case 1:
      image_sync_.reset(
        new ApproximateEpsilonImageSync(
          ApproximateEpsilonImageSyncPolicy(
            queue_size, rclcpp::Duration::from_seconds(approx_sync_epsilon)),
          *image_subs_.at(0), pass_through_, pass_through_, pass_through_, pass_through_,
          pass_through_, pass_through_, pass_through_, pass_through_));
      break;
    case 2:
      image_sync_.reset(
        new ApproximateEpsilonImageSync(
          ApproximateEpsilonImageSyncPolicy(
            queue_size, rclcpp::Duration::from_seconds(approx_sync_epsilon)),
          *image_subs_.at(0), *image_subs_.at(1), pass_through_, pass_through_, pass_through_,
          pass_through_, pass_through_, pass_through_, pass_through_));
      break;
    case 3:
      image_sync_.reset(
        new ApproximateEpsilonImageSync(
          ApproximateEpsilonImageSyncPolicy(
            queue_size, rclcpp::Duration::from_seconds(approx_sync_epsilon)),
          *image_subs_.at(0), *image_subs_.at(1), *image_subs_.at(2), pass_through_,
          pass_through_, pass_through_, pass_through_, pass_through_, pass_through_));
      break;
    case 4:
      image_sync_.reset(
        new ApproximateEpsilonImageSync(
          ApproximateEpsilonImageSyncPolicy(
            queue_size, rclcpp::Duration::from_seconds(approx_sync_epsilon)),
          *image_subs_.at(0), *image_subs_.at(1), *image_subs_.at(2), *image_subs_.at(3),
          pass_through_, pass_through_, pass_through_, pass_through_, pass_through_));
      break;
    case 5:
      image_sync_.reset(
        new ApproximateEpsilonImageSync(
          ApproximateEpsilonImageSyncPolicy(
            queue_size, rclcpp::Duration::from_seconds(approx_sync_epsilon)),
          *image_subs_.at(0), *image_subs_.at(1), *image_subs_.at(2), *image_subs_.at(3),
          *image_subs_.at(4), pass_through_, pass_through_, pass_through_, pass_through_));
      break;
    case 6:
      image_sync_.reset(
        new ApproximateEpsilonImageSync(
          ApproximateEpsilonImageSyncPolicy(
            queue_size, rclcpp::Duration::from_seconds(approx_sync_epsilon)),
          *image_subs_.at(0), *image_subs_.at(1), *image_subs_.at(2), *image_subs_.at(3),
          *image_subs_.at(4), *image_subs_.at(5), pass_through_, pass_through_, pass_through_));
      break;
    case 7:
      image_sync_.reset(
        new ApproximateEpsilonImageSync(
          ApproximateEpsilonImageSyncPolicy(
            queue_size, rclcpp::Duration::from_seconds(approx_sync_epsilon)),
          *image_subs_.at(0), *image_subs_.at(1), *image_subs_.at(2), *image_subs_.at(3),
          *image_subs_.at(4), *image_subs_.at(5), *image_subs_.at(6), pass_through_, pass_through_));
      break;
    case 8:
      image_sync_.reset(
        new ApproximateEpsilonImageSync(
          ApproximateEpsilonImageSyncPolicy(
            queue_size, rclcpp::Duration::from_seconds(approx_sync_epsilon)),
          *image_subs_.at(0), *image_subs_.at(1), *image_subs_.at(2), *image_subs_.at(3),
          *image_subs_.at(4), *image_subs_.at(5), *image_subs_.at(6), *image_subs_.at(7),
          pass_through_));
      break;
    case 9:
      image_sync_.reset(
        new ApproximateEpsilonImageSync(
          ApproximateEpsilonImageSyncPolicy(
            queue_size, rclcpp::Duration::from_seconds(approx_sync_epsilon)),
          *image_subs_.at(0), *image_subs_.at(1), *image_subs_.at(2), *image_subs_.at(3),
          *image_subs_.at(4), *image_subs_.at(5), *image_subs_.at(6), *image_subs_.at(7),
          *image_subs_.at(8)));
      break;
    default:
      return;
  }

  image_sync_->registerCallback(
    std::bind(
      &SimpleFusionNode::onImage, this, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
      std::placeholders::_7, std::placeholders::_8, std::placeholders::_9));

  fusion_sync_.reset(
    new ApproximateEpsilonFusionSync(
      ApproximateEpsilonFusionSyncPolicy(
        queue_size, rclcpp::Duration::from_seconds(approx_sync_epsilon)),
      *lidar_signal_sub_, *camera_signal_sub_));

  fusion_sync_->registerCallback(
    std::bind(
      &SimpleFusionNode::onMultiSensorSignal, this, std::placeholders::_1, std::placeholders::_2));
}

void SimpleFusionNode::onImage(
  const Image::ConstSharedPtr & image0,
  const Image::ConstSharedPtr & image1,
  const Image::ConstSharedPtr & image2,
  const Image::ConstSharedPtr & image3,
  const Image::ConstSharedPtr & image4,
  const Image::ConstSharedPtr & image5,
  const Image::ConstSharedPtr & image6,
  const Image::ConstSharedPtr & image7,
  const Image::ConstSharedPtr & image8)
{
  RCLCPP_INFO_STREAM(
    get_logger(), "image0: " << image0->header.stamp.sec << "." << image0->header.stamp.nanosec);

  RCLCPP_INFO_STREAM(
    get_logger(), "image1: " << image1->header.stamp.sec << "." << image1->header.stamp.nanosec);

  RCLCPP_INFO_STREAM(
    get_logger(), "image2: " << image2->header.stamp.sec << "." << image2->header.stamp.nanosec);

  RCLCPP_INFO_STREAM(
    get_logger(), "image3: " << image3->header.stamp.sec << "." << image3->header.stamp.nanosec);

  RCLCPP_INFO_STREAM(
    get_logger(), "image4: " << image4->header.stamp.sec << "." << image4->header.stamp.nanosec);

  RCLCPP_INFO_STREAM(
    get_logger(), "image5: " << image5->header.stamp.sec << "." << image5->header.stamp.nanosec);

  RCLCPP_INFO_STREAM(
    get_logger(), "image6: " << image6->header.stamp.sec << "." << image6->header.stamp.nanosec);

  RCLCPP_INFO_STREAM(
    get_logger(), "image7: " << image7->header.stamp.sec << "." << image7->header.stamp.nanosec);

  RCLCPP_INFO_STREAM(
    get_logger(), "image8: " << image8->header.stamp.sec << "." << image8->header.stamp.nanosec);
  Signal signal;
  signal.header = image0->header;
  camera_signal_pub_->publish(signal);
}

void SimpleFusionNode::onMultiSensorSignal(
  const simple_fusion::msg::Signal::ConstSharedPtr & lidar_signal,
  const simple_fusion::msg::Signal::ConstSharedPtr & camera_signal)
{
  RCLCPP_INFO_STREAM(
    get_logger(),
    "lidar: " << lidar_signal->header.stamp.sec << "." << lidar_signal->header.stamp.nanosec);
  RCLCPP_INFO_STREAM(
    get_logger(),
    "camera: " << camera_signal->header.stamp.sec << "." << camera_signal->header.stamp.nanosec);
}

}  // namespace simple_fusion

RCLCPP_COMPONENTS_REGISTER_NODE(simple_fusion::SimpleFusionNode)
