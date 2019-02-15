/**
 * A class to wrap the Extrinsic Calibrator within ROS messages
 */
#pragma once

#include <apriltag_tracker/AprilTagTrackerRosWrapper.h>
#include <extrinsics_calibrator/ExtrinsicCalibration.h>
#include <ros/ros.h>

#include <rosbag/bag.h>
#include <rosbag/message_instance.h>
#include <rosbag/query.h>
#include <rosbag/view.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>
/**
 * Inherits from message_filters::SimpleFilter<M>
 * to use protected signalMessage function
 */
template <class M>
class BagSubscriber : public message_filters::SimpleFilter<M> {
 public:
  void newMessage(const boost::shared_ptr<M const> &msg) {
    this->signalMessage(msg);
  }
};

class ExtrinsicCalibrationRosWrapper {
 public:
  struct extrinsic_calib_ros_params_t_ {
    std::string image_topic, odom_topic;
    std::string bag_file;
  };

 private:
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          nav_msgs::Odometry>
      ExtrinsicCalibratorPolicy;
  // Set up fake subscribers to capture images
  typedef BagSubscriber<sensor_msgs::Image> ImageSubscriber;
  typedef BagSubscriber<nav_msgs::Odometry> OdomSubscriber;

  std::shared_ptr<ImageSubscriber> image_sub_;
  std::shared_ptr<OdomSubscriber> odom_sub_;
  std::shared_ptr<message_filters::Synchronizer<ExtrinsicCalibratorPolicy>>
      synchronizer_;

  std::shared_ptr<ros::NodeHandle> nh_;
  tf2_ros::TransformBroadcaster broadcaster_;

  std::shared_ptr<ExtrinsicCalibration> calib_;
  std::shared_ptr<AprilTagTrackerRosWrapper> apriltag_detector_;

  extrinsic_calib_ros_params_t_ params_;

  Eigen::MatrixXd last_pose_, first_pose_;
  Eigen::MatrixXd accumulated_pose_;
  uint tuple_counter_;

  bool init_flag_;
  double curr_t_;

 public:
  ExtrinsicCalibrationRosWrapper(const std::string &config_file,
                                 const ros::NodeHandle &nh)
      : last_pose_(Eigen::Matrix4d::Identity()),
        first_pose_(Eigen::Matrix4d::Identity()),
        accumulated_pose_(Eigen::Matrix4d::Identity()),
        tuple_counter_(0),
        init_flag_(false),
        curr_t_(0.0) {
    nh_ = std::make_shared<ros::NodeHandle>(nh);
    calib_ = std::make_shared<ExtrinsicCalibration>(config_file);
    apriltag_detector_ =
        std::make_shared<AprilTagTrackerRosWrapper>(config_file);
    calib_->set_apriltag_params(get_apriltag_params());
    readConfig(config_file);
    initSubscribers();
  }
  apriltag_params get_apriltag_params() {
    return apriltag_detector_->get_params();
  }
  void calibrate();
  void readConfig(const std::string &config_file);
  void readBag(const std::string &bag_file);
  void initSubscribers();
  void callback(const sensor_msgs::Image::ConstPtr &image_msg,
                const nav_msgs::Odometry::ConstPtr &odom_msg);
  void publishLandmarks(
      const std::vector<std::pair<uint, gtsam::Pose3>> &landmarks,
      const double &stamp);
};

void ExtrinsicCalibrationRosWrapper::readConfig(
    const std::string &config_file) {
  try {
    YAML::Node params = YAML::LoadFile(config_file);
    std::cout << "[ExtrinsicsCalibrationRosWrapper] Reading from "
              << config_file << "\n";
    YAML::Node extrinsics_calib_node = params["extrinsics_calib_params"];
    if (!extrinsics_calib_node) {
      ROS_FATAL("Could not read extrinsics_calib_node!");
      exit(-1);
    } else {
      params_.bag_file = extrinsics_calib_node["bag_file"].as<std::string>();
      std::cout
          << "[CameraExtrinsicCalibrationRosWrapper] Reading from bag file "
          << params_.bag_file << "\n";

      params_.image_topic =
          extrinsics_calib_node["image_topic"].as<std::string>();
      params_.odom_topic =
          extrinsics_calib_node["odom_topic"].as<std::string>();
      std::cout << "[CameraExtrinsicCalibrationRosWrapper] Image topic: "
                << params_.image_topic << "\n";
      std::cout << "[CameraExtrinsicCalibrationRosWrapper] Odom topic: "
                << params_.odom_topic << "\n";
    }
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    exit(-1);
  }
}

void ExtrinsicCalibrationRosWrapper::calibrate() {
  readBag(params_.bag_file);
  std::vector<std::pair<uint, gtsam::Pose3>> landmarks;
  std::cout << "And the final answer is \n" << calib_->solve(landmarks) << "\n";
  publishLandmarks(landmarks, curr_t_);
}

void ExtrinsicCalibrationRosWrapper::publishLandmarks(
    const std::vector<std::pair<uint, gtsam::Pose3>> &landmarks,
    const double &stamp) {
  geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.stamp = ros::Time(stamp);
  transformStamped.header.frame_id = "world";
  for (auto landmark : landmarks) {
    transformStamped.child_frame_id = "l" + std::to_string(landmark.first);
    Eigen::Matrix4d theta = landmark.second.matrix();
    Eigen::Quaterniond quat(theta.block<3, 3>(0, 0));
    quat.normalize();

    transformStamped.transform.translation.x = theta(0, 3);
    transformStamped.transform.translation.y = theta(1, 3);
    transformStamped.transform.translation.z = theta(2, 3);

    transformStamped.transform.rotation.x = quat.x();
    transformStamped.transform.rotation.y = quat.y();
    transformStamped.transform.rotation.z = quat.z();
    transformStamped.transform.rotation.w = quat.w();

    broadcaster_.sendTransform(transformStamped);
  }
}

void ExtrinsicCalibrationRosWrapper::initSubscribers() {
  image_sub_ = std::make_shared<ImageSubscriber>();
  odom_sub_ = std::make_shared<OdomSubscriber>();

  // Initialize the ApproximateTimeSynchronizer
  synchronizer_ = std::make_shared<
      message_filters::Synchronizer<ExtrinsicCalibratorPolicy>>(
      ExtrinsicCalibratorPolicy(100), *(image_sub_), *(odom_sub_));

  // Register callbacks
  synchronizer_->registerCallback(
      boost::bind(&ExtrinsicCalibrationRosWrapper::callback, this, _1, _2));
}

void ExtrinsicCalibrationRosWrapper::callback(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const nav_msgs::Odometry::ConstPtr &odom_msg) {
  // Just call the Extrinsic Calibration addEdge
  Eigen::Quaterniond quat(
      odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
      odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);
  Eigen::Vector3d pos(odom_msg->pose.pose.position.x,
                      odom_msg->pose.pose.position.y,
                      odom_msg->pose.pose.position.z);
  Eigen::MatrixXd curr_pose = Eigen::Matrix4d::Identity();
  quat.normalize();
  curr_pose.block<3, 3>(0, 0) = quat.matrix();
  curr_pose.block<3, 1>(0, 3) = pos;

  // If this is the very first data tuple we're getting, set the init pose using
  // mocap
  if (!init_flag_) {
    calib_->initGraph(curr_pose);
    init_flag_ = true;
    first_pose_ = curr_pose;
    return;
  }

  // Call the apriltag tracker library
  std::map<uint, std::vector<cv::Point2f>> detections;
  cv::Mat debug_img;
  // Call ROS Wrapper data callback method
  apriltag_detector_->callback(image_msg);
  apriltag_detector_->detectTags(detections, debug_img);

  if (!detections.empty()) {
    //    Eigen::MatrixXd curr_pose_in_first = first_pose_.inverse() *
    //    curr_pose;
    //    Eigen::MatrixXd delta_pose = last_pose_.inverse() *
    //    curr_pose_in_first;
    //    accumulated_pose_ = accumulated_pose_ * delta_pose;
    //    last_pose_ = curr_pose_in_first;
    //    calib_->addEdge(gtsam::Pose3(delta_pose), detections);
    // Since we have absolute mocap ground truth poses, add an absolute edge
    calib_->addAbsoluteEdge(curr_pose, detections);
    tuple_counter_++;
    //    std::vector<std::pair<uint, gtsam::Pose3>> landmarks;
    //    calib_->landmark_init_list(landmarks);
    //    publishLandmarks(landmarks, image_msg->header.stamp.toSec());
  }
  // And also broadcast the current pose
  Eigen::Matrix4d theta = curr_pose.matrix();
  quat = Eigen::Quaterniond(theta.block<3, 3>(0, 0));
  geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.frame_id = "world";
  transformStamped.child_frame_id = "curr";
  transformStamped.header.stamp = image_msg->header.stamp;
  curr_t_ = image_msg->header.stamp.toSec();
  transformStamped.transform.translation.x = theta(0, 3);
  transformStamped.transform.translation.y = theta(1, 3);
  transformStamped.transform.translation.z = theta(2, 3);

  transformStamped.transform.rotation.x = quat.x();
  transformStamped.transform.rotation.y = quat.y();
  transformStamped.transform.rotation.z = quat.z();
  transformStamped.transform.rotation.w = quat.w();

  broadcaster_.sendTransform(transformStamped);

  // And accumulated
  //  transformStamped.header.frame_id = "world";
  //  transformStamped.child_frame_id = "accum";
  //  transformStamped.header.stamp = image_msg->header.stamp;
  //  curr_t_ = image_msg->header.stamp.toSec();
  //  theta = accumulated_pose_.matrix();
  //  quat = Eigen::Quaterniond(theta.block<3, 3>(0, 0));
  //  transformStamped.transform.translation.x = theta(0, 3);
  //  transformStamped.transform.translation.y = theta(1, 3);
  //  transformStamped.transform.translation.z = theta(2, 3);
  //
  //  transformStamped.transform.rotation.x = quat.x();
  //  transformStamped.transform.rotation.y = quat.y();
  //  transformStamped.transform.rotation.z = quat.z();
  //  transformStamped.transform.rotation.w = quat.w();
  //
  //  broadcaster_.sendTransform(transformStamped);
}

void ExtrinsicCalibrationRosWrapper::readBag(const std::string &bag_file) {
  rosbag::Bag bag;
  bag.open(bag_file, rosbag::bagmode::Read);

  std::vector<std::string> topics;
  topics.push_back(params_.image_topic);
  topics.push_back(params_.odom_topic);

  rosbag::View view(bag, rosbag::TopicQuery(topics));
  std::cout
      << "\n[CameraExtrinsicCalibrationRosWrapper] Beginning reading bag\n";
  double start_time = view.getBeginTime().toSec();
  double end_time = view.getEndTime().toSec();
  double percent_multiplier = 100.0 / (end_time - start_time);
  double start_percent = 1.0, end_percent = 90.0;
  for (auto m : view) {
    float percent = (m.getTime().toSec() - start_time) * percent_multiplier;
    if (!ros::ok()) {
      break;
    }
    if (percent >= end_percent || percent < start_percent) {
      continue;
    }
    sensor_msgs::Image::ConstPtr s = m.instantiate<sensor_msgs::Image>();
    if (s != NULL) {
      image_sub_->newMessage(s);
    } else {
      nav_msgs::Odometry::ConstPtr i = m.instantiate<nav_msgs::Odometry>();
      if (i != NULL) {
        odom_sub_->newMessage(i);
      }
    }
    std::cout << "\r[CameraExtrinsicCalibrationRosWrapper] Read Percentage: "
              << percent << "%\tTuples Read: " << tuple_counter_;
  }
  bag.close();
  std::cout << "\n[CameraExtrinsicCalibrationRosWrapper] Done Reading bag!\n";
}
