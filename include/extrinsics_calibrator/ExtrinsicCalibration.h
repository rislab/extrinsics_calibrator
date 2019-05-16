/*
 * ExtrinsicCalibration.h
 *
 *  A class to determine the extrinsic calibration between body frame and camera
 */

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/CalibratedCamera.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
// Camera observations of landmarks (i.e. pixel coordinates) will be stored as
// Point2 (x, y).
#include <gtsam/geometry/Point2.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam_unstable/slam/ProjectionFactorPPP.h>
#include <gtsam_unstable/slam/ProjectionFactorPPPC.h>

#include <boost/variant.hpp>
#include <map>

#include <Eigen/Dense>

#include <apriltag_tracker/AprilTagTracker.h>

#include <extrinsics_calibrator/FiducialPoseProjectionFactor.h>

#include <fstream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

void get_pnp_pose(const std::vector<cv::Point2f> &p_img, gtsam::Pose3 &pose, const apriltag_params &april_params)
{
  float s = april_params.tag_size / 2.;
  // tag corners in tag frame, which we call object
  std::vector<cv::Point3f> p_obj;
  p_obj.push_back(cv::Point3f(-s, -s, 0));
  p_obj.push_back(cv::Point3f(s, -s, 0));
  p_obj.push_back(cv::Point3f(s, s, 0));
  p_obj.push_back(cv::Point3f(-s, s, 0));
  cv::Mat rvec, tvec;
  // The output is in float since all our parameters passed are floats...
  cv::solvePnP(p_obj, p_img, april_params.K, april_params.D, rvec, tvec);
  pose = gtsam::Pose3(
      gtsam::Rot3::Rodrigues(rvec.at<float>(0), rvec.at<float>(1),
                             rvec.at<float>(2)),
      gtsam::Point3(tvec.at<float>(0), tvec.at<float>(1), tvec.at<float>(2)));
}

class ExtrinsicCalibration
{
public:
  enum calibration_mode_t
  {
    EXTRINSICS_ONLY = 0,
    FULL
  };

  struct extrinsics_params_t_
  {
    gtsam::Vector6 odom_noise, landmark_noise, extrinsics_noise;
    gtsam::Vector5 intrinsics_noise;
    float measurement_noise;
    calibration_mode_t mode;
    apriltag_params april_params;
    extrinsics_params_t_()
      : mode(EXTRINSICS_ONLY)
      , odom_noise((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished())
      , extrinsics_noise((gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished())
      , intrinsics_noise((gtsam::Vector(5) << 10.0, 10.0, 1.0, 1.0, 1.0).finished())
      , measurement_noise(2.0)
    {
    }
  };

private:
  gtsam::Cal3_S2::shared_ptr intrinsics_guess_;
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initial_;

  extrinsics_params_t_ params_;

  gtsam::noiseModel::Diagonal::shared_ptr odom_noise_;
  gtsam::noiseModel::Diagonal::shared_ptr landmark_noise_;
  gtsam::noiseModel::Diagonal::shared_ptr extrinsics_noise_;
  gtsam::noiseModel::Diagonal::shared_ptr intrinsics_noise_;
  gtsam::noiseModel::Base::shared_ptr measurement_noise_;

  gtsam::Key last_key_;
  gtsam::Pose3 extrinsics_guess_;
  gtsam::Pose3 last_in_first_;
  std::vector<uint> landmarks_added_;

public:
  ExtrinsicCalibration(const std::string &config_file)
    : last_key_(0)
    , extrinsics_guess_(gtsam::Pose3::identity())
    , last_in_first_(gtsam::Pose3::identity())
    , intrinsics_guess_(boost::make_shared<gtsam::Cal3_S2>())
  {
    readConfig(config_file);

    odom_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
        params_.odom_noise);  // 0.05 radian Rotation, 0.1m Translation
    extrinsics_noise_ =
        gtsam::noiseModel::Diagonal::Sigmas(params_.extrinsics_noise);
    intrinsics_noise_ =
        gtsam::noiseModel::Diagonal::Sigmas(params_.intrinsics_noise);
    const gtsam::noiseModel::Base::shared_ptr iso_meas_noise =
        gtsam::noiseModel::Isotropic::Sigma(8, params_.measurement_noise);
    measurement_noise_ = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(8),
                                                     iso_meas_noise);
    std::cout << "Using Intrinsics guess : " << intrinsics_guess_
              << "\n and extrinsics guess \n"
              << extrinsics_guess_.matrix() << "\n";
  }
  void set_apriltag_params(apriltag_params params)
  {
    params_.april_params = params;
  }
  void readConfig(const std::string &params_file);
  void addRelativeEdge(const gtsam::Pose3 &odom_edge, const std::map<uint, std::vector<cv::Point2f>> &detections);
  void addAbsoluteEdge(const gtsam::Pose3 &curr_pose, const std::map<uint, std::vector<cv::Point2f>> &detections);
  void initGraph(const gtsam::Pose3 &first_pose = gtsam::Pose3::identity());
  gtsam::Pose3 solve(std::vector<std::pair<uint, gtsam::Pose3>> &landmark_list);
  void landmark_init_list(std::vector<std::pair<uint, gtsam::Pose3>> &landmark_list);
};

std::ostream &operator<<(std::ostream &o, const ExtrinsicCalibration::extrinsics_params_t_ &params)
{
  o << "Odometry Noise: " << params.odom_noise.transpose()
    << "\nExtrinsics Noise: " << params.extrinsics_noise.transpose()
    << "\nIntrinsics Noise: " << params.intrinsics_noise.transpose()
    << "\nLandmark Noise: " << params.landmark_noise.transpose() << "\nMeasurement Noise: " << params.measurement_noise
    << "\n";
  return o;
}

void ExtrinsicCalibration::readConfig(const std::string &params_file)
{
  std::cout << "[CameraExtrinsicCalibration] Reading from file " << params_file << "\n";
  try
  {
    YAML::Node params = YAML::LoadFile(params_file);
    YAML::Node extrinsics_calib_node = params["extrinsics_calib_params"];
    if (!extrinsics_calib_node)
    {
      std::cerr << "[CameraExtrinsicCalibration] Could not read "
                   "extrinsics_calib_node!";
      //      exit(-1);
    }
    else
    {
      YAML::Node extrinsics_guess = extrinsics_calib_node["extrinsics_guess"];
      if (extrinsics_guess)
      {
        Eigen::MatrixXd guess = Eigen::Matrix4d::Identity();
        for (int i = 0; i < 4; i++)
        {
          for (int j = 0; j < 4; j++)
          {
            guess(i, j) = extrinsics_guess[4 * i + j].as<double>();
          }
        }
        extrinsics_guess_ = gtsam::Pose3(guess);
      }

      YAML::Node intrinsics_guess = extrinsics_calib_node["intrinsics_guess"];
      if (intrinsics_guess)
      {
        Eigen::VectorXd guess(5);
        for (int i = 0; i < 5; i++)
        {
          guess[i] = intrinsics_guess[i].as<double>();
        }
        intrinsics_guess_ = boost::make_shared<gtsam::Cal3_S2>(gtsam::Cal3_S2(guess));
      }

      YAML::Node odom_noise_node = extrinsics_calib_node["odom_noise"];
      if (odom_noise_node)
      {
        for (int i = 0; i < 6; i++)
        {
          params_.odom_noise[i] = odom_noise_node[i].as<double>();
        }
      }

      YAML::Node landmark_noise_node = extrinsics_calib_node["landmark_noise"];
      if (landmark_noise_node)
      {
        for (int i = 0; i < 6; i++)
        {
          params_.landmark_noise[i] = landmark_noise_node[i].as<double>();
        }
      }

      YAML::Node extrinsics_noise_node = extrinsics_calib_node["extrinsics_noise"];
      if (extrinsics_noise_node)
      {
        for (int i = 0; i < 6; i++)
        {
          params_.extrinsics_noise[i] = extrinsics_noise_node[i].as<double>();
        }
      }

      YAML::Node intrinsics_noise_node = extrinsics_calib_node["intrinsics_noise"];
      if (intrinsics_noise_node)
      {
        for (int i = 0; i < 5; i++)
        {
          params_.intrinsics_noise[i] = intrinsics_noise_node[i].as<double>();
        }
      }

      YAML::Node measurement_noise_node = extrinsics_calib_node["measurement_noise"];
      if (measurement_noise_node)
      {
        params_.measurement_noise = measurement_noise_node.as<double>();
      }
    }
    std::cout << "[CameraExtrinsicCalibration] CameraExtrinsicCalibration params : \n" << params_;
  }
  catch (const std::runtime_error &e)
  {
    std::cerr << e.what() << std::endl;
    exit(-1);
  }
}

void ExtrinsicCalibration::initGraph(const gtsam::Pose3 &prior_pose)
{
  // Create a very certain noise model for the prior
  gtsam::noiseModel::Diagonal::shared_ptr priorNoise =
      gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001).finished());

  // Add node for extrinsics
  last_in_first_ = prior_pose;
  initial_.insert(gtsam::Symbol('t', 0), extrinsics_guess_);
  // Add a prior
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('t', 0), extrinsics_guess_, extrinsics_noise_));

  // Add a node for the first pose
  initial_.insert(gtsam::Symbol('o', 0), prior_pose);
  // Add a prior
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('o', 0), prior_pose, priorNoise));

  //  if (params_.mode == FULL) {
  //    // Add node for calibration
  //    initial_.insert(gtsam::symbol('k', 0), intrinsics_guess_);
  //    // Add prior
  //    graph_.add(
  //        gtsam::PriorFactor<gtsam::Cal3_S2>(gtsam::Symbol('k', 0),
  //                                           *intrinsics_guess_,
  //                                           intrinsics_noise_));
  //  }
}

void ExtrinsicCalibration::addAbsoluteEdge(const gtsam::Pose3 &curr_pose,
                                           const std::map<uint, std::vector<cv::Point2f>> &detections)
{
  // Add initial estimate of this node
  initial_.insert(gtsam::Symbol('o', last_key_ + 1), curr_pose);
  // Add edge
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('o', last_key_ + 1), curr_pose, odom_noise_));
  // Now run through all the detections
  for (auto tag : detections)
  {
    std::vector<gtsam::Point2> measurement;
    for (const auto &point : tag.second)
    {
      measurement.push_back(gtsam::Point2(point.x, point.y));
    }
    // Is this the first time we're seeing this landmark?
    if (std::find(landmarks_added_.begin(), landmarks_added_.end(), tag.first) == landmarks_added_.end())
    {
      // Let's provide an initial value to this new landmark node seeded with
      // our current estimate
      gtsam::Pose3 tag_in_cam;
      get_pnp_pose(tag.second, tag_in_cam, params_.april_params);
      // Rotate this transform into the body frame
      gtsam::Pose3 tag_in_body = gtsam::Pose3(extrinsics_guess_.compose(tag_in_cam));
      // Rotate this using current estimate (i.e. express in world, aka, first
      // frame coordinates)?
      gtsam::Pose3 tag_in_world = curr_pose.compose(tag_in_body);
      initial_.insert(gtsam::Symbol('l', tag.first), tag_in_world);
      landmarks_added_.push_back(tag.first);
    }

    switch (params_.mode)
    {
      case EXTRINSICS_ONLY:
        // Create a FiducialPPPfactor
        graph_.add(
            gtsam::FiducialPoseProjectionFactor<gtsam::Pose3, gtsam::Pose3>(
                measurement, measurement_noise_,
                gtsam::Symbol('o', last_key_ + 1), gtsam::Symbol('t', 0),
                gtsam::Symbol('l', tag.first), intrinsics_guess_,
                params_.april_params.tag_size));
        break;
      default:
        throw std::runtime_error("Invalid mode!");
    }
  }
  // Finally increment the last_key_
  last_key_ += 1;
}

void ExtrinsicCalibration::addRelativeEdge(const gtsam::Pose3 &odom_edge,
                                           const std::map<uint, std::vector<cv::Point2f>> &detections)
{
  gtsam::Pose3 accumulated_pose = last_in_first_.compose(odom_edge);
  // Add initial estimate of this node
  initial_.insert(gtsam::Symbol('o', last_key_ + 1), accumulated_pose);
  // Add edge
  graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol('o', last_key_), gtsam::Symbol('o', last_key_ + 1),
                                                odom_edge, odom_noise_));
  // Now run through all the detections
  for (auto tag : detections)
  {
    std::vector<gtsam::Point2> measurement;
    for (const auto &point : tag.second)
    {
      measurement.push_back(gtsam::Point2(point.x, point.y));
    }
    // Is this the first time we're seeing this landmark?
    if (std::find(landmarks_added_.begin(), landmarks_added_.end(), tag.first) == landmarks_added_.end())
    {
      // Let's provide an initial value to this new landmark node seeded with
      // our current estimate
      gtsam::Pose3 tag_in_cam;
      get_pnp_pose(tag.second, tag_in_cam, params_.april_params);
      // Rotate this transform into the body frame
      gtsam::Pose3 tag_in_body = gtsam::Pose3(extrinsics_guess_.compose(tag_in_cam));
      // Rotate this using current estimate (i.e. express in world, aka, first
      // frame coordinates)?
      gtsam::Pose3 tag_in_first = last_in_first_.compose(tag_in_body);
      initial_.insert(gtsam::Symbol('l', tag.first), tag_in_first);
      landmarks_added_.push_back(tag.first);
    }

    switch (params_.mode) {
      case EXTRINSICS_ONLY:
        // Create a PPPfactor
        graph_.add(
            gtsam::FiducialPoseProjectionFactor<gtsam::Pose3, gtsam::Pose3>(
                measurement, measurement_noise_, gtsam::Symbol('o', last_key_),
                gtsam::Symbol('t', 0), gtsam::Symbol('l', tag.first),
                intrinsics_guess_, params_.april_params.tag_size));
        break;
      default:
        throw std::runtime_error("Invalid mode!");
    }
  }
  // Finally increment the last_key_
  last_key_ += 1;
  last_in_first_ = accumulated_pose;
}

gtsam::Pose3 ExtrinsicCalibration::solve(std::vector<std::pair<uint, gtsam::Pose3>> &landmark_list)
{
  gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_);
  gtsam::Values result = optimizer.optimize();
  //  result.print("Final Result:\n");
  gtsam::Pose3 extrinsics = result.at(gtsam::Symbol('t', 0)).cast<gtsam::Pose3>();
  landmark_list.clear();
  for (auto landmark_id : landmarks_added_)
  {
    landmark_list.push_back(
        std::make_pair(landmark_id, result.at(gtsam::Symbol('l', landmark_id)).cast<gtsam::Pose3>()));
  }
  return extrinsics;
}

void ExtrinsicCalibration::landmark_init_list(std::vector<std::pair<uint, gtsam::Pose3>> &landmark_list)
{
  landmark_list.clear();
  for (auto landmark_id : landmarks_added_)
  {
    landmark_list.push_back(
        std::make_pair(landmark_id, initial_.at(gtsam::Symbol('l', landmark_id)).cast<gtsam::Pose3>()));
  }
}
