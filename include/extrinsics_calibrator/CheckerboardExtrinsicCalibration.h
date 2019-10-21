/**
 * Copyright (C) 2019 Kumar Shaurya Shankar <kumarsha at cs dot cmu dot edu>
 * (Carnegie Mellon University)
 *
 */
#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/CalibratedCamera.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/geometry/Point2.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam_unstable/slam/ProjectionFactorPPP.h>
#include <gtsam_unstable/slam/ProjectionFactorPPPC.h>

#include <boost/variant.hpp>
#include <map>

#include <Eigen/Dense>

#include <extrinsics_calibrator/Cal3DS3.h>
#include <extrinsics_calibrator/CheckerboardFactorPPPC.h>
#include <extrinsics_calibrator/CheckerboardPoseProjectionFactor.h>

#include <memory>
#include <opencv2/opencv.hpp>

#include <yaml-cpp/yaml.h>

#define TREAT_CHECKERBOARD_CORNERS_INDEPENDENTLY 1
#define USE_DISTORTION 0

#if USE_DISTORTION
#define CAM_TYPE Cal3DS3
#else
#define CAM_TYPE Cal3_S2
#endif

class CheckerboardExtrinsicCalibration {
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit> MRow;

 public:
  enum calibration_mode_t {
    EXTRINSICS_ONLY = 0,
    FULL
  };
  struct extrinsics_params_t_ {
    gtsam::Vector6 odom_noise, extrinsics_noise;
#if TREAT_CHECKERBOARD_CORNERS_INDEPENDENTLY
    gtsam::Vector3 landmark_noise;
#else
    gtsam::Vector6 landmark_noise;
#endif
    gtsam::Vector intrinsics_noise;
    float measurement_noise;
    calibration_mode_t mode;
    int rows, cols;
    float s;
    extrinsics_params_t_()
        : mode(EXTRINSICS_ONLY), odom_noise((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()), extrinsics_noise((gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished()), intrinsics_noise(gtsam::Vector(gtsam::CAM_TYPE::Dim())), measurement_noise(2.0), rows(0), cols(0), s(0.0f) {
    }
  };

 private:
  gtsam::CAM_TYPE::shared_ptr intrinsics_guess_;
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initial_;

  extrinsics_params_t_ params_;

  gtsam::noiseModel::Diagonal::shared_ptr odom_noise_;
  gtsam::noiseModel::Diagonal::shared_ptr landmark_noise_;
  gtsam::noiseModel::Diagonal::shared_ptr extrinsics_noise_;
  gtsam::noiseModel::Diagonal::shared_ptr intrinsics_noise_;
  gtsam::noiseModel::Base::shared_ptr measurement_noise_;

  gtsam::Pose3 extrinsics_guess_;
  size_t num_poses_;
  bool init_;

 public:
  CheckerboardExtrinsicCalibration(const std::string &config_file) : extrinsics_guess_(gtsam::Pose3::identity()), intrinsics_guess_(boost::make_shared<gtsam::CAM_TYPE>()), num_poses_(0), init_(false) {
    // Read config file for params
    readConfig(config_file);
    odom_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
        params_.odom_noise);  // 0.05 radian Rotation, 0.1m Translation
    extrinsics_noise_ =
        gtsam::noiseModel::Diagonal::Sigmas(params_.extrinsics_noise);
    intrinsics_noise_ =
        gtsam::noiseModel::Diagonal::Sigmas(params_.intrinsics_noise);
    const gtsam::noiseModel::Base::shared_ptr iso_meas_noise =
#if TREAT_CHECKERBOARD_CORNERS_INDEPENDENTLY
        gtsam::noiseModel::Isotropic::Sigma(2, params_.measurement_noise);
#else
        gtsam::noiseModel::Isotropic::Sigma(2 * params_.rows * params_.cols, params_.measurement_noise);
#endif
    measurement_noise_ = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(1.345),
        iso_meas_noise);
    // measurement_noise_ = iso_meas_noise;
    landmark_noise_ =
        gtsam::noiseModel::Diagonal::Sigmas(params_.landmark_noise);
    std::cout << "Using Intrinsics guess : " << intrinsics_guess_->K()
              << "\n and extrinsics guess \n"
              << extrinsics_guess_.matrix() << "\n";
  }
  void readConfig(const std::string &);
  void addMeasurement(const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &);
  void solve(Eigen::Ref<MRow>, Eigen::Ref<MRow>, Eigen::Ref<MRow>, Eigen::Ref<MRow>, Eigen::Ref<MRow>);
};