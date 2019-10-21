#include <extrinsics_calibrator/CheckerboardExtrinsicCalibration.h>
#include <extrinsics_calibrator/FiducialPoseProjectionFactor.h>

std::ostream &operator<<(std::ostream &o, const CheckerboardExtrinsicCalibration::extrinsics_params_t_ &params) {
  o << "Odometry Noise: " << params.odom_noise.transpose()
    << "\nExtrinsics Noise: " << params.extrinsics_noise.transpose()
    << "\nIntrinsics Noise: " << params.intrinsics_noise.transpose()
    << "\nLandmark Noise: " << params.landmark_noise.transpose()
    << "\nMeasurement Noise: " << params.measurement_noise
    << "\nCheckerboard Rows: " << params.rows
    << "\nCheckerboard Cols: " << params.cols
    << "\nCheckerboard Size: " << params.s
    << "\n";
  return o;
}

void CheckerboardExtrinsicCalibration::readConfig(const std::string &params_file) {
  std::cout << "[CheckerboardExtrinsicCalibration] Reading from file " << params_file << "\n";
  try {
    YAML::Node params = YAML::LoadFile(params_file);
    YAML::Node extrinsics_calib_node = params["extrinsics_calib_params"];
    YAML::Node checkerboard_node = params["checkerboard_params"];

    if (!checkerboard_node) {
      std::cerr << "[CheckerboardExtrinsicCalibration] Could not read checkerboard_params!";
    } else {
      params_.rows = checkerboard_node["rows"].as<int>();
      params_.cols = checkerboard_node["cols"].as<int>();
      params_.s = checkerboard_node["s"].as<float>();
    }

    if (!extrinsics_calib_node) {
      std::cerr << "[CheckerboardExtrinsicCalibration] Could not read "
                   "extrinsics_calib_node!";
      //      exit(-1);
    } else {
      YAML::Node extrinsics_guess = extrinsics_calib_node["extrinsics_guess"];
      if (extrinsics_guess) {
        Eigen::MatrixXd guess = Eigen::Matrix4d::Identity();
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 4; j++) {
            guess(i, j) = extrinsics_guess[4 * i + j].as<double>();
          }
        }
        extrinsics_guess_ = gtsam::Pose3(guess);
      }

      YAML::Node intrinsics_guess = extrinsics_calib_node["intrinsics_guess"];
      if (intrinsics_guess) {
        Eigen::VectorXd guess(gtsam::CAM_TYPE::Dim());
        for (int i = 0; i < gtsam::CAM_TYPE::Dim(); i++) {
          guess[i] = intrinsics_guess[i].as<double>();
        }
        intrinsics_guess_ = boost::make_shared<gtsam::CAM_TYPE>(gtsam::CAM_TYPE(guess));
      }

      YAML::Node odom_noise_node = extrinsics_calib_node["odom_noise"];
      if (odom_noise_node) {
        for (int i = 0; i < 6; i++) {
          params_.odom_noise[i] = odom_noise_node[i].as<double>();
        }
      }

      YAML::Node landmark_noise_node = extrinsics_calib_node["landmark_noise"];
      if (landmark_noise_node) {
#ifdef TREAT_CHECKERBOARD_CORNERS_INDEPENDENTLY
        for (int i = 0; i < 3; i++) {
#else
        for (int i = 0; i < 6; i++) {
#endif
          params_.landmark_noise[i] = landmark_noise_node[i].as<double>();
        }
      }

      YAML::Node extrinsics_noise_node = extrinsics_calib_node["extrinsics_noise"];
      if (extrinsics_noise_node) {
        for (int i = 0; i < 6; i++) {
          params_.extrinsics_noise[i] = extrinsics_noise_node[i].as<double>();
        }
      }

      YAML::Node intrinsics_noise_node = extrinsics_calib_node["intrinsics_noise"];
      if (intrinsics_noise_node) {
        for (int i = 0; i < gtsam::CAM_TYPE::Dim(); i++) {
          params_.intrinsics_noise[i] = intrinsics_noise_node[i].as<double>();
        }
      }

      YAML::Node measurement_noise_node = extrinsics_calib_node["measurement_noise"];
      if (measurement_noise_node) {
        params_.measurement_noise = measurement_noise_node.as<float>();
      }

      YAML::Node mode_node = extrinsics_calib_node["full_mode"];
      if (mode_node) {
        params_.mode = mode_node.as<bool>() ? FULL : EXTRINSICS_ONLY;
      }
    }
    std::cout << "[CheckerboardExtrinsicCalibration] CheckerboardExtrinsicCalibration params : \n"
              << params_;
  } catch (const std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
    exit(-1);
  }
}

void CheckerboardExtrinsicCalibration::addMeasurement(const Eigen::MatrixXd &body_to_world, const Eigen::MatrixXd &board_to_world, const Eigen::MatrixXd &measurements) {
  if (!init_) {
    // Add the symbol for the extrinsics
    initial_.insert(gtsam::Symbol('t', 0), extrinsics_guess_);

#ifdef TREAT_CHECKERBOARD_CORNERS_INDEPENDENTLY
    // IF TREATING EACH CORNER INDEPENDENTLY
    // Add a reasonable initial estimate for all the checkerboard corners
    for (int i = 0; i < params_.rows; ++i) {
      for (int j = 0; j < params_.cols; ++j) {
        // The corners are provided in row major order
        gtsam::Point3 corner(j * params_.s, i * params_.s, 0.0f);
        gtsam::Matrix4 tag_in_board;
        tag_in_board << 0, 1, 0, 0,
            1, 0, 0, 0,
            0, 0, -1, 0,
            0, 0, 0, 1;
        // Transform to world frame
        gtsam::Point3 corner_in_board = (tag_in_board * corner.homogeneous()).head<3>();
        // if (i == 0 && j == 0) {
        // Add a prior factor for the first landmark
        // graph_.add(
        //     gtsam::PriorFactor<gtsam::Point3>(
        //         gtsam::Symbol('l', i * params_.cols + j), corner_in_board, landmark_noise_));
        // }
        // Also add tight between factors between this point and the others
        gtsam::noiseModel::Diagonal::shared_ptr checkerboard_noise =
            gtsam::noiseModel::Isotropic::Sigma(3, 0.001);

        // for (int v = 0; v < params_.rows; ++v) {
        //   for (int u = 0; u < params_.cols; ++u) {
        //     if (u != j && v != i && (v * params_.cols + u > i * params_.cols + j)) {
        // if (!(i == 0 && j == 0)) {
        //   graph_.add(
        //       gtsam::BetweenFactor<gtsam::Point3>(
        //           gtsam::Symbol('l', 0),
        //           gtsam::Symbol('l', i * params_.cols + j),
        //           gtsam::Point3(j * params_.s, i * params_.s, 0),
        //           checkerboard_noise));
        // }
        //   }
        // }

        initial_.insert(gtsam::Symbol('l', i * params_.cols + j), corner_in_board);
      }
    }

#else
    // IF TREATING ALL CORNERS TOGETHER
    // Just add an initial value for the checkerboard pose - this is tag_in_board
    // since we assume the origin to be at the mocap checkerboard frame
    gtsam::Matrix4 tag_in_board;
    tag_in_board << 0, 1, 0, 0,
        1, 0, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1;

    initial_.insert(gtsam::Symbol('l', 0), gtsam::Pose3(tag_in_board));
    // // Also, since we're reasonably sure of this transformation
    // graph_.add(
    //     gtsam::PriorFactor<gtsam::Pose3>(
    //         gtsam::Symbol('l', 0),
    //         gtsam::Pose3(tag_in_board),
    //         landmark_noise_));
#endif
    // TODO: Add a prior for the first pose?
    // If we're in full mode, add a prior for K
    if (params_.mode == FULL) {
      initial_.insert(gtsam::Symbol('k', 0), *intrinsics_guess_);
      // Also add prior
      graph_.add(gtsam::PriorFactor<gtsam::CAM_TYPE>(gtsam::Symbol('k', 0), *intrinsics_guess_, intrinsics_noise_));
    }
    init_ = true;
  }
  // Alright, measurements of the same landmarks again! Add them in!
  initial_.insert(gtsam::Symbol('x', num_poses_), gtsam::Pose3(board_to_world.inverse() * body_to_world));
  // Since these are coming via mocap, let's put some string unary priors as well
  graph_.add(
      gtsam::PriorFactor<gtsam::Pose3>(
          gtsam::Symbol('x', num_poses_),
          gtsam::Pose3(board_to_world.inverse() * body_to_world),
          odom_noise_));
  switch (params_.mode) {
    case EXTRINSICS_ONLY: {
#ifdef TREAT_CHECKERBOARD_CORNERS_INDEPENDENTLY
      for (int i = 0; i < measurements.rows(); ++i) {
        graph_.add(
            gtsam::ProjectionFactorPPP<gtsam::Pose3, gtsam::Point3, gtsam::CAM_TYPE>(
                gtsam::Point2(measurements.row(i)),
                measurement_noise_,
                gtsam::Symbol('x', num_poses_),
                gtsam::Symbol('t', 0),
                gtsam::Symbol('l', i),
                intrinsics_guess_));
      }
#else
      std::vector<gtsam::Point2> measurement;
      for (int i = 0; i < measurements.rows(); ++i) {
        measurement.push_back(gtsam::Point2(measurements.row(i)));
      }
      graph_.add(
          gtsam::CheckerboardPoseProjectionFactor<gtsam::Pose3, gtsam::Pose3>(
              measurement,
              measurement_noise_,
              gtsam::Symbol('x', num_poses_),
              gtsam::Symbol('t', 0),
              gtsam::Symbol('l', 0),
              intrinsics_guess_,
              params_.rows,
              params_.cols,
              params_.s));
#endif
      break;
    }
    case FULL: {
#ifdef TREAT_CHECKERBOARD_CORNERS_INDEPENDENTLY
      for (int i = 0; i < measurements.rows(); ++i) {
        graph_.add(
            gtsam::ProjectionFactorPPPC<gtsam::Pose3, gtsam::Point3, gtsam::CAM_TYPE>(
                gtsam::Point2(measurements.row(i)),
                measurement_noise_,
                gtsam::Symbol('x', num_poses_),
                gtsam::Symbol('t', 0),
                gtsam::Symbol('l', i),
                gtsam::Symbol('k', 0)));
      }
#else
      std::vector<gtsam::Point2> measurement;
      for (int i = 0; i < measurements.rows(); ++i) {
        measurement.push_back(gtsam::Point2(measurements.row(i)));
      }
      graph_.add(
          gtsam::CheckerboardFactorPPPC<gtsam::Pose3, gtsam::Pose3>(
              measurement,
              measurement_noise_,
              gtsam::Symbol('x', num_poses_),
              gtsam::Symbol('t', 0),
              gtsam::Symbol('l', 0),
              gtsam::Symbol('k', 0),
              params_.rows,
              params_.cols,
              params_.s));
#endif
      break;
    }
  }
  ++num_poses_;
}

void CheckerboardExtrinsicCalibration::solve(Eigen::Ref<MRow> cam_to_body, Eigen::Ref<MRow> tag_to_board, Eigen::Ref<MRow> cam_poses, Eigen::Ref<MRow> landmarks, Eigen::Ref<MRow> K) {
  gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_);
  {
    std::ofstream os("test.dot");
    graph_.saveGraph(os, initial_);
  }
  gtsam::Values result = optimizer.optimize();
  result.print("Final Result:\n");

  gtsam::Pose3 cam_to_body_pose = result.at(gtsam::Symbol('t', 0)).cast<gtsam::Pose3>();
  cam_to_body = cam_to_body_pose.matrix();
  // Also print the evaluate error at first factor?
  graph_.printErrors(result);
#ifdef TREAT_CHECKERBOARD_CORNERS_INDEPENDENTLY
  for (int i = 0; i < params_.rows * params_.cols; ++i) {
    landmarks.row(i) = result.at(gtsam::Symbol('l', i)).cast<gtsam::Point3>();
  }
  for (int i = 0; i < num_poses_; ++i) {
    gtsam::Pose3 cam_pose = result.at(gtsam::Symbol('x', i)).cast<gtsam::Pose3>();

    cam_poses.row(i) = Eigen::Map<MRow>(cam_pose.matrix().data(), 1, 16);
    if (i == 0) {
      std::cout << " cam_pose ::\n " << cam_pose << "\n row is " << cam_to_body.row(i) << "\n";
    }
  }
#else
  gtsam::Pose3 tag_to_board_pose = result.at(gtsam::Symbol('l', 0)).cast<gtsam::Pose3>();
  tag_to_board = tag_to_board_pose.matrix();
#endif
  if (params_.mode == FULL) {
    K = result.at(gtsam::Symbol('k', 0)).cast<gtsam::CAM_TYPE>().K();
    std::cout << "Solved K was \n"
              << K;
#if USE_DISTORTION
    std::cout << "solved distortion params were \n"
              << result.at(gtsam::Symbol('k', 0)).cast<gtsam::CAM_TYPE>().k();
#endif
  }
  std::cout << " K was \n"
            << *intrinsics_guess_ << "\n";
  printf("\n\nError before optimisation::%f Error after optimisation::%f\n", graph_.error(initial_), graph_.error(result));
}