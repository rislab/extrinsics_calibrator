/**
 * Copyright (C) 2017 Kumar Shaurya Shankar <kumarsha at cs dot cmu dot edu>
 * (Carnegie Mellon University)
 *
 */
#pragma once

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <boost/optional.hpp>
namespace gtsam {

/**
 * Factor to measure a checkerboard from a given pose
 */
template <class POSE, class LANDMARK, class CALIBRATION>
class CheckerboardFactorPPPC
    : public NoiseModelFactor4<POSE, POSE, LANDMARK, CALIBRATION> {
 protected:
  // Keep a copy of measurement and calibration for I/O
  std::vector<Point2> measured_;  ///< 2D measurement
  int rows_, cols_;
  double s_;
  std::vector<Point3> checkerboard_points_;

  // verbosity handling for Cheirality Exceptions
  bool throwCheirality_;    ///< If true, rethrows Cheirality exceptions (default:
                            ///false)
  bool verboseCheirality_;  ///< If true, prints text for Cheirality exceptions
                            ///(default: false)

 public:
  /// shorthand for base class type
  typedef NoiseModelFactor4<POSE, POSE, LANDMARK, CALIBRATION> Base;

  /// shorthand for this class
  typedef CheckerboardFactorPPPC<POSE, LANDMARK, CALIBRATION> This;

  /// shorthand for a smart pointer to a factor
  typedef boost::shared_ptr<This> shared_ptr;

  /// Default constructor
  CheckerboardFactorPPPC() : rows_(0), cols_(0), s_(0.0), throwCheirality_(false), verboseCheirality_(false) {}

  /// Constructor
  CheckerboardFactorPPPC(const std::vector<Point2>& measured,
                         const SharedNoiseModel& model,
                         Key poseKey,
                         Key transformKey,
                         Key checkerboardKey,
                         Key calibKey,
                         int rows, int cols,
                         const double& s)
      : Base(model, poseKey, transformKey, checkerboardKey, calibKey),
        measured_(measured),
        rows_(rows),
        cols_(cols),
        s_(s),
        throwCheirality_(false),
        verboseCheirality_(true) {
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        // The corners are provided in row major order
        gtsam::Point3 corner(j * s_, i * s_, 0.0f);
        checkerboard_points_.push_back(corner);
      }
    }
  }

  /** Virtual destructor */
  virtual ~CheckerboardFactorPPPC() {}

  /// @return a deep copy of this factor
  virtual gtsam::NonlinearFactor::shared_ptr clone() const {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /**
   * print
   * @param s optional string naming the factor
   * @param keyFormatter optional formatter useful for printing Symbols
   */
  void print(const std::string& s = "",
             const KeyFormatter& keyFormatter = DefaultKeyFormatter) const {
    std::cout << s << "CheckerboardFactorPPPC, z = ";
    for (auto point : measured_) {
      traits<Point2>::Print(point);
    }
    Base::print("", keyFormatter);
  }

  /// equals
  virtual bool equals(const NonlinearFactor& p, double tol = 1e-9) const {
    const This* e = dynamic_cast<const This*>(&p);
    bool point_equals = true;
    for (int i = 0; i < measured_.size(); i++) {
      if (!traits<Point2>::Equals(this->measured_[i], e->measured_[i], tol)) {
        point_equals = false;
        break;
      }
    }
    bool dims_equals = this->rows_ == e->rows_ && this->cols_ == e->cols_ && this->s_ == e->s_;
    return e && Base::equals(p, tol) && point_equals && dims_equals;
  }

  /// Evaluate error h(x)-z and optionally derivatives
  Vector evaluateError(const Pose3& pose,
                       const Pose3& transform,
                       const Pose3& object,
                       const CALIBRATION& K,
                       boost::optional<Matrix&> H1 = boost::none,
                       boost::optional<Matrix&> H2 = boost::none,
                       boost::optional<Matrix&> H3 = boost::none,
                       boost::optional<Matrix&> H4 = boost::none) const {
    try {
      if (H1 || H2 || H3 || H4) {
        gtsam::Matrix H_worldcam_wrt_body, H_worldcam_wrt_bodycam;
        PinholeCamera<CALIBRATION>
            camera(pose.compose(transform, H_worldcam_wrt_body, H_worldcam_wrt_bodycam), K);
        Vector reprojectionError(2 * checkerboard_points_.size());
        *H1 = Matrix::Zero(2 * checkerboard_points_.size(), 6);
        *H2 = Matrix::Zero(2 * checkerboard_points_.size(), 6);
        *H3 = Matrix::Zero(2 * checkerboard_points_.size(), 6);  // checkerboardObject is also a pose!
        // @todo: Make sure that the measurement dimensionality is the same as
        // checkerboard
        for (int i = 0; i < checkerboard_points_.size(); i++) {
          Matrix h_worldpoint_wrt_worldcam_deltapose(2, 6),
              h_worldpoint_wrt_worldcam_deltapoint(2, 3),
              h_objectpoint_wrt_deltapose(3, 6),
              h_worldpoint_wrt_cam;
          Point3 object_point_in_world = object.transformFrom(
              checkerboard_points_[i], h_objectpoint_wrt_deltapose);
          Point2 error(
              // H1 is the jacobian wrt 6DoF pose 2x6
              // H2 is the jacobian wrt 3D point 2x3
              camera.project(
                  object_point_in_world, h_worldpoint_wrt_worldcam_deltapose,
                  h_worldpoint_wrt_worldcam_deltapoint, h_worldpoint_wrt_cam) -
              measured_[i]);
          // Accumulate error and jacobians, since the error function is just a
          // sum
          reprojectionError[2 * i] = error[0];
          reprojectionError[2 * i + 1] = error[1];
          // Compose with composition jacobian
          H1->block<2, 6>(2 * i, 0) =
              h_worldpoint_wrt_worldcam_deltapose * H_worldcam_wrt_body;
          H2->block<2, 6>(2 * i, 0) =
              h_worldpoint_wrt_worldcam_deltapose *
              H_worldcam_wrt_bodycam;  // Which should be I3x3 for the second
                                       // term
          H3->block<2, 6>(2 * i, 0) = h_worldpoint_wrt_worldcam_deltapoint *
                                      h_objectpoint_wrt_deltapose;
          H4->block<2, FixedDimension<CALIBRATION>::value>(2 * i, 0) = h_worldpoint_wrt_cam;
        }
        return reprojectionError;
      } else {
        Vector reprojectionError(2 * checkerboard_points_.size());
        PinholeCamera<CALIBRATION> camera(pose.compose(transform), K);
        for (int i = 0; i < checkerboard_points_.size(); i++) {
          Point3 object_point_in_world =
              object.transformFrom(checkerboard_points_[i]);
          Point2 error(camera.project(object_point_in_world) - measured_[i]);
          reprojectionError[2 * i] = error[0];
          reprojectionError[2 * i + 1] = error[1];
        }
        return reprojectionError;
      }
    } catch (CheiralityException& e) {
      if (H1) *H1 = Matrix::Zero(2 * checkerboard_points_.size(), 6);
      if (H2) *H2 = Matrix::Zero(2 * checkerboard_points_.size(), 6);
      if (H3) *H3 = Matrix::Zero(2 * checkerboard_points_.size(), 3);
      if (H4) *H4 = Matrix::Zero(2 * checkerboard_points_.size(), CALIBRATION::Dim());
      if (verboseCheirality_)
        std::cout << e.what() << ": checkerboardObject "
                  << DefaultKeyFormatter(this->key3())
                  << " moved behind camera "
                  << DefaultKeyFormatter(this->key1()) << std::endl;
      if (throwCheirality_) throw e;
    }
    return Vector::Constant(2 * checkerboard_points_.size(), 2.0 * K.fx());
  }

  /** return the measurement */
  const std::vector<Point2>& measured() const { return measured_; }

  /** return verbosity */
  inline bool verboseCheirality() const { return verboseCheirality_; }

  /** return flag for throwing cheirality exceptions */
  inline bool throwCheirality() const { return throwCheirality_; }

 private:
  /// Serialization function
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(Base);
    ar& BOOST_SERIALIZATION_NVP(measured_);
    ar& BOOST_SERIALIZATION_NVP(throwCheirality_);
    ar& BOOST_SERIALIZATION_NVP(verboseCheirality_);
    ar& BOOST_SERIALIZATION_NVP(rows_);
    ar& BOOST_SERIALIZATION_NVP(cols_);
    ar& BOOST_SERIALIZATION_NVP(s_);
    ar& BOOST_SERIALIZATION_NVP(checkerboard_points_);
  }
};

template <class POSE, class LANDMARK, class CALIBRATION>
struct traits<CheckerboardFactorPPPC<POSE, LANDMARK, CALIBRATION>>
    : public Testable<
          CheckerboardFactorPPPC<POSE, LANDMARK, CALIBRATION>> {};
}  // namespace gtsam