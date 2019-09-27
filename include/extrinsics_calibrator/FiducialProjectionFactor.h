/**
  * Copyright (C) 2017 Kumar Shaurya Shankar <kumarsha at cs dot cmu dot edu>
 * (Carnegie Mellon University)
 *
  */
#pragma once

#include <gtsam/geometry/SimpleCamera.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <boost/optional.hpp>
namespace gtsam {

/**
 * Factor to measure a fiducial from a given pose
 */
template <class CALIBRATION = Cal3_S2>
class FiducialProjectionFactor : public NoiseModelFactor2<Pose3, Pose3> {
 protected:
  // Keep a copy of measurement and calibration for I/O
  std::vector<Point2> measured_;      ///< 2D measurement
  boost::shared_ptr<CALIBRATION> K_;  ///< shared pointer to calibration object
  boost::optional<Pose3>
      body_P_sensor_;  ///< The pose of the sensor in the body frame
  double tag_size_;
  std::vector<Point3> fiducial_points_;

  // verbosity handling for Cheirality Exceptions
  bool throwCheirality_;  ///< If true, rethrows Cheirality exceptions (default:
                          ///false)
  bool verboseCheirality_;  ///< If true, prints text for Cheirality exceptions
                            ///(default: false)

 public:
  /// shorthand for base class type
  typedef NoiseModelFactor2<Pose3, Pose3> Base;

  /// shorthand for this class
  typedef FiducialProjectionFactor<CALIBRATION> This;

  /// shorthand for a smart pointer to a factor
  typedef boost::shared_ptr<This> shared_ptr;

  /// Constructor
  FiducialProjectionFactor(const std::vector<Point2>& measured,
                           const SharedNoiseModel& model, Key poseKey,
                           Key fiducialKey,
                           const boost::shared_ptr<CALIBRATION>& K,
                           const double& tag_size,
                           boost::optional<Pose3> body_P_sensor = boost::none)
      : Base(model, poseKey, fiducialKey),
        measured_(measured),
        K_(K),
        tag_size_(tag_size),
        body_P_sensor_(body_P_sensor),
        throwCheirality_(false),
        verboseCheirality_(true) {
    fiducial_points_.push_back(
        gtsam::Point3(-tag_size_ / 2.0, -tag_size_ / 2.0, 0));
    fiducial_points_.push_back(
        gtsam::Point3(tag_size_ / 2.0, -tag_size_ / 2.0, 0));
    fiducial_points_.push_back(
        gtsam::Point3(tag_size_ / 2.0, tag_size_ / 2.0, 0));
    fiducial_points_.push_back(
        gtsam::Point3(-tag_size_ / 2.0, tag_size_ / 2.0, 0));
  }

  /** Virtual destructor */
  virtual ~FiducialProjectionFactor() {}

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
    std::cout << s << "FiducialProjectionFactor, z = ";
    for (auto point : measured_) {
      traits<Point2>::Print(point);
    }
    if (this->body_P_sensor_)
      this->body_P_sensor_->print("  sensor pose in body frame: ");
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
    return e && Base::equals(p, tol) && point_equals &&
           this->K_->equals(*e->K_, tol) &&
           ((!body_P_sensor_ && !e->body_P_sensor_) ||
            (body_P_sensor_ && e->body_P_sensor_ &&
             body_P_sensor_->equals(*e->body_P_sensor_)));
  }

  /// Evaluate error h(x)-z and optionally derivatives
  Vector evaluateError(const Pose3& pose, const Pose3& object,
                       boost::optional<Matrix&> H1 = boost::none,
                       boost::optional<Matrix&> H2 = boost::none) const {
    try {
      if (body_P_sensor_) {
        if (H1) {
          gtsam::Matrix H0;
          PinholeCamera<CALIBRATION> camera(pose.compose(*body_P_sensor_, H0),
                                            *K_);

          Vector reprojectionError(8);
          *H1 = Matrix::Zero(8, 6);
          *H2 = Matrix::Zero(8, 6);  // FiducialObject is also a pose!
          // @todo: Make sure that the measurement dimensionality is the same as
          // fiducial
          for (int i = 0; i < 4; i++) {
            Matrix h1(2, 6), h2(2, 3), h0(3, 6);
            Point3 object_point_in_world =
                object.transformFrom(fiducial_points_[i], h0);
            Point2 error(
                // H1 is the jacobian wrt 6DoF pose 2x6
                // H2 is the jacobian wrt 3D point 2x3
                camera.project(object_point_in_world, h1, h2, boost::none) -
                measured_[i]);
            // Accumulate error and jacobians, since the error function is just
            // a sum
            reprojectionError[2 * i] = error[0];
            reprojectionError[2 * i + 1] = error[1];
            // Compose with composition jacobian
            H1->block<2, 6>(2 * i, 0) = h1 * H0;
            H2->block<2, 6>(2 * i, 0) = h2 * h0;
          }
          return reprojectionError;
        } else {
          Vector reprojectionError(8);
          PinholeCamera<CALIBRATION> camera(pose.compose(*body_P_sensor_), *K_);
          for (int i = 0; i < 4; i++) {
            Point3 object_point_in_world =
                object.transformFrom(fiducial_points_[i]);
            Point2 error(camera.project(object_point_in_world) - measured_[i]);
            reprojectionError[2 * i] = error[0];
            reprojectionError[2 * i + 1] = error[1];
          }
          return reprojectionError;
        }
      }
      //      else {
      //        PinholeCamera<CALIBRATION> camera(pose, *K_);
      //        Vector reprojectionError(8);
      //
      //        if (H1) {
      //          *H1 = Matrix::Zero(8, 6);
      //          *H2 = Matrix::Zero(8, 6);
      //        }
      //        for (int i = 0; i < 4; i++) {
      //          if (H1) {
      //            Matrix h1(2, 6), h2(2, 3), h0(3, 6);
      //            Point3 object_point_in_world = object.transformFrom(
      //                object.points_[i], h0);
      //            Point2 error(
      //            // H1 is the jacobian wrt 6DoF pose 2x6
      //            // H2 is the jacobian wrt 3D point 2x3
      //                camera.project(object_point_in_world, h1, h2,
      //                boost::none)
      //                    - measured_[i]);
      //            reprojectionError[2 * i] = error[0];
      //            reprojectionError[2 * i + 1] = error[1];
      //            // Compose with composition jacobian
      //            H1->block<2, 6>(2 * i, 0) = h1;
      //            H2->block<2, 6>(2 * i, 0) = h2 * h0;
      //          } else {
      //            Point3 object_point_in_world = object.transformFrom(
      //                object.points_[i]);
      //            Point2 error(
      //                camera.project(object_point_in_world, H1, H2,
      //                boost::none)
      //                    - measured_[i]);
      //            reprojectionError[2 * i] = error[0];
      //            reprojectionError[2 * i + 1] = error[1];
      //          }
      //        }
      //        return reprojectionError;
      //      }
    } catch (CheiralityException& e) {
      if (H1) *H1 = Matrix::Zero(8, 6);
      if (H2) *H2 = Matrix::Zero(8, 6);
      if (verboseCheirality_)
        std::cout << e.what() << ": FiducialObject "
                  << DefaultKeyFormatter(this->key2())
                  << " moved behind camera "
                  << DefaultKeyFormatter(this->key1()) << std::endl;
      if (throwCheirality_) throw e;
    }
    return Vector8::Constant(2.0 * K_->fx());
  }

  /** return the measurement */
  const std::vector<Point2>& measured() const { return measured_; }

  /** return the calibration object */
  inline const boost::shared_ptr<CALIBRATION> calibration() const { return K_; }

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
    ar& BOOST_SERIALIZATION_NVP(K_);
    ar& BOOST_SERIALIZATION_NVP(body_P_sensor_);
    ar& BOOST_SERIALIZATION_NVP(throwCheirality_);
    ar& BOOST_SERIALIZATION_NVP(verboseCheirality_);
    ar& BOOST_SERIALIZATION_NVP(tag_size_);
    ar& BOOST_SERIALIZATION_NVP(fiducial_points_);
  }
};
/// traits
template <>
struct traits<FiducialProjectionFactor<> >
    : public Testable<FiducialProjectionFactor<> > {};
}  // gtsam
