/**
 * Copyright (C) 2017 Kumar Shaurya Shankar <kumarsha at cs dot cmu dot edu>
 * (Carnegie Mellon University)
 *
 */
#pragma once

#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>

#include <vector>

namespace gtsam {

/**
 * @brief Represents a fiducial in 3D, which is composed of an object pose
 * and 3D points corresponding to the object
 */

class GTSAM_EXPORT FiducialObject : public Pose3 {
 public:
  std::vector<Point3>
      points_;  ///< The set of 3D points describing the fiducial

  /// @name Constructors
  /// @{
  /// Default constructor
  FiducialObject() : points_() {}

  FiducialObject(const Pose3 &pose) : Pose3(pose) {
    // Workaround for not being able to downcast implicitly :(
  }

  FiducialObject(const FiducialObject &obj)
      : points_(obj.points_), Pose3(obj) {}

  FiducialObject(const Rot3 &R, const Point3 &t,
                 const std::vector<Point3> &points)
      : Pose3(R, t), points_(points) {}

  FiducialObject(const Pose3 &pose, const std::vector<Point3> &points)
      : Pose3(pose), points_(points) {}

  FiducialObject(const Matrix &T, const std::vector<Point3> &points)
      : Pose3(T), points_(points) {}

  /**
   * Manifold
   */

  //  // Chart at origin, depends on compile-time flag GTSAM_POSE3_EXPMAP
  //  struct ChartAtOrigin {
  //    static FiducialObject Retract(const Vector6& v, ChartJacobian H =
  //                                      boost::none) {
  //      return FiducialObject(Pose3::Retract(v, H));
  //    }
  //  };
};

template <>
struct traits<FiducialObject> : public internal::LieGroup<FiducialObject> {};

template <>
struct traits<const FiducialObject>
    : public internal::LieGroup<FiducialObject> {};
}  // namespace gtsam
