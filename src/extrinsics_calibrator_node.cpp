/**
* Copyright (C) 2017 Kumar Shaurya Shankar <kumarsha at cs dot cmu dot edu>
* (Carnegie Mellon University)
*
*/

#include <extrinsics_calibrator/ExtrinsicCalibrationRosWrapper.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "extrinsics_calibrator");
  ros::NodeHandle nh("~");
  std::string params_file;
  nh.param<std::string>("params_file", params_file, "../config/params.yaml");
  std::cout << "[ExtrinsicCalibrator] Params File : " << params_file << "\n";
  ExtrinsicCalibrationRosWrapper calibrator(params_file, nh);

  calibrator.calibrate();
  ros::spin();
  return EXIT_SUCCESS;
}
