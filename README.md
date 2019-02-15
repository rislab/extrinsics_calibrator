# REQUIREMENTS #
* https://github.com/rislab/gtsam_catkin.git
* https://github.com/rislab/apriltag_tracker

Package for a simple camera extrinsics calibration node that performs batch optimization to obtain a camera to body transformation from an odometry source (mocap) and detected apriltags. This is done by utilizing multiple views of the apriltag detected pixels and incorporating them into a graph using a custom factor.

Parameters need to be set in the config file. As a precaution, ensure that the tags can be detected by enabling debug mode in [https://github.com/rislab/apriltag_tracker] and visualizing the output. Further, ensure that the initial extrinsic guess is correct at least in rotation, and the intrinsic guess matches that of the calibrated camera.

Extrinsics reported is the transform that takes points in the camera frame to the body frame.
