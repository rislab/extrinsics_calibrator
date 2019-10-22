#!/usr/bin/env python2.7
import numpy as np
import cv2
import yaml
import rospy
import tf
import rosbag
import pdb
import time
from geometry import se3, SE3
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.msg import LinkStates, LinkState
from gazebo_msgs.srv import SetLinkState, GetLinkState
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image

from extrinsics_calibrator.ExtrinsicCalibPyModules import CheckerboardExtrinsicCalibration


def transform_matrix_from_odom(msg):
    translation = np.array(
        [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
    quaternion = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                           msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
    T = tf.transformations.quaternion_matrix(quaternion)
    T[:3, 3] = translation
    return T


def pose_msg_from_matrix(matrix):
    msg = Pose()
    msg.position.x = matrix[0, 3]
    msg.position.y = matrix[1, 3]
    msg.position.z = matrix[2, 3]
    quaternion = tf.transformations.quaternion_from_matrix(matrix)
    msg.orientation.x = quaternion[0]
    msg.orientation.y = quaternion[1]
    msg.orientation.z = quaternion[2]
    msg.orientation.w = quaternion[3]
    return msg


def get_measurements(board_gt_pose, tag_in_board, tag_pts, gt_cam_odoms):
    board_gt_pts = np.dot(
        board_gt_pose, np.dot(tag_in_board, tag_pts)).T
    measurements = []
    for cam_odom in gt_cam_odoms:
        # Transform the board gt points into the cam frame
        gt_pts_in_cam = np.dot(np.linalg.inv(cam_odom), board_gt_pts.T)
        # Project these into the image
        projected = np.dot(K, gt_pts_in_cam[:3, :])
        # Normalize
        projected /= projected[2]
        measurements.append(projected[:2, :].T)
    return measurements


def generate_images(cam_poses):
    state = LinkState()
    state.reference_frame = 'world'
    success = False
    for cam_pose in cam_poses:
        # Call gazebo to generate a view of the checkerboard
        state.pose = pose_msg_from_matrix(cam_pose)
        state.link_name = 'simple_rgbd_cam::link1'
        success = False
        while not success:
            response = set_link_service(state)
            success = response.success
            time.sleep(0.2)
        # Pull the image from the image topic
        images.append(latest_rgb_img)
        print 'appended, moving on'


latest_rgb_img = None
bridge = CvBridge()


def rgb_img_callback(msg):
    global latest_rgb_img, bridge
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError as e:
        print(e)
    latest_rgb_img = cv_image.copy()


# Read the config file and populate params
config_file = '../config/checkerboard_extrinsics_calib.yaml'
visualise = True

with open('../config/checkerboard_extrinsics_calib.yaml', 'r') as f:
    file_dict = yaml.load(f)
    s = np.array(file_dict['checkerboard_params']['s'])
    rows = np.array(file_dict['checkerboard_params']['rows'])
    cols = np.array(file_dict['checkerboard_params']['cols'])

    intrinsics = np.array(
        file_dict['extrinsics_calib_params']['intrinsics_guess'])
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[3]
    K[1, 2] = intrinsics[4]


# Create a virtual checkerboard at a location in space
objp = np.zeros((cols*rows, 3))
objp[:, :2] = s * np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
tag_pts = np.concatenate(
    (objp, np.ones((objp.shape[0], 1))), axis=1).transpose()

board_gt_pose = np.eye(4)
board_gt_pose = tf.transformations.quaternion_matrix(
    [0.420, 0.337, -0.666, 0.514])
board_gt_pose[:3, 3] = [0.214, 0.817, 0.574]

# Call gazebo to assign board pose
rospy.wait_for_service('/gazebo/set_link_state')
set_link_service = rospy.ServiceProxy('gazebo/set_link_state', SetLinkState)
get_link_service = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
state = LinkState()
state.pose = pose_msg_from_matrix(board_gt_pose)
state.link_name = 'checkerboard_5_7_0_08::checkerboard_5_7_0_08_body'
state.reference_frame = 'world'
success = False
while not success:
    response = set_link_service(state)
    success = response.success
    time.sleep(0.2)

# Create measurements from a bunch of different poses
# For now just use the same poses as from the bag file
bag_path = '/home/icoderaven/ext_calib_realsense_new_small_board.bag'
odom_topic = '/realsense_rig_new/vicon_odom'
img_topic = '/camera/color/image_raw'
rospy.init_node('test_checkerboard')
image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, rgb_img_callback)
gt_body_odoms = []
images = []
actual_images = []
with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        if topic == odom_topic:
            gt_body_odoms.append(transform_matrix_from_odom(msg))
        elif topic == img_topic:
            actual_images.append(bridge.imgmsg_to_cv2(msg, 'bgr8'))

# Visualise test
import matplotlib.pyplot as plt
cam_to_body_init = np.array([[0.9512, -0.2919, -0.0999, -0.0742],
                             [0.277,  0.9506, -0.1402, -0.0349],
                             [0.1359,  0.1057,  0.9851,  0.0826],
                             [0.,  0.,  0.,  1.]])
cam_body_odoms = np.dot(gt_body_odoms, cam_to_body_init)
generate_images(cam_body_odoms)
l = len(actual_images)
f, axes = plt.subplots(4, l/2)
for i in range(l/2):
    axes[0, i].imshow(actual_images[i])
    axes[1, i].imshow(images[i])
    axes[2, i].imshow(actual_images[i+l/2])
    axes[3, i].imshow(images[i+l/2])
plt.show()

# See if we can obtain the extrinsics to a high degree of accuracy
# with increasing amount of deviation from identity

deviation_norms = np.linspace(0.0, 1.0, 11)
tag_board_deviation_norms = [0.0]  # , 0.01, 0.05, 0.1]
num_eval_per_deviation = 10

successful_calibs = []
for dev_id, deviation_norm in enumerate(deviation_norms):
    successful_calibs.append([])

    tag_in_board_base = np.array(
        [[0.0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    for tag_norm in tag_board_deviation_norms:
        correct_extrinsics_found = []
        # Generate a random vector in tangent space with norm == deviation_norm
        tag_tangent_vector = np.random.rand(6)
        tag_tangent_vector *= tag_norm / np.linalg.norm(tag_tangent_vector)
        # Now generate the equivalent SE3 group from this vector
        tag_board_deviation = SE3.group_from_algebra(
            se3.algebra_from_vector(tag_tangent_vector))
        tag_board_deviation = np.eye(4)
        tag_board_deviation[2, 3] = tag_norm
        tag_in_board = np.dot(tag_in_board_base, tag_board_deviation)

        for eval_id in range(num_eval_per_deviation):
            # Initialise a ground truth cam_to_body offset
            cam_to_body_gt = np.eye(4)
            # Generate a random vector in tangent space with norm == deviation_norm
            extrinsic_tangent_vector = np.random.rand(6)
            extrinsic_tangent_vector *= deviation_norm / \
                np.linalg.norm(extrinsic_tangent_vector)
            # Now generate the equivalent SE3 group from this vector
            cam_to_body_gt = SE3.group_from_algebra(
                se3.algebra_from_vector(extrinsic_tangent_vector))

            gt_cam_odoms = [np.dot(gt_body_odom, cam_to_body_gt)
                            for gt_body_odom in gt_body_odoms]
            gt_body_in_board_odoms = [np.dot(np.linalg.inv(board_gt_pose), gt_body_odom)
                                      for gt_body_odom in gt_body_odoms]

            measurements = get_measurements(
                board_gt_pose, tag_in_board, tag_pts, gt_cam_odoms)
            # Create calibration object
            calibrator = CheckerboardExtrinsicCalibration(config_file)

            # Add measurements
            for i, body_odom in enumerate(gt_body_odoms):
                calibrator.add_measurement(
                    body_odom, board_gt_pose, measurements[i])

            # Ask to solve
            cam_to_body = np.eye(4)
            t_in_board = np.eye(4)
            body_poses_in_board = np.zeros((len(gt_body_odoms), 16))
            landmark_pts = np.zeros((rows*cols, 3))
            K_calib = np.zeros((3, 3))

            calibrator.solve(cam_to_body, t_in_board,
                             body_poses_in_board, landmark_pts, K_calib)
            correct_extrinsics_found.append(np.allclose(
                cam_to_body_gt, cam_to_body, atol=0.5*1e-4))

            if visualise:
                cam_odoms = [np.dot(gt_body_odom, cam_to_body)
                             for gt_body_odom in gt_body_odoms]
                measurements = get_measurements(
                    board_gt_pose, tag_in_board, tag_pts, cam_odoms)

        successful_calibs[dev_id].append(
            1.0*np.count_nonzero(correct_extrinsics_found)/num_eval_per_deviation)
fig = plt.figure('ExtrinsicsValidator')
successful_calibs = np.array(successful_calibs)
for i in range(len(tag_board_deviation_norms)):
    plt.plot(deviation_norms, successful_calibs[:, i], label=str(
        tag_board_deviation_norms[i]))
plt.title('Successful Extrinsic Calibration Found')
plt.legend()
plt.xlabel('Extrinsic norm')
plt.ylabel('Success Ratio')
plt.grid(True)
plt.tight_layout()
plt.show()
pdb.set_trace()
