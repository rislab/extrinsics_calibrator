#!/usr/bin/env python2.7
'''
Script to determine real-world latency between image feed and odom data using a cyclic motion
(Yawing about camera Z looking at a chessboard)
'''
import numpy as np
import sys
import cv2
import tf
import pdb
import yaml

import rosbag
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge

import matplotlib.pyplot as plt


def transform_matrix_from_odom(msg):
    return transform_matrix_from_posestamped(msg.pose)


def transform_matrix_from_posestamped(msg):
    translation = np.array(
        [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    quaternion = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                           msg.pose.orientation.z, msg.pose.orientation.w])
    T = tf.transformations.quaternion_matrix(quaternion)
    T[:3, 3] = translation
    return T


prev_R = np.eye(3)
prev_c0 = np.array([0.0, 0.0])
flipped = True


def get_pnp_pose(colour_img, K, dist_coeffs, cols, rows):
    global prev_R, prev_c0, flipped
    gray = cv2.cvtColor(colour_img, cv2.COLOR_RGB2GRAY)
    debug_img = colour_img.copy()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    axis = np.float32([[s*cols, 0, 0], [0, s*rows, 0],
                       [0, 0, 0.5]]).reshape(-1, 3)

    ret, corners = cv2.findChessboardCorners(
        gray, (cols, rows), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    cv2.drawChessboardCorners(debug_img, (cols, rows), corners, ret)
    cv2.imshow('PreRefinement', debug_img)

    if ret == False:
        cv2.waitKey(1)
        return None
    else:
        # Use a radius of half the minimum distance between corners
        min_colwise_dist = np.min(
            np.sum(np.diff(corners.reshape(rows, cols, 2), axis=0)**2, axis=2))
        min_rowwise_dist = np.min(
            np.sum(np.diff(corners.reshape(rows, cols, 2), axis=1)**2, axis=2))
        min_dist = np.sqrt(min(min_colwise_dist, min_rowwise_dist))
        radius = int(np.ceil(min_dist*0.5))

        corners2 = cv2.cornerSubPix(
            gray, corners, (radius, radius), (-1, -1), criteria)
        corners2.shape = (corners2.shape[0], corners2.shape[-1])

        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv2.solvePnP(
            objp, corners2, K, dist_coeffs)
        rodRotMat = cv2.Rodrigues(rvecs)[0]

        if np.linalg.norm(tf.transformations.euler_from_matrix(np.dot(np.linalg.inv(prev_R), rodRotMat))) > 0.1:
            flipped = not flipped
        prev_R = rodRotMat

        # Check if the order has flipped
        if flipped:
            # pdb.set_trace()
            print 'flipping'
            corners2 = np.copy(np.flipud(corners2))
            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, K, dist_coeffs)

        if ret == True:
            rodRotMat = cv2.Rodrigues(rvecs)
            tag_in_cam = np.eye(4)
            tag_in_cam[:3, :3] = rodRotMat[0]
            tag_in_cam[:3, 3] = tvecs[:, 0]

            cv2.drawChessboardCorners(
                debug_img, (cols, rows), corners2, ret)
            cv2.imshow('Final corners', debug_img)
            cv2.waitKey(1)
            return tag_in_cam
        else:
            return None


# def calc_latency(body_in_world_tuples, tag_in_cam_tuples, board_in_world, tag_in_board, cam_in_body):
def calc_latency(body_in_world_tuples, tag_in_cam_tuples, cam_in_body):
    # Compute pnp odom and corresponding mocap odom
    # We compare the yaw differences
    tag_in_cam_tuples = np.array(tag_in_cam_tuples)
    body_in_world_tuples = np.array(body_in_world_tuples)

    pnp_deltas = [np.dot(np.linalg.inv(np.linalg.inv(a)), np.linalg.inv(
        b)) for a, b in zip(tag_in_cam_tuples[:, 0], tag_in_cam_tuples[1:, 0])]
    body_in_cam = np.linalg.inv(cam_in_body)
    pnp_deltas_in_body = [np.dot(cam_in_body, np.dot(
        delta, body_in_cam)) for delta in pnp_deltas]

    gt_deltas = [np.dot(np.linalg.inv(a), b) for a, b in zip(
        body_in_world_tuples[:, 0], body_in_world_tuples[1:, 0])]
    # cams_in_world_through_tag =
    #     [np.dot(board_in_world,  np.dot(tag_in_board, np.linalg.inv(t[0])))
    #      for t in tag_in_cam_tuples]
    pnp_timestamps = [t[1] for t in tag_in_cam_tuples[1:]]

    # cams_in_world_mocap =
    #     [np.dot(t[0], cam_in_body)
    #      for t in body_in_world_tuples]
    gt_timestamps = [t[1] for t in body_in_world_tuples[1:]]

    # Extract the yaw
    sampled_pnp_odom = [tf.transformations.euler_from_matrix(
        t)[2] for t in pnp_deltas_in_body]
    sampled_gt_odom = [tf.transformations.euler_from_matrix(t)[
        2] for t in gt_deltas]

    # Mini optimisation to find offset
    def mini_cost(dt):
        sampled_odoms = np.interp(
            np.array(pnp_timestamps) + dt, np.array(gt_timestamps), np.array(sampled_gt_odom))
        return np.sum(
            (np.array(sampled_pnp_odom) - np.array(sampled_odoms))**2)

    sample_pts_for_mini = np.linspace(-1.0, 0.0, 1001)
    mini_costs = [mini_cost(dt)
                  for dt in sample_pts_for_mini]

    min_idx_mini_cost = np.argmin(mini_costs)
    print 'mini_costs min is {0} corresponding to dt of {1}'.format(
        min_idx_mini_cost, sample_pts_for_mini[min_idx_mini_cost])
    offset_dt = sample_pts_for_mini[min_idx_mini_cost]

    average_gt_dt = np.mean(np.diff(gt_timestamps))
    move_idx = int(offset_dt/(average_gt_dt))
    if offset_dt < 0.0:
        # PnP lags behind gt
        # Find number of indices we have to move gt forward
        shifted_gt_odom = sampled_gt_odom[:move_idx]
        shifted_gt_timestamps = gt_timestamps[-move_idx:]
    else:
        shifted_gt_odom = sampled_gt_odom[:-move_idx]
        shifted_gt_timestamps = gt_timestamps[move_idx:]

    # Plot!
    f, axes = plt.subplots(1, 2)
    axes[0].set_title('Error norm by time offset')
    axes[0].plot(sample_pts_for_mini, mini_costs)
    axes[0].set_xlabel('Time offset (s)')
    axes[0].set_ylabel('Error')
    axes[0].grid(True)

    axes[1].set_title('Yaw plots for aligned PnP and Ground Truth')
    axes[1].plot(pnp_timestamps, sampled_pnp_odom, label='PnP')
    axes[1].plot(gt_timestamps, sampled_gt_odom, label='gt', linestyle='--')
    axes[1].plot(shifted_gt_timestamps, shifted_gt_odom, label='shifted_gt')
    axes[1].legend()
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Yaw')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    pdb.set_trace()


if __name__ == "__main__":
    # Read bag file
    path = '/media/icoderaven/Dumps/bagfiles/lag_detect_2019-11-14-19-57-40.bag'
    odom_topic = '/vicon/realsense_rig_new'
    img_topic = '/camera/infra1/image_rect_raw'
    topics_to_read = [odom_topic, img_topic]
    # Initialise parameters
    cam_to_body = None
    K = None
    tag_pts = None
    cols = None
    rows = None
    s = None
    dist_coeffs = None
    with open('../config/checkerboard_extrinsics_calib.yaml', 'r') as f:
        file_node = yaml.load(f)
        ext_calib_node = file_node['extrinsics_calib_params']
        # TODO: Check if a cam_to_body exists, if not, then just use the extrinsics guess
        cam_to_body = np.array(
            ext_calib_node['extrinsics_guess']).reshape(4, 4)
        K = np.eye(3)
        K_node = ext_calib_node['intrinsics_guess']
        K[0, 0] = K_node[0]
        K[1, 1] = K_node[1]
        K[0, 2] = K_node[3]
        K[1, 2] = K_node[4]
        # k1, k2, p1, p2, k3
        dist_coeffs = np.array(
            [K_node[7], K_node[8], K_node[5], K_node[6], K_node[9]])

        checkerboard_params_node = file_node['checkerboard_params']
        cols = checkerboard_params_node['cols']
        rows = checkerboard_params_node['rows']
        s = checkerboard_params_node['s']

        objp = np.zeros((cols*rows, 3))
        objp[:, : 2] = s * np.mgrid[0: cols, 0: rows].T.reshape(-1, 2)
        tag_pts = np.concatenate(
            (objp, np.ones((objp.shape[0], 1))), axis=1).transpose()

    bridge = CvBridge()

    body_in_world_tuples = []
    tag_in_cam_tuples = []
    actual_images = []
    with rosbag.Bag(path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics_to_read):
            if (len(tag_in_cam_tuples) > 400):
                break
            if topic == odom_topic:
                if msg._type == 'nav_msgs/Odometry':
                    body_in_world_tuples.append(
                        [transform_matrix_from_odom(msg), msg.header.stamp.to_sec()])
                elif msg._type == 'geometry_msgs/PoseStamped':
                    body_in_world_tuples.append(
                        [transform_matrix_from_posestamped(msg), msg.header.stamp.to_sec()])
            elif topic == img_topic:
                img = bridge.imgmsg_to_cv2(msg, 'bgr8')
                actual_images.append(img)
                # Get pnp pose
                pose = get_pnp_pose(img, K, dist_coeffs, cols, rows)
                if pose is not None:
                    tag_in_cam_tuples.append([pose, msg.header.stamp.to_sec()])

    print 'Read {0} odoms from gt odom and {1} odoms from PnP'.format(
        len(body_in_world_tuples), len(tag_in_cam_tuples))

    # Now call the offset estimator
    calc_latency(body_in_world_tuples, tag_in_cam_tuples, cam_to_body)
